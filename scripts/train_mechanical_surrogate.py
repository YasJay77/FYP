"""Fit mechanical surrogates: relative density -> stiffness tensor for SC and FCC.

Compares Gibson-Ashby power law (C_ij = C1 * rho^n * E), GPR and XGBoost. Only
~10-20 points per lattice, so LOOCV is the only sane way to gauge generalisation.
Mechanical voxels are pore-free — fine here since d/L <= 0.3, where the pores
barely touch the elastic tensor.
"""

import argparse
import json
import os
import pickle
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import pearsonr
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.metrics import r2_score, mean_absolute_percentage_error
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Note: xgboost not installed — skipping XGBoost method. "
          "Install with: pip install xgboost")

SCRIPTS_DIR  = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS_DIR.parent
OUTPUTS_DIR  = PROJECT_ROOT / "results" / "property_datasets"
OUTPUTS_DIR.mkdir(exist_ok=True)


def load_datasets(sc_csv, fcc_csv):
    """load SC + FCC HomPy CSVs, tag lattice_type"""
    sc  = pd.read_csv(sc_csv)
    fcc = pd.read_csv(fcc_csv)
    sc['lattice_type']  = 'SC'
    fcc['lattice_type'] = 'FCC'

    # 0.025 floor
    sc_filt  = sc[sc['t_over_L']  >= 0.025].copy()
    fcc_filt = fcc[fcc['t_over_L'] >= 0.025].copy()

    n_sc_dropped  = len(sc)  - len(sc_filt)
    n_fcc_dropped = len(fcc) - len(fcc_filt)
    if n_sc_dropped > 0:
        print(f"  SC:  dropped {n_sc_dropped} rows with t/L < 0.025")
    if n_fcc_dropped > 0:
        print(f"  FCC: dropped {n_fcc_dropped} rows with t/L < 0.025")

    # sanity-check tensors before fitting
    for df, name in [(sc_filt, 'SC'), (fcc_filt, 'FCC')]:
        bad = df[df['C11'] <= df['C12']]
        if len(bad) > 0:
            print(f"  WARNING: {name} has {len(bad)} rows where C11 <= C12 "
                  f"(cubic symmetry violation). Rows: {bad.index.tolist()}")
        neg = df[(df['C11'] <= 0) | (df['C44'] <= 0)]
        if len(neg) > 0:
            print(f"  WARNING: {name} has {len(neg)} rows with C11<=0 or C44<=0")

    combined = pd.concat([sc_filt, fcc_filt], ignore_index=True)
    print(f"  Loaded: {len(sc_filt)} SC rows, {len(fcc_filt)} FCC rows "
          f"({len(combined)} total)")
    return sc_filt, fcc_filt, combined


# Gibson-Ashby power law

def gibson_ashby_model(rho, C1, n):
    """C_ij = C1 * rho^n (E normalised to 1)"""
    return C1 * np.power(np.abs(rho), n)


def fit_gibson_ashby(rho, C_vals, target_name="C11"):
    """fit C1, n for one stiffness component."""
    try:
        popt, pcov = curve_fit(
            gibson_ashby_model, rho, C_vals,
            p0=[0.5, 1.0],
            bounds=([0.0, 0.1], [10.0, 5.0]),
            maxfev=10000
        )
        C1_fit, n_fit = popt
        C_pred = gibson_ashby_model(rho, *popt)
        fit_r2 = r2_score(C_vals, C_pred)
        return C1_fit, n_fit, fit_r2
    except RuntimeError as e:
        print(f"  WARNING: Gibson-Ashby fit failed for {target_name}: {e}")
        return 0.5, 1.0, 0.0


def loocv_gibson_ashby(rho, C_vals):
    """LOOCV for the GA fit."""
    loo = LeaveOneOut()
    y_true, y_pred = [], []
    for train_idx, test_idx in loo.split(rho):
        rho_train = rho[train_idx]
        C_train   = C_vals[train_idx]
        rho_test  = rho[test_idx]
        C_test    = C_vals[test_idx]
        try:
            popt, _ = curve_fit(
                gibson_ashby_model, rho_train, C_train,
                p0=[0.5, 1.0],
                bounds=([0.0, 0.1], [10.0, 5.0]),
                maxfev=5000
            )
            y_pred.append(float(gibson_ashby_model(rho_test[0], *popt)))
        except RuntimeError:
            y_pred.append(float(np.mean(C_train)))
        y_true.append(float(C_test[0]))
    return np.array(y_true), np.array(y_pred)


# GPR

def build_gpr(n_restarts=10):
    """GPR with RBF + white-noise kernel"""
    kernel = (ConstantKernel(1.0, (1e-3, 1e3)) *
              RBF(length_scale=0.1, length_scale_bounds=(1e-3, 10.0)) +
              WhiteKernel(noise_level=1e-4, noise_level_bounds=(1e-6, 1e-1)))
    return GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=n_restarts,
        normalize_y=True,
        alpha=1e-8
    )


def loocv_gpr(X, y):
    """LOOCV for GPR."""
    loo = LeaveOneOut()
    y_true, y_pred = [], []
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    for train_idx, test_idx in loo.split(X_scaled):
        gpr = build_gpr(n_restarts=5)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gpr.fit(X_scaled[train_idx], y[train_idx])
        pred = gpr.predict(X_scaled[test_idx])
        y_pred.append(float(pred[0]))
        y_true.append(float(y[test_idx[0]]))
    return np.array(y_true), np.array(y_pred)


# XGBoost

def loocv_xgboost(X, y):
    """LOOCV for XGBoost."""
    if not XGBOOST_AVAILABLE:
        return None, None
    loo = LeaveOneOut()
    y_true, y_pred = [], []
    for train_idx, test_idx in loo.split(X):
        model = xgb.XGBRegressor(
            n_estimators=100, max_depth=3,
            learning_rate=0.1, subsample=0.8,
            verbosity=0, random_state=42
        )
        model.fit(X[train_idx], y[train_idx])
        y_pred.append(float(model.predict(X[test_idx])[0]))
        y_true.append(float(y[test_idx[0]]))
    return np.array(y_true), np.array(y_pred)


def compute_metrics(y_true, y_pred, name):
    """print + return R2 and MAPE"""
    r2   = r2_score(y_true, y_pred)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    print(f"    {name:25s}  R²={r2:.4f}  MAPE={mape:.2f}%")
    return r2, mape


COLORS = {'SC': '#1f77b4', 'FCC': '#ff7f0e'}
MARKERS = {'SC': 'o', 'FCC': 's'}


def plot_validation(results_dict, save_path):
    """LOOCV parity plots per method/target"""
    targets  = ['C11', 'C12', 'C44']
    methods  = [m for m in ['Gibson-Ashby', 'GPR', 'XGBoost']
                if m in results_dict]
    n_rows   = len(methods)
    n_cols   = len(targets)

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(4.5 * n_cols, 4 * n_rows),
                             squeeze=False)
    fig.suptitle("Mechanical Surrogate — LOOCV Validation\n"
                 "(Predicted vs True, leave-one-out)",
                 fontsize=13, y=1.01)

    for row, method in enumerate(methods):
        for col, target in enumerate(targets):
            ax = axes[row][col]
            if target not in results_dict[method]:
                ax.axis('off')
                continue

            yt, yp, r2, mape, lattice_labels = results_dict[method][target]

            # colour/marker per lattice
            for lt in ['SC', 'FCC']:
                mask = np.array(lattice_labels) == lt
                if mask.any():
                    ax.scatter(yt[mask], yp[mask],
                               color=COLORS[lt], marker=MARKERS[lt],
                               s=60, label=lt, alpha=0.85, zorder=3)

            # 1:1 line
            lo = min(yt.min(), yp.min()) * 0.95
            hi = max(yt.max(), yp.max()) * 1.05
            ax.plot([lo, hi], [lo, hi], 'k--', lw=1, alpha=0.5, zorder=2)

            ax.set_xlabel(f"True {target}/E")
            ax.set_ylabel(f"Predicted {target}/E")
            ax.set_title(f"{method} — {target}\nR²={r2:.3f}  MAPE={mape:.1f}%",
                         fontsize=10)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_curves(sc_df, fcc_df, ga_params, save_path):
    """C11/C12/C44 vs t/L and vs rho: GA fit lines + HomPy scatter"""
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle("Structure-Property Relationships — Plate Lattice Surrogates",
                 fontsize=13)

    targets = ['C11', 'C12', 'C44']
    t_dense = np.linspace(0.025, 0.22, 200)

    for col, target in enumerate(targets):
        # top row: vs t/L
        ax = axes[0][col]
        for lt, df in [('SC', sc_df), ('FCC', fcc_df)]:
            ax.scatter(df['t_over_L'], df[target],
                       color=COLORS[lt], marker=MARKERS[lt],
                       s=50, label=f'{lt} HomPy', zorder=3)
            # GA takes rho, so map t/L -> rho first
            if lt in ga_params and target in ga_params[lt]:
                C1, n = ga_params[lt][target]
                rho_fit = np.polyfit(df['t_over_L'], df['rho'], 1)
                rho_dense = np.polyval(rho_fit, t_dense)
                rho_dense = np.clip(rho_dense, 0.01, 1.0)
                C_curve = gibson_ashby_model(rho_dense, C1, n)
                ls = '-' if lt == 'SC' else '--'
                ax.plot(t_dense, C_curve, color=COLORS[lt],
                        ls=ls, lw=1.8, label=f'{lt} GA fit', zorder=2)

        ax.set_xlabel("t/L")
        ax.set_ylabel(f"{target}/E")
        ax.set_title(f"{target} vs t/L")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        # shade acoustic t/L window
        ax.axvspan(0.03, 0.18, alpha=0.08, color='green',
                   label='Acoustic range' if col == 0 else '')

        # bottom row: vs relative density
        ax = axes[1][col]
        rho_dense_plot = np.linspace(0.01, 0.55, 200)
        for lt, df in [('SC', sc_df), ('FCC', fcc_df)]:
            ax.scatter(df['rho'], df[target],
                       color=COLORS[lt], marker=MARKERS[lt],
                       s=50, label=f'{lt} HomPy', zorder=3)
            if lt in ga_params and target in ga_params[lt]:
                C1, n = ga_params[lt][target]
                C_curve = gibson_ashby_model(rho_dense_plot, C1, n)
                ls = '-' if lt == 'SC' else '--'
                ax.plot(rho_dense_plot, C_curve, color=COLORS[lt],
                        ls=ls, lw=1.8, label=f'{lt} GA fit', zorder=2)

        ax.set_xlabel("Relative density ρ*")
        ax.set_ylabel(f"{target}/E")
        ax.set_title(f"{target} vs ρ* (Gibson-Ashby)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path}")


# prediction fn used by the optimiser and the dataset merger

def make_predict_fn(ga_params, gpr_models, gpr_scalers, best_method):
    """build the predict_mechanical closure (C11/C12/C44 normalised by E)"""
    def predict_mechanical(t_over_L, lattice_type, rho=None, E_material=1.0):
        """predict C11, C12, C44 for one geometry. rho recovered from t/L if None"""
        lt = lattice_type.upper()

        # no density given -> recover from rho-t/L fit
        if rho is None:
            slope, intercept = ga_params[lt]['rho_fit']
            rho = float(np.clip(slope * t_over_L + intercept, 0.01, 1.0))

        results = {}
        for target in ['C11', 'C12', 'C44']:
            if best_method == 'Gibson-Ashby' and lt in ga_params:
                C1, n = ga_params[lt][target]
                results[target] = float(gibson_ashby_model(rho, C1, n))
            elif best_method == 'GPR' and lt in gpr_models:
                X = np.array([[rho, 1.0 if lt == 'SC' else 0.0]])
                X_sc = gpr_scalers[lt][target].transform(X)
                results[target] = float(gpr_models[lt][target].predict(X_sc)[0])
            else:
                # fall back to GA
                C1, n = ga_params[lt][target]
                results[target] = float(gibson_ashby_model(rho, C1, n))

        return (results['C11'] * E_material,
                results['C12'] * E_material,
                results['C44'] * E_material)

    return predict_mechanical


def main():
    """parse args, fit the GPR + Gibson-Ashby surrogates and merge them into the dataset"""
    parser = argparse.ArgumentParser(
        description="Train mechanical surrogate for SC+FCC plate lattices."
    )
    parser.add_argument('--sc_csv',  type=str,
        default=str(PROJECT_ROOT / 'results' / 'data' / 'sc_mech_res100_dilation.csv'))
    parser.add_argument('--fcc_csv', type=str,
        default=str(PROJECT_ROOT / 'results' / 'data' / 'fcc_mech_res120_dilation.csv'))
    parser.add_argument('--property_dataset', type=str,
        default=str(PROJECT_ROOT / 'results' / 'property_datasets' / 'property_dataset.csv'),
        help='Acoustic property dataset to merge mechanical outputs into')
    parser.add_argument('--skip_merge', action='store_true',
        help='Skip merging into property dataset (if it does not exist yet)')
    args = parser.parse_args()

    print("=" * 60)
    print("Mechanical Surrogate Training")
    print("=" * 60)

    print("\n1. Loading HomPy datasets...")
    sc_df, fcc_df, combined = load_datasets(args.sc_csv, args.fcc_csv)

    targets = ['C11', 'C12', 'C44']

    print("\n2. Fitting Gibson-Ashby power laws...")
    ga_params = {}

    for lt, df in [('SC', sc_df), ('FCC', fcc_df)]:
        rho = df['rho'].values
        ga_params[lt] = {}

        # keep rho-t/L fit so rho can be recovered from t/L alone
        rho_fit = np.polyfit(df['t_over_L'].values, rho, 1)
        ga_params[lt]['rho_fit'] = rho_fit.tolist()

        print(f"\n  {lt} (n={len(df)} points):")
        for target in targets:
            C_vals = df[target].values
            C1, n, fit_r2 = fit_gibson_ashby(rho, C_vals, target)
            ga_params[lt][target] = [float(C1), float(n)]
            print(f"    {target}: C1={C1:.4f}  n={n:.3f}  fit R²={fit_r2:.4f}")

    ga_path = OUTPUTS_DIR / 'surrogate_gibsonashby.json'
    with open(ga_path, 'w') as f:
        json.dump(ga_params, f, indent=2)
    print(f"\n  Saved: {ga_path}")

    print("\n3. Running LOOCV validation...")

    # GPR/XGB features: [rho, is_SC]
    rho_all     = combined['rho'].values
    is_sc       = (combined['lattice_type'] == 'SC').astype(float).values
    X_all       = np.column_stack([rho_all, is_sc])
    lt_labels   = combined['lattice_type'].values.tolist()

    results_dict = {}

    # GA LOOCV: per lattice, then pool
    print("\n  Gibson-Ashby:")
    ga_loocv = {'Gibson-Ashby': {}}
    for target in targets:
        yt_all, yp_all, ll_all = [], [], []
        for lt, df in [('SC', sc_df), ('FCC', fcc_df)]:
            rho = df['rho'].values
            C_vals = df[target].values
            yt, yp = loocv_gibson_ashby(rho, C_vals)
            yt_all.extend(yt.tolist())
            yp_all.extend(yp.tolist())
            ll_all.extend([lt] * len(yt))
        yt_all = np.array(yt_all)
        yp_all = np.array(yp_all)
        r2, mape = compute_metrics(yt_all, yp_all, f"GA {target}")
        ga_loocv['Gibson-Ashby'][target] = (yt_all, yp_all, r2, mape, ll_all)
    results_dict.update(ga_loocv)

    # GPR LOOCV
    print("\n  GPR:")
    gpr_models  = {'SC': {}, 'FCC': {}}
    gpr_scalers = {'SC': {}, 'FCC': {}}
    gpr_loocv   = {'GPR': {}}

    for target in targets:
        y_all = combined[target].values
        yt, yp = loocv_gpr(X_all, y_all)
        r2, mape = compute_metrics(yt, yp, f"GPR {target}")
        gpr_loocv['GPR'][target] = (yt, yp, r2, mape, lt_labels)

        # refit on full data for prediction time
        for lt, df in [('SC', sc_df), ('FCC', fcc_df)]:
            X_lt = np.column_stack([df['rho'].values,
                                    np.ones(len(df)) if lt == 'SC'
                                    else np.zeros(len(df))])
            scaler = StandardScaler()
            X_lt_sc = scaler.fit_transform(X_lt)
            gpr = build_gpr()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                gpr.fit(X_lt_sc, df[target].values)
            gpr_models[lt][target]  = gpr
            gpr_scalers[lt][target] = scaler

    results_dict.update(gpr_loocv)

    gpr_path = OUTPUTS_DIR / 'surrogate_gpr.pkl'
    with open(gpr_path, 'wb') as f:
        pickle.dump({'models': gpr_models, 'scalers': gpr_scalers}, f)
    print(f"\n  Saved: {gpr_path}")

    # XGBoost LOOCV
    if XGBOOST_AVAILABLE:
        print("\n  XGBoost:")
        xgb_loocv = {'XGBoost': {}}
        xgb_models = {}
        for target in targets:
            y_all = combined[target].values
            yt, yp = loocv_xgboost(X_all, y_all)
            if yt is not None:
                r2, mape = compute_metrics(yt, yp, f"XGB {target}")
                xgb_loocv['XGBoost'][target] = (yt, yp, r2, mape, lt_labels)
                # refit on full data
                model = xgb.XGBRegressor(
                    n_estimators=100, max_depth=3,
                    learning_rate=0.1, subsample=0.8,
                    verbosity=0, random_state=42)
                model.fit(X_all, y_all)
                xgb_models[target] = model
                model.save_model(str(OUTPUTS_DIR / f'surrogate_xgboost_{target}.json'))
        results_dict.update(xgb_loocv)
        print(f"  Saved XGBoost models to {OUTPUTS_DIR}/")

    print("\n4. Method comparison (LOOCV R²) — shown for reference only:")
    method_scores = {}
    for method, method_results in results_dict.items():
        r2_vals = [v[2] for v in method_results.values()]
        method_scores[method] = np.mean(r2_vals)
        print(f"  {method:20s}  mean R² = {method_scores[method]:.4f}")

    # GA is primary regardless of LOOCV rank: with <20 geometries GPR/XGB just
    # interpolate every point, so the score says nothing about generalisation.
    best_method = 'Gibson-Ashby'
    print(f"\n  Selected: Gibson-Ashby  (physical model — interpretable exponents,")
    print(f"  theoretically prescribed for stretch-dominated plate lattices)")
    print(f"  GPR and XGBoost LOOCV scores shown above for comparison only.")

    best_path = OUTPUTS_DIR / 'surrogate_best_method.json'
    with open(best_path, 'w') as f:
        json.dump({'best_method': best_method,
                   'rationale': (
                       'Gibson-Ashby selected as primary: n_unique_training '
                       'geometries is too small (<20) for GPR LOOCV to reflect '
                       'genuine generalisation over interpolation. GA has direct '
                       'theoretical justification for stretch-dominated plate lattices.'
                   ),
                   'loocv_r2': {k: float(v)
                                for k, v in method_scores.items()}}, f, indent=2)

    print("\n5. Generating validation plots...")
    plot_validation(results_dict,
                    OUTPUTS_DIR / 'fig_surrogate_validation.png')
    plot_curves(sc_df, fcc_df, ga_params,
                OUTPUTS_DIR / 'fig_surrogate_curves.png')

    predict_fn = make_predict_fn(ga_params, gpr_models, gpr_scalers,
                                  best_method)

    print("\n6. Sanity check predictions:")
    for lt in ['SC', 'FCC']:
        C11, C12, C44 = predict_fn(0.10, lt)
        print(f"  {lt} t/L=0.10: C11={C11:.4f}  C12={C12:.4f}  C44={C44:.4f}"
              f"  (C11>C12: {C11>C12})")

    # merge into property dataset
    prop_path = Path(args.property_dataset)
    if args.skip_merge or not prop_path.exists():
        if not prop_path.exists():
            print(f"\n7. Property dataset not found at {prop_path}")
            print("   Run acoustic TMM script first to generate it.")
            print("   Skipping merge — run again with --property_dataset "
                  "once it exists.")
        else:
            print("\n7. Skipping merge (--skip_merge flag set).")
    else:
        print(f"\n7. Merging mechanical properties into {prop_path}...")
        prop_df = pd.read_csv(prop_path)
        print(f"  Loaded {len(prop_df)} rows from property dataset")

        C11_col, C12_col, C44_col = [], [], []
        skipped = 0
        for _, row in prop_df.iterrows():
            lt = str(row.get('lattice_type', 'SC')).upper()
            t  = float(row['t_over_L'])
            rho_val = row.get('rho', None)
            rho_val = float(rho_val) if pd.notna(rho_val) else None

            # flag rows outside the fitted t/L range
            if t < 0.025 or t > 0.22:
                skipped += 1

            try:
                C11, C12, C44 = predict_fn(t, lt, rho=rho_val)
            except Exception:
                C11 = C12 = C44 = float('nan')

            C11_col.append(C11)
            C12_col.append(C12)
            C44_col.append(C44)

        prop_df['C11'] = C11_col
        prop_df['C12'] = C12_col
        prop_df['C44'] = C44_col
        prop_df['surrogate_method'] = best_method

        out_path = OUTPUTS_DIR / 'property_dataset_with_mechanical.csv'
        prop_df.to_csv(out_path, index=False)
        print(f"  Saved merged dataset: {out_path}  ({len(prop_df)} rows)")
        if skipped > 0:
            print(f"  Note: {skipped} rows had t/L outside training range "
                  f"[0.025, 0.22] — extrapolation used")

        print("\n  Mechanical property summary:")
        for lt in prop_df['lattice_type'].unique():
            sub = prop_df[prop_df['lattice_type'] == lt]
            print(f"    {lt}: C11 [{sub['C11'].min():.4f}, {sub['C11'].max():.4f}]"
                  f"  C44 [{sub['C44'].min():.4f}, {sub['C44'].max():.4f}]")

    print("\n" + "=" * 60)
    print("Done. Key outputs:")
    print(f"  {OUTPUTS_DIR}/surrogate_gibsonashby.json")
    print(f"  {OUTPUTS_DIR}/surrogate_gpr.pkl")
    print(f"  {OUTPUTS_DIR}/fig_surrogate_validation.png")
    print(f"  {OUTPUTS_DIR}/fig_surrogate_curves.png")
    if prop_path.exists() and not args.skip_merge:
        print(f"  {OUTPUTS_DIR}/property_dataset_with_mechanical.csv")
    print("=" * 60)
    print()
    print("Next step: multi-objective optimisation")
    print("  python scripts/run_optimisation.py \\")
    print(f"    --dataset {OUTPUTS_DIR}/property_dataset_with_mechanical.csv")

    return predict_fn, ga_params


if __name__ == "__main__":
    main()
