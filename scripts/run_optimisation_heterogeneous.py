"""Heterogeneous SC plate-lattice panels - per-layer unit cell size L_i.

Continuous search here, unlike the homogeneous discrete case: for fixed N the
vars [L_1..L_N, t/L, d] are smooth, so gradient methods work. N is an integer
outer loop. Two scalarisations (weighted-sum + eps-constraint) span the full
front including concave bits. Each sub-problem: DE global search then SLSQP polish.

Stack stiffness uses the Reuss (series-compliance) bound for axial z-loading,
1/C11_stack = sum_i (L_i/sum_j L_j) * (1/C11_i). t/L is scalar here so C11_i is
equal across layers and this collapses to the single-layer value - kept general
for a future per-layer t/L. Use --quick for a smoke test.
"""

import argparse
import json
import pickle
import sys
import warnings
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize, NonlinearConstraint

SCRIPTS_DIR  = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS_DIR.parent
OUTPUTS_DIR  = PROJECT_ROOT / "results" / "property_datasets"
OUTPUTS_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(SCRIPTS_DIR))
from tmm_script import (                                        # noqa: E402
    heterogeneous_slab, input_impedance_rigid, alpha_from_Zin,
    band_mean, sc_geometric_params,
)

# must match the dataset's frequency grid
F = np.linspace(200.0, 5000.0, 1000)

plt.rcParams.update({
    "font.family": "serif",
    "font.size":   11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "legend.fontsize": 9,
    "figure.dpi": 150,
})

COLORS = {"SC": "#1f77b4", "FCC": "#ff7f0e", "FCC_face": "#2ca02c"}


def make_objectives(N, gpr_models, gpr_scalers, ga_sc, c11_max):
    """(alpha_fn, c11_norm_fn) for fixed N. x = [L_1..L_N, t_over_L, d_mm]"""
    slope, intercept = ga_sc["rho_fit"]

    def band_mean_alpha(x):
        """mean absorption over the band"""
        L_mm     = np.asarray(x[:N], dtype=float)
        t_over_L = float(x[N])
        d_m      = float(x[N + 1]) * 1e-3       # mm -> m
        L_m      = L_mm * 1e-3                  # mm -> m
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            Tslab = heterogeneous_slab(F, t_over_L, d_m, L_m)
            Zin   = input_impedance_rigid(Tslab)
            alpha = alpha_from_Zin(Zin)
        return float(band_mean(alpha, F))

    def c11_norm(x):
        """C11 normalised to [0,1]"""
        L_mm     = np.asarray(x[:N], dtype=float)
        t_over_L = float(x[N])
        # rel density from Gibson-Ashby fit; scalar t/L so same across layers
        rho = float(np.clip(slope * t_over_L + intercept, 0.01, 1.0))
        Xf  = np.array([[rho, 1.0]])            # [rho, SC indicator]
        c11_single = float(
            gpr_models["C11"].predict(gpr_scalers["C11"].transform(Xf))[0]
        )
        # Reuss series-compliance bound for axial z-loading; full form kept for
        # a future per-layer t/L (here it's just 1/c11_single)
        weights   = L_mm / np.sum(L_mm)
        c11_stack = 1.0 / float(np.sum(weights / c11_single))
        return float(np.clip(c11_stack / c11_max, 0.0, 1.0))

    return band_mean_alpha, c11_norm


def make_slsqp_constraints(N, max_thickness_mm=200.0):
    """SLSQP constraints; each returns >= 0 when satisfied"""
    return [
        # packaging: total stack thickness <= max
        {"type": "ineq", "fun": lambda x, N=N:
            max_thickness_mm - np.sum(x[:N])},
        # pore independence: L_i >= 6 d on every layer
        {"type": "ineq", "fun": lambda x, N=N:
            np.min(x[:N]) - 6.0 * x[N + 1]},
    ]


def make_de_constraints(N, max_thickness_mm=200.0):
    """same constraints as NonlinearConstraint, for DE"""
    return [
        NonlinearConstraint(
            lambda x, N=N: max_thickness_mm - np.sum(x[:N]),
            0.0, np.inf,
        ),
        NonlinearConstraint(
            lambda x, N=N: np.min(x[:N]) - 6.0 * x[N + 1],
            0.0, np.inf,
        ),
    ]


def augmented_slsqp_constraints(base_slsqp, extra_fn):
    """tack one extra ineq onto a base list (returns a new list)"""
    return base_slsqp + [{"type": "ineq", "fun": extra_fn}]


def augmented_de_constraints(base_de, extra_fn):
    """tack one extra NonlinearConstraint onto a base list (new list)"""
    return base_de + [NonlinearConstraint(extra_fn, 0.0, np.inf)]


def solve_two_stage(obj_fn, bounds, slsqp_cons, de_cons,
                    de_seed, de_popsize, de_maxiter, de_tol):
    """DE global search then SLSQP polish"""
    # stage 1: global DE
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            de_res = differential_evolution(
                obj_fn,
                bounds=bounds,
                constraints=de_cons,
                seed=de_seed,
                popsize=de_popsize,
                maxiter=de_maxiter,
                tol=de_tol,
                polish=False,
                mutation=(0.5, 1.0),
                recombination=0.7,
            )
    except Exception:
        return None, None

    x0 = de_res.x

    # stage 2: SLSQP polish from the DE point
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            slsqp_res = minimize(
                obj_fn,
                x0,
                method="SLSQP",
                bounds=bounds,
                constraints=slsqp_cons,
                options={"ftol": 1e-9, "maxiter": 500, "disp": False},
            )
        x_cand = slsqp_res.x
    except Exception:
        x_cand = x0

    # only keep the polish if it stays feasible
    def is_feasible(x):
        """packaging + pore-independence constraints all satisfied"""
        return all(c["fun"](x) >= -1e-6 for c in slsqp_cons)

    if is_feasible(x_cand):
        return x_cand.copy(), float(obj_fn(x_cand))

    # else fall back to DE
    if is_feasible(x0):
        return x0.copy(), float(obj_fn(x0))

    return None, None


def pareto_mask_2d(alpha_arr, c11_arr):
    """boolean mask of non-dominated points (maximise both)"""
    n    = len(alpha_arr)
    mask = np.ones(n, dtype=bool)
    for i in range(n):
        if not mask[i]:
            continue
        dominated = (
            (alpha_arr >= alpha_arr[i]) &
            (c11_arr   >= c11_arr[i]) &
            ((alpha_arr > alpha_arr[i]) | (c11_arr > c11_arr[i]))
        )
        dominated[i] = False
        if dominated.any():
            mask[i] = False
    return mask


def run_heterogeneous_optimisation(
    gpr_models, gpr_scalers, ga_sc, c11_max,
    n_levels, max_thickness_mm, n_range,
    de_seed, de_popsize, de_maxiter, de_tol,
):
    """weighted-sum and eps-constraint sweeps for each N. x = [L_1..L_N, t_over_L, d_mm]"""
    rows = []

    for N in n_range:
        bounds      = [(10.0, 40.0)] * N + [(0.03, 0.18), (0.5, 3.0)]
        slsqp_cons  = make_slsqp_constraints(N, max_thickness_mm)
        de_cons     = make_de_constraints(N, max_thickness_mm)
        alpha_fn, c11_fn = make_objectives(N, gpr_models, gpr_scalers, ga_sc, c11_max)

        n_before = len(rows)

        # (a) weighted sum
        for lam in np.linspace(0.0, 1.0, n_levels):
            def obj_ws(x, lam=lam):
                """weighted-sum scalarisation objective"""
                return -(lam * alpha_fn(x) + (1.0 - lam) * c11_fn(x))

            x_opt, _ = solve_two_stage(
                obj_ws, bounds, slsqp_cons, de_cons,
                de_seed, de_popsize, de_maxiter, de_tol,
            )
            if x_opt is not None:
                rows.append({
                    "N":              int(N),
                    "L_array":        json.dumps(x_opt[:N].tolist()),
                    "t_over_L":       float(x_opt[N]),
                    "d_mm":           float(x_opt[N + 1]),
                    "band_mean_alpha": alpha_fn(x_opt),
                    "C11_norm":       c11_fn(x_opt),
                    "method":         "weighted_sum",
                    "lambda_or_eps":  float(lam),
                })

        # (b) eps-constraint: max <alpha> s.t. C11_norm >= eps
        for eps in np.linspace(0.05, 1.0, n_levels):
            sc  = augmented_slsqp_constraints(
                slsqp_cons, lambda x, e=eps: c11_fn(x) - e
            )
            dc  = augmented_de_constraints(
                de_cons, lambda x, e=eps: c11_fn(x) - e
            )

            def obj_ep1(x):
                """eps-constraint objective: maximise alpha"""
                return -alpha_fn(x)

            x_opt, _ = solve_two_stage(
                obj_ep1, bounds, sc, dc,
                de_seed, de_popsize, de_maxiter, de_tol,
            )
            if x_opt is not None:
                rows.append({
                    "N":              int(N),
                    "L_array":        json.dumps(x_opt[:N].tolist()),
                    "t_over_L":       float(x_opt[N]),
                    "d_mm":           float(x_opt[N + 1]),
                    "band_mean_alpha": alpha_fn(x_opt),
                    "C11_norm":       c11_fn(x_opt),
                    "method":         "eps_C11",
                    "lambda_or_eps":  float(eps),
                })

        # (b) flip it: max C11_norm s.t. <alpha> >= eps
        for eps in np.linspace(0.05, 0.75, n_levels):
            sc  = augmented_slsqp_constraints(
                slsqp_cons, lambda x, e=eps: alpha_fn(x) - e
            )
            dc  = augmented_de_constraints(
                de_cons, lambda x, e=eps: alpha_fn(x) - e
            )

            def obj_ep2(x):
                """eps-constraint objective: maximise C11_norm"""
                return -c11_fn(x)

            x_opt, _ = solve_two_stage(
                obj_ep2, bounds, sc, dc,
                de_seed, de_popsize, de_maxiter, de_tol,
            )
            if x_opt is not None:
                rows.append({
                    "N":              int(N),
                    "L_array":        json.dumps(x_opt[:N].tolist()),
                    "t_over_L":       float(x_opt[N]),
                    "d_mm":           float(x_opt[N + 1]),
                    "band_mean_alpha": alpha_fn(x_opt),
                    "C11_norm":       c11_fn(x_opt),
                    "method":         "eps_alpha",
                    "lambda_or_eps":  float(eps),
                })

        n_found = len(rows) - n_before
        print(f"    N={N:2d}  found {n_found:3d} optima  "
              f"(total so far: {len(rows)})")

    return pd.DataFrame(rows)


def compute_gap(df_het, df_baseline):
    """Improvement of each het point over the baseline at matched C11_norm.

    List of (c11, alpha_het, alpha_base, gap, N).
    """
    if len(df_baseline) < 2:
        return []

    base_s    = df_baseline.sort_values("C11_norm")
    base_c11  = base_s["C11_norm"].values
    base_alph = base_s["band_mean_alpha"].values
    c11_lo, c11_hi = base_c11.min(), base_c11.max()

    gaps = []
    for _, row in df_het.iterrows():
        c = row["C11_norm"]
        if c < c11_lo or c > c11_hi:
            continue
        a_base = float(np.interp(c, base_c11, base_alph))
        gaps.append((c, row["band_mean_alpha"], a_base,
                     row["band_mean_alpha"] - a_base, int(row["N"])))
    return gaps


def plot_overlay(df_het_pareto, df_disc, df_cont_pareto, save_path):
    """overlay: discrete homog front, continuous envelope, het front (coloured by N)"""
    fig, ax = plt.subplots(figsize=(10, 7))

    # discrete homogeneous front
    for lt in ["SC", "FCC", "FCC_face"]:
        if "lattice_type" in df_disc.columns:
            sub = df_disc[df_disc["lattice_type"] == lt]
        else:
            sub = df_disc
        if len(sub) == 0:
            continue
        sub_s = sub.sort_values("C11_norm")
        ax.scatter(
            sub_s["C11_norm"], sub_s["band_mean_alpha"],
            color=COLORS.get(lt, "grey"), s=70, zorder=6,
            edgecolors="black", linewidths=0.7,
            label=f"{lt} (discrete front)",
        )
        ax.plot(
            sub_s["C11_norm"], sub_s["band_mean_alpha"],
            color=COLORS.get(lt, "grey"), lw=1.5, ls="--", zorder=5,
        )

    # continuous envelope (optional)
    if df_cont_pareto is not None and len(df_cont_pareto) >= 2:
        cont_s = df_cont_pareto.sort_values("C11_norm")
        ax.plot(
            cont_s["C11_norm"], cont_s["band_mean_alpha"],
            color="steelblue", lw=2.0, ls="--", zorder=5,
            label="Homogeneous continuous envelope",
        )

    # het front coloured by N
    df_env = df_het_pareto.sort_values("C11_norm").reset_index(drop=True)
    n_vals = df_env["N"].values.astype(float)
    n_min, n_max = int(n_vals.min()), int(n_vals.max())

    cmap_n = matplotlib.colormaps.get_cmap("plasma").resampled(n_max - n_min + 1)
    norm_n = mcolors.BoundaryNorm(
        boundaries=np.arange(n_min - 0.5, n_max + 1.5, 1.0),
        ncolors=cmap_n.N,
    )

    pts  = np.column_stack([df_env["C11_norm"].values,
                            df_env["band_mean_alpha"].values])
    segs = [[pts[i], pts[i + 1]] for i in range(len(pts) - 1)]
    seg_n = (n_vals[:-1] + n_vals[1:]) / 2.0
    lc = LineCollection(segs, cmap=cmap_n, norm=norm_n,
                        linewidths=2.5, zorder=7, label="_nolegend_")
    lc.set_array(seg_n)
    ax.add_collection(lc)

    sc_het = ax.scatter(
        df_env["C11_norm"], df_env["band_mean_alpha"],
        c=df_env["N"], cmap=cmap_n, norm=norm_n,
        s=50, zorder=8, edgecolors="crimson", linewidths=0.6,
        label="Heterogeneous Pareto front",
    )
    cbar = fig.colorbar(sc_het, ax=ax, pad=0.02)
    cbar.set_label("N (layers per panel)", fontsize=10)

    # annotate the gap vs whichever homogeneous baseline we have
    df_base = df_cont_pareto if (df_cont_pareto is not None and len(df_cont_pareto) >= 2) \
              else df_disc
    gaps = compute_gap(df_het_pareto, df_base)
    if gaps:
        best = max(gaps, key=lambda t: t[3])
        c11_at, alpha_het, alpha_base, gap, best_N = best
        if gap > 0:
            ax.annotate(
                f"Max Δ⟨α⟩ = {gap:+.4f}  (N={best_N})",
                xy=(c11_at, alpha_het),
                xytext=(c11_at + 0.09, alpha_het - 0.06),
                arrowprops=dict(arrowstyle="->", color="crimson", lw=1.2),
                fontsize=9, color="crimson",
                bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.85),
            )
            ax.annotate(
                "", xy=(c11_at, alpha_base),
                xytext=(c11_at, alpha_het),
                arrowprops=dict(arrowstyle="<->", color="crimson", lw=1.5),
            )

    ax.set_xlabel("Normalised stiffness  C₁₁/C₁₁_max")
    ax.set_ylabel("Band-mean absorption  ⟨α⟩  [500–2000 Hz]")
    ax.set_title(
        "Heterogeneous vs Homogeneous Pareto Fronts\n"
        "SC plate lattice  (packaging: ΣLᵢ ≤ 200 mm,  variable L per layer)"
    )
    ax.axhline(0.5, color="green", lw=1, ls="--", alpha=0.5)
    ax.text(0.02, 0.52, "α = 0.5 target", color="green", fontsize=8, alpha=0.7)
    ax.set_xlim(-0.02, 1.05)
    ax.set_ylim(-0.02, 1.05)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", markerscale=1.4)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def print_summary(df_pareto, df_disc, df_cont_pareto, max_thickness_mm):
    """concise summary to stdout"""
    print("\n" + "=" * 65)
    print("  Heterogeneous Optimisation Summary")
    print("=" * 65)

    print(f"\n  Heterogeneous Pareto points: {len(df_pareto)}")

    # N spread
    n_counts = df_pareto["N"].value_counts().sort_index()
    print(f"\n  N distribution across Pareto front:")
    for n_val, cnt in n_counts.items():
        print(f"    N={n_val:2d}: {cnt:3d} points")

    # improvement over the homogeneous baseline
    df_base  = df_cont_pareto if (df_cont_pareto is not None and len(df_cont_pareto) >= 2) \
               else df_disc
    base_lbl = "continuous envelope" if (df_cont_pareto is not None and len(df_cont_pareto) >= 2) \
               else "discrete front"
    gaps = compute_gap(df_pareto, df_base)
    if gaps:
        gap_vals = np.array([g[3] for g in gaps])
        print(f"\n  ⟨α⟩ improvement vs homogeneous {base_lbl} "
              f"({len(gaps)} comparison points):")
        print(f"    Mean improvement:  {gap_vals.mean():+.4f}")
        print(f"    Max  improvement:  {gap_vals.max():+.4f}")
        print(f"    Min  improvement:  {gap_vals.min():+.4f}  "
              f"({'negative = homogeneous beats heterogeneous' if gap_vals.min() < 0 else 'all positive'})")

    # best acoustic point
    best_row = df_pareto.loc[df_pareto["band_mean_alpha"].idxmax()]
    L_arr    = np.array(json.loads(best_row["L_array"]))
    L_sorted = np.sort(L_arr)
    spread   = L_arr.max() - L_arr.min()
    is_mono  = np.all(np.diff(L_arr) >= 0) or np.all(np.diff(L_arr) <= 0)
    pattern  = "monotonically ordered" if is_mono else "interleaved"

    print(f"\n  Best ⟨α⟩ point:")
    print(f"    ⟨α⟩ = {best_row['band_mean_alpha']:.4f}   "
          f"C₁₁_norm = {best_row['C11_norm']:.4f}   "
          f"N = {int(best_row['N'])}")
    print(f"    L_i (sorted, mm): {np.round(L_sorted, 2).tolist()}")
    print(f"    L spread:  {spread:.2f} mm   ({pattern})")
    print(f"    t/L = {best_row['t_over_L']:.4f}   d = {best_row['d_mm']:.3f} mm")

    # how often packaging binds
    def total_L(row):
        """stack thickness"""
        return float(np.sum(json.loads(row["L_array"])))

    thick = df_pareto.apply(total_L, axis=1)
    n_active = int((np.abs(thick - max_thickness_mm) <= 1.0).sum())
    pct      = 100.0 * n_active / max(len(df_pareto), 1)
    print(f"\n  Packaging constraint active (within 1 mm of {max_thickness_mm:.0f} mm): "
          f"{n_active}/{len(df_pareto)} ({pct:.0f}%)")
    print()


def main():
    """parse args, run the heterogeneous optimisation, save front + overlay figure"""
    parser = argparse.ArgumentParser(
        description=(
            "Multi-objective Pareto optimisation for heterogeneous SC plate "
            "lattice panels with variable L per layer."
        )
    )
    parser.add_argument(
        "--dataset",
        default=str(PROJECT_ROOT / "results" / "property_datasets" / "property_dataset_all_lattices.csv"),
        help="Full property dataset CSV (for C11_max normalisation)",
    )
    parser.add_argument(
        "--pareto_csv",
        default=str(OUTPUTS_DIR / "pareto_front.csv"),
        help="Discrete homogeneous Pareto front CSV",
    )
    parser.add_argument(
        "--cont_csv",
        default=str(OUTPUTS_DIR / "pareto_front_continuous.csv"),
        help="Continuous homogeneous Pareto envelope CSV (optional overlay)",
    )
    parser.add_argument(
        "--gpr_pkl",
        default=str(OUTPUTS_DIR / "surrogate_gpr.pkl"),
        help="Fitted GPR surrogate pickle",
    )
    parser.add_argument(
        "--ga_json",
        default=str(OUTPUTS_DIR / "surrogate_gibsonashby.json"),
        help="Gibson-Ashby params JSON (contains rho_fit)",
    )
    parser.add_argument(
        "--max_thickness", type=float, default=200.0,
        help="Packaging constraint: ΣL_i ≤ max_thickness mm (default 200)",
    )
    parser.add_argument(
        "--n_levels", type=int, default=40,
        help="Number of λ / ε levels per sweep (default 40)",
    )
    parser.add_argument(
        "--n_min", type=int, default=4,
        help="Minimum N (layers per panel, default 4)",
    )
    parser.add_argument(
        "--n_max", type=int, default=12,
        help="Maximum N (layers per panel, default 12)",
    )
    parser.add_argument(
        "--de_seed", type=int, default=0,
        help="Random seed for DE (default 0)",
    )
    parser.add_argument(
        "--de_popsize", type=int, default=15,
        help="DE population size multiplier (default 15)",
    )
    parser.add_argument(
        "--de_maxiter", type=int, default=200,
        help="DE maximum iterations (default 200)",
    )
    parser.add_argument(
        "--de_tol", type=float, default=1e-6,
        help="DE convergence tolerance (default 1e-6)",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Fast smoke-test mode: popsize=5, maxiter=50, n_levels=8",
    )
    parser.add_argument(
        "--out_csv",
        default=str(OUTPUTS_DIR / "pareto_front_heterogeneous.csv"),
        help="Output CSV for heterogeneous Pareto front",
    )
    parser.add_argument(
        "--out_fig",
        default=str(OUTPUTS_DIR / "fig_pareto_heterogeneous_overlay.png"),
        help="Output PNG for overlay figure",
    )
    args = parser.parse_args()

    if args.quick:
        args.de_popsize = 5
        args.de_maxiter = 50
        args.n_levels   = 8
        print("  [--quick] DE: popsize=5, maxiter=50, n_levels=8")

    print("=" * 65)
    print("  Heterogeneous Pareto Optimisation — SC Plate Lattice")
    print("  Objectives: maximise band_mean_alpha AND C11_norm")
    print("  Method:     DE + SLSQP two-stage, scalarisation sweeps")
    print("=" * 65)

    # load
    print("\n1. Loading inputs...")

    df_all  = pd.read_csv(args.dataset)
    c11_max = float(df_all[df_all["lattice_type"] == "SC"]["C11"].max())
    print(f"   Property dataset: {len(df_all):,} rows")
    print(f"   C11_max (SC) = {c11_max:.6f}")

    df_disc = pd.read_csv(args.pareto_csv)
    if "C11" in df_disc.columns and "C11_norm" not in df_disc.columns:
        df_disc["C11_norm"] = df_disc["C11"] / c11_max
    print(f"   Discrete Pareto front: {len(df_disc)} rows")

    df_cont_pareto = None
    if Path(args.cont_csv).exists():
        df_cont_pareto = pd.read_csv(args.cont_csv)
        print(f"   Continuous envelope: {len(df_cont_pareto)} rows")
    else:
        print(f"   Continuous envelope: not found at {args.cont_csv} — skipped")

    with open(args.gpr_pkl, "rb") as fh:
        gpr_data    = pickle.load(fh)
    gpr_models  = gpr_data["models"]["SC"]
    gpr_scalers = gpr_data["scalers"]["SC"]
    print("   GPR models loaded (SC: C11, C12, C44)")

    with open(args.ga_json) as fh:
        ga_params = json.load(fh)
    ga_sc = ga_params["SC"]
    print(f"   rho_fit (SC): slope={ga_sc['rho_fit'][0]:.4f}, "
          f"intercept={ga_sc['rho_fit'][1]:.4f}")

    # run it
    n_range = range(args.n_min, args.n_max + 1)
    n_N     = len(n_range)
    sweeps  = 3   # weighted_sum + eps_C11 + eps_alpha
    total   = n_N * sweeps * args.n_levels
    print(f"\n2. Running heterogeneous optimisation...")
    print(f"   N range: [{args.n_min}, {args.n_max}]  ({n_N} values)")
    print(f"   Levels per sweep: {args.n_levels}  ×  {sweeps} sweeps  × {n_N} N values")
    print(f"   = {total} DE+SLSQP sub-problems total")
    print(f"   DE: seed={args.de_seed}, popsize={args.de_popsize}, "
          f"maxiter={args.de_maxiter}\n")

    df_optima = run_heterogeneous_optimisation(
        gpr_models, gpr_scalers, ga_sc, c11_max,
        n_levels        = args.n_levels,
        max_thickness_mm= args.max_thickness,
        n_range         = n_range,
        de_seed         = args.de_seed,
        de_popsize      = args.de_popsize,
        de_maxiter      = args.de_maxiter,
        de_tol          = args.de_tol,
    )
    print(f"\n   Total successful optima: {len(df_optima)}")

    # filter to the envelope
    print("\n3. Filtering to non-dominated Pareto envelope...")
    if len(df_optima) == 0:
        print("   No feasible optima found — check constraints / bounds.")
        return

    pmask = pareto_mask_2d(
        df_optima["band_mean_alpha"].values,
        df_optima["C11_norm"].values,
    )
    df_het_pareto = (
        df_optima[pmask]
        .copy()
        .sort_values("C11_norm")
        .reset_index(drop=True)
    )
    print(f"   Heterogeneous Pareto points: {len(df_het_pareto)} "
          f"(from {len(df_optima)} total optima)")

    # save
    df_het_pareto.to_csv(args.out_csv, index=False)
    print(f"\n4. Saved: {args.out_csv}")

    # overlay figure
    print("\n5. Generating overlay figure...")
    plot_overlay(df_het_pareto, df_disc, df_cont_pareto, args.out_fig)

    print_summary(df_het_pareto, df_disc, df_cont_pareto, args.max_thickness)

    print("=" * 65)
    print("Done.")
    print(f"  {args.out_csv}")
    print(f"  {args.out_fig}")
    print("=" * 65)


if __name__ == "__main__":
    main()
