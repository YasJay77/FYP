"""Sanity-check the discrete Pareto front against a continuous SC-geometry optimisation.

N is integer so it's an outer loop; for fixed N the 3 vars [t/L, d, L] are a
clean 3D SLSQP problem solved per N. Two scalarisations together (weighted-sum +
eps-constraint) trace the whole front including concave bits weighted-sum misses.
Constraints g(x) >= 0: packaging N*L <= 200, pore independence L/d >= 6.
"""

import argparse
import json
import os
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
from scipy.optimize import minimize

SCRIPTS_DIR  = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS_DIR.parent
OUTPUTS_DIR  = PROJECT_ROOT / "results" / "property_datasets"
OUTPUTS_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(SCRIPTS_DIR))
from tmm_script import compute_alpha, sc_geometric_params, band_mean   # noqa: E402

# must match the dataset's frequency grid
F = np.linspace(200.0, 5000.0, 1000)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "legend.fontsize": 9,
    "figure.dpi": 150,
})

COLORS = {"SC": "#1f77b4", "FCC": "#ff7f0e", "FCC_face": "#2ca02c"}


def make_objectives(N, gpr_models, gpr_scalers, ga_sc, c11_max):
    """build the alpha and c11_norm objective fns for fixed N"""
    slope, intercept = ga_sc["rho_fit"]

    def band_mean_alpha(x):
        """mean absorption over the band"""
        t_over_L, d_mm, L_mm = x
        d_m = d_mm * 1e-3
        L_m = L_mm * 1e-3
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            alpha = compute_alpha(F, float(t_over_L), d_m, L_m, int(N))
        return float(band_mean(alpha, F))

    def c11_norm(x):
        """C11 normalised to [0,1]"""
        t_over_L = float(x[0])
        rho = float(np.clip(slope * t_over_L + intercept, 0.01, 1.0))
        # GPR features [rho, SC indicator]
        X = np.array([[rho, 1.0]])
        X_sc = gpr_scalers["C11"].transform(X)
        c11 = float(gpr_models["C11"].predict(X_sc)[0])
        return float(np.clip(c11 / c11_max, 0.0, 1.0))

    return band_mean_alpha, c11_norm


def make_constraints(N, max_thickness_mm):
    """SLSQP constraints for fixed N; each returns >= 0 when satisfied"""
    return [
        # packaging: N*L <= max
        {"type": "ineq", "fun": lambda x, N=N: max_thickness_mm - N * x[2]},
        # pore independence: L/d >= 6
        {"type": "ineq", "fun": lambda x: x[2] - 6.0 * x[1]},
    ]


def solve_subproblem(obj_fn, constraints, bounds, n_starts, rng):
    """minimise obj_fn under constraints with random restarts"""
    x_best = None
    val_best = np.inf

    lb = np.array([b[0] for b in bounds])
    ub = np.array([b[1] for b in bounds])

    for _ in range(n_starts):
        x0 = lb + rng.random(len(bounds)) * (ub - lb)
        try:
            res = minimize(
                obj_fn,
                x0,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"ftol": 1e-9, "maxiter": 300, "disp": False},
            )
        except Exception:
            continue

        if not res.success:
            continue

        # SLSQP can call a marginal point feasible, so re-check
        feasible = all(
            c["fun"](res.x) >= -1e-6 for c in constraints
        )
        if not feasible:
            continue

        if res.fun < val_best:
            val_best = res.fun
            x_best = res.x.copy()

    return x_best, val_best


def pareto_mask_2d(alpha_arr, c11_arr):
    """boolean mask of non-dominated points (maximise both)"""
    n = len(alpha_arr)
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


def run_continuous_optimisation(
    gpr_models, gpr_scalers, ga_sc, c11_max,
    n_levels, n_starts, max_thickness_mm, n_range, rng
):
    """weighted-sum and eps-constraint sweeps for each N"""
    bounds = [
        (0.03, 0.18),   # t_over_L
        (0.5,  5.0),    # d_mm
        (5.0,  50.0),   # L_mm
    ]

    rows = []

    for N in n_range:
        constraints = make_constraints(N, max_thickness_mm)
        alpha_fn, c11_fn = make_objectives(N, gpr_models, gpr_scalers, ga_sc, c11_max)

        # (a) weighted sum
        lambdas = np.linspace(0.0, 1.0, n_levels)
        for lam in lambdas:
            # negate, minimize() minimises
            def obj_ws(x, lam=lam):
                """weighted-sum scalarisation objective"""
                return -(lam * alpha_fn(x) + (1.0 - lam) * c11_fn(x))

            x_opt, _ = solve_subproblem(obj_ws, constraints, bounds, n_starts, rng)
            if x_opt is not None:
                rows.append({
                    "N":              int(N),
                    "t_over_L":       float(x_opt[0]),
                    "d_mm":           float(x_opt[1]),
                    "L_mm":           float(x_opt[2]),
                    "band_mean_alpha": alpha_fn(x_opt),
                    "C11_norm":       c11_fn(x_opt),
                    "method":         "weighted_sum",
                    "lambda_or_eps":  float(lam),
                })

        # (b) eps-constraint: max <alpha> s.t. C11_norm >= eps
        eps_c11 = np.linspace(0.05, 1.0, n_levels)
        for eps in eps_c11:
            eps_cons = constraints + [
                {"type": "ineq", "fun": lambda x, e=eps: c11_fn(x) - e}
            ]

            def obj_ep1(x):
                """eps-constraint objective: maximise alpha"""
                return -alpha_fn(x)

            x_opt, _ = solve_subproblem(obj_ep1, eps_cons, bounds, n_starts, rng)
            if x_opt is not None:
                rows.append({
                    "N":              int(N),
                    "t_over_L":       float(x_opt[0]),
                    "d_mm":           float(x_opt[1]),
                    "L_mm":           float(x_opt[2]),
                    "band_mean_alpha": alpha_fn(x_opt),
                    "C11_norm":       c11_fn(x_opt),
                    "method":         "eps_constraint",
                    "lambda_or_eps":  float(eps),
                })

        # (b) flip it: max C11_norm s.t. <alpha> >= eps
        eps_alpha = np.linspace(0.05, 0.75, n_levels)
        for eps in eps_alpha:
            eps_cons = constraints + [
                {"type": "ineq", "fun": lambda x, e=eps: alpha_fn(x) - e}
            ]

            def obj_ep2(x):
                """eps-constraint objective: maximise C11_norm"""
                return -c11_fn(x)

            x_opt, _ = solve_subproblem(obj_ep2, eps_cons, bounds, n_starts, rng)
            if x_opt is not None:
                rows.append({
                    "N":              int(N),
                    "t_over_L":       float(x_opt[0]),
                    "d_mm":           float(x_opt[1]),
                    "L_mm":           float(x_opt[2]),
                    "band_mean_alpha": alpha_fn(x_opt),
                    "C11_norm":       c11_fn(x_opt),
                    "method":         "eps_constraint",
                    "lambda_or_eps":  float(eps),
                })

        print(f"    N={N:2d}  collected {len(rows)} optima so far")

    return pd.DataFrame(rows)


def plot_overlay(df_all, df_cont_pareto, df_disc, save_path):
    """overlay figure: background scatter, continuous envelope, discrete front"""
    fig, ax = plt.subplots(figsize=(10, 7))

    # background: all dataset points
    for lt in ["SC", "FCC", "FCC_face"]:
        sub = df_all[df_all["lattice_type"] == lt]
        if len(sub) == 0:
            continue
        ax.scatter(
            sub["C11_norm"], sub["band_mean_alpha"],
            color=COLORS[lt], alpha=0.12, s=3,
            label=f"{lt} (all)" if lt == "SC" else f"_{lt} (all)",
        )

    # continuous envelope, segments coloured by N
    df_env = df_cont_pareto.sort_values("C11_norm").reset_index(drop=True)
    n_vals = df_env["N"].values
    n_min, n_max = int(n_vals.min()), int(n_vals.max())

    cmap_n = matplotlib.colormaps.get_cmap("plasma").resampled(n_max - n_min + 1)
    norm_n = mcolors.BoundaryNorm(
        boundaries=np.arange(n_min - 0.5, n_max + 1.5, 1.0),
        ncolors=cmap_n.N,
    )

    # connect consecutive envelope points
    pts = np.column_stack([df_env["C11_norm"].values, df_env["band_mean_alpha"].values])
    segs = [[pts[i], pts[i + 1]] for i in range(len(pts) - 1)]
    seg_n = (n_vals[:-1] + n_vals[1:]) / 2.0      # mid-N -> segment colour
    lc = LineCollection(segs, cmap=cmap_n, norm=norm_n, linewidths=2.5, zorder=6)
    lc.set_array(seg_n)
    ax.add_collection(lc)

    # envelope points coloured by N
    sc_env = ax.scatter(
        df_env["C11_norm"], df_env["band_mean_alpha"],
        c=df_env["N"], cmap=cmap_n, norm=norm_n,
        s=40, zorder=7, edgecolors="black", linewidths=0.4,
        label="Continuous Pareto envelope",
    )
    cbar = fig.colorbar(sc_env, ax=ax, pad=0.02)
    cbar.set_label("N (layers)", fontsize=10)

    # discrete front
    for lt in ["SC", "FCC", "FCC_face"]:
        sub = df_disc[df_disc["lattice_type"] == lt]
        if len(sub) == 0:
            continue
        sub_s = sub.sort_values("C11_norm")
        ax.scatter(
            sub_s["C11_norm"], sub_s["band_mean_alpha"],
            color=COLORS[lt], s=80, zorder=8,
            edgecolors="black", linewidths=0.8,
            label=f"{lt} (discrete Pareto)",
        )
        ax.plot(
            sub_s["C11_norm"], sub_s["band_mean_alpha"],
            color=COLORS[lt], lw=1.5, ls="--", zorder=7,
        )

    # annotate the largest alpha gap at matched C11_norm
    gap_info = compute_gap_annotation(df_cont_pareto, df_disc)
    if gap_info is not None:
        c11_at_gap, alpha_cont, alpha_disc, gap = gap_info
        ax.annotate(
            f"Max Δ⟨α⟩ = {gap:.4f}\nat C₁₁/max = {c11_at_gap:.3f}",
            xy=(c11_at_gap, alpha_cont),
            xytext=(c11_at_gap + 0.08, alpha_cont - 0.05),
            arrowprops=dict(arrowstyle="->", color="red", lw=1.2),
            fontsize=9, color="red",
            bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.85),
        )
        ax.annotate(
            "", xy=(c11_at_gap, alpha_disc),
            xytext=(c11_at_gap, alpha_cont),
            arrowprops=dict(arrowstyle="<->", color="red", lw=1.5),
        )

    # flag N values on the continuous front but off the discrete grid
    disc_N = set(df_disc["N"].unique()) if "N" in df_disc.columns else set()
    cont_N = set(df_cont_pareto["N"].unique())
    extra_N = sorted(cont_N - disc_N)
    if extra_N:
        ax.text(
            0.02, 0.02,
            f"N values on continuous front only: {extra_N}",
            transform=ax.transAxes, fontsize=8,
            va="bottom", color="purple",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
        )

    ax.set_xlabel("Normalised stiffness  C₁₁/C₁₁_max")
    ax.set_ylabel("Band-mean absorption  ⟨α⟩  [500–2000 Hz]")
    ax.set_title(
        "Continuous Optimisation vs Discrete Pareto Front\n"
        "SC plate lattice  (packaging: N×L ≤ 200 mm)"
    )
    ax.axhline(0.5, color="green", lw=1, ls="--", alpha=0.5)
    ax.text(0.02, 0.52, "α = 0.5 target", color="green", fontsize=8, alpha=0.7)
    ax.set_xlim(-0.02, 1.05)
    ax.set_ylim(-0.02, 1.05)
    ax.grid(True, alpha=0.25)

    # legend: lattice patches + the envelope
    from matplotlib.patches import Patch
    handles, labels = ax.get_legend_handles_labels()
    lattice_patches = [
        Patch(facecolor=COLORS[lt], label=f"{lt}", alpha=0.6)
        for lt in ["SC", "FCC", "FCC_face"]
    ]
    ax.legend(
        handles=handles + lattice_patches,
        labels=labels + [lt for lt in ["SC", "FCC", "FCC_face"]],
        loc="upper right", markerscale=1.5,
    )

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def compute_gap_annotation(df_cont, df_disc):
    """envelope point with the biggest <alpha> gain over the discrete front at matched C11_norm"""
    if len(df_disc) < 2:
        return None

    disc_s = df_disc.sort_values("C11_norm")
    disc_c11 = disc_s["C11_norm"].values
    disc_alpha = disc_s["band_mean_alpha"].values

    c11_min = disc_c11.min()
    c11_max = disc_c11.max()

    gaps = []
    for _, row in df_cont.iterrows():
        c = row["C11_norm"]
        if c < c11_min or c > c11_max:
            continue
        alpha_disc_interp = float(np.interp(c, disc_c11, disc_alpha))
        gap = row["band_mean_alpha"] - alpha_disc_interp
        gaps.append((c, row["band_mean_alpha"], alpha_disc_interp, gap))

    if not gaps:
        return None
    return max(gaps, key=lambda t: t[3])


def print_summary(df_cont_pareto, df_disc, max_thickness_mm):
    """concise validation summary to stdout"""
    print("\n" + "=" * 65)
    print("  Continuous Optimisation Summary")
    print("=" * 65)

    print(f"\n  Continuous Pareto points:  {len(df_cont_pareto)}")

    # N values on the front
    cont_N = sorted(int(v) for v in df_cont_pareto["N"].unique())
    print(f"  N values on continuous front: {cont_N}")

    # how often packaging binds
    if "L_mm" in df_cont_pareto.columns:
        thickness = df_cont_pareto["N"] * df_cont_pareto["L_mm"]
        n_active = (np.abs(thickness - max_thickness_mm) <= 1.0).sum()
        pct = 100.0 * n_active / max(len(df_cont_pareto), 1)
        print(f"  Packaging constraint active: {n_active}/{len(df_cont_pareto)} "
              f"({pct:.0f}%) of continuous Pareto points")

    # improvement in <alpha> over the discrete front
    if len(df_disc) >= 2:
        disc_s = df_disc.sort_values("C11_norm")
        disc_c11 = disc_s["C11_norm"].values
        disc_alpha = disc_s["band_mean_alpha"].values
        c11_min, c11_max_d = disc_c11.min(), disc_c11.max()

        improvements = []
        for _, row in df_cont_pareto.iterrows():
            c = row["C11_norm"]
            if c11_min <= c <= c11_max_d:
                alpha_d = float(np.interp(c, disc_c11, disc_alpha))
                improvements.append(row["band_mean_alpha"] - alpha_d)

        if improvements:
            arr = np.array(improvements)
            print(f"\n  ⟨α⟩ improvement over discrete front "
                  f"(at matched C₁₁_norm, {len(arr)} comparison points):")
            print(f"    Mean:  {arr.mean():+.4f}")
            print(f"    Max:   {arr.max():+.4f}")
            print(f"    Min:   {arr.min():+.4f}  (negative = discrete beats continuous)")

    # flag any non-SC on the discrete front
    print("\n  Discrete front lattice types:")
    for lt, cnt in df_disc["lattice_type"].value_counts().items():
        flag = "" if lt == "SC" else "  *** NON-SC LATTICE ON DISCRETE FRONT ***"
        print(f"    {lt}: {cnt} points{flag}")
    print()


def main():
    """parse args, run the continuous optimisation, save envelope + overlay figure"""
    parser = argparse.ArgumentParser(
        description=(
            "Validate the discrete Pareto front via continuous per-N "
            "multi-objective optimisation of the SC plate lattice."
        )
    )
    parser.add_argument(
        "--dataset",
        default=str(PROJECT_ROOT / "results" / "property_datasets" / "property_dataset_all_lattices.csv"),
        help="Full property dataset CSV (21,720 rows)",
    )
    parser.add_argument(
        "--pareto_csv",
        default=str(PROJECT_ROOT / "results" / "property_datasets" / "pareto_front.csv"),
        help="Discrete Pareto front CSV from run_optimisation.py",
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
        help="Packaging constraint: N × L_mm ≤ max_thickness (default 200 mm)",
    )
    parser.add_argument(
        "--n_levels", type=int, default=50,
        help="Number of λ / ε levels per sweep (default 50)",
    )
    parser.add_argument(
        "--n_starts", type=int, default=5,
        help="Random restarts per sub-problem (default 5)",
    )
    parser.add_argument(
        "--n_min", type=int, default=5,
        help="Minimum N (default 5)",
    )
    parser.add_argument(
        "--n_max", type=int, default=30,
        help="Maximum N (default 30)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--out_csv",
        default=str(OUTPUTS_DIR / "pareto_front_continuous.csv"),
        help="Output CSV for continuous Pareto envelope",
    )
    parser.add_argument(
        "--out_fig",
        default=str(OUTPUTS_DIR / "fig_pareto_continuous_overlay.png"),
        help="Output PNG for overlay figure",
    )
    args = parser.parse_args()

    print("=" * 65)
    print("  Continuous Optimisation — SC Plate Lattice Validation")
    print("  Objectives: maximise band_mean_alpha AND C11_norm")
    print("=" * 65)

    # load
    print("\n1. Loading inputs...")

    df_all = pd.read_csv(args.dataset)
    print(f"   Property dataset: {len(df_all):,} rows")

    # SC-only C11_max, matching run_optimisation.py
    c11_max = df_all[df_all["lattice_type"] == "SC"]["C11"].max()
    # should equal the global max while SC dominates mechanically
    global_c11_max = df_all["C11"].max()
    assert abs(c11_max - global_c11_max) < 1e-8, (
        f"SC C11 max ({c11_max:.6f}) ≠ global max ({global_c11_max:.6f}) — "
        "FCC or FCC_face now dominates mechanically; update normalisation."
    )
    print(f"   C11_max = {c11_max:.6f}  (SC, equals global max — confirmed)")

    df_all["C11_norm"] = df_all["C11"] / c11_max

    df_disc = pd.read_csv(args.pareto_csv)
    print(f"   Discrete Pareto front: {len(df_disc)} rows")
    disc_lt = df_disc["lattice_type"].value_counts().to_dict()
    print(f"   Discrete front lattice distribution: {disc_lt}")

    non_sc = {lt: cnt for lt, cnt in disc_lt.items() if lt != "SC"}
    if non_sc:
        print(f"   *** FLAG: FCC/FCC_face on discrete front: {non_sc} ***")
    else:
        print("   SC dominates discrete front — continuous optimisation targets SC only")

    with open(args.gpr_pkl, "rb") as f:
        gpr_data = pickle.load(f)
    gpr_models   = gpr_data["models"]["SC"]
    gpr_scalers  = gpr_data["scalers"]["SC"]
    print("   GPR models loaded (SC: C11, C12, C44)")

    with open(args.ga_json) as f:
        ga_params = json.load(f)
    ga_sc = ga_params["SC"]   # has rho_fit
    print(f"   rho_fit (SC): slope={ga_sc['rho_fit'][0]:.4f}, "
          f"intercept={ga_sc['rho_fit'][1]:.4f}")

    # run it
    n_range = range(args.n_min, args.n_max + 1)
    total_subproblems = len(n_range) * 3 * args.n_levels
    print(f"\n2. Running continuous optimisation...")
    print(f"   N range: [{args.n_min}, {args.n_max}]  ({len(n_range)} values)")
    print(f"   Levels per sweep: {args.n_levels}  ×  3 sweeps  × {len(n_range)} N values")
    print(f"   = {total_subproblems} sub-problems × {args.n_starts} random starts each")
    print(f"   = {total_subproblems * args.n_starts:,} SLSQP calls total\n")

    rng = np.random.default_rng(args.seed)

    df_optima = run_continuous_optimisation(
        gpr_models, gpr_scalers, ga_sc, c11_max,
        n_levels=args.n_levels,
        n_starts=args.n_starts,
        max_thickness_mm=args.max_thickness,
        n_range=n_range,
        rng=rng,
    )

    print(f"\n   Total successful optima: {len(df_optima)}")

    # filter to the envelope
    print("\n3. Filtering to non-dominated (Pareto) envelope...")
    if len(df_optima) == 0:
        print("   No feasible optima found — check constraints/bounds.")
        return

    pmask = pareto_mask_2d(
        df_optima["band_mean_alpha"].values,
        df_optima["C11_norm"].values,
    )
    df_cont_pareto = df_optima[pmask].copy().reset_index(drop=True)
    df_cont_pareto = df_cont_pareto.sort_values("C11_norm").reset_index(drop=True)
    print(f"   Continuous Pareto points: {len(df_cont_pareto)} "
          f"(from {len(df_optima)} total optima)")

    # save
    df_cont_pareto.to_csv(args.out_csv, index=False)
    print(f"\n4. Saved: {args.out_csv}")

    # overlay figure
    print("\n5. Generating overlay figure...")
    plot_overlay(df_all, df_cont_pareto, df_disc, args.out_fig)

    print_summary(df_cont_pareto, df_disc, args.max_thickness)


if __name__ == "__main__":
    main()
