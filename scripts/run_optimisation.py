"""Stiffness vs absorption Pareto front over the plate-lattice property dataset.

Runs in property space, not geometry - pick the best (C11, ..., alpha) targets
here, recover geometry with the inversion net downstream. Grid is pre-computed,
so the front is just exact non-dominated sorting; a gradient search only earns
its keep when the grid is too big to enumerate (the heterogeneous case).
Both objectives maximised. Packaging constraint: N x L_mm <= max_thickness_mm.
"""

import argparse
import os
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "legend.fontsize": 9,
    "figure.dpi": 150,
})

SCRIPTS_DIR  = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS_DIR.parent
OUTPUTS_DIR  = PROJECT_ROOT / "results" / "property_datasets"
OUTPUTS_DIR.mkdir(exist_ok=True)


def fast_non_dominated_sort(costs):
    """fast non-dominated sort (assumes minimisation, so negate to maximise)"""
    n = len(costs)
    domination_count = np.zeros(n, dtype=int)   # how many points dominate i
    dominated_set    = [[] for _ in range(n)]   # points i dominates

    for i in range(n):
        for j in range(i + 1, n):
            ci, cj = costs[i], costs[j]
            # does j dominate i?
            if np.all(cj <= ci) and np.any(cj < ci):
                domination_count[i] += 1
                dominated_set[j].append(i)
            # or i dominate j?
            elif np.all(ci <= cj) and np.any(ci < cj):
                domination_count[j] += 1
                dominated_set[i].append(j)

    fronts = []
    rank   = np.zeros(n, dtype=int)
    current_front = [i for i in range(n) if domination_count[i] == 0]
    r = 1
    while current_front:
        fronts.append(current_front)
        for i in current_front:
            rank[i] = r
        next_front = []
        for i in current_front:
            for j in dominated_set[i]:
                domination_count[j] -= 1
                if domination_count[j] == 0:
                    next_front.append(j)
        current_front = next_front
        r += 1

    return fronts, rank


def crowding_distance(costs, front_indices):
    """crowding distance - ranks points within a front, favouring spread"""
    n = len(front_indices)
    if n <= 2:
        return np.full(n, np.inf)

    dist = np.zeros(n)
    sub  = costs[front_indices]

    for m in range(sub.shape[1]):
        order = np.argsort(sub[:, m])
        dist[order[0]]  = np.inf
        dist[order[-1]] = np.inf
        rng = sub[order[-1], m] - sub[order[0], m]
        if rng == 0:
            continue
        for k in range(1, n - 1):
            dist[order[k]] += (sub[order[k + 1], m] - sub[order[k - 1], m]) / rng

    return dist


def vectorised_pareto_front(df, obj1='band_mean_alpha', obj2='C11'):
    """boolean mask of Pareto-optimal rows (maximises both objectives)"""
    costs = -df[[obj1, obj2]].values   # negate so maximise -> minimise

    n = len(costs)
    is_efficient = np.ones(n, dtype=bool)

    for i in range(n):
        if not is_efficient[i]:
            continue
        others_idx = np.where(is_efficient)[0]
        others     = costs[others_idx]
        dominates_i = (np.all(others <= costs[i], axis=1) &
                       np.any(others  < costs[i], axis=1))
        # don't count self
        self_mask = others_idx == i
        dominates_i[self_mask] = False
        if dominates_i.any():
            is_efficient[i] = False

    return is_efficient


def apply_constraints(df, max_thickness_mm):
    """packaging constraint N x L_mm <= max_thickness_mm"""
    df = df.copy()
    df['thickness_mm'] = df['N'] * df['L_mm']
    mask = df['thickness_mm'] <= max_thickness_mm
    n_removed = (~mask).sum()
    df_filt = df[mask].copy()
    return df_filt, n_removed


def weighted_scalarisation_sweep(df, n_weights=100,
                                  obj1='band_mean_alpha_norm',
                                  obj2='C11_norm'):
    """sweep lambda in [0,1] to trace the front, picking up intermediate points"""
    pareto_indices = set()
    weights = np.linspace(0, 1, n_weights)

    o1 = df[obj1].values
    o2 = df[obj2].values

    for lam in weights:
        scalarised = lam * o1 + (1 - lam) * o2
        best_idx   = np.argmax(scalarised)
        pareto_indices.add(best_idx)

    return df.iloc[sorted(pareto_indices)].copy()


COLORS = {'SC': '#1f77b4', 'FCC': '#ff7f0e', 'FCC_face': '#2ca02c'}


def plot_pareto_front(df_all, df_pareto, save_path, max_thickness_mm):
    """main Pareto figure: acoustic vs mechanical property space"""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # left: full property space + front
    ax = axes[0]
    for lt in ['SC', 'FCC', 'FCC_face']:
        sub = df_all[df_all['lattice_type'] == lt]
        if len(sub) == 0:
            continue
        ax.scatter(sub['C11_norm'], sub['band_mean_alpha'],
                   color=COLORS[lt], alpha=0.15, s=4, label=f'{lt} (all)')

    # front points
    for lt in ['SC', 'FCC', 'FCC_face']:
        sub = df_pareto[df_pareto['lattice_type'] == lt]
        if len(sub) > 0:
            sub_s = sub.sort_values('C11_norm')
            ax.scatter(sub_s['C11_norm'], sub_s['band_mean_alpha'],
                       color=COLORS[lt], s=60, zorder=5,
                       edgecolors='black', linewidths=0.6,
                       label=f'{lt} (Pareto)')
            ax.plot(sub_s['C11_norm'], sub_s['band_mean_alpha'],
                    color=COLORS[lt], lw=1.5, zorder=4)

    ax.set_xlabel("Normalised stiffness  C₁₁/C₁₁_max")
    ax.set_ylabel("Band-mean absorption  ⟨α⟩  [500–2000 Hz]")
    ax.set_title(f"Pareto Front — Acoustic vs Mechanical\n"
                 f"(packaging: N×L ≤ {max_thickness_mm} mm)")
    ax.legend(markerscale=2)
    ax.grid(True, alpha=0.25)
    ax.set_xlim(-0.02, 1.05)
    ax.set_ylim(-0.02, 1.05)

    ax.axhline(0.5, color='green', lw=1, ls='--', alpha=0.5)
    ax.text(0.02, 0.52, 'α = 0.5 target', color='green', fontsize=8, alpha=0.7)

    # right: front properties vs design params
    ax2 = axes[1]
    df_p = df_pareto.sort_values('band_mean_alpha', ascending=False)

    x = np.arange(len(df_p))
    width = 0.35
    bars1 = ax2.bar(x - width/2, df_p['band_mean_alpha'],
                    width, label='Band-mean α', color='steelblue', alpha=0.8)
    ax2b = ax2.twinx()
    bars2 = ax2b.bar(x + width/2, df_p['C11_norm'],
                     width, label='C₁₁/C₁₁_max', color='darkorange', alpha=0.8)

    ax2.set_xlabel("Pareto point index (sorted by ⟨α⟩)")
    ax2.set_ylabel("Band-mean α", color='steelblue')
    ax2b.set_ylabel("Normalised C₁₁", color='darkorange')
    ax2.set_title("Pareto-Optimal Points — Property Values")
    ax2.set_ylim(0, 1)
    ax2b.set_ylim(0, 1)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"P{i+1}" for i in range(len(df_p))], fontsize=8)
    ax2.grid(True, alpha=0.2, axis='y')

    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2b.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_pareto_geometry(df_pareto, save_path):
    """design-parameter distributions across the Pareto points"""
    df_p = df_pareto.sort_values('band_mean_alpha', ascending=False).reset_index(drop=True)
    n    = len(df_p)
    idx  = np.arange(n)
    labels = [f"P{i+1}" for i in range(n)]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("Design Parameters of Pareto-Optimal Points", fontsize=13)

    params = [
        ('t_over_L',       't/L  (plate thickness ratio)'),
        ('d_mm',           'Pore diameter d  (mm)'),
        ('L_mm',           'Unit cell size L  (mm)'),
        ('N',              'Number of layers N'),
        ('thickness_mm',   'Total thickness N×L  (mm)'),
        ('band_mean_alpha','Band-mean α  [500–2000 Hz]'),
    ]

    for ax, (col, label) in zip(axes.flat, params):
        colors = [COLORS.get(lt, 'grey') for lt in df_p['lattice_type']]
        bars = ax.bar(idx, df_p[col], color=colors, alpha=0.8, edgecolor='black', lw=0.4)
        ax.set_xticks(idx)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.grid(True, alpha=0.2, axis='y')
        for bar, val in zip(bars, df_p[col]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                    f'{val:.3f}' if col != 'N' else f'{int(val)}',
                    ha='center', va='bottom', fontsize=7)

    # legend: only types actually on the front
    from matplotlib.patches import Patch
    present = df_pareto['lattice_type'].unique()
    handles = [Patch(facecolor=COLORS[lt], label=lt)
               for lt in COLORS if lt in present]
    fig.legend(handles=handles, loc='lower right', fontsize=10)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path}")


def main():
    """parse args and run the property-space pareto optimisation end to end"""
    parser = argparse.ArgumentParser(
        description="Multi-objective property-space optimisation for plate lattice metamaterials."
    )
    parser.add_argument('--dataset', type=str,
        default=str(PROJECT_ROOT / 'results' / 'property_datasets' / 'property_dataset_all_lattices.csv'),
        help='Merged property dataset CSV (all three lattice types, fully populated)')
    parser.add_argument('--max_thickness', type=float, default=200.0,
        help='Maximum panel thickness N×L in mm (default 200)')
    parser.add_argument('--top_n', type=int, default=None,
        help='Keep only top N Pareto points by crowding distance (default: all)')
    parser.add_argument('--out_pareto', type=str,
        default=str(OUTPUTS_DIR / 'pareto_front.csv'),
        help='Output CSV for Pareto front points')
    args = parser.parse_args()

    print("=" * 65)
    print("  Multi-Objective Property-Space Optimisation")
    print("  Objectives: maximise band_mean_alpha AND C11")
    print("=" * 65)

    # load
    print(f"\n1. Loading dataset: {args.dataset}")
    df = pd.read_csv(args.dataset)
    print(f"   {len(df):,} rows, {df['lattice_type'].nunique()} lattice types")
    print(f"   band_mean_alpha: [{df['band_mean_alpha'].min():.4f}, {df['band_mean_alpha'].max():.4f}]")
    print(f"   C11:             [{df['C11'].min():.4f}, {df['C11'].max():.4f}]")

    # packaging constraint
    print(f"\n2. Applying packaging constraint: N × L ≤ {args.max_thickness:.0f} mm")
    df_filt, n_removed = apply_constraints(df, args.max_thickness)
    print(f"   Removed {n_removed:,} rows  →  {len(df_filt):,} remaining")
    print(f"   Lattice types: {df_filt.lattice_type.value_counts().to_dict()}")

    # normalise objectives to [0,1]
    print("\n3. Normalising objectives to [0, 1]")
    # global max so all three lattice types share one scale
    c11_max = df['C11'].max()
    df_filt = df_filt.copy()
    df_filt['C11_norm'] = df_filt['C11'] / c11_max
    # alpha already in [0,1]
    df_filt['band_mean_alpha_norm'] = df_filt['band_mean_alpha']
    print(f"   C11 normalised by {c11_max:.4f} (global max)")

    # the front
    print("\n4. Computing Pareto front (non-dominated sorting)...")
    pareto_mask = vectorised_pareto_front(
        df_filt, obj1='band_mean_alpha', obj2='C11_norm'
    )
    df_pareto = df_filt[pareto_mask].copy()
    print(f"   Pareto front: {len(df_pareto)} points out of {len(df_filt):,}")
    print(f"   Lattice distribution: {df_pareto.lattice_type.value_counts().to_dict()}")

    # scalarisation sweep
    print("\n5. Weighted scalarisation sweep (λ ∈ [0,1], 200 weights)...")
    df_scalar = weighted_scalarisation_sweep(
        df_filt, n_weights=200,
        obj1='band_mean_alpha_norm', obj2='C11_norm'
    )
    # union with the front
    combined_idx = set(df_pareto.index) | set(df_scalar.index)
    df_pareto_full = df_filt.loc[sorted(combined_idx)].copy()
    # re-filter the union down to non-dominated
    pareto_mask2 = vectorised_pareto_front(
        df_pareto_full, obj1='band_mean_alpha', obj2='C11_norm'
    )
    df_pareto_full = df_pareto_full[pareto_mask2].copy()
    print(f"   After scalarisation union: {len(df_pareto_full)} Pareto points")

    # optionally trim by crowding distance
    if args.top_n is not None and len(df_pareto_full) > args.top_n:
        costs = -df_pareto_full[['band_mean_alpha', 'C11_norm']].values
        cd = crowding_distance(costs, np.arange(len(df_pareto_full)))
        top_idx = np.argsort(cd)[::-1][:args.top_n]
        df_pareto_full = df_pareto_full.iloc[sorted(top_idx)].copy()
        print(f"   Trimmed to top {args.top_n} by crowding distance")

    df_pareto_full = df_pareto_full.sort_values(
        'band_mean_alpha', ascending=False
    ).reset_index(drop=True)

    # summary table
    print("\n6. Pareto-optimal designs (sorted by acoustic performance):")
    print(f"\n   {'#':>3}  {'Type':>4}  {'t/L':>6}  {'d(mm)':>7}  {'L(mm)':>7}  "
          f"{'N':>3}  {'NxL(mm)':>8}  {'⟨α⟩':>7}  {'C11/max':>8}")
    print("   " + "-" * 72)
    for i, row in df_pareto_full.iterrows():
        print(f"   {int(i)+1:>3}  {row['lattice_type']:>4}  "
              f"{row['t_over_L']:>6.4f}  {row['d_mm']:>7.3f}  "
              f"{row['L_mm']:>7.3f}  {int(row['N']):>3}  "
              f"{row['thickness_mm']:>8.1f}  "
              f"{row['band_mean_alpha']:>7.4f}  {row['C11_norm']:>8.4f}")

    print(f"\n   Best acoustic:   ⟨α⟩ = {df_pareto_full['band_mean_alpha'].max():.4f}")
    print(f"   Best stiffness:  C11/max = {df_pareto_full['C11_norm'].max():.4f}")
    print(f"   Trade-off range: Δ⟨α⟩ = "
          f"{df_pareto_full['band_mean_alpha'].max() - df_pareto_full['band_mean_alpha'].min():.4f}  "
          f"ΔC11/max = "
          f"{df_pareto_full['C11_norm'].max() - df_pareto_full['C11_norm'].min():.4f}")

    # save csv
    out_cols = ['lattice_type', 't_over_L', 'd_mm', 'L_mm', 'N',
                'thickness_mm', 'band_mean_alpha', 'alpha_1000hz',
                'C11', 'C12', 'C44', 'C11_norm', 'band_mean_alpha_norm']
    out_cols = [c for c in out_cols if c in df_pareto_full.columns]
    df_pareto_full[out_cols].to_csv(args.out_pareto, index=False)
    print(f"\n7. Saved Pareto front: {args.out_pareto}")
    print(f"   ({len(df_pareto_full)} rows, columns: {out_cols})")

    # plots
    print("\n8. Generating plots...")
    plot_pareto_front(
        df_filt, df_pareto_full,
        OUTPUTS_DIR / 'fig_pareto_front.png',
        args.max_thickness
    )
    plot_pareto_geometry(
        df_pareto_full,
        OUTPUTS_DIR / 'fig_pareto_geometry.png'
    )

    print("\n" + "=" * 65)
    print("Done.")
    print(f"  {args.out_pareto}")
    print(f"  {OUTPUTS_DIR}/fig_pareto_front.png")
    print(f"  {OUTPUTS_DIR}/fig_pareto_geometry.png")
    print("=" * 65)
    print()
    print("Next step: train inversion network")
    print("  python scripts/train_inversion.py \\")
    print(f"    --pareto_csv {args.out_pareto} \\")
    print(f"    --dataset    {args.dataset}")


if __name__ == "__main__":
    main()