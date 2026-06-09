"""Re-run the property-space optimisation with the pore knockdown applied.

Gives the pore-resolved Pareto front and shows how the mix shifts vs the
pore-free baseline. Read-only on the repo, only writes into pore_investigation/.
Front logic is lifted from run_optimisation.py.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
# pore-free baseline on the 21,720-row dataset (22-pt front)
DATA = ROOT / "results" / "property_datasets" / "property_dataset_spectral.csv"
FITS = ROOT / "results" / "pore_investigation" / "knockdown_fits.json"
MAX_THICK = 200.0


def _pow(x, a, p):
    """power-law knockdown 1 - a x^p"""
    return 1.0 - a * np.power(np.maximum(x, 0.0), p)
def _quad(x, b, c):
    """quadratic knockdown 1 - b x - c x^2"""
    return 1.0 - b * x - c * x * x
def _exp(x, a, p):
    """exponential knockdown exp(-a x^p)"""
    return np.exp(-a * np.power(np.maximum(x, 0.0), p))
_FORM = {"A_power_minus": _pow, "B_quadratic": _quad, "C_exp_power": _exp}


def knockdown(fits, family, comp, dL):
    """evaluate the fitted knockdown k for a family/component at d/L"""
    info = fits[family][f"k_{comp}"]
    return _FORM[info["best"]](np.asarray(dL, float), *info["params"])


def apply_constraints(df, max_thickness_mm):
    """drop designs whose total thickness exceeds the cap"""
    df = df.copy()
    df["thickness_mm"] = df["N"] * df["L_mm"]
    return df[df["thickness_mm"] <= max_thickness_mm].copy()


def vectorised_pareto_front(df, obj1="band_mean_alpha", obj2="C11_norm"):
    """boolean mask of the non-dominated rows for the two objectives"""
    costs = -df[[obj1, obj2]].values
    n = len(costs)
    eff = np.ones(n, dtype=bool)
    for i in range(n):
        if not eff[i]:
            continue
        idx = np.where(eff)[0]
        oth = costs[idx]
        dom = np.all(oth <= costs[i], axis=1) & np.any(oth < costs[i], axis=1)
        dom[idx == i] = False
        if dom.any():
            eff[i] = False
    return eff


def weighted_scalarisation_sweep(df, n_weights=200, obj1="band_mean_alpha_norm", obj2="C11_norm"):
    """sweep the scalarisation weight and collect each weighted-best design"""
    idxs = set()
    o1, o2 = df[obj1].values, df[obj2].values
    for lam in np.linspace(0, 1, n_weights):
        idxs.add(int(np.argmax(lam * o1 + (1 - lam) * o2)))
    return df.iloc[sorted(idxs)].copy()


def compute_front(df_full, c11_col="C11"):
    """build the combined Pareto + scalarisation front for one dataset"""
    c11_max = df_full[c11_col].max()
    df = apply_constraints(df_full, MAX_THICK)
    df["C11_norm"] = df[c11_col] / c11_max
    df["band_mean_alpha_norm"] = df["band_mean_alpha"]
    mask = vectorised_pareto_front(df, "band_mean_alpha", "C11_norm")
    dp = df[mask].copy()
    ds = weighted_scalarisation_sweep(df, 200, "band_mean_alpha_norm", "C11_norm")
    comb = sorted(set(dp.index) | set(ds.index))
    dpf = df.loc[comb].copy()
    dpf = dpf[vectorised_pareto_front(dpf, "band_mean_alpha", "C11_norm")].copy()
    return dpf.sort_values("band_mean_alpha", ascending=False).reset_index(drop=True)


def main():
    """run pore-free and pore-resolved fronts and report the lattice-mix shift"""
    fits = json.load(open(FITS))
    df = pd.read_csv(DATA)
    print(f"dataset: {len(df):,} rows  {DATA.name}")

    pf = compute_front(df, "C11")
    mix_pf = pf.lattice_type.value_counts().to_dict()
    print("[pore-free]   front points:", len(pf), " mix:", mix_pf)

    dpc = df.copy()
    dpc["d_over_L"] = dpc["d_mm"] / dpc["L_mm"]
    for comp in ("C11", "C12", "C44"):
        k = np.array([knockdown(fits, fam, comp, d)
                      for fam, d in zip(dpc["lattice_type"], dpc["d_over_L"])])
        dpc[comp] = dpc[comp].values * k

    pr = compute_front(dpc, "C11")
    mix_pr = pr.lattice_type.value_counts().to_dict()
    print("[pore-resolved] front points:", len(pr), " mix:", mix_pr)

    cols = ["lattice_type", "t_over_L", "d_mm", "L_mm", "N", "thickness_mm",
            "band_mean_alpha", "alpha_1000hz", "C11", "C12", "C44", "C11_norm"]
    pf[[c for c in cols if c in pf]].to_csv(HERE / "pareto_front_porefree_repro.csv", index=False)
    pr[[c for c in cols if c in pr]].to_csv(HERE / "pareto_front_pore_resolved.csv", index=False)

    summary = {
        "dataset": str(DATA.relative_to(ROOT)),
        "max_thickness_mm": MAX_THICK,
        "knockdown": "results/pore_investigation/knockdown_fits.json (validated <2.8pp vs direct per-design FEA)",
        "pore_free": {"n": int(len(pf)), "lattice_mix": {k: int(v) for k, v in mix_pf.items()}},
        "pore_resolved": {"n": int(len(pr)), "lattice_mix": {k: int(v) for k, v in mix_pr.items()}},
    }
    json.dump(summary, open(HERE / "front_shift_summary.json", "w"), indent=2)

    print("\n=== FRONT SHIFT (pore-free -> pore-resolved) ===")
    fams = sorted(set(mix_pf) | set(mix_pr))
    print(f"  {'family':10s} {'pore-free':>10s} {'pore-resolved':>14s}")
    for f in fams:
        print(f"  {f:10s} {mix_pf.get(f, 0):>10d} {mix_pr.get(f, 0):>14d}")
    print(f"  {'TOTAL':10s} {len(pf):>10d} {len(pr):>14d}")
    print("\nwrote: pareto_front_porefree_repro.csv, pareto_front_pore_resolved.csv, front_shift_summary.json")


if __name__ == "__main__":
    main()
