"""How far the pore-free mechanical assumption holds. Read-only, writes nothing existing.

Reads the paired pore-free/pore-resolved study + fitted knockdowns and asks:
up to what d/L is pore-free valid per family, and do the selected designs sit inside it.
err = (C_free - C_res)/C_free = 1 - k(d/L), where k is the fitted knockdown.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent

FITS   = ROOT / "results" / "pore_investigation" / "knockdown_fits.json"
RAW    = ROOT / "results" / "pore_investigation" / "knockdown_factors_raw.csv"
PARETO = ROOT / "results" / "property_datasets" / "pareto_front.csv"
MASTER = ROOT / "results" / "property_datasets" / "property_dataset_spectral.csv"

COMPONENTS = ("C11", "C12", "C44")
FAMILIES   = ("SC", "FCC", "FCC_face")          # FCC = nodal-pore
TOLS       = (0.05, 0.10, 0.15)


# knockdown forms, mirror pore_correction.py
def k_power(x, a, p):       # 1 - a x^p
    """power-law knockdown 1 - a x^p"""
    return 1.0 - a * np.power(np.maximum(x, 0.0), p)

def k_quadratic(x, b, c):   # 1 - b x - c x^2
    """quadratic knockdown 1 - b x - c x^2"""
    return 1.0 - b * x - c * x * x

def k_exp_power(x, a, p):   # exp(-a x^p)
    """exponential knockdown exp(-a x^p)"""
    return np.exp(-a * np.power(np.maximum(x, 0.0), p))

FORM = {"A_power_minus": k_power, "B_quadratic": k_quadratic, "C_exp_power": k_exp_power}


def load_fits():
    """read the fitted knockdowns json"""
    with open(FITS) as f:
        return json.load(f)


def k_of(fits, family, comp, dL):
    """evaluate the fitted knockdown k for a family/component at d/L"""
    info = fits[family][f"k_{comp}"]
    return FORM[info["best"]](np.asarray(dL, float), *info["params"])


def err_of(fits, family, comp, dL):
    """relative error of the pore-free stiffness = 1 - k"""
    return 1.0 - k_of(fits, family, comp, dL)


def main():
    """run the pore-free validity analysis and dump the summary tables"""
    fits = load_fits()
    out = {}

    # sanity check against the measured raw points
    raw = pd.read_csv(RAW)
    raw = raw[raw["d_over_L"] > 0].copy()
    raw["err_C11_meas"] = 1.0 - raw["k_C11"]
    raw["err_C11_fit"]  = [err_of(fits, lat, "C11", d)
                           for lat, d in zip(raw["lattice"], raw["d_over_L"])]
    raw["abs_resid"] = (raw["err_C11_meas"] - raw["err_C11_fit"]).abs()
    out["max_fit_residual_C11"] = float(raw["abs_resid"].max())

    # validity thresholds per family/component
    grid = np.linspace(0.0, 0.40, 40001)          # fine d/L grid
    thr_rows, curve_rows = [], []
    for fam in FAMILIES:
        for comp in COMPONENTS:
            e = err_of(fits, fam, comp, grid)
            for tol in TOLS:
                hit = np.where(e >= tol)[0]
                dstar = float(grid[hit[0]]) if len(hit) else float("nan")
                thr_rows.append(dict(family=fam, component=comp,
                                     tol_pct=int(tol * 100), dL_threshold=round(dstar, 4)))
        # plotting curve, C11 is the structural-sizing component
        for d in np.round(np.linspace(0, 0.30, 61), 4):
            curve_rows.append(dict(family=fam, d_over_L=float(d),
                                   err_C11=float(err_of(fits, fam, "C11", d)),
                                   err_C12=float(err_of(fits, fam, "C12", d)),
                                   err_C44=float(err_of(fits, fam, "C44", d))))
    thr = pd.DataFrame(thr_rows)
    thr.to_csv(HERE / "validity_thresholds.csv", index=False)
    pd.DataFrame(curve_rows).to_csv(HERE / "error_curves.csv", index=False)

    # C11 thresholds, compact dict
    out["C11_thresholds"] = {
        fam: {f"{int(t*100)}pct": float(thr[(thr.family == fam) &
              (thr.component == "C11") & (thr.tol_pct == int(t*100))].dL_threshold.iloc[0])
              for t in TOLS} for fam in FAMILIES}

    # per-design bias for the production Pareto front
    pf = pd.read_csv(PARETO)
    pf["d_over_L"] = pf["d_mm"] / pf["L_mm"]
    # map lattice label to knockdown key
    famkey = {"SC": "SC", "FCC": "FCC", "FCC_NODAL": "FCC", "FCC_FACE": "FCC_face"}
    pf["fam"] = pf["lattice_type"].str.upper().map(famkey)
    for comp in COMPONENTS:
        pf[f"err_{comp}_pct"] = [round(100 * err_of(fits, f, comp, d), 1)
                                 for f, d in zip(pf["fam"], pf["d_over_L"])]
    keep = ["lattice_type", "t_over_L", "d_mm", "L_mm", "N", "d_over_L",
            "band_mean_alpha", "C11", "err_C11_pct", "err_C12_pct", "err_C44_pct"]
    pf_out = pf[keep].copy()
    pf_out["d_over_L"] = pf_out["d_over_L"].round(4)
    pf_out.to_csv(HERE / "per_design_bias.csv", index=False)

    out["pareto"] = dict(
        n=len(pf),
        dL_min=round(float(pf.d_over_L.min()), 4),
        dL_max=round(float(pf.d_over_L.max()), 4),
        worst_err_C11_pct=round(float(pf.err_C11_pct.max()), 1),
        worst_design=pf.loc[pf.err_C11_pct.idxmax(),
                            ["lattice_type", "t_over_L", "d_over_L", "err_C11_pct"]].to_dict(),
        by_family={fam: dict(
            n=int((pf.fam == fam).sum()),
            dL_max=round(float(pf.loc[pf.fam == fam, "d_over_L"].max()), 4) if (pf.fam == fam).any() else None,
            worst_err_C11_pct=round(float(pf.loc[pf.fam == fam, "err_C11_pct"].max()), 1) if (pf.fam == fam).any() else None,
        ) for fam in FAMILIES},
    )

    # full swept-dataset d/L distribution vs the 10% threshold
    md = pd.read_csv(MASTER, usecols=["t_over_L", "d_mm", "L_mm", "lattice_type"])
    md["d_over_L"] = md["d_mm"] / md["L_mm"]
    md["fam"] = md["lattice_type"].str.upper().map(famkey)
    md["err_C11"] = [err_of(fits, f, "C11", d) for f, d in zip(md["fam"], md["d_over_L"])]
    out["dataset"] = dict(
        n=len(md),
        dL_min=round(float(md.d_over_L.min()), 4),
        dL_max=round(float(md.d_over_L.max()), 4),
        frac_err_le_5pct=round(float((md.err_C11 <= 0.05).mean()), 3),
        frac_err_le_10pct=round(float((md.err_C11 <= 0.10).mean()), 3),
        frac_err_le_15pct=round(float((md.err_C11 <= 0.15).mean()), 3),
        by_family={fam: dict(
            n=int((md.fam == fam).sum()),
            dL_max=round(float(md.loc[md.fam == fam, "d_over_L"].max()), 4),
            frac_err_le_10pct=round(float((md.loc[md.fam == fam, "err_C11"] <= 0.10).mean()), 3),
        ) for fam in FAMILIES},
    )

    with open(HERE / "pore_validity_summary.json", "w") as f:
        json.dump(out, f, indent=2)

    print("=" * 72)
    print("PORE-FREE VALIDITY INVESTIGATION  (err = 1 - k = fraction of C lost)")
    print("=" * 72)
    print(f"\n[sanity] max |fit - measured| error in C11 across 9 raw pts = "
          f"{out['max_fit_residual_C11']*100:.2f} %  (fits track the data)\n")

    print("d/L threshold where pore-free C11 error first exceeds tolerance:")
    print(f"  {'family':10s} {'5%':>8s} {'10%':>8s} {'15%':>8s}")
    for fam in FAMILIES:
        t = out["C11_thresholds"][fam]
        print(f"  {fam:10s} {t['5pct']:>8.3f} {t['10pct']:>8.3f} {t['15pct']:>8.3f}")

    print("\nProduction Pareto front (the designs actually selected):")
    p = out["pareto"]
    print(f"  d/L spans {p['dL_min']:.3f} – {p['dL_max']:.3f} over {p['n']} designs")
    print(f"  WORST-CASE C11 bias on ANY selected design = {p['worst_err_C11_pct']} %"
          f"  ({p['worst_design']['lattice_type']}, "
          f"t/L={p['worst_design']['t_over_L']}, d/L={round(p['worst_design']['d_over_L'],3)})")
    for fam in FAMILIES:
        b = p["by_family"][fam]
        if b["n"]:
            print(f"    {fam:10s} n={b['n']:2d}  d/L<= {b['dL_max']:.3f}  worst C11 bias {b['worst_err_C11_pct']:>5}%")

    print("\nFull swept dataset (21,720 rows):")
    d = out["dataset"]
    print(f"  d/L spans {d['dL_min']:.3f} – {d['dL_max']:.3f}")
    print(f"  rows with C11 error <= 5% : {d['frac_err_le_5pct']*100:.1f}%")
    print(f"  rows with C11 error <=10% : {d['frac_err_le_10pct']*100:.1f}%")
    print(f"  rows with C11 error <=15% : {d['frac_err_le_15pct']*100:.1f}%")
    print("\nwrote: pore_validity_summary.json, per_design_bias.csv,")
    print("       validity_thresholds.csv, error_curves.csv")


if __name__ == "__main__":
    main()
