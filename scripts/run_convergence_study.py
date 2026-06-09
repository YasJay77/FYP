"""HomPy mesh-convergence check for SC and FCC at a fixed pore-free t/L = 0.10.

Re-homogenises across voxel resolutions and records C11/C12/C44 plus the per-step
relative change, to show the tensor has converged. SC is voxelised at nominal
t/L = 0.09 (the dilation generator lands on effective 0.10); FCC takes 0.10 directly.
Runs in-process. Writes results/data/hompy_convergence.csv.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from lattice.sc_plate_voxels import gen_sc
from lattice.fcc_plate_voxels import gen_fcc
from hompy import homogenize

OUT = ROOT / "results" / "data" / "hompy_convergence.csv"

RESOLUTIONS = [30, 50, 70, 90, 110]
SC_NOMINAL_TL = 0.09   # dilation lands on effective t/L = 0.10 at these resolutions
FCC_TL = 0.10          # exact rasteriser, passed through directly

# match the dataset generators
E, NU = 1.0, 0.30
LAM = E * NU / ((1 + NU) * (1 - 2 * NU))
MU = E / (2 * (1 + NU))


def homogenise_one(lattice, res):
    """voxelise + homogenise one lattice at a given resolution, return the C_ij row"""
    if lattice == "SC":
        n_iter = max(1, round(SC_NOMINAL_TL * res / 2))
        eff_tL = (2 * n_iter + 1) / res
        vox = gen_sc(resolution=res, t_over_L=SC_NOMINAL_TL)
    else:
        n_iter = np.nan
        eff_tL = FCC_TL
        vox = gen_fcc(resolution=res, t_over_L=FCC_TL)
    rho = float(vox.mean())
    t0 = time.time()
    C = homogenize(1.0, 1.0, 1.0, [LAM], [MU], vox, False)
    return {
        "lattice": lattice, "res": res, "n_iter": n_iter, "eff_tL": eff_tL,
        "rho": rho, "C11": float(C[0, 0]), "C12": float(C[0, 1]),
        "C44": float(C[3, 3]), "time_s": time.time() - t0,
    }


def main():
    """sweep resolutions for SC+FCC, compute per-step change, write the convergence csv"""
    rows = []
    for lattice in ("SC", "FCC"):
        for res in RESOLUTIONS:
            print(f"  {lattice}  res={res} ...", flush=True)
            rows.append(homogenise_one(lattice, res))
    df = pd.DataFrame(rows).sort_values(["lattice", "res"]).reset_index(drop=True)

    # per-step relative change (%) within each lattice
    for comp in ("C11", "C12", "C44"):
        df[f"d{comp}"] = np.nan
    for lat in df["lattice"].unique():
        m = df["lattice"] == lat
        for comp in ("C11", "C12", "C44"):
            v = df.loc[m, comp].to_numpy()
            df.loc[m, f"d{comp}"] = [np.nan] + [
                100 * abs(v[i] - v[i - 1]) / v[i] for i in range(1, len(v))
            ]
    df["max_d"] = df[["dC11", "dC12", "dC44"]].max(axis=1)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"\nWrote {OUT}  ({len(df)} rows)")


if __name__ == "__main__":
    main()
