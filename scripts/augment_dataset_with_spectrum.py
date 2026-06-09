"""Tack the per-frequency absorption spectrum onto the property dataset.

Runs the TMM at 16 freqs (500-2000 Hz) for every design and adds them as
alpha_500hz...alpha_2000hz. This is what the inversion net trains on.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from tmm_script import compute_alpha, compute_alpha_fcc, compute_alpha_fcc_face

SRC = ROOT / "results" / "property_datasets" / "property_dataset_all_lattices.csv"
DST = ROOT / "results" / "property_datasets" / "property_dataset_spectral.csv"

# 16 freqs, 500 to 2000 Hz in 100 Hz steps
FREQS_HZ = np.arange(500, 2001, 100, dtype=float)
FREQ_COLS = [f"alpha_{int(f)}hz" for f in FREQS_HZ]


def _alpha_at(lt, t_over_L, d_mm, L_mm, N, freqs):
    """TMM absorption for one lattice/geometry at given freqs"""
    d_m = d_mm / 1000.0
    L_m = L_mm / 1000.0
    if lt == "SC":
        return compute_alpha(freqs, t_over_L, d_m, L_m, int(N))
    if lt == "FCC":
        return compute_alpha_fcc(freqs, t_over_L, d_m, L_m, int(N))
    if lt == "FCC_face":
        return compute_alpha_fcc_face(freqs, t_over_L, d_m, L_m, int(N))
    raise ValueError(f"Unknown lattice_type: {lt!r}")


def main():
    """run the TMM over every design and write the spectral dataset"""
    print(f"[READ ] {SRC}")
    df = pd.read_csv(SRC)
    n = len(df)
    print(f"[INFO ] {n} rows | evaluating {len(FREQS_HZ)} frequencies per design")

    spectra = np.empty((n, len(FREQS_HZ)), dtype=np.float32)
    report_every = max(1, n // 20)
    for i, row in enumerate(df.itertuples(index=False)):
        spectra[i] = _alpha_at(row.lattice_type, float(row.t_over_L),
                               float(row.d_mm), float(row.L_mm), int(row.N),
                               FREQS_HZ)
        if (i + 1) % report_every == 0 or i == n - 1:
            print(f"  {i+1:>6d}/{n}  ({(i+1)/n*100:5.1f}%)")

    for j, col in enumerate(FREQ_COLS):
        df[col] = spectra[:, j]

    # core cols first, then the spectral ones in freq order
    core_cols = [c for c in df.columns if c not in FREQ_COLS]
    df = df[core_cols + FREQ_COLS]

    print(f"[WRITE] {DST}")
    df.to_csv(DST, index=False)
    print(f"[DONE ] {len(df.columns)} columns")


if __name__ == "__main__":
    main()
