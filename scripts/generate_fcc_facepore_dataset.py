"""Builds the FCC face-pore acoustic dataset via the TMM model in tmm_script.py.

Each {111} plate carries 4 pores (one per sub-triangle), 16 active pores total.
Pores sit ~L/4 apart so we keep L/d >= 12 to stop them interacting; that filter
runs before alpha so invalid combos are skipped. Grid matches the SC+FCC dataset.
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

SCRIPTS_DIR  = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS_DIR.parent
sys.path.insert(0, str(SCRIPTS_DIR))

from tmm_script import (
    compute_alpha_fcc_face,
    band_mean,
    alpha_at_frequency,
)

# grid matches the SC + FCC dataset
TL_GRID  = np.linspace(0.03, 0.18, 15)
D_MM_GRID = np.linspace(0.5, 3.0, 15)           # mm
L_MM_GRID = np.linspace(10.0, 40.0, 10)         # mm
N_GRID    = [5, 10, 15, 20]

FREQ      = np.arange(500, 2001, 10)             # 500-2000 Hz
LATTICE   = "FCC_face"

# face-pore needs L/d >= 12 (pores ~L/4 apart)
PORE_CRITERION = 12.0


def generate_dataset(out_path: Path) -> None:
    """sweep the grid, filter interacting pores, write the FCC_face acoustic CSV"""
    total_grid    = len(TL_GRID) * len(D_MM_GRID) * len(L_MM_GRID) * len(N_GRID)
    filtered_out  = 0
    rows_written  = 0

    fieldnames = [
        "t_over_L", "d_mm", "L_mm", "N",
        "lattice_type", "band_mean_alpha", "alpha_1000hz",
    ]

    print("=" * 60)
    print("FCC Face-Pore Acoustic Dataset Generator")
    print("Reference: Liu et al. 2022, Mat. Design 223, 111122")
    print("=" * 60)
    print(f"Parameter grid: {len(TL_GRID)} t/L × {len(D_MM_GRID)} d × "
          f"{len(L_MM_GRID)} L × {len(N_GRID)} N = {total_grid:,} combinations")
    print(f"Pore interaction filter: L/d ≥ {PORE_CRITERION:.0f}  "
          f"(spacing ≈ L/4, criterion (L/4)/d ≥ 3)")
    print(f"Output: {out_path}\n")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    milestone = max(1, total_grid // 10)
    count = 0

    with open(out_path, "w", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()

        for N in N_GRID:
            for L_mm in L_MM_GRID:
                L = L_mm / 1000.0          # metres
                for d_mm in D_MM_GRID:
                    # drop interacting pores before touching alpha
                    if L_mm / d_mm < PORE_CRITERION:
                        filtered_out += len(TL_GRID)
                        count += len(TL_GRID)
                        continue

                    d = d_mm / 1000.0      # metres

                    for tL in TL_GRID:
                        alpha  = compute_alpha_fcc_face(FREQ, tL, d, L, N)
                        bm     = band_mean(alpha, FREQ)
                        a1k    = alpha_at_frequency(alpha, FREQ, 1000.0)

                        writer.writerow({
                            "t_over_L":        round(float(tL), 6),
                            "d_mm":            round(float(d_mm), 4),
                            "L_mm":            round(float(L_mm), 4),
                            "N":               int(N),
                            "lattice_type":    LATTICE,
                            "band_mean_alpha": round(float(bm), 6),
                            "alpha_1000hz":    round(float(a1k), 6),
                        })
                        rows_written += 1
                        count += 1

                    if count % milestone < len(TL_GRID):
                        pct = 100 * count / total_grid
                        print(f"  {count:>6}/{total_grid}  ({pct:.0f}%)  "
                              f"written={rows_written}  filtered={filtered_out}")

    print(f"\n{'─'*60}")
    print(f"Total grid combinations : {total_grid:,}")
    print(f"Filtered (L/d < {PORE_CRITERION:.0f})    : {filtered_out:,}  "
          f"({100*filtered_out/total_grid:.1f}%)")
    print(f"Rows written            : {rows_written:,}  "
          f"({100*rows_written/total_grid:.1f}%)")

    # read back for the summary
    import pandas as pd
    df = pd.read_csv(out_path)
    print(f"\nband_mean_alpha: {df['band_mean_alpha'].min():.4f} "
          f"— {df['band_mean_alpha'].max():.4f}  "
          f"(mean={df['band_mean_alpha'].mean():.4f})")
    print(f"alpha_1000hz  : {df['alpha_1000hz'].min():.4f} "
          f"— {df['alpha_1000hz'].max():.4f}  "
          f"(mean={df['alpha_1000hz'].mean():.4f})")
    print(f"\nRow distribution by N:")
    print(df['N'].value_counts().sort_index().to_string())
    print(f"\nSaved: {out_path}")

    return df


def merge_into_all_lattices(fcc_face_path: Path) -> None:
    """Concat FCC_face onto the SC+FCC mechanical dataset -> all_lattices.csv.

    FCC_face C11/C12/C44 start NaN; populate_fcc_facepore_mechanical.py fills them.
    """
    import pandas as pd

    mech_path = PROJECT_ROOT / "results" / "property_datasets" / "property_dataset_with_mechanical.csv"
    out_path  = PROJECT_ROOT / "results" / "property_datasets" / "property_dataset_all_lattices.csv"

    if not mech_path.exists():
        print(f"\n[SKIP merge] {mech_path} not found — run train_mechanical_surrogate.py first.")
        return

    print(f"\n{'='*60}")
    print("Merging into property_dataset_all_lattices.csv")
    print(f"{'='*60}")

    df_mech = pd.read_csv(mech_path)
    df_face = pd.read_csv(fcc_face_path)

    print(f"  Existing (SC + FCC): {len(df_mech):,} rows")
    print(f"  FCC_face acoustic  : {len(df_face):,} rows")

    # no mech data for FCC_face yet
    for col in ["C11", "C12", "C44"]:
        if col not in df_face.columns:
            df_face[col] = float("nan")

    if "surrogate_method" in df_mech.columns and "surrogate_method" not in df_face.columns:
        df_face["surrogate_method"] = float("nan")

    # union the columns, NaN-fill the gaps
    all_cols = list(df_mech.columns)
    for col in df_face.columns:
        if col not in all_cols:
            all_cols.append(col)

    df_mech = df_mech.reindex(columns=all_cols)
    df_face = df_face.reindex(columns=all_cols)

    df_all = pd.concat([df_mech, df_face], ignore_index=True)
    df_all.to_csv(out_path, index=False)

    print(f"\n  Row counts by lattice_type:")
    print(df_all["lattice_type"].value_counts().to_string())
    nan_counts = df_all[["C11", "C12", "C44"]].isna().sum()
    print(f"\n  NaN counts in mechanical columns:")
    print(f"    C11: {nan_counts['C11']:,}  C12: {nan_counts['C12']:,}  C44: {nan_counts['C44']:,}")
    print(f"\n  Saved: {out_path}  ({len(df_all):,} rows total)")


def main():
    """generate the FCC_face acoustic dataset and optionally merge it in"""
    parser = argparse.ArgumentParser(
        description="Generate FCC face-pore acoustic TMM dataset."
    )
    parser.add_argument(
        "--out", type=str,
        default=str(PROJECT_ROOT / "results" / "property_datasets" / "property_dataset_fcc_face.csv"),
        help="Output CSV path",
    )
    parser.add_argument(
        "--no_merge", action="store_true",
        help="Skip merging into property_dataset_all_lattices.csv",
    )
    args = parser.parse_args()

    out_path = Path(args.out)
    generate_dataset(out_path)

    if not args.no_merge:
        merge_into_all_lattices(out_path)


if __name__ == "__main__":
    main()
