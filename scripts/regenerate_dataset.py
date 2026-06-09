"""Rebuilds property_dataset.csv and property_dataset_fcc_face.csv with the
corrected pore impedance (end correction sqrt(2)/16*kd*d/t, not the old 2*eps*Rs).

Writes SC+FCC, FCC face-pore, the L/d-filtered SC+FCC, and SC+FCC+mechanical.
The all-lattices merge is handled by generate_fcc_facepore_dataset.py.
"""

import sys
import os
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTDIR = PROJECT_ROOT / "results" / "property_datasets"
OUTDIR.mkdir(exist_ok=True)

sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
os.chdir(PROJECT_ROOT)

from tmm_script import build_property_dataset, compute_alpha_fcc_face, band_mean
from tmm_script import fcc_facepore_geometric_params

# freq array matches tmm_script.py main()
F_MIN, F_MAX, N_FREQ = 200.0, 5000.0, 1000
f = np.linspace(F_MIN, F_MAX, N_FREQ)

# SC + FCC acoustic
print("=" * 60)
print("Step 1: Regenerating SC + FCC property dataset (18,000 rows)")
print("=" * 60)
build_property_dataset(f, str(OUTDIR))

# FCC face-pore acoustic
print()
print("=" * 60)
print("Step 2: Regenerating FCC face-pore dataset")
print("=" * 60)

import csv as csv_mod

tL_grid = np.linspace(0.03, 0.18, 15)
d_grid  = np.linspace(0.5e-3, 3.0e-3, 15)
L_grid  = np.linspace(10e-3, 40e-3, 10)
N_grid  = [5, 10, 15, 20]

rows_face = []
skipped = 0
total = len(tL_grid) * len(d_grid) * len(L_grid) * len(N_grid)
count = 0
milestone = max(1, total // 10)

for tL in tL_grid:
    for d in d_grid:
        for L in L_grid:
            # face-pore needs L/d >= 12 to keep pores independent
            if (L / d) < 12.0:
                skipped += len(N_grid)
                count   += len(N_grid)
                continue
            for N in N_grid:
                alpha = compute_alpha_fcc_face(f, tL, d, L, N)
                bm    = band_mean(alpha, f)
                idx   = int(np.argmin(np.abs(f - 1000.0)))
                a1k   = float(np.clip(alpha[idx], 0.0, 1.0))
                rows_face.append({
                    "t_over_L":        round(float(tL), 6),
                    "d_mm":            round(float(d * 1e3), 4),
                    "L_mm":            round(float(L * 1e3), 4),
                    "N":               int(N),
                    "lattice_type":    "FCC_face",
                    "band_mean_alpha": round(float(bm), 6),
                    "alpha_1000hz":    round(float(a1k), 6),
                })
                count += 1
                if count % milestone == 0:
                    print(f"  {count:>6}/{total}  ({100*count/total:.0f}%)")

face_csv = OUTDIR / "property_dataset_fcc_face.csv"
fieldnames = ["t_over_L", "d_mm", "L_mm", "N", "lattice_type",
              "band_mean_alpha", "alpha_1000hz"]
with open(face_csv, "w", newline="") as fout:
    writer = csv_mod.DictWriter(fout, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows_face)
print(f"  Saved: {face_csv}  ({len(rows_face)} rows, {skipped} skipped by L/d<12)")

# pore-interaction filter on SC + FCC
print()
print("=" * 60)
print("Step 3: Applying pore interaction filter (L/d >= 6 for SC/FCC)")
print("=" * 60)

import pandas as pd

prop_df = pd.read_csv(OUTDIR / "property_dataset.csv")
n_before = len(prop_df)
mask = (prop_df["L_mm"] / prop_df["d_mm"]) >= 6.0
prop_filtered = prop_df[mask].copy()
n_after = len(prop_filtered)
prop_filtered.to_csv(OUTDIR / "property_dataset_filtered.csv", index=False)
print(f"  SC+FCC: {n_before} rows → {n_after} after filter "
      f"({n_before - n_after} removed)")

# merge acoustic + mechanical predictions
print()
print("=" * 60)
print("Step 4: Merging acoustic dataset with mechanical predictions")
print("=" * 60)

import json, pickle

ga_path  = OUTDIR / "surrogate_gibsonashby.json"
gpr_path = OUTDIR / "surrogate_gpr.pkl"

if not ga_path.exists():
    print("  WARNING: surrogate_gibsonashby.json not found — skipping merge.")
    print("  Run train_mechanical_surrogate.py first.")
else:
    with open(ga_path) as f_ga:
        ga_params = json.load(f_ga)

    with open(gpr_path, "rb") as f_gpr:
        gpr_models = pickle.load(f_gpr)

    # use whatever method surrogate_best_method.json recorded
    best_path = OUTDIR / "surrogate_best_method.json"
    best_method = "Gibson-Ashby"  # default
    if best_path.exists():
        with open(best_path) as f_bm:
            bm_data = json.load(f_bm)
            best_method = bm_data.get("best_method", "Gibson-Ashby")
    print(f"  Using surrogate method: {best_method}")

    # GA json: rho_fit = [slope, intercept]; C_ij = [C1, n] -> C1 * rho^n
    prop_df2 = pd.read_csv(OUTDIR / "property_dataset.csv")

    for lt in ["SC", "FCC"]:
        mask_lt  = prop_df2["lattice_type"] == lt
        tL_vals  = prop_df2.loc[mask_lt, "t_over_L"].values
        ga_lt    = ga_params.get(lt, {})

        # rho from the t/L fit
        rho_fit  = ga_lt.get("rho_fit", [3.0, 0.0])
        rho_vals = np.clip(rho_fit[0] * tL_vals + rho_fit[1], 0.0, 1.0)

        if best_method.lower() == "gpr":
            # GPR features = [rho, is_SC_flag]
            is_sc_flag = 1.0 if lt == "SC" else 0.0
            X_pred_raw = np.column_stack([rho_vals,
                                          np.full_like(rho_vals, is_sc_flag)])
            for comp in ["C11", "C12", "C44"]:
                try:
                    scaler = gpr_models["scalers"][lt][comp]
                    model  = gpr_models["models"][lt][comp]
                    X_sc   = scaler.transform(X_pred_raw)
                    pred   = model.predict(X_sc)
                except (KeyError, Exception) as e:
                    print(f"  GPR predict failed for {lt}/{comp}: {e}; falling back to Gibson-Ashby")
                    c1, n = ga_lt.get(comp, [1.0, 1.0])
                    pred  = c1 * rho_vals**n
                prop_df2.loc[mask_lt, comp] = pred
        else:
            # GA power law: C_ij = C1 * rho^n
            for comp in ["C11", "C12", "C44"]:
                c1, n = ga_lt.get(comp, [1.0, 1.0])
                prop_df2.loc[mask_lt, comp] = c1 * rho_vals**n

    prop_df2["surrogate_method"] = best_method

    # L/d >= 6 filter
    mask_valid = (prop_df2["L_mm"] / prop_df2["d_mm"]) >= 6.0
    prop_df2 = prop_df2[mask_valid].reset_index(drop=True)

    out_mech = OUTDIR / "property_dataset_with_mechanical.csv"
    prop_df2.to_csv(out_mech, index=False)
    print(f"  Saved: {out_mech}  ({len(prop_df2)} rows)")

print()
print("=" * 60)
print("Dataset regeneration complete.")
print("  results/property_datasets/property_dataset.csv              (SC + FCC acoustic)")
print("  results/property_datasets/property_dataset_fcc_face.csv     (FCC face-pore acoustic)")
print("  results/property_datasets/property_dataset_filtered.csv     (SC + FCC filtered)")
if ga_path.exists():
    print("  results/property_datasets/property_dataset_with_mechanical.csv (SC + FCC + mech)")
print()
print("Next steps:")
print("  1. Run generate_fcc_facepore_dataset.py to rebuild all-lattices CSV")
print("  2. Retrain InversionNet: python scripts/train_inversion.py")
print("  3. Re-run optimisation: python scripts/run_optimisation.py")
print("=" * 60)
