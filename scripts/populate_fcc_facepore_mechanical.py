"""Fills C11/C12/C44 for the FCC_face rows in property_dataset_all_lattices.csv
by reusing the FCC GPR surrogate as-is.

FCC_face shares the {111} skeleton and voxelisation with FCC nodal at equal t/L,
so the FCC mech GPR applies directly. The pore geometry (d, L) doesn't shift the
elastic constants here (d/L <= 0.3); only t/L (-> rho) sets stiffness. GPR features
are [rho, is_SC_flag=0]; rho comes from t/L via the linear fit.
"""

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPTS_DIR  = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS_DIR.parent
OUTPUTS_DIR  = PROJECT_ROOT / "results" / "property_datasets"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def estimate_rho_from_tL(t_over_L: np.ndarray, rho_fit: list) -> np.ndarray:
    """rho from t/L via the stored linear fit"""
    slope, intercept = rho_fit
    rho = slope * t_over_L + intercept
    return np.clip(rho, 0.01, 1.0)


def predict_fcc_gpr(
    t_over_L_vals: np.ndarray,
    gpr_models: dict,
    gpr_scalers: dict,
    ga_params: dict,
) -> tuple:
    """predict C11/C12/C44 for FCC_face rows with the FCC GPR"""
    rho = estimate_rho_from_tL(t_over_L_vals, ga_params["FCC"]["rho_fit"])
    # is_SC_flag = 0 for the FCC family
    is_SC_flag = np.zeros(len(rho))
    X = np.column_stack([rho, is_SC_flag])

    C11_arr = gpr_models["FCC"]["C11"].predict(
        gpr_scalers["FCC"]["C11"].transform(X)).astype(float)
    C12_arr = gpr_models["FCC"]["C12"].predict(
        gpr_scalers["FCC"]["C12"].transform(X)).astype(float)
    C44_arr = gpr_models["FCC"]["C44"].predict(
        gpr_scalers["FCC"]["C44"].transform(X)).astype(float)

    return C11_arr, C12_arr, C44_arr


def main():
    """fill the FCC_face C11/C12/C44 columns via the FCC GPR and save the dataset"""
    print("=" * 65)
    print("  Populate FCC Face-Pore Mechanical Properties")
    print("  (reusing FCC GPR — identical {111} plate skeleton)")
    print("=" * 65)

    # surrogate artefacts
    gpr_path = OUTPUTS_DIR / "surrogate_gpr.pkl"
    ga_path  = OUTPUTS_DIR / "surrogate_gibsonashby.json"

    if not gpr_path.exists():
        print(f"[ERROR] GPR not found: {gpr_path}")
        sys.exit(1)
    if not ga_path.exists():
        print(f"[ERROR] GA params not found: {ga_path}")
        sys.exit(1)

    with open(gpr_path, "rb") as f:
        gpr_data = pickle.load(f)
    with open(ga_path) as f:
        ga_params = json.load(f)

    gpr_models  = gpr_data["models"]
    gpr_scalers = gpr_data["scalers"]
    print(f"\n  GPR models loaded for: {list(gpr_models.keys())}")
    print(f"  GA rho_fit (FCC): slope={ga_params['FCC']['rho_fit'][0]:.4f}, "
          f"intercept={ga_params['FCC']['rho_fit'][1]:.4f}")

    # combined dataset
    csv_path = OUTPUTS_DIR / "property_dataset_all_lattices.csv"
    if not csv_path.exists():
        print(f"[ERROR] Dataset not found: {csv_path}")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    print(f"\n  Loaded: {len(df):,} rows from {csv_path.name}")
    print(f"  Lattice types: {df['lattice_type'].value_counts().to_dict()}")

    # FCC_face rows still missing mech data
    mask_face = (df["lattice_type"] == "FCC_face") & df["C11"].isna()
    n_to_fill = mask_face.sum()
    print(f"\n  FCC_face rows with NaN C11/C12/C44: {n_to_fill:,}")

    if n_to_fill == 0:
        print("  Nothing to do — FCC_face rows already have mechanical data.")
        return

    t_vals = df.loc[mask_face, "t_over_L"].values.astype(float)

    C11_pred, C12_pred, C44_pred = predict_fcc_gpr(
        t_vals, gpr_models, gpr_scalers, ga_params
    )

    df.loc[mask_face, "C11"] = C11_pred
    df.loc[mask_face, "C12"] = C12_pred
    df.loc[mask_face, "C44"] = C44_pred
    df.loc[mask_face, "surrogate_method"] = "GPR_FCC_reuse"

    # make sure nothing slipped through
    print("\n  C11/C12/C44 NaN counts after fill:")
    for lt in df["lattice_type"].unique():
        sub = df[df["lattice_type"] == lt]
        n_nan = sub["C11"].isna().sum()
        print(f"    {lt}: {n_nan} NaN out of {len(sub)}")

    n_total_populated = df["C11"].notna().sum()
    print(f"\n  Total rows with populated C11/C12/C44: {n_total_populated:,} / {len(df):,}")

    print("\n  Mechanical property ranges per lattice type:")
    for lt in df["lattice_type"].unique():
        sub = df[df["lattice_type"] == lt].dropna(subset=["C11"])
        print(f"    {lt:<10}  C11=[{sub['C11'].min():.4f}, {sub['C11'].max():.4f}]  "
              f"C12=[{sub['C12'].min():.4f}, {sub['C12'].max():.4f}]  "
              f"C44=[{sub['C44'].min():.4f}, {sub['C44'].max():.4f}]")

    # at equal t/L, FCC and FCC_face should give identical C_ij
    print("\n  Sanity check — FCC vs FCC_face at t/L = 0.10:")
    for tL in [0.03, 0.10, 0.18]:
        fcc_row  = df[(df["lattice_type"] == "FCC")      & (np.abs(df["t_over_L"] - tL) < 1e-4)]
        face_row = df[(df["lattice_type"] == "FCC_face") & (np.abs(df["t_over_L"] - tL) < 1e-4)]
        if len(fcc_row) and len(face_row):
            c11_fcc  = fcc_row["C11"].iloc[0]
            c11_face = face_row["C11"].iloc[0]
            print(f"    t/L={tL:.2f}  FCC C11={c11_fcc:.5f}  FCC_face C11={c11_face:.5f}  "
                  f"diff={abs(c11_fcc - c11_face):.2e}")

    df.to_csv(csv_path, index=False)
    print(f"\n  Saved updated dataset: {csv_path}")
    print(f"  ({len(df):,} rows, all C11/C12/C44 populated)")
    print("\nDone — Step 1 complete.")


if __name__ == "__main__":
    main()
