"""FCC mechanical FEA on the stellated-octahedron skeleton.

HomPy homogenisation at 20 t/L values in [0.03, 0.18] (the acoustic range — the
surrogate never queries outside it). Each run is appended to the CSV as it
finishes; on relaunch any t/L already present (matched by voxel_hash) is skipped,
so it's safe to kill and restart. Built-in per-row density sanity check.
Writes results/data/fcc_mech_res120_dilation.csv.
"""

import argparse
import csv
import gc
import hashlib
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from hompy import homogenize
from lattice.fcc_plate_voxels import gen_fcc


# 20 t/L values over the acoustic range [0.03, 0.18]
TL_GRID: np.ndarray = np.linspace(0.03, 0.18, 20)

RESOLUTION = 120    # production FCC resolution
E_BULK     = 1.0
NU         = 0.30
LX = LY = LZ = 1.0
DIRECT     = False   # iterative solver

# density-check thresholds, loosened for resolution noise
LIU_STOP_PCT    = 30.0   # STOP only on real geometry failure
LIU_NOTE_PCT    = 15.0
LIU_FIRST_WARN  = 35.0   # relaxed for the 1-layer t/L=0.03 plate

DISC_WARN_TL_MAX = 0.031

# new columns appended last
CSV_HEADER = [
    "t_over_L", "rho", "voxel_hash",
    "C11", "C12", "C44",
    "time_s", "direct_solution", "resolution",
    "E", "nu", "lx", "ly", "lz",
    "rho_liu",                 # pore-free density prediction (rR=0)
    "discretization_warning",  # True only at t/L=0.03
]


def lame_from_E_nu(E: float, nu: float) -> Tuple[float, float]:
    """lame parameters from E and nu"""
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu  = E / (2 * (1 + nu))
    return lam, mu


def voxel_hash(vox: np.ndarray) -> str:
    """md5 of the voxel array to spot duplicate geometries"""
    return hashlib.md5(vox.tobytes()).hexdigest()


def liu_rho(tR: float) -> float:
    """pore-free relative density (rR = 0)"""
    return 6.867 * tR - 15.79 * tR ** 2


def load_completed_hashes(csv_path: Path) -> Set[str]:
    """voxel hashes already in the CSV (old-schema files get moved aside)"""
    if not csv_path.exists():
        return set()

    # peek at the header
    with csv_path.open("r", newline="") as f:
        header_line = f.readline().strip()

    if "discretization_warning" not in header_line:
        stale = csv_path.with_suffix(".csv.stale_schema_bak")
        csv_path.rename(stale)
        print(f"  [INFO] Old-schema CSV detected — moved to {stale.name}.")
        print(f"         Starting fresh with new schema.\n")
        return set()

    hashes: Set[str] = set()
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("voxel_hash"):
                hashes.add(row["voxel_hash"])
    return hashes


def append_row(csv_path: Path, row: Dict, header: List[str]) -> None:
    """append one row; write the header if the file is new"""
    is_new = not csv_path.exists()
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if is_new:
            writer.writeheader()
        writer.writerow(row)
        f.flush()


def liu_check(tL: float, rho_vox: float, is_first: bool) -> str:
    """check predicted vs voxelised rho, raises SystemExit on a mid-range STOP"""
    rho_l = liu_rho(tL)
    err   = 100.0 * (rho_vox - rho_l) / max(rho_l, 1e-9)
    warn_limit = LIU_FIRST_WARN if is_first else LIU_NOTE_PCT
    stop_limit = None           if is_first else LIU_STOP_PCT

    if stop_limit is not None and abs(err) > stop_limit:
        msg = (
            f"\n{'='*68}\n"
            f"  STOP — Liu sanity check failed at t/L={tL:.4f}\n"
            f"  rho_vox={rho_vox:.4f}  Liu={rho_l:.4f}  error={err:+.2f}%\n"
            f"  Threshold: |error| > {stop_limit:.0f}% for mid-range points\n"
            f"  This indicates a geometry problem in the corrected skeleton.\n"
            f"  FEA results written so far are preserved in the CSV.\n"
            f"{'='*68}"
        )
        print(msg)
        sys.exit(1)

    if abs(err) > warn_limit:
        tag = "  <-- STOP" if (stop_limit and abs(err) > stop_limit) else "  <-- warn"
    elif abs(err) > LIU_NOTE_PCT and not is_first:
        tag = "  <-- note"
    else:
        tag = ""

    return f"rho_Liu={rho_l:.4f}  error={err:+.2f}%{tag}"


def main() -> None:
    """run the FCC exact-t/L FEA sweep, resuming from the CSV"""
    parser = argparse.ArgumentParser(
        description="Phase 4 FCC FEA on corrected stellated-octahedron skeleton."
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Preview t/L values and expected densities; do not run FEA."
    )
    parser.add_argument(
        "--out", type=str,
        default=str(PROJECT_ROOT / "results" / "data" / "fcc_mech_res120_dilation.csv"),
        help="Output CSV path  (default: results/data/fcc_mech_res120_dilation.csv)."
    )
    args = parser.parse_args()

    out_path = Path(args.out)

    print("=" * 68)
    print("  Phase 4 — FCC FEA: corrected stellated-octahedron skeleton")
    print("=" * 68)
    print(f"\n  Output:      {out_path}")
    print(f"  Resolution:  {RESOLUTION}")
    print(f"  Material:    E={E_BULK}, nu={NU},  solver={'direct' if DIRECT else 'iterative'}")
    print(f"  t/L grid:    linspace({TL_GRID[0]:.3f}, {TL_GRID[-1]:.3f}, {len(TL_GRID)})"
          f"  step={TL_GRID[1]-TL_GRID[0]:.4f}")
    print()

    # pre-flight density sweep
    print(f"  {'#':>3}  {'t/L':>8}  {'rho_vox':>9}  {'rho_Liu':>9}  "
          f"{'error_%':>9}  disc_warn")
    print("  " + "-" * 60)

    vox_cache: List[Tuple[float, np.ndarray, float]] = []
    for i, tL in enumerate(TL_GRID):
        vox   = gen_fcc(RESOLUTION, tL, 0.0)
        rho_v = float(vox.mean())
        rho_l = liu_rho(tL)
        err   = 100.0 * (rho_v - rho_l) / max(rho_l, 1e-9)
        is_first = (tL <= DISC_WARN_TL_MAX)

        warn_thr = LIU_FIRST_WARN if is_first else LIU_STOP_PCT
        flag = ("  <-- STOP" if (not is_first and abs(err) > LIU_STOP_PCT) else
                "  <-- warn" if abs(err) > warn_thr else
                "  <-- note" if abs(err) > LIU_NOTE_PCT else "")
        disc_str = "True " if is_first else "False"
        print(f"  {i+1:>3}  {tL:>8.4f}  {rho_v:>9.4f}  {rho_l:>9.4f}  "
              f"{err:>+9.2f}%{flag:<12}  {disc_str}")
        vox_cache.append((tL, vox, rho_v))

    if args.dry_run:
        print("\n  [DRY RUN] Exiting without FEA.")
        return

    # skip what's already done
    completed = load_completed_hashes(out_path)
    n_done    = len(completed)
    if n_done > 0:
        print(f"\n  Resuming: {n_done} voxel hash(es) already in {out_path.name}.")
    else:
        print(f"\n  Starting fresh.")

    remaining = [(tL, vox, rho_v)
                 for (tL, vox, rho_v) in vox_cache
                 if voxel_hash(vox) not in completed]

    if not remaining:
        print("  All 20 runs already complete — nothing to do.")
        return

    print(f"  Runs to execute: {len(remaining)} / {len(TL_GRID)}")
    print(f"  Estimated time:  ~{len(remaining) * 13:.0f} min "
          f"({len(remaining) * 13 / 60:.1f} h)\n")

    lam, mu = lame_from_E_nu(E_BULK, NU)
    t_total_start = time.time()
    done_this_session = 0

    for run_idx, (tL, vox, rho_v) in enumerate(remaining, start=1):
        h        = voxel_hash(vox)
        is_first = (tL <= DISC_WARN_TL_MAX)

        print(f"  [{run_idx:02d}/{len(remaining)}] t/L={tL:.6f}  "
              f"rho={rho_v:.4f}  hash={h[:10]}...")

        # density check (raises SystemExit on STOP)
        check_str = liu_check(tL, rho_v, is_first)
        print(f"          Liu check: {check_str}")

        # wrap the solve so a crash still keeps earlier rows
        try:
            print(f"          Running homogenize()...", end="", flush=True)
            t0 = time.time()
            C  = homogenize(LX, LY, LZ, [lam], [mu], vox, DIRECT)
            dt = time.time() - t0
            print(f" {dt:.1f} s")
            print(f"          C11={C[0,0]:.5g}  C12={C[0,1]:.5g}  "
                  f"C44={C[3,3]:.5g}")
        except Exception:
            print()
            print("  ERROR during homogenize() — skipping this point.")
            traceback.print_exc()
            print(f"  Completed runs are safe in {out_path.name}.")
            continue

        # one row per completed run
        row = {
            "t_over_L":             round(float(tL), 10),
            "rho":                  round(float(rho_v), 6),
            "voxel_hash":           h,
            "C11":                  float(C[0, 0]),
            "C12":                  float(C[0, 1]),
            "C44":                  float(C[3, 3]),
            "time_s":               round(dt, 4),
            "direct_solution":      int(DIRECT),
            "resolution":           RESOLUTION,
            "E":                    E_BULK,
            "nu":                   NU,
            "lx":                   LX,
            "ly":                   LY,
            "lz":                   LZ,
            "rho_liu":              round(liu_rho(tL), 6),
            "discretization_warning": is_first,
        }
        append_row(out_path, row, CSV_HEADER)
        done_this_session += 1
        print(f"          Saved to {out_path.name}  "
              f"({done_this_session} this session)")

        # free HomPy's big intermediate arrays before the next iteration
        gc.collect()
        print()

    t_total = time.time() - t_total_start
    total_in_file = len(load_completed_hashes(out_path))

    print("=" * 68)
    print(f"  Phase 4 FEA complete.")
    print(f"  Session: {done_this_session} runs in {t_total/60:.1f} min")
    print(f"  Total rows in {out_path.name}: {total_in_file} / {len(TL_GRID)}")
    print()

    if total_in_file == len(TL_GRID):
        # full density-check table once everything's in
        print("  Final Liu 2022 sanity check:")
        print(f"  {'t/L':>8}  {'rho_vox':>9}  {'rho_Liu':>9}  "
              f"{'error_%':>9}  disc_warn  status")
        print("  " + "-" * 65)
        all_pass = True
        for tL, _vox, rho_v in vox_cache:
            rho_l    = liu_rho(tL)
            err      = 100.0 * (rho_v - rho_l) / max(rho_l, 1e-9)
            is_first = (tL <= DISC_WARN_TL_MAX)
            thresh   = LIU_FIRST_WARN if is_first else LIU_STOP_PCT
            ok       = abs(err) <= thresh
            status   = "OK" if ok else "FAIL"
            if not ok:
                all_pass = False
            print(f"  {tL:>8.4f}  {rho_v:>9.4f}  {rho_l:>9.4f}  "
                  f"{err:>+9.2f}%  {'True ' if is_first else 'False'}      {status}")
        print()
        print(f"  Overall: {'ALL PASS' if all_pass else 'FAILURES DETECTED — review above'}")
        print()
        print("  Next step:")
        print("    python scripts/train_mechanical_surrogate.py")
    else:
        remaining_n = len(TL_GRID) - total_in_file
        print(f"  {remaining_n} run(s) still pending.  Re-launch this script to resume.")

    print("=" * 68)


if __name__ == "__main__":
    main()
