"""SC mechanical FEA on the exact-geometry (no dilation) voxel model.

20 t/L values in [0.03, 0.18] (acoustic range). Mirrors run_fcc_fea_exact_tL.py.
Writes results/data/sc_mech_res100_dilation.csv (filename kept for downstream
compatibility, though geometry is now exact).
"""

import argparse, csv, gc, hashlib, sys, time, traceback
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from hompy import homogenize
from lattice.sc_plate_voxels import gen_sc

TL_GRID    = np.linspace(0.03, 0.18, 20)
RESOLUTION = 100
E_BULK, NU = 1.0, 0.30
LX = LY = LZ = 1.0
DIRECT     = False

CSV_HEADER = [
    "t_over_L", "rho", "voxel_hash",
    "C11", "C12", "C44",
    "time_s", "direct_solution", "resolution",
    "E", "nu", "lx", "ly", "lz",
    "rho_liu_approx",          # rough density prediction for sanity
    "discretization_warning",  # True for the thin t/L=0.03 plate
]

DISC_WARN_TL = 0.031   # thinnest point

def lame(E, nu):
    """lame parameters from E and nu"""
    return E*nu/((1+nu)*(1-2*nu)), E/(2*(1+nu))
def vhash(v):
    """md5 of the voxel array to spot duplicate geometries"""
    return hashlib.md5(v.tobytes()).hexdigest()

def sc_ref_rho(tR):
    """approx SC density: 3 orthogonal plates, inclusion-exclusion"""
    return 3*tR - 3*tR**2 + tR**3

def load_hashes(path: Path) -> Set[str]:
    """hashes already in the CSV for resume"""
    if not path.exists(): return set()
    with path.open("r", newline="") as f:
        hdr = f.readline()
    if "discretization_warning" not in hdr:
        stale = path.with_suffix(".csv.stale_schema_bak")
        path.rename(stale)
        print(f"  Old-schema CSV moved to {stale.name} — starting fresh.")
        return set()
    hashes = set()
    with path.open("r", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("voxel_hash"): hashes.add(row["voxel_hash"])
    return hashes

def append_row(path: Path, row: Dict, header: List[str]):
    """append one row to the CSV"""
    new = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if new: w.writeheader()
        w.writerow(row); f.flush()


def main():
    """run the SC exact-t/L FEA sweep, resuming from the CSV"""
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", default=str(PROJECT_ROOT/"results"/"data"/"sc_mech_res100_dilation.csv"))
    args = ap.parse_args()
    out  = Path(args.out)

    print("=" * 66)
    print("  SC FEA — exact geometry (no dilation), stellated-midplane plates")
    print("=" * 66)
    print(f"\n  Output:     {out}")
    print(f"  Resolution: {RESOLUTION}")
    print(f"  Material:   E={E_BULK}, nu={NU}")
    print(f"  t/L grid:   linspace({TL_GRID[0]:.3f}, {TL_GRID[-1]:.3f}, {len(TL_GRID)})\n")

    # pre-flight sweep
    print(f"  {'#':>3}  {'t/L':>8}  {'rho_vox':>9}  {'rho_approx':>12}  "
          f"{'error_%':>9}  disc_warn")
    print("  " + "-" * 58)
    cache = []
    for i, tL in enumerate(TL_GRID):
        vox   = gen_sc(RESOLUTION, tL)
        rho_v = float(vox.mean())
        rho_a = sc_ref_rho(tL)
        err   = 100*(rho_v - rho_a)/max(rho_a, 1e-9)
        warn  = tL <= DISC_WARN_TL
        print(f"  {i+1:>3}  {tL:>8.4f}  {rho_v:>9.4f}  {rho_a:>12.4f}  "
              f"{err:>+9.2f}%  {'True' if warn else 'False'}")
        cache.append((tL, vox, rho_v))

    if args.dry_run:
        print("\n  [DRY RUN] Done."); return

    done = load_hashes(out)
    remaining = [(tL, v, r) for tL,v,r in cache if vhash(v) not in done]
    if not remaining:
        print("  All rows already complete."); return
    print(f"\n  Runs to execute: {len(remaining)} / {len(TL_GRID)}")
    print(f"  Est. time:  ~{len(remaining)*8:.0f} min\n")

    lam, mu = lame(E_BULK, NU)
    done_n  = 0

    for run_i, (tL, vox, rho_v) in enumerate(remaining, 1):
        h    = vhash(vox)
        warn = tL <= DISC_WARN_TL
        print(f"  [{run_i:02d}/{len(remaining)}] t/L={tL:.6f}  rho={rho_v:.4f}  "
              f"hash={h[:10]}...  disc_warn={warn}")
        try:
            print(f"          homogenize()...", end="", flush=True)
            t0 = time.time()
            C  = homogenize(LX, LY, LZ, [lam], [mu], vox, DIRECT)
            dt = time.time() - t0
            print(f" {dt:.1f}s")
            print(f"          C11={C[0,0]:.5g}  C12={C[0,1]:.5g}  C44={C[3,3]:.5g}")
        except Exception:
            print(); traceback.print_exc()
            print(f"  ERROR — skipping. Completed rows safe in {out.name}.")
            continue

        append_row(out, {
            "t_over_L": round(float(tL), 10), "rho": round(rho_v, 6),
            "voxel_hash": h, "C11": float(C[0,0]), "C12": float(C[0,1]),
            "C44": float(C[3,3]), "time_s": round(dt, 4),
            "direct_solution": int(DIRECT), "resolution": RESOLUTION,
            "E": E_BULK, "nu": NU, "lx": LX, "ly": LY, "lz": LZ,
            "rho_liu_approx": round(sc_ref_rho(tL), 6),
            "discretization_warning": warn,
        }, CSV_HEADER)
        done_n += 1
        gc.collect()
        print()

    total = len(load_hashes(out))
    print("=" * 66)
    print(f"  Done. {done_n} runs this session. Total unique in file: {total}/20")
    if total == 20:
        print("  Next: python scripts/train_mechanical_surrogate.py")
    print("=" * 66)

if __name__ == "__main__":
    main()
