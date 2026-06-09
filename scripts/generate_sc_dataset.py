import time
import csv
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import numpy as np
import hashlib

from hompy import homogenize
from lattice.sc_plate_voxels import gen_sc


def lame_from_E_nu(E: float, nu: float) -> Tuple[float, float]:
    """lame parameters from E and nu"""
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    return lam, mu


def voxel_hash(vox: np.ndarray) -> str:
    """md5 of the voxel array to spot duplicate geometries"""
    # fingerprint to spot duplicate geometries
    return hashlib.md5(vox.tobytes()).hexdigest()


def rho_of_t(t_over_L: float, resolution: int) -> float:
    """relative density from t/L"""
    vox = gen_sc(resolution=resolution, t_over_L=float(t_over_L))
    return float(vox.mean())


def infer_t_range_from_rho(
    rho_min: float,
    rho_max: float,
    resolution: int,
    t_scan_min: float = 0.005,
    t_scan_max: float = 0.40,
    n_scan: int = 500,
) -> Tuple[float, float]:
    """scan for the t/L window whose voxel density sits in [rho_min, rho_max]"""
    if not (0.0 < rho_min < rho_max < 1.0):
        raise ValueError("Require 0 < rho_min < rho_max < 1")

    ts = np.linspace(t_scan_min, t_scan_max, n_scan)
    rhos = np.array([rho_of_t(t, resolution) for t in ts], dtype=float)

    idx_low = np.where(rhos >= rho_min)[0]
    idx_high = np.where(rhos <= rho_max)[0]

    if len(idx_low) == 0:
        raise ValueError(
            f"rho_min={rho_min} not reachable in scan range "
            f"[{t_scan_min},{t_scan_max}] at resolution={resolution}. "
            f"Max rho in scan: {rhos.max():.4f}"
        )
    if len(idx_high) == 0:
        raise ValueError(
            f"rho_max={rho_max} not reachable in scan range "
            f"[{t_scan_min},{t_scan_max}] at resolution={resolution}. "
            f"Min rho in scan: {rhos.min():.4f}"
        )

    t_low = float(ts[idx_low[0]])
    t_high = float(ts[idx_high[-1]])

    if t_low >= t_high:
        raise ValueError(
            f"Inferred thickness interval empty: t_low={t_low:.6f}, t_high={t_high:.6f}. "
            f"Try adjusting rho bounds or scan bounds."
        )

    return t_low, t_high


def run_one_sample(
    t_over_L: float,
    resolution: int,
    E: float,
    nu: float,
    lx: float,
    ly: float,
    lz: float,
    direct_solution: bool,
    rho_min: float,
    rho_max: float,
) -> Optional[Dict[str, Any]]:
    """voxelise one t/L, density-filter, homogenise, return the row dict"""
    vox = gen_sc(resolution=resolution, t_over_L=float(t_over_L))
    rho = float(vox.mean())

    if rho < rho_min or rho > rho_max:
        return None

    h = voxel_hash(vox)

    lam, mu = lame_from_E_nu(E, nu)

    t0 = time.time()
    C = homogenize(lx, ly, lz, [lam], [mu], vox, direct_solution)
    dt = time.time() - t0

    return {
        "t_over_L": float(t_over_L),
        "rho": float(rho),
        "voxel_hash": h,
        "C11": float(C[0, 0]),
        "C12": float(C[0, 1]),
        "C44": float(C[3, 3]),
        "time_s": float(dt),
        "direct_solution": int(bool(direct_solution)),
        "resolution": int(resolution),
        "E": float(E),
        "nu": float(nu),
        "lx": float(lx),
        "ly": float(ly),
        "lz": float(lz),
    }


def load_existing_hashes(csv_path: Path) -> Tuple[set, int]:
    """read back an existing CSV, return (hashes, row count)"""
    if not csv_path.exists():
        return set(), 0

    hashes = set()
    n = 0
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            n += 1
            if "voxel_hash" in row and row["voxel_hash"]:
                hashes.add(row["voxel_hash"])
    return hashes, n


def append_row(csv_path: Path, row: Dict[str, Any], header: list):
    """append one row to the CSV"""
    new_file = not csv_path.exists()
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if new_file:
            writer.writeheader()
        writer.writerow(row)
        f.flush()


def main():
    """run the SC mech dataset generation end to end"""
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--n_target", type=int, default=250)
    p.add_argument("--resolution", type=int, default=30)

    # density bounds drive everything else
    p.add_argument("--rho_min", type=float, required=True)
    p.add_argument("--rho_max", type=float, required=True)

    # scan that maps density bounds to a t/L range
    p.add_argument("--t_scan_min", type=float, default=0.005)
    p.add_argument("--t_scan_max", type=float, default=0.40)
    p.add_argument("--n_scan", type=int, default=500)

    # thicknesses sampled inside that range
    p.add_argument("--n_candidates", type=int, default=5000,
                   help="number of candidate thickness values to try (voxelised + hashed)")

    # homogenisation params
    p.add_argument("--E", type=float, default=1.0)
    p.add_argument("--nu", type=float, default=0.30)
    p.add_argument("--lx", type=float, default=1.0)
    p.add_argument("--ly", type=float, default=1.0)
    p.add_argument("--lz", type=float, default=1.0)
    p.add_argument("--direct", action="store_true")

    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    out_path = Path(args.out) if args.out else Path(f"results/data/sc_mech_res{args.resolution}.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    header = [
        "t_over_L", "rho", "voxel_hash", "C11", "C12", "C44", "time_s",
        "direct_solution", "resolution", "E", "nu", "lx", "ly", "lz"
    ]

    # density window -> thickness range
    print("Inferring thickness range from density bounds (voxel-only scan)...")
    t_low, t_high = infer_t_range_from_rho(
        rho_min=args.rho_min,
        rho_max=args.rho_max,
        resolution=args.resolution,
        t_scan_min=args.t_scan_min,
        t_scan_max=args.t_scan_max,
        n_scan=args.n_scan,
    )
    print(f"Target density range: [{args.rho_min:.4f}, {args.rho_max:.4f}]")
    print(f"Inferred t/L range:  [{t_low:.5f}, {t_high:.5f}]")
    print()

    # resume from whatever's already in the file
    seen_hashes, n_existing = load_existing_hashes(out_path)
    if n_existing > 0:
        print(f"Resuming; existing rows: {n_existing} in {out_path}")
        print(f"Unique voxel hashes already present: {len(seen_hashes)}")
    else:
        print("Starting new dataset.")
    print()

    rng = np.random.default_rng(args.seed)

    # spread candidates over [t_low, t_high]s
    base = np.linspace(t_low, t_high, args.n_candidates, endpoint=True)
    jitter = rng.normal(scale=(t_high - t_low) / (10 * args.n_candidates), size=args.n_candidates)
    candidates = np.clip(base + jitter, t_low, t_high)

    accepted = 0
    tried = 0
    unique_found = 0

    for t in candidates:
        t = float(t)
        tried += 1

        # voxelise + filter + hash
        vox = gen_sc(resolution=args.resolution, t_over_L=t)
        rho = float(vox.mean())
        if rho < args.rho_min or rho > args.rho_max:
            continue

        h = voxel_hash(vox)
        if h in seen_hashes:
            continue

        # new geometry
        seen_hashes.add(h)
        unique_found += 1

        lam, mu = lame_from_E_nu(args.E, args.nu)
        t0 = time.time()
        C = homogenize(args.lx, args.ly, args.lz, [lam], [mu], vox, args.direct)
        dt = time.time() - t0

        row = {
            "t_over_L": t,
            "rho": rho,
            "voxel_hash": h,
            "C11": float(C[0, 0]),
            "C12": float(C[0, 1]),
            "C44": float(C[3, 3]),
            "time_s": float(dt),
            "direct_solution": int(bool(args.direct)),
            "resolution": int(args.resolution),
            "E": float(args.E),
            "nu": float(args.nu),
            "lx": float(args.lx),
            "ly": float(args.ly),
            "lz": float(args.lz),
        }
        append_row(out_path, row, header)

        accepted += 1
        print(f"[{accepted:03d}/{args.n_target}] t/L={t:.5f} rho={rho:.4f} "
              f"C11={row['C11']:.4g} time={dt:.1f}s  (unique #{unique_found}, tried {tried})")

        if accepted >= args.n_target:
            break

    if accepted < args.n_target:
        print("\nWARNING: did not reach n_target.")
        print(f"Accepted {accepted}, needed {args.n_target}.")
        print("Increase --n_candidates, widen density bounds, or increase resolution.")

    print("\nDone.")
    print(f"Wrote: {out_path}")
    print(f"Unique geometries in file now: {len(seen_hashes)}")


if __name__ == "__main__":
    main()
