"""Fires off the SC and FCC HomPy jobs in parallel subprocesses. Run from project root."""

import subprocess
import sys
import threading
import time
import argparse
from pathlib import Path

SCRIPTS_DIR  = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS_DIR.parent

SC_DEFAULTS = dict(
    script=str(SCRIPTS_DIR / "generate_sc_dataset.py"),
    rho_min=0.05,
    rho_max=0.50,
    resolution=150,
    n_target=20,
    out=str(PROJECT_ROOT / "results" / "data" / "sc_mech_res150_dilation.csv"),
)

FCC_DEFAULTS = dict(
    script=str(SCRIPTS_DIR / "generate_fcc_dataset.py"),
    rho_min=0.02,
    rho_max=0.50,
    resolution=150,
    n_target=20,
    out=str(PROJECT_ROOT / "results" / "data" / "fcc_mech_res150_dilation.csv"),
)

def stream_output(proc: subprocess.Popen, label: str) -> None:
    """pipe a subprocess's stdout to ours, line by line, with a label"""
    for line in proc.stdout:
        print(f"[{label}] {line}", end="", flush=True)


def build_cmd(cfg: dict) -> list:
    """build the dataset-generator command line from a config dict"""
    return [
        sys.executable, cfg["script"],
        "--rho_min",    str(cfg["rho_min"]),
        "--rho_max",    str(cfg["rho_max"]),
        "--resolution", str(cfg["resolution"]),
        "--n_target",   str(cfg["n_target"]),
        "--out",        cfg["out"],
    ]


def run_jobs(sc_cfg: dict, fcc_cfg: dict) -> None:
    """launch the SC and FCC jobs in parallel and wait for both"""
    (PROJECT_ROOT / "results" / "data").mkdir(exist_ok=True)

    sc_cmd  = build_cmd(sc_cfg)
    fcc_cmd = build_cmd(fcc_cfg)

    print("=" * 60)
    print(f"Project root: {PROJECT_ROOT}")
    print()
    print("Starting SC job:")
    print("  " + " ".join(sc_cmd))
    print()
    print("Starting FCC job:")
    print("  " + " ".join(fcc_cmd))
    print("=" * 60)
    print()

    t0 = time.time()

    import os
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(PROJECT_ROOT) + (os.pathsep + existing if existing else "")

    sc_proc = subprocess.Popen(
        sc_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        cwd=str(PROJECT_ROOT),
        env=env,
    )
    fcc_proc = subprocess.Popen(
        fcc_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        cwd=str(PROJECT_ROOT),
        env=env,
    )

    sc_thread  = threading.Thread(target=stream_output, args=(sc_proc,  "SC "), daemon=True)
    fcc_thread = threading.Thread(target=stream_output, args=(fcc_proc, "FCC"), daemon=True)
    sc_thread.start()
    fcc_thread.start()

    sc_ret  = sc_proc.wait()
    fcc_ret = fcc_proc.wait()
    sc_thread.join()
    fcc_thread.join()

    elapsed = time.time() - t0
    h, m = divmod(int(elapsed), 3600)
    m, s = divmod(m, 60)

    print()
    print("=" * 60)
    print(f"Both jobs finished in {h:02d}h {m:02d}m {s:02d}s")
    print(f"  SC  exit code: {sc_ret}   -> {sc_cfg['out']}")
    print(f"  FCC exit code: {fcc_ret}   -> {fcc_cfg['out']}")
    if sc_ret != 0 or fcc_ret != 0:
        print("  WARNING: one or both jobs exited with non-zero code.")
    print("=" * 60)


def main() -> None:
    """parse args, build SC/FCC configs, kick off the parallel run"""
    parser = argparse.ArgumentParser(
        description="Run SC + FCC HomPy dataset generation in parallel."
    )
    parser.add_argument("--sc_res",  type=int, default=None)
    parser.add_argument("--fcc_res", type=int, default=None)
    parser.add_argument("--sc_out",  type=str, default=None)
    parser.add_argument("--fcc_out", type=str, default=None)
    parser.add_argument("--test", action="store_true",
                        help="Run 1 geometry each to verify setup before overnight run")
    args = parser.parse_args()

    sc_cfg  = dict(SC_DEFAULTS)
    fcc_cfg = dict(FCC_DEFAULTS)

    if args.sc_res:
        sc_cfg["resolution"] = args.sc_res
        sc_cfg["out"] = str(PROJECT_ROOT / "results" / "data" / f"sc_mech_res{args.sc_res}_dilation.csv")
    if args.fcc_res:
        fcc_cfg["resolution"] = args.fcc_res
        fcc_cfg["out"] = str(PROJECT_ROOT / "results" / "data" / f"fcc_mech_res{args.fcc_res}_dilation.csv")
    if args.sc_out:
        sc_cfg["out"] = args.sc_out
    if args.fcc_out:
        fcc_cfg["out"] = args.fcc_out

    if args.test:
        print("TEST MODE — 1 geometry each.")
        sc_cfg["n_target"]  = 1
        fcc_cfg["n_target"] = 1
        sc_cfg["out"]  = str(PROJECT_ROOT / "results" / "data" / "sc_test_dilation.csv")
        fcc_cfg["out"] = str(PROJECT_ROOT / "results" / "data" / "fcc_test_dilation.csv")

    run_jobs(sc_cfg, fcc_cfg)


if __name__ == "__main__":
    main()
