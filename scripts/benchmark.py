#!/usr/bin/env python3
"""Timing harness for NavSolver — establishes a baseline before optimizing.

Runs the (currently serial) solver across a matrix of grid sizes with a
fixed, convergence-independent workload (convergenceTol=0 forces every
config to run exactly maxTimeSteps, so runs are apples-to-apples), times
wall-clock execution, and records throughput in seconds/step and
seconds/(active-cell*step) so results are comparable across grid sizes and,
later, across the OpenMP/CUDA/MPI implementations.

Usage:
    python3 scripts/benchmark.py                    # default size matrix
    python3 scripts/benchmark.py --sizes small,medium
    python3 scripts/benchmark.py --repeats 5 --steps 100
    python3 scripts/benchmark.py --skip-build        # reuse existing ./navsolver

    # OpenMP strong-scaling sweep:
    python3 scripts/benchmark.py --binary navsolver_omp --threads 1,2,4,6,12

No external Python dependencies (stdlib only) so it runs without the venv.
Plotting a scaling curve from the resulting CSV: scripts/plot_scaling.py.
"""

import argparse
import csv
import os
import platform
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
NAVSOLVER = REPO_ROOT / "navsolver"

# Grid sizes only — geometry is fixed to AbruptExpansion (indices derived
# purely from grid size, no baseUnit to misconfigure) so this harness
# measures raw solver throughput, not geometry-specific behavior.
SIZE_MATRIX = {
    "tiny":   (24, 12, 6),
    "small":  (48, 24, 12),
    "medium": (96, 48, 24),
    "large":  (144, 72, 36),
}

ACTIVE_CELLS_RE = re.compile(r"Active cells:\s*(\d+)")
STEPS_RE = re.compile(r"Total steps\s*:\s*(\d+)")


def build(make_target="all"):
    print(f"Building (make {make_target})...", file=sys.stderr)
    subprocess.run(["make", "clean"], cwd=REPO_ROOT, check=True,
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(["make", make_target], cwd=REPO_ROOT, check=True,
                    stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def make_config(nx, ny, nz, steps, num_pressure_iter, out_dir):
    return f"""\
numCellsX      = {nx}
numCellsY      = {ny}
numCellsZ      = {nz}
reynoldsNumber = 100.0
hyperViscousStart = 0
geometryShape  = AbruptExpansion
lateralBC      = SolidWall
outletBC       = ZeroFirstDeriv
initialProfile = InletProfile
flowType       = SteadyMarching
maxTimeSteps   = {steps}
convergenceTol = 0
numPressureIter = {num_pressure_iter}
reportEveryN   = {steps + 1}
outputDir      = {out_dir}
runName        = bench
"""


def run_once(binary, cfg_path, threads=None):
    env = dict(os.environ)
    if threads is not None:
        env["OMP_NUM_THREADS"] = str(threads)
    start = time.perf_counter()
    result = subprocess.run([str(binary), str(cfg_path)],
                             capture_output=True, text=True, cwd=REPO_ROOT, env=env)
    elapsed = time.perf_counter() - start
    if result.returncode != 0:
        raise RuntimeError(f"{binary} exited {result.returncode}:\n{result.stderr}\n{result.stdout}")
    m = ACTIVE_CELLS_RE.search(result.stdout)
    active_cells = int(m.group(1)) if m else None
    m = STEPS_RE.search(result.stdout)
    steps_ran = int(m.group(1)) if m else None
    return elapsed, active_cells, steps_ran


def git_commit():
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              cwd=REPO_ROOT, capture_output=True, text=True, check=True)
        commit = out.stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=REPO_ROOT,
                                capture_output=True, text=True, check=True).stdout.strip()
        return commit + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sizes", default="tiny,small,medium",
                     help=f"comma-separated subset of {list(SIZE_MATRIX)} (default: tiny,small,medium)")
    ap.add_argument("--steps", type=int, default=50, help="maxTimeSteps per run (default: 50)")
    ap.add_argument("--repeats", type=int, default=3, help="repeats per size, min is reported (default: 3)")
    ap.add_argument("--num-pressure-iter", type=int, default=5)
    ap.add_argument("--skip-build", action="store_true", help="reuse existing binary")
    ap.add_argument("--binary", default="navsolver",
                     help="binary name under the repo root to run (default: navsolver; "
                          "use navsolver_omp for an OpenMP strong-scaling sweep)")
    ap.add_argument("--threads", default=None,
                     help="comma-separated OMP_NUM_THREADS values to sweep (default: none -- "
                          "don't set the env var, i.e. a single run at the binary's default). "
                          "Ignored (harmlessly) by the plain serial binary.")
    ap.add_argument("--out", type=Path, default=None,
                     help="CSV output path (default: experiments/results/benchmarks/bench_<ts>.csv)")
    args = ap.parse_args()

    sizes = [s.strip() for s in args.sizes.split(",")]
    for s in sizes:
        if s not in SIZE_MATRIX:
            ap.error(f"unknown size '{s}', choose from {list(SIZE_MATRIX)}")

    binary_path = REPO_ROOT / args.binary
    threads_list = [int(t) for t in args.threads.split(",")] if args.threads else [None]

    if not args.skip_build:
        make_target = "openmp" if args.binary == "navsolver_omp" else "all"
        build(make_target)
    elif not binary_path.exists():
        ap.error(f"{binary_path} not found; run without --skip-build first")

    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out_path = args.out or (REPO_ROOT / "experiments" / "results" / "benchmarks" / f"bench_{timestamp}.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    commit = git_commit()
    cpu = platform.processor() or platform.machine()
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
    except FileNotFoundError:
        pass

    rows = []
    print(f"{'size':<8} {'threads':>7} {'grid':<14} {'cells':>10} {'min s':>10} {'s/step':>10} {'ns/cell-step':>14}")
    with tempfile.TemporaryDirectory() as tmp:
        for size in sizes:
            nx, ny, nz = SIZE_MATRIX[size]
            for threads in threads_list:
                tag = f"{size}_{threads}" if threads else size
                out_dir = Path(tmp) / tag
                cfg_path = Path(tmp) / f"{tag}.cfg"
                cfg_path.write_text(make_config(nx, ny, nz, args.steps, args.num_pressure_iter, out_dir))

                times, active_cells, steps_ran = [], None, None
                for _ in range(args.repeats):
                    elapsed, active_cells, steps_ran = run_once(binary_path, cfg_path, threads)
                    times.append(elapsed)

                if steps_ran != args.steps:
                    print(f"  WARNING: {tag} ran {steps_ran} steps, expected {args.steps} "
                          f"(convergenceTol=0 should prevent early stop)", file=sys.stderr)

                best = min(times)
                mean = sum(times) / len(times)
                per_step = best / args.steps
                per_cell_step = (per_step / active_cells) * 1e9 if active_cells else None

                rows.append({
                    "size": size, "numCellsX": nx, "numCellsY": ny, "numCellsZ": nz,
                    "active_cells": active_cells, "maxTimeSteps": args.steps,
                    "threads": threads or 1, "binary": args.binary,
                    "repeats": args.repeats, "min_seconds": best, "mean_seconds": mean,
                    "seconds_per_step": per_step, "ns_per_active_cell_step": per_cell_step,
                    "git_commit": commit, "cpu": cpu, "cores": platform.os.cpu_count(),
                    "timestamp_utc": timestamp,
                })
                print(f"{size:<8} {threads or 1:>7} {f'{nx}x{ny}x{nz}':<14} {active_cells or 0:>10} "
                      f"{best:>10.4f} {per_step:>10.5f} {per_cell_step or 0:>14.1f}")

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
