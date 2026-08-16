#!/usr/bin/env python3
"""Roofline analysis for NavSolver's two hottest kernels.

Combines three things, each measured/derived separately and documented:

1. Peak ceilings — empirically measured on THIS machine via the
   microbenchmarks in scripts/roofline/ (not vendor spec sheets):
   peak FMA GFLOP/s, peak memory bandwidth, peak std::exp() throughput.
   Hardware performance counters (perf/LIKWID) aren't reliably usable under
   this WSL2 sandbox, so this is the documented analytical/empirical
   fallback methodology, not the industry-standard automated path (Intel
   Advisor/LIKWID) — see docs/roofline.md.

2. Analytical FLOP/byte/exp-call counts per interior cell for
   computeAccelerations and solvePressurePoisson, derived by hand from
   src/backends/serial/Physics.cpp (see docs/roofline.md for the full derivation —
   these constants are NOT re-derived automatically, they're transcribed
   from that manual count, so they must be updated by hand if the kernels
   change).

3. Achieved performance — a clean `-O3` Release build's total wall-clock
   time (never the -pg build, which distorts absolute timing) apportioned
   by a `-pg`/gprof profile's *relative* self-time breakdown, on the same
   workload used in docs/serial-optimization.md's baseline (AbruptExpansion,
   96x48x24, 100 steps, SteadyMarching). This hybrid avoids trusting either
   measurement for what it's bad at.

Usage:
    python3 scripts/roofline.py
"""

import csv
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import REPO_ROOT  # noqa: E402

ROOFLINE_DIR = REPO_ROOT / "scripts" / "roofline"
RESULTS_DIR = REPO_ROOT / "experiments" / "results" / "roofline"
FIGURES_DIR = REPO_ROOT / "experiments" / "figures"

WORKLOAD_CFG = """\
numCellsX = 96
numCellsY = 48
numCellsZ = 24
reynoldsNumber = 100.0
hyperViscousStart = 0
geometryShape = AbruptExpansion
lateralBC = SolidWall
outletBC = ZeroFirstDeriv
initialProfile = InletProfile
flowType = SteadyMarching
maxTimeSteps = 100
convergenceTol = 0
numPressureIter = 5
reportEveryN = 101
outputDir = {out}
runName = roofline
"""

# ── Analytical counts, transcribed from the hand-derivation in docs/roofline.md ──
# Per INTERIOR cell, per call to the kernel (i.e. already summed across all
# 3 direction sweeps for computeAccelerations). "Bytes" is the conservative
# no-cross-pass-register-reuse estimate: every distinct field-array element
# access, in every separate pass/statement, counted as one 8-byte DRAM
# transaction. Real achieved bytes are likely higher still for the X/Y
# sweeps specifically, due to non-unit-stride access wasting most of each
# fetched 64-byte cache line — see docs/roofline.md for that discussion,
# which is the actual point of this analysis.
ACCEL_FLOPS_PER_CELL = 237.0     # 79 FLOPs/sweep x 3 sweeps (X/Y/Z treated as structurally equal cost)
ACCEL_EXP_CALLS_PER_CELL = 6.0   # 2 exp() calls/sweep x 3 sweeps
ACCEL_BYTES_PER_CELL = 2400.0    # ~800 bytes/sweep x 3 sweeps (2+9+9 = 20 field-array accesses/sweep x 8 bytes x ~5 for read-modify-write on accel arrays across passes 2 and 5 -- see docs/roofline.md

PRESSURE_FLOPS_PER_CELL = 13.0   # SOR-blended Gauss-Seidel update
PRESSURE_BYTES_PER_CELL = 72.0   # 7 neighbor+source reads + 1 self-read + 1 write, no reuse assumed

CALLS_PER_STEP = {"computeAccelerations": 1, "solvePressurePoisson": 1}  # SteadyMarching; RK4Transient is 5 and 4 respectively -- not used by this workload

ACTIVE_CELLS_RE = re.compile(r"Active cells:\s*(\d+)")
GPROF_FLAT_RE = re.compile(
    r"^\s*[\d.]+\s+[\d.]+\s+([\d.]+)\s+\d+\s+[\d.]+\s+[\d.]+\s+(\w+)\(SimState")


def run(cmd, cwd=None, check=True):
    return subprocess.run(cmd, cwd=cwd or REPO_ROOT, capture_output=True, text=True, check=check)


def build_microbenchmarks():
    print("Building roofline microbenchmarks...", file=sys.stderr)
    flags = ["-O3", "-march=native", "-funroll-loops"]
    for name in ("peak_flops", "peak_bandwidth", "peak_exp_throughput"):
        src = ROOFLINE_DIR / f"{name}.cpp"
        run(["g++", *flags, "-o", str(ROOFLINE_DIR / name), str(src)])
        run(["g++", *flags, "-fopenmp", "-DUSE_OPENMP", "-o", str(ROOFLINE_DIR / f"{name}_omp"), str(src)])


def measure_peaks():
    peaks = {}
    r = run([str(ROOFLINE_DIR / "peak_flops")])
    peaks["fma_gflops_1t"] = float(re.search(r"total_gflops=([\d.]+)", r.stdout).group(1))
    best = 0.0
    for _ in range(3):
        r = run([str(ROOFLINE_DIR / "peak_flops_omp")])
        best = max(best, float(re.search(r"total_gflops=([\d.]+)", r.stdout).group(1)))
    peaks["fma_gflops_nt"] = best

    r = run([str(ROOFLINE_DIR / "peak_bandwidth")])
    peaks["bw_gbps_1t"] = float(re.search(r"bandwidth_gbps=([\d.]+)", r.stdout).group(1))
    best = 0.0
    for _ in range(3):
        r = run([str(ROOFLINE_DIR / "peak_bandwidth_omp")])
        best = max(best, float(re.search(r"bandwidth_gbps=([\d.]+)", r.stdout).group(1)))
    peaks["bw_gbps_nt"] = best

    r = run([str(ROOFLINE_DIR / "peak_exp_throughput")])
    peaks["exp_per_sec_1t"] = float(re.search(r"exp_calls_per_sec=([\d.e+]+)", r.stdout).group(1))
    best = 0.0
    for _ in range(3):
        r = run([str(ROOFLINE_DIR / "peak_exp_throughput_omp")])
        best = max(best, float(re.search(r"exp_calls_per_sec=([\d.e+]+)", r.stdout).group(1)))
    peaks["exp_per_sec_nt"] = best
    return peaks


def measure_achieved():
    """Clean -O3 total time x gprof's relative self-time share, per kernel."""
    workload = RESULTS_DIR / "workload.cfg"
    out_dir = RESULTS_DIR / "workload_out"
    workload.parent.mkdir(parents=True, exist_ok=True)
    workload.write_text(WORKLOAD_CFG.format(out=out_dir))

    print("Building profile (-pg) binary for relative attribution...", file=sys.stderr)
    run(["make", "clean"], check=False)
    run(["make", "profile"])
    navsolver = REPO_ROOT / "navsolver"
    profile_run = run([str(navsolver), str(workload)])
    gmon = REPO_ROOT / "gmon.out"
    gprof_out = run(["gprof", str(navsolver), str(gmon)]).stdout

    self_times = {}
    for line in gprof_out.splitlines():
        m = GPROF_FLAT_RE.match(line)
        if m:
            t, name = float(m.group(1)), m.group(2)
            self_times[name] = t

    m = ACTIVE_CELLS_RE.search(profile_run.stdout)
    active_cells = int(m.group(1)) if m else None

    print("Building clean Release binary for absolute time...", file=sys.stderr)
    run(["make", "clean"])
    run(["make"])
    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        run([str(navsolver), str(workload)])
        times.append(time.perf_counter() - t0)
    clean_total = min(times)

    return self_times, clean_total, active_cells, gprof_out


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    build_microbenchmarks()
    peaks = measure_peaks()
    print("Peak ceilings:", peaks, file=sys.stderr)

    self_times, clean_total, active_cells, gprof_raw = measure_achieved()
    (RESULTS_DIR / "gprof_raw.txt").write_text(gprof_raw)
    print(f"gprof self-times: {self_times}, clean total: {clean_total:.4f}s, "
          f"active_cells: {active_cells}", file=sys.stderr)

    total_self = sum(self_times.values())
    kernels = {}
    for name, flops_c, exp_c, bytes_c in [
        ("computeAccelerations", ACCEL_FLOPS_PER_CELL, ACCEL_EXP_CALLS_PER_CELL, ACCEL_BYTES_PER_CELL),
        ("solvePressurePoisson", PRESSURE_FLOPS_PER_CELL, 0.0, PRESSURE_BYTES_PER_CELL),
    ]:
        if name not in self_times or not active_cells:
            print(f"WARNING: no gprof data for {name}, skipping", file=sys.stderr)
            continue
        share = self_times[name] / total_self
        achieved_time = clean_total * share
        calls = CALLS_PER_STEP[name] * 100  # 100 steps in WORKLOAD_CFG
        total_flops = flops_c * active_cells * calls
        total_bytes = bytes_c * active_cells * calls
        total_exp = exp_c * active_cells * calls

        achieved_gflops = total_flops / achieved_time / 1e9
        ai = total_flops / total_bytes
        exp_ceiling_gflops = (flops_c / exp_c * peaks["exp_per_sec_1t"] / 1e9) if exp_c > 0 else None

        kernels[name] = {
            "gprof_share": share, "achieved_time_s": achieved_time,
            "achieved_gflops": achieved_gflops, "arithmetic_intensity": ai,
            "exp_ceiling_gflops_1t": exp_ceiling_gflops,
        }

    with open(RESULTS_DIR / "roofline_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["kernel", "gprof_share", "achieved_time_s", "achieved_gflops",
                    "arithmetic_intensity", "exp_ceiling_gflops_1t",
                    "fma_gflops_1t", "fma_gflops_nt", "bw_gbps_1t", "bw_gbps_nt"])
        for name, k in kernels.items():
            w.writerow([name, k["gprof_share"], k["achieved_time_s"], k["achieved_gflops"],
                        k["arithmetic_intensity"], k["exp_ceiling_gflops_1t"],
                        peaks["fma_gflops_1t"], peaks["fma_gflops_nt"],
                        peaks["bw_gbps_1t"], peaks["bw_gbps_nt"]])

    print("\n=== Roofline summary ===")
    for name, k in kernels.items():
        print(f"{name}: AI={k['arithmetic_intensity']:.3f} FLOP/byte, "
              f"achieved={k['achieved_gflops']:.3f} GFLOP/s, "
              f"bandwidth-implied ceiling at this AI = "
              f"{k['arithmetic_intensity']*peaks['bw_gbps_1t']:.2f} GFLOP/s (1T), "
              f"exp-ceiling={k['exp_ceiling_gflops_1t']}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 6))
        ai_range = [0.01, 100]
        ax.plot(ai_range, [ai * peaks["bw_gbps_1t"] for ai in ai_range], "b--", label=f"BW roof 1T ({peaks['bw_gbps_1t']:.1f} GB/s)")
        ax.axhline(peaks["fma_gflops_1t"], color="r", linestyle="--", label=f"FMA peak 1T ({peaks['fma_gflops_1t']:.1f} GFLOP/s)")
        for name, k in kernels.items():
            ax.plot(k["arithmetic_intensity"], k["achieved_gflops"], "o", markersize=10, label=f"{name} (achieved)")
            if k["exp_ceiling_gflops_1t"]:
                ax.axhline(k["exp_ceiling_gflops_1t"], color="g", linestyle=":", label=f"{name} exp() ceiling 1T")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("Arithmetic Intensity (FLOP/byte)")
        ax.set_ylabel("Performance (GFLOP/s)")
        ax.set_title("NavSolver serial roofline (Intel Core 7 150U, single-thread ceilings)")
        ax.legend(fontsize=8)
        ax.grid(True, which="both", alpha=0.3)
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "roofline.png", dpi=150)
        print(f"\nWrote {FIGURES_DIR / 'roofline.png'}", file=sys.stderr)
    except ImportError:
        print("\nmatplotlib not installed -- skipping plot. "
              "pip install matplotlib into .venv to enable it.", file=sys.stderr)

    print(f"\nWrote {RESULTS_DIR / 'roofline_summary.csv'}", file=sys.stderr)


if __name__ == "__main__":
    main()
