#!/usr/bin/env python3
"""OpenMP implementation-equivalence checks for NavSolver.

Distinct purpose from scripts/validate.py: that script checks PHYSICS
correctness (conservation, analytical solutions); this one checks that
navsolver_omp's answer doesn't depend on thread count, and doesn't vary
run-to-run at a fixed thread count. Two checks, both comparing
navsolver_omp against ITSELF, never against the separate serial binary:

1. **Thread-count equivalence**: navsolver_omp at each thread count in
   --threads must match navsolver_omp at 1 thread, within a documented
   relative tolerance (not bit-exact — floating-point addition isn't
   associative, and OpenMP's accumulation order differs by thread count).
   Deliberately NOT compared against navsolver (serial): the OpenMP build
   uses red-black SOR, a genuinely different iterative algorithm from
   serial's sequential Gauss-Seidel/SOR (needed for parallel-safety, see
   docs/openmp-parallelization.md) that only converges to the same answer
   as serial at HIGH sweep counts (scripts/validate.py's red-black tier
   checks that, gauge-fixed, at numPressureIter=3000) -- at the production
   sweep counts used here (5), comparing red-black to serial GS directly
   would fail from the algorithm difference alone, unrelated to whether
   OpenMP itself is correct. Comparing every thread count against
   1-thread-OpenMP isolates exactly the thing this script exists to check.

2. **Determinism**: navsolver_omp run TWICE at the SAME thread count, same
   config, must match tightly (near bit-exact). This is what actually
   catches races — schedule(static) makes work partitioning identical
   across runs at a fixed thread count, so if results differ run-to-run,
   something is racing (this exact check caught a real bug during
   development; see docs/openmp-parallelization.md).

Usage:
    python3 scripts/validate_parallel.py
    python3 scripts/validate_parallel.py --threads 1,2,4,8 --skip-build
    cmake --build build --target validate_parallel
"""

import argparse
from contextlib import nullcontext, redirect_stdout
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import (NAVSOLVER, NAVSOLVER_OMP,  # noqa: E402
                     checks_envelope, emit_json,
                     build, compare_velocity_fields, run)

EQUIVALENCE_TOL = 1e-10   # relative, navsolver_omp vs navsolver (floating-point reordering)
DETERMINISM_TOL = 1e-12   # relative, navsolver_omp vs itself at a fixed thread count

CONFIG_TEMPLATE = """\
numCellsX = {nx}
numCellsY = {ny}
numCellsZ = {nz}
baseUnit = {base_unit}
reynoldsNumber = {re}
hyperViscousStart = 0
geometryShape = {shape}
geometryType = Curved
lateralBC = {lateral}
outletBC = ZeroFirstDeriv
initialProfile = {profile}
flowType = SteadyMarching
maxTimeSteps = 20
convergenceTol = 0
numPressureIter = 5
reportEveryN = 20
outputDir = {out}
runName = vp
"""

# Two geometries: the non-periodic solid-wall path and the periodic path --
# these exercise different branches in updateColor()/computeAccelerations().
CONFIGS = {
    "abrupt_expansion": dict(nx=48, ny=24, nz=12, base_unit=0, re=100.0,
                              shape="AbruptExpansion", lateral="SolidWall",
                              profile="InletProfile"),
    "rounded_corner_periodic": dict(nx=60, ny=20, nz=10, base_unit=8, re=1000.0,
                                     shape="RoundedCorner", lateral="Periodic",
                                     profile="PotentialFlow"),
}


def check_thread_equivalence(threads_list):
    print("== Thread-count equivalence: navsolver_omp @ N threads vs @ 1 thread ==")
    ok = True
    other_threads = [t for t in threads_list if t != 1]
    if not other_threads:
        print("  (only 1 in --threads, nothing to compare)")
        return True
    with tempfile.TemporaryDirectory() as tmp:
        for name, params in CONFIGS.items():
            out_1t = Path(tmp) / f"{name}_1t"
            cfg_1t = Path(tmp) / f"{name}_1t.cfg"
            cfg_1t.write_text(CONFIG_TEMPLATE.format(out=out_1t, **params))
            run(NAVSOLVER_OMP, cfg_1t, threads=1)
            vtk_1t = out_1t / "vp_t000020.vtk"

            for t in other_threads:
                out_omp = Path(tmp) / f"{name}_t{t}"
                cfg_omp = Path(tmp) / f"{name}_t{t}.cfg"
                cfg_omp.write_text(CONFIG_TEMPLATE.format(out=out_omp, **params))
                run(NAVSOLVER_OMP, cfg_omp, threads=t)
                vtk_omp = out_omp / "vp_t000020.vtk"
                ok = compare_velocity_fields(vtk_1t, vtk_omp, EQUIVALENCE_TOL,
                                              f"{name}: {t} threads vs 1 thread") and ok
    return ok


def check_determinism(threads_list):
    print("\n== Determinism: navsolver_omp vs itself at a fixed thread count ==")
    ok = True
    max_threads = max(threads_list)
    with tempfile.TemporaryDirectory() as tmp:
        for name, params in CONFIGS.items():
            outs = []
            for run_idx in range(2):
                out_dir = Path(tmp) / f"{name}_det_{run_idx}"
                cfg_path = Path(tmp) / f"{name}_det_{run_idx}.cfg"
                cfg_path.write_text(CONFIG_TEMPLATE.format(out=out_dir, **params))
                run(NAVSOLVER_OMP, cfg_path, threads=max_threads)
                outs.append(out_dir / "vp_t000020.vtk")
            ok = compare_velocity_fields(outs[0], outs[1], DETERMINISM_TOL,
                                          f"{name} @ {max_threads} threads, run1 vs run2") and ok
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--threads", default="1,2,4,6,12",
                     help="comma-separated OMP_NUM_THREADS values to check (default: 1,2,4,6,12)")
    ap.add_argument("--skip-build", action="store_true")
    ap.add_argument("--json", action="store_true",
                     help="emit structured results on stdout")
    args = ap.parse_args()

    threads_list = [int(t) for t in args.threads.split(",")]

    if not args.skip_build:
        build("navsolver", "navsolver_omp", required=("navsolver",))
    elif not (NAVSOLVER.exists() and NAVSOLVER_OMP.exists()):
        ap.error(f"{NAVSOLVER} / {NAVSOLVER_OMP} not found; run without --skip-build first")

    if not NAVSOLVER_OMP.exists():
        print("navsolver_omp not built (no OpenMP toolchain?) — nothing to check.", file=sys.stderr)
        sys.exit(0)

    with redirect_stdout(sys.stderr) if args.json else nullcontext():
        ok = check_thread_equivalence(threads_list)
        ok = check_determinism(threads_list) and ok

    out = sys.stderr if args.json else sys.stdout
    print(file=out)
    if ok:
        print("All parallel-equivalence checks passed.", file=out)
    else:
        print("Parallel-equivalence checks FAILED.", file=out)

    if args.json:
        emit_json(checks_envelope("parallel"))

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
