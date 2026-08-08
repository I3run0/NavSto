#!/usr/bin/env python3
"""CUDA implementation-correctness checks for NavSolver.

Structurally mirrors scripts/validate_parallel.py (same
build-then-compare-VTK-fields shape), adapted for the two things that
differ about the CUDA port:

1. **Red-black equivalence, gauge-fixed** (navsolver_cuda vs navsolver):
   solvePressurePoisson on the CUDA path uses the SAME red-black SOR
   restructuring as navsolver_omp (see src/common/RedBlackIndexing.hpp,
   src/cuda/Physics.cu's solvePressurePoissonCuda) -- a different iterative
   algorithm from the serial solver's sequential Gauss-Seidel/SOR, needed
   for GPU parallel-safety. It converges to the SAME fixed point via a
   DIFFERENT path, so -- exactly like scripts/validate.py's tier 3 -- this
   compares gauge-fixed (each field's own mean subtracted) final pressure
   at a large sweep count on a well-posed Straight-geometry config, not a
   raw pointwise comparison at production sweep counts.

2. **Determinism**: navsolver_cuda run multiple times on the SAME config
   must match near-bit-exactly. mirrorGhostCells is now two kernels
   (cross-row writes, then same-row writes, sequential launch order) --
   see docs/cuda-port.md's "Follow-up" section for why that's still
   race-free despite CUDA having no in-launch ordering guarantee. No
   race detector (compute-sanitizer fails to launch in this environment)
   was available to verify that claim directly, so this determinism check
   -- run at both the default small config AND a multi-block stress config
   -- is the actual verification evidence for that fix, not just a sanity
   check.

VRAM sizing note: the largest config below (48x24x12 = 13,824 cells, plus
ghost padding) allocates roughly a few hundred field/scratch buffers on
that order times 8 bytes -- order 10-50 MiB total, negligible next to the
MX570's 4 GiB (see docs/cuda-port.md for the actual measured figure).

Usage:
    python3 scripts/validate_cuda.py
    python3 scripts/validate_cuda.py --skip-build
    make validate-cuda
"""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from validate import (REPO_ROOT, NAVSOLVER, parse_vtk_scalar,  # noqa: E402
                       REDBLACK_CFG_TEMPLATE, REDBLACK_TOL)
from validate_parallel import compare_velocity_fields  # noqa: E402

NAVSOLVER_CUDA = REPO_ROOT / "navsolver_cuda"

DETERMINISM_TOL = 1e-12

DET_CFG_TEMPLATE = """\
numCellsX = 24
numCellsY = 12
numCellsZ = 6
reynoldsNumber = 100.0
hyperViscousStart = 0
geometryShape = AbruptExpansion
lateralBC = SolidWall
outletBC = ZeroFirstDeriv
initialProfile = InletProfile
flowType = SteadyMarching
maxTimeSteps = 20
convergenceTol = 0
numPressureIter = 5
reportEveryN = 20
outputDir = {out}
runName = cuda_det
"""


def build():
    print("Building (Release, CMake)...", file=sys.stderr)
    build_dir = REPO_ROOT / "build"
    subprocess.run(["cmake", "-S", str(REPO_ROOT), "-B", str(build_dir),
                     "-DCMAKE_BUILD_TYPE=Release"], check=True,
                    stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    subprocess.run(["cmake", "--build", str(build_dir), "--target", "navsolver", "-j"],
                    check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    result = subprocess.run(["cmake", "--build", str(build_dir), "--target", "navsolver_cuda", "-j"],
                             stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    if result.returncode == 0:
        built_bin = build_dir / "navsolver_cuda"
        if built_bin.exists():
            import shutil
            shutil.copy(built_bin, NAVSOLVER_CUDA)
        built_serial = build_dir / "navsolver"
        if built_serial.exists():
            import shutil
            shutil.copy(built_serial, NAVSOLVER)
    return result.returncode == 0 and NAVSOLVER_CUDA.exists()


def run(binary, cfg_path):
    result = subprocess.run([str(binary), str(cfg_path)], capture_output=True,
                             text=True, cwd=REPO_ROOT)
    if result.returncode != 0:
        raise RuntimeError(f"{binary} exited {result.returncode}:\n{result.stderr}\n{result.stdout}")


def check_redblack_equivalence():
    print("== Red-black equivalence: navsolver vs navsolver_cuda (gauge-fixed) ==")
    with tempfile.TemporaryDirectory() as tmp:
        cfg_path = Path(tmp) / "redblack.cfg"

        out_serial = Path(tmp) / "serial"
        cfg_path.write_text(REDBLACK_CFG_TEMPLATE.format(out=out_serial))
        run(NAVSOLVER, cfg_path)

        out_cuda = Path(tmp) / "cuda"
        cfg_path.write_text(REDBLACK_CFG_TEMPLATE.format(out=out_cuda))
        run(NAVSOLVER_CUDA, cfg_path)

        vtk_serial = out_serial / "redblack_check_t000001.vtk"
        vtk_cuda = out_cuda / "redblack_check_t000001.vtk"
        (nx, ny, nz), press_serial, _ = parse_vtk_scalar(vtk_serial, "Pressure")
        (nx2, ny2, nz2), press_cuda, _ = parse_vtk_scalar(vtk_cuda, "Pressure")
        if (nx, ny, nz) != (nx2, ny2, nz2):
            print("  FAIL: grid size mismatch between serial and CUDA output")
            return False

        mean_serial = sum(press_serial) / len(press_serial)
        mean_cuda = sum(press_cuda) / len(press_cuda)
        diffs_sq = 0.0
        serial_sq = 0.0
        for ps, pc in zip(press_serial, press_cuda):
            d = (ps - mean_serial) - (pc - mean_cuda)
            diffs_sq += d * d
            serial_sq += (ps - mean_serial) ** 2
        l2_rel = (diffs_sq / max(serial_sq, 1e-30)) ** 0.5

        if l2_rel > REDBLACK_TOL:
            print(f"  FAIL: gauge-fixed pressure fields disagree "
                  f"(L2_rel={l2_rel:.3e}, tol={REDBLACK_TOL:.0e})")
            return False
        print(f"  PASS: gauge-fixed pressure fields agree (L2_rel={l2_rel:.3e})")
        return True


def check_determinism(repeats=5):
    # 5 repeats, not 2: matches the rigor of the OpenMP mirrorGhostCells
    # verification (docs/openmp-parallelization.md's Round 2 -- "5
    # repeated runs, byte-for-byte identical") now that the CUDA
    # mirrorGhostCells kernel is also parallel (two-pass, one thread per
    # row) instead of the original single-thread <<<1,1>>> launch. CUDA
    # has weaker in-launch ordering guarantees than OpenMP's
    # schedule(static), so more repeats buys more confidence per run.
    print(f"\n== Determinism: navsolver_cuda vs itself, same config ({repeats} repeats) ==")
    with tempfile.TemporaryDirectory() as tmp:
        outs = []
        for run_idx in range(repeats):
            out_dir = Path(tmp) / f"det_{run_idx}"
            cfg_path = Path(tmp) / f"det_{run_idx}.cfg"
            cfg_path.write_text(DET_CFG_TEMPLATE.format(out=out_dir))
            run(NAVSOLVER_CUDA, cfg_path)
            outs.append(out_dir / "cuda_det_t000020.vtk")
        ok = True
        for run_idx in range(1, repeats):
            ok = compare_velocity_fields(outs[0], outs[run_idx], DETERMINISM_TOL,
                                          f"run1 vs run{run_idx + 1}") and ok
        return ok


# gridFor(numCellsX) with CUDA_BLOCK=256 means DET_CFG_TEMPLATE's
# numCellsX=24 keeps mirrorGhostCells's two kernels to a SINGLE block --
# under-stresses any real cross-block race, since a single block's threads
# have much more implicit lockstep/scheduling correlation than independent
# blocks. This config forces numCellsX=320 > 256, i.e. >1 block, the
# regime where a genuine ordering bug is most likely to surface.
STRESS_CFG_TEMPLATE = """\
numCellsX = 320
numCellsY = 24
numCellsZ = 12
reynoldsNumber = 100.0
hyperViscousStart = 0
geometryShape = AbruptExpansion
lateralBC = SolidWall
outletBC = ZeroFirstDeriv
initialProfile = InletProfile
flowType = SteadyMarching
maxTimeSteps = 5
convergenceTol = 0
numPressureIter = 5
reportEveryN = 5
outputDir = {out}
runName = cuda_stress
"""


def check_determinism_multiblock(repeats=3):
    print(f"\n== Determinism (multi-block stress, numCellsX=320): "
          f"navsolver_cuda vs itself ({repeats} repeats) ==")
    with tempfile.TemporaryDirectory() as tmp:
        outs = []
        for run_idx in range(repeats):
            out_dir = Path(tmp) / f"stress_{run_idx}"
            cfg_path = Path(tmp) / f"stress_{run_idx}.cfg"
            cfg_path.write_text(STRESS_CFG_TEMPLATE.format(out=out_dir))
            run(NAVSOLVER_CUDA, cfg_path)
            outs.append(out_dir / "cuda_stress_t000005.vtk")
        ok = True
        for run_idx in range(1, repeats):
            ok = compare_velocity_fields(outs[0], outs[run_idx], DETERMINISM_TOL,
                                          f"run1 vs run{run_idx + 1}") and ok
        return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-build", action="store_true")
    args = ap.parse_args()

    if not args.skip_build:
        if not build():
            print("navsolver_cuda build failed / no CUDA toolchain -- nothing to check.", file=sys.stderr)
            sys.exit(0)
    elif not (NAVSOLVER.exists() and NAVSOLVER_CUDA.exists()):
        ap.error(f"{NAVSOLVER} / {NAVSOLVER_CUDA} not found; run without --skip-build first")

    if not NAVSOLVER_CUDA.exists():
        print("navsolver_cuda not built (no CUDA toolchain?) — nothing to check.", file=sys.stderr)
        sys.exit(0)

    ok = check_redblack_equivalence()
    ok = check_determinism() and ok
    ok = check_determinism_multiblock() and ok

    print()
    if ok:
        print("All CUDA-equivalence checks passed.")
        sys.exit(0)
    else:
        print("CUDA-equivalence checks FAILED.")
        sys.exit(1)


if __name__ == "__main__":
    main()
