#!/usr/bin/env python3
"""Physics correctness checks for NavSolver.

Two tiers:

1. Conservation/stability sanity checks ("blow-up detector"): run a few
   representative configs, parse their convergence CSV, and fail if
   ResidMax/DilMax ever go non-finite or exceed a generous ceiling. This is
   intentionally coarse — it catches crashes/divergence/NaN regressions in
   CI, it is NOT a substitute for deeper physics validation.

2. Analytical check (plane Poiseuille flow): geometryShape=Straight +
   lateralBC=Periodic + initialProfile=InletProfile seeds the exact
   closed-form solution u(y) = 6*yNorm*(1-yNorm)*Umax as the initial
   condition (see experiments/configs/poiseuille.cfg). A correct
   discretization should reproduce that solution almost exactly (residual
   near machine epsilon, not just "small") for the entire run, since it's
   already the steady state. This tests discretization self-consistency —
   whether the discrete operators return ~0 residual when evaluated AT the
   exact solution — which is a real, if narrow, verification technique. It
   does NOT test whether the solver converges to the right answer from an
   arbitrary starting point; that needs a grid-convergence study, which is
   future work (see README).

3. Red-black equivalence (only if navsolver_omp exists, built via
   `make openmp`): src/openmp/Physics.cpp's solvePressurePoisson uses
   red-black SOR instead of the serial solver's sequential Gauss-Seidel/SOR
   (needed for OpenMP parallel-safety, see docs/openmp-parallelization.md).
   They converge to the SAME fixed point via a DIFFERENT iteration path,
   so this compares the two solvers' final pressure fields at a large
   sweep count -- GAUGE-FIXED (each field's own mean subtracted first).
   That's not optional bookkeeping: pressure Poisson has a null space (a
   uniform additive offset never affects velocity, since only ∇p matters),
   and that offset turned out to be a slowly-converging mode under both
   solvers -- comparing raw values at any practically-reachable sweep count
   shows a real, if physically meaningless, gap. Comparing gauge-fixed
   fields is what actually tests whether the two solvers agree on the part
   that matters.

Usage:
    python3 scripts/validate.py                # both tiers
    python3 scripts/validate.py --skip-build
    make validate
"""

import argparse
import csv
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Where to look for the solver binaries. Defaults to the repo root (where
# `make` drops them), overridable via NAVSOLVER_BIN_DIR so a CMake build
# tree can be checked without copying binaries around -- that's how these
# scripts are registered as ctest tests (see CMakeLists.txt).
#
# Deliberately an env var rather than a CLI flag: validate_parallel.py and
# validate_cuda.py do `from validate import NAVSOLVER, ...`, which binds
# their own module-level names at import time. A flag parsed in main()
# could not rebind those; an env var read here, before those imports
# resolve, propagates to every caller.
BIN_DIR = Path(os.environ.get("NAVSOLVER_BIN_DIR", REPO_ROOT))
NAVSOLVER = BIN_DIR / "navsolver"
NAVSOLVER_OMP = BIN_DIR / "navsolver_omp"

# Blow-up detector: intentionally generous, just needs to catch crashes/NaN/divergence.
CONSERVATION_CEILING = 1e6

CONSERVATION_CONFIGS = {
    "abrupt_expansion": """\
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
runName = conservation_ae
""",
    "rounded_corner_periodic": """\
numCellsX = 60
numCellsY = 20
numCellsZ = 10
baseUnit = 8
reynoldsNumber = 1000.0
hyperViscousRe = 20.0
hyperViscousStart = 0
geometryShape = RoundedCorner
geometryType = Curved
lateralBC = Periodic
outletBC = ZeroFirstDeriv
initialProfile = PotentialFlow
flowType = SteadyMarching
maxTimeSteps = 20
convergenceTol = 0
numPressureIter = 5
reportEveryN = 20
outputDir = {out}
runName = conservation_rc
""",
}

POISEUILLE_CFG = REPO_ROOT / "experiments" / "configs" / "poiseuille.cfg"
POISEUILLE_RESID_TOL = 1e-6      # residual should stay near machine epsilon
POISEUILLE_PROFILE_TOL = 1e-4    # relative L2/Linf error of u(y) vs analytical

# One step, a large sweep count -- isolates one pressure-solve's convergence
# without confounding it with many steps' varying velocity/RHS. 3000 was
# empirically checked (see docs/openmp-parallelization.md) to bring the
# gauge-fixed gap to ~0.1%; the tolerance below has margin over that.
#
# geometryShape=Straight deliberately, not AbruptExpansion: AbruptExpansion's
# stepped geometry showed the SERIAL solver's own mean pressure still
# drifting (not converging) even at 20000 sweeps -- a pre-existing property
# of that geometry's discretization at sweep counts far beyond real usage
# (production configs use 5-20, never 3000+), unrelated to red-black. Testing
# algorithm equivalence needs a well-posed case; Straight's simple rectangular
# domain converges cleanly and isolates the actual thing being tested.
REDBLACK_CFG_TEMPLATE = """\
numCellsX = 48
numCellsY = 24
numCellsZ = 12
reynoldsNumber = 100.0
hyperViscousStart = 0
geometryShape = Straight
lateralBC = SolidWall
outletBC = ZeroFirstDeriv
initialProfile = InletProfile
flowType = SteadyMarching
maxTimeSteps = 1
convergenceTol = 0
numPressureIter = 3000
reportEveryN = 1
outputDir = {out}
runName = redblack_check
"""
REDBLACK_TOL = 1e-2  # relative L2, gauge-fixed (see module docstring tier 3)


def run_solver(cfg_path):
    result = subprocess.run([str(NAVSOLVER), str(cfg_path)],
                             capture_output=True, text=True, cwd=REPO_ROOT)
    if result.returncode != 0:
        raise RuntimeError(f"navsolver exited {result.returncode}:\n{result.stderr}\n{result.stdout}")
    return result.stdout


def parse_convergence_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def parse_vtk_velocity(path):
    lines = Path(path).read_text().splitlines()
    dims = None
    start = None
    for i, line in enumerate(lines):
        if line.startswith("DIMENSIONS"):
            dims = tuple(int(x) for x in line.split()[1:])
        if line.startswith("VECTORS Velocity"):
            start = i + 1
            break
    if dims is None or start is None:
        raise RuntimeError(f"could not parse {path}")
    nx, ny, nz = dims
    vel = [tuple(float(x) for x in line.split()) for line in lines[start:start + nx * ny * nz]]

    def at(i, j, k):
        return vel[(k * ny + j) * nx + i]

    return dims, at


def parse_vtk_scalar(path, field_name):
    """Generic SCALARS reader (e.g. field_name='Pressure') -- values follow
    the 'LOOKUP_TABLE default' line right after 'SCALARS <field_name> ...'."""
    lines = Path(path).read_text().splitlines()
    dims = None
    start = None
    for i, line in enumerate(lines):
        if line.startswith("DIMENSIONS"):
            dims = tuple(int(x) for x in line.split()[1:])
        if line.startswith(f"SCALARS {field_name}"):
            start = i + 2  # skip the SCALARS line and the LOOKUP_TABLE line
            break
    if dims is None or start is None:
        raise RuntimeError(f"could not parse {field_name} from {path}")
    nx, ny, nz = dims
    vals = [float(line) for line in lines[start:start + nx * ny * nz]]

    def at(i, j, k):
        return vals[(k * ny + j) * nx + i]

    return dims, vals, at


def build():
    print("Building (Release)...", file=sys.stderr)
    subprocess.run(["make", "clean"], cwd=REPO_ROOT, check=True,
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(["make", "all"], cwd=REPO_ROOT, check=True,
                    stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def build_openmp():
    """Best-effort: returns True if navsolver_omp was built, False if the
    toolchain isn't available (e.g. no OpenMP) -- tier 3 is skipped, not
    failed, in that case."""
    print("Building (OpenMP)...", file=sys.stderr)
    result = subprocess.run(["make", "openmp"], cwd=REPO_ROOT,
                             stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        print(f"  (skipping tier 3: make openmp failed: {result.stderr.strip()[-300:]})",
              file=sys.stderr)
        return False
    return NAVSOLVER_OMP.exists()


def check_conservation():
    print("== Conservation / stability sanity checks ==")
    ok = True
    with tempfile.TemporaryDirectory() as tmp:
        for name, template in CONSERVATION_CONFIGS.items():
            out_dir = Path(tmp) / name
            cfg_path = Path(tmp) / f"{name}.cfg"
            cfg_path.write_text(template.format(out=out_dir))
            run_solver(cfg_path)

            run_name = re.search(r"runName\s*=\s*(\S+)", template).group(1)
            csv_path = out_dir / f"{run_name}_convergence.csv"
            rows = parse_convergence_csv(csv_path)
            if not rows:
                print(f"  FAIL {name}: no convergence rows written")
                ok = False
                continue

            bad = []
            for row in rows:
                for key in ("ResidMax", "DilMax"):
                    val = float(row[key])
                    if not (val == val) or val in (float("inf"), float("-inf")):  # NaN check
                        bad.append(f"step {row['step']}: {key}={row[key]} (non-finite)")
                    elif abs(val) > CONSERVATION_CEILING:
                        bad.append(f"step {row['step']}: {key}={val:.3e} exceeds ceiling {CONSERVATION_CEILING:.0e}")
            if bad:
                print(f"  FAIL {name}:")
                for b in bad[:5]:
                    print(f"    {b}")
                ok = False
            else:
                final = rows[-1]
                print(f"  PASS {name}: {len(rows)} steps, "
                      f"final ResidMax={float(final['ResidMax']):.3e} DilMax={float(final['DilMax']):.3e}")
    return ok


def check_poiseuille():
    print("\n== Analytical check: plane Poiseuille flow ==")
    if not POISEUILLE_CFG.exists():
        print(f"  FAIL: {POISEUILLE_CFG} not found")
        return False

    run_solver(POISEUILLE_CFG)

    cfg_text = POISEUILLE_CFG.read_text()
    out_dir = REPO_ROOT / re.search(r"outputDir\s*=\s*(\S+)", cfg_text).group(1)
    run_name = re.search(r"runName\s*=\s*(\S+)", cfg_text).group(1)
    max_steps = int(re.search(r"maxTimeSteps\s*=\s*(\S+)", cfg_text).group(1))

    csv_path = out_dir / f"{run_name}_convergence.csv"
    rows = parse_convergence_csv(csv_path)
    ok = True

    max_resid = max(abs(float(r["ResidMax"])) for r in rows)
    max_dil = max(abs(float(r["DilMax"])) for r in rows)
    if max_resid > POISEUILLE_RESID_TOL or max_dil > POISEUILLE_RESID_TOL:
        print(f"  FAIL: residual grew away from the exact solution "
              f"(max ResidMax={max_resid:.3e}, max DilMax={max_dil:.3e}, tol={POISEUILLE_RESID_TOL:.0e})")
        ok = False
    else:
        print(f"  PASS: residual stayed near machine epsilon "
              f"(max ResidMax={max_resid:.3e}, max DilMax={max_dil:.3e})")

    vtk_path = out_dir / f"{run_name}_t{max_steps:06d}.vtk"
    (nx, ny, nz), at = parse_vtk_velocity(vtk_path)
    i_mid, k_mid = nx // 2, nz // 2

    sq_err_sum, max_err, max_u = 0.0, 0.0, 0.0
    for j in range(ny):
        y_norm = j / (ny - 1)
        u_exact = 6.0 * y_norm * (1.0 - y_norm)
        u_num = at(i_mid, j, k_mid)[0]
        err = abs(u_num - u_exact)
        sq_err_sum += err * err
        max_err = max(max_err, err)
        max_u = max(max_u, u_exact)

    l2_rel = (sq_err_sum / ny) ** 0.5 / max_u
    linf_rel = max_err / max_u
    if l2_rel > POISEUILLE_PROFILE_TOL or linf_rel > POISEUILLE_PROFILE_TOL:
        print(f"  FAIL: u(y) profile deviates from analytical solution "
              f"(L2_rel={l2_rel:.2e}, Linf_rel={linf_rel:.2e}, tol={POISEUILLE_PROFILE_TOL:.0e})")
        ok = False
    else:
        print(f"  PASS: u(y) matches analytical parabola "
              f"(L2_rel={l2_rel:.2e}, Linf_rel={linf_rel:.2e})")

    return ok


def check_redblack_equivalence():
    print("\n== Red-black equivalence: navsolver vs navsolver_omp (gauge-fixed) ==")
    with tempfile.TemporaryDirectory() as tmp:
        cfg_path = Path(tmp) / "redblack.cfg"

        out_serial = Path(tmp) / "serial"
        cfg_path.write_text(REDBLACK_CFG_TEMPLATE.format(out=out_serial))
        subprocess.run([str(NAVSOLVER), str(cfg_path)], capture_output=True, text=True,
                        cwd=REPO_ROOT, check=True)

        out_omp = Path(tmp) / "omp"
        cfg_path.write_text(REDBLACK_CFG_TEMPLATE.format(out=out_omp))
        subprocess.run([str(NAVSOLVER_OMP), str(cfg_path)], capture_output=True, text=True,
                        cwd=REPO_ROOT, check=True)

        vtk_serial = out_serial / "redblack_check_t000001.vtk"
        vtk_omp = out_omp / "redblack_check_t000001.vtk"
        (nx, ny, nz), press_serial, _ = parse_vtk_scalar(vtk_serial, "Pressure")
        (nx2, ny2, nz2), press_omp, _ = parse_vtk_scalar(vtk_omp, "Pressure")
        if (nx, ny, nz) != (nx2, ny2, nz2):
            print("  FAIL: grid size mismatch between serial and openmp output")
            return False

        mean_serial = sum(press_serial) / len(press_serial)
        mean_omp = sum(press_omp) / len(press_omp)
        diffs_sq = 0.0
        serial_sq = 0.0
        for ps, po in zip(press_serial, press_omp):
            d = (ps - mean_serial) - (po - mean_omp)
            diffs_sq += d * d
            serial_sq += (ps - mean_serial) ** 2
        l2_rel = (diffs_sq / max(serial_sq, 1e-30)) ** 0.5

        if l2_rel > REDBLACK_TOL:
            print(f"  FAIL: gauge-fixed pressure fields disagree "
                  f"(L2_rel={l2_rel:.3e}, tol={REDBLACK_TOL:.0e})")
            return False
        print(f"  PASS: gauge-fixed pressure fields agree (L2_rel={l2_rel:.3e})")
        return True


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-build", action="store_true")
    args = ap.parse_args()

    if not args.skip_build:
        build()
    elif not NAVSOLVER.exists():
        ap.error(f"{NAVSOLVER} not found; run without --skip-build first")

    ok = check_conservation()
    ok = check_poiseuille() and ok

    if args.skip_build:
        omp_available = NAVSOLVER_OMP.exists()
    else:
        omp_available = build_openmp()
    if omp_available:
        ok = check_redblack_equivalence() and ok
    else:
        print("\n== Red-black equivalence: SKIPPED (navsolver_omp not built) ==")

    print()
    if ok:
        print("All physics checks passed.")
        sys.exit(0)
    else:
        print("Physics checks FAILED.")
        sys.exit(1)


if __name__ == "__main__":
    main()
