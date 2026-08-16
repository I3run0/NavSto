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

3. Red-black equivalence (only if navsolver_omp exists, i.e. the
   navsolver_omp target built): src/backends/openmp/Physics.cpp's solvePressurePoisson uses
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
    cmake --build build --target validate_physics
"""

import argparse
import re
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import (REPO_ROOT, NAVSOLVER, NAVSOLVER_OMP,  # noqa: E402
                     build, compare_pressure_gauge_fixed, parse_convergence_csv,
                     parse_vtk_velocity, run)

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


def check_conservation():
    print("== Conservation / stability sanity checks ==")
    ok = True
    with tempfile.TemporaryDirectory() as tmp:
        for name, template in CONSERVATION_CONFIGS.items():
            out_dir = Path(tmp) / name
            cfg_path = Path(tmp) / f"{name}.cfg"
            cfg_path.write_text(template.format(out=out_dir))
            run(NAVSOLVER, cfg_path)

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

    run(NAVSOLVER, POISEUILLE_CFG)

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
        vtks = []
        for label, binary in (("serial", NAVSOLVER), ("omp", NAVSOLVER_OMP)):
            out_dir = Path(tmp) / label
            cfg_path.write_text(REDBLACK_CFG_TEMPLATE.format(out=out_dir))
            run(binary, cfg_path)
            vtks.append(out_dir / "redblack_check_t000001.vtk")
        return compare_pressure_gauge_fixed(vtks[0], vtks[1], REDBLACK_TOL,
                                            "serial vs omp")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-build", action="store_true")
    args = ap.parse_args()

    if not args.skip_build:
        build("navsolver", required=("navsolver",))
    elif not NAVSOLVER.exists():
        ap.error(f"{NAVSOLVER} not found; run without --skip-build first")

    ok = check_conservation()
    ok = check_poiseuille() and ok

    if not args.skip_build:
        build("navsolver_omp")
    omp_available = NAVSOLVER_OMP.exists()
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
