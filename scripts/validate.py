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

Usage:
    python3 scripts/validate.py                # both tiers
    python3 scripts/validate.py --skip-build
    make validate
"""

import argparse
import csv
import re
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
NAVSOLVER = REPO_ROOT / "navsolver"

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


def build():
    print("Building (Release)...", file=sys.stderr)
    subprocess.run(["make", "clean"], cwd=REPO_ROOT, check=True,
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(["make", "all"], cwd=REPO_ROOT, check=True,
                    stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


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

    print()
    if ok:
        print("All physics checks passed.")
        sys.exit(0)
    else:
        print("Physics checks FAILED.")
        sys.exit(1)


if __name__ == "__main__":
    main()
