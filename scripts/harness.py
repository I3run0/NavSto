#!/usr/bin/env python3
"""Shared build/run/parse plumbing for the validation and benchmark scripts.

Everything here was previously copy-pasted: build() lived in five scripts (one
of them still driving the deleted Makefile), and the VTK readers and field
comparison in two or three each.

BIN_DIR is an env var, not a CLI flag, because importers bind these names at
import time — a flag parsed in main() could not rebind them.
"""

import csv
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BIN_DIR = Path(os.environ.get("NAVSOLVER_BIN_DIR", REPO_ROOT / "build"))

NAVSOLVER = BIN_DIR / "navsolver"
NAVSOLVER_OMP = BIN_DIR / "navsolver_omp"
NAVSOLVER_CUDA = BIN_DIR / "navsolver_cuda"


# ── Build ────────────────────────────────────────────────────────────────────

def configure():
    """Configure the Release build tree BIN_DIR points at. Idempotent."""
    subprocess.run(["cmake", "-S", str(REPO_ROOT), "-B", str(BIN_DIR),
                    "-DCMAKE_BUILD_TYPE=Release"], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def build_target(target):
    """Best-effort build of one target; True if it succeeded. Callers decide
    whether a failure is fatal or just skips a tier."""
    return subprocess.run(["cmake", "--build", str(BIN_DIR), "--target", target, "-j"],
                          stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                          text=True).returncode == 0


def build(*targets, required=()):
    """Configure, then build each target. Raises if a target in `required`
    fails; returns the set of targets that built."""
    print(f"Building (Release): {', '.join(targets)}...", file=sys.stderr)
    configure()
    built = set()
    for target in targets:
        if build_target(target):
            built.add(target)
        elif target in required:
            raise RuntimeError(f"{target} build failed")
        else:
            print(f"  (skipping {target}: build failed or unavailable)", file=sys.stderr)
    return built


# ── Run ──────────────────────────────────────────────────────────────────────

def run(binary, cfg_path, threads=None):
    """Run a solver binary on a config. Raises on non-zero exit."""
    env = dict(os.environ)
    if threads is not None:
        env["OMP_NUM_THREADS"] = str(threads)
    result = subprocess.run([str(binary), str(cfg_path)], capture_output=True,
                            text=True, cwd=REPO_ROOT, env=env)
    if result.returncode != 0:
        raise RuntimeError(f"{binary} exited {result.returncode}:\n"
                           f"{result.stderr}\n{result.stdout}")
    return result.stdout


# ── Parse ────────────────────────────────────────────────────────────────────

def parse_convergence_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _dims_and_start(lines, marker, skip):
    dims = None
    for i, line in enumerate(lines):
        if line.startswith("DIMENSIONS"):
            dims = tuple(int(x) for x in line.split()[1:])
        if line.startswith(marker):
            return dims, i + skip
    return dims, None


def parse_vtk_velocity(path):
    lines = Path(path).read_text().splitlines()
    dims, start = _dims_and_start(lines, "VECTORS Velocity", 1)
    if dims is None or start is None:
        raise RuntimeError(f"could not parse {path}")
    nx, ny, nz = dims
    vel = [tuple(float(x) for x in line.split())
           for line in lines[start:start + nx * ny * nz]]

    def at(i, j, k):
        return vel[(k * ny + j) * nx + i]

    return dims, at


def parse_vtk_scalar(path, field_name):
    """Generic SCALARS reader; values follow the LOOKUP_TABLE line."""
    lines = Path(path).read_text().splitlines()
    dims, start = _dims_and_start(lines, f"SCALARS {field_name}", 2)
    if dims is None or start is None:
        raise RuntimeError(f"could not parse {field_name} from {path}")
    nx, ny, nz = dims
    vals = [float(line) for line in lines[start:start + nx * ny * nz]]

    def at(i, j, k):
        return vals[(k * ny + j) * nx + i]

    return dims, vals, at


# ── Compare ──────────────────────────────────────────────────────────────────

def compare_velocity_fields(vtk_a, vtk_b, tol, label):
    dims_a, at_a = parse_vtk_velocity(vtk_a)
    dims_b, at_b = parse_vtk_velocity(vtk_b)
    if dims_a != dims_b:
        print(f"  FAIL {label}: grid size mismatch {dims_a} vs {dims_b}")
        return False

    nx, ny, nz = dims_a
    sq_err, sq_ref, max_rel = 0.0, 0.0, 0.0
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                ua, ub = at_a(i, j, k), at_b(i, j, k)
                for c in range(3):
                    d = ua[c] - ub[c]
                    sq_err += d * d
                    sq_ref += ua[c] * ua[c]
                    max_rel = max(max_rel, abs(d) / max(abs(ua[c]), 1e-12))
    l2_rel = (sq_err / max(sq_ref, 1e-30)) ** 0.5

    if l2_rel > tol:
        print(f"  FAIL {label}: L2_rel={l2_rel:.3e} "
              f"(tol={tol:.0e}, max pointwise rel={max_rel:.3e})")
        return False
    print(f"  PASS {label}: L2_rel={l2_rel:.3e}")
    return True


def compare_pressure_gauge_fixed(vtk_a, vtk_b, tol, label):
    """Relative L2 between two pressure fields with each field's own mean
    removed. The pressure Poisson problem has a null space -- a uniform offset
    never affects velocity -- and that offset is a slowly-converging mode, so
    raw values disagree at any reachable sweep count even when the solvers do
    agree on the part that matters."""
    dims_a, press_a, _ = parse_vtk_scalar(vtk_a, "Pressure")
    dims_b, press_b, _ = parse_vtk_scalar(vtk_b, "Pressure")
    if dims_a != dims_b:
        print(f"  FAIL {label}: grid size mismatch {dims_a} vs {dims_b}")
        return False

    mean_a = sum(press_a) / len(press_a)
    mean_b = sum(press_b) / len(press_b)
    diffs_sq = ref_sq = 0.0
    for pa, pb in zip(press_a, press_b):
        d = (pa - mean_a) - (pb - mean_b)
        diffs_sq += d * d
        ref_sq += (pa - mean_a) ** 2
    l2_rel = (diffs_sq / max(ref_sq, 1e-30)) ** 0.5

    if l2_rel > tol:
        print(f"  FAIL {label}: gauge-fixed pressure fields disagree "
              f"(L2_rel={l2_rel:.3e}, tol={tol:.0e})")
        return False
    print(f"  PASS {label}: gauge-fixed pressure fields agree (L2_rel={l2_rel:.3e})")
    return True
