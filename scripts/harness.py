#!/usr/bin/env python3
"""Shared build/run/parse plumbing for the validation and benchmark scripts.

Everything here was previously copy-pasted: build() lived in five scripts (one
of them still driving the deleted Makefile), and the VTK readers and field
comparison in two or three each.

BIN_DIR is an env var, not a CLI flag, because importers bind these names at
import time — a flag parsed in main() could not rebind them.
"""

import csv
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BIN_DIR = Path(os.environ.get("NAVSOLVER_BIN_DIR", REPO_ROOT / "build"))

NAVSOLVER = BIN_DIR / "navsolver"
NAVSOLVER_OMP = BIN_DIR / "navsolver_omp"
NAVSOLVER_CUDA = BIN_DIR / "navsolver_cuda"
KBENCH = BIN_DIR / "navsolver_kbench"
KBENCH_OMP = BIN_DIR / "navsolver_kbench_omp"

# Pinning, applied to every timed run and recorded in the envelope. Unpinned
# OpenMP threads migrate between cores mid-run, which is a large part of the
# run-to-run swing that otherwise looks like a real speedup.
BENCH_ENV = {"OMP_PROC_BIND": "close", "OMP_PLACES": "cores"}

# A candidate must beat the baseline by more than NOISE_K x the measured noise
# floor to count. Anything less is drift, and an optimizer that accepts drift
# will happily "improve" code that did not change.
NOISE_K = 2.0


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

def run(binary, cfg_path, threads=None, pinned=False):
    """Run a solver binary on a config. Raises on non-zero exit."""
    env = dict(os.environ)
    if threads is not None:
        env["OMP_NUM_THREADS"] = str(threads)
    if pinned:
        env.update(BENCH_ENV)
    result = subprocess.run([str(binary), str(cfg_path)], capture_output=True,
                            text=True, cwd=REPO_ROOT, env=env)
    if result.returncode != 0:
        raise RuntimeError(f"{binary} exited {result.returncode}:\n"
                           f"{result.stderr}\n{result.stdout}")
    return result.stdout


def kbench(binary, kernel="all", grid="96x48x24", iters=20, warmup=3,
           pressure_iter=5, threads=None):
    """Run navsolver_kbench and return its parsed JSON.

    This is the inner loop of an optimization cycle -- one kernel on a
    realistic state in about a second, versus tens of seconds for a whole
    simulation. Absolute numbers run ~15% under the whole-program profiler
    because a tight loop keeps caches hot; the offset is consistent across
    kernels, so relative comparisons are what this is for.
    """
    env = dict(os.environ)
    env.update(BENCH_ENV)
    if threads is not None:
        env["OMP_NUM_THREADS"] = str(threads)
    cmd = [str(binary), "--kernel", kernel, "--grid", grid,
           "--iters", str(iters), "--warmup", str(warmup),
           "--pressure-iter", str(pressure_iter)]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT, env=env)
    if r.returncode != 0:
        raise RuntimeError(f"{binary} exited {r.returncode}:\n{r.stderr}")
    return json.loads(r.stdout)


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


# ── Check records ────────────────────────────────────────────────────────────
#
# A bare exit code tells an optimizer "something broke" and nothing else. These
# records carry the metric and the tolerance, so it can see how much margin a
# change ate even when everything still passes.

CHECKS = []


def record(check, passed, metric=None, tolerance=None, detail=None):
    CHECKS.append({
        "check": check,
        "status": "pass" if passed else "fail",
        "metric": metric,
        "tolerance": tolerance,
        "margin": (tolerance / metric) if (metric and tolerance and metric > 0) else None,
        "detail": detail,
    })
    return passed


def checks_envelope(tool, **extra):
    ok = all(c["status"] == "pass" for c in CHECKS)
    return envelope(tool=tool, status="pass" if ok else "fail",
                    checks=list(CHECKS), **extra)


# ── Compare ──────────────────────────────────────────────────────────────────

def compare_velocity_fields(vtk_a, vtk_b, tol, label):
    dims_a, at_a = parse_vtk_velocity(vtk_a)
    dims_b, at_b = parse_vtk_velocity(vtk_b)
    if dims_a != dims_b:
        print(f"  FAIL {label}: grid size mismatch {dims_a} vs {dims_b}")
        return record(label, False, detail=f"grid mismatch {dims_a} vs {dims_b}")

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

    ok = l2_rel <= tol
    if ok:
        print(f"  PASS {label}: L2_rel={l2_rel:.3e}")
    else:
        print(f"  FAIL {label}: L2_rel={l2_rel:.3e} "
              f"(tol={tol:.0e}, max pointwise rel={max_rel:.3e})")
    return record(label, ok, metric=l2_rel, tolerance=tol,
                  detail=f"max pointwise rel={max_rel:.3e}")


def parse_kernel_csv(path):
    """<runName>_kernels.csv from a -DNAVSOLVER_PROFILE=ON build."""
    rows = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            rows[r["kernel"]] = {
                "calls": int(r["calls"]),
                "total_seconds": float(r["total_seconds"]),
                "mean_ms": float(r["mean_ms"]),
                "share_percent": float(r["share_percent"]),
            }
    return rows


def profile_kernels(cfg_text, run_name, target="navsolver", threads=None,
                    build_dir=None):
    """Build a profiling tree, run one config, return the per-kernel table.

    Its own build dir: -DNAVSOLVER_PROFILE=ON changes codegen, and the profiled
    TOTAL is not comparable to a normal build (the CUDA timer synchronises per
    kernel). The per-kernel SHARES are the usable part.
    """
    import tempfile
    prof_dir = Path(build_dir) if build_dir else (BIN_DIR.parent / "build-profile-auto")
    subprocess.run(["cmake", "-S", str(REPO_ROOT), "-B", str(prof_dir),
                    "-DCMAKE_BUILD_TYPE=Release", "-DNAVSOLVER_PROFILE=ON"],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    subprocess.run(["cmake", "--build", str(prof_dir), "--target", target, "-j"],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp) / "out"
        cfg = Path(tmp) / "profile.cfg"
        cfg.write_text(cfg_text.format(out=out_dir))
        run(prof_dir / target, cfg, threads=threads, pinned=True)
        return parse_kernel_csv(out_dir / f"{run_name}_kernels.csv")


# ── Statistics ───────────────────────────────────────────────────────────────

def stats(samples):
    """min/median/mean/stddev plus the relative spread a verdict keys off."""
    s = sorted(samples)
    mean = statistics.fmean(s)
    stdev = statistics.stdev(s) if len(s) > 1 else 0.0
    return {
        "min": s[0], "max": s[-1], "median": statistics.median(s),
        "mean": mean, "stddev": stdev,
        "rel_spread": (stdev / mean) if mean else 0.0,
        "samples": len(s),
    }


def noise_floor(measure_fn, repeats=5):
    """Relative spread of the SAME binary measured repeatedly.

    This is the number that decides whether a delta is real. Without it an
    optimizer cannot distinguish a 3% win from thermal drift -- on this machine
    an earlier sweep produced 9.89s at 1 thread and 11.29s at 4, which was pure
    noise.
    """
    return stats([measure_fn() for _ in range(repeats)])


def verdict(baseline, candidate, noise_rel, k=NOISE_K):
    """improved / regressed / within_noise, for lower-is-better metrics."""
    if baseline <= 0:
        return {"verdict": "unknown", "speedup": None}
    speedup = baseline / candidate if candidate > 0 else None
    rel_delta = (baseline - candidate) / baseline
    threshold = k * noise_rel
    if rel_delta > threshold:
        v = "improved"
    elif rel_delta < -threshold:
        v = "regressed"
    else:
        v = "within_noise"
    return {
        "verdict": v, "speedup": speedup, "rel_delta": rel_delta,
        "noise_rel": noise_rel, "threshold": threshold,
    }


# ── Provenance / output ──────────────────────────────────────────────────────

def git_commit():
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT,
                             capture_output=True, text=True, check=True)
        commit = out.stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=REPO_ROOT,
                               capture_output=True, text=True, check=True).stdout.strip()
        return commit + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def cpu_model():
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except FileNotFoundError:
        pass
    return platform.processor() or platform.machine()


def envelope(**extra):
    """Common provenance wrapper. A measurement without the commit it came from
    is not comparable to anything."""
    env = {
        "git_commit": git_commit(),
        "cpu": cpu_model(),
        "cores": os.cpu_count(),
        "timestamp_utc": time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()),
        "pinning": dict(BENCH_ENV),
    }
    env.update(extra)
    return env


def emit_json(obj):
    """JSON on stdout, so human tables can keep using stderr in the same run."""
    json.dump(obj, sys.stdout, indent=2, default=str)
    sys.stdout.write("\n")


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
        return record(label, False, detail=f"grid mismatch {dims_a} vs {dims_b}")

    mean_a = sum(press_a) / len(press_a)
    mean_b = sum(press_b) / len(press_b)
    diffs_sq = ref_sq = 0.0
    for pa, pb in zip(press_a, press_b):
        d = (pa - mean_a) - (pb - mean_b)
        diffs_sq += d * d
        ref_sq += (pa - mean_a) ** 2
    l2_rel = (diffs_sq / max(ref_sq, 1e-30)) ** 0.5

    ok = l2_rel <= tol
    if ok:
        print(f"  PASS {label}: gauge-fixed pressure fields agree (L2_rel={l2_rel:.3e})")
    else:
        print(f"  FAIL {label}: gauge-fixed pressure fields disagree "
              f"(L2_rel={l2_rel:.3e}, tol={tol:.0e})")
    return record(label, ok, metric=l2_rel, tolerance=tol,
                  detail="gauge-fixed pressure L2")
