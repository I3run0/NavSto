#!/usr/bin/env python3
"""One entrypoint that emits the signals an optimization loop needs.

Four signals, and a loop needs all of them:

  gate         did it stay correct, and by how much margin
  kernel       per-operator time in isolation -- the INNER loop, ~1s
  reward       whole-program wall clock, I/O excluded
  attribution  which operator owns the step time
  headroom     how far from the roofline (slow; run it rarely)

Everything lands in one JSON envelope on stdout, carrying the commit it came
from. Human progress goes to stderr, so `measure.py --tier full > run.json`
works while you watch it.

With --baseline, each metric is compared against a previous envelope and gets a
verdict of improved / regressed / within_noise. A delta only counts if it beats
NOISE_K x the measured run-to-run spread -- without that an optimizer accepts
thermal drift as a win.

Usage:
    python3 scripts/measure.py --tier kernel
    python3 scripts/measure.py --tier full --save
    python3 scripts/measure.py --tier reward --baseline experiments/results/baselines/<f>.json
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import (BIN_DIR, KBENCH, KBENCH_OMP, NOISE_K,  # noqa: E402
                     REPO_ROOT, build, emit_json, envelope, kbench,
                     profile_kernels, stats, verdict)

BASELINE_DIR = REPO_ROOT / "experiments" / "results" / "baselines"

TIERS = ("gate", "kernel", "reward", "attribution", "headroom", "full")

PROFILE_CFG = """\
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
maxTimeSteps = 50
convergenceTol = 0
numPressureIter = 5
reportEveryN = 0
outputDir = {out}
runName = measure
"""


def log(*a):
    print(*a, file=sys.stderr)


def _script(name, *args):
    """Run one of the sibling scripts with --json and return its parsed output."""
    cmd = [sys.executable, str(REPO_ROOT / "scripts" / name), "--json", *args]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT)
    payload = json.loads(r.stdout) if r.stdout.strip() else {"status": "error"}
    payload["exit_code"] = r.returncode
    return payload


# ── Tiers ────────────────────────────────────────────────────────────────────

def tier_gate(args):
    log("[gate] correctness validators...")
    out = {}
    for name, script in (("physics", "validate.py"),
                         ("parallel", "validate_parallel.py"),
                         ("cuda", "validate_cuda.py")):
        out[name] = _script(script, "--skip-build")
    out["status"] = "pass" if all(v.get("status") == "pass" for v in out.values()
                                  if isinstance(v, dict)) else "fail"
    return out


def tier_kernel(args):
    """Repeat the whole measurement, and report the spread OF THE MINIMA.

    The noise floor has to describe the statistic being compared. kbench's own
    rel_spread is the within-run iteration spread, which is dominated by
    warm-up outliers -- using it gave thresholds of +/-60% to +/-144%, so no
    real improvement could ever clear the bar. Run-to-run spread of min_ms is
    ~2-3%, which is the number a verdict should key off.
    """
    binary = KBENCH_OMP if args.backend == "openmp" else KBENCH
    log(f"[kernel] {binary.name} {args.grid} x{args.iters}, {args.repeats} runs...")

    runs = [kbench(binary, kernel=args.kernel, grid=args.grid, iters=args.iters,
                   threads=args.threads) for _ in range(args.repeats)]

    merged = []
    for entry in runs[0]["kernels"]:
        name = entry["kernel"]
        mins = [k["min_ms"] for r in runs for k in r["kernels"] if k["kernel"] == name]
        st = stats(mins)
        merged.append({
            "kernel": name,
            "min_ms": st["min"],
            "median_ms": st["median"],
            # run-to-run spread of the minima: the actual noise floor
            "rel_spread": st["rel_spread"],
            "within_run_spread": entry["rel_spread"],
            "ns_per_active_cell": entry["ns_per_active_cell"] * st["min"] / entry["min_ms"]
                                  if entry["min_ms"] else 0.0,
            "runs": len(mins),
        })
        log(f"    {name:<26} {st['min']:8.3f} ms  noise={st['rel_spread']:.1%}")

    return {**runs[0], "kernels": merged, "repeats": args.repeats}


def tier_reward(args):
    log("[reward] whole-program benchmark (I/O excluded)...")
    extra = ["--skip-build", "--sizes", args.sizes, "--steps", str(args.steps),
             "--repeats", str(args.repeats), "--binary",
             "navsolver_omp" if args.backend == "openmp" else "navsolver"]
    if args.threads:
        extra += ["--threads", str(args.threads)]
    data = _script("benchmark.py", *extra)
    # rel_spread over the repeats IS the noise floor: same binary, same input.
    for r in data.get("runs", []):
        log(f"    {r['size']:<8} {r['min_seconds']:.4f}s  "
            f"{r['ns_per_active_cell_step']:.1f} ns/cell-step  "
            f"noise={r['rel_spread']:.1%}")
    return data


def tier_attribution(args):
    log("[attribution] per-kernel shares from a profiling build...")
    kernels = profile_kernels(PROFILE_CFG, run_name="measure",
                              target="navsolver_omp" if args.backend == "openmp" else "navsolver",
                              threads=args.threads)
    for name, k in sorted(kernels.items(), key=lambda kv: -kv[1]["share_percent"]):
        log(f"    {name:<26} {k['share_percent']:5.1f}%  {k['mean_ms']:8.3f} ms/call")
    return {"kernels": kernels}


def tier_headroom(args):
    log("[headroom] roofline (slow)...")
    return _script("roofline.py")


# ── Baseline comparison ──────────────────────────────────────────────────────

def compare(baseline, candidate):
    """Verdicts per kernel and per benchmark size, lower-is-better."""
    out = {"noise_k": NOISE_K, "kernels": {}, "reward": {}}

    b_k = {k["kernel"]: k for k in baseline.get("kernel", {}).get("kernels", [])}
    c_k = {k["kernel"]: k for k in candidate.get("kernel", {}).get("kernels", [])}
    for name, c in c_k.items():
        if name not in b_k:
            continue
        # Use the noisier of the two spreads: a verdict is only as trustworthy
        # as the shakier measurement behind it.
        noise = max(c["rel_spread"], b_k[name]["rel_spread"])
        out["kernels"][name] = verdict(b_k[name]["min_ms"], c["min_ms"], noise)

    b_r = {r["size"]: r for r in baseline.get("reward", {}).get("runs", [])}
    c_r = {r["size"]: r for r in candidate.get("reward", {}).get("runs", [])}
    for size, c in c_r.items():
        if size not in b_r:
            continue
        noise = max(c["rel_spread"], b_r[size]["rel_spread"])
        out["reward"][size] = verdict(b_r[size]["min_seconds"], c["min_seconds"], noise)

    verdicts = [v["verdict"] for v in
                list(out["kernels"].values()) + list(out["reward"].values())]
    out["summary"] = ("regressed" if "regressed" in verdicts
                      else "improved" if "improved" in verdicts
                      else "within_noise")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tier", default="kernel", choices=TIERS)
    ap.add_argument("--backend", default="serial", choices=("serial", "openmp"))
    ap.add_argument("--kernel", default="all", help="kbench --kernel (default: all)")
    ap.add_argument("--grid", default="96x48x24")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--sizes", default="small,medium")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--repeats", type=int, default=5,
                    help="also the noise-floor sample count (default: 5)")
    ap.add_argument("--baseline", type=Path, default=None)
    ap.add_argument("--save", action="store_true",
                    help=f"also write the envelope under {BASELINE_DIR}")
    ap.add_argument("--skip-build", action="store_true")
    args = ap.parse_args()

    if not args.skip_build:
        targets = ["navsolver", "navsolver_kbench"]
        if args.backend == "openmp":
            targets += ["navsolver_omp", "navsolver_kbench_omp"]
        build(*targets, required=("navsolver",))

    wanted = TIERS[:-1] if args.tier == "full" else (args.tier,)
    result = envelope(tool="measure", tier=args.tier, backend=args.backend)

    runners = {"gate": tier_gate, "kernel": tier_kernel, "reward": tier_reward,
               "attribution": tier_attribution, "headroom": tier_headroom}
    for t in wanted:
        try:
            result[t] = runners[t](args)
        except Exception as e:                      # a dead tier must not hide the rest
            log(f"  [{t}] FAILED: {e}")
            result[t] = {"status": "error", "error": str(e)}

    if args.baseline:
        base = json.loads(Path(args.baseline).read_text())
        result["comparison"] = compare(base, result)
        log(f"\n  verdict: {result['comparison']['summary']} "
            f"(vs {base.get('git_commit', '?')})")

    if args.save:
        BASELINE_DIR.mkdir(parents=True, exist_ok=True)
        path = BASELINE_DIR / f"{result['timestamp_utc']}_{result['git_commit']}.json"
        path.write_text(json.dumps(result, indent=2, default=str))
        log(f"  saved baseline -> {path}")

    emit_json(result)

    gate = result.get("gate")
    if gate and gate.get("status") == "fail":
        sys.exit(1)
    cmp_ = result.get("comparison")
    sys.exit(2 if cmp_ and cmp_["summary"] == "regressed" else 0)


if __name__ == "__main__":
    main()
