#!/usr/bin/env python3
"""Interleaved paired A/B between two kbench binaries.

measure.py compares a candidate against a saved baseline envelope, which is
fine for a large win and wrong for a small one: the two runs happen minutes
apart, and this machine drifts by more than the within-session noise floor in
that time. Here both arms run in the same session, alternating which goes
first, so drift lands on both equally and the statistic is the median of the
paired ratios rather than a difference of two independent minima.

Read the win count, not just the median: 7/7 pairs at 1.02x is a result,
4/7 pairs at 1.05x is not.

CAVEAT, and it is not a small one: two binaries differ in code layout as well
as in the change, and function alignment alone moved an untouched kernel by 4%
while measuring the change that prompted this script. A kernel the change
cannot reach that still moves consistently is evidence of layout, not of the
change -- confirm such a case by putting both paths in ONE binary behind a
runtime switch and running this against itself.

Usage:
    python3 scripts/paired.py --base build/kbench_head --cand build/navsolver_kbench
    python3 scripts/paired.py --base-ref HEAD --pairs 7 --grid 240x120x60
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import BENCH_ENV, KBENCH, REPO_ROOT  # noqa: E402


def build_ref(ref, dest):
    """Build kbench from `ref` in a throwaway worktree, so the working tree is
    never stashed -- a crashed script must not leave the user's edits parked."""
    with tempfile.TemporaryDirectory() as tmp:
        tree = Path(tmp) / "tree"
        subprocess.run(["git", "worktree", "add", "--detach", str(tree), ref],
                       cwd=REPO_ROOT, check=True, capture_output=True)
        try:
            build = tree / "build"
            subprocess.run(["cmake", "-S", str(tree), "-B", str(build),
                            "-DCMAKE_BUILD_TYPE=Release"], check=True, capture_output=True)
            subprocess.run(["cmake", "--build", str(build), "--target",
                            "navsolver_kbench", "-j"], check=True, capture_output=True)
            dest.write_bytes((build / "navsolver_kbench").read_bytes())
            dest.chmod(0o755)
        finally:
            subprocess.run(["git", "worktree", "remove", "--force", str(tree)],
                           cwd=REPO_ROOT, check=False, capture_output=True)
    return dest


def run(binary, grid, iters, threads):
    env = {**os.environ, **BENCH_ENV}
    if threads is not None:
        env["OMP_NUM_THREADS"] = str(threads)
    r = subprocess.run([str(binary), "--grid", grid, "--iters", str(iters)],
                       capture_output=True, text=True, env=env, check=True)
    return {k["kernel"]: k["min_ms"] for k in json.loads(r.stdout)["kernels"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", type=Path, help="baseline kbench binary")
    ap.add_argument("--base-ref", help="build the baseline from this git ref instead")
    ap.add_argument("--cand", type=Path, default=KBENCH)
    ap.add_argument("--grid", default="96x48x24")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--pairs", type=int, default=7)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        base = args.base
        if args.base_ref:
            base = build_ref(args.base_ref, Path(tmp) / "kbench_base")
        if not base:
            ap.error("pass --base or --base-ref")

        ratios, order = {}, []
        for p in range(args.pairs):
            first = base if p % 2 == 0 else args.cand
            second = args.cand if p % 2 == 0 else base
            a = run(first, args.grid, args.iters, args.threads)
            b = run(second, args.grid, args.iters, args.threads)
            bres, cres = (a, b) if p % 2 == 0 else (b, a)
            for k in bres:
                ratios.setdefault(k, []).append(bres[k] / cres[k])
                if k not in order:
                    order.append(k)

    result = {"grid": args.grid, "pairs": args.pairs, "kernels": {}}
    for k in order:
        r = ratios[k]
        result["kernels"][k] = {
            "speedup_median": statistics.median(r),
            "min": min(r), "max": max(r),
            "pairs_favouring_candidate": sum(1 for x in r if x > 1.0),
            "pairs": len(r),
        }

    if args.json:
        print(json.dumps(result, indent=2))
        return
    print(f"{args.grid}, {args.pairs} interleaved pairs — >1 means the candidate is faster")
    for k, v in result["kernels"].items():
        print(f"  {k:<26} {v['speedup_median']:6.3f}x  "
              f"(min {v['min']:.3f} max {v['max']:.3f}, "
              f"{v['pairs_favouring_candidate']}/{v['pairs']} pairs favour it)")


if __name__ == "__main__":
    main()
