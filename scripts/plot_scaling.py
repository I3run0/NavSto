#!/usr/bin/env python3
"""Plots a strong-scaling curve (speedup vs. thread count) from a
scripts/benchmark.py CSV produced with --threads.

Usage:
    python3 scripts/benchmark.py --binary navsolver_omp --threads 1,2,4,6,8,12 \\
        --sizes medium --repeats 5 --out /tmp/scaling.csv
    python3 scripts/plot_scaling.py /tmp/scaling.csv

Needs matplotlib (same one-time `.venv` addition as scripts/roofline.py):
    source .venv/bin/activate && pip install matplotlib
"""

import csv
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <benchmark_csv>", file=sys.stderr)
        sys.exit(1)
    csv_path = Path(sys.argv[1])

    by_size = defaultdict(list)
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            by_size[row["size"]].append((int(row["threads"]), float(row["min_seconds"])))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed -- pip install matplotlib into .venv", file=sys.stderr)
        sys.exit(1)

    fig, ax = plt.subplots(figsize=(8, 6))
    max_threads = 1
    for size, points in sorted(by_size.items()):
        points.sort()
        threads = [t for t, _ in points]
        max_threads = max(max_threads, max(threads))
        t1 = next(sec for t, sec in points if t == 1) if 1 in threads else points[0][1]
        speedup = [t1 / sec for _, sec in points]
        ax.plot(threads, speedup, "o-", label=size)

    ideal = list(range(1, max_threads + 1))
    ax.plot(ideal, ideal, "k--", alpha=0.4, label="ideal linear")
    ax.set_xlabel("OMP_NUM_THREADS")
    ax.set_ylabel("Speedup vs. 1 thread")
    ax.set_title("NavSolver OpenMP strong scaling")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out_path = REPO_ROOT / "experiments" / "figures" / "openmp_scaling.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
