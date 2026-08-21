"""Interleaved paired wall-clock A/B for a whole solver binary.

The CUDA backend has no kbench, so a device-side change is measured end to end:
two binaries, alternating which runs first, ratio per pair. Read the win count
with the median. Give it a config with convergenceTol = 0 so both arms do
exactly maxTimeSteps of work.

Usage:
    python3 scripts/paired_app.py <base-binary> <cand-binary> <config> <pairs>
"""
import statistics, subprocess, sys, time, os, tempfile, pathlib
base, cand, cfg, pairs = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
env = {**os.environ, "OMP_PROC_BIND": "close", "OMP_PLACES": "cores"}

def run(binary):
    with tempfile.TemporaryDirectory() as tmp:
        c = pathlib.Path(tmp) / "run.cfg"
        c.write_text(f"outputDir = {tmp}\nrunName = t\n" + pathlib.Path(cfg).read_text())
        t0 = time.perf_counter()
        r = subprocess.run([binary, str(c)], capture_output=True, env=env)
        if r.returncode != 0:
            raise SystemExit(f"{binary} failed:\n{r.stderr.decode()[-800:]}")
        return time.perf_counter() - t0

ratios = []
for p in range(pairs):
    if p % 2 == 0: a, b = run(base), run(cand)
    else:          b = run(cand); a = run(base)
    ratios.append(a / b)
    print(f"  pair {p+1}: base {a:.3f}s  cand {b:.3f}s  -> {a/b:.3f}x", flush=True)
print(f"median {statistics.median(ratios):.3f}x  "
      f"({sum(1 for x in ratios if x > 1)}/{len(ratios)} pairs favour candidate)")
