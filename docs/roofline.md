# Roofline analysis: is the serial code compute-bound or memory-bound?

**Date:** 2026-08-07
**Builds on:** `docs/serial-optimization.md`
**Machine:** Intel Core 7 150U, 6 cores/12 threads, single socket/NUMA node,
AVX2+FMA, L1d 288 KiB (48 KiB/core), L2 7.5 MiB (1.25 MiB/core), L3 12 MiB.

## Why this exists

`docs/serial-optimization.md` identified a likely-large remaining
optimization (the X/Y-direction sweeps in `computeAccelerations` don't
match `GridField`'s memory layout) but explicitly left it unfixed pending
evidence that it's actually worth the risk — a layout fix only helps if the
kernel is memory-bound; if it's compute-bound, parallelism (not layout)
is the right next lever. This analysis answers that question with
measurements instead of intuition.

## Methodology

Hardware performance counters (`perf`, LIKWID, PAPI) are not reliably
usable under this WSL2 sandbox — confirmed earlier in this project (`perf`
binary present but missing the matching kernel package; WSL2's Hyper-V
layer typically doesn't expose real PMU access to the guest). This is the
documented analytical/empirical fallback for that situation, not the
industry-standard automated path (Intel Advisor / LIKWID) — see the
"industry standard" discussion earlier in this project's history for that
distinction. Three pieces, each independently reproducible:

1. **Peak ceilings, measured empirically on this machine** (not vendor
   spec sheets) via `scripts/roofline/*.cpp`, compiled with the project's
   own Release flags (`-O3 -march=native -funroll-loops`).
2. **Analytical FLOP/byte/exp-call counts**, hand-derived from
   `src/serial/Physics.cpp`, transcribed as constants at the top of
   `scripts/roofline.py`.
3. **Achieved performance**: a clean `-O3` Release build's total wall-clock
   time (never the `-pg` build — confirmed in the prior serial-optimization
   pass that its per-call instrumentation overhead distorts absolute
   timing) apportioned by a `-pg`/`gprof` profile's *relative* self-time
   breakdown, on the same baseline workload as `docs/serial-optimization.md`
   (`AbruptExpansion`, 96×48×24, 100 steps, `SteadyMarching`).

Run via `python3 scripts/roofline.py` (needs `matplotlib` in `.venv`, added
this session — `benchmark.py`/`validate.py` stay stdlib-only).

## Peak ceilings measured on this machine

| Ceiling | 1 thread | 12 threads (best of 3) |
|---|---|---|
| FMA GFLOP/s | 25.7 | 255.4 |
| Memory bandwidth (STREAM-triad) | 22.3 GB/s | 34.0 GB/s |
| `std::exp()` throughput | 2.86×10⁸ calls/s | 1.26×10⁹ calls/s |

Vectorization confirmed for the FMA microbenchmark (not just assumed from
the flag): `-fopt-info-vec-optimized` reports "loop vectorized using 32
byte vectors" (AVX2, 256-bit) and the compiled binary contains 32 `vfmadd`
instructions.

**Notable finding, not part of the original ask but relevant to Phase 3
(OpenMP) planning**: memory bandwidth saturates almost immediately —
2 threads (33.8 GB/s) is nearly indistinguishable from 12 threads
(29.6-43.0 GB/s across repeated runs, noisy but flat, no clear upward
trend past ~2 threads). If a kernel turns out memory-bandwidth-bound,
**OpenMP thread count won't help it much on this specific machine** —
worth remembering when interpreting Phase 3's scaling curves; a
memory-bound kernel scaling poorly with threads here would confirm this,
not indicate a bug in the parallelization.

## Analytical per-cell counts (hand-derived from `src/serial/Physics.cpp`)

**`computeAccelerations`** (summed across all 3 direction sweeps, treated
as structurally equal cost — X/Y/Z differ only in which velocity component
leads and which neighbor offsets are used, not in operation count):

- ~79 FLOPs/cell/sweep × 3 sweeps = **237 FLOPs/cell**
- 2 `std::exp()` calls/cell/sweep (inside `computeExponentialWeights`,
  called twice per sweep — once for face coefficients, once for the
  cross-term correction) × 3 sweeps = **6 exp() calls/cell**
- **~2400 bytes/cell**, conservative estimate: every distinct field-array
  element access (`velX`/`velY`/`velZ`/`accelX`/`accelY`/`accelZ`), in
  every separate pass, counted as one fresh 8-byte DRAM transaction — i.e.
  assuming **no** reuse across the 5 sequential passes per plane. Flagged
  explicitly as likely too conservative — see Results below, where the
  measurement itself falsifies this assumption.

Cross-check: 6 exp() calls/cell × 105,984 active cells × 100 steps =
63.6M — matches the ~60.3M `computeExponentialWeights`-adjacent calls
measured in `docs/serial-optimization.md`'s baseline profile closely
enough (same order, same workload family) to trust the per-cell counting
methodology.

**`solvePressurePoisson`**, per interior cell:
- **13 FLOPs** (10 for the 7-point stencil update + 3 for the SOR blend)
- **72 bytes**, same no-reuse-assumed methodology (7 neighbor+source reads
  + 1 self-read for the SOR blend + 1 write, no cross-cell reuse assumed)

## Results

| Kernel | Arithmetic Intensity | Achieved | vs. FMA peak (25.7) | vs. bandwidth-implied ceiling at this AI |
|---|---|---|---|---|
| `computeAccelerations` | 0.099 FLOP/byte | 5.06 GFLOP/s | 20% | **139%** — exceeds it |
| `solvePressurePoisson` | 0.181 FLOP/byte | 0.22 GFLOP/s | 0.8% | 5% |

![Roofline plot](../experiments/figures/roofline.png)

## Honest methodological finding: the conservative byte-count isn't a tight bound

`computeAccelerations`'s achieved throughput (5.06 GFLOP/s) sits *above*
the bandwidth ceiling implied by its own analytically-derived arithmetic
intensity (2.20 GFLOP/s) — visible in the plot as the blue dot sitting
above the dashed bandwidth roof line at its x-position. That's not
physically possible for a true DRAM-bandwidth ceiling; it means the
"no-reuse-across-passes" byte-counting assumption is measurably too
conservative. The working set touched per `(j,k)` plane in one pass
(roughly one cache line's worth of values across the swept dimension) is
small enough to plausibly stay resident in L1/L2 across the 5 passes,
so real DRAM traffic is lower — and real arithmetic intensity higher —
than the naive count assumes. **Reported plainly rather than adjusting the
constant after the fact to make the plot look consistent**: the specific
"X GB/s away from the DRAM roof" framing shouldn't be trusted at face
value for this kernel. What *does* still hold regardless of exactly which
byte-count is right: achieved throughput (5.06 GFLOP/s) is well below both
the FMA peak (25.7, ~20%) and the `computeAccelerations`-specific
exp()-throughput ceiling (11.3 GFLOP/s-equivalent, ~45%) — so there is real
headroom being lost to *something* beyond the exp() calls themselves,
consistent with (though this analysis alone doesn't prove) the
non-unit-stride access pattern identified in `docs/serial-optimization.md`.

`solvePressurePoisson` has no such anomaly — achieved (0.22 GFLOP/s) sits
cleanly *below* both its bandwidth-implied ceiling (4.03 GFLOP/s, achieving
only ~5% of it) and the FMA peak (~0.8%). This is a much larger, more
reliable gap than `computeAccelerations`'s, and the likely explanation
isn't primarily memory bandwidth at all: Gauss-Seidel/SOR's per-cell update
reads a neighbor that was *just written* in the same sweep — a genuine
read-after-write dependency on every iteration that limits instruction-
level parallelism regardless of how fast memory is. Its `j`-innermost loop
(sweep order `for(i) for(k) for(j)`) also doesn't match `GridField`'s
`k`-fastest-varying storage, compounding the effect, but the RAW-hazard
story is likely dominant given how far below even the generous ceiling it
sits.

## Verdict

**`computeAccelerations`: proceed with the Phase 2 loop-order
restructuring.** The low arithmetic intensity and the ~20%-of-FMA-peak /
~45%-of-exp-ceiling achieved performance both point the same direction —
real headroom, likely a memory/cache-access-pattern problem — even though
the specific DRAM-bandwidth-ceiling number shouldn't be over-trusted given
the finding above. This is evidence *for* proceeding, not proof of the
exact mechanism; the golden-field regression test specified in the
approved plan is what actually validates the fix, not this analysis alone.

**`solvePressurePoisson`: the red-black restructuring already planned for
Phase 3 is doing double duty.** It wasn't scoped as a serial speedup — it
was scoped as an OpenMP/CUDA correctness prerequisite (Gauss-Seidel's
sequential dependency can't be parallelized as-is). This analysis adds a
second, independent reason to want it: breaking that same dependency chain
(processing color-independent cells that don't depend on each other) is
also very likely to improve *serial* instruction-level parallelism, given
how far below even a generous ceiling the current implementation sits.
Worth measuring serial wall-clock before/after the red-black change lands,
not just correctness — a plausible free serial win alongside the
parallelization work.

## Reproducing this

```bash
source .venv/bin/activate   # matplotlib
python3 scripts/roofline.py
# writes experiments/results/roofline/roofline_summary.csv,
#        experiments/results/roofline/gprof_raw.txt,
#        experiments/figures/roofline.png
```
