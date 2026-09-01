# NavSolver — working agreement

3-D incompressible Navier-Stokes on a staggered Cartesian grid: UNIFAES
exponential scheme, explicit fractional-step projection, steady marching or RK4.
Three backends — serial, OpenMP, CUDA — share `src/core`, `src/solver`, `src/io`
and each own their kernels under `src/backends/<name>`. `README.md` is the
reference for build/run/config; this file is the contract for changing the code.

The work in progress is **testing the UNIFAES method and extracting maximum
performance from it**. Optimization is the default task, not an occasional one.

## The gate: strict Pareto, no exceptions

A change is accepted only if it is **at least as accurate and at least as fast
as HEAD, everywhere measured**. Both halves are hard constraints:

- **Accuracy may not be traded for speed.** Host backends stay `double`.
  Validator tolerances, golden fields and convergence criteria are fixed inputs
  — never widen one to make a change pass. A change that moves a validated
  number is wrong until proven otherwise, even when it is faster.
  The bar is *algebraically* identical, not byte-identical: FMA contraction may
  appear or disappear when branches move, which changes the last bit and is an
  improvement per operation, not a loss. Prove it is only that — rebuild both
  sides with `-DCMAKE_CXX_FLAGS_RELEASE="-O3 -DNDEBUG -march=native
  -funroll-loops -ffp-contract=off"` and compare byte for byte — and say in the
  commit how far the difference grows over a run. Anything that survives
  `-ffp-contract=off` is a real change in the arithmetic and is not allowed.
  When it does shift, every backend the validators cross-check has to shift
  with it in the same commit, or `validate_parallel`'s 1e-10 equivalence fails.
- **No configuration may get slower.** Every kernel and every benchmark size
  must come back `improved` or `within_noise`. One `regressed` rejects the whole
  change, however large the win elsewhere.

So: speed comes from restructuring — memory layout, loop order, fusion, reduced
traffic, scheduling, better algorithms — never from doing less arithmetic or
less carefully.

## Measuring

`scripts/measure.py` is the only source of a verdict; it pins threads, takes
minima over repeats, and gates deltas at `2 x` the measured run-to-run noise so
thermal drift cannot read as a win.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
python3 scripts/measure.py --tier loop --baseline <baseline.json>   # one iteration
python3 scripts/measure.py --tier loop --save                       # new baseline
python3 scripts/measure.py --tier attribution                       # where time goes
python3 scripts/measure.py --tier headroom                          # roofline (slow)
```

Tiers: `gate` (three validators), `kernel` (per-operator, ~1s, the inner loop),
`reward` (whole program, I/O excluded), `attribution`, `headroom`; `loop` =
gate+kernel+reward in one envelope, `full` = everything. Exit codes: **0** clean,
**1** gate did not pass, **2** something regressed. Baselines live in
`experiments/results/baselines/`; re-baseline after every accepted commit, and
never compare across machines — the envelope records the CPU for that reason.

`--backend openmp` measures the OpenMP kernels. CUDA has no kernel tier: use
`validate_cuda.py` plus a whole-application timing, paired and interleaved with
the serial run, since GPU and CPU thermals drift independently.

### What this machine can actually resolve

Measured 2026-08-21 on the Core 7 150U under WSL2 (run-to-run spread of the
minima, so the smallest delta a `2 x noise` verdict can call):

| signal | noise | resolvable win |
|---|---|---|
| serial kernel, 96x48x24 | 0.7–5% | ~2–10% |
| OpenMP kernel, 192x96x48, 6 threads | 8–12% (small kernels 20–38%) | ~23%+ |
| whole-program reward | 6–7% | ~13% |

Two further hazards, both met in practice on 2026-08-21:

- **A saved baseline is a different session, and this machine drifts more than
  its within-session noise floor.** For anything under ~10%, decide with
  `scripts/paired.py` — both arms in one session, alternating which runs first,
  reported as the median of the paired ratios plus a win count. 7/7 pairs at
  1.02x is a result; 4/7 at 1.05x is not. `measure.py --baseline` now refuses a
  baseline whose backend, CPU, grid or thread count differs from the run.
- **Two binaries differ in code layout as well as in the change**, and the
  effect is per-kernel. Measured by pairing two *functionally identical* builds
  that differ only by an appended unused function: `computeAccelerations` stays
  inside 0.992-1.013x, while `updateVelocities` swings **0.695-1.359x**. A 1.5%
  move in the first is signal; a 4% move in the second is nothing. Get a
  kernel's layout band that way before believing a small delta in it, or put
  both paths in one binary behind a runtime switch and pair it against itself.

Grid choice for `kbench` on this machine: **96x48x24 is the trustworthy point**
(untouched kernels sit at 1.000 in a layout-neutral pair), 144x72x36 is usable,
and **240x120x60 is not** — at 7 GB of RAM its footprint puts kernels the change
cannot touch anywhere between 0.76x and 1.47x. Production-size claims come from
the whole-program tiers, not from kbench.

**Multi-threaded kernel timing is not a signal on this machine — at all.** Two
binaries whose relevant code is byte-identical measured 1.85x apart at 6
threads, 7/7 pairs, while tying to within 1% at one thread; the same binary
ranges 0.30–1.20 ms run to run. Explicit `OMP_PLACES` does not fix it. So:

- Measure an OpenMP **code** change at `--threads 1`, interleaved. That is the
  same code path, and it is reproducible to ~2%. The buildPressureSource port
  reads 2.29x there and an unusable 1.07–2.06x at six.
- Take **threading speedup** from `--tier reward --backend openmp` — whole
  program, where the jitter averages out — or do not claim a number.

Absolute levels drift hard over minutes: the same kernel and binary measured
0.72 ms and 0.50 ms twenty minutes apart. Only ratios from an interleaved
window mean anything; never compare two numbers from different windows. A sub-13% whole-program win is not provable here
at all — say so rather than claiming it.

### Check whether the kernel vectorises before tuning it

```bash
g++ -std=gnu++20 -O3 -DNDEBUG -march=native -funroll-loops -Isrc/core -Isrc/solver \
    -Isrc/io -Isrc/backends/serial -Isrc/backends/host \
    -c src/backends/serial/Physics.cpp -o /dev/null -fopt-info-vec-all
```

`buildPressureSource` was scalar — its 48 unit-stride loads per cell could not
be vectorised because the store through `GridField::operator()` might, as far as
the compiler knows, land on the fields' own `sJ`/`sK` members. Hoisting
`__restrict` pointers and lifting the row into its own function made it 2.26x.
Two traps found doing it: a `km` that wraps for periodic z is not affine in `k`
and blocks the whole loop (peel that iteration), and a loop body in a lambda is
outlined into an `isra` clone unless the vectorised form lives in a real
function. Use the project's own flags in that command — plain `-O3` reports
`V2DF` because it misses `-march=native`, which is a different question.

## Iteration protocol

1. Baseline at HEAD (`--tier loop --save`) if there is not a current one.
   Never gate with `--skip-build`: a stale binary passes every validator, and
   a change that does not compile on a backend you are not building reads as
   green. `measure.py` builds by default — let it.
2. One hypothesis, one change. Pick it from `attribution`/`headroom`, not by
   guessing; state the mechanism you expect to pay off before measuring.
3. Rebuild, run `--tier loop --baseline <baseline>`.
4. **Accept** only on gate `pass`, zero `regressed`, at least one `improved` —
   then commit it alone, and re-baseline.
   **Reject** otherwise, revert the working tree, and record why in the notes
   below if the negative result is informative.
5. Never batch two optimizations into one commit; a mixed commit cannot be
   bisected into a win and a regression.

Commit messages follow the existing style: `perf(<area>): what changed — the
measured number`, then the mechanism, the paired numbers, and explicitly where
the theory was wrong. Negative results get committed as documentation when the
code change is reverted but the finding is worth keeping.

## Standing exception, needs a decision

`c561e2d` made the **CUDA device default to `float`** (1.17x, host untouched and
bit-identical). Under the accuracy rule above that default is a live exception:
it agrees with the fp64 device build to six significant figures, not exactly,
and `validate_cuda.py` widens its RK4 tolerance from 1e-12 to 1e-5 to admit it.
Rebuild with `-DNAVSOLVER_CUDA_REAL=double` to restore fp64; the startup
  banner reports `CUDA (fp32)` or `CUDA (fp64)`, and is the thing to check
  — that flag was accepted but unwired until `83b12f5`, so it silently
  produced a float build. Do not extend the
fp32 pattern to any other backend, and do not widen another tolerance to match.

## Do not

- Loosen a tolerance, shrink a validation grid, or regenerate a golden file to
  make a change pass. Golden files change only when the physics intentionally
  changes, in their own commit.
- Time anything from `build-debug` (sanitizers) or a `-pg` profile build; those
  measure instrumentation. `NAVSOLVER_PROFILE=ON` totals are not comparable
  either — only its per-kernel shares are.
- Edit `reference/navsto_dynamic.cpp` or `reference/harwelEabRe1000m120p0.txt`.
  They are the Pascal-derived ground truth this port is checked against.
- Commit anything under `experiments/results/` except baselines, or leave the
  build trees, `navsolver*` binaries or `navsolver.log` in a commit.
