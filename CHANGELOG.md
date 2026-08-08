# Changelog

All notable changes to this project are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project intends to adopt [Semantic Versioning](https://semver.org/)
once the first tagged release is cut. Until then, entries are grouped
under `[Unreleased]`.

## [Unreleased]

### Added
- `src/openmp/` (copy of `src/serial/`, build wired via `make openmp` and
  CMake's `find_package(OpenMP)`-guarded `navsolver_omp` target) +
  `src/common/RedBlackIndexing.hpp` — red-black restructuring of
  `solvePressurePoisson`, **serially verified, no `#pragma omp` yet**: the
  serial solver's Gauss-Seidel/SOR has a loop-carried dependency that can't
  be correctly parallelized as-is; red-black splits cells by `(i+j+k)`
  parity so each color updates independently of the other. Built as shared
  infrastructure (also the planned foundation for CUDA's pressure kernel).
  Two real findings during verification, both documented in
  `docs/openmp-parallelization.md`: (1) pressure has a null space (a
  uniform additive offset never affects velocity), so comparisons must be
  gauge-fixed (subtract each field's own mean) or they show a large,
  physically-meaningless gap; (2) `AbruptExpansion`'s stepped geometry
  doesn't fully converge even in the *original serial* solver at very
  large sweep counts (a pre-existing property, unrelated to red-black,
  never hit by real usage since production configs use 5-20 sweeps not
  thousands) — the equivalence test uses the simpler `Straight` geometry
  instead. Verified: `scripts/validate.py`'s new red-black equivalence
  tier passes (gauge-fixed `L2_rel=1.15e-3`), the OpenMP binary
  independently passes the Poiseuille analytical check at machine epsilon,
  and ASan/UBSan are clean across both solid-wall and periodic boundary
  configs.
- **Actual OpenMP parallelism** on top of the red-black prerequisite above:
  per-thread scratch buffers for `computeAccelerations` (`SimState`-owned,
  sized `maxThreads × scratchLen`, `#ifdef _OPENMP`-guarded so the serial
  build is unaffected), one `#pragma omp parallel` region wrapping all
  three direction sweeps with `#pragma omp for schedule(static)` per sweep,
  and `#pragma omp parallel for schedule(static)` on `solvePressurePoisson`'s
  `updateColor`. **Found and fixed a real data race** while verifying:
  initially also parallelized `mirrorGhostCells` on an "ownership" argument
  that turned out wrong — caught via a determinism check (same config, same
  thread count, different `DilMax` every run, which can't happen under
  `schedule(static)` if race-free), isolated by binary search (disable one
  pragma at a time) since ThreadSanitizer isn't usable in this WSL2 sandbox
  (`FATAL: unexpected memory mapping` at startup — same class of limitation
  as `perf`). Root cause: `mirrorGhostCells`'s same-row `j`-boundary writes
  can collide with a different thread's cross-row `im`/`ip` write landing
  in that same row. Fixed by leaving `mirrorGhostCells` serial (cheap
  relative to `updateColor`'s real work, not worth the risk without a
  sanitizer to verify a corrected version). Re-verified: bit-identical
  results across `OMP_NUM_THREADS ∈ {1,2,4,6,8,12}` and 5 repeats at 12
  threads, on two geometries, plus ASan/UBSan clean. Full writeup in
  `docs/openmp-parallelization.md`.
- `scripts/validate_parallel.py` (`make validate-parallel`, in CI) — formalizes
  the thread-count-equivalence and determinism checks used to find/verify
  the race above into automated tests. Both bit-identical (`L2_rel=0.0`)
  across `OMP_NUM_THREADS ∈ {1,2,4,6,12}` — expected in retrospect, since
  neither parallelized kernel does cross-thread reduction.
- `scripts/benchmark.py --threads`/`--binary` + `scripts/plot_scaling.py` —
  OpenMP strong-scaling measurement. Real finding, honestly not a good
  result: peak speedup is only ~1.2-1.3x at 2 threads, then flat-to-declining
  — by 12 threads the small grid runs *slower* than single-threaded. Matches
  what `docs/roofline.md` predicted before any of this parallelization work
  started (memory bandwidth here saturates almost immediately past ~2
  threads). Full numbers and plot in `docs/openmp-parallelization.md`; the
  practical implication is that on this machine, further single-thread
  efficiency work (the still-unfinished Y-sweep restructuring) has more
  headroom than adding OpenMP threads does.
- `docs/roofline.md`, `scripts/roofline.py`, `scripts/roofline/*.cpp` —
  roofline analysis for `computeAccelerations` and `solvePressurePoisson`:
  three empirically-measured ceilings (FMA peak, memory bandwidth, and a
  `std::exp()`-throughput ceiling — added because `computeExponentialWeights`
  is ~13% of profiled time and is exp()-bound, not FMA-bound, so a
  standard FMA-only roofline would give a misleading verdict), analytical
  FLOP/byte/exp-call counts hand-derived from the kernels, and achieved
  performance apportioned from a clean `-O3` build's wall time by a
  `gprof` profile's relative breakdown. Verdict: `computeAccelerations`
  shows a real, if methodologically caveated, memory/cache-bound signal
  (see the doc for an honest report of where the conservative byte-count
  assumption was falsified by the measurement) — proceed with the
  loop-order restructuring identified in `docs/serial-optimization.md`.
  `solvePressurePoisson` is not primarily bandwidth-bound (only ~5% of its
  own generous bandwidth ceiling) — more likely limited by its Gauss-Seidel
  read-after-write dependency chain, which is a second, independent reason
  (beyond OpenMP/CUDA correctness) to want the red-black restructuring
  already planned as a parallelization prerequisite. Also measured: memory
  bandwidth on this machine saturates almost immediately past ~2 threads
  (22→34 GB/s from 1→12 threads) — a memory-bound kernel won't scale well
  with OpenMP thread count here, independent of the parallelization being
  correct.
- `docs/serial-optimization-loop-order.md` — acted on the roofline verdict:
  restructured `computeAccelerations`'s X-direction sweep from
  `for(j) for(k) for(i){5 passes}` (`i` innermost, non-unit-stride) to
  `for(j) for(pass) for(i) for(k)` (`k` innermost, matching `GridField`'s
  storage) by widening its scratch buffers from 1-D to 2-D
  (`ppieXK`/`ppiwXK`/`qsieXK`/`KuXK`/`KvXK`/`KwXK` in `SimState`) — same
  math, same `i`-recurrence order, only which loop is innermost changed.
  Added `tests/serial/GoldenFieldTest.cpp` + a captured golden acceleration
  field on a non-equilibrium config (the existing Poiseuille analytical
  check alone can't catch a subtle bug here, since it tests self-consistency
  at equilibrium where residuals are already ~0) — passes, bit-for-bit
  equivalent to the pre-restructuring implementation. Measured effect:
  `computeAccelerations`'s own self-time dropped ~17% (`gprof`, medium
  grid); whole-program speedup 1.02x-1.12x depending on grid size (only
  one of three direction sweeps fixed so far). Y-sweep has the identical
  problem and is the natural next step, not yet done.
- `CHANGELOG.md` (this file).
- `tests/` unit-test scaffold (lightweight header-only harness, no external
  dependencies) with initial coverage for `GridField`, `ConfigParser`, and
  regression tests for the three memory-safety fixes below, wired into both
  CMake (`ctest`) and the Makefile (`make test`).
- `.github/workflows/ci.yml` — builds the solver (Release + Debug) and runs
  `make check` plus the unit tests on every push/PR.
- Validation in `initSimulation()`: `SharpCorner`/`RoundedCorner` geometries
  now reject an incompatible `baseUnit` (relative to grid size) with a clear
  `std::invalid_argument` instead of silently corrupting heap memory.
- `scripts/benchmark.py` (`make bench`) — timing harness establishing a
  serial baseline before any OpenMP/CUDA/MPI optimization work: runs a fixed,
  convergence-independent workload (`convergenceTol=0`, so every run does
  exactly `maxTimeSteps`) across a grid-size matrix, reports `seconds/step`
  and a size-normalized `ns/(active-cell·step)` throughput metric, and writes
  results to `experiments/results/benchmarks/*.csv`. Current serial baseline
  on this machine (Intel Core 7 150U, 1 thread): ~115-200 ns per
  active-cell-step across grids from 1.6k to 358k active cells — i.e.
  roughly flat per-cell cost, no obvious algorithmic blowup with grid size
  at this range.
- `GeometryShape::Straight` — an explicit, honestly-named case for a
  straight channel (full domain active at every x). Previously this exact
  behavior only existed as an unnamed `default:` fallback in
  `initSimulation()`'s geometry switch, reached only by setting one of the
  not-yet-implemented shape names (`OpenCavity`, `GradualExpansion`, etc) —
  an undocumented quirk rather than a supported feature. `default:` still
  falls back to the same straight-channel behavior for forward-compatibility
  with those not-yet-implemented shapes.
- `scripts/validate.py` (`make validate`) — physics correctness checks, two
  tiers: (1) a conservation/stability "blow-up detector" across a couple of
  representative configs (fails on non-finite or unbounded `ResidMax`/`DilMax`);
  (2) an analytical check using plane Poiseuille flow — see the fix to
  `poiseuille.cfg` below for how that's set up, and the README's
  "Correctness checks" section for what this test does and doesn't cover.
- `make profile` — `-O2 -g -pg` gprof-instrumented build target.
- `cfg.sorOmega` (default `1.7`) — SOR relaxation factor for the pressure
  solve, see Changed below.
- `docs/serial-optimization.md` — a real profiling pass (gprof baseline,
  three fixes applied, before/after wall-clock numbers, and an honestly-
  reported case where a naive fix attempt turned out unstable and wasn't
  shipped). Start here before doing any more serial-performance work.

### Changed
- **`solvePressurePoisson()`: Gauss-Seidel → SOR.** Same per-cell cost
  (one extra FMA) but converges markedly faster per sweep — an isolated test
  showed SOR (`omega=1.7`) matching 20-sweep plain-Gauss-Seidel accuracy in
  just 5 sweeps. `numPressureIter` (the fixed per-step sweep count) was
  deliberately left unchanged in every shipped config — a naive attempt to
  cash in the faster convergence by cutting it aggressively (`5→2`) passed
  an isolated single-step check but *diverged over a real multi-step run*;
  `5→3` is verified stable but is documented as a recommendation, not
  applied to configs, since it needs the same per-config full-run check.
  Full writeup, including the matched-iteration-count wall-clock numbers
  (small win at small grids, a wash at larger ones — SOR's real benefit
  needs the sweep-count reduction to show up as wall-clock speedup) in
  `docs/serial-optimization.md`.
- **`computeAccelerations()`'s 12 scratch buffers** (`ppie`, `ppiw`, `Ku`, ...)
  moved from being `std::vector`-allocated fresh every call to
  `SimState`-owned storage allocated once. Verified bit-identical results
  before/after (numerically a no-op, purely removes allocation overhead) —
  measured effect was modest (`gprof` never showed allocation as a distinct
  cost center; see docs).
- **`VtkExporter::writeConvergenceCSV()`** now keeps one file handle open
  for the process lifetime (with an explicit `.flush()` after each write so
  it's still readable mid-run) instead of opening, `fs::exists()`-checking,
  and closing the file every timestep — this was invisible in a 100-step
  `gprof` profile (blocking I/O doesn't register as CPU-sample time) but is
  200,000 open/stat/close cycles for `production_can.cfg`'s
  `maxTimeSteps=200000`.
- Renamed `test_scripts/` to `reference/` — it holds the legacy
  Pascal-derived C++ translation (`navsto_dynamic.cpp`) and validation data
  (`harwelEabRe1000m120p0.txt`), not an automated test suite. The name was
  colliding with the project's actual `tests/` directory.
- Updated `README.md`'s project structure section to match.

### Fixed
- **Heap-buffer-overflow in `computeAccelerations()`** (`src/serial/Physics.cpp`):
  the `ppie`/`ppiw`/`ppin`/`ppis`/`ppiu`/`ppid`/`qsie`/`qsin`/`qsiu`/`Ku`/`Kv`/`Kw`
  temporary coefficient buffers were sized `maxDim+2`, but several boundary
  writes in all three direction sweeps (X/Y/Z) need logical index `maxDim+1`,
  which needs physical slot `maxDim+2` — one past the last valid element.
  Reproduced on the built-in smoke-test grid and on the README's example
  48×24×24 config; confirmed unrelated to this branch's other changes by
  reproducing from a clean `HEAD` checkout first. Fixed by sizing the buffers
  `maxDim+3`; the write/read index formulas themselves were already
  self-consistent (traced every write to its later read), so this only adds
  headroom and doesn't change any computed value.
- **Heap-buffer-overflow in `initSimulation()`**: `SharpCorner`/`RoundedCorner`
  derive `jLow`/`jHigh` index bounds directly from `cfg.baseUnit` (e.g.
  `degreeIndex2 = baseUnit`), with no check that `baseUnit` is compatible with
  the grid size. The built-in smoke-test config left `baseUnit` at its default
  (20) against a 12-cell grid, corrupting heap memory. All real experiment
  configs already keep `baseUnit` at or below `min(numCellsX, numCellsY)/2`
  (the original Pascal convention), so this is now enforced as validation
  (see Added) rather than a silent crash.
- **Heap-buffer-overflow in `solvePressurePoisson()`**: the Gauss-Seidel
  Neumann boundary-mirroring writes one cell past the top of each axis (e.g.
  `press(numCellsX+1, j, k)`) so the next sweep can read a ghost value there,
  but `SimState::allocateFields()` sized fields to exactly the physical
  domain (`numCells+1`) with no ghost layer. This triggered on literally
  every configuration at step 1, regardless of geometry — no run could
  previously complete even one full time step. Fixed by allocating one extra
  ghost layer per axis (`numCells+2`); confirmed nothing else depends on the
  exact field size (`VtkExporter` and all loops iterate the explicit
  `0..numCells` domain, never the raw grid-storage size).
- `.gitignore`'s `results/*` rule didn't match the actual output location
  `experiments/results/*` (missing the `experiments/` prefix), which let
  generated VTK/CSV run output get committed to git. Corrected the pattern
  and stopped tracking the previously-committed output files.
- Stopped tracking compiled binaries and run logs that had been committed
  by mistake: `navsolver`, `navsolver.log`, `reference/navsto` (formerly
  `test_scripts/navsto`).
- `experiments/configs/poiseuille.cfg` didn't actually solve anything: it
  used `numCellsZ=1` with `lateralBC=SolidWall`, which collapses the
  Z-direction loop bound (`KKfim = numCellsZm1 = 0`) to zero iterations —
  and that same bound also gates the per-k loops in the X/Y sweeps, so
  `computeAccelerations()` did no work at all and the residual was trivially
  `0` from step 1 (previously listed as a known issue, not a fix, since
  correcting it needed a real straight-channel geometry that didn't exist
  yet). Fixed by switching to `geometryShape=Straight` (new, see Added) and
  `lateralBC=Periodic`, which makes a thin Z genuinely translation-invariant
  instead of walled-and-degenerate. Now used as the analytical correctness
  check (see `scripts/validate.py` above).

All three fixes verified under AddressSanitizer across the smoke-test grid
and representative `AbruptExpansion` / `RoundedCorner+Periodic` configs, plus
`make check`, `make test`, and `ctest` on a Release build.

### Known issues
- The analytical Poiseuille check in `scripts/validate.py` verifies
  discretization self-consistency (residual ≈ 0 when evaluated at a known
  exact solution used as the initial condition), not convergence to that
  solution from an arbitrary starting point. A full grid-convergence study
  (run the same case at increasing resolution, confirm the error shrinks at
  the expected discretization order) would be a stronger check and is
  natural follow-up work, not done here.

## Project history (pre-changelog)

Summarized from git history prior to this file's introduction:

- **2026-05-02** — Initial commit; repository rearranged into its current
  `src/`, `experiments/`, `test_scripts/` layout; began the C++ translation
  of the original Pascal UNIFAES solver; `experiments/` and `plots/`
  directories added; README written.
- **2026-05-03** — General repository updates; `experiments/` content added.
- **2026-05-17** — `.venv/` added to `.gitignore`.
