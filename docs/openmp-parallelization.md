# OpenMP parallelization

**Date:** 2026-08-07 (started)
**Builds on:** `docs/roofline.md`, `docs/serial-optimization-loop-order.md`

## Why red-black first

The serial solver's `solvePressurePoisson` is plain Gauss-Seidel/SOR: each
cell's update reads a neighbor that was just written earlier in the same
sweep — a genuine loop-carried dependency. That can't be correctly
parallelized as-is; naive `#pragma omp parallel for` on it either races
(wrong, non-reproducible results depending on thread scheduling) or has to
serialize (defeating the point).

Red-black splits cells into two colors by `(i+j+k)` parity. Every cell's 6
face-neighbors are always the *other* color, so within one color, no
cell's update depends on another cell of the same color — updating a whole
color is embarrassingly parallel. Built and verified **serially first**
(no `#pragma omp` anywhere yet), deliberately: conflating "did I break the
algorithm" with "did I introduce a race" would make both nearly
undebuggable together, and this is a genuine algorithm change (same fixed
point, different iteration path), not just an engineering refactor.

Also built as shared infrastructure (`src/solver/Geometry.hpp`),
not OpenMP-specific — the same red/black index lists are the planned
foundation for the CUDA pressure-solve kernels too, so this is built once
rather than re-derived per backend.

## What changed

- `src/solver/Geometry.hpp` — `buildRedBlackIndices(SimState&)`
  fills `s.redCells`/`s.blackCells` (flat `CellIndex{i,j,k}` lists) for the
  active interior domain, using the identical bounds logic
  `solvePressurePoisson` already used for its `jLoopS`/`jLoopN`
  active-span computation. Built once (lazily, on first
  `solvePressurePoisson` call) and reused for the whole run — geometry is
  fixed after `initSimulation()`. **Superseded by the row-list/strided
  scheme in "Round 2" below** — kept here as the accurate historical
  record of what was first built and verified.
- `src/backends/openmp/` created as a full copy of `src/backends/serial/` (deliberately
  duplicative, not `#ifdef`-branched into the serial file — lower risk,
  doesn't destabilize the already-verified serial path while editing for
  parallelism; the tradeoff is ~900 duplicated lines, with a noted future
  cleanup once both paths stabilize: extract the unchanged, non-hot
  functions into `src/core/`).
- `solvePressurePoisson` in `src/backends/openmp/Physics.cpp` restructured: the
  serial version's *inline* Neumann ghost-cell mirroring (interleaved
  mid-sweep, using whatever `press(i,j,k)` holds at that point in a single
  continuous traversal) can't survive reordering into red-black — colors
  update out of that original traversal order. Factored into its own
  explicit `mirrorGhostCells()` pass, run **twice per sweep**: once before
  updating RED, once before updating BLACK (BLACK's boundary cells need to
  see RED's just-updated interior values reflected in the ghost layer,
  which a single shared mirror pass wouldn't provide).
- Both build systems wired: `make openmp` → `navsolver_omp`; CMake
  `find_package(OpenMP)` (not `REQUIRED`) guards a `navsolver_omp` target,
  so configure still succeeds on a machine without OpenMP.

## Two real findings while verifying, not glossed over

**1. Pressure has a null space — comparisons must be gauge-fixed.** A
uniform additive offset to the whole pressure field never affects
velocity (only `∇p` matters), so serial and red-black converging to
solutions that differ by such an offset would still be "correct" — but a
naive value-by-value comparison would show a large, misleading gap. Traced
by checking whether subtracting each field's own mean before comparing
closed most of the gap (it did — from ~2% raw down to ~0.1%). `scripts/validate.py`'s
red-black equivalence tier now explicitly gauge-fixes before comparing;
see the module docstring there for the full reasoning.

**2. `AbruptExpansion`'s stepped geometry doesn't fully converge even in
the *serial* (original) solver at very large sweep counts.** Testing the
equivalence check at increasing `numPressureIter` (200 → 3000 → 20000) to
see the gap close instead showed it *growing* (14% → 30% → 42%
gauge-fixed), and critically the **serial** solver's own mean pressure was
still drifting upward at 20000 sweeps (−0.045 → 0.096 → 0.220) — not a
red-black artifact, a property of the original algorithm on this
geometry's stepped boundary at sweep counts far beyond anything real usage
hits (production configs use 5-20 sweeps, never thousands). The simpler
`Straight` geometry (full rectangular domain, no stepped boundary)
converges cleanly and was used for the equivalence test instead — testing
algorithm correctness needs a well-posed case, and this drift is a
separate, pre-existing question about the original solver's long-run
behavior on complex geometry that's out of scope for this parallelization
work. Flagged here so it isn't silently rediscovered later; not
investigated further in this pass.

## Verification so far (before any `#pragma omp`)

- `make test`: 15/15 (unaffected — no `src/openmp` tests yet, this counts
  the existing serial suite staying green through the `SimState.hpp`
  additions shared by both paths).
- `scripts/validate.py`: all three tiers pass, including the new red-black
  equivalence tier (gauge-fixed `L2_rel = 1.15e-3`, well under the `1e-2`
  tolerance).
- The `navsolver_omp` binary (serial red-black, no parallelism yet)
  independently passes the Poiseuille analytical check at machine epsilon
  (`ResidMax ≈ 1.5e-14`) — strong evidence the red-black port is solving
  the actual physics correctly, not just "close to serial."
- AddressSanitizer + UBSan clean across `AbruptExpansion`/`SolidWall` and
  `RoundedCorner`/`Periodic` (exercises both the non-periodic ghost-mirror
  path and the periodic-wrap path in `updateColor`).

## Parallelization

**`computeAccelerations`**: per-thread scratch buffers. The shared
`SimState`-owned buffers from `docs/serial-optimization.md`/
`docs/serial-optimization-loop-order.md` (`ppin`/`ppis`/.../`Ku`/`Kv`/`Kw`,
and the X-sweep's 2-D `ppieXK`/etc.) are a real race hazard once multiple
threads process different planes concurrently — sized `maxThreads ×
scratchLen` (`SimState::allocateFields()`, `maxThreads = 1` in the serial
build via `#ifdef _OPENMP`, identical behavior to before per-thread buffers
existed), each thread's slice padded to a 64-byte (8-double) boundary to
reduce false sharing at slice boundaries. `VM`/`VM2` changed from
`std::vector<double>&` to `double*` so each thread just computes its own
offset pointer once at the top of one `#pragma omp parallel` region
wrapping all three sweeps, with `#pragma omp for schedule(static)` on each
sweep's outer loop (`j` for X, `i` for Y and Z) — `schedule(static)` both
because per-plane cost is fairly uniform for this geometry and because it
makes results deterministic at a fixed thread count, which the
verification below depends on.

**`solvePressurePoisson`**: `#pragma omp parallel for schedule(static)` on
`updateColor`'s per-cell loop (converted from range-based to index-based —
required for OpenMP's canonical loop form).

### A real data race, found and fixed

Initially also parallelized `mirrorGhostCells` (same O(active-cells) cost
as one color's update, called twice per sweep — leaving it serial looked
like a real Amdahl's-law loss) based on an "ownership" argument: each
ghost-cell write seemed attributable to exactly one owning `i`. **That
argument was wrong.** Caught it via a determinism check — running the same
config at a fixed thread count (12) multiple times gave *different*
`DilMax` every run, which cannot happen under `schedule(static)` if the
code is race-free. ThreadSanitizer isn't usable to pinpoint it directly
(same WSL2 limitation as `perf`, see `docs/roofline.md` — TSan fails with
`FATAL: unexpected memory mapping` at startup in this sandbox), so isolated
it by binary search: disable `updateColor`'s pragma only (still
non-deterministic) vs. disable `mirrorGhostCells`'s pragma only (became
deterministic) — confirmed the race was in `mirrorGhostCells`.

Root cause: the `j==jLoopS`/`j==jLoopN` ghost extensions write
`press(i, j∓1, k)` — same row `i` as the writing thread — but a
*different* thread processing `i+1` can also write into row `i` via its
own `im=i` target, whenever `i+1 == s.iLow[j']+1` for some `j'` that
happens to equal this thread's `j∓1` target. Two threads writing the same
address with different source values — exactly the collision the original
"no two `i` values collide" analysis missed (it only checked `im`/`ip`
cross-`i` writes against each other, not against the same-row `j`-shift
writes).

**Fix (round 1): left `mirrorGhostCells` serial.** It's cheap relative to
`updateColor`'s actual linear-algebra work, and without TSan available to
formally verify a corrected parallel version, not worth the correctness
risk. Re-verified after the fix: bit-identical `DilMax` across
`OMP_NUM_THREADS ∈ {1,2,4,6,8,12}`, 5 repeats at 12 threads, on two
different geometries (`AbruptExpansion`/`SolidWall` and
`RoundedCorner`/`Periodic`), plus a clean ASan/UBSan pass at 1/6/12 threads
on both.

**Revisited in round 2 (below) — a corrected parallel version was found,
implemented, and verified**, once the exact collision pattern above was
understood precisely enough to split it out. See "Parallelizing
`mirrorGhostCells`, this time correctly" under Round 2.

## Verification

- `make test`: 15/15.
- `scripts/validate.py`: all three tiers pass, including the red-black
  equivalence tier (gauge-fixed `L2_rel = 1.15e-3`).
- `navsolver_omp` independently passes the Poiseuille analytical check at
  machine epsilon (`ResidMax ≈ 1.5e-14`).
- `scripts/validate_parallel.py` (`make validate-parallel`, wired into CI)
  formalizes the manual checks used during debugging into two automated
  tiers: **thread-count equivalence** (every `OMP_NUM_THREADS` compared
  against 1-thread `navsolver_omp`, deliberately *not* against serial
  `navsolver` — red-black only matches serial at high sweep counts, an
  unrelated fact already covered by `scripts/validate.py`'s red-black
  tier; comparing against serial here would conflate two different
  questions) and **determinism** (same config run twice at a fixed thread
  count). Both bit-identical (`L2_rel = 0.0`) across
  `OMP_NUM_THREADS ∈ {1,2,4,6,12}` on both geometries — makes sense in
  retrospect: neither `computeAccelerations` nor `updateColor` do any
  cross-thread reduction (each cell is written by exactly one thread), so
  there's no floating-point reordering source to begin with, unlike a
  typical OpenMP reduction pattern.
- AddressSanitizer + UBSan clean at 1/6/12 threads on both geometries.
- `mirrorGhostCells` remaining serial was a known, documented limitation as
  of round 1 — **round 2 (below) found and verified a correct parallel
  version**, so this is no longer current; kept here as the historical
  record of the round-1 state.

## Round 2: further OpenMP tuning (2026-08-08)

Prompted by a direct question: *is this the best possible use of OpenMP
here, given what round 1 already found?* Round 1 left three concrete,
named gaps: `mirrorGhostCells` staying serial (a real Amdahl's-law tax),
one parallel-region open/close per `updateColor` call (10 team spawns per
pressure solve at the default `numPressureIter=5`), and an untested
hypothesis that `solvePressurePoisson`'s combination of low arithmetic
intensity *and* only ~5% of the roofline's bandwidth ceiling meant it was
latency-bound (a per-cell index-list *gather*), not actually bandwidth
bound. This round attacks all three, plus checks a fourth lever
(vectorization) that was flagged but never verified.

### Killing the red-black gather

`RedBlackIndexing.hpp`'s round-1 `s.redCells`/`s.blackCells` were flat
`std::vector<CellIndex>` lists — each cell access in `updateColor` needed
an indirect load of `(i,j,k)` *before* the real `press()`/`pressureSource()`
addresses were known, a gather pattern that defeats hardware prefetching.

The fix exploits a fact the per-cell list threw away: `k` is never masked
by the active-domain geometry (only `i,j` are), so every active `(i,j)` row
spans the *full* `k ∈ [1,numCellsZ]`. A row plus a color is therefore
enough to derive that color's `k`-stride directly:
`kStart(red) = ((i+j)%2==0) ? 2 : 1`, the opposite for black, both
stepping by 2. `s.redCells`/`s.blackCells` became a single `s.activeRows`
list of `RowIndex{i,j}` (used for *both* colors — a row holds cells of both
colors at different `k` parity), and `updateColor`'s inner loop became a
direct strided loop instead of an index dereference. Same cells touched,
same math, same convergence path — verified: `scripts/validate.py`'s
red-black-vs-serial tier gives the *identical* `L2_rel=1.152e-03` before
and after (the same digits, not just "close").

### Fusing parallel regions

Round 1's `updateColor` opened its own `#pragma omp parallel for` on every
call. With the default `numPressureIter=5`, one `solvePressurePoisson` call
spawned/joined a thread team 10 times (5 sweeps × 2 colors), every single
timestep — a fixed cost that doesn't scale with grid size, and a plausible
driver of round 1's small-grid-regresses-at-high-thread-count result.
`solvePressurePoisson` now opens *one* `#pragma omp parallel` region
spanning the whole sweep loop; `updateColor` and `mirrorGhostCells` use
orphaned `#pragma omp for` (standard OpenMP — a worksharing construct in a
called function binds to the enclosing team), relying on the implicit
barrier at the end of each `for` region (no `nowait` anywhere) for the same
ordering guarantees round 1 had.

### Parallelizing `mirrorGhostCells`, this time correctly

Round 1's race (see above) was between `im`/`ip` writes landing in an
*adjacent* row and same-row `j`-boundary writes from a *different* thread
processing that adjacent row. Stated that precisely, the fix is a matter of
temporal separation, not avoiding parallelism altogether: split into two
`#pragma omp for` passes with the implicit barrier between them —

- **Pass A**: only the cross-row (`im`/`ip`) writes, which target row `i-1`
  or `i+1`, never row `i` itself.
- **Pass B** (after the barrier from Pass A): only the same-row writes
  (`j`-boundary and, for `SolidWall`, the `k` ghost layer), which by
  construction only ever target `(i, ...)` — this thread's own row — so
  once Pass A has *fully* completed, no other thread can still be writing
  into row `i`.

(Pass A itself has no cross-thread collision either — two different `i`'s
`im`/`ip` writes could only target the same neighbor row if some row `j`
had `iLow[j] > iHigh[j]`, an inverted/empty active span that doesn't happen
for a valid active row; this reuses the "no two `i` values collide" half
of round 1's original analysis, which was correct as far as it went.)

Verified with the *same* determinism methodology that caught the original
race, since TSan still isn't usable in this sandbox: `OMP_NUM_THREADS=12`,
5 repeated runs of the same config, compared **byte-for-byte** (not just
displayed-precision `DilMax`) — the convergence CSV and VTK snapshot were
identical across all 5 runs, and identical again against the 1-thread
reference. `scripts/validate_parallel.py`'s automated equivalence and
determinism tiers also stayed at `L2_rel=0.000e+00` across
`OMP_NUM_THREADS ∈ {1,2,4,6,12}`. A fresh ASan+UBSan+leak build was clean
at 12 threads on the same config.

### Vectorization check

Compiled with `-fopt-info-vec-optimized -fopt-info-vec-missed` to check
whether `updateColor`'s inner `k` loop auto-vectorizes (motivated by
`solvePressurePoisson` having obvious headroom below its own bandwidth
ceiling, so SIMD is a free-standing lever independent of the gather fix
above). It doesn't: GCC reports `missed: not vectorized: control flow in
loop` at the inner loop. The blocking control flow is real and
correctness-load-bearing — the rare reference-pressure-pin `continue` and
the Neumann-boundary corner correction — not a `restrict`/aliasing hint GCC
is missing. Per the project's rule of not force-vectorizing physics-
affecting branches without a fresh accuracy-verification pass (the same
caution applied to the exp()-throughput ceiling in `docs/roofline.md`),
this is reported as a finding, not fixed here.

### Verification (round 2)

- `make test`: 15/15, unchanged.
- `scripts/validate.py`: all three tiers pass; red-black-vs-serial tier's
  `L2_rel` is bit-identical to round 1's `1.152e-03`.
- `scripts/validate_parallel.py`: thread-count equivalence and determinism
  both `L2_rel=0.000e+00` across `OMP_NUM_THREADS ∈ {1,2,4,6,12}`, both
  geometries.
- Manual byte-for-byte determinism check (CSV + VTK diff, not just a
  logged scalar): 5×12-thread runs identical, 12-thread vs 1-thread
  identical.
- Fresh `-fsanitize=address,undefined,leak` build clean at 1/12 threads.

### Performance — first pass was contended, here's the clean measurement

The first attempt at re-measuring this round's scaling ran on a machine
with an **unrelated, sustained, CPU-heavy process** already running
throughout (`navsolver`/`navsolver_omp` on
`experiments/configs/production_can.cfg`, Re=10000, ~1.2M cells, consuming
3-4 cores continuously for 100+ CPU-minutes — confirmed via `ps`/
`/proc/<pid>/cwd` to belong to a different, unrelated task on the same
shared machine). That run's 8/12-thread columns were flagged unreliable and
not used for conclusions; once that process was stopped, the sweep was
re-run clean. **The numbers below are from the clean re-run.**

Even clean, the 12-thread column (this CPU's full logical-core count,
leaving the OS/OpenMP runtime itself no slack) showed real run-to-run
variance on repeated measurement: `tiny`@12 gave `0.145s`, `0.039s`, then
`0.050s` across three separate 5-repeat batches taken minutes apart — a
~3.7x spread with no code or system change between them. `small`@12 and
`large`@12 showed smaller but still real swings (roughly 30%) across
repeat batches. The 1-8 thread columns were comparably stable across
repeated checks (single-digit-percent spread). The 12-thread numbers below
use the minimum of a larger 8-repeat batch specifically to damp this down,
but should still be read as noisier than the rest of the table — this
variance is itself a real, if only partially explained, finding about this
machine (WSL2 thread scheduling at full core count is the leading
suspect), not swept under the rug.

`scripts/benchmark.py --binary navsolver_omp --threads 1,2,4,6,8,12` on an
otherwise-idle machine, `AbruptExpansion`, 50 steps, min of 5 repeats (8 at
12 threads):

| threads | tiny  | small | medium | large |
|---|---|---|---|---|
| 1  | 1.00x | 1.00x | 1.00x | 1.00x |
| 2  | 1.07x | 1.36x | 1.25x | 1.26x |
| 4  | 0.74x | 1.22x | 1.14x | 1.23x |
| 6  | 0.86x | 1.10x | 1.13x | 1.20x |
| 8  | 0.51x | 1.27x | 1.16x | 1.13x |
| 12 | 0.31x | 1.00x | 1.27x | 1.02x |

![OpenMP strong scaling, round 2](../experiments/figures/openmp_scaling.png)

**Compared directly against round 1's own clean table** (reproduced below,
same methodology, same machine, before this round's changes):

| threads | small (r1 → r2) | medium (r1 → r2) | large (r1 → r2) |
|---|---|---|---|
| 2  | 1.20x → 1.36x | 1.25x → 1.25x | 1.27x → 1.26x |
| 4  | 1.00x → 1.22x | 1.03x → 1.14x | 1.17x → 1.23x |
| 6  | 0.99x → 1.10x | 1.01x → 1.13x | 1.14x → 1.20x |
| 8  | 1.03x → 1.27x | 1.00x → 1.16x | 1.11x → 1.13x |
| 12 | **0.53x → 1.00x** | 0.91x → **1.27x** | 1.02x → 1.02x |

**This is a real, broad win, not just the gather fix's serial-side speedup
carrying over.** Every 4/6/8-thread entry for `small`/`medium`/`large`
improved, several substantially. Most strikingly: round 1's small-grid
12-thread collapse (0.53x — *slower* than single-threaded) is gone
(1.00x, break-even); `medium`@12 threads went from round 1's worst-of-table
0.91x to this round's *best*-of-table 1.27x. Both are exactly what the
Amdahl's-law-tax hypothesis for `mirrorGhostCells` staying serial (round
1's stated but undiagnosed suspicion) predicts once that tax is removed —
the effect should show up most at higher thread counts, where the fixed
serial fraction previously ate a larger share of the wall clock, and
that's precisely where the improvement is largest. `large`@12 is the one
flat entry (1.02x → 1.02x) — plausibly because `large`'s much bigger
per-thread workload already amortized the fixed overheads round 1 was
losing to, leaving less for this round's fixes to recover; consistent with,
not contradicting, the Amdahl's-law explanation.

**`tiny` is the one real regression, and it's a legitimate result, not a
bug.** At 1656 active cells split across up to 12 threads, per-thread work
drops to ~138 cells — small enough that fixed per-thread cost (team
scheduling, cache-line padding overhead in the per-thread scratch buffers,
OS thread wake-up latency) dominates actual work, and scaling degrades
monotonically past 2 threads (1.07x → 0.74x → 0.86x → 0.51x → 0.31x).
Round 1 didn't benchmark a grid this small, so there's no direct
before/after for `tiny`, but the shape (monotonic decline once thread count
exceeds what the problem can usefully use) is exactly what's expected for
a fixed-size problem stretched across more workers than it has parallel
work to give them — not something either round's changes caused or could
fix; a bigger config or fewer threads is the correct response, not more
tuning.

**Bottom line on "is this the best possible OpenMP use now?"**: measurably
closer. The gather removal, region fusion, and corrected
`mirrorGhostCells` parallelization together produced a broad, verified
improvement at every thread count from 2-12 for every grid size `small`
and up — the small-grid 12-thread collapse that was round 1's most visible
weak point is specifically fixed. What *hasn't* changed: the roofline's
underlying ceiling (memory bandwidth saturating past ~2 threads on this
CPU) still caps how far OpenMP alone can go — peak speedup is still ~1.2-
1.4x, not the 4-8x more threads would suggest — and the still-unfinished
Y-sweep loop-order restructuring remains the other lever with headroom, per
round 1's original conclusion. Untried: thread affinity (`OMP_PROC_BIND`/
`OMP_PLACES`), which round 1 flagged as a free, zero-risk experiment and
this round didn't get to either.

## Performance (round 1): strong scaling is poor, and the roofline predicted why

`scripts/benchmark.py --binary navsolver_omp --threads 1,2,4,6,8,12`
(`AbruptExpansion`, 50 steps, min of 5 repeats, three grid sizes):

| threads | small speedup | medium speedup | large speedup |
|---|---|---|---|
| 1  | 1.00x | 1.00x | 1.00x |
| 2  | 1.20x | 1.25x | 1.27x |
| 4  | 1.00x | 1.03x | 1.17x |
| 6  | 0.99x | 1.01x | 1.14x |
| 8  | 1.03x | 1.00x | 1.11x |
| 12 | 0.53x | 0.91x | 1.02x |

![OpenMP strong scaling](../experiments/figures/openmp_scaling.png)

**Peak speedup is ~1.2-1.3x at 2 threads, then flat-to-declining — by 12
threads the small grid is actually *slower* than single-threaded.** This
isn't a bug (determinism and equivalence both check out); it's a real
performance ceiling, and `docs/roofline.md` predicted almost exactly this
shape before any of this parallelization work started: memory bandwidth
on this machine saturates almost immediately past ~2 threads (22→34 GB/s
measured going from 1 to 12 threads in the roofline microbenchmark). If
`computeAccelerations` is memory/cache-bound (the roofline's tentative
verdict), adding threads beyond the point where bandwidth is already
saturated can't help — and the decline beyond that point suggests it
actively hurts, plausibly from a combination of: (a) `mirrorGhostCells`
remaining serial, so its fixed cost becomes a proportionally larger
Amdahl's-law tax as the parallel portions get more threads without
shrinking that serial chunk; (b) per-call OpenMP parallel-region overhead
(thread spawn/join/barrier), paid fresh on every `computeAccelerations`/
`solvePressurePoisson` call (every timestep); (c) this CPU's heterogeneous
P-core/E-core mix (Meteor Lake) — 12 "threads" aren't 12 uniform cores,
and scheduling across that mix isn't investigated further here. Not
diagnosed further with hard numbers since there's no working per-phase
timing breakdown or hardware counters in this environment (see
`docs/roofline.md`'s methodology section) to separate these causes
cleanly — flagged as the natural next question rather than guessed at.

**Takeaway for the project's broader goal**: on *this* machine, OpenMP is
not the lever that gets the biggest win — the roofline's other finding
(computeAccelerations achieves only ~20% of FMA peak, ~45% of its exp()
ceiling) suggests there's more headroom in single-thread efficiency
(the still-unfinished Y-sweep loop-order fix from
`docs/serial-optimization-loop-order.md`, and revisiting whether
`mirrorGhostCells` can be made both correct and parallel) than in adding
threads to code that's already bandwidth-constrained at 2 threads. Real
multi-core wins would need either a genuinely different memory access
pattern (reducing bytes moved per cell, not just distributing the same
bytes across cores) or hardware with more memory bandwidth per core than
this machine has.
