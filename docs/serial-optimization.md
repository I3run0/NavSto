# Serial performance optimization pass

**Date:** 2026-08-07
**Machine:** Intel Core 7 150U, 6 cores / 12 threads, single socket/NUMA node (WSL2)
**Method:** `gprof` (software-instrumented; hardware perf counters — `perf`/LIKWID
— are not reliably available under this WSL2 environment), plus wall-clock
comparisons via clean `-O3` Release builds (never via the `-pg`-instrumented
binary, since `gprof`'s per-call instrumentation overhead distorts absolute
timing, especially for high-call-count functions).

Every change below was verified correct via `make test` (14/14) and
`python3 scripts/validate.py` (conservation blow-up checks + the analytical
Poiseuille check) before and after, and via AddressSanitizer on representative
configs. See `CHANGELOG.md` for the change list; this document is about the
*evidence*, not just the changes.

## Baseline profile

Workload: `AbruptExpansion`, 96×48×24 grid (~106k active cells), 100 steps,
`numPressureIter=5`. Built with `make profile` (`-O2 -g -pg` — `-O2` not
`-O3`, to keep function boundaries visible in the call graph; aggressive
inlining at `-O3` hides where time is actually spent).

| Function | % self time | self (s) | calls |
|---|---|---|---|
| `computeAccelerations` | 40.5% | 0.34 | 101 |
| `solvePressurePoisson` | 23.8% | 0.20 | 100 |
| `computeExponentialWeights` (called from `computeAccelerations`) | 13.1% | 0.11 | 60,318,513 |
| `updateVelocities` | 6.0% | 0.05 | 100 |
| `computeMomentumResidual` | 4.8% | 0.04 | 100 |
| `VtkExporter::writeSnapshot` | 4.8% | 0.04 | 2 |
| everything else (`buildPressureSource`, `computeDivergence`, `adaptTimeStep`, I/O, logging) | 6.9% | 0.06 | — |

**`computeAccelerations` + `solvePressurePoisson` account for ~64% of wall
time.** That set the priority order below. Two hypotheses from reading the
code *didn't* hold up under measurement, worth stating plainly rather than
quietly dropping:

- **Per-step CSV file open/close** (`VtkExporter::writeConvergenceCSV`
  opened+closed the file every timestep) attributed **0.00% self-time** in
  this profile. `gprof` samples CPU time, and blocking I/O syscalls mostly
  don't burn CPU cycles, so this wasn't visible here regardless of its real
  cost. Fixed anyway (see below) since it's a correctness-of-design issue
  independent of what one profile shows, and at `production_can.cfg`'s
  `maxTimeSteps=200000` the syscall count difference (200,000 vs 1) is large
  even if each syscall is fast.
- **Repeated heap allocation** of `computeAccelerations`'s 12 scratch
  `std::vector<double>` buffers didn't show up as a separate cost center in
  the call graph either — the allocations are small (~100 elements × 12,
  every call) and evidently fast enough on glibc's allocator not to register
  at `gprof`'s 0.01s sample granularity. Fixed anyway (moved to
  `SimState`-owned storage, allocated once) since it's free and safe, but
  the measured effect turned out to be modest — see below, and don't expect
  a large win from this class of fix in general.

## Changes made, in priority order

### 1. Gauss-Seidel → SOR in `solvePressurePoisson` (24% of baseline time)

Added `sorOmega` to `SimConfig` (default `1.7`); the pressure update became
`press += omega * (pNew - press)` (`omega=1.0` reduces to the original plain
Gauss-Seidel). Same per-cell cost as before plus one FMA; the actual benefit
is needing fewer sweeps for the same accuracy, since `numPressureIter` is a
**fixed** iteration count every step (not run-to-convergence).

Isolated, controlled test (single pressure solve, `AbruptExpansion`
48×24×12, comparing `DilMax` — the pressure solve's own target quantity —
at a fixed sweep budget):

| Method | Sweeps | `DilMax` |
|---|---|---|
| Gauss-Seidel (`omega=1.0`) | 20 | 0.04759 |
| SOR (`omega=1.7`) | **5** | **0.04701** |

**SOR matches (slightly beats) 20-sweep Gauss-Seidel accuracy in 5 sweeps —
a ~4x reduction in sweeps needed for equal accuracy**, on this config.

Two honest caveats found while checking this, not glossed over:

- **SOR's improvement isn't monotonic with more sweeps on this masked,
  irregular domain**: `omega=1.7` at 20 sweeps (`DilMax=0.04880`) is
  slightly *worse* than the same `omega` at 5 sweeps (`0.04701`). Classic
  SOR convergence theory assumes a simple rectangular domain; this solver's
  active-cell masking (stepped/curved geometry) means the theory doesn't
  apply exactly. The win at low sweep counts is real and reproducible; don't
  assume it keeps improving indefinitely with more sweeps.
- **A single-step test is not sufficient to validate a reduced
  `numPressureIter`.** `numPressureIter=2` with SOR looked fine in an
  isolated single-step comparison but **diverges over a real multi-step run**
  (`ResidMax` climbing from 0.65 at step 20 to 43 and still rising at step
  50, on the 96×48×24 config). `numPressureIter=3` is stable over 100 steps
  (bounded oscillation, same qualitative behavior as the original
  `numPressureIter=5`). Under-resolving the pressure Poisson equation
  compounds across steps in a way a single-step snapshot can't reveal —
  this is exactly the kind of thing `scripts/validate.py`'s conservation
  checks exist to catch, and it's why **`numPressureIter` was left
  unchanged in every shipped config** — only `sorOmega`'s default changed.
  Reducing `numPressureIter` per-config is real, available follow-up work,
  but needs the same full-run stability check per config/geometry, not a
  blanket reduction.

### 2. `computeAccelerations` scratch buffers moved to `SimState` (allocated once)

The 12 temporary `std::vector<double>` buffers (`ppie`, `ppiw`, `Ku`, `Kv`, ...)
are now `SimState` members, sized once in `allocateFields()` instead of
`std::vector`-constructed on every `computeAccelerations()` call (every
timestep, and multiple times per step for RK4 substeps). Verified safe to
reuse *without* re-zeroing between calls: the active index range each call
touches is fixed by geometry (set once in `initSimulation()`, static for the
whole run), and every entry in that range is written before it's read within
the same call — confirmed empirically too, since results are bit-identical
before/after this change on every test config.

### 3. `writeConvergenceCSV` keeps one file handle open for the run

Was `fs::exists()` + open + (implicit) close on every call. Now a
function-local `static std::ofstream`, opened once, with an explicit
`.flush()` after each write so the file stays readable by an external
tool tailing it mid-run (preserves that property without paying for a full
reopen/close every step).

## Real, measured wall-clock effect

Two comparisons, both Release builds (`make` / `-O3`, no `-pg`), min of 5
repeats, `AbruptExpansion`, 50 steps, against the pre-optimization commit
(`fbe4fdc`) as "before":

**(a) Matched iteration count** (`numPressureIter=5` both sides — isolates
just the three changes above, with SOR paying its per-cell cost but not
"cashing in" its faster-convergence benefit):

| Grid | before (s) | after (s) | speedup |
|---|---|---|---|
| tiny (24×12×6, ~1.6k cells) | 0.0136 | 0.0105 | **1.29x** |
| small (48×24×12, ~13k cells) | 0.0950 | 0.0744 | **1.28x** |
| medium (96×48×24, ~106k cells) | 0.6149 | 0.6449 | **0.95x** (slightly slower) |

At small problem sizes the allocation/CSV-I/O fixes win outright. At medium
size, SOR's added per-cell arithmetic (paid every sweep, at the *same*
sweep count) roughly cancels the other two fixes' savings — **this is the
honest matched-iteration-count picture, not a universal win.**

**(b) Retuned** (`numPressureIter=5→3`, verified stable over 100 steps —
see caveats above; this is a documented *recommendation*, not a change to
any shipped config):

| Grid | before, GS iter=5 (s) | after, SOR iter=3 (s) | speedup |
|---|---|---|---|
| tiny | 0.0136 | 0.0095 | **1.43x** |
| small | 0.0924 | 0.0679 | **1.36x** |
| medium | 0.6027 | 0.5741 | **1.05x** |

Retuning `numPressureIter` down (safely, with full-run stability
verification per config) is where the larger win actually lives — the code
changes alone are a modest, size-dependent win; realizing SOR's real benefit
needs that follow-up tuning step, deliberately not done here.

## Identified but not fixed: memory layout / loop order mismatch

`GridField` stores `(i,j,k)` with `k` fastest-varying (unit stride). In
`computeAccelerations`, the X-direction sweep loops `for(j) for(k) for(i)`
(`i` innermost, stride `sJ*sK`) and the Y-direction sweep loops
`for(i) for(k) for(j)` (`j` innermost, stride `sK`) — neither matches the
storage layout. Only the Z-direction sweep (`k` innermost) does.
`-march=native` confirmed AVX2+FMA are available on this CPU, but a loop
striding by thousands of elements per iteration gives the compiler nothing
to vectorize or prefetch effectively.

This plausibly explains why `computeAccelerations` remains the dominant
cost even after removing its allocation overhead — the real work per cell
looks small, but cache-unfriendly access inflates its effective cost. It is
**not fixed in this pass**: the temporary coefficient buffers (`ppie`,
`Ku`, ...) are recomputed per-`(j,k)`-plane through several sequential
passes over `i` that depend on each other *within* that plane, so this
isn't a simple loop-swap — it needs either a genuine restructuring of the
sweep (e.g. reorganizing around `(i,j)` with `k` innermost, which changes
how the per-plane multi-pass coefficient logic works) or a data-layout
change (trading which direction is cache-friendly), both of which carry
real risk of silently changing numerical results in a stencil this
intricate. This is the largest remaining optimization opportunity
identified in this pass, and the right next step before spending more
effort elsewhere — but it needs to be done carefully, with the
`scripts/validate.py` checks run after every intermediate step, not as a
single large rewrite.

## Reproducing this

```bash
make profile                          # -pg instrumented build
./navsolver <config>                  # generates gmon.out
gprof ./navsolver gmon.out | less     # flat profile + call graph

make bench                            # scripts/benchmark.py — timing baseline
make validate                         # scripts/validate.py — correctness gate
```
