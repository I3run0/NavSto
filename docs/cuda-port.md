# CUDA port

**Builds on:** [`docs/roofline.md`](roofline.md) (memory-bound verdict for
both hot kernels), [`docs/openmp-parallelization.md`](openmp-parallelization.md)
(the red-black restructuring this port reuses via
`src/common/RedBlackIndexing.hpp`, and — critically — the `mirrorGhostCells`
data race found there, which this port deliberately avoids re-introducing;
see below).

## Scope

Per the approved performance-study plan, this phase ports the *entire*
per-step loop to CUDA (`src/cuda/`), not just the two hot kernels — the
whole point of a GPU port is staying device-resident, and copying fields
back to the host every step just to run one leftover CPU function would
defeat that. Concretely, every function `src/serial/main.cpp`'s
`runSteady`/`runRK4` call has a device-kernel-backed equivalent in
`src/cuda/Physics.cu`:

| CPU (`src/serial/Physics.cpp`) | CUDA (`src/cuda/Physics.cu`) |
|---|---|
| `computeAccelerations` | `computeAccelerationsCuda` (3 sweep kernels) |
| `buildPressureSource` | `buildPressureSourceCuda` (2 boundary-zero kernels + 1 main kernel) |
| `solvePressurePoisson` | `solvePressurePoissonCuda` (mirror + red + mirror + black, per sweep) |
| `updateVelocities` | `updateVelocitiesCuda` (+ outlet-BC + periodic-copy kernels) |
| `computeMomentumResidual` | `computeMomentumResidualCuda` (kernel + `thrust` reduction) |
| `computeDivergence` | `computeDivergenceCuda` (kernel + `thrust` reduction) |
| `adaptTimeStep` | `adaptTimeStepCuda` (kernel + `thrust` reduction) |

`initSimulation` (geometry + initial conditions) is **not** ported — it's a
one-time, host-only setup cost, not a per-step hot path, so
`src/cuda/main.cu` links `src/serial/Physics.cpp` directly and calls the
existing host `initSimulation()` before uploading to the device. This is
safe: `Physics.cu`'s device-launching wrappers use `*Cuda`-suffixed names,
so there's no symbol clash with `Physics.cpp`'s CPU implementations of the
same seven functions — they're simply compiled in and never called on the
CUDA binary's path.

Host round-trips are limited to exactly two things, both intentional and
cheap: a full field download when a VTK snapshot is due (`reportEveryN`,
infrequent — every 500-1000+ steps by typical config), and the handful of
scalar downloads (`ResidMax`, `ResidRMS`, `DilMax`, `dt`) the log line and
convergence CSV need every step (a few 8-byte `thrust::reduce` results, not
a field-sized copy).

## Threading model

- **`computeAccelerations`** (X/Y/Z sweeps): one CUDA thread per **plane**,
  per the plan. Each sweep direction has a genuine sequential recurrence
  along its own axis (pass *n* depends on pass *n-1* at the neighboring
  index on that axis — nothing in the sweep body references the other two
  axes' neighbors), so that axis stays a normal `for` loop *inside* the
  thread; the other two axes parallelize across threads. Concretely: the
  X-sweep launches one thread per `(j,k)` pair, looping `i` internally; Y
  launches one thread per `(i,k)` pair, looping `j`; Z launches one thread
  per `(i,j)` pair, looping `k`. Each thread gets a private slice of the
  six per-sweep scratch buffers (`ppie/ppiw/qsie/Ku/Kv/Kw` and their Y/Z
  analogues) in global memory, sized `scratchLen` doubles each — this is
  the same "correct but unoptimized, in global memory" choice the plan
  calls for; nothing here uses shared memory yet (see Future work).
- **`solvePressurePoisson`**: one CUDA thread per active cell of one color
  (red or black), reusing `RedBlackIndexing.hpp`'s index lists **unchanged**
  — they're already built i-outer/j-middle/k-inner (compact, contiguous per
  color), which is the same ordering a GPU wants for coalescing-as-good-as-
  a-checkerboard-allows, exactly as that file's own comment anticipated.
  Sequential kernel-launch ordering on the default stream gives the
  inter-color barrier for free (no explicit sync needed): `mirror → red →
  mirror → black`, one iteration per sweep.
- **Everything else** (`buildPressureSource`, `updateVelocities`,
  `computeMomentumResidual`, `computeDivergence`, `adaptTimeStep`): one
  thread per cell over a generous flattened upper-bound domain with an
  in-kernel bounds check against the real (geometry-dependent) active
  range. Simple, correct, not the fastest possible occupancy — matches the
  "measure before you tune" discipline already used for the CPU roofline
  work.

## `mirrorGhostCells`: deliberately serial, again

This is the one place this port's design was **not** "port the parallel
structure and move on" — it required re-deriving the decision
`docs/openmp-parallelization.md` already made, and reaching the same
answer for a different reason.

Naively parallelizing `mirrorGhostCells` one-thread-per-`(i,k)` (mirroring
`RedBlackIndexing`'s natural granularity) reproduces the **exact race**
`docs/openmp-parallelization.md` found and fixed on the CPU: the `im`/`ip`
"cross-row" ghost writes and the `j`-boundary "same-row" writes can target
the same address from two different threads with different source values.
On CUDA this is actually a strictly *harder* problem than it was for
OpenMP: `schedule(static)` at least gives OpenMP threads contiguous,
timing-correlated iteration ranges; a CUDA kernel launch has **no**
ordering guarantee among threads at all. And there's no `cuda-memcheck
racecheck` run available in this environment to verify a fix the way one
normally would (same absent-tooling situation as the missing
ThreadSanitizer that shaped the OpenMP decision — see `docs/roofline.md`
for the parallel absent-`perf` situation).

A cross-row/same-row two-pass split (barrier via two separate kernel
launches, analogous to the fix sketched but not yet attempted for OpenMP)
is plausible in principle, but verifying it correct without a race
detector, under this phase's time budget, was judged not worth the risk of
shipping a silent GPU correctness bug. Instead, `mirrorGhostCellsKernel` is
launched as **`<<<1,1>>>`** — a single GPU thread executing the identical
control flow as the proven-correct serial CPU version, entirely on-device
(no D2H/H2D round trip, so GPU-residency is preserved). This is trivially
race-free (only one thread ever runs) and bit-for-bit equivalent to the
serial algorithm, at a real, measured performance cost — see below.

## Correctness verification

Per this project's standing rule (never trust a performance number before
correctness is verified), `scripts/validate_cuda.py` — structurally
mirroring `scripts/validate_parallel.py` — gates every result below:

1. **Red-black equivalence, gauge-fixed** (`navsolver` vs `navsolver_cuda`):
   same methodology as `scripts/validate.py`'s OpenMP tier 3 — `Straight`
   geometry, `numPressureIter=3000`, compare final pressure fields with
   each field's own mean subtracted first (the null-space caveat from
   `docs/openmp-parallelization.md` applies identically here).
   **Result: `L2_rel = 1.152e-03`** — matching the OpenMP red-black check's
   own `1.15e-3` almost to the last digit. That's a strong independent
   cross-check: two differently-implemented parallel backends (OpenMP
   threads vs CUDA threads), running the *same* red-black algorithm,
   converge to the same answer.
2. **Determinism**: `navsolver_cuda` run twice on an identical config must
   match near-bit-exactly. **Result: `L2_rel = 0.000e+00`** (exactly
   bit-identical) — every field write in every kernel here comes from
   exactly one thread by construction (see the threading model above), and
   `mirrorGhostCellsKernel`'s single-thread launch removes the one place
   that could plausibly have raced.
3. `make test` (15/15) and `python3 scripts/validate.py` (conservation,
   Poiseuille analytical check, OpenMP red-black tier) all still pass
   unmodified — this phase only *added* `src/cuda/` and a CMake target,
   touching nothing under `src/serial/`, `src/openmp/`, or `src/common/`
   except adding one `#include` of the already-shared
   `RedBlackIndexing.hpp` into the new `DeviceState.cuh`.

## Performance — honest result: slower than serial on this hardware

This is the part of this port that did **not** go the hoped-for direction,
reported in full per this project's no-cherry-picking discipline.

| Config (steady, `AbruptExpansion`, 50 steps) | serial | CUDA | ratio |
|---|---|---|---|
| 48×24×12, `numPressureIter=5` | 0.47 s | 4.52 s | **9.7× slower** |
| 96×48×24, `numPressureIter=5` (8× the cells) | 2.50 s | 14.8 s | **5.9× slower** |
| 48×24×12, `numPressureIter=50` (10× the sweeps) | 1.01 s | 18.8 s | **18.6× slower** |

Two things stand out:

- **Bigger grids get relatively less bad**, not relatively better in
  absolute terms — the 8×-larger grid's slowdown ratio improved from 9.7×
  to 5.9×, consistent with fixed per-launch overhead amortizing over more
  parallel work, exactly the mechanism discussed for OpenMP's own
  size-dependent scaling. But it's still solidly slower than serial at
  every size tested here.
- **More pressure sweeps make it dramatically worse**, disproportionately
  to serial's own slowdown: 10× the sweeps took serial from 0.47s→1.01s
  (2.1×) but CUDA from 4.52s→18.8s (4.2×) — CUDA's *relative* penalty
  roughly doubles when sweep count goes up 10×. This isolates the cause:
  `mirrorGhostCellsKernel` runs twice per sweep, `<<<1,1>>>`, entirely
  serially, on a single CUDA core — one of the weakest possible execution
  units for this branchy, low-arithmetic-intensity work, and its cost
  scales with sweep count exactly like the CUDA-vs-serial gap does. This
  matches (and sharpens) `docs/openmp-parallelization.md`'s own "Amdahl's
  law tax" hypothesis for `mirrorGhostCells` — on a GPU, a single serial
  thread is a far larger relative liability than a single serial CPU
  thread ever was in the OpenMP case, because the *parallel* portion
  (`updateColor`'s red/black kernels) is running on hardware built
  specifically to make that comparison worse, not better.

The hardware itself is also a real, disclosed factor: this machine's GPU
is an **NVIDIA GeForce MX570 A** — a 16-SM, 4 GiB, entry-level laptop dGPU
(Compute Capability 8.6), not remotely representative of the GPU hardware
this kind of workload is normally sized for (a real workstation/datacenter
GPU has an order of magnitude more SMs and bandwidth). VRAM was never a
constraint at any size tested here — the 48×24×12 validation config uses
roughly **6.7 MB** total device allocation (fields + per-thread scratch),
against 4096 MiB available.

## Future work (not attempted this phase)

- **Verify a parallel `mirrorGhostCells`.** The cross-row/same-row
  two-pass split sketched above is a real, evaluable idea — it just needs
  a race detector (`cuda-memcheck --tool racecheck` or equivalent) that
  wasn't available in this pass to verify before shipping. Given the
  measured cost above, this is very likely the single highest-value next
  step for CUDA performance specifically.
- **Shared memory for the sweep kernels.** Every sweep kernel currently
  reads/writes global memory for every access, including the per-thread
  scratch buffers — real global-memory-bound kernels, matching the
  project's "correct but unoptimized first" plan, revisit only with a
  GPU-side roofline once the above is fixed.
- **A real (non-entry-level) GPU.** Every number in this doc is directional
  only, from a laptop dGPU with a fraction of the SMs/bandwidth of
  workstation or datacenter hardware — the *relative* mirrorGhostCells
  finding should generalize, but the absolute serial-vs-CUDA crossover
  point will not.

## Build

CMake-only (not the Makefile) — deliberately asymmetric with the OpenMP
target, per the plan: CUDA needs `enable_language(CUDA)` and mixed
C++/CUDA linking that CMake handles natively and the hand-rolled Makefile
would have to reimplement. Uses `check_language(CUDA)` +
`enable_language(CUDA)` guarded exactly like `find_package(OpenMP)` is for
`navsolver_omp` — configure succeeds with or without a CUDA toolchain.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target navsolver_cuda
./build/navsolver_cuda config_re100_expansion.cfg

python3 scripts/validate_cuda.py          # correctness gate (builds serial + CUDA)
```

`CUDA_ARCHITECTURES 86` targets this machine's MX570 specifically — a
deliberately narrow choice (see `docs/roofline.md`/`docs/openmp-
parallelization.md` for the same "this machine, stated plainly" approach
to hardware-specific numbers), not a generic multi-arch build.
