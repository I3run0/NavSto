# NavSolver — 3-D Incompressible Navier-Stokes Solver

Solves the 3-D incompressible Navier-Stokes equations on a staggered Cartesian
grid using the **UNIFAES exponential scheme** and an explicit **fractional-step
(projection) method**, with both steady-state marching and **RK4 transient** integrators.

See [`CHANGELOG.md`](CHANGELOG.md) for notable changes, including known issues.

## Build

```bash
# Release (optimised)
make

# Debug + AddressSanitizer
make debug

# Run with example config
./navsolver config_re100_expansion.cfg

# Unit tests
make test

# Physics correctness checks (conservation + analytical Poiseuille check)
make validate
```

Requires **g++ ≥ 10** (or clang++ ≥ 12) with C++20 support (concepts — see
`src/core/FieldStorage.hpp`).  No external
libraries are needed.

A `CMakeLists.txt` is also provided (`cmake -S . -B build && cmake --build build
&& ctest --test-dir build`) and mirrors the Makefile's Release/Debug profiles
plus the `navsolver_tests` target.

## Benchmarking

```bash
make bench
# or directly:
python3 scripts/benchmark.py --sizes tiny,small,medium,large --steps 50 --repeats 3
```

Runs the solver across a fixed grid-size matrix with `convergenceTol=0` (so
every run does exactly `maxTimeSteps` of work — an apples-to-apples fixed
workload rather than "however many steps until convergence"), times
wall-clock execution (min of `--repeats`), and writes a CSV to
`experiments/results/benchmarks/` with `seconds/step` and a
grid-size-normalized `ns/(active-cell·step)` throughput metric. This is the
baseline the OpenMP/CUDA/MPI implementations should be compared against —
see [`CHANGELOG.md`](CHANGELOG.md) for the current serial baseline numbers
and known correctness caveats before optimizing further.

## Profiling

```bash
make profile                          # -O2 -pg instrumented build
./navsolver <config>                  # generates gmon.out
gprof ./navsolver gmon.out | less
```

`-O2` (not `-O3`) keeps function boundaries visible in the call graph —
`-O3`'s aggressive inlining hides where time is actually spent. Don't use
the `-pg` build to measure absolute wall-clock time — its per-call
instrumentation overhead distorts it, especially for high-call-count
functions; use `make bench` (a clean `-O3` build) for that instead. See
[`docs/serial-optimization.md`](docs/serial-optimization.md) for a full
profiling pass with real before/after numbers.

### Per-kernel timing

`gprof` answers "which function"; this answers "which of the seven per-step
operators, and by how much" — the question you actually ask when tuning one
backend.

```bash
make kprofile          # serial       (or: cmake -DNAVSOLVER_PROFILE=ON)
make kprofile-openmp   # OpenMP
./navsolver <config>   # table in the log + <runName>_kernels.csv
```

Off by default and zero-cost when off. Profiled **totals** are not comparable
to a normal build — the CUDA timer synchronizes per kernel, which serializes
work the GPU would overlap. The per-kernel **shares** are the point; take
end-to-end numbers from an uninstrumented build. See
[`src/io/KernelTimers.hpp`](src/io/KernelTimers.hpp).

## Documentation index

- [`docs/serial-optimization.md`](docs/serial-optimization.md) — gprof
  baseline, three applied fixes (SOR pressure solve, buffer reuse, CSV
  I/O), real before/after wall-clock numbers.
- [`docs/roofline.md`](docs/roofline.md) — is the code compute- or
  memory-bound? Empirically-measured peak ceilings (FMA, bandwidth,
  `exp()` throughput) vs. achieved performance for the two hottest kernels.
  ```bash
  source .venv/bin/activate && pip install matplotlib   # once, for the plot
  python3 scripts/roofline.py
  ```
- [`docs/serial-optimization-loop-order.md`](docs/serial-optimization-loop-order.md) —
  acted on the roofline verdict: restructured `computeAccelerations`'s
  X-sweep to match `GridField`'s memory layout. Golden-field regression
  test, before/after numbers, Y-sweep not yet done.
- [`docs/openmp-parallelization.md`](docs/openmp-parallelization.md) —
  red-black pressure solve (needed since plain Gauss-Seidel/SOR can't be
  parallelized as-is), per-thread scratch buffers, a real data race found
  and fixed during verification, and real strong-scaling numbers (peak
  ~1.2-1.3x at 2 threads, declining beyond — matches the roofline's
  memory-bandwidth-saturation prediction).
  ```bash
  make openmp                                           # navsolver_omp
  make validate-parallel                                # thread-count equivalence + determinism
  python3 scripts/benchmark.py --binary navsolver_omp --threads 1,2,4,6,12
  python3 scripts/plot_scaling.py <benchmark_csv>
  ```
- [`docs/cuda-port.md`](docs/cuda-port.md) — full CUDA port (not just the
  hot kernels — the whole per-step loop stays GPU-resident), one thread
  per plane for `computeAccelerations`, one thread per active cell of one
  color for the red-black pressure solve, `mirrorGhostCells` deliberately
  kept `<<<1,1>>>` (same race avoided as the OpenMP port). Correctness
  verified (`L2_rel=1.15e-3` vs serial, matching OpenMP's own result;
  bit-identical determinism); performance honestly reported as a **negative
  result** — 5.9×-18.6× slower than serial on this machine's entry-level
  GPU, dominated by the single-thread `mirrorGhostCells` kernel.
  ```bash
  cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
  cmake --build build --target navsolver_cuda
  python3 scripts/validate_cuda.py
  ```

## Correctness checks

```bash
make validate
# or directly:
python3 scripts/validate.py
```

Two tiers, both required to pass before trusting a performance number:

1. **Conservation/stability sanity checks** — runs a couple of representative
   configs and fails if `ResidMax`/`DilMax` ever go non-finite or blow up.
   Coarse by design (a crash/divergence detector for CI), not a substitute
   for #2.
2. **Analytical check (plane Poiseuille flow)** — `experiments/configs/poiseuille.cfg`
   uses `geometryShape=Straight` + `lateralBC=Periodic` +
   `initialProfile=InletProfile`, which seeds the exact closed-form solution
   `u(y) = 6·yNorm·(1-yNorm)·Umax` as the initial condition. A correct
   discretization should reproduce it almost exactly (residual near machine
   epsilon) for the whole run, since it's already the steady state — this
   tests whether the discrete operators are self-consistent with a known
   exact solution. It does *not* test convergence to that solution from an
   arbitrary start; a full grid-convergence study would be the next step for
   that.

## Usage

```
./navsolver [config.cfg]
```

Without a config file the solver runs with built-in defaults (Re=100,
48 × 24 × 24 abrupt-expansion channel, RK4 transient).

## Configuration

Edit `config_re100_expansion.cfg`.  All keys and allowed values:

| Key | Example | Description |
|-----|---------|-------------|
| `numCellsX/Y/Z` | `48 24 24` | Grid resolution |
| `reynoldsNumber` | `100.0` | Flow Reynolds number |
| `geometryShape` | `AbruptExpansion` | One of: `AbruptExpansion`, `AbruptContraction`, `SharpCorner`, `RoundedCorner` |
| `geometryType` | `Axial` | `Axial` or `Curved` |
| `lateralBC` | `Periodic` | `Periodic` or `SolidWall` |
| `outletBC` | `ZeroFirstDeriv` | `ZeroFirstDeriv` or `ZeroSecondDeriv` |
| `initialProfile` | `InletProfile` | `InletProfile` or `PotentialFlow` |
| `flowType` | `RK4Transient` | `SteadyMarching` or `RK4Transient` |
| `maxTimeSteps` | `10000` | Maximum number of time steps |
| `reportEveryN` | `500` | Write VTK snapshot every N steps |
| `convergenceTol` | `1e-6` | Stop when `ResidMax < tol` |
| `outputDir` | `results` | Directory for all outputs |
| `runName` | `navsolver_run` | Prefix for output filenames |

## Output Files (Industrial Standard)

| File | Format | Description |
|------|--------|-------------|
| `results/<runName>_t000000.vtk` | **VTK Legacy ASCII** | t=0 snapshot |
| `results/<runName>_t000500.vtk` | VTK Legacy ASCII | Snapshot at step 500 |
| `results/<runName>_convergence.csv` | CSV | Per-step diagnostics |
| `results/<runName>.cfg` | Config | Provenance snapshot |
| `navsolver.log` | Plain text | Timestamped run log |

### Opening in ParaView

1. `File → Open → results/<runName>_t*.vtk` (select the group)
2. Click **Apply**
3. Select `Velocity` in the colouring dropdown for velocity magnitude
4. Add `Filters → Glyph` for vector arrows
5. Add `Filters → Contour` on `Pressure` for isobars

### VTK Fields Exported

- **Velocity** (VECTORS) — u, v, w components
- **Pressure** (SCALARS)
- **VelocityMagnitude** (SCALARS) — |u|, useful for streamlines
- **MomentumResidual** (SCALARS) — local convergence indicator

## Project Structure

```
NavSolver/
├── src/
│   ├── core/                   # what a backend is built FROM: SimConfig,
│   │                           # SimState, GridField, FieldStorage, Physics.hpp
│   ├── solver/                 # shared physics + the program: Setup.cpp
│   │                           # (geometry/ICs), Driver.cpp (main + time loops)
│   ├── io/                     # ConfigParser, VtkExporter, Logger, KernelTimers
│   └── backends/               # each owns its layout, working set and kernels
│       ├── serial/             #   BackendConfig.hpp + Physics.cpp
│       ├── openmp/             #   + red-black rows, per-thread scratch
│       └── cuda/               #   + DeviceState, own Driver.cu
│
├── experiments/                # Self-contained experiments
│   ├── configs/                # All .cfg files (Re=100, Re=500, scaling, etc.)
│   ├── results/                # Output from C++ solvers (gitignored)
│   └── notebooks/              # Jupyter for loading results, plotting, comparison
│
├── tests/                      # Unit tests (header-only harness, no deps),
│                                # wired into `ctest` and `make test`
├── scripts/                    # benchmark.py (make bench), validate.py (make validate)
├── reference/                  # Legacy Pascal-derived C++ translation
│                                # (navsto_dynamic.cpp) and validation data,
│                                # kept for cross-checking the modernized solver
├── docs/                       # serial-optimization.md (profiling + tuning notes)
├── CHANGELOG.md                # Notable changes, Keep a Changelog format
└── .github/workflows/          # CI: build + unit tests + smoke test on push/PR
```

## Physics

- **Momentum**: explicit fractional-step (projection) method
- **Advection/Diffusion**: UNIFAES exponential scheme (Bernstein-Crank weights)
- **Pressure**: Gauss-Seidel solution of ∇²p = S
- **Time integration**: adaptive CFL, steady marching or RK4 transient

## Key Improvements over Original

| Aspect | Before | After |
|--------|--------|-------|
| Memory layout | Global flat arrays + `#define F3` macro | `GridField<T>` with `operator()`, bounds-checked in debug |
| Condition strings | `"prd"`, `"nao"`, `"Eab"` | Typed enums `LateralBC::Periodic`, etc. |
| Configuration | Hard-coded in `main()` | External `.cfg` file + `ConfigParser` |
| Output format | `.dat` text tables | **VTK Legacy** (ParaView-ready) + CSV convergence history |
| Error handling | None | `std::exception` + structured logger |
| Build system | None | `Makefile` with release / debug / sanitizer profiles |
| Compiler warnings | Unknown | `-Wall -Wextra -Wpedantic -Wshadow` |
