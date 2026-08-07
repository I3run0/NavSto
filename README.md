# NavSolver — 3-D Incompressible Navier-Stokes Solver

Solves the 3-D incompressible Navier-Stokes equations on a staggered Cartesian
grid using the **UNIFAES exponential scheme** and an explicit **fractional-step
(projection) method**, with both steady-state marching and **RK4 transient** integrators.

## Build

```bash
# Release (optimised)
make

# Debug + AddressSanitizer
make debug

# Run with example config
./navsolver config_re100_expansion.cfg
```

Requires **g++ ≥ 9** (or clang++ ≥ 10) with C++17 support.  No external
libraries are needed.

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
├── src/                        # C++ code only
│   ├── common/                 # Physics kernels (ONCE), config, logging, VTK
│   ├── serial/                 # Serial CPU: AoS layout, simple loops
│   ├── openmp/                 # OpenMP: AoS layout, #pragma omp
│   ├── cuda/                   # CUDA: SoA layout, GPU kernels
│   └── mpi_cuda/               # MPI+CUDA: 1D decomposition, halo exchange
│
├── experiments/                # Self-contained experiments
│   ├── configs/                # All .cfg files (Re=100, Re=500, scaling, etc.)
│   ├── results/                # Output from C++ solvers (gitignored)
│   └── notebooks/              # Jupyter for loading results, plotting, comparison
│
├── tests/                      # Unit and integration tests
├── scripts/                    # Build and run utilities
└── docs/                       # Documentation
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
