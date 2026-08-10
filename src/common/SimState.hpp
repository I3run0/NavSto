#pragma once
// =============================================================================
//  SimState.hpp  —  Complete simulation state (replaces scattered globals).
//
//  Every physical field, grid parameter, and I/O handle lives here so that:
//    • functions declare their dependencies explicitly via (SimState&),
//    • multiple independent simulations can coexist, and
//    • the struct can be serialised / checkpointed later.
//
//  ── Per-backend assembly ───────────────────────────────────────────────────
//  Two of the pieces below are supplied by the backend being built, not fixed
//  here: `Field` (the storage behind every 3-D field) and `Extras` (that
//  backend's scratch and precomputed indices), both defined in its own
//  BackendConfig.hpp. Each solver target compiles with its own backend
//  directory on the include path, so this header resolves to a different --
//  better-fitting -- SimState in each binary.
//
//  That is the same compile-time backend selection the repo already uses to
//  decide which Physics.cpp defines the kernels, extended from the code to
//  the data it works on. It exists so a backend can be tuned in isolation:
//  changing how OpenMP stores its fields, or what scratch it keeps, is an
//  edit to src/openmp/BackendConfig.hpp alone -- no shared header changes, no
//  other backend affected, no `#ifdef` here. This header used to carry an
//  `#ifdef _OPENMP` and size its buffers by omp_get_max_threads(), and to
//  hold a red-black row list the serial solver left empty for every run.
//
//  Consequence to respect: SimState is deliberately a different type per
//  binary. Never link objects built against two different backend configs
//  into one executable.
// =============================================================================

#include "BackendConfig.hpp"
#include "GridField.hpp"
#include "SimConfig.hpp"

#include <algorithm>
#include <fstream>
#include <vector>


// ---------------------------------------------------------------------------
//  SimState  —  the full mutable state of a running simulation.
// ---------------------------------------------------------------------------
struct SimState {

    // ── Configuration (set before allocateFields()) ───────────────────────────
    SimConfig cfg;

    // ── Derived / cached grid sizes ────────────────────────────────────────────
    GridSize g;
    int numCellsXm1 = 0, numCellsYm1 = 0, numCellsZm1 = 0;
    double cellSizeXsq = 0.0, cellSizeYsq = 0.0, cellSizeZsq = 0.0;

    // ── Geometry boundary index arrays ─────────────────────────────────────────
    std::vector<int> iLow, iHigh;   // leftmost/rightmost active i for row j
    std::vector<int> jLow, jHigh;   // lowest/highest active j for column i

    int degreeIndex1  = 0;
    int degreeIndex2  = 0;
    int degreeIndexY  = 0;
    int jLowInitial   = 0;
    int jHighFinal    = 0;
    int rampIndexX1   = 0;
    int rampIndexX2   = 0;
    int rampIndexY1   = 0;
    int rampIndexY2   = 0;
    int numActiveCells = 0;

    // ── Time integration state ─────────────────────────────────────────────────
    int    timeStep      = 0;
    int    midPlaneZ     = 0;
    int    iResidMax = 0, jResidMax = 0, kResidMax = 0;
    int    iDilMax   = 0, jDilMax   = 0, kDilMax   = 0;
    int    counter   = 0;
    bool   useHalfStep = false;   // true during RK sub-steps 1 & 2

    double timeStepSize    = 0.0;
    double simulationTime  = 0.0;

    double maxVelocityChange   = 0.0;
    double momentumResidMax    = 0.0;
    double momentumResidRMS    = 0.0;
    double dilatationMax       = 0.0;
    double intDivergence       = 0.0;
    double intAbsDivergence    = 0.0;
    double initialPressureGradX = 0.0;
    double initialPressureGradY = 0.0;
    double uMaxAtInlet  = 0.0;
    double vMaxAtInlet  = 0.0;
    double hyperViscousDecay = 0.0;

    // ── 3-D field arrays ───────────────────────────────────────────────────────
    Field velX;            ///< u — x-velocity
    Field velY;            ///< v — y-velocity
    Field velZ;            ///< w — z-velocity
    Field press;           ///< p — pressure
    Field accelX;          ///< Au — UNIFAES acceleration on u
    Field accelY;          ///< Av — UNIFAES acceleration on v
    Field accelZ;          ///< Aw — UNIFAES acceleration on w
    Field pressureSource;  ///< s  — RHS of pressure Poisson
    Field scratchField;    ///< temporary (stream func., residuals…)

    // ── Backend-private working memory ─────────────────────────────────────────
    // Scratch buffers, precomputed index lists — whatever THIS backend needs
    // and no other one has to know about. Defined in BackendConfig.hpp.
    Extras ext;

    // ── Log / output ───────────────────────────────────────────────────────────
    std::ofstream logFile;

    // ── Memory management ──────────────────────────────────────────────────────
    /// Must be called once cfg.numCells* are set.
    void allocateFields() {
        // The physical domain is indices 0..numCells{X,Y,Z} (size
        // numCells+1), but the Gauss-Seidel pressure solve mirrors
        // Neumann boundary values one cell past the top of each axis
        // (e.g. s.press(numCellsX+1, j, k)) so the *next* sweep can read
        // a ghost value there. Allocate one extra ghost layer (+2, not
        // +1) so those writes land in valid, zero-initialized memory;
        // every loop elsewhere iterates the explicit 0..numCells range
        // and never sees the ghost layer.
        g.sI = cfg.numCellsX + 2;
        g.sJ = cfg.numCellsY + 2;
        g.sK = cfg.numCellsZ + 2;

        velX         .resize(g);
        velY         .resize(g);
        velZ         .resize(g);
        press        .resize(g);
        accelX       .resize(g);
        accelY       .resize(g);
        accelZ       .resize(g);
        pressureSource.resize(g);
        scratchField .resize(g);

        iLow .assign(g.sJ, 0);
        iHigh.assign(g.sJ, 0);
        jLow .assign(g.sI, 0);
        jHigh.assign(g.sI, 0);

        // Sizing only — anything needing geometry (which initSimulation()
        // fixes later) is built by the backend after that call instead.
        ext.allocate(cfg);
    }
};
