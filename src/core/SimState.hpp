#pragma once
// =============================================================================
//  SimState.hpp — full mutable state of a running simulation.
//
//  `Field` (field storage) and `Extras` (backend-private working memory) come
//  from the backend's own BackendConfig.hpp, resolved by include path — so
//  SimState is deliberately a different type in each binary, and a backend can
//  be retuned without touching a shared header.
//
//  Never link objects built against two different backend configs.
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

    /// Last i of buildInitialPressure's ramp; the x where the domain steps.
    /// The other shape indices (degreeIndexY, the ramp corners) are locals in
    /// Setup.cpp — nothing outside geometry construction ever read them.
    int degreeIndex1  = 0;
    int degreeIndex2  = 0;
    int numActiveCells = 0;   ///< reported in the startup banner

    // ── Time integration state ─────────────────────────────────────────────────
    int    timeStep      = 0;
    bool   useHalfStep = false;   // true during RK sub-steps 1 & 2

    double timeStepSize    = 0.0;
    double simulationTime  = 0.0;

    // Diagnostics the driver logs and VtkExporter writes. Anything computed
    // per cell but never read (the argmax indices, maxVelocityChange) is gone:
    // it was work inside the kernels these benchmarks time.
    double momentumResidMax    = 0.0;
    double momentumResidRMS    = 0.0;
    double dilatationMax       = 0.0;
    double intDivergence       = 0.0;
    double intAbsDivergence    = 0.0;
    double initialPressureGradX = 0.0;
    double initialPressureGradY = 0.0;
    double uMaxAtInlet  = 0.0;
    double vMaxAtInlet  = 0.0;

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
