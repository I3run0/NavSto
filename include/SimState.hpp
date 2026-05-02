#pragma once
// =============================================================================
//  SimState.hpp  —  Complete simulation state (replaces scattered globals).
//
//  Every physical field, grid parameter, and I/O handle lives here so that:
//    • functions declare their dependencies explicitly via (SimState&),
//    • multiple independent simulations can coexist, and
//    • the struct can be serialised / checkpointed later.
// =============================================================================

#include "GridField.hpp"

#include <string>
#include <vector>
#include <fstream>
#include <filesystem>

// ---------------------------------------------------------------------------
//  Condition enumerations — replaces fragile string comparisons like "prd".
// ---------------------------------------------------------------------------
enum class GeometryType   { Axial, Curved };
enum class GeometryShape  { AbruptExpansion, AbruptContraction, OpenCavity,
                            GradualExpansion, UnilateralExpansion,
                            GradualContraction, UnilateralContraction,
                            SharpCorner, RoundedCorner };
enum class OutletBC       { ZeroFirstDeriv, ZeroSecondDeriv };
enum class LateralBC      { Periodic, SolidWall };
enum class InitialProfile { InletProfile, PotentialFlow };
enum class FlowType       { SteadyMarching, RK4Transient };

// ---------------------------------------------------------------------------
//  Configuration  —  all user-settable parameters in one plain-old-data pod.
//  These are filled from a config file or hard-coded defaults before INIT().
// ---------------------------------------------------------------------------
struct SimConfig {
    // Grid
    int numCellsX  = 240;   // II = 6*NN, NN=40
    int numCellsY  = 80;    // JJ = 2*NN
    int numCellsZ  = 40;    // KK = NN
    int baseUnit   = 40;    // NN

    // Physical domain dimensions (Cmp x Alt x Lrg in original)
    double domainLengthX = 6.0;   // Cmp
    double domainLengthY = 2.0;   // Alt
    double domainLengthZ = 1.0;   // Lrg

    // Cell sizes — derived from domain/grid; set automatically if left 0
    double cellSizeX = 0.0;   // dx = domainLengthX / numCellsX
    double cellSizeY = 0.0;   // dy = domainLengthY / numCellsY
    double cellSizeZ = 0.0;   // dz = domainLengthZ / numCellsZ

    // Physics
    double reynoldsNumber  = 9600.0;
    double hyperViscousRe  = 20.0;
    int    hyperViscousStart = 0;

    // Time integration
    int    maxTimeSteps    = 10000;
    int    reportEveryN    = 1000;
    double convergenceTol  = 1e-6;

    // Boundary / geometry conditions  — matches NavSto_dynamic.cpp main()
    GeometryType   geometryType    = GeometryType::Curved;           // TipoGeometria = "Crv"
    GeometryShape  geometryShape   = GeometryShape::RoundedCorner;   // Geometria     = "Cam"
    OutletBC       outletCondition = OutletBC::ZeroFirstDeriv;        // CondSaida     = "1d0"
    LateralBC      lateralCondition = LateralBC::Periodic;            // CondLater     = "prd"
    InitialProfile initialProfile   = InitialProfile::PotentialFlow;  // PerfilInicial = "LCP"
    FlowType       flowType         = FlowType::SteadyMarching;       // itp = 0

    // Output
    std::filesystem::path outputDir  = "results";
    std::string           runName    = "cam_re9600";
    int                   numPressureIter = 5;
};

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
    GridField<> velX;            ///< u — x-velocity
    GridField<> velY;            ///< v — y-velocity
    GridField<> velZ;            ///< w — z-velocity
    GridField<> press;           ///< p — pressure
    GridField<> accelX;          ///< Au — UNIFAES acceleration on u
    GridField<> accelY;          ///< Av — UNIFAES acceleration on v
    GridField<> accelZ;          ///< Aw — UNIFAES acceleration on w
    GridField<> pressureSource;  ///< s  — RHS of pressure Poisson
    GridField<> scratchField;    ///< temporary (stream func., residuals…)

    // ── Log / output ───────────────────────────────────────────────────────────
    std::ofstream logFile;

    // ── Memory management ──────────────────────────────────────────────────────
    /// Must be called once cfg.numCells* are set.
    void allocateFields() {
        g.sI = cfg.numCellsX + 1;
        g.sJ = cfg.numCellsY + 1;
        g.sK = cfg.numCellsZ + 1;

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
    }
};