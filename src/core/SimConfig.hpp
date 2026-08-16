#pragma once
// =============================================================================
//  SimConfig.hpp — user-settable parameters and their enums.
//
//  Split from SimState so BackendConfig.hpp can configure itself from cfg
//  without a circular include. Also the value type that must stay identical
//  across backends for a comparison to mean anything.
// =============================================================================

#include <filesystem>
#include <string>

// ---------------------------------------------------------------------------
//  Condition enumerations — replaces fragile string comparisons like "prd".
// ---------------------------------------------------------------------------
enum class GeometryType   { Axial, Curved };
/// Only shapes Setup.cpp actually builds. Five more were declared and parsed
/// but fell through to the Straight case, so a run asking for them got a
/// plausible result for a geometry it never used.
enum class GeometryShape  { AbruptExpansion, AbruptContraction,
                            SharpCorner, RoundedCorner, Straight };
enum class OutletBC       { ZeroFirstDeriv, ZeroSecondDeriv };
enum class LateralBC      { Periodic, SolidWall };
enum class InitialProfile { InletProfile, PotentialFlow };
enum class FlowType       { SteadyMarching, RK4Transient };

// ---------------------------------------------------------------------------
//  Configuration  —  all user-settable parameters in one plain-old-data pod.
//  These are filled from a config file or hard-coded defaults before INIT().
// ---------------------------------------------------------------------------
struct SimConfig {
    // Grid — THIS RANK'S cells. Allocation and every loop bound come from
    // these, so a decomposed backend sets them to its own slab.
    int numCellsX  = 120;   // II = 6*NN, NN=40
    int numCellsY  = 40;    // JJ = 2*NN
    int numCellsZ  = 20;    // KK = NN
    int baseUnit   = 20;    // NN

    // The full domain, and this rank's offset into it. Geometry is defined in
    // global coordinates (Setup.cpp) and VTK reports global extents, so those
    // two read these rather than the local sizes above.
    //
    // A single-process backend leaves them alone: normalizeDecomposition()
    // sets global = local and origin = 0, which makes every use below
    // arithmetically identical to reading the local size directly.
    int globalNumCellsX = 0, globalNumCellsY = 0, globalNumCellsZ = 0;
    int originX = 0, originY = 0, originZ = 0;

    /// Fills the global/origin fields for the undecomposed case. Called by
    /// initSimulation() before geometry is built; idempotent, and a backend
    /// that has already set them keeps its values.
    void normalizeDecomposition() {
        if (globalNumCellsX == 0) globalNumCellsX = numCellsX;
        if (globalNumCellsY == 0) globalNumCellsY = numCellsY;
        if (globalNumCellsZ == 0) globalNumCellsZ = numCellsZ;
    }

    // Physical domain dimensions (Cmp x Alt x Lrg in original)
    double domainLengthX = 6.0;   // Cmp
    double domainLengthY = 2.0;   // Alt
    double domainLengthZ = 1.0;   // Lrg

    // Cell sizes — derived from domain/grid; set automatically if left 0
    double cellSizeX = 0.0;   // dx = domainLengthX / numCellsX
    double cellSizeY = 0.0;   // dy = domainLengthY / numCellsY
    double cellSizeZ = 0.0;   // dz = domainLengthZ / numCellsZ

    // Physics
    double reynoldsNumber  = 100.0;
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

    // Output. Defaults under experiments/results/, which .gitignore covers --
    // a run's VTK snapshots are regenerable and large (hundreds of MB for a
    // production grid), so the default path must not be one git will commit.
    std::filesystem::path outputDir  = "experiments/results";
    std::string           runName    = "cam_re9600";
    int                   numPressureIter = 5;

    // SOR relaxation factor for the pressure Poisson solve (1.0 = plain
    // Gauss-Seidel). 1.7 converges noticeably faster than 1.0 for this
    // grid/BC mix without the divergence risk of pushing closer to 2.0;
    // see docs/serial-optimization.md for how this was chosen/verified.
    double sorOmega = 1.7;
};
