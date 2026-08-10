#pragma once
// =============================================================================
//  SimConfig.hpp  —  User-settable parameters, and the enums they use.
//
//  Split out of SimState.hpp so it can be included without pulling in the
//  state itself. That matters because SimState is now assembled from a
//  per-backend BackendConfig.hpp, whose pieces (AccelScratch, and whatever
//  else a backend needs) are configured FROM this config -- including
//  SimState there instead would be circular.
//
//  It is also the right seam on its own terms: this is the pure value type
//  that must be identical across every backend for a benchmark comparison to
//  mean anything. Nothing here depends on how any backend stores its data.
// =============================================================================

#include <filesystem>
#include <string>

// ---------------------------------------------------------------------------
//  Condition enumerations — replaces fragile string comparisons like "prd".
// ---------------------------------------------------------------------------
enum class GeometryType   { Axial, Curved };
enum class GeometryShape  { AbruptExpansion, AbruptContraction, OpenCavity,
                            GradualExpansion, UnilateralExpansion,
                            GradualContraction, UnilateralContraction,
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
    // Grid
    int numCellsX  = 120;   // II = 6*NN, NN=40
    int numCellsY  = 40;    // JJ = 2*NN
    int numCellsZ  = 20;    // KK = NN
    int baseUnit   = 20;    // NN

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

    // Output
    std::filesystem::path outputDir  = "results";
    std::string           runName    = "cam_re9600";
    int                   numPressureIter = 5;

    // SOR relaxation factor for the pressure Poisson solve (1.0 = plain
    // Gauss-Seidel). 1.7 converges noticeably faster than 1.0 for this
    // grid/BC mix without the divergence risk of pushing closer to 2.0;
    // see docs/serial-optimization.md for how this was chosen/verified.
    double sorOmega = 1.7;
};
