// =============================================================================
//  Driver.cpp — entry point and time-marching loops. Shared by CPU backends.
//
//  Shared because it decides WHAT WORK GETS DONE — operator sequence, RK4
//  weights, convergence test, diagnostic cadence. Backends disagreeing on any
//  of that would stop their timings measuring the same computation. The kernels
//  it dispatches to are what differ, selected at link time.
//
//  A backend whose loop genuinely differs opts out instead of branching here;
//  src/backends/cuda/Driver.cu does exactly that.
//
//  Usage: ./navsolver [config.cfg]   (defaults if omitted)
//  Outputs: <outputDir>/<runName>_t*.vtk, _convergence.csv, .cfg; navsolver.log
// =============================================================================

#include "SimState.hpp"
#include "Physics.hpp"
#include "VtkExporter.hpp"
#include "ConfigParser.hpp"
#include "KernelTimers.hpp"
#include "Logger.hpp"

#include <iostream>
#include <iomanip>
#include <vector>
#include <filesystem>
#include <stdexcept>

// ---------------------------------------------------------------------------
//  Steady-state time marching  (flowType == SteadyMarching)
// ---------------------------------------------------------------------------
static void runSteady(SimState& s)
{
    LOG_INFO("=== Steady-state marching ===");
    s.useHalfStep = false;

    do {
        NAVSOLVER_TIME(TimeStep, adaptTimeStep(s));
        ++s.timeStep;
        s.simulationTime += s.timeStepSize;

        NAVSOLVER_TIME(PressSource, buildPressureSource(s));
        NAVSOLVER_TIME(PressSolve, solvePressurePoisson(s));
        NAVSOLVER_TIME(UpdateVel, updateVelocities(s));
        NAVSOLVER_TIME(Accel, computeAccelerations(s));
        NAVSOLVER_TIME(Residual, computeMomentumResidual(s));
        NAVSOLVER_TIME(Divergence, computeDivergence(s));

        LOG_INFO("step=", std::setw(6), s.timeStep,
                 "  t=",  std::fixed, std::setprecision(5), s.simulationTime,
                 "  ResidMax=", std::setprecision(3), std::scientific, s.momentumResidMax,
                 "  DilMax=",   s.dilatationMax,
                 "  dt=",       s.timeStepSize);

        VtkExporter::writeConvergenceCSV(s);

        if (s.timeStep % s.cfg.reportEveryN == 0)
            VtkExporter::writeSnapshot(s, s.timeStep);

    } while (s.momentumResidMax >= s.cfg.convergenceTol
             && s.timeStep < s.cfg.maxTimeSteps);

    LOG_INFO("Steady solve finished.  Steps=", s.timeStep,
             "  FinalResid=", s.momentumResidMax);
}

// ---------------------------------------------------------------------------
//  RK4 transient simulation  (flowType == RK4Transient)
//
//  u_{n+1} = u_n + (K1 + 2*K2 + 2*K3 + K4) / 6
//  Stages 1-2 use dt/2 (halfStep); stages 3-4 use dt.
// ---------------------------------------------------------------------------
static void runRK4(SimState& s)
{
    LOG_INFO("=== RK4 transient simulation ===");
    const GridSize& g = s.g;

    // `Field`, not GridField<>: these are read and written in the same
    // expressions as s.velX below, so a backend that re-lays-out its state
    // fields would otherwise leave half the RK4 working set on the old layout.
    Field velX0(g), velY0(g), velZ0(g);
    Field Ku(g),    Kv(g),    Kw(g);

    constexpr double rkWeights[4] = {1.0, 2.0, 2.0, 1.0};
    const int NX = s.cfg.numCellsX;
    const int KKfim = (s.cfg.lateralCondition == LateralBC::SolidWall)
                    ? s.numCellsZm1 : s.cfg.numCellsZ;

    do {
        NAVSOLVER_TIME(TimeStep, adaptTimeStep(s));
        ++s.timeStep;
        s.simulationTime += s.timeStepSize;

        // Save velocity at start of step
        for (int i = 0; i <= NX; ++i)
            for (int j = 0; j <= s.cfg.numCellsY; ++j)
                for (int k = 0; k <= s.cfg.numCellsZ; ++k) {
                    velX0(i,j,k) = s.velX(i,j,k);
                    velY0(i,j,k) = s.velY(i,j,k);
                    velZ0(i,j,k) = s.velZ(i,j,k);
                    Ku(i,j,k)    = 0.0;
                    Kv(i,j,k)    = 0.0;
                    Kw(i,j,k)    = 0.0;
                }

        // Four RK sub-steps
        for (int stage = 1; stage <= 4; ++stage) {
            // Restore u_n
            for (int i = 0; i <= NX; ++i)
                for (int j = 0; j <= s.cfg.numCellsY; ++j)
                    for (int k = 0; k <= s.cfg.numCellsZ; ++k) {
                        s.velX(i,j,k) = velX0(i,j,k);
                        s.velY(i,j,k) = velY0(i,j,k);
                        s.velZ(i,j,k) = velZ0(i,j,k);
                    }

            s.useHalfStep = (stage <= 2);

            NAVSOLVER_TIME(PressSource, buildPressureSource(s));
            NAVSOLVER_TIME(PressSolve, solvePressurePoisson(s));
            NAVSOLVER_TIME(UpdateVel, updateVelocities(s));

            // Accumulate weighted increment
            const double w = rkWeights[stage - 1];
            for (int i = 1; i <= s.numCellsXm1; ++i)
                for (int j = s.jLow[i]+1; j <= s.jHigh[i]-1; ++j)
                    for (int k = 1; k <= KKfim; ++k) {
                        Ku(i,j,k) += w * (s.velX(i,j,k) - velX0(i,j,k));
                        Kv(i,j,k) += w * (s.velY(i,j,k) - velY0(i,j,k));
                        Kw(i,j,k) += w * (s.velZ(i,j,k) - velZ0(i,j,k));
                    }

            NAVSOLVER_TIME(Accel, computeAccelerations(s));
        }

        // Final RK4 update: u_{n+1} = u_n + (K1+2K2+2K3+K4)/6
        for (int i = 1; i <= s.numCellsXm1; ++i)
            for (int j = s.jLow[i]+1; j <= s.jHigh[i]-1; ++j)
                for (int k = 1; k <= KKfim; ++k) {
                    s.velX(i,j,k) = velX0(i,j,k) + Ku(i,j,k) / 6.0;
                    s.velY(i,j,k) = velY0(i,j,k) + Kv(i,j,k) / 6.0;
                    s.velZ(i,j,k) = velZ0(i,j,k) + Kw(i,j,k) / 6.0;
                }

        // Apply final BCs after RK update
        s.useHalfStep = false;
        if (s.cfg.outletCondition == OutletBC::ZeroFirstDeriv)
            for (int j = s.jLow[NX]+1; j <= s.jHigh[NX]-1; ++j)
                for (int k = 0; k <= KKfim; ++k) {
                    s.velX(NX,j,k) = s.velX(s.numCellsXm1,j,k);
                    s.velY(NX,j,k) = s.velY(s.numCellsXm1,j,k);
                    s.velZ(NX,j,k) = s.velZ(s.numCellsXm1,j,k);
                }
        if (s.cfg.lateralCondition == LateralBC::Periodic)
            for (int i = 1; i <= s.numCellsXm1; ++i)
                for (int j = s.jLow[i]+1; j <= s.jHigh[i]-1; ++j) {
                    s.velX(i,j,0) = s.velX(i,j,s.cfg.numCellsZ);
                    s.velY(i,j,0) = s.velY(i,j,s.cfg.numCellsZ);
                    s.velZ(i,j,0) = s.velZ(i,j,s.cfg.numCellsZ);
                }

        NAVSOLVER_TIME(Accel, computeAccelerations(s));
        NAVSOLVER_TIME(Residual, computeMomentumResidual(s));
        NAVSOLVER_TIME(Divergence, computeDivergence(s));

        LOG_INFO("step=", std::setw(6), s.timeStep,
                 "  t=",  std::fixed, std::setprecision(5), s.simulationTime,
                 "  ResidMax=", std::setprecision(3), std::scientific, s.momentumResidMax,
                 "  DilMax=",   s.dilatationMax);

        VtkExporter::writeConvergenceCSV(s);

        if (s.timeStep % s.cfg.reportEveryN == 0)
            VtkExporter::writeSnapshot(s, s.timeStep);

    } while (s.momentumResidMax >= s.cfg.convergenceTol
             && s.timeStep < s.cfg.maxTimeSteps);

    LOG_INFO("RK4 transient finished.  Steps=", s.timeStep,
             "  FinalResid=", s.momentumResidMax);
}

// ---------------------------------------------------------------------------
//  main
// ---------------------------------------------------------------------------
int main(int argc, char* argv[])
{
    // ── Logger setup ─────────────────────────────────────────────────────────
    Logger::instance().setLevel(Logger::Level::INFO);
    Logger::instance().setFile("navsolver.log");

    LOG_INFO("NavSolver — 3-D incompressible Navier-Stokes solver");
    LOG_INFO("Build: C++17, UNIFAES scheme, fractional-step projection");

    // ── Load configuration ────────────────────────────────────────────────────
    SimState s;

    if (argc >= 2) {
        try {
            ConfigParser::parse(argv[1], s.cfg);
        } catch (const std::exception& e) {
            LOG_ERROR("Config error: ", e.what());
            return 1;
        }
    } else {
        LOG_INFO("No config file given — using built-in defaults.");
        // Defaults are already set in SimConfig's inline initialisers.
        // Cell sizes: dx = Cmp/II, dy = Alt/JJ, dz = Lrg/KK
        s.cfg.cellSizeX = s.cfg.domainLengthX / s.cfg.numCellsX;
        s.cfg.cellSizeY = s.cfg.domainLengthY / s.cfg.numCellsY;
        s.cfg.cellSizeZ = s.cfg.domainLengthZ / s.cfg.numCellsZ;
    }

    // ── Create output directory ───────────────────────────────────────────────
    try {
        std::filesystem::create_directories(s.cfg.outputDir);
        ConfigParser::write(s.cfg.outputDir / (s.cfg.runName + ".cfg"), s.cfg);
    } catch (const std::exception& e) {
        LOG_WARN("Could not create output directory: ", e.what());
    }

    // ── Allocate memory & initialise ─────────────────────────────────────────
    LOG_INFO("Grid: ", s.cfg.numCellsX, " x ", s.cfg.numCellsY, " x ", s.cfg.numCellsZ,
             "  Re=", s.cfg.reynoldsNumber);
    LOG_INFO("Domain: ", s.cfg.domainLengthX, " x ", s.cfg.domainLengthY, " x ", s.cfg.domainLengthZ,
             "  dx=", s.cfg.cellSizeX, "  dy=", s.cfg.cellSizeY, "  dz=", s.cfg.cellSizeZ);

    s.allocateFields();

    try {
        initSimulation(s);
        computeAccelerations(s);
    } catch (const std::exception& e) {
        LOG_ERROR("Initialisation failed: ", e.what());
        return 1;
    }

    // Write t=0 snapshot
    VtkExporter::writeSnapshot(s, 0);
    VtkExporter::writeConvergenceCSV(s);

    // ── Time integration ──────────────────────────────────────────────────────
    try {
        if (s.cfg.flowType == FlowType::SteadyMarching)
            runSteady(s);
        else
            runRK4(s);
    } catch (const std::exception& e) {
        LOG_ERROR("Solver error at step ", s.timeStep, ": ", e.what());
        // Still write whatever we have
        VtkExporter::writeSnapshot(s, s.timeStep);
        return 1;
    }

    // ── Final output ──────────────────────────────────────────────────────────
    VtkExporter::writeSnapshot(s, s.timeStep);

    LOG_INFO("=== Run complete ===");
    LOG_INFO("  Final time   : ", s.simulationTime);
    LOG_INFO("  Total steps  : ", s.timeStep);
    LOG_INFO("  ResidMax     : ", s.momentumResidMax);
    LOG_INFO("  DilMax       : ", s.dilatationMax);
    LOG_INFO("  VTK files    → ", (s.cfg.outputDir / s.cfg.runName).string(), "_t*.vtk");
    LOG_INFO("  Convergence  → ", (s.cfg.outputDir / (s.cfg.runName + "_convergence.csv")).string());

#if NAVSOLVER_PROFILE_ENABLED
    // Profiling build only — see src/io/KernelTimers.hpp, in particular
    // why total runtime here is NOT comparable to a normal build's.
    {
        const auto csv = s.cfg.outputDir / (s.cfg.runName + "_kernels.csv");
        KernelProfile::instance().writeCsv(csv);

        std::ostringstream table;
        KernelProfile::instance().report(table);
        LOG_INFO(table.str());
        LOG_INFO("  Per-kernel   → ", csv.string());
    }
#endif

    return 0;
}