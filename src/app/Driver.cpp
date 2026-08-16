// =============================================================================
//  Driver.cpp — the program. One entry point and one time-marching loop for
//  every backend.
//
//  It fixes WHAT WORK GETS DONE — operator sequence, RK4 weights, convergence
//  test, diagnostic cadence — so backend timings measure the same computation.
//  Everything a backend does differently comes in through Physics.hpp's
//  operators and Backend.hpp's hooks, both resolved at link time.
//
//  Usage: ./navsolver[_omp|_cuda] [config.cfg]   (defaults if omitted)
//  Outputs: <outputDir>/<runName>_t*.vtk, _convergence.csv, .cfg; navsolver.log
// =============================================================================

#include "SimState.hpp"
#include "Backend.hpp"
#include "Physics.hpp"
#include "VtkExporter.hpp"
#include "ConfigParser.hpp"
#include "KernelTimers.hpp"
#include "Logger.hpp"

#include <iostream>
#include <iomanip>
#include <filesystem>
#include <stdexcept>

// ---------------------------------------------------------------------------
//  Output gating.
//
//  reportEveryN = 0 means write nothing at all -- no VTK, no convergence CSV.
//  That is what a benchmark wants: writing the t=0 and final snapshots
//  unconditionally put ~38% of a timed run into ASCII I/O the thing under
//  measurement cannot affect (144x72x36, 50 steps: 1.44s of 3.71s).
// ---------------------------------------------------------------------------
static bool outputEnabled(const SimState& s) { return s.cfg.reportEveryN > 0; }

static bool snapshotDue(const SimState& s)
{
    return outputEnabled(s) && (s.timeStep % s.cfg.reportEveryN == 0);
}

/// Snapshot — the only place field data has to be host-readable.
static void writeSnapshot(SimState& s, int step)
{
    if (!outputEnabled(s)) return;
    syncFieldsToHost(s);
    VtkExporter::writeSnapshot(s, step);
}

static void writeConvergence(const SimState& s)
{
    if (!outputEnabled(s)) return;
    VtkExporter::writeConvergenceCSV(s);
}

// ---------------------------------------------------------------------------
//  Steady-state time marching  (flowType == SteadyMarching)
// ---------------------------------------------------------------------------
static void runSteady(SimState& s)
{
    LOG_INFO("=== Steady-state marching (", backendName(), ") ===");
    s.useHalfStep = false;

    do {
        NAVSOLVER_DRIVER_TIME(TimeStep, adaptTimeStep(s));
        ++s.timeStep;
        s.simulationTime += s.timeStepSize;

        NAVSOLVER_DRIVER_TIME(PressSource, buildPressureSource(s));
        NAVSOLVER_DRIVER_TIME(PressSolve, solvePressurePoisson(s));
        NAVSOLVER_DRIVER_TIME(UpdateVel, updateVelocities(s));
        NAVSOLVER_DRIVER_TIME(Accel, computeAccelerations(s));
        NAVSOLVER_DRIVER_TIME(Residual, computeMomentumResidual(s));
        NAVSOLVER_DRIVER_TIME(Divergence, computeDivergence(s));

        LOG_INFO("step=", std::setw(6), s.timeStep,
                 "  t=",  std::fixed, std::setprecision(5), s.simulationTime,
                 "  ResidMax=", std::setprecision(3), std::scientific, s.momentumResidMax,
                 "  DilMax=",   s.dilatationMax,
                 "  dt=",       s.timeStepSize);

        writeConvergence(s);

        if (snapshotDue(s))
            writeSnapshot(s, s.timeStep);

    } while (s.momentumResidMax >= s.cfg.convergenceTol
             && s.timeStep < s.cfg.maxTimeSteps);

    LOG_INFO("Steady solve finished.  Steps=", s.timeStep,
             "  FinalResid=", s.momentumResidMax);
}

// ---------------------------------------------------------------------------
//  RK4 transient simulation  (flowType == RK4Transient)
//
//  u_{n+1} = u_n + (K1 + 2*K2 + 2*K3 + K4) / 6
//  Stages 1-2 use dt/2 (halfStep); stages 3-4 use dt. The rk4* calls move the
//  data; this loop owns the sequence and the weights.
// ---------------------------------------------------------------------------
static void runRK4(SimState& s)
{
    LOG_INFO("=== RK4 transient simulation (", backendName(), ") ===");

    constexpr double rkWeights[4] = {1.0, 2.0, 2.0, 1.0};

    do {
        NAVSOLVER_DRIVER_TIME(TimeStep, adaptTimeStep(s));
        ++s.timeStep;
        s.simulationTime += s.timeStepSize;

        rk4Save(s);

        for (int stage = 1; stage <= 4; ++stage) {
            rk4Restore(s);
            s.useHalfStep = (stage <= 2);

            NAVSOLVER_DRIVER_TIME(PressSource, buildPressureSource(s));
            NAVSOLVER_DRIVER_TIME(PressSolve, solvePressurePoisson(s));
            NAVSOLVER_DRIVER_TIME(UpdateVel, updateVelocities(s));

            rk4Accumulate(s, rkWeights[stage - 1]);

            NAVSOLVER_DRIVER_TIME(Accel, computeAccelerations(s));
        }

        rk4Combine(s);

        s.useHalfStep = false;
        rk4ApplyFinalBCs(s);

        NAVSOLVER_DRIVER_TIME(Accel, computeAccelerations(s));
        NAVSOLVER_DRIVER_TIME(Residual, computeMomentumResidual(s));
        NAVSOLVER_DRIVER_TIME(Divergence, computeDivergence(s));

        LOG_INFO("step=", std::setw(6), s.timeStep,
                 "  t=",  std::fixed, std::setprecision(5), s.simulationTime,
                 "  ResidMax=", std::setprecision(3), std::scientific, s.momentumResidMax,
                 "  DilMax=",   s.dilatationMax);

        writeConvergence(s);

        if (snapshotDue(s))
            writeSnapshot(s, s.timeStep);

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

    LOG_INFO("NavSolver (", backendName(), ") — 3-D incompressible Navier-Stokes solver");
    LOG_INFO("Build: C++20, UNIFAES scheme, fractional-step projection");

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
        backendStartup(s);
        computeAccelerations(s);
    } catch (const std::exception& e) {
        LOG_ERROR("Initialisation failed: ", e.what());
        backendShutdown(s);
        return 1;
    }

    // Write t=0 snapshot
    writeSnapshot(s, 0);
    writeConvergence(s);

    // ── Time integration ──────────────────────────────────────────────────────
    try {
        if (s.cfg.flowType == FlowType::SteadyMarching)
            runSteady(s);
        else
            runRK4(s);
    } catch (const std::exception& e) {
        LOG_ERROR("Solver error at step ", s.timeStep, ": ", e.what());
        // Still write whatever we have
        writeSnapshot(s, s.timeStep);
        backendShutdown(s);
        return 1;
    }

    // ── Final output ──────────────────────────────────────────────────────────
    writeSnapshot(s, s.timeStep);

    LOG_INFO("=== Run complete ===");
    LOG_INFO("  Backend      : ", backendName());
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

    backendShutdown(s);
    return 0;
}
