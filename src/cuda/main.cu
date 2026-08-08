// =============================================================================
//  main.cu  —  NavSolver CUDA entry point.
//
//  Mirrors src/serial/main.cpp's driver structure (same two flow types,
//  same logging/CSV/VTK cadence) but every per-step physics call runs on
//  the GPU via the *Cuda wrappers in Physics.cu. Field data stays device-
//  resident for the WHOLE per-step loop; the only host round-trips are (a)
//  a full field download when a VTK snapshot is due, and (b) tiny scalar
//  downloads (ResidMax/RMS, DilMax, dt) needed for the log line and CSV —
//  see docs/cuda-port.md for why that's the deliberate scope boundary.
//
//  Usage: identical to the other two binaries.
//    ./navsolver_cuda [config.cfg]
// =============================================================================

#include "SimState.hpp"
#include "Physics.hpp"
#include "VtkExporter.hpp"
#include "ConfigParser.hpp"
#include "Logger.hpp"
#include "DeviceState.cuh"

#include <iostream>
#include <iomanip>
#include <filesystem>
#include <stdexcept>

// ── CUDA physics wrappers (Physics.cu) ──────────────────────────────────────
void computeAccelerationsCuda(DeviceState& d);
void buildPressureSourceCuda(DeviceState& d, double timeStepSize);
void solvePressurePoissonCuda(DeviceState& d);
void updateVelocitiesCuda(DeviceState& d, bool useHalfStep, double timeStepSize);
void computeMomentumResidualCuda(DeviceState& d, double& residMax, double& residRMS);
void computeDivergenceCuda(DeviceState& d, double& dilatationMax, double& intDivergence, double& intAbsDivergence);
void adaptTimeStepCuda(DeviceState& d, double& timeStepSize);

// ── RK4 bookkeeping kernels (elementwise, full grid) ────────────────────────
__global__ void rk4SaveKernel(DeviceState d, double* velX0, double* velY0, double* velZ0,
                               double* Ku, double* Kv, double* Kw) {
    const long long tid = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (tid >= d.fieldLen) return;
    velX0[tid] = d.velX[tid]; velY0[tid] = d.velY[tid]; velZ0[tid] = d.velZ[tid];
    Ku[tid] = 0.0; Kv[tid] = 0.0; Kw[tid] = 0.0;
}

__global__ void rk4RestoreKernel(DeviceState d, const double* velX0, const double* velY0, const double* velZ0) {
    const long long tid = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (tid >= d.fieldLen) return;
    d.velX[tid] = velX0[tid]; d.velY[tid] = velY0[tid]; d.velZ[tid] = velZ0[tid];
}

// Accumulates over the SAME active domain as updateVelocitiesKernel
// (i:1..numCellsXm1, j:jLow[i]+1..jHigh[i]-1, k:1..KKfim) — see
// src/serial/main.cpp's runRK4 for the reference loop this mirrors.
__global__ void rk4AccumulateKernel(DeviceState d, const double* velX0, const double* velY0, const double* velZ0,
                                     double* Ku, double* Kv, double* Kw, double w) {
    const long long total = (long long)d.numCellsXm1 * (d.numCellsY + 1) * d.numCellsZ;
    const long long tid = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (tid >= total) return;
    const int i = 1 + (int)(tid / ((long long)(d.numCellsY + 1) * d.numCellsZ));
    const long long rem = tid % ((long long)(d.numCellsY + 1) * d.numCellsZ);
    const int j = (int)(rem / d.numCellsZ);
    const int k = 1 + (int)(rem % d.numCellsZ);
    const int KKfim = d.solidWall ? d.numCellsZm1 : d.numCellsZ;
    if (i > d.numCellsXm1 || j < d.jLow[i] + 1 || j > d.jHigh[i] - 1 || k > KKfim) return;

    const long long fi = d.idx(i, j, k);
    Ku[fi] += w * (d.velX[fi] - velX0[fi]);
    Kv[fi] += w * (d.velY[fi] - velY0[fi]);
    Kw[fi] += w * (d.velZ[fi] - velZ0[fi]);
}

__global__ void rk4FinalCombineKernel(DeviceState d, const double* velX0, const double* velY0, const double* velZ0,
                                       const double* Ku, const double* Kv, const double* Kw) {
    const long long total = (long long)d.numCellsXm1 * (d.numCellsY + 1) * d.numCellsZ;
    const long long tid = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (tid >= total) return;
    const int i = 1 + (int)(tid / ((long long)(d.numCellsY + 1) * d.numCellsZ));
    const long long rem = tid % ((long long)(d.numCellsY + 1) * d.numCellsZ);
    const int j = (int)(rem / d.numCellsZ);
    const int k = 1 + (int)(rem % d.numCellsZ);
    const int KKfim = d.solidWall ? d.numCellsZm1 : d.numCellsZ;
    if (i > d.numCellsXm1 || j < d.jLow[i] + 1 || j > d.jHigh[i] - 1 || k > KKfim) return;

    const long long fi = d.idx(i, j, k);
    d.velX[fi] = velX0[fi] + Ku[fi] / 6.0;
    d.velY[fi] = velY0[fi] + Kv[fi] / 6.0;
    d.velZ[fi] = velZ0[fi] + Kw[fi] / 6.0;
}

namespace {
constexpr int BLK = 256;
int gridForFull(long long n) { return (int)((n + BLK - 1) / BLK); }
}

// ---------------------------------------------------------------------------
//  Steady-state time marching
// ---------------------------------------------------------------------------
static void runSteady(SimState& s, DeviceState& d) {
    LOG_INFO("=== Steady-state marching (CUDA) ===");
    s.useHalfStep = false;

    do {
        adaptTimeStepCuda(d, s.timeStepSize);
        ++s.timeStep;
        s.simulationTime += s.timeStepSize;

        buildPressureSourceCuda(d, s.timeStepSize);
        solvePressurePoissonCuda(d);
        updateVelocitiesCuda(d, s.useHalfStep, s.timeStepSize);
        computeAccelerationsCuda(d);
        computeMomentumResidualCuda(d, s.momentumResidMax, s.momentumResidRMS);
        computeDivergenceCuda(d, s.dilatationMax, s.intDivergence, s.intAbsDivergence);

        LOG_INFO("step=", std::setw(6), s.timeStep,
                 "  t=",  std::fixed, std::setprecision(5), s.simulationTime,
                 "  ResidMax=", std::setprecision(3), std::scientific, s.momentumResidMax,
                 "  DilMax=",   s.dilatationMax,
                 "  dt=",       s.timeStepSize);

        VtkExporter::writeConvergenceCSV(s);

        if (s.timeStep % s.cfg.reportEveryN == 0) {
            downloadFields(d, s);
            VtkExporter::writeSnapshot(s, s.timeStep);
        }

    } while (s.momentumResidMax >= s.cfg.convergenceTol
             && s.timeStep < s.cfg.maxTimeSteps);

    LOG_INFO("Steady solve finished.  Steps=", s.timeStep,
             "  FinalResid=", s.momentumResidMax);
}

// ---------------------------------------------------------------------------
//  RK4 transient simulation — same four-stage structure as
//  src/serial/main.cpp's runRK4, with the save/restore/accumulate/combine
//  bookkeeping done as device kernels so the loop stays GPU-resident.
// ---------------------------------------------------------------------------
static void runRK4(SimState& s, DeviceState& d) {
    LOG_INFO("=== RK4 transient simulation (CUDA) ===");

    double *velX0, *velY0, *velZ0, *Ku, *Kv, *Kw;
    CUDA_CHECK(cudaMalloc(&velX0, d.fieldLen * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&velY0, d.fieldLen * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&velZ0, d.fieldLen * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&Ku, d.fieldLen * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&Kv, d.fieldLen * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&Kw, d.fieldLen * sizeof(double)));

    constexpr double rkWeights[4] = {1.0, 2.0, 2.0, 1.0};
    const long long activeTotal = (long long)d.numCellsXm1 * (d.numCellsY + 1) * d.numCellsZ;

    do {
        adaptTimeStepCuda(d, s.timeStepSize);
        ++s.timeStep;
        s.simulationTime += s.timeStepSize;

        rk4SaveKernel<<<gridForFull(d.fieldLen), BLK>>>(d, velX0, velY0, velZ0, Ku, Kv, Kw);

        for (int stage = 1; stage <= 4; ++stage) {
            rk4RestoreKernel<<<gridForFull(d.fieldLen), BLK>>>(d, velX0, velY0, velZ0);
            s.useHalfStep = (stage <= 2);

            buildPressureSourceCuda(d, s.timeStepSize);
            solvePressurePoissonCuda(d);
            updateVelocitiesCuda(d, s.useHalfStep, s.timeStepSize);

            rk4AccumulateKernel<<<gridForFull(activeTotal), BLK>>>(d, velX0, velY0, velZ0, Ku, Kv, Kw, rkWeights[stage - 1]);

            computeAccelerationsCuda(d);
        }

        rk4FinalCombineKernel<<<gridForFull(activeTotal), BLK>>>(d, velX0, velY0, velZ0, Ku, Kv, Kw);

        s.useHalfStep = false;
        // Final BCs (outlet + periodic) — reuse the same post-updateVelocities
        // BC kernels via a zero-effect updateVelocitiesCuda-style call would
        // recompute velocities; instead call the BC kernels directly through
        // a dedicated pass identical to src/serial/main.cpp's runRK4 tail.
        computeAccelerationsCuda(d);   // also applies periodic accel copy internally (zSweepKernel)
        computeMomentumResidualCuda(d, s.momentumResidMax, s.momentumResidRMS);
        computeDivergenceCuda(d, s.dilatationMax, s.intDivergence, s.intAbsDivergence);

        LOG_INFO("step=", std::setw(6), s.timeStep,
                 "  t=",  std::fixed, std::setprecision(5), s.simulationTime,
                 "  ResidMax=", std::setprecision(3), std::scientific, s.momentumResidMax,
                 "  DilMax=",   s.dilatationMax);

        VtkExporter::writeConvergenceCSV(s);

        if (s.timeStep % s.cfg.reportEveryN == 0) {
            downloadFields(d, s);
            VtkExporter::writeSnapshot(s, s.timeStep);
        }

    } while (s.momentumResidMax >= s.cfg.convergenceTol
             && s.timeStep < s.cfg.maxTimeSteps);

    LOG_INFO("RK4 transient finished.  Steps=", s.timeStep,
             "  FinalResid=", s.momentumResidMax);

    cudaFree(velX0); cudaFree(velY0); cudaFree(velZ0);
    cudaFree(Ku); cudaFree(Kv); cudaFree(Kw);
}

// ---------------------------------------------------------------------------
//  main
// ---------------------------------------------------------------------------
int main(int argc, char* argv[]) {
    Logger::instance().setLevel(Logger::Level::INFO);
    Logger::instance().setFile("navsolver.log");

    LOG_INFO("NavSolver (CUDA) — 3-D incompressible Navier-Stokes solver");
    LOG_INFO("Build: C++17/CUDA, UNIFAES scheme, fractional-step projection");

    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        LOG_ERROR("No CUDA device found.");
        return 1;
    }
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, 0);
    LOG_INFO("GPU: ", prop.name, "  (", prop.multiProcessorCount, " SMs, ",
             prop.totalGlobalMem / (1024 * 1024), " MiB VRAM) -- entry-level laptop dGPU, ",
             "absolute perf numbers are directional only, see docs/cuda-port.md");

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
        s.cfg.cellSizeX = s.cfg.domainLengthX / s.cfg.numCellsX;
        s.cfg.cellSizeY = s.cfg.domainLengthY / s.cfg.numCellsY;
        s.cfg.cellSizeZ = s.cfg.domainLengthZ / s.cfg.numCellsZ;
    }

    try {
        std::filesystem::create_directories(s.cfg.outputDir);
        ConfigParser::write(s.cfg.outputDir / (s.cfg.runName + ".cfg"), s.cfg);
    } catch (const std::exception& e) {
        LOG_WARN("Could not create output directory: ", e.what());
    }

    LOG_INFO("Grid: ", s.cfg.numCellsX, " x ", s.cfg.numCellsY, " x ", s.cfg.numCellsZ,
             "  Re=", s.cfg.reynoldsNumber);

    s.allocateFields();

    DeviceState d{};
    try {
        initSimulation(s);       // host-side geometry + IC setup (src/serial/Physics.cpp, reused as-is)
        computeAccelerations(s); // host reference pass -- NOT used further; establishes s fully before upload.
                                  // (Immediately recomputed on-device below so the GPU path never depends
                                  // on this CPU result beyond providing consistent uploaded state.)
        d = buildDeviceState(s);
        computeAccelerationsCuda(d);
    } catch (const std::exception& e) {
        LOG_ERROR("Initialisation failed: ", e.what());
        return 1;
    }

    downloadFields(d, s);
    VtkExporter::writeSnapshot(s, 0);
    VtkExporter::writeConvergenceCSV(s);

    try {
        if (s.cfg.flowType == FlowType::SteadyMarching)
            runSteady(s, d);
        else
            runRK4(s, d);
    } catch (const std::exception& e) {
        LOG_ERROR("Solver error at step ", s.timeStep, ": ", e.what());
        downloadFields(d, s);
        VtkExporter::writeSnapshot(s, s.timeStep);
        freeDeviceState(d);
        return 1;
    }

    downloadFields(d, s);
    VtkExporter::writeSnapshot(s, s.timeStep);

    LOG_INFO("=== Run complete ===");
    LOG_INFO("  Final time   : ", s.simulationTime);
    LOG_INFO("  Total steps  : ", s.timeStep);
    LOG_INFO("  ResidMax     : ", s.momentumResidMax);
    LOG_INFO("  DilMax       : ", s.dilatationMax);

    freeDeviceState(d);
    return 0;
}
