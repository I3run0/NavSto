// =============================================================================
//  CudaBackend.cu — Physics.hpp and Backend.hpp, implemented for the GPU.
//
//  The shims below are the whole reason the shared driver works here: they
//  give the *Cuda wrappers in Physics.cu the SimState& signatures the driver
//  calls, keeping field data device-resident across the step. Only two things
//  cross back to the host — a full download when a VTK snapshot is due, and
//  the scalars the log line and CSV need. See docs/cuda-port.md.
// =============================================================================

#include "Backend.hpp"
#include "Physics.hpp"
#include "CudaKernelTimer.cuh"
#include "DeviceState.cuh"
#include "Logger.hpp"

#include <stdexcept>

// ── CUDA physics wrappers (Physics.cu) ──────────────────────────────────────
void computeAccelerationsCuda(DeviceState& d);
void buildPressureSourceCuda(DeviceState& d, double timeStepSize);
void solvePressurePoissonCuda(DeviceState& d);
void updateVelocitiesCuda(DeviceState& d, bool useHalfStep, double timeStepSize);
void computeMomentumResidualCuda(DeviceState& d, double& residMax, double& residRMS);
void computeDivergenceCuda(DeviceState& d, double& dilatationMax, double& intDivergence, double& intAbsDivergence);
void adaptTimeStepCuda(DeviceState& d, double& timeStepSize);

// ── Velocity BC kernels (Physics.cu) — also used by rk4ApplyFinalBCs ────────
__global__ void outletBCKernel(DeviceState d, bool zeroSecondDeriv);
__global__ void periodicCopyVelKernel(DeviceState d);

namespace {
constexpr int BLK = 256;
int gridFor(long long n) { return (int)((n + BLK - 1) / BLK); }

/// The device state the driver reaches through SimState. Null before
/// backendStartup(), which is the only place it is built.
DeviceState& dev(SimState& s) { return *s.ext.dev; }

/// Upper bound on the active domain, shared by the accumulate/combine kernels.
long long activeDomain(const DeviceState& d) {
    return (long long)d.numCellsXm1 * (d.numCellsY + 1) * d.numCellsZ;
}
}  // namespace

// ═══════════════════════════════════════════════════════════════════════════
//  Physics.hpp — one shim per operator
// ═══════════════════════════════════════════════════════════════════════════

void computeAccelerations(SimState& s) {
    NAVSOLVER_TIME_CUDA(Accel, computeAccelerationsCuda(dev(s)));
}

void buildPressureSource(SimState& s) {
    NAVSOLVER_TIME_CUDA(PressSource, buildPressureSourceCuda(dev(s), s.timeStepSize));
}

void solvePressurePoisson(SimState& s) {
    NAVSOLVER_TIME_CUDA(PressSolve, solvePressurePoissonCuda(dev(s)));
}

void updateVelocities(SimState& s) {
    NAVSOLVER_TIME_CUDA(UpdateVel, updateVelocitiesCuda(dev(s), s.useHalfStep, s.timeStepSize));
}

void computeMomentumResidual(SimState& s) {
    NAVSOLVER_TIME_CUDA(Residual,
        computeMomentumResidualCuda(dev(s), s.momentumResidMax, s.momentumResidRMS));
}

void computeDivergence(SimState& s) {
    NAVSOLVER_TIME_CUDA(Divergence,
        computeDivergenceCuda(dev(s), s.dilatationMax, s.intDivergence, s.intAbsDivergence));
}

void adaptTimeStep(SimState& s) {
    NAVSOLVER_TIME_CUDA(TimeStep, adaptTimeStepCuda(dev(s), s.timeStepSize));
}

// ═══════════════════════════════════════════════════════════════════════════
//  Backend.hpp — lifecycle
// ═══════════════════════════════════════════════════════════════════════════

const char* backendName() { return kBackendName; }

void backendStartup(SimState& s) {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0)
        throw std::runtime_error("no CUDA device found");

    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, 0);
    LOG_INFO("GPU: ", prop.name, "  (", prop.multiProcessorCount, " SMs, ",
             prop.totalGlobalMem / (1024 * 1024), " MiB VRAM) -- entry-level laptop dGPU, ",
             "absolute perf numbers are directional only, see docs/cuda-port.md");

    s.ext.dev = new DeviceState(buildDeviceState(s));

    if (s.cfg.flowType == FlowType::RK4Transient) {
        DeviceState& d = dev(s);
        const size_t bytes = d.fieldLen * sizeof(double);
        CUDA_CHECK(cudaMalloc(&d.rk4VelX0, bytes));
        CUDA_CHECK(cudaMalloc(&d.rk4VelY0, bytes));
        CUDA_CHECK(cudaMalloc(&d.rk4VelZ0, bytes));
        CUDA_CHECK(cudaMalloc(&d.rk4Ku, bytes));
        CUDA_CHECK(cudaMalloc(&d.rk4Kv, bytes));
        CUDA_CHECK(cudaMalloc(&d.rk4Kw, bytes));
    }
}

void backendShutdown(SimState& s) {
    if (!s.ext.dev) return;

    DeviceState& d = dev(s);
    cudaFree(d.rk4VelX0); cudaFree(d.rk4VelY0); cudaFree(d.rk4VelZ0);
    cudaFree(d.rk4Ku);    cudaFree(d.rk4Kv);    cudaFree(d.rk4Kw);
    freeDeviceState(d);

    delete s.ext.dev;
    s.ext.dev = nullptr;
}

void syncFieldsToHost(SimState& s) {
    if (s.ext.dev) downloadFields(dev(s), s);
}

// ═══════════════════════════════════════════════════════════════════════════
//  Backend.hpp — RK4 bookkeeping, as device kernels so the loop stays on GPU
// ═══════════════════════════════════════════════════════════════════════════

__global__ void rk4SaveKernel(DeviceState d) {
    const long long tid = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (tid >= d.fieldLen) return;
    d.rk4VelX0[tid] = d.velX[tid];
    d.rk4VelY0[tid] = d.velY[tid];
    d.rk4VelZ0[tid] = d.velZ[tid];
    d.rk4Ku[tid] = 0.0; d.rk4Kv[tid] = 0.0; d.rk4Kw[tid] = 0.0;
}

__global__ void rk4RestoreKernel(DeviceState d) {
    const long long tid = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (tid >= d.fieldLen) return;
    d.velX[tid] = d.rk4VelX0[tid];
    d.velY[tid] = d.rk4VelY0[tid];
    d.velZ[tid] = d.rk4VelZ0[tid];
}

/// Same active domain as updateVelocitiesKernel — i:1..numCellsXm1,
/// j:jLow[i]+1..jHigh[i]-1, k:1..KKfim.
__global__ void rk4AccumulateKernel(DeviceState d, double w) {
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
    d.rk4Ku[fi] += w * (d.velX[fi] - d.rk4VelX0[fi]);
    d.rk4Kv[fi] += w * (d.velY[fi] - d.rk4VelY0[fi]);
    d.rk4Kw[fi] += w * (d.velZ[fi] - d.rk4VelZ0[fi]);
}

__global__ void rk4CombineKernel(DeviceState d) {
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
    d.velX[fi] = d.rk4VelX0[fi] + d.rk4Ku[fi] / 6.0;
    d.velY[fi] = d.rk4VelY0[fi] + d.rk4Kv[fi] / 6.0;
    d.velZ[fi] = d.rk4VelZ0[fi] + d.rk4Kw[fi] / 6.0;
}

void rk4Save(SimState& s) {
    DeviceState& d = dev(s);
    rk4SaveKernel<<<gridFor(d.fieldLen), BLK>>>(d);
    CUDA_CHECK(cudaGetLastError());
}

void rk4Restore(SimState& s) {
    DeviceState& d = dev(s);
    rk4RestoreKernel<<<gridFor(d.fieldLen), BLK>>>(d);
    CUDA_CHECK(cudaGetLastError());
}

void rk4Accumulate(SimState& s, double weight) {
    DeviceState& d = dev(s);
    rk4AccumulateKernel<<<gridFor(activeDomain(d)), BLK>>>(d, weight);
    CUDA_CHECK(cudaGetLastError());
}

void rk4Combine(SimState& s) {
    DeviceState& d = dev(s);
    rk4CombineKernel<<<gridFor(activeDomain(d)), BLK>>>(d);
    CUDA_CHECK(cudaGetLastError());
}

/// Mirrors HostBackend.cpp's rk4ApplyFinalBCs: the outlet copy runs only under
/// ZeroFirstDeriv, so the zeroSecondDeriv branch of outletBCKernel stays unused
/// here even though updateVelocitiesCuda drives both.
void rk4ApplyFinalBCs(SimState& s) {
    DeviceState& d = dev(s);

    if (d.outletZeroFirstDeriv)
        outletBCKernel<<<gridFor((long long)(d.numCellsY + 2) * (d.numCellsZ + 2)), BLK>>>(
            d, /*zeroSecondDeriv=*/false);

    if (d.periodic)
        periodicCopyVelKernel<<<gridFor((long long)d.numCellsXm1 * (d.numCellsY + 1)), BLK>>>(d);

    CUDA_CHECK(cudaGetLastError());
}
