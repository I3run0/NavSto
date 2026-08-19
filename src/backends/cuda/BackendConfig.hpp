#pragma once
// =============================================================================
//  BackendConfig.hpp (CUDA) — host staging plus a handle to the device state.
//
//  SimState stages the run on the host: initSimulation() fills it,
//  backendStartup() uploads it, syncFieldsToHost() downloads into it. The
//  per-step loop is device-resident, so there is no host scratch here; the
//  tuning surface for this backend is DeviceState.cuh.
// =============================================================================

#include "GridField.hpp"
#include "SimConfig.hpp"

/// Namespace SimState (and so every operator taking it) lands in, which is what
/// makes linking against another backend's objects a link error instead of a
/// silent layout mismatch. See SimState.hpp.
#define NAVSOLVER_BACKEND_NS backend_cuda

/// Device arithmetic precision.
///
/// GA107-class consumer GPUs run FP64 at 1/64 of FP32, so a double-precision
/// solver competes on the GPU's weakest axis -- this machine's CPU measured
/// 183.9 GFLOP/s FP64 against the card's ~77, and the double build ran 0.63x
/// serial. Float is the default HERE and only here: the host backends measured
/// SLOWER in float (conversion cost exceeds the halved traffic), so they stay
/// double and the host SimState is untouched. Narrowing happens at the upload
/// and download boundary, which runs once per snapshot.
///
/// Build with -DNAVSOLVER_CUDA_REAL=double to restore the old behaviour.
#ifndef NAVSOLVER_CUDA_REAL
#define NAVSOLVER_CUDA_REAL float
#endif
using Real = NAVSOLVER_CUDA_REAL;

/// Shown in the startup banner; see backendName(). Carries the precision so a
/// validation script can size its tolerance off the binary instead of guessing.
inline constexpr const char* kBackendName =
    sizeof(Real) == 4 ? "CUDA (fp32)" : "CUDA (fp64)";

/// Storage for every 3-D field in the host staging SimState.
using Field = GridField<>;

/// Launches are async, so a host clock around one would time the launch, not
/// the work. The *Cuda wrappers time themselves with cudaEvents instead
/// (CudaKernelTimer.cuh), which is why the driver adds nothing here.
#define NAVSOLVER_DRIVER_TIME(kernel, call) \
    do {                                    \
        call;                               \
    } while (0)

/// Opaque here on purpose: the shared driver is compiled by the host compiler,
/// which cannot parse DeviceState's __host__ __device__ members.
struct DeviceState;

/// Backend-private working memory, reachable as `s.ext`.
struct Extras {
    /// Owned by backendStartup()/backendShutdown(); null until the former runs.
    DeviceState* dev = nullptr;

    /// Called by SimState::allocateFields(), before initSimulation(). Nothing
    /// to size: the device allocations need geometry and happen later.
    void allocate(const SimConfig&) {}
};
