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

/// Shown in the startup banner; see backendName().
inline constexpr const char* kBackendName = "CUDA";

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
