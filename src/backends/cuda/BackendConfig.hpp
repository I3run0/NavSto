#pragma once
// =============================================================================
//  BackendConfig.hpp (CUDA) — host staging only.
//
//  SimState here just stages the run on the host: initSimulation() fills it,
//  buildDeviceState() uploads it, and VTK snapshots download into it. The
//  per-step loop is device-resident, so no CPU kernel and no scratch. The
//  tuning surface for this backend is DeviceState.cuh.
// =============================================================================

#include "GridField.hpp"
#include "SimConfig.hpp"

/// Storage for every 3-D field in the host staging SimState.
using Field = GridField<>;

/// No host-side working memory: the per-step loop runs on the device.
struct Extras {
    void allocate(const SimConfig&) {}
};
