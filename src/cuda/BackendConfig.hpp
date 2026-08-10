#pragma once
// =============================================================================
//  BackendConfig.hpp (CUDA) — this backend's HOST-side tuning surface.
//
//  See src/serial/BackendConfig.hpp for the mechanism.
//
//  Note what SimState is for on this path. The CUDA binary uses it only to
//  stage the simulation on the host: initSimulation() builds geometry and
//  initial conditions into it, buildDeviceState() uploads them, and from then
//  on the per-step loop is entirely device-resident (see src/cuda/main.cu and
//  docs/cuda-port.md). It is also the landing buffer for the field download
//  taken when a VTK snapshot is due.
//
//  So the tuning surface that matters for this backend is DeviceState.cuh,
//  not this file: device-side layout, block sizes and the per-cell red/black
//  lists live there. Host staging happens once per run and is not measured,
//  which is why Field here is just the stock GridField and Extras is empty --
//  a CPU per-step kernel is never compiled into this binary, so none of the
//  scratch the other backends keep has any use here.
// =============================================================================

#include "GridField.hpp"
#include "SimConfig.hpp"

/// Storage for every 3-D field in the host staging SimState.
using Field = GridField<>;

/// No host-side working memory: the per-step loop runs on the device.
struct Extras {
    void allocate(const SimConfig&) {}
};
