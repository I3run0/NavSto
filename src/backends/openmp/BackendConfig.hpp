#pragma once
// =============================================================================
//  BackendConfig.hpp (OpenMP) — this backend's storage and working memory.
//
//  See src/backends/serial/BackendConfig.hpp for the mechanism. Extras carries
//  what only this backend needs: one scratch slice per thread, and the
//  red-black row list its parallel pressure solve iterates.
// =============================================================================

#include "AccelScratch.hpp"
#include "GridField.hpp"
#include "KernelTimers.hpp"
#include "Rk4Workspace.hpp"
#include "RowIndex.hpp"

#include <omp.h>
#include <vector>

/// Shown in the startup banner; see backendName().
inline constexpr const char* kBackendName = "OpenMP";

/// Storage for every 3-D field in SimState. A NUMA first-touch variant used to
/// live beside this file; its resize() value-initialised serially before the
/// parallel loop ran, so every page was already faulted onto the master thread
/// and the "no measurable difference" it recorded was measuring nothing.
using Field = GridField<>;

/// Host-synchronous kernels, so the driver's own wall clock is valid here.
#define NAVSOLVER_DRIVER_TIME(kernel, call) NAVSOLVER_TIME(kernel, call)

/// Backend-private working memory, reachable as `s.ext`.
struct Extras {
    AccelScratch accel;

    /// Active (i,j) rows for the red-black pressure sweep; one row list drives
    /// BOTH colors (see Geometry.hpp). Geometry is fixed once initSimulation()
    /// has run, so solvePressurePoisson() fills it once on its first call.
    std::vector<RowIndex> activeRows;

    /// Allocated by backendStartup(), and only for RK4Transient runs.
    Rk4Workspace rk4;

    /// Called by SimState::allocateFields(), before initSimulation(). Only
    /// sizing happens here; activeRows needs geometry and is filled later.
    void allocate(const SimConfig& cfg) {
        accel.allocate(cfg, omp_get_max_threads());
    }
};
