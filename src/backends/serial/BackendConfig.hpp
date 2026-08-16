#pragma once
// =============================================================================
//  BackendConfig.hpp (serial) — this backend's storage and working memory.
//
//  SimState.hpp builds itself from the names below, and each target compiles
//  with its own backend directory on the include path, so SimState is a
//  different type per binary. Edit this file to retune this backend's memory;
//  no other backend reads it.
// =============================================================================

#include "AccelScratch.hpp"
#include "GridField.hpp"
#include "KernelTimers.hpp"
#include "Rk4Workspace.hpp"

/// Namespace SimState (and so every operator taking it) lands in, which is what
/// makes linking against another backend's objects a link error instead of a
/// silent layout mismatch. See SimState.hpp.
#define NAVSOLVER_BACKEND_NS backend_serial

/// Shown in the startup banner; see backendName().
inline constexpr const char* kBackendName = "serial";

/// Storage for every 3-D field in SimState.
using Field = GridField<>;

/// The driver times its own calls here: these kernels run synchronously on the
/// host, so a wall clock around the call measures the work.
#define NAVSOLVER_DRIVER_TIME(kernel, call) NAVSOLVER_TIME(kernel, call)

/// Backend-private working memory, reachable as `s.ext`.
struct Extras {
    AccelScratch accel;

    /// Allocated by backendStartup(), and only for RK4Transient runs.
    Rk4Workspace rk4;

    /// Called by SimState::allocateFields(), before initSimulation().
    void allocate(const SimConfig& cfg) {
        accel.allocate(cfg, /*numThreads=*/1);
    }
};
