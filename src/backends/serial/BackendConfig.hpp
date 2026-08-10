#pragma once
// =============================================================================
//  BackendConfig.hpp (serial) — this backend's storage and working memory.
//
//  SimState.hpp builds itself from the two names below, and each target
//  compiles with its own backend directory on the include path, so SimState is
//  a different type per binary. Edit this file to retune this backend's memory;
//  no other backend reads it.
//
//  Never link objects built against two different backend configs.
// =============================================================================

#include "AccelScratch.hpp"
#include "GridField.hpp"

/// Storage for every 3-D field in SimState.
using Field = GridField<>;

/// Backend-private working memory, reachable as `s.ext`.
struct Extras {
    AccelScratch accel;

    /// Called by SimState::allocateFields(), before initSimulation().
    void allocate(const SimConfig& cfg) {
        accel.allocate(cfg, /*numThreads=*/1);
    }
};
