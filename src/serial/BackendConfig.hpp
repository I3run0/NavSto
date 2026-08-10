#pragma once
// =============================================================================
//  BackendConfig.hpp (serial) — this backend's tuning surface.
//
//  SimState.hpp includes "BackendConfig.hpp" and builds itself out of the two
//  names below. Every backend directory provides its own copy, and each solver
//  target compiles with its own backend directory on the include path, so
//  `SimState` is a different (better-fitting) type in each binary. Same
//  compile-time selection the repo already uses to pick which Physics.cpp
//  defines the kernels -- extended from the code to the data it works on.
//
//  This is the file to edit when tuning THIS backend's memory:
//
//    Field   the storage behind every 3-D field. Swap in a padded, aligned,
//            SoA or first-touch-aware container and only this backend sees it
//            -- provided it still answers (i,j,k), resize() and data().
//    Extras  scratch and precomputed indices this backend needs. Members here
//            exist only in this backend's SimState; nothing shared, and no
//            other backend, has to know about them.
//
//  Serial needs nothing exotic: the stock GridField, and a single-threaded
//  slice of the acceleration scratch. It deliberately has no red-black row
//  list -- its solvePressurePoisson is a sequential Gauss-Seidel/SOR sweep
//  with a genuine loop-carried dependency and no use for one. The shared
//  SimState used to carry that list anyway, empty, for every serial run.
//
//  These types differ per binary by design. Do not link objects built against
//  two different backend configs into one executable.
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
