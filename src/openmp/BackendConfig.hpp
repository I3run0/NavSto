#pragma once
// =============================================================================
//  BackendConfig.hpp (OpenMP) — this backend's storage and working memory.
//
//  See src/serial/BackendConfig.hpp for the mechanism. Extras carries what only
//  this backend needs: one scratch slice per thread, and the red-black row list
//  its parallel pressure solve iterates.
// =============================================================================

#include "AccelScratch.hpp"
#include "FirstTouchField.hpp"
#include "GridField.hpp"
#include "RowIndex.hpp"

#include <omp.h>
#include <vector>

/// Storage for every 3-D field in SimState. FirstTouchField (this directory)
/// is a contract-checked alternative, not selected: measured inside the
/// run-to-run variance on this single-socket machine. Swapping is this line.
using Field = GridField<>;

/// Backend-private working memory, reachable as `s.ext`.
struct Extras {
    AccelScratch accel;

    /// Active (i,j) rows for the red-black pressure sweep; one row list drives
    /// BOTH colors (see Geometry.hpp). Geometry is fixed once initSimulation()
    /// has run, so this is filled once by buildActiveRows() after that call --
    /// replacing a lazy "already built?" flag that lived in SimState.
    std::vector<RowIndex> activeRows;

    /// Called by SimState::allocateFields(), before initSimulation(). Only
    /// sizing happens here; activeRows needs geometry and is filled later.
    void allocate(const SimConfig& cfg) {
        accel.allocate(cfg, omp_get_max_threads());
    }
};
