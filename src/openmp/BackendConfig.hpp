#pragma once
// =============================================================================
//  BackendConfig.hpp (OpenMP) — this backend's tuning surface.
//
//  See src/serial/BackendConfig.hpp for the mechanism. In short: SimState.hpp
//  builds itself out of the two names below, and each solver target compiles
//  with its own backend directory on the include path, so this is the file to
//  edit when tuning THIS backend's memory -- no shared header changes, and no
//  other backend affected.
//
//  What OpenMP needs beyond serial, and why it used to leak into SimState:
//
//    * one acceleration-scratch slice PER THREAD -- concurrent threads
//      writing a shared buffer would race. The thread count is a property of
//      this backend, so it is chosen here; the shared header used to
//      #include <omp.h> behind an #ifdef and size every buffer by
//      omp_get_max_threads() for all backends at once.
//
//    * the red-black active-row list its parallel pressure solve iterates.
//      SimState carried that vector for every backend, and the serial solver
//      left it empty for the whole of every run.
//
//  Field is still the stock GridField: nothing has yet needed a different
//  layout here. When something does -- padding, an SoA split, NUMA-aware
//  first touch -- this one line is where it changes, and serial and CUDA
//  will not notice.
// =============================================================================

#include "AccelScratch.hpp"
#include "GridField.hpp"
#include "RowIndex.hpp"

#include <omp.h>
#include <vector>

/// Storage for every 3-D field in SimState.
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
