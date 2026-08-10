#pragma once
// =============================================================================
//  AccelScratch.hpp — computeAccelerations()'s coefficient working buffers.
//
//  These used to live in SimState, which forced two backend-specific things
//  into the shared state header: an `#ifdef _OPENMP` include of <omp.h>, and
//  an allocateFields() that sized every buffer by omp_get_max_threads().
//  Whether a backend wants per-thread slices, and how many, is a property of
//  that backend -- not of the simulation -- so the thread count is now a
//  parameter and each backend's Workspace passes its own (serial: 1).
//
//  Reuse without re-zeroing is safe: the active index range each call touches
//  is fixed by geometry (set once in initSimulation() and never changed), and
//  every entry in that range is written before it is read within the same
//  call. If that ever stops holding -- e.g. adaptive or moving geometry --
//  these need re-zeroing per call.
//
//  This struct is a convenience, not a contract: a backend that wants a
//  different layout (SoA splits, explicit alignment, NUMA-aware first touch)
//  should stop using it and put its own buffers in its own Workspace. Nothing
//  outside the backend's own kernels reads these.
// =============================================================================

#include "SimConfig.hpp"

#include <algorithm>
#include <cstddef>
#include <vector>

struct AccelScratch {
    // 1-D coefficient buffers (logical index -1..maxDim+1, offset by +1).
    // Ku/Kv/Kw are shared by the Y and Z sweeps; the X sweep uses its own
    // 2-D KuXK/KvXK/KwXK below.
    std::vector<double> ppin, ppis, ppiu, ppid;
    std::vector<double> qsin, qsiu, Ku, Kv, Kw;
    int scratchLenPerThread = 0;  ///< one thread's slice length in the buffers above

    // X-sweep-only 2-D (logical i, k) scratch, flattened per thread as
    // buf[tid*xk2DLenPerThread + iLogical*scratchKLen + (k-1)]. See
    // docs/serial-optimization-loop-order.md for why 2-D with k innermost
    // (matching GridField's storage order).
    std::vector<double> ppieXK, ppiwXK, qsieXK, KuXK, KvXK, KwXK;
    int scratchKLen = 0;          ///< k-stride within one thread's (i,k) slice
    int xk2DLenPerThread = 0;     ///< one thread's slice length in the XK buffers

    /// numThreads slices are allocated; every thread indexes its own via
    /// tid*scratchLenPerThread (concurrent threads writing a shared buffer
    /// would race). Slice lengths are rounded up to a 64-byte cache line so
    /// adjacent threads' slices don't share one at their boundary.
    void allocate(const SimConfig& cfg, int numThreads) {
        auto padTo8 = [](std::size_t n) { return (n + 7) & ~std::size_t{7}; };

        const int maxDim = std::max({cfg.numCellsX, cfg.numCellsY, cfg.numCellsZ});

        // maxDim+3, not maxDim+2: several writes reach logical index
        // maxDim+1 (the VM(Ku, iEnd+1)-style boundary extrapolations), which
        // needs physical slot maxDim+2.
        const std::size_t scratchLen = padTo8(static_cast<std::size_t>(maxDim) + 3);
        scratchLenPerThread = static_cast<int>(scratchLen);
        const std::size_t total = scratchLen * static_cast<std::size_t>(numThreads);
        ppin.assign(total, 0.0); ppis.assign(total, 0.0);
        ppiu.assign(total, 0.0); ppid.assign(total, 0.0);
        qsin.assign(total, 0.0); qsiu.assign(total, 0.0);
        Ku  .assign(total, 0.0); Kv  .assign(total, 0.0); Kw  .assign(total, 0.0);

        // i-dimension sized like the 1-D buffers (headroom for the +1
        // logical-index padding the VM() offset scheme needs); k dimension
        // covers k=1..numCellsZ, the periodic-case upper bound for KKfim.
        const std::size_t iLen = static_cast<std::size_t>(maxDim) + 3;
        scratchKLen = cfg.numCellsZ + 1;
        const std::size_t xk2DLen = padTo8(iLen * static_cast<std::size_t>(scratchKLen));
        xk2DLenPerThread = static_cast<int>(xk2DLen);
        const std::size_t totalXK = xk2DLen * static_cast<std::size_t>(numThreads);
        ppieXK.assign(totalXK, 0.0); ppiwXK.assign(totalXK, 0.0);
        qsieXK.assign(totalXK, 0.0);
        KuXK  .assign(totalXK, 0.0); KvXK  .assign(totalXK, 0.0); KwXK.assign(totalXK, 0.0);
    }
};
