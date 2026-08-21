#pragma once
// =============================================================================
//  AccelScratch.hpp — computeAccelerations()'s coefficient buffers.
//
//  numThreads slices; each thread indexes its own via tid*scratchLenPerThread
//  (a shared buffer would race), padded to a cache line. Reuse without
//  re-zeroing is safe: geometry fixes the range each call touches, and every
//  entry in it is written before it is read.
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
    /// Only four (i,k) planes are needed. The fused sweeps produce the east
    /// coefficient and consume it in the same iteration -- it is a register --
    /// and the west coefficient is produced one iteration before it is read, so
    /// a single k-row of history suffices. Those two used to be full planes,
    /// which cost 2/6 of a 1.01 MB per-thread footprint; at 12 threads that put
    /// 12.16 MB against a ~12 MB L3 and inverted OpenMP scaling past 4 threads.
    std::vector<double> qsieXK, KuXK, KvXK, KwXK;
    std::vector<double> ppiwRow;  ///< west coefficient carried from i-1 (or j-1)

    /// Which acceleration cells computeAccelerations has to reset, from
    /// buildAccelResetPlan(). Empty until the first call builds it; geometry is
    /// not known yet when allocate() runs. Per column i, [zeroJLo, zeroJHi] is
    /// everything any sweep writes and [xJLo, xJHi] the part the X sweep
    /// assigns outright, so only the two j-runs outside it need zeroing.
    std::vector<int> zeroJLo, zeroJHi, xJLo, xJHi;

    /// One k-row of intermediate values, numThreads slices of rowLenPerThread.
    /// computeDivergence and computeMomentumResidual write their per-cell
    /// value here so the stencil can vectorise, then reduce it in order.
    std::vector<double> rowBuf;
    int rowLenPerThread = 0;
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
        qsieXK.assign(totalXK, 0.0);
        ppiwRow.assign(static_cast<std::size_t>(scratchKLen) * numThreads, 0.0);
        KuXK  .assign(totalXK, 0.0); KvXK  .assign(totalXK, 0.0); KwXK.assign(totalXK, 0.0);

        const std::size_t rowLen = padTo8(static_cast<std::size_t>(cfg.numCellsZ) + 2);
        rowLenPerThread = static_cast<int>(rowLen);
        rowBuf.assign(rowLen * static_cast<std::size_t>(numThreads), 0.0);
    }
};
