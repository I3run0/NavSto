#pragma once
// =============================================================================
//  RedBlackIndexing.hpp — precomputed red/black cell index lists for
//  checkerboard (red-black) SOR.
//
//  Plain Gauss-Seidel/SOR (src/serial/Physics.cpp's solvePressurePoisson)
//  has a genuine loop-carried dependency: each cell's update reads a
//  neighbor that was just written earlier in the same sweep. That can't be
//  correctly parallelized as-is. Red-black splits cells into two colors by
//  (i+j+k) parity; every cell's 6 face-neighbors are always the OTHER
//  color, so within one color, no cell's update depends on another cell of
//  the same color — updating a whole color is embarrassingly parallel.
//  Shared between the OpenMP and (planned) CUDA pressure-solve kernels —
//  built once here rather than re-derived per backend.
//
//  Geometry (s.iLow/iHigh/jLow/jHigh, s.degreeIndex2) is fixed after
//  initSimulation() runs and never changes for the rest of a run, so these
//  index lists only need to be built once, not per sweep or per timestep.
//
//  List ordering: i-outer, j-middle, k-inner (k fastest-varying, matching
//  GridField's storage) so consecutive list entries stay as close together
//  in memory as a checkerboard pattern allows. Within one (i,j) row a
//  single color is inherently stride-2 in k (that's what "checkerboard"
//  means), but ordering this way keeps same-row entries adjacent in the
//  list rather than scattered, and is the same ordering a GPU port would
//  want for coalescing-as-good-as-checkerboard-allows — see
//  docs/openmp-parallelization.md.
// =============================================================================

#include "SimState.hpp"

#include <cstddef>

/// Fills s.redCells / s.blackCells for solvePressurePoisson's active
/// interior domain (i in [1,numCellsX], k in [1,numCellsZ], j in the
/// per-i active span — identical bounds logic to solvePressurePoisson's
/// own loop, kept in sync with it by hand; see that function if this ever
/// needs updating). Call once after initSimulation(), reuse for the run —
/// idempotent guard is the caller's responsibility (see s.redBlackBuilt).
inline void buildRedBlackIndices(SimState& s) {
    const auto& cfg = s.cfg;
    s.redCells.clear();
    s.blackCells.clear();

    // Conservative capacity estimate to avoid reallocation during the
    // push_back loop below; the real active-cell count is smaller for a
    // masked (non-rectangular) domain, so this may over-reserve slightly.
    const std::size_t capacity =
        (static_cast<std::size_t>(cfg.numCellsX) *
         static_cast<std::size_t>(cfg.numCellsY) *
         static_cast<std::size_t>(cfg.numCellsZ)) / 2 + 1;
    s.redCells.reserve(capacity);
    s.blackCells.reserve(capacity);

    for (int i = 1; i <= cfg.numCellsX; ++i) {
        const int im = i - 1;
        int jLoopS, jLoopN;
        if      (s.jLow[im]  >= s.jLow[i])  jLoopS = s.jLow[i]+1;
        else                                  jLoopS = s.jLow[i];
        if      (s.jHigh[im] <= s.jHigh[i]) jLoopN = s.jHigh[i];
        else                                  jLoopN = s.jHigh[i]+1;
        if (i == s.degreeIndex2) { jLoopS = s.jLow[im]+1; jLoopN = s.jHigh[im]; }

        for (int j = jLoopS; j <= jLoopN; ++j) {
            for (int k = 1; k <= cfg.numCellsZ; ++k) {
                CellIndex c{i, j, k};
                if ((i + j + k) % 2 == 0) s.redCells.push_back(c);
                else                       s.blackCells.push_back(c);
            }
        }
    }
}
