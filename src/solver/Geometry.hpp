#pragma once
// =============================================================================
//  Geometry.hpp — derived-geometry queries over an initialised SimState.
//
//  Pure functions returning by value; whichever backend wants a result stores
//  it in its own Extras. The two j-range helpers were open-coded in eleven
//  places across the host backends and Setup.cpp; DeviceMath.cuh keeps its own
//  __device__ copies, which is the only duplication left.
// =============================================================================

#include "RowIndex.hpp"
#include "SimState.hpp"

#include <cstddef>
#include <vector>

/// Interior j-span of column i: the cells a kernel owns and writes.
inline void activeJRange(const SimState& s, int i, int& jS, int& jN) {
    const int im = i - 1;
    jS = (i != s.degreeIndex2) ? s.jLow[i]+1  : s.jLow[im]+1;
    jN = (i != s.degreeIndex2) ? s.jHigh[i]   : s.jHigh[im];
}

/// Wider j-span the pressure solve sweeps, covering the rows whose Neumann
/// ghost values it also has to mirror. Steps out by one wherever column i-1
/// is taller than column i, so the step in a non-rectangular domain is walled.
inline void mirrorJRange(const SimState& s, int i, int& jLoopS, int& jLoopN) {
    const int im = i - 1;
    jLoopS = (s.jLow[im]  >= s.jLow[i])  ? s.jLow[i]+1  : s.jLow[i];
    jLoopN = (s.jHigh[im] <= s.jHigh[i]) ? s.jHigh[i]   : s.jHigh[i]+1;
    if (i == s.degreeIndex2) { jLoopS = s.jLow[im]+1; jLoopN = s.jHigh[im]; }
}

/// Active-row list for solvePressurePoisson's interior domain, in the same
/// bounds as its own loop. Geometry is fixed once initSimulation() has run, so
/// callers build this once and reuse it; never call it before.
inline std::vector<RowIndex> buildActiveRows(const SimState& s) {
    const auto& cfg = s.cfg;
    std::vector<RowIndex> activeRows;

    // Over-reserves for a masked (non-rectangular) domain; the point is only
    // to keep the push_back loop from reallocating.
    activeRows.reserve(static_cast<std::size_t>(cfg.numCellsX) *
                       static_cast<std::size_t>(cfg.numCellsY) + 1);

    for (int i = 1; i <= cfg.numCellsX; ++i) {
        int jLoopS, jLoopN;
        mirrorJRange(s, i, jLoopS, jLoopN);
        for (int j = jLoopS; j <= jLoopN; ++j)
            activeRows.push_back(RowIndex{i, j});
    }

    return activeRows;
}
