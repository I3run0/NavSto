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

#include <algorithm>
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

/// Per-column j-span covering every cell computeAccelerations can write: the
/// X sweep addresses the domain by rows (iLow/iHigh), the Y and Z sweeps by
/// columns (jLow/jHigh), and this is the min/max of both — a superset by
/// construction, so nothing depends on the two descriptions agreeing. Geometry
/// is fixed once initSimulation() has run; build this once and reuse it.
inline void buildAccelZeroSpans(const SimState& s,
                                std::vector<int>& jLo, std::vector<int>& jHi) {
    const auto& cfg = s.cfg;
    jLo.assign(cfg.numCellsX + 2, 1);
    jHi.assign(cfg.numCellsX + 2, 0);          // empty span where nothing writes

    for (int i = 1; i <= s.numCellsXm1; ++i) { // Y and Z sweeps, by column
        jLo[i] = s.jLow[i];
        jHi[i] = s.jHigh[i];
    }
    for (int j = 1; j <= s.numCellsYm1; ++j) { // X sweep, by row
        for (int i = s.iLow[j]; i <= s.iHigh[j]; ++i) {
            jLo[i] = std::min(jLo[i], j);
            jHi[i] = std::max(jHi[i], j);
        }
    }
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
