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

/// What computeAccelerations has to reset before it accumulates.
///
/// Per column i: [jLo, jHi] spans every cell any sweep writes — the X sweep
/// addresses the domain by rows (iLow/iHigh), the Y and Z sweeps by columns
/// (jLow/jHigh), and taking the min/max of both makes it a superset by
/// construction rather than by assuming the two descriptions agree. [xJLo,
/// xJHi] is the sub-span the X sweep assigns outright; it is left empty unless
/// every j in it really is covered, so a geometry whose column coverage is not
/// contiguous falls back to zeroing the whole span rather than trusting a
/// min/max that would skip a cell the X sweep never writes.
///
/// Templated on the plan type so this header stays independent of any one
/// backend's Extras: the CUDA backend includes Geometry.hpp and has no
/// AccelScratch to name.
template <typename ResetPlan>
inline void buildAccelResetPlan(const SimState& s, ResetPlan& plan) {
    const auto& cfg = s.cfg;
    const int nI = cfg.numCellsX + 2;
    plan.zeroJLo.assign(nI, 1); plan.zeroJHi.assign(nI, 0);   // empty span
    plan.xJLo   .assign(nI, 1); plan.xJHi   .assign(nI, 0);

    for (int i = 1; i <= s.numCellsXm1; ++i) {   // Y and Z sweeps, by column
        plan.zeroJLo[i] = s.jLow[i];
        plan.zeroJHi[i] = s.jHigh[i];
    }

    std::vector<int> covered(nI, 0);             // j-count per column, for the check
    for (int j = 1; j <= s.numCellsYm1; ++j) {   // X sweep, by row
        for (int i = s.iLow[j]; i <= s.iHigh[j]; ++i) {
            plan.zeroJLo[i] = std::min(plan.zeroJLo[i], j);
            plan.zeroJHi[i] = std::max(plan.zeroJHi[i], j);
        }
        for (int i = s.iLow[j]+1; i <= s.iHigh[j]-1; ++i) {  // what it assigns
            plan.xJLo[i] = std::min(plan.xJLo[i], j);
            plan.xJHi[i] = std::max(plan.xJHi[i], j);
            ++covered[i];
        }
    }

    for (int i = 1; i <= s.numCellsXm1; ++i)
        if (covered[i] != plan.xJHi[i] - plan.xJLo[i] + 1)   // not contiguous
            { plan.xJLo[i] = 1; plan.xJHi[i] = 0; }
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
