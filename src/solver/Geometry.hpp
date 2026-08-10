#pragma once
// =============================================================================
//  Geometry.hpp — derived-geometry queries over an initialised SimState.
//
//  Pure functions returning by value; whichever backend wants a result stores
//  it in its own Extras, in whatever shape suits. buildActiveRows() used to
//  write into SimState::activeRows — OpenMP-only data the serial solver left
//  empty, and which CUDA re-expanded into a per-cell list anyway.
//
//  ── Active-row list for red-black SOR ──
// =============================================================================

#include "RowIndex.hpp"
#include "SimState.hpp"

#include <cstddef>
#include <vector>

/// Returns the active-row list for solvePressurePoisson's interior domain
/// (i in [1,numCellsX], j in the per-i active span — identical bounds logic
/// to solvePressurePoisson's own loop). Geometry is fixed once
/// initSimulation() has run, so callers build this once and reuse it for the
/// whole run; call it after initSimulation(), never before.
inline std::vector<RowIndex> buildActiveRows(const SimState& s) {
    const auto& cfg = s.cfg;
    std::vector<RowIndex> activeRows;

    // Conservative capacity estimate to avoid reallocation during the
    // push_back loop below; the real active-row count is smaller for a
    // masked (non-rectangular) domain, so this may over-reserve slightly.
    const std::size_t capacity =
        static_cast<std::size_t>(cfg.numCellsX) *
        static_cast<std::size_t>(cfg.numCellsY) + 1;
    activeRows.reserve(capacity);

    for (int i = 1; i <= cfg.numCellsX; ++i) {
        const int im = i - 1;
        int jLoopS, jLoopN;
        if      (s.jLow[im]  >= s.jLow[i])  jLoopS = s.jLow[i]+1;
        else                                  jLoopS = s.jLow[i];
        if      (s.jHigh[im] <= s.jHigh[i]) jLoopN = s.jHigh[i];
        else                                  jLoopN = s.jHigh[i]+1;
        if (i == s.degreeIndex2) { jLoopS = s.jLow[im]+1; jLoopN = s.jHigh[im]; }

        for (int j = jLoopS; j <= jLoopN; ++j) {
            activeRows.push_back(RowIndex{i, j});
        }
    }

    return activeRows;
}
