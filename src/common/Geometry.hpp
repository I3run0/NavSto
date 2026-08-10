#pragma once
// =============================================================================
//  Geometry.hpp — derived-geometry queries over an initialised SimState.
//
//  Everything here is a pure function of geometry that initSimulation() has
//  already fixed for the run. Nothing here is state: results are returned by
//  value, and whichever backend wants them stores them in its own Workspace,
//  in whatever layout suits it. buildActiveRows() used to write into
//  SimState::activeRows -- OpenMP-only data (the serial solver left it empty)
//  parked in the shared state struct because there was nowhere else to put
//  it, and which the CUDA backend then had to re-expand into a per-cell list
//  anyway. Two backends wanting the same logical data in two different
//  layouts is exactly why this returns a value instead of owning one.
//
//  ── Active-row list for checkerboard (red-black) SOR ──
//
//  Plain Gauss-Seidel/SOR (src/serial/Physics.cpp's solvePressurePoisson)
//  has a genuine loop-carried dependency: each cell's update reads a
//  neighbor that was just written earlier in the same sweep. That can't be
//  correctly parallelized as-is. Red-black splits cells into two colors by
//  (i+j+k) parity; every cell's 6 face-neighbors are always the OTHER
//  color, so within one color, no cell's update depends on another cell of
//  the same color — updating a whole color is embarrassingly parallel.
//
//  v1 of this file (see git history) materialized two flat
//  std::vector<CellIndex> lists (one per color) and had updateColor()
//  dereference them one cell at a time. That's a gather: each cell access
//  needs an indirect load of its (i,j,k) BEFORE the actual press()/
//  pressureSource() addresses are known, which defeats hardware
//  prefetching. This matched the roofline's finding that
//  solvePressurePoisson achieved only ~5% of the measured memory-bandwidth
//  ceiling despite low arithmetic intensity — low AI *and* low fraction of
//  the bandwidth ceiling is the signature of latency-bound, not
//  bandwidth-bound, execution (see docs/roofline.md).
//
//  v2 (this version) exploits a fact the per-cell list threw away: k is
//  NEVER masked by the active-domain geometry (only i,j are — see
//  s.iLow/iHigh/jLow/jHigh), so every active (i,j) row spans the FULL k in
//  [1,numCellsZ]. That means a row plus a color is enough to derive that
//  color's k-stride directly:
//      kStart(red)   = ((i+j) % 2 == 0) ? 2 : 1
//      kStart(black) = ((i+j) % 2 == 0) ? 1 : 2
//  (both step by 2 up to numCellsZ). So instead of one CellIndex per cell,
//  we store ONE (i,j) row per active row — used for BOTH colors, since a
//  row contains cells of both colors at different k parity — and the inner
//  k-loop in updateColor() becomes a direct strided loop, not a gather. The
//  index list also shrinks by roughly numCellsZ/2x (rows, not cells) and
//  is built once, not twice (no separate red/black lists to construct).
//
//  Geometry (s.iLow/iHigh/jLow/jHigh, s.degreeIndex2) is fixed after
//  initSimulation() runs and never changes for the rest of a run, so this
//  row list only needs to be built once, not per sweep or per timestep.
//
//  List ordering: i-outer, j-inner, matching the original per-i,j nesting
//  of solvePressurePoisson's own loop bounds (kept in sync with it by
//  hand — see that function if this ever needs updating).
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
