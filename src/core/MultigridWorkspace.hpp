#pragma once
// =============================================================================
//  MultigridWorkspace.hpp — the coarse grid the pressure correction needs.
//
//  Split from src/solver/PressureMultigrid.hpp (which holds the algorithms and
//  needs all of SimState) for the same reason RowIndex.hpp was split from
//  Geometry.hpp: a BackendConfig.hpp has to declare storage for this without a
//  circular include back through SimState.
// =============================================================================

#include "GridField.hpp"

#include <vector>

struct MultigridWorkspace {
    int nx = 0, ny = 0, nz = 0;          ///< coarse cell counts
    std::vector<int> jLow, jHigh;        ///< coarse active span per coarse i
    GridField<> corr;                    ///< coarse correction e
    GridField<> rhs;                     ///< restricted residual
    GridField<> fineRes;                 ///< residual on the fine grid
    bool built = false;
};
