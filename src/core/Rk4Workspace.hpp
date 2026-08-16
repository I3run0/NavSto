#pragma once
// =============================================================================
//  Rk4Workspace.hpp — the six buffers RK4 bookkeeping needs.
// =============================================================================

#include "GridField.hpp"

struct Rk4Workspace {
    GridField<> velX0, velY0, velZ0;   ///< u_n, snapshotted at the top of the step
    GridField<> Ku, Kv, Kw;            ///< weighted stage increments

    /// Only allocated for RK4Transient runs; steady marching never calls this.
    void allocate(const GridSize& g) {
        velX0.resize(g); velY0.resize(g); velZ0.resize(g);
        Ku.resize(g);    Kv.resize(g);    Kw.resize(g);
    }
};
