#pragma once
// =============================================================================
//  Rk4Workspace.hpp — the six buffers RK4 bookkeeping needs.
//
//  Templated on the backend's Field so the workspace shares the storage type
//  of the velocities it is read and written alongside.
// =============================================================================

#include "GridField.hpp"

template <typename F>
struct Rk4WorkspaceT {
    F velX0, velY0, velZ0;   ///< u_n, snapshotted at the top of the step
    F Ku, Kv, Kw;            ///< weighted stage increments

    /// Only allocated for RK4Transient runs; steady marching never calls this.
    void allocate(const GridSize& g) {
        velX0.resize(g); velY0.resize(g); velZ0.resize(g);
        Ku.resize(g);    Kv.resize(g);    Kw.resize(g);
    }
};
