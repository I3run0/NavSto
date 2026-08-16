#pragma once
// =============================================================================
//  VelocityBCs.hpp — outlet and lateral velocity boundary conditions.
//
//  Boundary-only work (an i-plane and a k-plane), identical in both host
//  backends and small enough that leaving it serial costs nothing measurable.
// =============================================================================

#include "SimState.hpp"

inline void applyVelocityBCs(SimState& s)
{
    const auto& cfg = s.cfg;
    const int NX = cfg.numCellsX;
    const int KKfim = (cfg.lateralCondition == LateralBC::SolidWall)
                    ? s.numCellsZm1 : cfg.numCellsZ;

    if (cfg.outletCondition == OutletBC::ZeroFirstDeriv) {
        for (int j = s.jLow[NX]+1; j <= s.jHigh[NX]-1; ++j)
            for (int k = 0; k <= KKfim; ++k) {
                s.velX(NX, j, k) = s.velX(s.numCellsXm1, j, k);
                s.velY(NX, j, k) = s.velY(s.numCellsXm1, j, k);
                s.velZ(NX, j, k) = s.velZ(s.numCellsXm1, j, k);
            }
    } else {  // ZeroSecondDeriv
        for (int j = s.jLow[NX]+1; j <= s.jHigh[NX]-1; ++j)
            for (int k = 0; k <= KKfim; ++k) {
                s.velX(NX, j, k) = 2.0*s.velX(s.numCellsXm1, j, k) - s.velX(NX-2, j, k);
                s.velY(NX, j, k) = 2.0*s.velY(s.numCellsXm1, j, k) - s.velY(NX-2, j, k);
                s.velZ(NX, j, k) = 2.0*s.velZ(s.numCellsXm1, j, k) - s.velZ(NX-2, j, k);
            }
    }

    if (cfg.lateralCondition == LateralBC::Periodic) {
        for (int i = 1; i <= s.numCellsXm1; ++i)
            for (int j = s.jLow[i]+1; j <= s.jHigh[i]-1; ++j) {
                s.velX(i, j, 0) = s.velX(i, j, cfg.numCellsZ);
                s.velY(i, j, 0) = s.velY(i, j, cfg.numCellsZ);
                s.velZ(i, j, 0) = s.velZ(i, j, cfg.numCellsZ);
            }
    }
}
