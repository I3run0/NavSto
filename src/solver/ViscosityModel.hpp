#pragma once
// =============================================================================
//  ViscosityModel.hpp — effective 1/Re for a host SimState.
//
//  A thin adapter: the formula itself is in SchemeMath.hpp, shared with the
//  device. Setup.cpp seeds the initial pressure with it and every backend's
//  computeAccelerations() calls it per i-column, so it stays inline.
// =============================================================================

#include "SchemeMath.hpp"
#include "SimState.hpp"

inline double effectiveInvRe(const SimState& s, int i)
{
    return effectiveInvReAt(i, s.cfg.numCellsX, s.cfg.hyperViscousStart,
                            s.cfg.reynoldsNumber, s.cfg.hyperViscousRe);
}
