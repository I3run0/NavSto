#pragma once
// =============================================================================
//  ViscosityModel.hpp — effective 1/Re, including the hyper-viscous sponge.
//
//  Lives in a header (rather than in one backend's Physics.cpp) because it is
//  the one setup/per-step boundary crosser: Setup.cpp's buildInitialPressure()
//  needs it to seed the initial pressure field, and every backend's
//  computeAccelerations() needs it per i-column in the hot loop. Keeping it
//  `inline` in a header means both get it with no call overhead and no
//  duplicated definition.
//
//  The CUDA backend deliberately does NOT use this: device code can't call a
//  host function, and DeviceState is a different type from SimState, so
//  src/cuda/Kernels.cuh carries a __device__ overload of the same formula.
//  The two must stay in sync -- if you change the sponge profile here, change
//  it there too.
// =============================================================================

#include "SimState.hpp"

#include <cmath>

// ---------------------------------------------------------------------------
//  Effective 1/Re accounting for the hyper-viscous sponge layer.
// ---------------------------------------------------------------------------
inline double effectiveInvRe(const SimState& s, int i)
{
    if (s.cfg.hyperViscousStart == 0 || i <= s.cfg.numCellsX - s.cfg.hyperViscousStart)
        return 1.0 / s.cfg.reynoldsNumber;

    const double hpi    = 2.0 * std::atan(1.0);  // π/2
    const int IIorig    = s.cfg.numCellsX - 5 * s.cfg.hyperViscousStart / 8;
    const int IIamp     = 3 * s.cfg.hyperViscousStart / 8;
    const double zReOrig = 0.5 * (1.0/s.cfg.hyperViscousRe + 1.0/s.cfg.reynoldsNumber);
    const double zReAmp  = 0.5 * (-1.0/s.cfg.hyperViscousRe + 1.0/s.cfg.reynoldsNumber);

    if (i < s.cfg.numCellsX - s.cfg.hyperViscousStart / 4)
        return zReOrig - zReAmp * std::sin(((i - IIorig) / (double)IIamp) * hpi);
    return 1.0 / s.cfg.hyperViscousRe;
}
