#pragma once
// =============================================================================
//  SchemeMath.hpp — the scheme's scalar math, one copy for host and device.
//
//  Pure functions of their arguments, so there is nothing backend-specific to
//  specialise: NAVSOLVER_HD makes them callable from CUDA kernels too, which is
//  what retired the hand-synced __device__ duplicates in DeviceMath.cuh.
//  Unqualified fabs/exp/sin/atan resolve to the device intrinsics under nvcc
//  and to <cmath>'s overloads on the host.
// =============================================================================

#include <cmath>

#ifdef __CUDACC__
  #define NAVSOLVER_HD __host__ __device__
#else
  #define NAVSOLVER_HD
#endif

/// UNIFAES weight π(Pe): blends upwind and central differencing so the scheme
/// is exact for 1-D steady advection-diffusion at any Péclet number.
NAVSOLVER_HD inline void computeExponentialWeights(double localRe, double DPe,
                                                   double& pip,
                                                   double& coeffEast, double& coeffWest)
{
    if (fabs(DPe) < 0.1) {
        // Polynomial approximation — numerically stable near Pe = 0
        pip = 1.0 / ((((0.05*DPe + 0.25)*DPe + 1.0)*DPe/6.0 + 0.5)*DPe + 1.0);
    } else if (fabs(DPe) <= 200.0) {
        pip = DPe / (exp(DPe) - 1.0);  // exact Bernstein-Crank formula
    } else if (DPe > 200.0) {
        pip = 0.0;    // advection strongly left-to-right; east weight vanishes
    } else {
        pip = -DPe;   // advection strongly right-to-left
    }

    const double pim = DPe + pip;
    coeffEast = pip / localRe;
    coeffWest = pim / localRe;
}

/// UNIFAES cross-term blending weight ξ.
NAVSOLVER_HD inline double computeQsi(double DPe, double pip, double xeOverDx)
{
    if (fabs(DPe) < 0.01)
        return DPe * (1.0 - DPe * DPe / 60.0) / 12.0 + xeOverDx - 0.5;
    return (pip - 1.0) / DPe + xeOverDx;
}

/// Effective 1/Re at column i, including the hyper-viscous sponge that ramps
/// viscosity up near the outlet. Takes the four scalars it needs rather than a
/// state struct, since host and device carry different ones.
NAVSOLVER_HD inline double effectiveInvReAt(int i, int numCellsX, int hyperViscousStart,
                                            double reynoldsNumber, double hyperViscousRe)
{
    if (hyperViscousStart == 0 || i <= numCellsX - hyperViscousStart)
        return 1.0 / reynoldsNumber;

    const double hpi     = 2.0 * atan(1.0);  // π/2
    const int    IIorig  = numCellsX - 5 * hyperViscousStart / 8;
    const int    IIamp   = 3 * hyperViscousStart / 8;
    const double zReOrig = 0.5 * ( 1.0/hyperViscousRe + 1.0/reynoldsNumber);
    const double zReAmp  = 0.5 * (-1.0/hyperViscousRe + 1.0/reynoldsNumber);

    if (i < numCellsX - hyperViscousStart / 4)
        return zReOrig - zReAmp * sin(((i - IIorig) / (double)IIamp) * hpi);
    return 1.0 / hyperViscousRe;
}
