#pragma once
// =============================================================================
//  SchemeMath.hpp — the scheme's scalar math, one copy for host and device.
//
//  Pure functions of their arguments, so there is nothing backend-specific to
//  specialise: NAVSOLVER_HD makes them callable from CUDA kernels too, which is
//  what retired the hand-synced __device__ duplicates in DeviceMath.cuh.
//
//  Templated on the scalar type so a backend can run them in float. The host
//  instantiates T = double and is bit-identical to the untemplated version;
//  every literal is cast to T so nothing silently promotes.
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
template <typename T>
NAVSOLVER_HD inline void computeExponentialWeights(T localRe, T DPe,
                                                   T& pip,
                                                   T& coeffEast, T& coeffWest)
{
    if (fabs(DPe) < T(0.1)) {
        // Polynomial approximation — numerically stable near Pe = 0
        pip = T(1) / ((((T(0.05)*DPe + T(0.25))*DPe + T(1))*DPe/T(6) + T(0.5))*DPe + T(1));
    } else if (fabs(DPe) <= T(200)) {
        pip = DPe / (exp(DPe) - T(1));  // exact Bernstein-Crank formula
    } else if (DPe > T(200)) {
        pip = T(0);    // advection strongly left-to-right; east weight vanishes
    } else {
        pip = -DPe;    // advection strongly right-to-left
    }

    const T pim = DPe + pip;
    coeffEast = pip / localRe;
    coeffWest = pim / localRe;
}

/// UNIFAES cross-term blending weight ξ.
template <typename T>
NAVSOLVER_HD inline T computeQsi(T DPe, T pip, T xeOverDx)
{
    if (fabs(DPe) < T(0.01))
        return DPe * (T(1) - DPe * DPe / T(60)) / T(12) + xeOverDx - T(0.5);
    return (pip - T(1)) / DPe + xeOverDx;
}

/// Effective 1/Re at column i, including the hyper-viscous sponge that ramps
/// viscosity up near the outlet. Takes the four scalars it needs rather than a
/// state struct, since host and device carry different ones.
template <typename T>
NAVSOLVER_HD inline T effectiveInvReAt(int i, int numCellsX, int hyperViscousStart,
                                       T reynoldsNumber, T hyperViscousRe)
{
    if (hyperViscousStart == 0 || i <= numCellsX - hyperViscousStart)
        return T(1) / reynoldsNumber;

    const T   hpi     = T(2) * atan(T(1));  // π/2
    const int IIorig  = numCellsX - 5 * hyperViscousStart / 8;
    const int IIamp   = 3 * hyperViscousStart / 8;
    const T   zReOrig = T(0.5) * ( T(1)/hyperViscousRe + T(1)/reynoldsNumber);
    const T   zReAmp  = T(0.5) * (-T(1)/hyperViscousRe + T(1)/reynoldsNumber);

    if (i < numCellsX - hyperViscousStart / 4)
        return zReOrig - zReAmp * sin(((i - IIorig) / (T)IIamp) * hpi);
    return T(1) / hyperViscousRe;
}
