#pragma once
// =============================================================================
//  DeviceMath.cuh  —  Device-side field accessors and UNIFAES math helpers
//  shared by every kernel in Physics.cu. Mirrors the anonymous-namespace
//  helpers in src/backends/serial/Physics.cpp exactly (same formulas, same variable
//  names where practical) so the two can be diffed side-by-side.
// =============================================================================

#include "DeviceState.cuh"
#include <cmath>

// ── Field accessors (d passed by value into every kernel; these take a
//    reference to that local copy — the pointers inside still point at
//    real device memory) ──────────────────────────────────────────────────
__device__ __forceinline__ double& VELX(DeviceState& d, int i, int j, int k) { return d.velX[d.idx(i, j, k)]; }
__device__ __forceinline__ double& VELY(DeviceState& d, int i, int j, int k) { return d.velY[d.idx(i, j, k)]; }
__device__ __forceinline__ double& VELZ(DeviceState& d, int i, int j, int k) { return d.velZ[d.idx(i, j, k)]; }
__device__ __forceinline__ double& PRES(DeviceState& d, int i, int j, int k) { return d.press[d.idx(i, j, k)]; }
__device__ __forceinline__ double& ACCX(DeviceState& d, int i, int j, int k) { return d.accelX[d.idx(i, j, k)]; }
__device__ __forceinline__ double& ACCY(DeviceState& d, int i, int j, int k) { return d.accelY[d.idx(i, j, k)]; }
__device__ __forceinline__ double& ACCZ(DeviceState& d, int i, int j, int k) { return d.accelZ[d.idx(i, j, k)]; }
__device__ __forceinline__ double& PSRC(DeviceState& d, int i, int j, int k) { return d.pressureSource[d.idx(i, j, k)]; }
__device__ __forceinline__ double& SCRATCH(DeviceState& d, int i, int j, int k) { return d.scratchField[d.idx(i, j, k)]; }

// ── UNIFAES weight π(Pe) — exact port of computeExponentialWeights ─────────
__device__ __forceinline__ void computeExponentialWeights(double localRe, double DPe,
                                                            double& pip, double& coeffEast, double& coeffWest) {
    if (fabs(DPe) < 0.1) {
        pip = 1.0 / ((((0.05 * DPe + 0.25) * DPe + 1.0) * DPe / 6.0 + 0.5) * DPe + 1.0);
    } else if (fabs(DPe) <= 200.0) {
        pip = DPe / (exp(DPe) - 1.0);
    } else if (DPe > 200.0) {
        pip = 0.0;
    } else {
        pip = -DPe;
    }
    const double pim = DPe + pip;
    coeffEast = pip / localRe;
    coeffWest = pim / localRe;
}

__device__ __forceinline__ double computeQsi(double DPe, double pip, double xeOverDx) {
    if (fabs(DPe) < 0.01)
        return DPe * (1.0 - DPe * DPe / 60.0) / 12.0 + xeOverDx - 0.5;
    return (pip - 1.0) / DPe + xeOverDx;
}

__device__ __forceinline__ double effectiveInvRe(const DeviceState& d, int i) {
    if (d.hyperViscousStart == 0 || i <= d.numCellsX - d.hyperViscousStart)
        return 1.0 / d.reynoldsNumber;

    const double hpi = 2.0 * atan(1.0);
    const int IIorig = d.numCellsX - 5 * d.hyperViscousStart / 8;
    const int IIamp = 3 * d.hyperViscousStart / 8;
    const double zReOrig = 0.5 * (1.0 / d.hyperViscousRe + 1.0 / d.reynoldsNumber);
    const double zReAmp = 0.5 * (-1.0 / d.hyperViscousRe + 1.0 / d.reynoldsNumber);

    if (i < d.numCellsX - d.hyperViscousStart / 4)
        return zReOrig - zReAmp * sin(((i - IIorig) / (double)IIamp) * hpi);
    return 1.0 / d.hyperViscousRe;
}

// ── Active-domain bound helpers — kept in sync BY HAND with
//    src/backends/serial/Physics.cpp's buildPressureSource/computeDivergence (jS/jN)
//    and solvePressurePoisson/Geometry.hpp's ghost-mirror bounds
//    (jLoopS/jLoopN), exactly like Geometry.hpp already documents
//    doing for the CPU paths. ──────────────────────────────────────────────
__device__ __forceinline__ void activeJRange(const DeviceState& d, int i, int& jS, int& jN) {
    const int im = i - 1;
    jS = (i != d.degreeIndex2) ? d.jLow[i] + 1 : d.jLow[im] + 1;
    jN = (i != d.degreeIndex2) ? d.jHigh[i]    : d.jHigh[im];
}

__device__ __forceinline__ void mirrorJRange(const DeviceState& d, int i, int& jLoopS, int& jLoopN) {
    const int im = i - 1;
    if (d.jLow[im] >= d.jLow[i]) jLoopS = d.jLow[i] + 1; else jLoopS = d.jLow[i];
    if (d.jHigh[im] <= d.jHigh[i]) jLoopN = d.jHigh[i]; else jLoopN = d.jHigh[i] + 1;
    if (i == d.degreeIndex2) { jLoopS = d.jLow[im] + 1; jLoopN = d.jHigh[im]; }
}

constexpr int CUDA_BLOCK = 256;
__host__ __forceinline__ int gridFor(long long n) { return (int)((n + CUDA_BLOCK - 1) / CUDA_BLOCK); }
