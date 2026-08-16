#pragma once
// =============================================================================
//  DeviceMath.cuh  —  Device-side field accessors and bound helpers.
//
//  The scalar formulas are NOT duplicated here any more: SchemeMath.hpp is
//  __host__ __device__ and shared with the CPU backends. What remains is the
//  part that genuinely differs — accessors into DeviceState's raw pointers,
//  and j-range helpers that read DeviceState instead of SimState.
// =============================================================================

#include "DeviceState.cuh"
#include "SchemeMath.hpp"
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

/// DeviceState-flavoured wrapper over SchemeMath.hpp's shared formula.
__device__ __forceinline__ double effectiveInvRe(const DeviceState& d, int i) {
    return effectiveInvReAt(i, d.numCellsX, d.hyperViscousStart,
                            d.reynoldsNumber, d.hyperViscousRe);
}

// ── Active-domain bound helpers. Same rules as Geometry.hpp's host versions,
//    against DeviceState's plain arrays instead of SimState's vectors; change
//    one and change the other. ──────────────────────────────────────────────
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
