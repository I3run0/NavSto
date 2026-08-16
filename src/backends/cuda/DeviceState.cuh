#pragma once
// =============================================================================
//  DeviceState.cuh  —  Device-resident mirror of SimState for the CUDA port.
//
//  A separate struct rather than device pointers on SimState, so the CPU
//  builds stay untouched. Built once from a host SimState after
//  initSimulation() (geometry is fixed from then on), freed at exit.
//
//  The layout matches GridField's flat (i*sJ+j)*sK+k indexing, so a whole
//  field uploads or downloads as one contiguous cudaMemcpy — no repacking.
// =============================================================================

#include "SimState.hpp"
#include "Geometry.hpp"

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err__ = (call);                                             \
        if (err__ != cudaSuccess) {                                             \
            std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,  \
                         cudaGetErrorString(err__));                            \
            std::exit(1);                                                       \
        }                                                                       \
    } while (0)

/// One (i,j,k) cell reference on the device. The CPU iterates rows and strides
/// k, but a GPU wants one thread per CELL for occupancy, so buildDeviceState()
/// expands Geometry.hpp's row list into per-cell red/black lists on the host
/// using the same k-parity formula updateColor() applies.
struct DeviceCellIndex { int i, j, k; };

// ---------------------------------------------------------------------------
//  DeviceState — every device-side allocation + the small scalar config
//  bundle every kernel needs. Passed BY VALUE into kernels (~200 bytes,
//  cheap) so kernels don't need to chase a pointer-to-struct.
// ---------------------------------------------------------------------------
struct DeviceState {
    // ── Grid ─────────────────────────────────────────────────────────────
    int sI = 0, sJ = 0, sK = 0;     // full padded extents (numCells+2)
    long long fieldLen = 0;         // sI*sJ*sK — length of every field array

    // ── Fields (flat, same layout as GridField) ─────────────────────────
    double *velX = nullptr, *velY = nullptr, *velZ = nullptr;
    double *press = nullptr;
    double *accelX = nullptr, *accelY = nullptr, *accelZ = nullptr;
    double *pressureSource = nullptr;
    double *scratchField = nullptr;      // also doubles as MomentumResidual VTK field
    double *divScratch = nullptr;        // transient per-cell divergence (computeDivergence)

    // ── Geometry index arrays (device copies of s.iLow/iHigh/jLow/jHigh) ──
    int *iLow = nullptr, *iHigh = nullptr;   // length sJ
    int *jLow = nullptr, *jHigh = nullptr;   // length sI

    // ── Red-black pressure-solve indices (built host-side, uploaded once) ─
    DeviceCellIndex *redCells = nullptr;
    DeviceCellIndex *blackCells = nullptr;
    int nRed = 0, nBlack = 0;

    // ── computeAccelerations scratch — one slice PER CUDA THREAD, i.e. per
    //  (plane) the thread owns; see docs/cuda-port.md for the indexing.
    //  X-sweep: thread = (j,k) pair, recurrence over i.
    //  Y-sweep: thread = (i,k) pair, recurrence over j.
    //  Z-sweep: thread = (i,j) pair, recurrence over k.
    double *ppieX = nullptr, *ppiwX = nullptr, *qsieX = nullptr;
    double *KuX = nullptr, *KvX = nullptr, *KwX = nullptr;
    int scratchLen = 0;          // per-thread slice length (maxDim+3, shared by all three sweeps)
    int numThreadsX = 0, numThreadsY = 0, numThreadsZ = 0;   // upper-bound slice counts

    double *ppinY = nullptr, *ppisY = nullptr, *qsinY = nullptr;
    double *KuY = nullptr, *KvY = nullptr, *KwY = nullptr;

    double *ppiuZ = nullptr, *ppidZ = nullptr, *qsiuZ = nullptr;
    double *KuZ = nullptr, *KvZ = nullptr, *KwZ = nullptr;

    // ── RK4 bookkeeping buffers. Allocated by backendStartup() and only for
    //  cfg.flowType == RK4Transient; null on every steady-marching run. ────
    double *rk4VelX0 = nullptr, *rk4VelY0 = nullptr, *rk4VelZ0 = nullptr;
    double *rk4Ku = nullptr, *rk4Kv = nullptr, *rk4Kw = nullptr;

    // ── Generic reusable reduction scratch (updateVelocities' maxChange,
    //  adaptTimeStep's per-component max) — sized to the largest per-cell
    //  upper-bound domain used anywhere (numCellsXm1*(numCellsY+1)*numCellsZ),
    //  reused sequentially across calls (never live concurrently). ─────────
    double *cellScratch1 = nullptr, *cellScratch2 = nullptr, *cellScratch3 = nullptr;
    long long maxCellDomain = 0;

    // ── Scalar config every kernel needs (copied by value, no device read) ─
    int numCellsX = 0, numCellsY = 0, numCellsZ = 0;
    int numCellsXm1 = 0, numCellsYm1 = 0, numCellsZm1 = 0;
    double cellSizeX = 0, cellSizeY = 0, cellSizeZ = 0;
    double cellSizeXsq = 0, cellSizeYsq = 0, cellSizeZsq = 0;
    double reynoldsNumber = 0, hyperViscousRe = 0;
    int hyperViscousStart = 0;
    bool periodic = false;          // lateralCondition == Periodic
    bool solidWall = false;         // lateralCondition == SolidWall
    bool outletZeroFirstDeriv = true;
    double sorOmega = 1.0;
    int degreeIndex2 = 0;
    int numPressureIter = 0;

    // Pressure reference-node pin (fixed once geometry is known — see
    // solvePressurePoisson's iRef/jRef/kRef in src/backends/serial/Physics.cpp).
    int iRef = 0, jRef = 0, kRef = 0;

    // Exact active-cell counts for RMS/divergence norms — computed once
    // host-side (see buildDeviceState) by literally counting the same loop
    // bounds the CPU version uses, rather than re-derived on-device per
    // step (this is fixed for the whole run, same reasoning as
    // s.numActiveCells in initSimulation()).
    long long residCellCount = 0;     // computeMomentumResidual's active-cell count
    long long divCellCount = 0;       // computeDivergence's active-cell count (unused directly, kept for symmetry/debug)

    __host__ __device__ inline long long idx(int i, int j, int k) const {
        return (static_cast<long long>(i) * sJ + j) * sK + k;
    }
};

/// Builds a DeviceState from a fully-initialised host SimState (must be
/// called AFTER initSimulation() + buildActiveRows()), allocating and
/// uploading every field/index array. Caller owns the returned DeviceState
/// and must pass it to freeDeviceState() when done.
DeviceState buildDeviceState(SimState& s);

/// Copies every field array back from device to the host SimState's
/// GridFields (velX/velY/velZ/press/accelX/accelY/accelZ/scratchField) —
/// call only when host-visible data is actually needed (VTK snapshot).
void downloadFields(const DeviceState& d, SimState& s);

void freeDeviceState(DeviceState& d);
