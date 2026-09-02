// =============================================================================
//  Physics.cu  —  CUDA port of src/backends/serial/Physics.cpp.
//
//  Builds on: docs/roofline.md (memory-bound verdict), docs/openmp-
//  parallelization.md (red-black restructuring + the mirrorGhostCells race
//  this port deliberately avoids re-introducing — see solvePressurePoissonCuda
//  below). See docs/cuda-port.md for the full threading-model writeup.
//
//  Threading model per the approved plan:
//    - computeAccelerations: one CUDA thread per PLANE (the sweep direction
//      stays a sequential per-thread loop — it's a genuine recurrence along
//      that axis; the two perpendicular axes parallelize across threads).
//    - solvePressurePoisson: one CUDA thread per active cell of one color
//      (red or black), reusing src/solver/Geometry.hpp's index
//      lists built once on the host and uploaded.
//    - Everything else in the per-step loop (buildPressureSource,
//      updateVelocities, computeMomentumResidual, computeDivergence,
//      adaptTimeStep): one thread per cell over a generous upper-bound
//      domain with an in-kernel bounds check, matching the project's
//      "correct but unoptimized first" discipline (see docs/roofline.md /
//      docs/serial-optimization.md for the same measure-before-tune order
//      applied to the CPU paths).
// =============================================================================

#include "DeviceState.cuh"
#include "DeviceMath.cuh"

#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>

#include <cmath>
#include <vector>
#include <algorithm>

// ═════════════════════════════════════════════════════════════════════════
//  DeviceState construction / teardown
// ═════════════════════════════════════════════════════════════════════════

namespace {

Real* uploadField(const Field& f) {
    const auto& data = f.data();
    Real* devPtr;
    CUDA_CHECK(cudaMalloc(&devPtr, data.size() * sizeof(Real)));
    // Host fields are double, device fields are Real. Stage the narrowing
    // here: this runs once per field at startup, not per step.
    std::vector<Real> staged(data.begin(), data.end());
    CUDA_CHECK(cudaMemcpy(devPtr, staged.data(), staged.size() * sizeof(Real), cudaMemcpyHostToDevice));
    return devPtr;
}

Real* allocField(long long n) {
    Real* p;
    CUDA_CHECK(cudaMalloc(&p, n * sizeof(Real)));
    return p;
}

int* uploadInts(const std::vector<int>& v) {
    int* p;
    CUDA_CHECK(cudaMalloc(&p, v.size() * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(p, v.data(), v.size() * sizeof(int), cudaMemcpyHostToDevice));
    return p;
}

DeviceCellIndex* uploadCells(const std::vector<DeviceCellIndex>& v) {
    if (v.empty()) return nullptr;
    DeviceCellIndex* p;
    CUDA_CHECK(cudaMalloc(&p, v.size() * sizeof(DeviceCellIndex)));
    CUDA_CHECK(cudaMemcpy(p, v.data(), v.size() * sizeof(DeviceCellIndex), cudaMemcpyHostToDevice));
    return p;
}

// Expands an active-row list (one (i,j) row, used for both colors — see
// Geometry.hpp) into a per-cell list for `wantParity`, replicating
// src/backends/openmp/Physics.cpp's updateColor() k-parity/stride formula exactly:
// kStart is the smallest k in {1,2} with (i+j+k) parity == wantParity,
// step 2 covers the rest of [1, numCellsZ].
//
// This is the per-cell layout the GPU wants, derived from the row layout
// the CPU wants -- the same logical data in two different shapes, which is
// why buildActiveRows() returns a value rather than storing one.
std::vector<DeviceCellIndex> expandRowsToCells(const SimState& s,
                                                const std::vector<RowIndex>& activeRows,
                                                int wantParity) {
    std::vector<DeviceCellIndex> cells;
    cells.reserve(activeRows.size() * (s.cfg.numCellsZ / 2 + 1));
    for (const auto& row : activeRows) {
        const int kStart = (((row.i + row.j) % 2) == wantParity) ? 2 : 1;
        for (int k = kStart; k <= s.cfg.numCellsZ; k += 2) {
            cells.push_back(DeviceCellIndex{row.i, row.j, k});
        }
    }
    return cells;
}

// Exact active-cell count for computeMomentumResidual's RMS norm — counted
// once here by literally replicating that loop's bounds (i:1..numCellsXm1,
// j:jLow[i]+1..jHigh[i]-1, k:1..KKfim), same "compute once, geometry is
// fixed" reasoning as s.numActiveCells in initSimulation().
long long countResidCells(const SimState& s) {
    const auto& cfg = s.cfg;
    const int KKfim = (cfg.lateralCondition == LateralBC::SolidWall) ? s.numCellsZm1 : cfg.numCellsZ;
    long long count = 0;
    for (int i = 1; i <= s.numCellsXm1; ++i)
        count += (long long)std::max(0, s.jHigh[i] - 1 - (s.jLow[i] + 1) + 1) * KKfim;
    return count;
}

} // namespace

DeviceState buildDeviceState(SimState& s) {
    DeviceState d{};
    const auto& cfg = s.cfg;

    d.sI = s.g.sI; d.sJ = s.g.sJ; d.sK = s.g.sK;
    d.fieldLen = (long long)d.sI * d.sJ * d.sK;

    d.velX = uploadField(s.velX);
    d.velY = uploadField(s.velY);
    d.velZ = uploadField(s.velZ);
    d.press = uploadField(s.press);
    d.accelX = uploadField(s.accelX);
    d.accelY = uploadField(s.accelY);
    d.accelZ = uploadField(s.accelZ);
    d.pressureSource = uploadField(s.pressureSource);
    d.scratchField = uploadField(s.scratchField);
    d.divScratch = allocField(d.fieldLen);

    d.iLow = uploadInts(s.iLow);
    d.iHigh = uploadInts(s.iHigh);
    d.jLow = uploadInts(s.jLow);
    d.jHigh = uploadInts(s.jHigh);

    const std::vector<RowIndex> activeRows = buildActiveRows(s);
    const std::vector<DeviceCellIndex> redCells = expandRowsToCells(s, activeRows, /*wantParity=*/0);
    const std::vector<DeviceCellIndex> blackCells = expandRowsToCells(s, activeRows, /*wantParity=*/1);
    d.redCells = uploadCells(redCells);
    d.blackCells = uploadCells(blackCells);
    d.nRed = (int)redCells.size();
    d.nBlack = (int)blackCells.size();

    d.numCellsX = cfg.numCellsX; d.numCellsY = cfg.numCellsY; d.numCellsZ = cfg.numCellsZ;
    d.numCellsXm1 = s.numCellsXm1; d.numCellsYm1 = s.numCellsYm1; d.numCellsZm1 = s.numCellsZm1;
    d.cellSizeX = cfg.cellSizeX; d.cellSizeY = cfg.cellSizeY; d.cellSizeZ = cfg.cellSizeZ;
    d.cellSizeXsq = s.cellSizeXsq; d.cellSizeYsq = s.cellSizeYsq; d.cellSizeZsq = s.cellSizeZsq;
    d.reynoldsNumber = cfg.reynoldsNumber; d.hyperViscousRe = cfg.hyperViscousRe;
    d.hyperViscousStart = cfg.hyperViscousStart;
    d.periodic = (cfg.lateralCondition == LateralBC::Periodic);
    d.solidWall = (cfg.lateralCondition == LateralBC::SolidWall);
    d.outletZeroFirstDeriv = (cfg.outletCondition == OutletBC::ZeroFirstDeriv);
    d.sorOmega = cfg.sorOmega;
    d.degreeIndex2 = s.degreeIndex2;
    d.numPressureIter = cfg.numPressureIter;

    d.iRef = cfg.numCellsX;
    d.jRef = (s.jHigh[cfg.numCellsX] + s.jLow[cfg.numCellsX]) / 2;
    d.kRef = (cfg.numCellsZ + 1) / 2;

    d.residCellCount = countResidCells(s);

    // Per-thread scratch for computeAccelerations' three sweeps — see
    // DeviceMath.cuh / the sweep kernels below for the addressing scheme.
    const int maxDim = std::max({cfg.numCellsX, cfg.numCellsY, cfg.numCellsZ});
    d.scratchLen = maxDim + 3;
    d.numThreadsX = d.numCellsYm1 * d.numCellsZ;              // (j,k) pairs, upper bound
    d.numThreadsY = d.numCellsXm1 * d.numCellsZ;              // (i,k) pairs, upper bound
    d.numThreadsZ = d.numCellsXm1 * (d.numCellsY + 1);        // (i,j) pairs, upper bound

    auto allocScratch = [&](long long nThreads) { return allocField(nThreads * d.scratchLen); };
    d.ppieX = allocScratch(d.numThreadsX); d.ppiwX = allocScratch(d.numThreadsX); d.qsieX = allocScratch(d.numThreadsX);
    d.KuX = allocScratch(d.numThreadsX); d.KvX = allocScratch(d.numThreadsX); d.KwX = allocScratch(d.numThreadsX);

    d.ppinY = allocScratch(d.numThreadsY); d.ppisY = allocScratch(d.numThreadsY); d.qsinY = allocScratch(d.numThreadsY);
    d.KuY = allocScratch(d.numThreadsY); d.KvY = allocScratch(d.numThreadsY); d.KwY = allocScratch(d.numThreadsY);

    d.ppiuZ = allocScratch(d.numThreadsZ); d.ppidZ = allocScratch(d.numThreadsZ); d.qsiuZ = allocScratch(d.numThreadsZ);
    d.KuZ = allocScratch(d.numThreadsZ); d.KvZ = allocScratch(d.numThreadsZ); d.KwZ = allocScratch(d.numThreadsZ);

    d.maxCellDomain = (long long)d.numCellsXm1 * (d.numCellsY + 1) * d.numCellsZ;
    d.cellScratch1 = allocField(std::max<long long>(d.maxCellDomain, 1));
    d.cellScratch2 = allocField(std::max<long long>(d.maxCellDomain, 1));
    d.cellScratch3 = allocField(std::max<long long>(d.maxCellDomain, 1));

    return d;
}

void downloadFields(const DeviceState& d, SimState& s) {
    std::vector<Real> staged(static_cast<std::size_t>(d.fieldLen));
    auto dl = [&](Real* dev, GridField<>& f) {
        CUDA_CHECK(cudaMemcpy(staged.data(), dev, d.fieldLen * sizeof(Real), cudaMemcpyDeviceToHost));
        std::copy(staged.begin(), staged.end(), f.data().begin());
    };
    dl(d.velX, s.velX); dl(d.velY, s.velY); dl(d.velZ, s.velZ);
    dl(d.press, s.press);
    dl(d.accelX, s.accelX); dl(d.accelY, s.accelY); dl(d.accelZ, s.accelZ);
    dl(d.scratchField, s.scratchField);
}

void freeDeviceState(DeviceState& d) {
    cudaFree(d.velX); cudaFree(d.velY); cudaFree(d.velZ); cudaFree(d.press);
    cudaFree(d.accelX); cudaFree(d.accelY); cudaFree(d.accelZ);
    cudaFree(d.pressureSource); cudaFree(d.scratchField); cudaFree(d.divScratch);
    cudaFree(d.iLow); cudaFree(d.iHigh); cudaFree(d.jLow); cudaFree(d.jHigh);
    cudaFree(d.redCells); cudaFree(d.blackCells);
    cudaFree(d.ppieX); cudaFree(d.ppiwX); cudaFree(d.qsieX); cudaFree(d.KuX); cudaFree(d.KvX); cudaFree(d.KwX);
    cudaFree(d.ppinY); cudaFree(d.ppisY); cudaFree(d.qsinY); cudaFree(d.KuY); cudaFree(d.KvY); cudaFree(d.KwY);
    cudaFree(d.ppiuZ); cudaFree(d.ppidZ); cudaFree(d.qsiuZ); cudaFree(d.KuZ); cudaFree(d.KvZ); cudaFree(d.KwZ);
    cudaFree(d.cellScratch1); cudaFree(d.cellScratch2); cudaFree(d.cellScratch3);
    d = DeviceState{};
}

// ═════════════════════════════════════════════════════════════════════════
//  computeAccelerations — X/Y/Z sweep kernels
// ═════════════════════════════════════════════════════════════════════════

// The X sweep, as four kernels of one thread per cell instead of one kernel of
// one thread per (j, k) looping over i.
//
// Nothing here is a recurrence: pass 1 writes each face's coefficients into
// scratch and the later passes read them back, so every i is independent. The
// old shape existed because the scratch was a per-thread slice; with the
// scratch indexed by element it does not have to be. At 96x48x24 that is 1,128
// threads becoming ~110,000, and the launches replace loop iterations rather
// than adding work.
//
// XSWEEP_SETUP maps a linear id to (i, j, k) with k fastest, so a warp walks k
// and both the velocity reads and the scratch reads coalesce.
#define XSWEEP_SETUP()                                                          \
    const int KKfim = d.periodic ? d.numCellsZ : d.numCellsZm1;                 \
    const long long perI = (long long)d.numCellsYm1 * KKfim;                    \
    const long long tid = blockIdx.x * (long long)blockDim.x + threadIdx.x;     \
    if (tid >= (long long)(d.numCellsX + 1) * perI) return;                      \
    const int i = (int)(tid / perI);   /* iLow[j] can be 0, so i starts at 0 */  \
    const long long rem = tid % perI;                                           \
    const int j = 1 + (int)(rem / KKfim);                                        \
    const int k = 1 + (int)(rem % KKfim);                                        \
    const long long slot = (long long)(j - 1) * d.numCellsZ + (k - 1);          \
    const long long sstr = d.numThreadsX;                                        \
    auto SC = [slot, sstr](Real* p, long long idx) -> Real& {                    \
        return p[idx * sstr + slot]; };                                          \
    const int iStart = d.iLow[j];                                                \
    const int iEnd = d.iHigh[j];

// Passes 1 and 2 of the X sweep in one kernel.
//
// They were split because pass 2 needs face coefficients that pass 1 evaluates
// at NEIGHBOURING i, and threads cannot hand those across without a barrier --
// so pass 1 wrote them to global memory and pass 2 read them back. On this
// device that is the wrong trade: a stand-in build with every division in the
// scheme math turned into a multiply runs only 4.1% faster, so the arithmetic
// is nearly free while the traffic is what costs. The thread at i therefore
// evaluates BOTH of its faces itself -- (i,i+1) and (i-1,i) -- from velocities
// it has already loaded, instead of round-tripping two arrays through global
// memory.
//
// Bit-identical: the recomputed face is the same expression on the same inputs
// the neighbouring thread used. The one thing that must match exactly is the
// Reynolds number -- the west face was evaluated by the thread at i-1 and so
// carries effectiveInvRe(i-1), not effectiveInvRe(i). They differ wherever the
// hyper-viscous sponge ramps.
__global__ void xSweepFusedKernel(DeviceState d) {
    XSWEEP_SETUP()
    if (i < iStart || i > iEnd - 1) return;
    Real* qsie = d.qsieX;

    const Real localRe = 1.0 / effectiveInvRe(d, i);
    const Real vxc = VELX(d, i, j, k), vxp = VELX(d, i + 1, j, k);

    // East face (i, i+1): its qsi feeds the cross-term kernel, and its east
    // coefficient is this cell's own.
    // uFace must land in a Real before it is used: 0.5 is a double literal, so
    // inlining this expression would keep the intermediate in double and shift
    // the last bit of every cell.
    const Real uFaceE = 0.5 * (vxp + vxc);
    const Real DPeE = localRe * uFaceE * d.cellSizeX;
    Real pipE, cipE, cimE;
    computeExponentialWeights(localRe, DPeE, pipE, cipE, cimE);
    SC(qsie, i + 1 + 1) = computeQsi(DPeE, pipE, Real(0.5));

    if (i < iStart + 1) return;          // pass 2 starts one cell later

    // West face (i-1, i), recomputed rather than reloaded. Note the Reynolds
    // number of cell i-1, which is what the thread that used to write this
    // value used.
    const Real localReW = 1.0 / effectiveInvRe(d, i - 1);
    const Real vxm = VELX(d, i - 1, j, k);
    const Real uFaceW = 0.5 * (vxc + vxm);
    const Real DPeW = localReW * uFaceW * d.cellSizeX;
    Real pipW, cipW, cimW;
    computeExponentialWeights(localReW, DPeW, pipW, cipW, cimW);

    Real* Ku = d.KuX; Real* Kv = d.KvX; Real* Kw = d.KwX;
    const Real invDx2 = 1.0 / (d.cellSizeX * d.cellSizeX);
    const Real coeff = invDx2;
    const Real ppie = cipE, ppiw = cimW;

    ACCX(d, i, j, k) += (ppie * (vxp - vxc)
                        + ppiw * (vxm - vxc)) * coeff;
    ACCY(d, i, j, k) += (ppie * (VELY(d, i + 1, j, k) - VELY(d, i, j, k))
                        + ppiw * (VELY(d, i - 1, j, k) - VELY(d, i, j, k))) * coeff;
    ACCZ(d, i, j, k) += (ppie * (VELZ(d, i + 1, j, k) - VELZ(d, i, j, k))
                        + ppiw * (VELZ(d, i - 1, j, k) - VELZ(d, i, j, k))) * coeff;

    const Real DPeC = localRe * vxc * d.cellSizeX;
    Real pipC, cipC, cimC;
    computeExponentialWeights(localRe, DPeC, pipC, cipC, cimC);
    cipC *= invDx2; cimC *= invDx2;
    SC(Ku, i + 1 + 1) = cipC * (vxc - vxp) + cimC * (vxc - vxm);
    SC(Kv, i + 1 + 1) = cipC * (VELY(d, i, j, k) - VELY(d, i + 1, j, k)) + cimC * (VELY(d, i, j, k) - VELY(d, i - 1, j, k));
    SC(Kw, i + 1 + 1) = cipC * (VELZ(d, i, j, k) - VELZ(d, i + 1, j, k)) + cimC * (VELZ(d, i, j, k) - VELZ(d, i - 1, j, k));
}

// The two boundary extrapolations, one thread per (j, k) as before: there are
// exactly six writes per row and they are not per-cell work.
__global__ void xSweepExtrapolateKernel(DeviceState d) {
    const int KKfim = d.periodic ? d.numCellsZ : d.numCellsZm1;
    const long long tid = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (tid >= (long long)d.numCellsYm1 * KKfim) return;
    const int j = 1 + (int)(tid / KKfim);
    const int k = 1 + (int)(tid % KKfim);
    const long long slot = (long long)(j - 1) * d.numCellsZ + (k - 1);
    const long long sstr = d.numThreadsX;
    auto SC = [slot, sstr](Real* p, long long idx) -> Real& { return p[idx * sstr + slot]; };
    const int iStart = d.iLow[j], iEnd = d.iHigh[j];
    Real* Ku = d.KuX; Real* Kv = d.KvX; Real* Kw = d.KwX;

    SC(Ku, iStart + 1 + 1) = 2.0 * SC(Ku, iStart + 2 + 1) - SC(Ku, iStart + 3 + 1);
    SC(Kv, iStart + 1 + 1) = 2.0 * SC(Kv, iStart + 2 + 1) - SC(Kv, iStart + 3 + 1);
    SC(Kw, iStart + 1 + 1) = 2.0 * SC(Kw, iStart + 2 + 1) - SC(Kw, iStart + 3 + 1);
    SC(Ku, iEnd + 1 + 1) = 2.0 * SC(Ku, iEnd + 1) - SC(Ku, iEnd - 1 + 1);
    SC(Kv, iEnd + 1 + 1) = 2.0 * SC(Kv, iEnd + 1) - SC(Kv, iEnd - 1 + 1);
    SC(Kw, iEnd + 1 + 1) = 2.0 * SC(Kw, iEnd + 1) - SC(Kw, iEnd - 1 + 1);
}

// Cross-term subtraction, with the face average folded in. The serial version
// averaged in place, which only works walking i upward; computing both
// averages here from the unaveraged values gives the same two numbers without
// a second buffer or a fifth launch.
__global__ void xSweepCrossTermKernel(DeviceState d) {
    XSWEEP_SETUP()
    if (i < iStart + 1 || i > iEnd - 1) return;
    Real* qsie = d.qsieX;
    Real* Ku = d.KuX; Real* Kv = d.KvX; Real* Kw = d.KwX;

    const Real kuHi = 0.5 * (SC(Ku, i + 1 + 1) + SC(Ku, i + 2 + 1));
    const Real kvHi = 0.5 * (SC(Kv, i + 1 + 1) + SC(Kv, i + 2 + 1));
    const Real kwHi = 0.5 * (SC(Kw, i + 1 + 1) + SC(Kw, i + 2 + 1));
    const Real kuLo = 0.5 * (SC(Ku, i + 1) + SC(Ku, i + 1 + 1));
    const Real kvLo = 0.5 * (SC(Kv, i + 1) + SC(Kv, i + 1 + 1));
    const Real kwLo = 0.5 * (SC(Kw, i + 1) + SC(Kw, i + 1 + 1));

    ACCX(d, i, j, k) -= (kuHi * SC(qsie, i + 1 + 1) - kuLo * SC(qsie, i + 1));
    ACCY(d, i, j, k) -= (kvHi * SC(qsie, i + 1 + 1) - kvLo * SC(qsie, i + 1));
    ACCZ(d, i, j, k) -= (kwHi * SC(qsie, i + 1 + 1) - kwLo * SC(qsie, i + 1));
}

__global__ void ySweepKernel(DeviceState d) {
    const int KKfim = d.periodic ? d.numCellsZ : d.numCellsZm1;
    const long long total = (long long)d.numCellsXm1 * KKfim;
    const long long tid = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (tid >= total) return;
    const int i = 1 + (int)(tid / KKfim);
    const int k = 1 + (int)(tid % KKfim);

    const long long slot = (long long)(i - 1) * d.numCellsZ + (k - 1);
    const long long sstr = d.numThreadsY;
    auto SC = [slot, sstr](Real* p, long long idx) -> Real& { return p[idx * sstr + slot]; };
    Real* ppin = d.ppinY; Real* ppis = d.ppisY; Real* qsin = d.qsinY;
    Real* Ku = d.KuY; Real* Kv = d.KvY; Real* Kw = d.KwY;
    auto VM = [SC](Real* buf, int li) -> Real& { return SC(buf, li + 1); };

    const int jStart = d.jLow[i];
    const int jEnd = d.jHigh[i];
    const Real invDy2 = 1.0 / (d.cellSizeY * d.cellSizeY);
    const Real localRe = 1.0 / effectiveInvRe(d, i);

    for (int j = jStart; j <= jEnd - 1; ++j) {
        const Real vFace = 0.5 * (VELY(d, i, j + 1, k) + VELY(d, i, j, k));
        const Real DPe = localRe * vFace * d.cellSizeY;
        Real pip, cin, cis;
        computeExponentialWeights(localRe, DPe, pip, cin, cis);
        VM(ppin, j + 1) = cin; VM(ppis, j + 2) = cis; VM(qsin, j + 1) = computeQsi(DPe, pip, Real(0.5));
    }
    for (int j = jStart + 1; j <= jEnd - 1; ++j) {
        const Real coeff = invDy2;
        ACCX(d, i, j, k) += (VM(ppin, j + 1) * (VELX(d, i, j + 1, k) - VELX(d, i, j, k))
                            + VM(ppis, j + 1) * (VELX(d, i, j - 1, k) - VELX(d, i, j, k))) * coeff;
        ACCY(d, i, j, k) += (VM(ppin, j + 1) * (VELY(d, i, j + 1, k) - VELY(d, i, j, k))
                            + VM(ppis, j + 1) * (VELY(d, i, j - 1, k) - VELY(d, i, j, k))) * coeff;
        ACCZ(d, i, j, k) += (VM(ppin, j + 1) * (VELZ(d, i, j + 1, k) - VELZ(d, i, j, k))
                            + VM(ppis, j + 1) * (VELZ(d, i, j - 1, k) - VELZ(d, i, j, k))) * coeff;
    }
    for (int j = jStart + 1; j <= jEnd - 1; ++j) {
        const Real vCell = VELY(d, i, j, k);
        const Real DPe = localRe * vCell * d.cellSizeY;
        Real pip, cin, cis;
        computeExponentialWeights(localRe, DPe, pip, cin, cis);
        cin *= invDy2; cis *= invDy2;
        VM(Ku, j + 1) = cin * (VELX(d, i, j, k) - VELX(d, i, j + 1, k)) + cis * (VELX(d, i, j, k) - VELX(d, i, j - 1, k));
        VM(Kv, j + 1) = cin * (VELY(d, i, j, k) - VELY(d, i, j + 1, k)) + cis * (VELY(d, i, j, k) - VELY(d, i, j - 1, k));
        VM(Kw, j + 1) = cin * (VELZ(d, i, j, k) - VELZ(d, i, j + 1, k)) + cis * (VELZ(d, i, j, k) - VELZ(d, i, j - 1, k));
    }
    VM(Ku, jStart + 1) = 2.0 * VM(Ku, jStart + 2) - VM(Ku, jStart + 3);
    VM(Kv, jStart + 1) = 2.0 * VM(Kv, jStart + 2) - VM(Kv, jStart + 3);
    VM(Kw, jStart + 1) = 2.0 * VM(Kw, jStart + 2) - VM(Kw, jStart + 3);
    VM(Ku, jEnd + 1) = 2.0 * VM(Ku, jEnd) - VM(Ku, jEnd - 1);
    VM(Kv, jEnd + 1) = 2.0 * VM(Kv, jEnd) - VM(Kv, jEnd - 1);
    VM(Kw, jEnd + 1) = 2.0 * VM(Kw, jEnd) - VM(Kw, jEnd - 1);
    for (int j = jStart; j <= jEnd - 1; ++j) {
        VM(Ku, j + 1) = 0.5 * (VM(Ku, j + 1) + VM(Ku, j + 2));
        VM(Kv, j + 1) = 0.5 * (VM(Kv, j + 1) + VM(Kv, j + 2));
        VM(Kw, j + 1) = 0.5 * (VM(Kw, j + 1) + VM(Kw, j + 2));
    }
    for (int j = jStart + 1; j <= jEnd - 1; ++j) {
        ACCX(d, i, j, k) -= (VM(Ku, j + 1) * VM(qsin, j + 1) - VM(Ku, j) * VM(qsin, j));
        ACCY(d, i, j, k) -= (VM(Kv, j + 1) * VM(qsin, j + 1) - VM(Kv, j) * VM(qsin, j));
        ACCZ(d, i, j, k) -= (VM(Kw, j + 1) * VM(qsin, j + 1) - VM(Kw, j) * VM(qsin, j));
    }
}

__global__ void zSweepKernel(DeviceState d) {
    const long long total = (long long)d.numCellsXm1 * (d.numCellsY + 1);
    const long long tid = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (tid >= total) return;
    const int i = 1 + (int)(tid / (d.numCellsY + 1));
    const int j = (int)(tid % (d.numCellsY + 1));
    if (i > d.numCellsXm1 || j < d.jLow[i] + 1 || j > d.jHigh[i] - 1) return;

    const int KKfim = d.periodic ? d.numCellsZ : d.numCellsZm1;
    const long long slot = tid;
    const long long sstr = d.numThreadsZ;
    auto SC = [slot, sstr](Real* p, long long idx) -> Real& { return p[idx * sstr + slot]; };
    Real* ppiu = d.ppiuZ; Real* ppid = d.ppidZ; Real* qsiu = d.qsiuZ;
    Real* Ku = d.KuZ; Real* Kv = d.KvZ; Real* Kw = d.KwZ;
    auto VM = [SC](Real* buf, int li) -> Real& { return SC(buf, li + 1); };

    const Real invDz2 = 1.0 / (d.cellSizeZ * d.cellSizeZ);
    const Real localRe = 1.0 / effectiveInvRe(d, i);

    for (int k = 0; k <= KKfim; ++k) {
        const int kp = (d.periodic && k == d.numCellsZ) ? 1 : k + 1;
        const Real wFace = 0.5 * (VELZ(d, i, j, kp) + VELZ(d, i, j, k));
        const Real DPe = localRe * wFace * d.cellSizeZ;
        Real pip, ciu, cid;
        computeExponentialWeights(localRe, DPe, pip, ciu, cid);
        VM(ppiu, k + 1) = ciu; VM(ppid, kp + 1) = cid; VM(qsiu, k + 1) = computeQsi(DPe, pip, Real(0.5));
    }
    for (int k = 1; k <= KKfim; ++k) {
        const int kp = (d.periodic && k == d.numCellsZ) ? 1 : k + 1;
        const int km = (d.periodic && k == 1) ? d.numCellsZ : k - 1;
        const Real coeff = invDz2;
        ACCX(d, i, j, k) += (VM(ppiu, k + 1) * (VELX(d, i, j, kp) - VELX(d, i, j, k))
                            + VM(ppid, k + 1) * (VELX(d, i, j, km) - VELX(d, i, j, k))) * coeff;
        ACCY(d, i, j, k) += (VM(ppiu, k + 1) * (VELY(d, i, j, kp) - VELY(d, i, j, k))
                            + VM(ppid, k + 1) * (VELY(d, i, j, km) - VELY(d, i, j, k))) * coeff;
        ACCZ(d, i, j, k) += (VM(ppiu, k + 1) * (VELZ(d, i, j, kp) - VELZ(d, i, j, k))
                            + VM(ppid, k + 1) * (VELZ(d, i, j, km) - VELZ(d, i, j, k))) * coeff;
    }
    for (int k = 1; k <= KKfim; ++k) {
        const int kp = (d.periodic && k == d.numCellsZ) ? 1 : k + 1;
        const int km = (d.periodic && k == 1) ? d.numCellsZ : k - 1;
        const Real wCell = VELZ(d, i, j, k);
        const Real DPe = localRe * wCell * d.cellSizeZ;
        Real pip, ciu, cid;
        computeExponentialWeights(localRe, DPe, pip, ciu, cid);
        ciu *= invDz2; cid *= invDz2;
        VM(Ku, k + 1) = ciu * (VELX(d, i, j, k) - VELX(d, i, j, kp)) + cid * (VELX(d, i, j, k) - VELX(d, i, j, km));
        VM(Kv, k + 1) = ciu * (VELY(d, i, j, k) - VELY(d, i, j, kp)) + cid * (VELY(d, i, j, k) - VELY(d, i, j, km));
        VM(Kw, k + 1) = ciu * (VELZ(d, i, j, k) - VELZ(d, i, j, kp)) + cid * (VELZ(d, i, j, k) - VELZ(d, i, j, km));
    }
    if (!d.periodic) {
        VM(Ku, 0 + 1) = 2.0 * VM(Ku, 1 + 1) - VM(Ku, 2 + 1);
        VM(Kv, 0 + 1) = 2.0 * VM(Kv, 1 + 1) - VM(Kv, 2 + 1);
        VM(Kw, 0 + 1) = 2.0 * VM(Kw, 1 + 1) - VM(Kw, 2 + 1);
        VM(Ku, d.numCellsZ + 1) = 2.0 * VM(Ku, d.numCellsZm1 + 1) - VM(Ku, d.numCellsZ - 1 + 1);
        VM(Kv, d.numCellsZ + 1) = 2.0 * VM(Kv, d.numCellsZm1 + 1) - VM(Kv, d.numCellsZ - 1 + 1);
        VM(Kw, d.numCellsZ + 1) = 2.0 * VM(Kw, d.numCellsZm1 + 1) - VM(Kw, d.numCellsZ - 1 + 1);
    }
    for (int k = 0; k <= KKfim; ++k) {
        const int kp = (d.periodic && k == d.numCellsZ) ? 1 : k + 1;
        VM(Ku, k + 1) = 0.5 * (VM(Ku, k + 1) + VM(Ku, kp + 1));
        VM(Kv, k + 1) = 0.5 * (VM(Kv, k + 1) + VM(Kv, kp + 1));
        VM(Kw, k + 1) = 0.5 * (VM(Kw, k + 1) + VM(Kw, kp + 1));
    }
    for (int k = 1; k <= KKfim; ++k) {
        const int km = (d.periodic && k == 1) ? d.numCellsZ : k - 1;
        ACCX(d, i, j, k) -= (VM(Ku, k + 1) * VM(qsiu, k + 1) - VM(Ku, km + 1) * VM(qsiu, km + 1));
        ACCY(d, i, j, k) -= (VM(Kv, k + 1) * VM(qsiu, k + 1) - VM(Kv, km + 1) * VM(qsiu, km + 1));
        ACCZ(d, i, j, k) -= (VM(Kw, k + 1) * VM(qsiu, k + 1) - VM(Kw, km + 1) * VM(qsiu, km + 1));
    }
    if (d.periodic) {
        ACCX(d, i, j, 0) = ACCX(d, i, j, d.numCellsZ);
        ACCY(d, i, j, 0) = ACCY(d, i, j, d.numCellsZ);
        ACCZ(d, i, j, 0) = ACCZ(d, i, j, d.numCellsZ);
    }
}

void computeAccelerationsCuda(DeviceState& d) {
    CUDA_CHECK(cudaMemset(d.accelX, 0, d.fieldLen * sizeof(Real)));
    CUDA_CHECK(cudaMemset(d.accelY, 0, d.fieldLen * sizeof(Real)));
    CUDA_CHECK(cudaMemset(d.accelZ, 0, d.fieldLen * sizeof(Real)));

    {
        const long long kk = d.periodic ? d.numCellsZ : d.numCellsZm1;
        const long long perI = (long long)d.numCellsYm1 * kk;
        const long long nCell = (long long)(d.numCellsX + 1) * perI;
        xSweepFusedKernel<<<gridFor(nCell), CUDA_BLOCK>>>(d);
        xSweepExtrapolateKernel<<<gridFor(perI), CUDA_BLOCK>>>(d);
        xSweepCrossTermKernel<<<gridFor(nCell), CUDA_BLOCK>>>(d);
    }
    ySweepKernel<<<gridFor((long long)d.numCellsXm1 * (d.periodic ? d.numCellsZ : d.numCellsZm1)), CUDA_BLOCK>>>(d);
    zSweepKernel<<<gridFor((long long)d.numCellsXm1 * (d.numCellsY + 1)), CUDA_BLOCK>>>(d);
    CUDA_CHECK(cudaGetLastError());
}

// ═════════════════════════════════════════════════════════════════════════
//  buildPressureSource
// ═════════════════════════════════════════════════════════════════════════

__global__ void zeroBoundaryIKKernel(DeviceState d) {
    const long long total = (long long)(d.numCellsX + 1) * (d.numCellsZ + 1);
    const long long tid = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (tid >= total) return;
    const int i = (int)(tid / (d.numCellsZ + 1));
    const int k = (int)(tid % (d.numCellsZ + 1));
    const int jB = d.jLow[i], jT = d.jHigh[i];
    ACCX(d, i, jB, k) = 0.0; ACCX(d, i, jT, k) = 0.0;
    ACCY(d, i, jB, k) = 0.0; ACCY(d, i, jT, k) = 0.0;
    ACCZ(d, i, jB, k) = 0.0; ACCZ(d, i, jT, k) = 0.0;
}

__global__ void zeroBoundaryJKKernel(DeviceState d) {
    const long long total = (long long)(d.numCellsY + 1) * (d.numCellsZ + 1);
    const long long tid = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (tid >= total) return;
    const int j = (int)(tid / (d.numCellsZ + 1));
    const int k = (int)(tid % (d.numCellsZ + 1));
    const int iL = d.iLow[j], iR = d.iHigh[j];
    ACCX(d, iL, j, k) = 0.0; ACCX(d, iR, j, k) = 0.0;
    ACCY(d, iL, j, k) = 0.0; ACCY(d, iR, j, k) = 0.0;
    ACCZ(d, iL, j, k) = 0.0; ACCZ(d, iR, j, k) = 0.0;
}

__global__ void pressureSourceKernel(DeviceState d, Real invDt) {
    const long long total = (long long)d.numCellsX * (d.numCellsY + 1) * d.numCellsZ;
    const long long tid = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (tid >= total) return;
    const int i = 1 + (int)(tid / ((long long)(d.numCellsY + 1) * d.numCellsZ));
    if (i > d.numCellsX) return;
    const long long rem = tid % ((long long)(d.numCellsY + 1) * d.numCellsZ);
    const int j = (int)(rem / d.numCellsZ);
    const int k = 1 + (int)(rem % d.numCellsZ);

    int jS, jN; activeJRange(d, i, jS, jN);
    if (j < jS || j > jN) return;

    const int im = i - 1, jm = j - 1;
    int km = k - 1;
    if (d.periodic && k == 1) km = d.numCellsZ;

    const Real qInvDx = 0.25 / d.cellSizeX, qInvDy = 0.25 / d.cellSizeY, qInvDz = 0.25 / d.cellSizeZ;

    const Real divU = (VELX(d,i,j,k)-VELX(d,im,j,k)+VELX(d,i,jm,k)-VELX(d,im,jm,k)
                        + VELX(d,i,j,km)-VELX(d,im,j,km)+VELX(d,i,jm,km)-VELX(d,im,jm,km)) * qInvDx;
    const Real divV = (VELY(d,i,j,k)-VELY(d,i,jm,k)+VELY(d,im,j,k)-VELY(d,im,jm,k)
                        + VELY(d,i,j,km)-VELY(d,i,jm,km)+VELY(d,im,j,km)-VELY(d,im,jm,km)) * qInvDy;
    const Real divW = (VELZ(d,i,j,k)+VELZ(d,i,jm,k)+VELZ(d,im,j,k)+VELZ(d,im,jm,k)
                        - VELZ(d,i,j,km)-VELZ(d,i,jm,km)-VELZ(d,im,j,km)-VELZ(d,im,jm,km)) * qInvDz;
    const Real divAu = (ACCX(d,i,j,k)-ACCX(d,im,j,k)+ACCX(d,i,jm,k)-ACCX(d,im,jm,k)
                        + ACCX(d,i,j,km)-ACCX(d,im,j,km)+ACCX(d,i,jm,km)-ACCX(d,im,jm,km)) * qInvDx;
    const Real divAv = (ACCY(d,i,j,k)-ACCY(d,i,jm,k)+ACCY(d,im,j,k)-ACCY(d,im,jm,k)
                        + ACCY(d,i,j,km)-ACCY(d,i,jm,km)+ACCY(d,im,j,km)-ACCY(d,im,jm,km)) * qInvDy;
    const Real divAw = (ACCZ(d,i,j,k)+ACCZ(d,i,jm,k)+ACCZ(d,im,j,k)+ACCZ(d,im,jm,k)
                        - ACCZ(d,i,j,km)-ACCZ(d,i,jm,km)-ACCZ(d,im,j,km)-ACCZ(d,im,jm,km)) * qInvDz;

    PSRC(d, i, j, k) = (divU + divV + divW) * invDt + (divAu + divAv + divAw);
}

void buildPressureSourceCuda(DeviceState& d, Real timeStepSize) {
    zeroBoundaryIKKernel<<<gridFor((long long)(d.numCellsX+1)*(d.numCellsZ+1)), CUDA_BLOCK>>>(d);
    zeroBoundaryJKKernel<<<gridFor((long long)(d.numCellsY+1)*(d.numCellsZ+1)), CUDA_BLOCK>>>(d);
    pressureSourceKernel<<<gridFor((long long)d.numCellsX*(d.numCellsY+1)*d.numCellsZ), CUDA_BLOCK>>>(d, 1.0 / timeStepSize);
    CUDA_CHECK(cudaGetLastError());
}

// ═════════════════════════════════════════════════════════════════════════
//  solvePressurePoisson — red-black SOR.
//
//  mirrorGhostCells is split into two kernels, ported directly from the
//  two-pass structure verified on the OpenMP path (src/backends/openmp/Physics.cpp,
//  docs/openmp-parallelization.md's "Round 2" section): kernelA does only
//  the cross-row (im/ip) writes, one thread per row i; then, on kernel-
//  launch-boundary ordering (this file uses no explicit CUDA streams, so
//  the default stream serializes these two launches exactly like OpenMP's
//  implicit barrier between its two `#pragma omp for` passes), kernelB
//  does only the same-row (j/k-boundary) writes, one thread per row i.
//  kernelB's writes are then provably confined to each thread's own row i
//  since kernelA has fully completed (and its device-wide effects are
//  visible — a completed kernel launch is a full device-wide memory fence,
//  stronger than OpenMP's barrier needed here). This was previously a
//  single <<<1,1>>>-launched serial kernel, kept that way because CUDA has
//  no ordering guarantee AT ALL among threads within one launch (weaker
//  than OpenMP's schedule(static)) and no race detector was available to
//  verify a parallel version — see the verification section in
//  docs/cuda-port.md for how this version was checked (determinism across
//  repeated runs plus compute-sanitizer racecheck) before shipping.
// ═════════════════════════════════════════════════════════════════════════

// Ghost mirroring, one write per thread instead of one j-scan per thread.
//
// These used to run a thread per (i,k) that walked the whole j range and wrote
// only where a boundary test fired -- O(nx*ny*nz) iterations to perform
// O(ny*nz + nx*nz + nx*ny) writes. At 180x60x30 that is ~324k iterations for a
// few thousand stores, and it measured 6.9% of the CUDA run (probe: keep the
// launches, empty the bodies). The mapping is simply inverted so each thread
// lands directly on the cells it writes.
//
// Same stores, same values, so the pressure field is bit-identical. Every
// destination is written by exactly one thread, as before.

// The i-direction ghosts: a thread per (j,k) writes its row's two ends
// directly, rather than every column testing whether it is one of them.
__global__ void mirrorGhostCellsCrossRowKernel(DeviceState d) {
    const long long tid = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (tid >= (long long)d.numCellsY * d.numCellsZ) return;
    const int j = 1 + (int)(tid / d.numCellsZ);
    const int k = 1 + (int)(tid % d.numCellsZ);

    // A column i mirrors into row j only if j lies in that column's own
    // mirror range -- the condition the j-scan applied implicitly.
    auto inRange = [&](int i) {
        if (i < 1 || i > d.numCellsX) return false;
        int s_, n_; mirrorJRange(d, i, s_, n_);
        return j >= s_ && j <= n_;
    };
    if (inRange(1))                       PRES(d, 0, j, k) = PRES(d, 1, j, k);
    const int iL = d.iLow[j] + 1;
    if (iL != 1 && inRange(iL))           PRES(d, iL - 1, j, k) = PRES(d, iL, j, k);

    if (inRange(d.numCellsX))             PRES(d, d.numCellsX + 1, j, k) = PRES(d, d.numCellsX, j, k);
    const int iH = d.iHigh[j];
    if (iH != d.numCellsX && inRange(iH)) PRES(d, iH + 1, j, k) = PRES(d, iH, j, k);
}

// The j-direction ghosts (two per (i,k), no scan needed) and, for the two
// k-planes that have them, the z-direction ghosts. Only 2 of numCellsZ threads
// take the j loop, so it costs nothing on the rest.
__global__ void mirrorGhostCellsSameRowKernel(DeviceState d) {
    const long long tid = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (tid >= (long long)d.numCellsX * d.numCellsZ) return;
    const int i = 1 + (int)(tid / d.numCellsZ);
    const int k = 1 + (int)(tid % d.numCellsZ);
    int jLoopS, jLoopN; mirrorJRange(d, i, jLoopS, jLoopN);
    if (jLoopS > jLoopN) return;

    PRES(d, i, jLoopS - 1, k) = PRES(d, i, jLoopS, k);
    PRES(d, i, jLoopN + 1, k) = PRES(d, i, jLoopN, k);

    if (d.solidWall && (k == 1 || k == d.numCellsZ)) {
        const int kg = (k == 1) ? k - 1 : k + 1;
        for (int j = jLoopS; j <= jLoopN; ++j) PRES(d, i, j, kg) = PRES(d, i, j, k);
    }
}

__global__ void updateColorKernel(DeviceState d, const DeviceCellIndex* cells, int n,
                                   Real cX, Real cY, Real cZ, Real invDiag, Real pRef) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    const DeviceCellIndex c = cells[idx];
    const int i = c.i, j = c.j, k = c.k;
    const int im = i - 1, ip = i + 1, jm = j - 1, jp = j + 1;
    int km = k - 1, kp = k + 1;
    if (!d.solidWall) {
        if (k == 1) km = d.numCellsZ;
        if (k == d.numCellsZ) kp = 1;
    }

    if (i == d.iRef && j == d.jRef && k == d.kRef) {
        PRES(d, i, j, k) = pRef;
        return;
    }
    Real pNew = (cY * (PRES(d,i,jp,k) + PRES(d,i,jm,k))
                 + cX * (PRES(d,ip,j,k) + PRES(d,im,j,k))
                 + cZ * (PRES(d,i,j,kp) + PRES(d,i,j,km))
                 - PSRC(d,i,j,k)) * invDiag;
    if ((i == 1 || i == d.numCellsX) && (j == d.jLow[i] + 1 || j == d.jHigh[i])) {
        pNew -= PSRC(d, i, j, k) * invDiag;
        if (d.solidWall && (k == 1 || k == d.numCellsZ))
            pNew -= 2.0 * PSRC(d, i, j, k) * invDiag;
    }
    PRES(d, i, j, k) += d.sorOmega * (pNew - PRES(d, i, j, k));
}

void solvePressurePoissonCuda(DeviceState& d) {
    const Real cX = 1.0 / d.cellSizeXsq, cY = 1.0 / d.cellSizeYsq, cZ = 1.0 / d.cellSizeZsq;
    const Real invDiag = 0.5 / (cX + cY + cZ);

    Real pRef;
    CUDA_CHECK(cudaMemcpy(&pRef, d.press + d.idx(d.iRef, d.jRef, d.kRef), sizeof(Real), cudaMemcpyDeviceToHost));

    for (int sweep = 0; sweep < d.numPressureIter; ++sweep) {
        mirrorGhostCellsCrossRowKernel<<<gridFor((long long)d.numCellsY * d.numCellsZ), CUDA_BLOCK>>>(d);
        mirrorGhostCellsSameRowKernel<<<gridFor((long long)d.numCellsX * d.numCellsZ), CUDA_BLOCK>>>(d);
        updateColorKernel<<<gridFor(d.nRed), CUDA_BLOCK>>>(d, d.redCells, d.nRed, cX, cY, cZ, invDiag, pRef);
        mirrorGhostCellsCrossRowKernel<<<gridFor((long long)d.numCellsY * d.numCellsZ), CUDA_BLOCK>>>(d);
        mirrorGhostCellsSameRowKernel<<<gridFor((long long)d.numCellsX * d.numCellsZ), CUDA_BLOCK>>>(d);
        updateColorKernel<<<gridFor(d.nBlack), CUDA_BLOCK>>>(d, d.blackCells, d.nBlack, cX, cY, cZ, invDiag, pRef);
    }
    CUDA_CHECK(cudaGetLastError());
}

// ═════════════════════════════════════════════════════════════════════════
//  updateVelocities
// ═════════════════════════════════════════════════════════════════════════

__global__ void updateVelocitiesKernel(DeviceState d, Real dtEff) {
    const long long total = (long long)d.numCellsXm1 * (d.numCellsY + 1) * d.numCellsZ;
    const long long tid = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (tid >= total) return;
    const int i = 1 + (int)(tid / ((long long)(d.numCellsY + 1) * d.numCellsZ));
    const long long rem = tid % ((long long)(d.numCellsY + 1) * d.numCellsZ);
    const int j = (int)(rem / d.numCellsZ);
    const int k = 1 + (int)(rem % d.numCellsZ);

    const int KKfim = d.solidWall ? d.numCellsZm1 : d.numCellsZ;
    if (i > d.numCellsXm1 || j < d.jLow[i] + 1 || j > d.jHigh[i] - 1 || k > KKfim) return;
    const int ip = i + 1, jp = j + 1;
    const int kp = (k < d.numCellsZ) ? k + 1 : 1;
    const Real qInvDx = 0.25 / d.cellSizeX, qInvDy = 0.25 / d.cellSizeY, qInvDz = 0.25 / d.cellSizeZ;

    const Real dpu = (PRES(d,ip,j,kp)-PRES(d,i,j,kp)+PRES(d,ip,jp,kp)-PRES(d,i,jp,kp)
                       +PRES(d,ip,j,k) -PRES(d,i,j,k) +PRES(d,ip,jp,k) -PRES(d,i,jp,k)) * qInvDx;
    const Real dpv = (PRES(d,i,jp,kp)-PRES(d,i,j,kp)+PRES(d,ip,jp,kp)-PRES(d,ip,j,kp)
                       +PRES(d,i,jp,k) -PRES(d,i,j,k) +PRES(d,ip,jp,k) -PRES(d,ip,j,k)) * qInvDy;
    const Real dpw = (PRES(d,i,jp,kp)+PRES(d,i,j,kp)+PRES(d,ip,jp,kp)+PRES(d,ip,j,kp)
                       -PRES(d,i,jp,k) -PRES(d,i,j,k) -PRES(d,ip,jp,k) -PRES(d,ip,j,k)) * qInvDz;

    const Real du = ACCX(d,i,j,k) - dpu;
    const Real dv = ACCY(d,i,j,k) - dpv;
    const Real dw = ACCZ(d,i,j,k) - dpw;

    VELX(d,i,j,k) += du * dtEff;
    VELY(d,i,j,k) += dv * dtEff;
    VELZ(d,i,j,k) += dw * dtEff;
}

__global__ void outletBCKernel(DeviceState d, bool zeroSecondDeriv) {
    const int NX = d.numCellsX;
    const int KKfim = d.solidWall ? d.numCellsZm1 : d.numCellsZ;
    const int jS = d.jLow[NX] + 1, jN = d.jHigh[NX] - 1;
    const int nJ = jN - jS + 1;
    if (nJ <= 0) return;
    const long long total = (long long)nJ * (KKfim + 1);
    const long long tid = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (tid >= total) return;
    const int j = jS + (int)(tid / (KKfim + 1));
    const int k = (int)(tid % (KKfim + 1));
    if (!zeroSecondDeriv) {
        VELX(d,NX,j,k) = VELX(d,NX-1,j,k);
        VELY(d,NX,j,k) = VELY(d,NX-1,j,k);
        VELZ(d,NX,j,k) = VELZ(d,NX-1,j,k);
    } else {
        VELX(d,NX,j,k) = 2.0*VELX(d,NX-1,j,k) - VELX(d,NX-2,j,k);
        VELY(d,NX,j,k) = 2.0*VELY(d,NX-1,j,k) - VELY(d,NX-2,j,k);
        VELZ(d,NX,j,k) = 2.0*VELZ(d,NX-1,j,k) - VELZ(d,NX-2,j,k);
    }
}

__global__ void periodicCopyVelKernel(DeviceState d) {
    const long long total = (long long)d.numCellsXm1 * (d.numCellsY + 1);
    const long long tid = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (tid >= total) return;
    const int i = 1 + (int)(tid / (d.numCellsY + 1));
    const int j = (int)(tid % (d.numCellsY + 1));
    if (i > d.numCellsXm1 || j < d.jLow[i] + 1 || j > d.jHigh[i] - 1) return;
    VELX(d,i,j,0) = VELX(d,i,j,d.numCellsZ);
    VELY(d,i,j,0) = VELY(d,i,j,d.numCellsZ);
    VELZ(d,i,j,0) = VELZ(d,i,j,d.numCellsZ);
}

void updateVelocitiesCuda(DeviceState& d, bool useHalfStep, Real timeStepSize) {
    const Real dtEff = useHalfStep ? 0.5 * timeStepSize : timeStepSize;
    updateVelocitiesKernel<<<gridFor(d.maxCellDomain), CUDA_BLOCK>>>(d, dtEff);

    outletBCKernel<<<gridFor((long long)(d.numCellsY+2)*(d.numCellsZ+2)), CUDA_BLOCK>>>(d, !d.outletZeroFirstDeriv);
    if (d.periodic)
        periodicCopyVelKernel<<<gridFor((long long)d.numCellsXm1*(d.numCellsY+1)), CUDA_BLOCK>>>(d);
    CUDA_CHECK(cudaGetLastError());
}

// ═════════════════════════════════════════════════════════════════════════
//  computeMomentumResidual
// ═════════════════════════════════════════════════════════════════════════

__global__ void momentumResidualKernel(DeviceState d) {
    const long long total = (long long)d.numCellsXm1 * (d.numCellsY + 1) * d.numCellsZ;
    const long long tid = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (tid >= total) return;
    const int i = 1 + (int)(tid / ((long long)(d.numCellsY + 1) * d.numCellsZ));
    const long long rem = tid % ((long long)(d.numCellsY + 1) * d.numCellsZ);
    const int j = (int)(rem / d.numCellsZ);
    const int k = 1 + (int)(rem % d.numCellsZ);

    const int KKfim = d.solidWall ? d.numCellsZm1 : d.numCellsZ;
    if (i > d.numCellsXm1 || j < d.jLow[i] + 1 || j > d.jHigh[i] - 1 || k > KKfim) return;

    const int ip = i + 1, jp = j + 1;
    const int kp = (k < d.numCellsZ) ? k + 1 : 1;
    const Real qInvDx = 0.25 / d.cellSizeX, qInvDy = 0.25 / d.cellSizeY, qInvDz = 0.25 / d.cellSizeZ;

    const Real gradPx = (PRES(d,ip,j,k)-PRES(d,i,j,k)+PRES(d,ip,jp,k)-PRES(d,i,jp,k)
                          + PRES(d,ip,j,kp)-PRES(d,i,j,kp)+PRES(d,ip,jp,kp)-PRES(d,i,jp,kp)) * qInvDx;
    const Real gradPy = (PRES(d,i,jp,k)-PRES(d,i,j,k)+PRES(d,ip,jp,k)-PRES(d,ip,j,k)
                          + PRES(d,i,jp,kp)-PRES(d,i,j,kp)+PRES(d,ip,jp,kp)-PRES(d,ip,j,kp)) * qInvDy;
    const Real gradPz = (-PRES(d,i,jp,k)-PRES(d,i,j,k)-PRES(d,ip,jp,k)-PRES(d,ip,j,k)
                          + PRES(d,i,jp,kp)+PRES(d,i,j,kp)+PRES(d,ip,jp,kp)+PRES(d,ip,j,kp)) * qInvDz;

    const Real resU = ACCX(d,i,j,k) - gradPx;
    const Real resV = ACCY(d,i,j,k) - gradPy;
    const Real resW = ACCZ(d,i,j,k) - gradPz;
    SCRATCH(d, i, j, k) = sqrt(resU*resU + resV*resV + resW*resW);
}

// Max reductions run in Real -- a maximum is exact at any precision. SUMS stay
// in double on purpose: these accumulate over ~2.7M cells, and a float
// accumulator loses far more to cancellation than the device gains from the
// narrower type. The functors below therefore widen for the sum paths.
struct SquareFunctor { __device__ double operator()(Real v) const { return double(v) * double(v); } };
struct AbsFunctor    { __device__ double operator()(Real v) const { return fabs(double(v)); } };
struct AbsFunctorR   { __device__ Real   operator()(Real v) const { return fabs(v); } };

void computeMomentumResidualCuda(DeviceState& d, double& residMax, double& residRMS) {
    CUDA_CHECK(cudaMemset(d.scratchField, 0, d.fieldLen * sizeof(Real)));
    momentumResidualKernel<<<gridFor(d.maxCellDomain), CUDA_BLOCK>>>(d);
    CUDA_CHECK(cudaGetLastError());

    thrust::device_ptr<Real> sf(d.scratchField);
    residMax = thrust::reduce(thrust::device, sf, sf + d.fieldLen, Real(0), thrust::maximum<Real>());
    const double sumSq = thrust::transform_reduce(thrust::device, sf, sf + d.fieldLen, SquareFunctor(), 0.0, thrust::plus<double>());
    residRMS = (d.residCellCount > 0) ? std::sqrt(sumSq / (double)d.residCellCount) : 0.0;
}

// ═════════════════════════════════════════════════════════════════════════
//  computeDivergence
// ═════════════════════════════════════════════════════════════════════════

__global__ void divergenceKernel(DeviceState d) {
    const long long total = (long long)d.numCellsX * (d.numCellsY + 1) * d.numCellsZ;
    const long long tid = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (tid >= total) return;
    const int i = 1 + (int)(tid / ((long long)(d.numCellsY + 1) * d.numCellsZ));
    if (i > d.numCellsX) return;
    const long long rem = tid % ((long long)(d.numCellsY + 1) * d.numCellsZ);
    const int j = (int)(rem / d.numCellsZ);
    const int k = 1 + (int)(rem % d.numCellsZ);

    int jS, jN; activeJRange(d, i, jS, jN);
    if (j < jS || j > jN) return;

    const int im = i - 1, jm = j - 1;
    const int km = (d.periodic && k == 1) ? d.numCellsZ : k - 1;
    const Real qInvDx = 0.25 / d.cellSizeX, qInvDy = 0.25 / d.cellSizeY, qInvDz = 0.25 / d.cellSizeZ;

    const Real div = (VELX(d,i,j,k)-VELX(d,im,j,k)+VELX(d,i,jm,k)-VELX(d,im,jm,k)
                       + VELX(d,i,j,km)-VELX(d,im,j,km)+VELX(d,i,jm,km)-VELX(d,im,jm,km)) * qInvDx
                      + (VELY(d,i,j,k)-VELY(d,i,jm,k)+VELY(d,im,j,k)-VELY(d,im,jm,k)
                       + VELY(d,i,j,km)-VELY(d,i,jm,km)+VELY(d,im,j,km)-VELY(d,im,jm,km)) * qInvDy
                      + (VELZ(d,i,j,k)+VELZ(d,i,jm,k)+VELZ(d,im,j,k)+VELZ(d,im,jm,k)
                       - VELZ(d,i,j,km)-VELZ(d,i,jm,km)-VELZ(d,im,j,km)-VELZ(d,im,jm,km)) * qInvDz;

    d.divScratch[d.idx(i, j, k)] = div;
}

void computeDivergenceCuda(DeviceState& d, double& dilatationMax, double& intDivergence, double& intAbsDivergence) {
    CUDA_CHECK(cudaMemset(d.divScratch, 0, d.fieldLen * sizeof(Real)));
    divergenceKernel<<<gridFor((long long)d.numCellsX*(d.numCellsY+1)*d.numCellsZ), CUDA_BLOCK>>>(d);
    CUDA_CHECK(cudaGetLastError());

    thrust::device_ptr<Real> dv(d.divScratch);
    intDivergence = thrust::reduce(thrust::device, dv, dv + d.fieldLen, 0.0, thrust::plus<double>());
    intAbsDivergence = thrust::transform_reduce(thrust::device, dv, dv + d.fieldLen, AbsFunctor(), 0.0, thrust::plus<double>());
    dilatationMax = thrust::transform_reduce(thrust::device, dv, dv + d.fieldLen, AbsFunctorR(), Real(0), thrust::maximum<Real>());

    const Real cellVol = d.cellSizeX * d.cellSizeY * d.cellSizeZ;
    intDivergence *= cellVol;
    intAbsDivergence *= cellVol;
}

// ═════════════════════════════════════════════════════════════════════════
//  adaptTimeStep
// ═════════════════════════════════════════════════════════════════════════

// Three max reductions in one pass. The previous version wrote |u|, |v| and
// |w| into three full-length arrays and then ran three thrust::reduce calls,
// each of which reads its array back and synchronises: three device round
// trips and six field-sized transfers to produce three numbers. Here a
// grid-stride loop reduces into shared memory, one partial triple per block,
// and a single second launch folds those. max is exactly associative, so the
// tree shape does not change the answer.
__global__ void velMaxBlockKernel(DeviceState d, Real* partials, int nBlocks) {
    __shared__ Real su[CUDA_BLOCK], sv[CUDA_BLOCK], sw[CUDA_BLOCK];
    const long long total = (long long)d.numCellsXm1 * (d.numCellsY + 1) * d.numCellsZ;
    const long long stride = (long long)gridDim.x * blockDim.x;

    Real u = 0, v = 0, w = 0;
    for (long long tid = blockIdx.x * (long long)blockDim.x + threadIdx.x;
         tid < total; tid += stride) {
        const int i = 1 + (int)(tid / ((long long)(d.numCellsY + 1) * d.numCellsZ));
        const long long rem = tid % ((long long)(d.numCellsY + 1) * d.numCellsZ);
        const int j = (int)(rem / d.numCellsZ);
        const int k = 1 + (int)(rem % d.numCellsZ);
        if (i > d.numCellsXm1 || j < d.jLow[i] + 1 || j > d.jHigh[i] - 1) continue;
        u = fmax(u, fabs(VELX(d, i, j, k)));
        v = fmax(v, fabs(VELY(d, i, j, k)));
        w = fmax(w, fabs(VELZ(d, i, j, k)));
    }
    su[threadIdx.x] = u; sv[threadIdx.x] = v; sw[threadIdx.x] = w;
    __syncthreads();

    for (int half = blockDim.x / 2; half > 0; half >>= 1) {
        if (threadIdx.x < half) {
            su[threadIdx.x] = fmax(su[threadIdx.x], su[threadIdx.x + half]);
            sv[threadIdx.x] = fmax(sv[threadIdx.x], sv[threadIdx.x + half]);
            sw[threadIdx.x] = fmax(sw[threadIdx.x], sw[threadIdx.x + half]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        partials[blockIdx.x]               = su[0];
        partials[blockIdx.x + nBlocks]     = sv[0];
        partials[blockIdx.x + 2 * nBlocks] = sw[0];
    }
}

__global__ void velMaxFinalKernel(Real* partials, int nBlocks) {
    Real u = 0, v = 0, w = 0;
    for (int b = threadIdx.x; b < nBlocks; b += blockDim.x) {
        u = fmax(u, partials[b]);
        v = fmax(v, partials[b + nBlocks]);
        w = fmax(w, partials[b + 2 * nBlocks]);
    }
    __shared__ Real su[CUDA_BLOCK], sv[CUDA_BLOCK], sw[CUDA_BLOCK];
    su[threadIdx.x] = u; sv[threadIdx.x] = v; sw[threadIdx.x] = w;
    __syncthreads();
    for (int half = blockDim.x / 2; half > 0; half >>= 1) {
        if (threadIdx.x < half) {
            su[threadIdx.x] = fmax(su[threadIdx.x], su[threadIdx.x + half]);
            sv[threadIdx.x] = fmax(sv[threadIdx.x], sv[threadIdx.x + half]);
            sw[threadIdx.x] = fmax(sw[threadIdx.x], sw[threadIdx.x + half]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) { partials[0] = su[0]; partials[1] = sv[0]; partials[2] = sw[0]; }
}

void adaptTimeStepCuda(DeviceState& d, double& timeStepSize) {
    // Cap the block count so the partials fit in one final block's stride and
    // the grid-stride loop keeps every block busy on a small grid.
    const int nBlocks = (int)std::min<long long>(
        (d.maxCellDomain + CUDA_BLOCK - 1) / CUDA_BLOCK, 256);
    velMaxBlockKernel<<<nBlocks, CUDA_BLOCK>>>(d, d.cellScratch1, nBlocks);
    velMaxFinalKernel<<<1, CUDA_BLOCK>>>(d.cellScratch1, nBlocks);
    CUDA_CHECK(cudaGetLastError());

    Real maxima[3];
    CUDA_CHECK(cudaMemcpy(maxima, d.cellScratch1, 3 * sizeof(Real), cudaMemcpyDeviceToHost));
    const Real uMax = maxima[0], vMax = maxima[1], wMax = maxima[2];

    const Real reForDiff = (d.hyperViscousStart == 0) ? d.reynoldsNumber : d.hyperViscousRe;
    const Real dtViscous = 0.5 * reForDiff / (1.0/d.cellSizeXsq + 1.0/d.cellSizeYsq + 1.0/d.cellSizeZsq);
    const Real dtAdv = std::min({d.cellSizeX / (uMax + 1e-30),
                                    d.cellSizeY / (vMax + 1e-30),
                                    d.cellSizeZ / (wMax + 1e-30)});
    timeStepSize = 0.35 * std::min(dtViscous, dtAdv);
}
