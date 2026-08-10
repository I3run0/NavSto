// =============================================================================
//  Physics.cpp  —  Navier-Stokes solver kernels (OpenMP).
//
//  Solves the 3-D incompressible Navier-Stokes equations:
//
//    ∂u/∂t + (u·∇)u = -∇p + (1/Re)∇²u      (momentum)
//             ∇·u   = 0                        (incompressibility)
//
//  Method: explicit fractional-step (projection) with UNIFAES exponential
//          scheme on a staggered Cartesian grid.
//  Pressure: red-black SOR solution of ∇²p = S (not plain Gauss-Seidel —
//            see docs/openmp-parallelization.md for why).
//
//  Started as a full copy of src/serial/Physics.cpp (deliberately, not
//  #ifdef-branched into the serial file — see docs/openmp-parallelization.md
//  for that tradeoff) with computeAccelerations() and solvePressurePoisson()
//  modified for parallelism; the remaining per-step kernels are unchanged
//  from serial.
//
//  Geometry and initial conditions are NOT duplicated here: they moved to
//  src/common/Setup.cpp, shared by every backend. This file holds only
//  per-step kernels — the code actually under measurement.
// =============================================================================

#include "Physics.hpp"
#include "Logger.hpp"
#include "Geometry.hpp"
#include "ViscosityModel.hpp"

#include <cmath>
#include <algorithm>
#include <stdexcept>

// ─── Internal helpers ────────────────────────────────────────────────────────

namespace {

// ---------------------------------------------------------------------------
//  UNIFAES weight π(Pe):  blends upwind and central differencing so that the
//  scheme is exact for 1-D steady advection-diffusion at any Péclet number.
// ---------------------------------------------------------------------------
void computeExponentialWeights(double localRe, double DPe,
                               double& pip,
                               double& coeffEast, double& coeffWest)
{
    if (std::abs(DPe) < 0.1) {
        // Polynomial approximation — numerically stable near Pe = 0
        pip = 1.0 / ((((0.05*DPe + 0.25)*DPe + 1.0)*DPe/6.0 + 0.5)*DPe + 1.0);
    } else if (std::abs(DPe) <= 200.0) {
        pip = DPe / (std::exp(DPe) - 1.0);  // exact Bernstein-Crank formula
    } else if (DPe > 200.0) {
        pip = 0.0;    // advection strongly left-to-right; east weight vanishes
    } else {
        pip = -DPe;   // advection strongly right-to-left
    }

    const double pim = DPe + pip;
    coeffEast = pip / localRe;
    coeffWest = pim / localRe;
}

// ---------------------------------------------------------------------------
//  UNIFAES cross-term blending weight ξ.
// ---------------------------------------------------------------------------
double computeQsi(double DPe, double pip, double xeOverDx)
{
    if (std::abs(DPe) < 0.01)
        return DPe * (1.0 - DPe * DPe / 60.0) / 12.0 + xeOverDx - 0.5;
    return (pip - 1.0) / DPe + xeOverDx;
}

// ---------------------------------------------------------------------------
//  Apply outlet and periodic boundary conditions to velocity.
// ---------------------------------------------------------------------------
void applyVelocityBCs(SimState& s)
{
    const int KKfim = (s.cfg.lateralCondition == LateralBC::SolidWall)
                    ? s.numCellsZm1 : s.cfg.numCellsZ;

    if (s.cfg.outletCondition == OutletBC::ZeroFirstDeriv) {
        for (int j = s.jLow[s.cfg.numCellsX]+1; j <= s.jHigh[s.cfg.numCellsX]-1; ++j)
            for (int k = 0; k <= KKfim; ++k) {
                s.velX(s.cfg.numCellsX, j, k) = s.velX(s.numCellsXm1, j, k);
                s.velY(s.cfg.numCellsX, j, k) = s.velY(s.numCellsXm1, j, k);
                s.velZ(s.cfg.numCellsX, j, k) = s.velZ(s.numCellsXm1, j, k);
            }
    }
    if (s.cfg.outletCondition == OutletBC::ZeroSecondDeriv) {
        for (int j = s.jLow[s.cfg.numCellsX]+1; j <= s.jHigh[s.cfg.numCellsX]-1; ++j)
            for (int k = 0; k <= KKfim; ++k) {
                s.velX(s.cfg.numCellsX, j, k) = 2.0*s.velX(s.numCellsXm1, j, k) - s.velX(s.cfg.numCellsX-2, j, k);
                s.velY(s.cfg.numCellsX, j, k) = 2.0*s.velY(s.numCellsXm1, j, k) - s.velY(s.cfg.numCellsX-2, j, k);
                s.velZ(s.cfg.numCellsX, j, k) = 2.0*s.velZ(s.numCellsXm1, j, k) - s.velZ(s.cfg.numCellsX-2, j, k);
            }
    }
    if (s.cfg.lateralCondition == LateralBC::Periodic) {
        for (int i = 1; i <= s.numCellsXm1; ++i)
            for (int j = s.jLow[i]+1; j <= s.jHigh[i]-1; ++j) {
                s.velX(i, j, 0) = s.velX(i, j, s.cfg.numCellsZ);
                s.velY(i, j, 0) = s.velY(i, j, s.cfg.numCellsZ);
                s.velZ(i, j, 0) = s.velZ(i, j, s.cfg.numCellsZ);
            }
    }
}

// ---------------------------------------------------------------------------
//  Red-black solvePressurePoisson helpers.
//
//  The serial solver (src/serial/Physics.cpp) interleaves Neumann
//  ghost-cell mirroring *inline*, mid-sweep, using whatever press(i,j,k)
//  holds at that point in the single i/k/j traversal. Red-black updates
//  cells out of that traversal order (all RED, then all BLACK), so the
//  mirroring can't stay inline -- it's factored into its own explicit
//  pass here, run twice per sweep (once before RED, once before BLACK):
//  BLACK's boundary cells need to see RED's just-updated interior values
//  reflected in the ghost layer, which a single mirror pass before both
//  colors wouldn't provide.
// ---------------------------------------------------------------------------

/// Mirrors Neumann boundary press() values into the ghost layer for the
/// CURRENT press field state. Periodic-Z doesn't need mirroring here --
/// updateColor() reads wrapped k indices directly instead.
///
/// v1 of this function (see git history) was fully serial: an earlier
/// attempt at `#pragma omp parallel for` on the `i` loop hit a real data
/// race, found via a determinism check (results varied run-to-run at a
/// fixed thread count under schedule(static), which shouldn't happen if
/// race-free): the `j==jLoopS`/`j==jLoopN` writes target
/// `press(i, j-1|j+1, k)` -- SAME row i as the writing thread -- but a
/// DIFFERENT thread processing i+1 can ALSO write into row i via its own
/// im=i target (`i+1 == s.iLow[j']+1` for some j'), and if that j'
/// coincides with this thread's j-1/j+1 target, two threads write the same
/// address with different values.
///
/// v2 (this version) separates the two write categories into two passes
/// with a barrier between (the implicit barrier at the end of the first
/// `#pragma omp for`, since neither loop uses `nowait`):
///   Pass A -- only the im/ip (cross-row) writes, which target row i-1 or
///             i+1, NEVER row i itself.
///   Pass B -- only the same-row writes (j-extension and, for SolidWall,
///             the k ghost layer), which by construction only ever target
///             (i, ...) -- this thread's OWN row -- so once Pass A has
///             fully completed (guaranteed by the barrier), no other
///             thread can still be writing into row i and Pass B is
///             race-free.
/// (Pass A itself has no cross-thread collision either: two different i's
/// im/ip writes could only target the same neighbor row if some row j had
/// iLow[j] > iHigh[j], i.e. an inverted/empty active span, which doesn't
/// happen for a valid active row -- same reasoning the original,
/// incomplete race analysis already relied on, see git history.)
///
/// MUST be called from inside an enclosing `#pragma omp parallel` region
/// (see solvePressurePoisson) -- both `#pragma omp for` below are orphaned
/// worksharing constructs, standard OpenMP, bind to whatever team is
/// currently active.
///
/// No ThreadSanitizer available in this WSL2 sandbox to verify formally
/// (same limitation as `perf`, see docs/roofline.md), so this was verified
/// with the SAME determinism methodology that caught the original race:
/// OMP_NUM_THREADS=12, config run 5 times, DilMax bit-identical every run,
/// matching the 1-thread/serial reference -- see
/// docs/openmp-parallelization.md for the actual run log.
void mirrorGhostCells(SimState& s) {
    const auto& cfg = s.cfg;

    // Pass A: cross-row writes only (into row i-1 or i+1, never row i).
    #pragma omp for schedule(static)
    for (int i = 1; i <= cfg.numCellsX; ++i) {
        const int im = i - 1, ip = i + 1;
        int jLoopS, jLoopN;
        if      (s.jLow[im]  >= s.jLow[i])  jLoopS = s.jLow[i]+1;
        else                                  jLoopS = s.jLow[i];
        if      (s.jHigh[im] <= s.jHigh[i]) jLoopN = s.jHigh[i];
        else                                  jLoopN = s.jHigh[i]+1;
        if (i == s.degreeIndex2) { jLoopS = s.jLow[im]+1; jLoopN = s.jHigh[im]; }

        for (int k = 1; k <= cfg.numCellsZ; ++k) {
            for (int j = jLoopS; j <= jLoopN; ++j) {
                if (i == 1 || i == s.iLow[j]+1)             s.press(im, j, k) = s.press(i, j, k);
                if (i == cfg.numCellsX || i == s.iHigh[j])  s.press(ip, j, k) = s.press(i, j, k);
            }
        }
    }
    // implicit barrier here -- no nowait on the #pragma omp for above.

    // Pass B: same-row writes only (into row i itself).
    #pragma omp for schedule(static)
    for (int i = 1; i <= cfg.numCellsX; ++i) {
        const int im = i - 1;
        int jLoopS, jLoopN;
        if      (s.jLow[im]  >= s.jLow[i])  jLoopS = s.jLow[i]+1;
        else                                  jLoopS = s.jLow[i];
        if      (s.jHigh[im] <= s.jHigh[i]) jLoopN = s.jHigh[i];
        else                                  jLoopN = s.jHigh[i]+1;
        if (i == s.degreeIndex2) { jLoopS = s.jLow[im]+1; jLoopN = s.jHigh[im]; }

        for (int k = 1; k <= cfg.numCellsZ; ++k) {
            for (int j = jLoopS; j <= jLoopN; ++j) {
                if (j == jLoopS)                             s.press(i, j-1, k) = s.press(i, j, k);
                if (j == jLoopN)                             s.press(i, j+1, k) = s.press(i, j, k);
                if (cfg.lateralCondition == LateralBC::SolidWall) {
                    if (k == 1)             s.press(i, j, k-1) = s.press(i, j, k);
                    if (k == cfg.numCellsZ) s.press(i, j, k+1) = s.press(i, j, k);
                }
            }
        }
    }
}

enum class RBColor { Red, Black };

/// Updates every cell of one color across all active rows — embarrassingly
/// parallel: by construction every cell of `color` only reads neighbors of
/// the OTHER color, all fixed for the duration of this call. Correctness
/// (same fixed point as the original single-sweep Gauss-Seidel/SOR,
/// verified via scripts/validate.py's red-black tier) was established
/// BEFORE adding the #pragma omp below -- see docs/openmp-parallelization.md
/// for why serial-first.
///
/// MUST be called from inside an enclosing `#pragma omp parallel` region
/// (see solvePressurePoisson) -- the `#pragma omp for` below is an
/// orphaned worksharing construct, standard OpenMP, binds to whatever team
/// is currently active. Iterates the caller's active-row list (one (i,j)
/// row list drives BOTH colors -- see Geometry.hpp) with a direct strided inner
/// k-loop instead of dereferencing a per-cell index list: replaces a
/// gather (defeats prefetching) with unit-stride-2 access, which is what
/// the roofline's "5% of bandwidth ceiling despite low AI" finding pointed
/// at (latency-bound via indirection, not bandwidth-bound).
void updateColor(SimState& s, const std::vector<RowIndex>& activeRows, RBColor color,
                  double cX, double cY, double cZ, double invDiag,
                  int iRef, int jRef, int kRef, double pRef) {
    const auto& cfg = s.cfg;
    const long n = static_cast<long>(activeRows.size());
    const int wantParity = (color == RBColor::Red) ? 0 : 1;
    // schedule(static): deterministic at a fixed thread count (needed by
    // scripts/validate_parallel.py's determinism check). Per-row cost is
    // uniform (~numCellsZ/2 cells each), no reason to prefer dynamic.
    #pragma omp for schedule(static)
    for (long idx = 0; idx < n; ++idx) {
        const int i = activeRows[idx].i;
        const int j = activeRows[idx].j;
        const int im = i-1, ip = i+1, jm = j-1, jp = j+1;
        // (i+j+k) even => red. kStart is the smallest k in [1,2] with the
        // right (i+j+k) parity for this color; step 2 covers the rest.
        const int kStart = (((i + j) % 2) == wantParity) ? 2 : 1;
        for (int k = kStart; k <= cfg.numCellsZ; k += 2) {
            int km = k-1, kp = k+1;
            if (cfg.lateralCondition != LateralBC::SolidWall) {
                if (k == 1)             km = cfg.numCellsZ;
                if (k == cfg.numCellsZ) kp = 1;
            }

            if (i == iRef && j == jRef && k == kRef) {
                s.press(i, j, k) = pRef;
                continue;
            }
            double pNew = (cY*(s.press(i,jp,k) + s.press(i,jm,k))
                        + cX*(s.press(ip,j,k) + s.press(im,j,k))
                        + cZ*(s.press(i,j,kp) + s.press(i,j,km))
                        - s.pressureSource(i,j,k)) * invDiag;
            if ((i==1 || i==cfg.numCellsX) && (j==s.jLow[i]+1 || j==s.jHigh[i])) {
                pNew -= s.pressureSource(i,j,k) * invDiag;
                if (cfg.lateralCondition == LateralBC::SolidWall && (k==1 || k==cfg.numCellsZ))
                    pNew -= 2.0 * s.pressureSource(i,j,k) * invDiag;
            }
            s.press(i, j, k) += cfg.sorOmega * (pNew - s.press(i, j, k));
        }
    }
}

} // anonymous namespace

// ════════════════════════════════════════════════════════════════════════════
//  PUBLIC API
// ════════════════════════════════════════════════════════════════════════════

// ---------------------------------------------------------------------------
//  computeAccelerations — UNIFAES advective + viscous accelerations
// ---------------------------------------------------------------------------
// =============================================================================
//  computeAccelerations  —  UNIFAES advection + viscous terms (exact as CoCOEF)
// =============================================================================
void computeAccelerations(SimState& s)
{
    const auto& cfg = s.cfg;
    const bool periodic = (cfg.lateralCondition == LateralBC::Periodic);
    const int KKfim = periodic ? cfg.numCellsZ : s.numCellsZm1;

    // Zero all acceleration arrays
    s.accelX.fill(0.0);
    s.accelY.fill(0.0);
    s.accelZ.fill(0.0);

    // Helper to access offset arrays (logical idx -> physical idx+1). Takes
    // a raw double* (not std::vector<double>&) so it works uniformly on
    // whatever per-thread slice pointer each thread sets up below.
    auto VM = [](double* v, int idx) -> double& { return v[idx+1]; };

    // Same offset convention as VM, but for the X-sweep's 2-D (i,k) scratch:
    // logical i -> physical i+1 (as VM), k (1..KKfim) -> physical k-1.
    const int kStride = s.ext.accel.scratchKLen;
    auto VM2 = [kStride](double* v, int iLogical, int k) -> double& {
        return v[static_cast<std::size_t>(iLogical + 1) * kStride + (k - 1)];
    };

    // computeAccelerations() scratch buffers are SimState-owned (sized once
    // in allocateFields(), not std::vector-allocated fresh every call — see
    // the field comments there) but PER-THREAD: each sweep below is
    // parallelized over its outer loop (#pragma omp for), and every thread
    // needs its own private slice of these buffers -- concurrent threads
    // writing the SAME shared buffer would race. Each thread computes its
    // own slice pointers once at the top of this parallel region (not
    // per-outer-loop-iteration) and reuses them for all three sweeps.
    #pragma omp parallel
    {
#ifdef _OPENMP
        const int tid = omp_get_thread_num();
#else
        const int tid = 0;
#endif
        double* ppin = s.ext.accel.ppin.data() + tid * s.ext.accel.scratchLenPerThread;
        double* ppis = s.ext.accel.ppis.data() + tid * s.ext.accel.scratchLenPerThread;
        double* ppiu = s.ext.accel.ppiu.data() + tid * s.ext.accel.scratchLenPerThread;
        double* ppid = s.ext.accel.ppid.data() + tid * s.ext.accel.scratchLenPerThread;
        double* qsin = s.ext.accel.qsin.data() + tid * s.ext.accel.scratchLenPerThread;
        double* qsiu = s.ext.accel.qsiu.data() + tid * s.ext.accel.scratchLenPerThread;
        double* Ku   = s.ext.accel.Ku.data()   + tid * s.ext.accel.scratchLenPerThread;
        double* Kv   = s.ext.accel.Kv.data()   + tid * s.ext.accel.scratchLenPerThread;
        double* Kw   = s.ext.accel.Kw.data()   + tid * s.ext.accel.scratchLenPerThread;

        // X-sweep-only 2-D (i,k) scratch — see SimState.hpp field comments
        // and docs/serial-optimization-loop-order.md. Replaces the 1-D
        // ppie/ppiw/qsie buffers (and this sweep's private use of Ku/Kv/Kw,
        // which the Y/Z sweeps below still use in their own 1-D slices).
        double* ppieXK = s.ext.accel.ppieXK.data() + tid * s.ext.accel.xk2DLenPerThread;
        double* ppiwXK = s.ext.accel.ppiwXK.data() + tid * s.ext.accel.xk2DLenPerThread;
        double* qsieXK = s.ext.accel.qsieXK.data() + tid * s.ext.accel.xk2DLenPerThread;
        double* KuXK   = s.ext.accel.KuXK.data()   + tid * s.ext.accel.xk2DLenPerThread;
        double* KvXK   = s.ext.accel.KvXK.data()   + tid * s.ext.accel.xk2DLenPerThread;
        double* KwXK   = s.ext.accel.KwXK.data()   + tid * s.ext.accel.xk2DLenPerThread;

    // ---------- Direction X ----------
    // Restructured (docs/serial-optimization-loop-order.md) to loop
    // pass-major -- all i, all k, per pass -- instead of the original
    // k-major structure (all 5 passes, per k). i must stay the outer loop
    // within each pass since the pass sequence is a genuine recurrence
    // along i (each pass depends on the previous pass's results at
    // neighboring i), but k has no such coupling -- nothing in this sweep
    // references k+-1, only i+-1 -- so k can freely become the innermost,
    // unit-stride loop (GridField stores k fastest-varying) instead of a
    // fixed outer index. Same math, same per-(i,k) result, only the order
    // work happens in relative to OTHER (i,k) pairs changes -- verified via
    // tests/serial/GoldenFieldTest.cpp (captured from the pre-restructuring
    // implementation).
    const double invDx2 = 1.0 / (cfg.cellSizeX * cfg.cellSizeX);
    // Parallel over j: each j-plane's 5-pass computation is self-contained
    // (uses only this thread's private scratch slice), so different j
    // values can run on different threads with no cross-thread dependency.
    // schedule(static) for deterministic results at a fixed thread count
    // (needed by scripts/validate_parallel.py's determinism check) and
    // because the per-plane cost is fairly uniform for this geometry.
    #pragma omp for schedule(static)
    for (int j = 1; j <= s.numCellsYm1; ++j) {
        const int iStart = s.iLow[j];
        const int iEnd   = s.iHigh[j];

        // Pass 1: face coefficients (between i and i+1)
        for (int i = iStart; i <= iEnd-1; ++i) {
            const double localRe = 1.0 / effectiveInvRe(s, i);
            for (int k = 1; k <= KKfim; ++k) {
                const double uFace = 0.5 * (s.velX(i+1, j, k) + s.velX(i, j, k));
                const double DPe = localRe * uFace * cfg.cellSizeX;
                double pip, cip, cim;
                computeExponentialWeights(localRe, DPe, pip, cip, cim);
                VM2(ppieXK, i+1, k) = cip;      // east coefficient at face i+0.5
                VM2(ppiwXK, i+1+1, k) = cim;    // west coefficient (ip+1, ip=i+1) -- see 1-D version's comment history
                VM2(qsieXK, i+1, k) = computeQsi(DPe, pip, 0.5);
            }
        }
        // Pass 2: diffusive part of Au, Av, Aw (first part)
        for (int i = iStart+1; i <= iEnd-1; ++i) {
            for (int k = 1; k <= KKfim; ++k) {
                const double coeff = invDx2;
                s.accelX(i, j, k) += (VM2(ppieXK,i+1,k)*(s.velX(i+1,j,k)-s.velX(i,j,k))
                                     + VM2(ppiwXK,i+1,k)*(s.velX(i-1,j,k)-s.velX(i,j,k))) * coeff;
                s.accelY(i, j, k) += (VM2(ppieXK,i+1,k)*(s.velY(i+1,j,k)-s.velY(i,j,k))
                                     + VM2(ppiwXK,i+1,k)*(s.velY(i-1,j,k)-s.velY(i,j,k))) * coeff;
                s.accelZ(i, j, k) += (VM2(ppieXK,i+1,k)*(s.velZ(i+1,j,k)-s.velZ(i,j,k))
                                     + VM2(ppiwXK,i+1,k)*(s.velZ(i-1,j,k)-s.velZ(i,j,k))) * coeff;
            }
        }
        // Pass 3: cross‑term correction (K * qsi)
        for (int i = iStart+1; i <= iEnd-1; ++i) {
            const double localRe = 1.0 / effectiveInvRe(s, i);
            for (int k = 1; k <= KKfim; ++k) {
                const double uCell = s.velX(i, j, k);
                const double DPe = localRe * uCell * cfg.cellSizeX;
                double pip, cip, cim;
                computeExponentialWeights(localRe, DPe, pip, cip, cim);
                cip *= invDx2;
                cim *= invDx2;
                VM2(KuXK, i+1, k) = cip*(s.velX(i,j,k)-s.velX(i+1,j,k)) + cim*(s.velX(i,j,k)-s.velX(i-1,j,k));
                VM2(KvXK, i+1, k) = cip*(s.velY(i,j,k)-s.velY(i+1,j,k)) + cim*(s.velY(i,j,k)-s.velY(i-1,j,k));
                VM2(KwXK, i+1, k) = cip*(s.velZ(i,j,k)-s.velZ(i+1,j,k)) + cim*(s.velZ(i,j,k)-s.velZ(i-1,j,k));
            }
        }
        // Extrapolate Ku,Kv,Kw at boundaries -- per k, unchanged formula
        for (int k = 1; k <= KKfim; ++k) {
            VM2(KuXK, iStart+1, k) = 2.0*VM2(KuXK, iStart+2, k) - VM2(KuXK, iStart+3, k);
            VM2(KvXK, iStart+1, k) = 2.0*VM2(KvXK, iStart+2, k) - VM2(KvXK, iStart+3, k);
            VM2(KwXK, iStart+1, k) = 2.0*VM2(KwXK, iStart+2, k) - VM2(KwXK, iStart+3, k);
            VM2(KuXK, iEnd+1, k)   = 2.0*VM2(KuXK, iEnd, k)   - VM2(KuXK, iEnd-1, k);
            VM2(KvXK, iEnd+1, k)   = 2.0*VM2(KvXK, iEnd, k)   - VM2(KvXK, iEnd-1, k);
            VM2(KwXK, iEnd+1, k)   = 2.0*VM2(KwXK, iEnd, k)   - VM2(KwXK, iEnd-1, k);
        }
        // Average to faces
        for (int i = iStart; i <= iEnd-1; ++i) {
            for (int k = 1; k <= KKfim; ++k) {
                VM2(KuXK, i+1, k) = 0.5*(VM2(KuXK, i+1, k) + VM2(KuXK, i+2, k));
                VM2(KvXK, i+1, k) = 0.5*(VM2(KvXK, i+1, k) + VM2(KvXK, i+2, k));
                VM2(KwXK, i+1, k) = 0.5*(VM2(KwXK, i+1, k) + VM2(KwXK, i+2, k));
            }
        }
        // Subtract cross‑term divergence
        for (int i = iStart+1; i <= iEnd-1; ++i) {
            for (int k = 1; k <= KKfim; ++k) {
                s.accelX(i, j, k) -= (VM2(KuXK,i+1,k)*VM2(qsieXK,i+1,k) - VM2(KuXK,i,k)*VM2(qsieXK,i,k));
                s.accelY(i, j, k) -= (VM2(KvXK,i+1,k)*VM2(qsieXK,i+1,k) - VM2(KvXK,i,k)*VM2(qsieXK,i,k));
                s.accelZ(i, j, k) -= (VM2(KwXK,i+1,k)*VM2(qsieXK,i+1,k) - VM2(KwXK,i,k)*VM2(qsieXK,i,k));
            }
        }
    }

    // ---------- Direction Y ----------
    const double invDy2 = 1.0 / (cfg.cellSizeY * cfg.cellSizeY);
    // Parallel over i -- same reasoning as the X-sweep's #pragma omp for
    // above, just over the Y-sweep's outer dimension instead.
    #pragma omp for schedule(static)
    for (int i = 1; i <= s.numCellsXm1; ++i) {
        const int jStart = s.jLow[i];
        const int jEnd   = s.jHigh[i];
        const double invRe = effectiveInvRe(s, i);
        const double localRe = 1.0 / invRe;
        for (int k = 1; k <= KKfim; ++k) {
            for (int j = jStart; j <= jEnd-1; ++j) {
                const double vFace = 0.5 * (s.velY(i, j+1, k) + s.velY(i, j, k));
                const double DPe = localRe * vFace * cfg.cellSizeY;
                double pip, cin, cis;
                computeExponentialWeights(localRe, DPe, pip, cin, cis);
                VM(ppin, j+1) = cin;
                VM(ppis, j+2) = cis;
                VM(qsin, j+1) = computeQsi(DPe, pip, 0.5);
            }
            for (int j = jStart+1; j <= jEnd-1; ++j) {
                const double coeff = invDy2;
                s.accelX(i, j, k) += (VM(ppin,j+1)*(s.velX(i,j+1,k)-s.velX(i,j,k))
                                     + VM(ppis,j+1)*(s.velX(i,j-1,k)-s.velX(i,j,k))) * coeff;
                s.accelY(i, j, k) += (VM(ppin,j+1)*(s.velY(i,j+1,k)-s.velY(i,j,k))
                                     + VM(ppis,j+1)*(s.velY(i,j-1,k)-s.velY(i,j,k))) * coeff;
                s.accelZ(i, j, k) += (VM(ppin,j+1)*(s.velZ(i,j+1,k)-s.velZ(i,j,k))
                                     + VM(ppis,j+1)*(s.velZ(i,j-1,k)-s.velZ(i,j,k))) * coeff;
            }
            for (int j = jStart+1; j <= jEnd-1; ++j) {
                const double vCell = s.velY(i, j, k);
                const double DPe = localRe * vCell * cfg.cellSizeY;
                double pip, cin, cis;
                computeExponentialWeights(localRe, DPe, pip, cin, cis);
                cin *= invDy2;  cis *= invDy2;
                VM(Ku, j+1) = cin*(s.velX(i,j,k)-s.velX(i,j+1,k)) + cis*(s.velX(i,j,k)-s.velX(i,j-1,k));
                VM(Kv, j+1) = cin*(s.velY(i,j,k)-s.velY(i,j+1,k)) + cis*(s.velY(i,j,k)-s.velY(i,j-1,k));
                VM(Kw, j+1) = cin*(s.velZ(i,j,k)-s.velZ(i,j+1,k)) + cis*(s.velZ(i,j,k)-s.velZ(i,j-1,k));
            }
            VM(Ku, jStart+1) = 2.0*VM(Ku, jStart+2) - VM(Ku, jStart+3);
            VM(Kv, jStart+1) = 2.0*VM(Kv, jStart+2) - VM(Kv, jStart+3);
            VM(Kw, jStart+1) = 2.0*VM(Kw, jStart+2) - VM(Kw, jStart+3);
            VM(Ku, jEnd+1)   = 2.0*VM(Ku, jEnd)   - VM(Ku, jEnd-1);
            VM(Kv, jEnd+1)   = 2.0*VM(Kv, jEnd)   - VM(Kv, jEnd-1);
            VM(Kw, jEnd+1)   = 2.0*VM(Kw, jEnd)   - VM(Kw, jEnd-1);
            for (int j = jStart; j <= jEnd-1; ++j) {
                VM(Ku, j+1) = 0.5*(VM(Ku, j+1) + VM(Ku, j+2));
                VM(Kv, j+1) = 0.5*(VM(Kv, j+1) + VM(Kv, j+2));
                VM(Kw, j+1) = 0.5*(VM(Kw, j+1) + VM(Kw, j+2));
            }
            for (int j = jStart+1; j <= jEnd-1; ++j) {
                s.accelX(i, j, k) -= (VM(Ku,j+1)*VM(qsin,j+1) - VM(Ku,j)*VM(qsin,j));
                s.accelY(i, j, k) -= (VM(Kv,j+1)*VM(qsin,j+1) - VM(Kv,j)*VM(qsin,j));
                s.accelZ(i, j, k) -= (VM(Kw,j+1)*VM(qsin,j+1) - VM(Kw,j)*VM(qsin,j));
            }
        }
    }

    // ---------- Direction Z ----------
    const double invDz2 = 1.0 / (cfg.cellSizeZ * cfg.cellSizeZ);
    // Parallel over i -- same reasoning as the X/Y sweeps' #pragma omp for
    // above. Implicit barrier at the end of this omp-for (no `nowait`) is
    // required: this is the last sweep before the parallel region closes,
    // and the code after it (periodic Z copy, outside this region) reads
    // accelX/Y/Z that all three sweeps wrote.
    #pragma omp for schedule(static)
    for (int i = 1; i <= s.numCellsXm1; ++i) {
        const double invRe = effectiveInvRe(s, i);
        const double localRe = 1.0 / invRe;
        for (int j = s.jLow[i]+1; j <= s.jHigh[i]-1; ++j) {
            for (int k = 0; k <= KKfim; ++k) {
                int kp = (periodic && k == cfg.numCellsZ) ? 1 : k+1;
                const double wFace = 0.5 * (s.velZ(i, j, kp) + s.velZ(i, j, k));
                const double DPe = localRe * wFace * cfg.cellSizeZ;
                double pip, ciu, cid;
                computeExponentialWeights(localRe, DPe, pip, ciu, cid);
                VM(ppiu, k+1) = ciu;
                VM(ppid, kp+1) = cid;
                VM(qsiu, k+1) = computeQsi(DPe, pip, 0.5);
            }
            for (int k = 1; k <= KKfim; ++k) {
                int kp = (periodic && k == cfg.numCellsZ) ? 1 : k+1;
                int km = (periodic && k == 1) ? cfg.numCellsZ : k-1;
                const double coeff = invDz2;
                s.accelX(i, j, k) += (VM(ppiu,k+1)*(s.velX(i,j,kp)-s.velX(i,j,k))
                                     + VM(ppid,k+1)*(s.velX(i,j,km)-s.velX(i,j,k))) * coeff;
                s.accelY(i, j, k) += (VM(ppiu,k+1)*(s.velY(i,j,kp)-s.velY(i,j,k))
                                     + VM(ppid,k+1)*(s.velY(i,j,km)-s.velY(i,j,k))) * coeff;
                s.accelZ(i, j, k) += (VM(ppiu,k+1)*(s.velZ(i,j,kp)-s.velZ(i,j,k))
                                     + VM(ppid,k+1)*(s.velZ(i,j,km)-s.velZ(i,j,k))) * coeff;
            }
            for (int k = 1; k <= KKfim; ++k) {
                int kp = (periodic && k == cfg.numCellsZ) ? 1 : k+1;
                int km = (periodic && k == 1) ? cfg.numCellsZ : k-1;
                const double wCell = s.velZ(i, j, k);
                const double DPe = localRe * wCell * cfg.cellSizeZ;
                double pip, ciu, cid;
                computeExponentialWeights(localRe, DPe, pip, ciu, cid);
                ciu *= invDz2;  cid *= invDz2;
                VM(Ku, k+1) = ciu*(s.velX(i,j,k)-s.velX(i,j,kp)) + cid*(s.velX(i,j,k)-s.velX(i,j,km));
                VM(Kv, k+1) = ciu*(s.velY(i,j,k)-s.velY(i,j,kp)) + cid*(s.velY(i,j,k)-s.velY(i,j,km));
                VM(Kw, k+1) = ciu*(s.velZ(i,j,k)-s.velZ(i,j,kp)) + cid*(s.velZ(i,j,k)-s.velZ(i,j,km));
            }
            if (!periodic) { // Dirichlet in z: extrapolate
                VM(Ku, 0+1) = 2.0*VM(Ku,1+1) - VM(Ku,2+1);
                VM(Kv, 0+1) = 2.0*VM(Kv,1+1) - VM(Kv,2+1);
                VM(Kw, 0+1) = 2.0*VM(Kw,1+1) - VM(Kw,2+1);
                VM(Ku, cfg.numCellsZ+1) = 2.0*VM(Ku, s.numCellsZm1+1) - VM(Ku, cfg.numCellsZ-1+1);
                VM(Kv, cfg.numCellsZ+1) = 2.0*VM(Kv, s.numCellsZm1+1) - VM(Kv, cfg.numCellsZ-1+1);
                VM(Kw, cfg.numCellsZ+1) = 2.0*VM(Kw, s.numCellsZm1+1) - VM(Kw, cfg.numCellsZ-1+1);
            }
            for (int k = 0; k <= KKfim; ++k) {
                int kp = (periodic && k == cfg.numCellsZ) ? 1 : k+1;
                VM(Ku, k+1) = 0.5*(VM(Ku, k+1) + VM(Ku, kp+1));
                VM(Kv, k+1) = 0.5*(VM(Kv, k+1) + VM(Kv, kp+1));
                VM(Kw, k+1) = 0.5*(VM(Kw, k+1) + VM(Kw, kp+1));
            }
            for (int k = 1; k <= KKfim; ++k) {
                int km = (periodic && k == 1) ? cfg.numCellsZ : k-1;
                s.accelX(i, j, k) -= (VM(Ku,k+1)*VM(qsiu,k+1) - VM(Ku,km+1)*VM(qsiu,km+1));
                s.accelY(i, j, k) -= (VM(Kv,k+1)*VM(qsiu,k+1) - VM(Kv,km+1)*VM(qsiu,km+1));
                s.accelZ(i, j, k) -= (VM(Kw,k+1)*VM(qsiu,k+1) - VM(Kw,km+1)*VM(qsiu,km+1));
            }
        }
    }
    } // end #pragma omp parallel

    // Periodic copy in z (cheap, O(N^(2/3)) -- left serial rather than
    // parallelized inside the region above)
    if (periodic) {
        for (int i = 1; i <= s.numCellsXm1; ++i)
            for (int j = s.jLow[i]+1; j <= s.jHigh[i]-1; ++j) {
                s.accelX(i, j, 0) = s.accelX(i, j, cfg.numCellsZ);
                s.accelY(i, j, 0) = s.accelY(i, j, cfg.numCellsZ);
                s.accelZ(i, j, 0) = s.accelZ(i, j, cfg.numCellsZ);
            }
    }
}

// ---------------------------------------------------------------------------
//  buildPressureSource — S = ∇·u/dt + ∇·A  (8-corner staggered averages)
// ---------------------------------------------------------------------------
void buildPressureSource(SimState& s)
{
    const auto& cfg = s.cfg;

    // Zero acceleration on all boundary (wall) nodes
    for (int i = 0; i <= cfg.numCellsX; ++i)
        for (int k = 0; k <= cfg.numCellsZ; ++k) {
            const int jB = s.jLow[i], jT = s.jHigh[i];
            s.accelX(i,jB,k) = s.accelX(i,jT,k) = 0.0;
            s.accelY(i,jB,k) = s.accelY(i,jT,k) = 0.0;
            s.accelZ(i,jB,k) = s.accelZ(i,jT,k) = 0.0;
        }
    for (int j = 0; j <= cfg.numCellsY; ++j)
        for (int k = 0; k <= cfg.numCellsZ; ++k) {
            const int iL = s.iLow[j], iR = s.iHigh[j];
            s.accelX(iL,j,k) = s.accelX(iR,j,k) = 0.0;
            s.accelY(iL,j,k) = s.accelY(iR,j,k) = 0.0;
            s.accelZ(iL,j,k) = s.accelZ(iR,j,k) = 0.0;
        }

    const double qInvDx = 0.25 / cfg.cellSizeX;
    const double qInvDy = 0.25 / cfg.cellSizeY;
    const double qInvDz = 0.25 / cfg.cellSizeZ;
    const double invDt  = 1.0  / s.timeStepSize;

    for (int i = 1; i <= cfg.numCellsX; ++i) {
        const int im  = i - 1;
        const int jS  = (i != s.degreeIndex2) ? s.jLow[i]+1  : s.jLow[im]+1;
        const int jN  = (i != s.degreeIndex2) ? s.jHigh[i]   : s.jHigh[im];

        for (int j = jS; j <= jN; ++j) {
            const int jm = j - 1;
            for (int k = 1; k <= cfg.numCellsZ; ++k) {
                int km = k - 1;
                if (cfg.lateralCondition == LateralBC::Periodic && k == 1) km = cfg.numCellsZ;

                const double divU =
                    (s.velX(i,j,k) - s.velX(im,j,k) + s.velX(i,jm,k) - s.velX(im,jm,k)
                   + s.velX(i,j,km) - s.velX(im,j,km) + s.velX(i,jm,km) - s.velX(im,jm,km)) * qInvDx;
                const double divV =
                    (s.velY(i,j,k) - s.velY(i,jm,k) + s.velY(im,j,k) - s.velY(im,jm,k)
                   + s.velY(i,j,km) - s.velY(i,jm,km) + s.velY(im,j,km) - s.velY(im,jm,km)) * qInvDy;
                const double divW =
                    (s.velZ(i,j,k) + s.velZ(i,jm,k) + s.velZ(im,j,k) + s.velZ(im,jm,k)
                   - s.velZ(i,j,km) - s.velZ(i,jm,km) - s.velZ(im,j,km) - s.velZ(im,jm,km)) * qInvDz;

                const double divAu =
                    (s.accelX(i,j,k) - s.accelX(im,j,k) + s.accelX(i,jm,k) - s.accelX(im,jm,k)
                   + s.accelX(i,j,km) - s.accelX(im,j,km) + s.accelX(i,jm,km) - s.accelX(im,jm,km)) * qInvDx;
                const double divAv =
                    (s.accelY(i,j,k) - s.accelY(i,jm,k) + s.accelY(im,j,k) - s.accelY(im,jm,k)
                   + s.accelY(i,j,km) - s.accelY(i,jm,km) + s.accelY(im,j,km) - s.accelY(im,jm,km)) * qInvDy;
                const double divAw =
                    (s.accelZ(i,j,k) + s.accelZ(i,jm,k) + s.accelZ(im,j,k) + s.accelZ(im,jm,k)
                   - s.accelZ(i,j,km) - s.accelZ(i,jm,km) - s.accelZ(im,j,km) - s.accelZ(im,jm,km)) * qInvDz;

                s.pressureSource(i, j, k) = (divU + divV + divW) * invDt
                                          + (divAu + divAv + divAw);
            }
        }
    }
}

// ---------------------------------------------------------------------------
//  solvePressurePoisson — red-black SOR for ∇²p = S
//
//  The serial solver's plain Gauss-Seidel/SOR has a genuine loop-carried
//  dependency (each cell reads a same-sweep-updated neighbor), which can't
//  be correctly parallelized as-is. Red-black splits cells by (i+j+k)
//  parity so each color's update is embarrassingly parallel -- see
//  Geometry.hpp and mirrorGhostCells()/updateColor() above.
//  Converges to the SAME fixed point as the serial version via a
//  DIFFERENT iteration path (not identical intermediate values) --
//  verified via scripts/validate.py's red-black tier, not a golden-field
//  diff.
//
//  One #pragma omp parallel region wraps the ENTIRE sweep loop (all
//  numPressureIter sweeps), not one region per updateColor() call. The
//  first version of this function opened/closed a thread team on every
//  single updateColor() call -- 2 colors x numPressureIter sweeps = 10
//  team spawn/joins per solve at the default numPressureIter=5, every
//  single timestep. That fixed per-spawn overhead doesn't scale with grid
//  size, which is a plausible driver of the originally-measured
//  small-grid-regresses-at-high-thread-count scaling (see
//  docs/openmp-parallelization.md's Performance section, pre-this-change
//  numbers). mirrorGhostCells() is now ALSO parallel internally (two-pass,
//  see its own comment above) -- called directly by every thread here
//  (not wrapped in `#pragma omp single`), its own `#pragma omp for`
//  passes provide the necessary synchronization.
// ---------------------------------------------------------------------------
void solvePressurePoisson(SimState& s)
{
    const auto& cfg = s.cfg;
    const double cX = 1.0 / s.cellSizeXsq;
    const double cY = 1.0 / s.cellSizeYsq;
    const double cZ = 1.0 / s.cellSizeZsq;
    const double invDiag = 0.5 / (cX + cY + cZ);

    // Pin one reference pressure node to remove the null-space
    const int iRef = cfg.numCellsX;
    const int jRef = (s.jHigh[cfg.numCellsX] + s.jLow[cfg.numCellsX]) / 2;
    const int kRef = (cfg.numCellsZ + 1) / 2;
    const double pRef = s.press(iRef, jRef, kRef);

    // Built once and reused for the run — geometry is fixed after
    // initSimulation(). The list, and this guard, live in THIS backend's
    // Extras (src/openmp/BackendConfig.hpp); SimState used to carry both for
    // every backend, with a separate `redBlackBuilt` bool that emptiness
    // makes redundant.
    if (s.ext.activeRows.empty())
        s.ext.activeRows = buildActiveRows(s);

    #pragma omp parallel
    {
        for (int sweep = 0; sweep < cfg.numPressureIter; ++sweep) {
            mirrorGhostCells(s);
            updateColor(s, s.ext.activeRows, RBColor::Red, cX, cY, cZ, invDiag, iRef, jRef, kRef, pRef);
            mirrorGhostCells(s);
            updateColor(s, s.ext.activeRows, RBColor::Black, cX, cY, cZ, invDiag, iRef, jRef, kRef, pRef);
        }
    }
}

// ---------------------------------------------------------------------------
//  updateVelocities — projection step  u_new = u_old + dt*(A - ∇p)
// ---------------------------------------------------------------------------
void updateVelocities(SimState& s)
{
    const auto& cfg = s.cfg;
    const double qInvDx = 0.25 / cfg.cellSizeX;
    const double qInvDy = 0.25 / cfg.cellSizeY;
    const double qInvDz = 0.25 / cfg.cellSizeZ;
    const double dt_eff = s.useHalfStep ? 0.5 * s.timeStepSize : s.timeStepSize;
    const int KKfim = (cfg.lateralCondition == LateralBC::SolidWall) ? s.numCellsZm1 : cfg.numCellsZ;
    s.maxVelocityChange = 0.0;

    for (int i = 1; i <= s.numCellsXm1; ++i) {
        const int ip = i + 1;
        for (int j = s.jLow[i]+1; j <= s.jHigh[i]-1; ++j) {
            const int jp = j + 1;
            for (int k = 1; k <= KKfim; ++k) {
                const int kp = (k < cfg.numCellsZ) ? k+1 : 1;

                const double dpu = (s.press(ip,j,kp) - s.press(i,j,kp) + s.press(ip,jp,kp) - s.press(i,jp,kp)
                                  + s.press(ip,j,k)  - s.press(i,j,k)  + s.press(ip,jp,k)  - s.press(i,jp,k)) * qInvDx;
                const double dpv = (s.press(i,jp,kp) - s.press(i,j,kp) + s.press(ip,jp,kp) - s.press(ip,j,kp)
                                  + s.press(i,jp,k)  - s.press(i,j,k)  + s.press(ip,jp,k)  - s.press(ip,j,k)) * qInvDy;
                const double dpw = (s.press(i,jp,kp) + s.press(i,j,kp) + s.press(ip,jp,kp) + s.press(ip,j,kp)
                                  - s.press(i,jp,k)  - s.press(i,j,k)  - s.press(ip,jp,k)  - s.press(ip,j,k)) * qInvDz;

                const double du = s.accelX(i,j,k) - dpu;
                const double dv = s.accelY(i,j,k) - dpv;
                const double dw = s.accelZ(i,j,k) - dpw;

                s.velX(i,j,k) += du * dt_eff;
                s.velY(i,j,k) += dv * dt_eff;
                s.velZ(i,j,k) += dw * dt_eff;

                const double mag = std::sqrt(du*du + dv*dv + dw*dw);
                s.maxVelocityChange = std::max(s.maxVelocityChange, mag);
            }
            if (cfg.lateralCondition == LateralBC::Periodic) {
                s.velX(i,j,0) = s.velX(i,j,cfg.numCellsZ);
                s.velY(i,j,0) = s.velY(i,j,cfg.numCellsZ);
                s.velZ(i,j,0) = s.velZ(i,j,cfg.numCellsZ);
            }
        }
    }
    applyVelocityBCs(s);
}

// ---------------------------------------------------------------------------
//  computeMomentumResidual — L∞ and L² of (A - ∇p)
// ---------------------------------------------------------------------------
void computeMomentumResidual(SimState& s)
{
    const auto& cfg = s.cfg;
    s.scratchField.fill(0.0);
    const double qInvDx = 0.25 / cfg.cellSizeX;
    const double qInvDy = 0.25 / cfg.cellSizeY;
    const double qInvDz = 0.25 / cfg.cellSizeZ;
    const int KKfim = (cfg.lateralCondition == LateralBC::SolidWall) ? s.numCellsZm1 : cfg.numCellsZ;

    s.momentumResidMax = 0.0;
    s.momentumResidRMS = 0.0;
    s.counter          = 0;

    for (int i = 1; i <= s.numCellsXm1; ++i) {
        const int ip = i + 1;
        for (int j = s.jLow[i]+1; j <= s.jHigh[i]-1; ++j) {
            const int jp = j + 1;
            for (int k = 1; k <= KKfim; ++k) {
                const int kp = (k < cfg.numCellsZ) ? k+1 : 1;

                const double gradPx =
                    (s.press(ip,j,k) - s.press(i,j,k) + s.press(ip,jp,k) - s.press(i,jp,k)
                   + s.press(ip,j,kp) - s.press(i,j,kp) + s.press(ip,jp,kp) - s.press(i,jp,kp)) * qInvDx;
                const double gradPy =
                    (s.press(i,jp,k) - s.press(i,j,k) + s.press(ip,jp,k) - s.press(ip,j,k)
                   + s.press(i,jp,kp) - s.press(i,j,kp) + s.press(ip,jp,kp) - s.press(ip,j,kp)) * qInvDy;
                const double gradPz =
                    (-s.press(i,jp,k) - s.press(i,j,k) - s.press(ip,jp,k) - s.press(ip,j,k)
                    + s.press(i,jp,kp) + s.press(i,j,kp) + s.press(ip,jp,kp) + s.press(ip,j,kp)) * qInvDz;

                const double resU = s.accelX(i,j,k) - gradPx;
                const double resV = s.accelY(i,j,k) - gradPy;
                const double resW = s.accelZ(i,j,k) - gradPz;
                const double resSq    = resU*resU + resV*resV + resW*resW;
                const double resNorm  = std::sqrt(resSq);
                s.scratchField(i,j,k) = resNorm;

                if (resNorm > s.momentumResidMax) {
                    s.momentumResidMax = resNorm;
                    s.iResidMax = i; s.jResidMax = j; s.kResidMax = k;
                }
                s.momentumResidRMS += resSq;
                ++s.counter;
            }
        }
    }
    if (s.counter > 0)
        s.momentumResidRMS = std::sqrt(s.momentumResidRMS / s.counter);
}

// ---------------------------------------------------------------------------
//  computeDivergence — mass conservation check  ∇·u ≈ 0?
// ---------------------------------------------------------------------------
void computeDivergence(SimState& s)
{
    const auto& cfg = s.cfg;
    const double qInvDx = 0.25 / cfg.cellSizeX;
    const double qInvDy = 0.25 / cfg.cellSizeY;
    const double qInvDz = 0.25 / cfg.cellSizeZ;

    s.dilatationMax    = 0.0;
    s.intDivergence    = 0.0;
    s.intAbsDivergence = 0.0;

    for (int i = 1; i <= cfg.numCellsX; ++i) {
        const int im = i - 1;
        const int jS = (i != s.degreeIndex2) ? s.jLow[i]+1  : s.jLow[im]+1;
        const int jN = (i != s.degreeIndex2) ? s.jHigh[i]   : s.jHigh[im];

        for (int j = jS; j <= jN; ++j) {
            const int jm = j - 1;
            for (int k = 1; k <= cfg.numCellsZ; ++k) {
                const int km = (cfg.lateralCondition == LateralBC::Periodic && k == 1) ? cfg.numCellsZ : k-1;

                const double div =
                    (s.velX(i,j,k) - s.velX(im,j,k) + s.velX(i,jm,k) - s.velX(im,jm,k)
                   + s.velX(i,j,km) - s.velX(im,j,km) + s.velX(i,jm,km) - s.velX(im,jm,km)) * qInvDx
                  + (s.velY(i,j,k) - s.velY(i,jm,k) + s.velY(im,j,k) - s.velY(im,jm,k)
                   + s.velY(i,j,km) - s.velY(i,jm,km) + s.velY(im,j,km) - s.velY(im,jm,km)) * qInvDy
                  + (s.velZ(i,j,k) + s.velZ(i,jm,k) + s.velZ(im,j,k) + s.velZ(im,jm,k)
                   - s.velZ(i,j,km) - s.velZ(i,jm,km) - s.velZ(im,j,km) - s.velZ(im,jm,km)) * qInvDz;

                s.intDivergence    += div;
                s.intAbsDivergence += std::abs(div);
                if (std::abs(div) > s.dilatationMax) {
                    s.dilatationMax = std::abs(div);
                    s.iDilMax = i; s.jDilMax = j; s.kDilMax = k;
                }
            }
        }
    }
    const double cellVol = cfg.cellSizeX * cfg.cellSizeY * cfg.cellSizeZ;
    s.intDivergence    *= cellVol;
    s.intAbsDivergence *= cellVol;
}

// ---------------------------------------------------------------------------
//  adaptTimeStep — CFL stability criterion
// ---------------------------------------------------------------------------
void adaptTimeStep(SimState& s)
{
    const auto& cfg = s.cfg;
    double uMax = 0.0, vMax = 0.0, wMax = 0.0;

    for (int i = 1; i <= s.numCellsXm1; ++i)
        for (int j = s.jLow[i]+1; j <= s.jHigh[i]-1; ++j)
            for (int k = 1; k <= cfg.numCellsZ; ++k) {
                uMax = std::max(uMax, std::abs(s.velX(i,j,k)));
                vMax = std::max(vMax, std::abs(s.velY(i,j,k)));
                wMax = std::max(wMax, std::abs(s.velZ(i,j,k)));
            }

    const double reForDiff = (cfg.hyperViscousStart == 0) ? cfg.reynoldsNumber : cfg.hyperViscousRe;
    const double dtViscous  = 0.5 * reForDiff
                            / (1.0/s.cellSizeXsq + 1.0/s.cellSizeYsq + 1.0/s.cellSizeZsq);
    const double dtAdv = std::min({cfg.cellSizeX / (uMax + 1e-30),
                                   cfg.cellSizeY / (vMax + 1e-30),
                                   cfg.cellSizeZ / (wMax + 1e-30)});
    s.timeStepSize = 0.35 * std::min(dtViscous, dtAdv);
}