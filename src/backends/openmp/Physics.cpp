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
//  Started as a full copy of src/backends/serial/Physics.cpp (deliberately, not
//  #ifdef-branched into the serial file — see docs/openmp-parallelization.md
//  for that tradeoff). Everything that is not backend-specific now lives in
//  src/solver/ and is included by both.
//
//  Every kernel here is parallel. Five of them used to be verbatim serial
//  copies, which put 37% of per-step time outside the thread team and capped
//  the measurable speedup at ~2.3x however many cores the machine had.
//
//  Geometry and initial conditions are NOT duplicated here: they moved to
//  src/solver/Setup.cpp, shared by every backend. This file holds only
//  per-step kernels — the code actually under measurement.
// =============================================================================

#include "Physics.hpp"
#include "Logger.hpp"
#include "Geometry.hpp"
#include "VelocityBCs.hpp"
#include "ViscosityModel.hpp"

#include <cmath>
#include <algorithm>
#include <stdexcept>

// ─── Internal helpers ────────────────────────────────────────────────────────

namespace {

// ---------------------------------------------------------------------------
//  Red-black solvePressurePoisson helpers.
//
//  The serial solver (src/backends/serial/Physics.cpp) interleaves Neumann
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
/// CURRENT press field state. Periodic-Z needs no mirroring -- updateColor()
/// reads wrapped k indices directly.
///
/// Split into two passes to be race-free, and the split is load-bearing:
///   Pass A -- cross-row writes only, targeting row i-1 or i+1, never row i.
///   Pass B -- same-row writes only, targeting this thread's own row i.
/// Interleaved, a thread on row i+1 can write into row i via its own im
/// target while the thread on row i writes the same address via j+-1. The
/// implicit barrier between the passes (neither uses nowait) is what makes
/// Pass B safe. See docs/openmp-parallelization.md.
///
/// MUST be called from inside an enclosing `#pragma omp parallel` region --
/// both `#pragma omp for` below are orphaned worksharing constructs.
void mirrorGhostCells(SimState& s) {
    const auto& cfg = s.cfg;

    // Pass A: cross-row writes only (into row i-1 or i+1, never row i).
    #pragma omp for schedule(static)
    for (int i = 1; i <= cfg.numCellsX; ++i) {
        const int im = i - 1, ip = i + 1;
        int jLoopS, jLoopN; mirrorJRange(s, i, jLoopS, jLoopN);

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
        int jLoopS, jLoopN; mirrorJRange(s, i, jLoopS, jLoopN);

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
/// parallel: a cell of `color` only reads neighbors of the OTHER color, all
/// fixed for this call's duration.
///
/// MUST be called from inside an enclosing `#pragma omp parallel` region --
/// the `#pragma omp for` below is an orphaned worksharing construct. Walks
/// the row list with a strided inner k-loop rather than a per-cell index
/// list: unit-stride-2 instead of a gather, which is what the roofline's
/// "5% of bandwidth ceiling despite low AI" finding pointed at.
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

    // Reset the accumulators only where a sweep writes; see the serial
    // backend for why everything outside that region is already zero. The
    // span table is built here, outside the parallel region, because the
    // first call would otherwise have every thread building it at once.
    auto& plan = s.ext.accel;
    if (plan.zeroJLo.empty()) buildAccelResetPlan(s, plan);

    // One run per column, not one per row: j is the middle index, so a column's
    // whole j-span is contiguous in memory. See the serial backend.
    const int sK = s.accelX.gridSize().sK;
    auto zeroRun = [&](int i, int jFrom, int jTo) {
        const std::size_t n = static_cast<std::size_t>(jTo - jFrom + 1) * sK;
        std::fill_n(&s.accelX(i, jFrom, 0), n, 0.0);
        std::fill_n(&s.accelY(i, jFrom, 0), n, 0.0);
        std::fill_n(&s.accelZ(i, jFrom, 0), n, 0.0);
    };
    #pragma omp parallel for schedule(static)
    for (int i = 1; i <= s.numCellsXm1; ++i) {
        const int lo = plan.zeroJLo[i], hi = plan.zeroJHi[i];
        if (hi < lo) continue;
        if (plan.xJHi[i] < plan.xJLo[i]) { zeroRun(i, lo, hi); continue; }
        if (plan.xJLo[i] > lo) zeroRun(i, lo, plan.xJLo[i] - 1);
        if (plan.xJHi[i] < hi) zeroRun(i, plan.xJHi[i] + 1, hi);
    }

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
        double* ppiu = s.ext.accel.ppiu.data() + tid * s.ext.accel.scratchLenPerThread;
        double* ppid = s.ext.accel.ppid.data() + tid * s.ext.accel.scratchLenPerThread;
        double* qsiu = s.ext.accel.qsiu.data() + tid * s.ext.accel.scratchLenPerThread;
        double* Ku   = s.ext.accel.Ku.data()   + tid * s.ext.accel.scratchLenPerThread;
        double* Kv   = s.ext.accel.Kv.data()   + tid * s.ext.accel.scratchLenPerThread;
        double* Kw   = s.ext.accel.Kw.data()   + tid * s.ext.accel.scratchLenPerThread;

        // 2-D (i,k) scratch, used by the X sweep and then re-used by the Y
        // sweep as (j,k) once X is finished — see the alias block there. The
        // 1-D ppin/ppis/qsin slices these replaced are gone; only the Z sweep
        // still needs 1-D buffers.
        double* qsieXK = s.ext.accel.qsieXK.data() + tid * s.ext.accel.xk2DLenPerThread;
        double* ppiwRow = s.ext.accel.ppiwRow.data() + tid * s.ext.accel.scratchKLen;
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
    // tests/backend/GoldenFieldTests.cpp (captured from the pre-restructuring
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

        // Passes 1-3, fused into one traversal.
        //
        // All three read velX/velY/velZ at i-1, i and i+1. Run separately they
        // streamed those three fields three times over, and at the production
        // grid one field is 21.9 MB against a 12 MB L3 -- nothing survived from
        // one pass to the next, so each re-read went to DRAM.
        //
        // Fusing is legal because the only cross-pass dependency resolves
        // inside a single (i, k) iteration: pass 2 needs ppie[i+1] (pass 1
        // writes it at this same i and k, just above) and ppiw[i+1] (pass 1
        // wrote it at i-1, the previous iteration). Pass 3 reads no pass
        // output at all. Every arithmetic expression below is kept in its
        // original association -- floating-point addition is not associative,
        // and the golden-field test pins this kernel to 1e-12.
        //
        // i = iStart is peeled: passes 2 and 3 start at iStart+1, and peeling
        // keeps a branch out of the fused loop.
        {
            const int i = iStart;
            const double localRe = 1.0 / effectiveInvRe(s, i);
            for (int k = 1; k <= KKfim; ++k) {
                const double uFace = 0.5 * (s.velX(i+1, j, k) + s.velX(i, j, k));
                const double DPe = localRe * uFace * cfg.cellSizeX;
                double pip, cip, cim;
                computeExponentialWeights(localRe, DPe, pip, cip, cim);
                ppiwRow[k-1] = cim;             // west coefficient, consumed at i+1
                VM2(qsieXK, i+1, k) = computeQsi(DPe, pip, 0.5);
            }
        }
        for (int i = iStart+1; i <= iEnd-1; ++i) {
            const double localRe = 1.0 / effectiveInvRe(s, i);
            for (int k = 1; k <= KKfim; ++k) {
                // One load of each field per (i,k), shared by all three passes.
                const double vxm = s.velX(i-1,j,k), vx0 = s.velX(i,j,k), vxp = s.velX(i+1,j,k);
                const double vym = s.velY(i-1,j,k), vy0 = s.velY(i,j,k), vyp = s.velY(i+1,j,k);
                const double vzm = s.velZ(i-1,j,k), vz0 = s.velZ(i,j,k), vzp = s.velZ(i+1,j,k);

                // Pass 1: face coefficients (between i and i+1)
                const double uFace = 0.5 * (vxp + vx0);
                const double DPeFace = localRe * uFace * cfg.cellSizeX;
                double pipF, cipF, cimF;
                computeExponentialWeights(localRe, DPeFace, pipF, cipF, cimF);
                // Read the carried value BEFORE overwriting the slot: this row
                // holds i-1's west coefficient, and pass 1 is about to replace
                // it with i+1's.
                const double ppiw = ppiwRow[k-1];
                ppiwRow[k-1] = cimF;
                const double ppie = cipF;                // consumed this iteration
                VM2(qsieXK, i+1, k) = computeQsi(DPeFace, pipF, 0.5);

                // Pass 2: diffusive part of Au, Av, Aw (first part)
                // Assign, not accumulate: this is the first write to every cell
                // in the X sweep's range, so the reset above skips them
                // entirely -- three fields' worth of stores, and the read half
                // of the read-modify-write, both gone.
                const double coeff = invDx2;
                s.accelX(i, j, k) = (ppie*(vxp-vx0) + ppiw*(vxm-vx0)) * coeff;
                s.accelY(i, j, k) = (ppie*(vyp-vy0) + ppiw*(vym-vy0)) * coeff;
                s.accelZ(i, j, k) = (ppie*(vzp-vz0) + ppiw*(vzm-vz0)) * coeff;

                // Pass 3: cross-term correction (K * qsi)
                const double uCell = vx0;
                const double DPeCell = localRe * uCell * cfg.cellSizeX;
                double pipC, cipC, cimC;
                computeExponentialWeights(localRe, DPeCell, pipC, cipC, cimC);
                cipC *= invDx2;
                cimC *= invDx2;
                VM2(KuXK, i+1, k) = cipC*(vx0-vxp) + cimC*(vx0-vxm);
                VM2(KvXK, i+1, k) = cipC*(vy0-vyp) + cimC*(vy0-vym);
                VM2(KwXK, i+1, k) = cipC*(vz0-vzp) + cimC*(vz0-vzm);
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
    // Same restructuring the X-sweep already had, for the same reason: j
    // carries the pass-to-pass recurrence and cannot move, but nothing here
    // references k+-1, so k becomes the innermost unit-stride loop and the
    // scratch widens to (j, k). The 2-D buffers are the X-sweep's, reused --
    // that sweep is complete and its scratch dead by this point, and both
    // dimensions are sized off maxDim, so no extra allocation is needed.
    double* qsinJK = qsieXK;
    double* KuJK = KuXK; double* KvJK = KvXK; double* KwJK = KwXK;

    const double invDy2 = 1.0 / (cfg.cellSizeY * cfg.cellSizeY);
    // Parallel over i -- same reasoning as the X-sweep's #pragma omp for
    // above, just over the Y-sweep's outer dimension instead.
    #pragma omp for schedule(static)
    for (int i = 1; i <= s.numCellsXm1; ++i) {
        const int jStart = s.jLow[i];
        const int jEnd   = s.jHigh[i];
        const double invRe = effectiveInvRe(s, i);
        const double localRe = 1.0 / invRe;

        // Passes 1-3 fused, exactly as the X sweep above and for the same
        // reason: all three read velX/velY/velZ at j-1, j and j+1, and the only
        // cross-pass dependency (pass 2 needs ppin[j+1] from pass 1 at this j,
        // and ppis[j+1] from pass 1 at j-1) resolves inside one iteration.
        {
            const int j = jStart;
            for (int k = 1; k <= KKfim; ++k) {
                const double vFace = 0.5 * (s.velY(i, j+1, k) + s.velY(i, j, k));
                const double DPe = localRe * vFace * cfg.cellSizeY;
                double pip, cin, cis;
                computeExponentialWeights(localRe, DPe, pip, cin, cis);
                ppiwRow[k-1] = cis;             // south coefficient, consumed at j+1
                VM2(qsinJK, j+1, k) = computeQsi(DPe, pip, 0.5);
            }
        }
        for (int j = jStart+1; j <= jEnd-1; ++j) {
            for (int k = 1; k <= KKfim; ++k) {
                // One load of each field per (j,k), shared by all three passes.
                const double vxm = s.velX(i,j-1,k), vx0 = s.velX(i,j,k), vxp = s.velX(i,j+1,k);
                const double vym = s.velY(i,j-1,k), vy0 = s.velY(i,j,k), vyp = s.velY(i,j+1,k);
                const double vzm = s.velZ(i,j-1,k), vz0 = s.velZ(i,j,k), vzp = s.velZ(i,j+1,k);

                // Pass 1: face coefficients (between j and j+1)
                const double vFace = 0.5 * (vyp + vy0);
                const double DPeFace = localRe * vFace * cfg.cellSizeY;
                double pipF, cinF, cisF;
                computeExponentialWeights(localRe, DPeFace, pipF, cinF, cisF);
                const double ppis = ppiwRow[k-1];        // carried from j-1
                ppiwRow[k-1] = cisF;                     // for j+1
                const double ppin = cinF;                // consumed this iteration
                VM2(qsinJK, j+1, k) = computeQsi(DPeFace, pipF, 0.5);

                // Pass 2: diffusive part of Au, Av, Aw
                const double coeff = invDy2;
                s.accelX(i, j, k) += (ppin*(vxp-vx0) + ppis*(vxm-vx0)) * coeff;
                s.accelY(i, j, k) += (ppin*(vyp-vy0) + ppis*(vym-vy0)) * coeff;
                s.accelZ(i, j, k) += (ppin*(vzp-vz0) + ppis*(vzm-vz0)) * coeff;

                // Pass 3: cross-term correction (K * qsi)
                const double vCell = vy0;
                const double DPeCell = localRe * vCell * cfg.cellSizeY;
                double pipC, cinC, cisC;
                computeExponentialWeights(localRe, DPeCell, pipC, cinC, cisC);
                cinC *= invDy2;  cisC *= invDy2;
                VM2(KuJK, j+1, k) = cinC*(vx0-vxp) + cisC*(vx0-vxm);
                VM2(KvJK, j+1, k) = cinC*(vy0-vyp) + cisC*(vy0-vym);
                VM2(KwJK, j+1, k) = cinC*(vz0-vzp) + cisC*(vz0-vzm);
            }
        }
        for (int k = 1; k <= KKfim; ++k) {
            VM2(KuJK, jStart+1, k) = 2.0*VM2(KuJK, jStart+2, k) - VM2(KuJK, jStart+3, k);
            VM2(KvJK, jStart+1, k) = 2.0*VM2(KvJK, jStart+2, k) - VM2(KvJK, jStart+3, k);
            VM2(KwJK, jStart+1, k) = 2.0*VM2(KwJK, jStart+2, k) - VM2(KwJK, jStart+3, k);
            VM2(KuJK, jEnd+1, k)   = 2.0*VM2(KuJK, jEnd, k)   - VM2(KuJK, jEnd-1, k);
            VM2(KvJK, jEnd+1, k)   = 2.0*VM2(KvJK, jEnd, k)   - VM2(KvJK, jEnd-1, k);
            VM2(KwJK, jEnd+1, k)   = 2.0*VM2(KwJK, jEnd, k)   - VM2(KwJK, jEnd-1, k);
        }
        for (int j = jStart; j <= jEnd-1; ++j) {
            for (int k = 1; k <= KKfim; ++k) {
                VM2(KuJK, j+1, k) = 0.5*(VM2(KuJK, j+1, k) + VM2(KuJK, j+2, k));
                VM2(KvJK, j+1, k) = 0.5*(VM2(KvJK, j+1, k) + VM2(KvJK, j+2, k));
                VM2(KwJK, j+1, k) = 0.5*(VM2(KwJK, j+1, k) + VM2(KwJK, j+2, k));
            }
        }
        for (int j = jStart+1; j <= jEnd-1; ++j) {
            for (int k = 1; k <= KKfim; ++k) {
                s.accelX(i, j, k) -= (VM2(KuJK,j+1,k)*VM2(qsinJK,j+1,k) - VM2(KuJK,j,k)*VM2(qsinJK,j,k));
                s.accelY(i, j, k) -= (VM2(KvJK,j+1,k)*VM2(qsinJK,j+1,k) - VM2(KvJK,j,k)*VM2(qsinJK,j,k));
                s.accelZ(i, j, k) -= (VM2(KwJK,j+1,k)*VM2(qsinJK,j+1,k) - VM2(KwJK,j,k)*VM2(qsinJK,j,k));
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
            // Passes 2 and 3 fused: both read velX/velY/velZ at km, k and kp,
            // the same nine loads twice over. Pass 3 consumes no pass output,
            // and pass 2 only needs pass 1's completed arrays.
            //
            // Pass 1 canNOT join them, unlike the X and Y sweeps. Its k loop
            // wraps -- at k == numCellsZ the periodic kp folds back to 1 and
            // writes ppid(2), which pass 2 reads at k == 1 -- so pass 1 has a
            // backward dependency and must finish first.
            for (int k = 1; k <= KKfim; ++k) {
                const int kp = (periodic && k == cfg.numCellsZ) ? 1 : k+1;
                const int km = (periodic && k == 1) ? cfg.numCellsZ : k-1;

                const double vxm = s.velX(i,j,km), vx0 = s.velX(i,j,k), vxp = s.velX(i,j,kp);
                const double vym = s.velY(i,j,km), vy0 = s.velY(i,j,k), vyp = s.velY(i,j,kp);
                const double vzm = s.velZ(i,j,km), vz0 = s.velZ(i,j,k), vzp = s.velZ(i,j,kp);

                // Pass 2: diffusive part of Au, Av, Aw
                const double coeff = invDz2;
                const double ppiuK = VM(ppiu,k+1), ppidK = VM(ppid,k+1);
                s.accelX(i, j, k) += (ppiuK*(vxp-vx0) + ppidK*(vxm-vx0)) * coeff;
                s.accelY(i, j, k) += (ppiuK*(vyp-vy0) + ppidK*(vym-vy0)) * coeff;
                s.accelZ(i, j, k) += (ppiuK*(vzp-vz0) + ppidK*(vzm-vz0)) * coeff;

                // Pass 3: cross-term correction (K * qsi)
                const double wCell = vz0;
                const double DPe = localRe * wCell * cfg.cellSizeZ;
                double pip, ciu, cid;
                computeExponentialWeights(localRe, DPe, pip, ciu, cid);
                ciu *= invDz2;  cid *= invDz2;
                VM(Ku, k+1) = ciu*(vx0-vxp) + cid*(vx0-vxm);
                VM(Kv, k+1) = ciu*(vy0-vyp) + cid*(vy0-vym);
                VM(Kw, k+1) = ciu*(vz0-vzp) + cid*(vz0-vzm);
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

    // Zero acceleration on all boundary (wall) nodes. Two separate loops with
    // the implicit barrier between them: iteration i writes only column i and
    // iteration j only row j, so neither loop races with itself, and the
    // corners both of them touch are written 0.0 either way.
    #pragma omp parallel for schedule(static)
    for (int i = 0; i <= cfg.numCellsX; ++i)
        for (int k = 0; k <= cfg.numCellsZ; ++k) {
            const int jB = s.jLow[i], jT = s.jHigh[i];
            s.accelX(i,jB,k) = s.accelX(i,jT,k) = 0.0;
            s.accelY(i,jB,k) = s.accelY(i,jT,k) = 0.0;
            s.accelZ(i,jB,k) = s.accelZ(i,jT,k) = 0.0;
        }
    #pragma omp parallel for schedule(static)
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

    // Parallel over i: iteration i writes only pressureSource(i,·,·) and reads
    // vel/accel at i and i-1, which no iteration modifies.
    #pragma omp parallel for schedule(static)
    for (int i = 1; i <= cfg.numCellsX; ++i) {
        const int im = i - 1;
        int jS, jN; activeJRange(s, i, jS, jN);

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
//  ONE parallel region wraps all numPressureIter sweeps, not one per
//  updateColor() call: that would spawn and join a team 2*numPressureIter
//  times per solve, per timestep, at a fixed cost that does not shrink with
//  grid size. mirrorGhostCells() is called by every thread here, not under
//  `omp single` -- its own orphaned `omp for` passes synchronise it.
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

    // Built once — geometry is fixed after initSimulation(). Lives in this
    // backend's Extras; emptiness replaces the old redBlackBuilt flag.
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

    // Parallel over i: iteration i writes only vel*(i,·,·); press and accel are
    // read-only here, so nothing another iteration writes is read.
    #pragma omp parallel for schedule(static)
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

    // residMax is a max reduction, so it is exact regardless of how the team
    // is scheduled -- which matters because the driver's convergence test
    // reads it, and a thread-count-dependent stopping step would make the
    // backends incomparable. residRMS is a sum and so is order-dependent at
    // the last couple of ULPs; it is reported, never used for control flow.
    double residMax = 0.0, residSumSq = 0.0;
    long long count = 0;

    #pragma omp parallel for schedule(static) \
            reduction(max:residMax) reduction(+:residSumSq,count)
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

                residMax = std::max(residMax, resNorm);
                residSumSq += resSq;
                ++count;
            }
        }
    }
    s.momentumResidMax = residMax;
    if (count > 0)
        s.momentumResidRMS = std::sqrt(residSumSq / static_cast<double>(count));
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

    double dilMax = 0.0, intDiv = 0.0, intAbsDiv = 0.0;

    #pragma omp parallel for schedule(static) \
            reduction(max:dilMax) reduction(+:intDiv,intAbsDiv)
    for (int i = 1; i <= cfg.numCellsX; ++i) {
        const int im = i - 1;
        int jS, jN; activeJRange(s, i, jS, jN);

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

                intDiv    += div;
                intAbsDiv += std::abs(div);
                dilMax     = std::max(dilMax, std::abs(div));
            }
        }
    }
    const double cellVol = cfg.cellSizeX * cfg.cellSizeY * cfg.cellSizeZ;
    s.dilatationMax    = dilMax;
    s.intDivergence    = intDiv * cellVol;
    s.intAbsDivergence = intAbsDiv * cellVol;
}

// ---------------------------------------------------------------------------
//  adaptTimeStep — CFL stability criterion
// ---------------------------------------------------------------------------
void adaptTimeStep(SimState& s)
{
    const auto& cfg = s.cfg;
    double uMax = 0.0, vMax = 0.0, wMax = 0.0;

    // Max reductions only, so dt is bit-identical at any thread count -- and
    // at any value of the if() below. It feeds every subsequent step, so an
    // order-dependent dt would make two runs of the same config diverge.
    //
    // The if(): this is by far the cheapest kernel (three reads per cell, no
    // stores), so on a small grid a 12-thread fork/join costs more than the
    // whole loop -- measured 0.140ms serial, 0.070ms on 2 threads, 0.716ms on
    // 12 at 96x48x24. Demanding ~10k iterations per thread before going
    // parallel keeps the win at production sizes without that cliff.
    const long long cells = static_cast<long long>(cfg.numCellsX) *
                            cfg.numCellsY * cfg.numCellsZ;
    const bool worthThreading = cells >= 10000LL * omp_get_max_threads();

    #pragma omp parallel for schedule(static) reduction(max:uMax,vMax,wMax) \
            if(worthThreading)
    for (int i = 1; i <= s.numCellsXm1; ++i)
        for (int j = s.jLow[i]+1; j <= s.jHigh[i]-1; ++j)
            for (int k = 1; k <= cfg.numCellsZ; ++k) {
                uMax = std::max(uMax, static_cast<double>(std::abs(s.velX(i,j,k))));
                vMax = std::max(vMax, static_cast<double>(std::abs(s.velY(i,j,k))));
                wMax = std::max(wMax, static_cast<double>(std::abs(s.velZ(i,j,k))));
            }

    const double reForDiff = (cfg.hyperViscousStart == 0) ? cfg.reynoldsNumber : cfg.hyperViscousRe;
    const double dtViscous  = 0.5 * reForDiff
                            / (1.0/s.cellSizeXsq + 1.0/s.cellSizeYsq + 1.0/s.cellSizeZsq);
    const double dtAdv = std::min({cfg.cellSizeX / (uMax + 1e-30),
                                   cfg.cellSizeY / (vMax + 1e-30),
                                   cfg.cellSizeZ / (wMax + 1e-30)});
    s.timeStepSize = 0.35 * std::min(dtViscous, dtAdv);
}