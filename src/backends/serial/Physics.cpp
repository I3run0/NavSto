// =============================================================================
//  Physics.cpp  —  Navier-Stokes solver kernels.
//
//  Solves the 3-D incompressible Navier-Stokes equations:
//
//    ∂u/∂t + (u·∇)u = -∇p + (1/Re)∇²u      (momentum)
//             ∇·u   = 0                        (incompressibility)
//
//  Method: explicit fractional-step (projection) with UNIFAES exponential
//          scheme on a staggered Cartesian grid.
//  Pressure: Gauss-Seidel solution of ∇²p = S.
// =============================================================================

#include "Physics.hpp"
#include "Geometry.hpp"
#include "PressureMultigrid.hpp"
#include "KernelRows.hpp"
#include "Logger.hpp"
#include "VelocityBCs.hpp"
#include "ViscosityModel.hpp"

#include <cmath>
#include <algorithm>
#include <stdexcept>

// ---------------------------------------------------------------------------
//  computeAccelerations — UNIFAES advection + viscous terms
// ---------------------------------------------------------------------------
void computeAccelerations(SimState& s)
{
    const auto& cfg = s.cfg;
    const bool periodic = (cfg.lateralCondition == LateralBC::Periodic);
    const int KKfim = periodic ? cfg.numCellsZ : s.numCellsZm1;

    // Reset only what the X sweep below does not assign outright. Cells no
    // sweep writes are never anything but zero -- GridField's constructor put
    // it there and only explicit `= 0.0` assignments touch them since -- so
    // zeroing the whole allocation every call was pure store traffic.
    auto& plan = s.ext.accel;
    if (plan.zeroJLo.empty()) buildAccelResetPlan(s, plan);

    // One run per column, not one per row: j is the middle index, so a column's
    // whole j-span is contiguous in memory. Row-at-a-time fills of numCellsZ+1
    // doubles measured slower than the full fill they replaced.
    const int sK = s.accelX.gridSize().sK;
    auto zeroRun = [&](int i, int jFrom, int jTo) {
        const std::size_t n = static_cast<std::size_t>(jTo - jFrom + 1) * sK;
        std::fill_n(&s.accelX(i, jFrom, 0), n, 0.0);
        std::fill_n(&s.accelY(i, jFrom, 0), n, 0.0);
        std::fill_n(&s.accelZ(i, jFrom, 0), n, 0.0);
    };
    for (int i = 1; i <= s.numCellsXm1; ++i) {
        const int lo = plan.zeroJLo[i], hi = plan.zeroJHi[i];
        if (hi < lo) continue;
        if (plan.xJHi[i] < plan.xJLo[i]) { zeroRun(i, lo, hi); continue; }
        if (plan.xJLo[i] > lo) zeroRun(i, lo, plan.xJLo[i] - 1);
        if (plan.xJHi[i] < hi) zeroRun(i, plan.xJHi[i] + 1, hi);
    }

    // Temporary 1‑D arrays (logical index -1 .. maxDim+1) offset by +1.
    // Sized maxDim+3 (not maxDim+2) in AccelScratch::allocate(): several
    // writes below reach logical index maxDim+1 (e.g. VM(ppiw, i+2) at
    // i==iEnd-1==maxDim-1, and the VM(Ku, iEnd+1)/VM(Ku, jEnd+1)/
    // VM(Ku, numCellsZ+1) boundary extrapolations), which needs physical
    // slot maxDim+2. Owned by SimState's backend Extras and reused across calls
    // instead of being allocated fresh every call — see AccelScratch.hpp for why
    // that's safe without re-zeroing.
    auto& ppiu = s.ext.accel.ppiu; auto& ppid = s.ext.accel.ppid;
    double* __restrict wNum = s.ext.accel.wNum.data();
    double* __restrict wDen = s.ext.accel.wDen.data();
    double* __restrict wDPe = s.ext.accel.wDPe.data();
    double* __restrict wQMask = s.ext.accel.wQMask.data();
    double* __restrict wQDen = s.ext.accel.wQDen.data();
    double* __restrict wQAdd = s.ext.accel.wQAdd.data();
    auto& qsiu = s.ext.accel.qsiu;
    auto& Ku = s.ext.accel.Ku; auto& Kv = s.ext.accel.Kv; auto& Kw = s.ext.accel.Kw;

    // X-sweep-only 2-D (i,k) scratch — see AccelScratch.hpp field comments
    // and docs/serial-optimization-loop-order.md. Replaces the 1-D
    // ppie/ppiw/qsie buffers (and this sweep's private use of Ku/Kv/Kw,
    // which the Y/Z sweeps below still use in their original 1-D form).
    auto& qsieXK = s.ext.accel.qsieXK;
    auto& ppiwRow = s.ext.accel.ppiwRow;   // west coefficient carried from i-1 / j-1
    auto& KuXK = s.ext.accel.KuXK; auto& KvXK = s.ext.accel.KvXK; auto& KwXK = s.ext.accel.KwXK;

    // Helper to access offset arrays (logical idx -> physical idx+1)
    auto VM = [](std::vector<double>& v, int idx) -> double& { return v[idx+1]; };

    // Same offset convention as VM, but for the X-sweep's 2-D (i,k) scratch:
    // logical i -> physical i+1 (as VM), k (1..KKfim) -> physical k-1.
    const int kStride = s.ext.accel.scratchKLen;
    auto VM2 = [kStride](std::vector<double>& v, int iLogical, int k) -> double& {
        return v[static_cast<std::size_t>(iLogical + 1) * kStride + (k - 1)];
    };

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
        for (int i = iStart; i <= iEnd-1; ++i)
            faceAverageRow(&VM2(KuXK,i+1,1), &VM2(KvXK,i+1,1), &VM2(KwXK,i+1,1),
                           &VM2(KuXK,i+2,1), &VM2(KvXK,i+2,1), &VM2(KwXK,i+2,1), KKfim);

        // Subtract cross‑term divergence
        for (int i = iStart+1; i <= iEnd-1; ++i)
            crossTermRow(&s.accelX(i,j,1), &s.accelY(i,j,1), &s.accelZ(i,j,1),
                         &VM2(KuXK,i+1,1), &VM2(KvXK,i+1,1), &VM2(KwXK,i+1,1), &VM2(qsieXK,i+1,1),
                         &VM2(KuXK,i,1),   &VM2(KvXK,i,1),   &VM2(KwXK,i,1),   &VM2(qsieXK,i,1),
                         KKfim);
    }

    // ---------- Direction Y ----------
    // Same restructuring the X-sweep already had, for the same reason: j
    // carries the pass-to-pass recurrence and cannot move, but nothing here
    // references k+-1, so k becomes the innermost unit-stride loop and the
    // scratch widens to (j, k). The 2-D buffers are the X-sweep's, reused --
    // that sweep is complete and its scratch dead by this point, and both
    // dimensions are sized off maxDim, so no extra allocation is needed.
    auto& qsinJK = s.ext.accel.qsieXK;
    auto& KuJK = s.ext.accel.KuXK; auto& KvJK = s.ext.accel.KvXK; auto& KwJK = s.ext.accel.KwXK;

    const double invDy2 = 1.0 / (cfg.cellSizeY * cfg.cellSizeY);
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
        for (int j = jStart; j <= jEnd-1; ++j)
            faceAverageRow(&VM2(KuJK,j+1,1), &VM2(KvJK,j+1,1), &VM2(KwJK,j+1,1),
                           &VM2(KuJK,j+2,1), &VM2(KvJK,j+2,1), &VM2(KwJK,j+2,1), KKfim);

        for (int j = jStart+1; j <= jEnd-1; ++j)
            crossTermRow(&s.accelX(i,j,1), &s.accelY(i,j,1), &s.accelZ(i,j,1),
                         &VM2(KuJK,j+1,1), &VM2(KvJK,j+1,1), &VM2(KwJK,j+1,1), &VM2(qsinJK,j+1,1),
                         &VM2(KuJK,j,1),   &VM2(KvJK,j,1),   &VM2(KwJK,j,1),   &VM2(qsinJK,j,1),
                         KKfim);
    }

    // ---------- Direction Z ----------
    const double invDz2 = 1.0 / (cfg.cellSizeZ * cfg.cellSizeZ);

    // Peel bounds for this sweep's periodic wrap, hoisted out of the i/j
    // loops. KKfim is numCellsZ when periodic and numCellsZm1 otherwise, so
    // the non-periodic case has no wrapping iteration to peel at all and
    // every loop below runs its full original range.
    const int zP1Hi  = periodic ? KKfim - 1 : KKfim;   // pass 1:      kp = k+1
    const int zP23Lo = periodic ? 2 : 1;               // passes 2-3:  km = k-1
    const int zP23Hi = periodic ? KKfim - 1 : KKfim;   //              kp = k+1
    const int zXtLo  = periodic ? 2 : 1;               // cross term:  km = k-1
    for (int i = 1; i <= s.numCellsXm1; ++i) {
        const double invRe = effectiveInvRe(s, i);
        const double localRe = 1.0 / invRe;
        for (int j = s.jLow[i]+1; j <= s.jHigh[i]-1; ++j) {
            // Pass 1. The periodic wrap -- kp folds back to 1 at k ==
            // numCellsZ -- is this loop's only non-affine index, and it cost
            // the whole loop: `periodic` is a runtime bool, so the select
            // survived into every iteration and the vectoriser reported
            // "control flow in loop". Peeled, the range below is plain k+1.
            auto zPassOne = [&](int k, int kp) __attribute__((always_inline)) {
                const double wFace = 0.5 * (s.velZ(i, j, kp) + s.velZ(i, j, k));
                const double DPe = localRe * wFace * cfg.cellSizeZ;
                double pip, ciu, cid;
                computeExponentialWeights(localRe, DPe, pip, ciu, cid);
                VM(ppiu, k+1) = ciu;
                VM(ppid, kp+1) = cid;
                VM(qsiu, k+1) = computeQsi(DPe, pip, 0.5);
            };
            // Split in two: a scalar pass that only selects operands -- the
            // branch chain and the exp live there -- and a division pass with
            // no control flow, which vectorises. The four divisions are ~29%
            // of this kernel and were scalar only because the branches around
            // them blocked the loop; see docs/roofline.md.
            for (int k = 0; k <= zP1Hi; ++k) {
                const double wFace = 0.5 * (s.velZ(i, j, k+1) + s.velZ(i, j, k));
                const double DPe = localRe * wFace * cfg.cellSizeZ;
                wDPe[k] = DPe;
                weightOperands(DPe, 0.5, wNum[k], wDen[k], wQMask[k], wQDen[k], wQAdd[k]);
            }
            weightDivideRow(wNum, wDen, wDPe, wQMask, wQDen, wQAdd,
                            &VM(ppiu,1), &VM(ppid,2), &VM(qsiu,1), zP1Hi + 1, localRe);
            if (periodic) zPassOne(cfg.numCellsZ, 1);

            // Passes 2 and 3 fused: both read velX/velY/velZ at km, k and kp,
            // the same nine loads twice over. Pass 3 consumes no pass output,
            // and pass 2 only needs pass 1's completed arrays.
            //
            // Pass 1 canNOT join them, unlike the X and Y sweeps. Its k loop
            // wraps -- at k == numCellsZ the periodic kp folds back to 1 and
            // writes ppid(2), which pass 2 reads at k == 1 -- so pass 1 has a
            // backward dependency and must finish first.
            //
            // Both wrapping ends are peeled here, for the same reason as pass
            // 1 and at both ends because this body carries two folded indices
            // (km at k == 1, kp at k == numCellsZ). No iteration depends on
            // another -- every write is to its own k -- so the peels may run
            // either side of the interior without changing a single result.
            auto zPassTwoThree = [&](int k, int kp, int km) __attribute__((always_inline)) {
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
            };
            if (periodic) zPassTwoThree(1, cfg.numCellsZ == 1 ? 1 : 2, cfg.numCellsZ);
            // Same split for pass 3, which needs no qsi and scales both
            // coefficients by 1/dz^2 exactly where the fused body did.
            for (int k = zP23Lo; k <= zP23Hi; ++k) {
                const double DPe = localRe * s.velZ(i,j,k) * cfg.cellSizeZ;
                wDPe[k] = DPe;
                expWeightOperands(DPe, wNum[k], wDen[k]);
            }
            if (zP23Hi >= zP23Lo)
                zDiffusionCrossRow(&s.accelX(i,j,zP23Lo), &s.accelY(i,j,zP23Lo), &s.accelZ(i,j,zP23Lo),
                                   &s.velX(i,j,zP23Lo), &s.velY(i,j,zP23Lo), &s.velZ(i,j,zP23Lo),
                                   &VM(ppiu,zP23Lo+1), &VM(ppid,zP23Lo+1),
                                   wNum + zP23Lo, wDen + zP23Lo, wDPe + zP23Lo,
                                   &VM(Ku,zP23Lo+1), &VM(Kv,zP23Lo+1), &VM(Kw,zP23Lo+1),
                                   zP23Hi - zP23Lo + 1, invDz2, localRe);
            if (periodic && cfg.numCellsZ > 1)
                zPassTwoThree(cfg.numCellsZ, 1, cfg.numCellsZ-1);

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
            // Cross-term divergence, the X and Y sweeps' row kernel at last.
            // Only k == 1 folds km back to numCellsZ, so peeling that one
            // iteration leaves a row whose Hi and Lo operands are the same
            // scratch offset by one. They overlap, but both are read-only
            // here -- crossTermRow writes only the three accel rows, which
            // are a different allocation -- so the restrict qualifiers hold.
            if (periodic) {
                s.accelX(i, j, 1) -= (VM(Ku,2)*VM(qsiu,2) - VM(Ku,cfg.numCellsZ+1)*VM(qsiu,cfg.numCellsZ+1));
                s.accelY(i, j, 1) -= (VM(Kv,2)*VM(qsiu,2) - VM(Kv,cfg.numCellsZ+1)*VM(qsiu,cfg.numCellsZ+1));
                s.accelZ(i, j, 1) -= (VM(Kw,2)*VM(qsiu,2) - VM(Kw,cfg.numCellsZ+1)*VM(qsiu,cfg.numCellsZ+1));
            }
            if (KKfim >= zXtLo)
                crossTermRow(&s.accelX(i,j,zXtLo), &s.accelY(i,j,zXtLo), &s.accelZ(i,j,zXtLo),
                             &VM(Ku,zXtLo+1), &VM(Kv,zXtLo+1), &VM(Kw,zXtLo+1), &VM(qsiu,zXtLo+1),
                             &VM(Ku,zXtLo),   &VM(Kv,zXtLo),   &VM(Kw,zXtLo),   &VM(qsiu,zXtLo),
                             KKfim - zXtLo + 1);
        }
    }

    // Periodic copy in z
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

    // Raw pointers and hoisted strides, because the accessor form does not
    // vectorise: the compiler cannot prove the store misses the GridFields'
    // own sJ/sK members, so the loads' base stops being loop-invariant and it
    // gives up on a k-loop whose 48 loads are all unit-stride. Same
    // expressions in the same association -- only the addressing changes.
    const auto gs = s.velX.gridSize();
    const std::size_t sJ = static_cast<std::size_t>(gs.sJ), sK = static_cast<std::size_t>(gs.sK);
    const double* __restrict vx = s.velX.data().data();
    const double* __restrict vy = s.velY.data().data();
    const double* __restrict vz = s.velZ.data().data();
    const double* __restrict ax = s.accelX.data().data();
    const double* __restrict ay = s.accelY.data().data();
    const double* __restrict az = s.accelZ.data().data();
    double* __restrict psrc = s.pressureSource.data().data();

    // k = 1 is peeled: with periodic z its km wraps to numCellsZ, and a km that
    // is not affine in k stops the loop vectorising on its own.
    const bool periodic = (cfg.lateralCondition == LateralBC::Periodic);

    for (int i = 1; i <= cfg.numCellsX; ++i) {
        const int im = i - 1;
        int jS, jN; activeJRange(s, i, jS, jN);

        for (int j = jS; j <= jN; ++j) {
            const int jm = j - 1;
            // Row pointers for the four (i,j) corners: each is a contiguous
            // run in k, so the loop below indexes them with a plain int and
            // the accesses stay affine. Going through size_t index arithmetic
            // instead left the vectoriser reporting "no vectype for stmt".
            const std::size_t a = (static_cast<std::size_t>(i)  * sJ + j ) * sK;
            const std::size_t b = (static_cast<std::size_t>(im) * sJ + j ) * sK;
            const std::size_t c = (static_cast<std::size_t>(i)  * sJ + jm) * sK;
            const std::size_t d = (static_cast<std::size_t>(im) * sJ + jm) * sK;

            for (int pass = 0; pass < 2; ++pass) {
                const int kFrom = pass ? 2 : 1;
                const int kTo   = pass ? cfg.numCellsZ : 1;
                const int kmOff = pass ? 1 : (periodic ? 1 - cfg.numCellsZ : 1);
                pressureSourceRow(vx+a, vx+b, vx+c, vx+d, vy+a, vy+b, vy+c, vy+d,
                                  vz+a, vz+b, vz+c, vz+d, ax+a, ax+b, ax+c, ax+d,
                                  ay+a, ay+b, ay+c, ay+d, az+a, az+b, az+c, az+d,
                                  psrc+a, kFrom, kTo, kmOff,
                                  qInvDx, qInvDy, qInvDz, invDt);
            }
        }
    }
}

// ---------------------------------------------------------------------------
//  gaussSeidelSweeps — n SOR sweeps over the active domain.
//
//  Factored out of solvePressurePoisson unchanged so the multigrid path can use
//  the identical smoother; nothing about the iteration moved.
// ---------------------------------------------------------------------------
static void gaussSeidelSweeps(SimState& s, int nSweeps)
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

    // Swept (i, j, k) with k — GridField's unit-stride index — innermost.
    // The original order was (i, k, j); both are lexicographic orders whose
    // predecessor set for this 7-point star is exactly {im,jm,km}, so the
    // Gauss-Seidel iteration is bit-identical, but j-innermost strode sK
    // doubles per cell. Everything the k-loop cannot vary is hoisted.
    const bool solidWall = (cfg.lateralCondition == LateralBC::SolidWall);
    const int  nZ = cfg.numCellsZ;
    const double omega = cfg.sorOmega;

    for (int sweep = 0; sweep < nSweeps; ++sweep) {
        for (int i = 1; i <= cfg.numCellsX; ++i) {
            const int im = i-1, ip = i+1;
            int jLoopS, jLoopN; mirrorJRange(s, i, jLoopS, jLoopN);
            const bool iIsLeftEdge  = (i == 1);
            const bool iIsRightEdge = (i == cfg.numCellsX);

            for (int j = jLoopS; j <= jLoopN; ++j) {
                const int jm = j-1, jp = j+1;

                // Neumann ghost-cell mirroring — which faces mirror is fixed
                // for the whole k-row.
                const bool mirrorW = iIsLeftEdge  || (i == s.iLow[j]+1);
                const bool mirrorE = iIsRightEdge || (i == s.iHigh[j]);
                const bool mirrorS = (j == jLoopS);
                const bool mirrorN = (j == jLoopN);
                const bool corner  = (iIsLeftEdge || iIsRightEdge)
                                  && (j == s.jLow[i]+1 || j == s.jHigh[i]);
                const bool rowHasRef = (i == iRef && j == jRef);

                // The common row mirrors nothing, holds no reference node and
                // is not a corner, so every test in the general loop below is
                // loop-invariant and false there. Run it without them, peeling
                // the only two k that wrap or write a z ghost.
                if (!mirrorW && !mirrorE && !mirrorS && !mirrorN
                    && !corner && !rowHasRef && nZ >= 2) {
                    auto plainCell = [&](int k, int km, int kp) {
                        const double pNew = (cY*(s.press(i,jp,k) + s.press(i,jm,k))
                                          + cX*(s.press(ip,j,k) + s.press(im,j,k))
                                          + cZ*(s.press(i,j,kp) + s.press(i,j,km))
                                          - s.pressureSource(i,j,k)) * invDiag;
                        s.press(i, j, k) += omega * (pNew - s.press(i, j, k));
                    };
                    if (solidWall) s.press(i, j, 0) = s.press(i, j, 1);
                    plainCell(1, solidWall ? 0 : nZ, 2);
                    for (int k = 2; k <= nZ-1; ++k) plainCell(k, k-1, k+1);
                    if (solidWall) s.press(i, j, nZ+1) = s.press(i, j, nZ);
                    plainCell(nZ, nZ-1, solidWall ? nZ+1 : 1);
                    continue;
                }

                for (int k = 1; k <= nZ; ++k) {
                    int km = k-1, kp = k+1;

                    if (mirrorW) s.press(im, j, k) = s.press(i, j, k);
                    if (mirrorE) s.press(ip, j, k) = s.press(i, j, k);
                    if (mirrorS) s.press(i, jm, k) = s.press(i, j, k);
                    if (mirrorN) s.press(i, jp, k) = s.press(i, j, k);
                    if (solidWall) {
                        if (k == 1)  s.press(i, j, km) = s.press(i, j, k);
                        if (k == nZ) s.press(i, j, kp) = s.press(i, j, k);
                    } else {
                        if (k == 1)  km = nZ;
                        if (k == nZ) kp = 1;
                    }

                    if (rowHasRef && k == kRef) {
                        s.press(i, j, k) = pRef;
                    } else {
                        double pNew = (cY*(s.press(i,jp,k) + s.press(i,jm,k))
                                    + cX*(s.press(ip,j,k) + s.press(im,j,k))
                                    + cZ*(s.press(i,j,kp) + s.press(i,j,km))
                                    - s.pressureSource(i,j,k)) * invDiag;
                        // Corner correction
                        if (corner) {
                            pNew -= s.pressureSource(i,j,k) * invDiag;
                            if (solidWall && (k == 1 || k == nZ))
                                pNew -= 2.0 * s.pressureSource(i,j,k) * invDiag;
                        }
                        // SOR: over-relax the Gauss-Seidel update (omega=1
                        // reduces to plain Gauss-Seidel) — same per-cell
                        // cost, converges to the same fixed point in fewer
                        // sweeps.
                        s.press(i, j, k) += omega * (pNew - s.press(i, j, k));
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
//  solvePressurePoisson — fixed sweeps, or one coarse-grid correction.
//
//  Gauss-Seidel removes high-frequency error in a sweep or two and then stalls:
//  measured on the production shape, the interior residual improves only 3.7x
//  over 500 sweeps and is still moving, so the 5 sweeps actually run leave the
//  projection about 2x less converged than the discretisation allows. The
//  multigrid path attacks exactly that smooth error. Default is unchanged.
// ---------------------------------------------------------------------------
void solvePressurePoisson(SimState& s)
{
    const auto& cfg = s.cfg;

    if (cfg.pressureSolver == PressureSolver::GaussSeidel) {
        gaussSeidelSweeps(s, cfg.numPressureIter);
        return;
    }

    MultigridWorkspace& mg = s.ext.mg;
    if (!mg.built) mgBuild(s, mg);

    gaussSeidelSweeps(s, cfg.mgPreSweeps);      // damp what GS is good at
    mgFineResidual(s, mg.fineRes);              // what it left behind
    mgRestrict(s, mg);                          // onto the coarse grid
    mgCoarseSolve(s, mg, cfg.mgCoarseSweeps);   // smooth error, cheaply
    mgProlongAdd(s, mg);                        // correct the fine solution
    gaussSeidelSweeps(s, cfg.mgPostSweeps);     // clean up interpolation error
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

    // Same shape as buildPressureSource: restrict pointers, and the periodic
    // wrap peeled so kp stays affine. See KernelRows.hpp.
    const auto gs = s.press.gridSize();
    const std::size_t sJ = static_cast<std::size_t>(gs.sJ), sK = static_cast<std::size_t>(gs.sK);
    const double* __restrict pr = s.press.data().data();
    const double* __restrict ax = s.accelX.data().data();
    const double* __restrict ay = s.accelY.data().data();
    const double* __restrict az = s.accelZ.data().data();
    double* __restrict vx = s.velX.data().data();
    double* __restrict vy = s.velY.data().data();
    double* __restrict vz = s.velZ.data().data();

    const bool wraps = (KKfim == cfg.numCellsZ);   // kp = 1 on the last k
    const int kAffineTo = wraps ? KKfim - 1 : KKfim;

    for (int i = 1; i <= s.numCellsXm1; ++i) {
        const int ip = i + 1;
        for (int j = s.jLow[i]+1; j <= s.jHigh[i]-1; ++j) {
            const int jp = j + 1;
            const std::size_t a = (static_cast<std::size_t>(i)  * sJ + j ) * sK;
            const std::size_t b = (static_cast<std::size_t>(ip) * sJ + j ) * sK;
            const std::size_t c = (static_cast<std::size_t>(i)  * sJ + jp) * sK;
            const std::size_t d = (static_cast<std::size_t>(ip) * sJ + jp) * sK;
            const std::size_t o = (static_cast<std::size_t>(i)  * sJ + j ) * sK;

            velocityUpdateRow(pr+a, pr+b, pr+c, pr+d, ax+o, ay+o, az+o,
                              vx+o, vy+o, vz+o, 1, kAffineTo, 1,
                              qInvDx, qInvDy, qInvDz, dt_eff);
            if (wraps)
                velocityUpdateRow(pr+a, pr+b, pr+c, pr+d, ax+o, ay+o, az+o,
                                  vx+o, vy+o, vz+o, KKfim, KKfim,
                                  1 - cfg.numCellsZ, qInvDx, qInvDy, qInvDz, dt_eff);

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
    const double qInvDx = 0.25 / cfg.cellSizeX;
    const double qInvDy = 0.25 / cfg.cellSizeY;
    const double qInvDz = 0.25 / cfg.cellSizeZ;
    const int KKfim = (cfg.lateralCondition == LateralBC::SolidWall) ? s.numCellsZm1 : cfg.numCellsZ;

    s.momentumResidMax = 0.0;
    s.momentumResidRMS = 0.0;
    long long count = 0;

    // No scratchField.fill: this kernel is the only writer, and the cells it
    // skips have held zero since the field was constructed. Same reasoning as
    // computeAccelerations' reset.
    //
    // The stencil goes through momentumResidualRow so it vectorises; the max
    // and the RMS sum stay scalar and in order, since a vectorised sum would
    // reassociate.
    const auto gs = s.press.gridSize();
    const std::size_t sJ = static_cast<std::size_t>(gs.sJ), sK = static_cast<std::size_t>(gs.sK);
    const double* __restrict pr = s.press.data().data();
    const double* __restrict ax = s.accelX.data().data();
    const double* __restrict ay = s.accelY.data().data();
    const double* __restrict az = s.accelZ.data().data();
    double* __restrict scr = s.scratchField.data().data();
    double* __restrict sq  = s.ext.accel.rowBuf.data();

    const bool wraps = (KKfim == cfg.numCellsZ);
    const int kAffineTo = wraps ? KKfim - 1 : KKfim;

    for (int i = 1; i <= s.numCellsXm1; ++i) {
        const int ip = i + 1;
        for (int j = s.jLow[i]+1; j <= s.jHigh[i]-1; ++j) {
            const int jp = j + 1;
            const std::size_t a = (static_cast<std::size_t>(i)  * sJ + j ) * sK;
            const std::size_t b = (static_cast<std::size_t>(ip) * sJ + j ) * sK;
            const std::size_t c = (static_cast<std::size_t>(i)  * sJ + jp) * sK;
            const std::size_t d = (static_cast<std::size_t>(ip) * sJ + jp) * sK;

            momentumResidualRow(pr+a, pr+b, pr+c, pr+d, ax+a, ay+a, az+a,
                                scr+a, sq, 1, kAffineTo, 1, qInvDx, qInvDy, qInvDz);
            if (wraps)
                momentumResidualRow(pr+a, pr+b, pr+c, pr+d, ax+a, ay+a, az+a,
                                    scr+a, sq, KKfim, KKfim, 1 - cfg.numCellsZ,
                                    qInvDx, qInvDy, qInvDz);

            for (int k = 1; k <= KKfim; ++k) {
                s.momentumResidMax = std::max(s.momentumResidMax, scr[a+k]);
                s.momentumResidRMS += sq[k];
                ++count;
            }
        }
    }
    if (count > 0)
        s.momentumResidRMS = std::sqrt(s.momentumResidRMS / static_cast<double>(count));
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

    // The stencil goes through divergenceRow so it vectorises; the three
    // reductions stay here, scalar and in the original order, because a
    // vectorised sum would reassociate and stop being bit-identical.
    const auto gs = s.velX.gridSize();
    const std::size_t sJ = static_cast<std::size_t>(gs.sJ), sK = static_cast<std::size_t>(gs.sK);
    const double* __restrict vx = s.velX.data().data();
    const double* __restrict vy = s.velY.data().data();
    const double* __restrict vz = s.velZ.data().data();
    double* __restrict row = s.ext.accel.rowBuf.data();
    const bool periodic = (cfg.lateralCondition == LateralBC::Periodic);

    for (int i = 1; i <= cfg.numCellsX; ++i) {
        const int im = i - 1;
        int jS, jN; activeJRange(s, i, jS, jN);

        for (int j = jS; j <= jN; ++j) {
            const int jm = j - 1;
            const std::size_t a = (static_cast<std::size_t>(i)  * sJ + j ) * sK;
            const std::size_t b = (static_cast<std::size_t>(im) * sJ + j ) * sK;
            const std::size_t c = (static_cast<std::size_t>(i)  * sJ + jm) * sK;
            const std::size_t d = (static_cast<std::size_t>(im) * sJ + jm) * sK;

            for (int pass = 0; pass < 2; ++pass) {
                const int kFrom = pass ? 2 : 1;
                const int kTo   = pass ? cfg.numCellsZ : 1;
                const int kmOff = pass ? 1 : (periodic ? 1 - cfg.numCellsZ : 1);
                divergenceRow(vx+a, vx+b, vx+c, vx+d, vy+a, vy+b, vy+c, vy+d,
                              vz+a, vz+b, vz+c, vz+d, row, kFrom, kTo, kmOff,
                              qInvDx, qInvDy, qInvDz);
            }

            for (int k = 1; k <= cfg.numCellsZ; ++k) {
                const double div = row[k];
                s.intDivergence    += div;
                s.intAbsDivergence += std::abs(div);
                s.dilatationMax     = std::max(s.dilatationMax, std::abs(div));
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