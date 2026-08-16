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

    // Zero all acceleration arrays
    s.accelX.fill(0.0);
    s.accelY.fill(0.0);
    s.accelZ.fill(0.0);

    // Temporary 1‑D arrays (logical index -1 .. maxDim+1) offset by +1.
    // Sized maxDim+3 (not maxDim+2) in AccelScratch::allocate(): several
    // writes below reach logical index maxDim+1 (e.g. VM(ppiw, i+2) at
    // i==iEnd-1==maxDim-1, and the VM(Ku, iEnd+1)/VM(Ku, jEnd+1)/
    // VM(Ku, numCellsZ+1) boundary extrapolations), which needs physical
    // slot maxDim+2. Owned by SimState's backend Extras and reused across calls
    // instead of being allocated fresh every call — see AccelScratch.hpp for why
    // that's safe without re-zeroing.
    auto& ppiu = s.ext.accel.ppiu; auto& ppid = s.ext.accel.ppid;
    auto& qsiu = s.ext.accel.qsiu;
    auto& Ku = s.ext.accel.Ku; auto& Kv = s.ext.accel.Kv; auto& Kw = s.ext.accel.Kw;

    // X-sweep-only 2-D (i,k) scratch — see AccelScratch.hpp field comments
    // and docs/serial-optimization-loop-order.md. Replaces the 1-D
    // ppie/ppiw/qsie buffers (and this sweep's private use of Ku/Kv/Kw,
    // which the Y/Z sweeps below still use in their original 1-D form).
    auto& ppieXK = s.ext.accel.ppieXK; auto& ppiwXK = s.ext.accel.ppiwXK;
    auto& qsieXK = s.ext.accel.qsieXK;
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
                VM2(ppieXK, i+1, k) = cip;      // east coefficient at face i+0.5
                VM2(ppiwXK, i+1+1, k) = cim;    // west coefficient (ip+1, ip=i+1)
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
                VM2(ppieXK, i+1, k) = cipF;
                VM2(ppiwXK, i+1+1, k) = cimF;
                VM2(qsieXK, i+1, k) = computeQsi(DPeFace, pipF, 0.5);

                // Pass 2: diffusive part of Au, Av, Aw (first part)
                const double coeff = invDx2;
                const double ppie = VM2(ppieXK,i+1,k);   // written just above
                const double ppiw = VM2(ppiwXK,i+1,k);   // written at i-1
                s.accelX(i, j, k) += (ppie*(vxp-vx0) + ppiw*(vxm-vx0)) * coeff;
                s.accelY(i, j, k) += (ppie*(vyp-vy0) + ppiw*(vym-vy0)) * coeff;
                s.accelZ(i, j, k) += (ppie*(vzp-vz0) + ppiw*(vzm-vz0)) * coeff;

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
    auto& ppinJK = s.ext.accel.ppieXK; auto& ppisJK = s.ext.accel.ppiwXK;
    auto& qsinJK = s.ext.accel.qsieXK;
    auto& KuJK = s.ext.accel.KuXK; auto& KvJK = s.ext.accel.KvXK; auto& KwJK = s.ext.accel.KwXK;

    const double invDy2 = 1.0 / (cfg.cellSizeY * cfg.cellSizeY);
    for (int i = 1; i <= s.numCellsXm1; ++i) {
        const int jStart = s.jLow[i];
        const int jEnd   = s.jHigh[i];
        const double invRe = effectiveInvRe(s, i);
        const double localRe = 1.0 / invRe;

        for (int j = jStart; j <= jEnd-1; ++j) {
            for (int k = 1; k <= KKfim; ++k) {
                const double vFace = 0.5 * (s.velY(i, j+1, k) + s.velY(i, j, k));
                const double DPe = localRe * vFace * cfg.cellSizeY;
                double pip, cin, cis;
                computeExponentialWeights(localRe, DPe, pip, cin, cis);
                VM2(ppinJK, j+1, k) = cin;
                VM2(ppisJK, j+2, k) = cis;
                VM2(qsinJK, j+1, k) = computeQsi(DPe, pip, 0.5);
            }
        }
        for (int j = jStart+1; j <= jEnd-1; ++j) {
            for (int k = 1; k <= KKfim; ++k) {
                const double coeff = invDy2;
                s.accelX(i, j, k) += (VM2(ppinJK,j+1,k)*(s.velX(i,j+1,k)-s.velX(i,j,k))
                                     + VM2(ppisJK,j+1,k)*(s.velX(i,j-1,k)-s.velX(i,j,k))) * coeff;
                s.accelY(i, j, k) += (VM2(ppinJK,j+1,k)*(s.velY(i,j+1,k)-s.velY(i,j,k))
                                     + VM2(ppisJK,j+1,k)*(s.velY(i,j-1,k)-s.velY(i,j,k))) * coeff;
                s.accelZ(i, j, k) += (VM2(ppinJK,j+1,k)*(s.velZ(i,j+1,k)-s.velZ(i,j,k))
                                     + VM2(ppisJK,j+1,k)*(s.velZ(i,j-1,k)-s.velZ(i,j,k))) * coeff;
            }
        }
        for (int j = jStart+1; j <= jEnd-1; ++j) {
            for (int k = 1; k <= KKfim; ++k) {
                const double vCell = s.velY(i, j, k);
                const double DPe = localRe * vCell * cfg.cellSizeY;
                double pip, cin, cis;
                computeExponentialWeights(localRe, DPe, pip, cin, cis);
                cin *= invDy2;  cis *= invDy2;
                VM2(KuJK, j+1, k) = cin*(s.velX(i,j,k)-s.velX(i,j+1,k)) + cis*(s.velX(i,j,k)-s.velX(i,j-1,k));
                VM2(KvJK, j+1, k) = cin*(s.velY(i,j,k)-s.velY(i,j+1,k)) + cis*(s.velY(i,j,k)-s.velY(i,j-1,k));
                VM2(KwJK, j+1, k) = cin*(s.velZ(i,j,k)-s.velZ(i,j+1,k)) + cis*(s.velZ(i,j,k)-s.velZ(i,j-1,k));
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
//  solvePressurePoisson — Gauss-Seidel for ∇²p = S
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

    // Swept (i, j, k) with k — GridField's unit-stride index — innermost.
    // The original order was (i, k, j); both are lexicographic orders whose
    // predecessor set for this 7-point star is exactly {im,jm,km}, so the
    // Gauss-Seidel iteration is bit-identical, but j-innermost strode sK
    // doubles per cell. Everything the k-loop cannot vary is hoisted.
    const bool solidWall = (cfg.lateralCondition == LateralBC::SolidWall);
    const int  nZ = cfg.numCellsZ;
    const double omega = cfg.sorOmega;

    for (int sweep = 0; sweep < cfg.numPressureIter; ++sweep) {
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
    long long count = 0;

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

                s.momentumResidMax = std::max(s.momentumResidMax, resNorm);
                s.momentumResidRMS += resSq;
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