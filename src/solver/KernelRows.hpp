#pragma once
// =============================================================================
//  KernelRows.hpp — one k-row of a kernel, for both host backends.
//
//  These live here because of how they compile, not to share code:
//  written in place the vectoriser reports "no vectype for stmt" and leaves the
//  row scalar, while the identical loop in its own function vectorises to
//  32-byte vectors. noinline keeps it that way; one call per (i,j) row costs
//  nothing against numCellsZ cells.
// =============================================================================

#include <cmath>

__attribute__((noinline)) inline
void pressureSourceRow(const double* __restrict vxa, const double* __restrict vxb,
                       const double* __restrict vxc, const double* __restrict vxd,
                       const double* __restrict vya, const double* __restrict vyb,
                       const double* __restrict vyc, const double* __restrict vyd,
                       const double* __restrict vza, const double* __restrict vzb,
                       const double* __restrict vzc, const double* __restrict vzd,
                       const double* __restrict axa, const double* __restrict axb,
                       const double* __restrict axc, const double* __restrict axd,
                       const double* __restrict aya, const double* __restrict ayb,
                       const double* __restrict ayc, const double* __restrict ayd,
                       const double* __restrict aza, const double* __restrict azb,
                       const double* __restrict azc, const double* __restrict azd,
                       double* __restrict out, int kFrom, int kTo, int kmOff,
                       double qInvDx, double qInvDy, double qInvDz, double invDt)
{
    for (int k = kFrom; k <= kTo; ++k) {
        const int q = k - kmOff;

        const double divU = (vxa[k] - vxb[k] + vxc[k] - vxd[k]
                           + vxa[q] - vxb[q] + vxc[q] - vxd[q]) * qInvDx;
        const double divV = (vya[k] - vyc[k] + vyb[k] - vyd[k]
                           + vya[q] - vyc[q] + vyb[q] - vyd[q]) * qInvDy;
        const double divW = (vza[k] + vzc[k] + vzb[k] + vzd[k]
                           - vza[q] - vzc[q] - vzb[q] - vzd[q]) * qInvDz;

        const double divAu = (axa[k] - axb[k] + axc[k] - axd[k]
                            + axa[q] - axb[q] + axc[q] - axd[q]) * qInvDx;
        const double divAv = (aya[k] - ayc[k] + ayb[k] - ayd[k]
                            + aya[q] - ayc[q] + ayb[q] - ayd[q]) * qInvDy;
        const double divAw = (aza[k] + azc[k] + azb[k] + azd[k]
                            - aza[q] - azc[q] - azb[q] - azd[q]) * qInvDz;

        out[k] = (divU + divV + divW) * invDt + (divAu + divAv + divAw);
    }
}

// One k-row of updateVelocities. Same three reasons as above: restrict
// pointers so the velocity stores cannot be thought to hit press's own
// strides, a kp that stays affine (the periodic wrap is peeled by the caller),
// and its own function so the loop is not inside an outlined clone.
__attribute__((noinline)) inline
void velocityUpdateRow(const double* __restrict pa, const double* __restrict pb,
                       const double* __restrict pc, const double* __restrict pd,
                       const double* __restrict ax, const double* __restrict ay,
                       const double* __restrict az,
                       double* __restrict vx, double* __restrict vy,
                       double* __restrict vz, int kFrom, int kTo, int kpOff,
                       double qInvDx, double qInvDy, double qInvDz, double dtEff)
{
    for (int k = kFrom; k <= kTo; ++k) {
        const int kp = k + kpOff;

        const double dpu = (pb[kp] - pa[kp] + pd[kp] - pc[kp]
                          + pb[k]  - pa[k]  + pd[k]  - pc[k])  * qInvDx;
        const double dpv = (pc[kp] - pa[kp] + pd[kp] - pb[kp]
                          + pc[k]  - pa[k]  + pd[k]  - pb[k])  * qInvDy;
        const double dpw = (pc[kp] + pa[kp] + pd[kp] + pb[kp]
                          - pc[k]  - pa[k]  - pd[k]  - pb[k])  * qInvDz;

        vx[k] += (ax[k] - dpu) * dtEff;
        vy[k] += (ay[k] - dpv) * dtEff;
        vz[k] += (az[k] - dpw) * dtEff;
    }
}

// Face average of the three cross-term coefficient planes, in place. Same
// reason as the rows above: written inline against the sweep's VM2 indexing
// the compiler reports "evolution of base is not affine" and leaves it scalar.
__attribute__((noinline)) inline
void faceAverageRow(double* __restrict ku, double* __restrict kv, double* __restrict kw,
                    const double* __restrict kuNext, const double* __restrict kvNext,
                    const double* __restrict kwNext, int n)
{
    for (int k = 0; k < n; ++k) {
        ku[k] = 0.5*(ku[k] + kuNext[k]);
        kv[k] = 0.5*(kv[k] + kvNext[k]);
        kw[k] = 0.5*(kw[k] + kwNext[k]);
    }
}

// Cross-term divergence subtracted from the three acceleration rows.
__attribute__((noinline)) inline
void crossTermRow(double* __restrict ax, double* __restrict ay, double* __restrict az,
                  const double* __restrict kuHi, const double* __restrict kvHi,
                  const double* __restrict kwHi, const double* __restrict qHi,
                  const double* __restrict kuLo, const double* __restrict kvLo,
                  const double* __restrict kwLo, const double* __restrict qLo, int n)
{
    for (int k = 0; k < n; ++k) {
        ax[k] -= (kuHi[k]*qHi[k] - kuLo[k]*qLo[k]);
        ay[k] -= (kvHi[k]*qHi[k] - kvLo[k]*qLo[k]);
        az[k] -= (kwHi[k]*qHi[k] - kwLo[k]*qLo[k]);
    }
}

// x / c, computed from a reciprocal instead of a divide -- and bit-identical to
// the divide, not an approximation of it. With r = fl(1/c), q0 = fl(x*r), the
// residual e = fma(-c, q0, x) is exact, and fma(e, r, q0) is then the
// correctly-rounded quotient (Markstein). That lets the division hoist out of
// the loop when c is loop-invariant, which localRe is: it depends only on i.
//
// Worth 11-23% of computeAccelerations, measured against the divide. The claim
// of exactness is not taken on faith -- the accuracy matrix byte-compares every
// output, and this is reverted if a single bit moves.
static inline double divInv(double x, double c, double r)
{
    const double q0 = x * r;
    const double e  = std::fma(-c, q0, x);
    return std::fma(e, r, q0);
}

// Z-sweep passes 2 and 3 for one k-row, doing pass 3's divisions inline.
// Fusing them saves a row-kernel call and two scratch round-trips per (i,j);
// split out they cost more than the vectorisation bought at short rows.
// Nothing branches, so the divisions and the stencil vectorise together.
__attribute__((noinline)) inline
void zDiffusionCrossRow(double* __restrict ax, double* __restrict ay, double* __restrict az,
                        const double* __restrict vx, const double* __restrict vy,
                        const double* __restrict vz,
                        const double* __restrict ppiu, const double* __restrict ppid,
                        const double* __restrict num, const double* __restrict den,
                        const double* __restrict dpe,
                        double* __restrict ku, double* __restrict kv, double* __restrict kw,
                        int n, double coeff, double localRe)
{
    const double rLocal = 1.0 / localRe;
    for (int t = 0; t < n; ++t) {
        const double vxm = vx[t-1], vx0 = vx[t], vxp = vx[t+1];
        const double vym = vy[t-1], vy0 = vy[t], vyp = vy[t+1];
        const double vzm = vz[t-1], vz0 = vz[t], vzp = vz[t+1];

        const double ppiuK = ppiu[t], ppidK = ppid[t];
        ax[t] += (ppiuK*(vxp-vx0) + ppidK*(vxm-vx0)) * coeff;
        ay[t] += (ppiuK*(vyp-vy0) + ppidK*(vym-vy0)) * coeff;
        az[t] += (ppiuK*(vzp-vz0) + ppidK*(vzm-vz0)) * coeff;

        const double pip = num[t] / den[t];
        const double pim = dpe[t] + pip;
        const double ciuK = (divInv(pip, localRe, rLocal)) * coeff;
        const double cidK = (divInv(pim, localRe, rLocal)) * coeff;
        ku[t] = ciuK*(vx0-vxp) + cidK*(vx0-vxm);
        kv[t] = ciuK*(vy0-vyp) + cidK*(vy0-vym);
        kw[t] = ciuK*(vz0-vzp) + cidK*(vz0-vzm);
    }
}

// The division half of the UNIFAES weight evaluation, one k-row at a time.
// The caller's scalar pass picked every operand, so nothing here branches and
// all four divisions vectorise -- which is the point: they are ~29% of
// computeAccelerations and were scalar only because the branch chain they sat
// inside blocked the loop. See docs/roofline.md.
__attribute__((noinline)) inline
void weightDivideRow(const double* __restrict num, const double* __restrict den,
                     const double* __restrict dpe, const double* __restrict qMask,
                     const double* __restrict qDen, const double* __restrict qAdd,
                     double* __restrict cEast, double* __restrict cWest,
                     double* __restrict qsi, int n, double localRe)
{
    const double rLocal = 1.0 / localRe;
    for (int t = 0; t < n; ++t) {
        const double pip = num[t] / den[t];
        const double pim = dpe[t] + pip;
        cEast[t] = divInv(pip, localRe, rLocal);
        cWest[t] = divInv(pim, localRe, rLocal);
        qsi[t]   = ((pip - 1.0) * qMask[t]) / qDen[t] + qAdd[t];
    }
}

// The X sweep's fused passes 1-3, one (i,j) k-row. Identical to ySweepRow
// below except that pass 2 assigns rather than accumulates -- X runs first,
// so this is the only write to those cells, and the reset plan skips them.
__attribute__((noinline)) inline
void xSweepRow(double* __restrict ax, double* __restrict ay, double* __restrict az,
               const double* __restrict vxm, const double* __restrict vx0c, const double* __restrict vxp,
               const double* __restrict vym, const double* __restrict vy0c, const double* __restrict vyp,
               const double* __restrict vzm, const double* __restrict vz0c, const double* __restrict vzp,
               const double* __restrict numF, const double* __restrict denF,
               const double* __restrict dpeF, const double* __restrict qMask,
               const double* __restrict qDen, const double* __restrict qAdd,
               const double* __restrict numC, const double* __restrict denC,
               const double* __restrict dpeC,
               const double* __restrict ppiwIn, double* __restrict cimOut,
               double* __restrict qsi,
               double* __restrict ku, double* __restrict kv, double* __restrict kw,
               int n, double coeff, double localRe)
{
    const double rLocal = 1.0 / localRe;
    for (int t = 0; t < n; ++t) {
        const double vxmT = vxm[t], vx0 = vx0c[t], vxpT = vxp[t];
        const double vymT = vym[t], vy0 = vy0c[t], vypT = vyp[t];
        const double vzmT = vzm[t], vz0 = vz0c[t], vzpT = vzp[t];

        // Pass 1: face coefficients (between i and i+1)
        const double pipF = numF[t] / denF[t];
        const double pimF = dpeF[t] + pipF;
        const double ppie = divInv(pipF, localRe, rLocal);
        cimOut[t] = divInv(pimF, localRe, rLocal);
        const double ppiw = ppiwIn[t];
        qsi[t] = ((pipF - 1.0) * qMask[t]) / qDen[t] + qAdd[t];

        // Pass 2: assign, not accumulate
        ax[t] = (ppie*(vxpT-vx0) + ppiw*(vxmT-vx0)) * coeff;
        ay[t] = (ppie*(vypT-vy0) + ppiw*(vymT-vy0)) * coeff;
        az[t] = (ppie*(vzpT-vz0) + ppiw*(vzmT-vz0)) * coeff;

        // Pass 3: cross-term correction (K * qsi)
        const double pipC = numC[t] / denC[t];
        const double pimC = dpeC[t] + pipC;
        const double cipC = (divInv(pipC, localRe, rLocal)) * coeff;
        const double cimC = (divInv(pimC, localRe, rLocal)) * coeff;
        ku[t] = cipC*(vx0-vxpT) + cimC*(vx0-vxmT);
        kv[t] = cipC*(vy0-vypT) + cimC*(vy0-vymT);
        kw[t] = cipC*(vz0-vzpT) + cimC*(vz0-vzmT);
    }
}

// One (i,j) k-row of the Y sweep's fused passes 1-3, doing every division
// here. The caller's scalar pass picked the operands for both weight
// evaluations -- the face one, which also feeds qsi, and the cell one -- so
// nothing branches and all six divisions vectorise. Kept as one row rather
// than a division pass plus a stencil pass: at short rows the extra call and
// scratch round-trip cost more than the vectorisation returns.
//
// ppisIn and cisOut are separate buffers the caller swaps after each j, which
// is what the single in-place ppiwRow did when this was a scalar loop: read
// j-1's south coefficient, write j's for j+1.
__attribute__((noinline)) inline
void ySweepRow(double* __restrict ax, double* __restrict ay, double* __restrict az,
               const double* __restrict vxm, const double* __restrict vx0c, const double* __restrict vxp,
               const double* __restrict vym, const double* __restrict vy0c, const double* __restrict vyp,
               const double* __restrict vzm, const double* __restrict vz0c, const double* __restrict vzp,
               const double* __restrict numF, const double* __restrict denF,
               const double* __restrict dpeF, const double* __restrict qMask,
               const double* __restrict qDen, const double* __restrict qAdd,
               const double* __restrict numC, const double* __restrict denC,
               const double* __restrict dpeC,
               const double* __restrict ppisIn, double* __restrict cisOut,
               double* __restrict qsi,
               double* __restrict ku, double* __restrict kv, double* __restrict kw,
               int n, double coeff, double localRe)
{
    const double rLocal = 1.0 / localRe;
    for (int t = 0; t < n; ++t) {
        const double vxmT = vxm[t], vx0 = vx0c[t], vxpT = vxp[t];
        const double vymT = vym[t], vy0 = vy0c[t], vypT = vyp[t];
        const double vzmT = vzm[t], vz0 = vz0c[t], vzpT = vzp[t];

        // Pass 1: face coefficients (between j and j+1)
        const double pipF = numF[t] / denF[t];
        const double pimF = dpeF[t] + pipF;
        const double ppin = divInv(pipF, localRe, rLocal);
        cisOut[t] = divInv(pimF, localRe, rLocal);
        const double ppis = ppisIn[t];
        qsi[t] = ((pipF - 1.0) * qMask[t]) / qDen[t] + qAdd[t];

        // Pass 2: diffusive part of Au, Av, Aw
        ax[t] += (ppin*(vxpT-vx0) + ppis*(vxmT-vx0)) * coeff;
        ay[t] += (ppin*(vypT-vy0) + ppis*(vymT-vy0)) * coeff;
        az[t] += (ppin*(vzpT-vz0) + ppis*(vzmT-vz0)) * coeff;

        // Pass 3: cross-term correction (K * qsi)
        const double pipC = numC[t] / denC[t];
        const double pimC = dpeC[t] + pipC;
        const double cinC = (divInv(pipC, localRe, rLocal)) * coeff;
        const double cisC = (divInv(pimC, localRe, rLocal)) * coeff;
        ku[t] = cinC*(vx0-vxpT) + cisC*(vx0-vxmT);
        kv[t] = cinC*(vy0-vypT) + cisC*(vy0-vymT);
        kw[t] = cinC*(vz0-vzpT) + cisC*(vz0-vzmT);
    }
}

// One k-row of computeDivergence's 8-corner divergence, into a scratch row.
// The kernel's reductions must stay scalar and in order to keep the sums
// bit-identical; only the stencil moves here, where it vectorises.
__attribute__((noinline)) inline
void divergenceRow(const double* __restrict vxa, const double* __restrict vxb,
                   const double* __restrict vxc, const double* __restrict vxd,
                   const double* __restrict vya, const double* __restrict vyb,
                   const double* __restrict vyc, const double* __restrict vyd,
                   const double* __restrict vza, const double* __restrict vzb,
                   const double* __restrict vzc, const double* __restrict vzd,
                   double* __restrict out, int kFrom, int kTo, int kmOff,
                   double qInvDx, double qInvDy, double qInvDz)
{
    for (int k = kFrom; k <= kTo; ++k) {
        const int q = k - kmOff;
        out[k] = (vxa[k] - vxb[k] + vxc[k] - vxd[k]
                + vxa[q] - vxb[q] + vxc[q] - vxd[q]) * qInvDx
               + (vya[k] - vyc[k] + vyb[k] - vyd[k]
                + vya[q] - vyc[q] + vyb[q] - vyd[q]) * qInvDy
               + (vza[k] + vzc[k] + vzb[k] + vzd[k]
                - vza[q] - vzc[q] - vzb[q] - vzd[q]) * qInvDz;
    }
}

// One k-row of computeMomentumResidual. Both outputs are needed: the norm goes
// to scratchField for the VTK export and feeds the max, and the square feeds
// the RMS sum -- recomputing one from the other would not be bit-identical.
__attribute__((noinline)) inline
void momentumResidualRow(const double* __restrict pa, const double* __restrict pb,
                         const double* __restrict pc, const double* __restrict pd,
                         const double* __restrict ax, const double* __restrict ay,
                         const double* __restrict az,
                         double* __restrict outNorm, double* __restrict outSq,
                         int kFrom, int kTo, int kpOff,
                         double qInvDx, double qInvDy, double qInvDz)
{
    for (int k = kFrom; k <= kTo; ++k) {
        const int kp = k + kpOff;

        const double gradPx = (pb[k]  - pa[k]  + pd[k]  - pc[k]
                             + pb[kp] - pa[kp] + pd[kp] - pc[kp]) * qInvDx;
        const double gradPy = (pc[k]  - pa[k]  + pd[k]  - pb[k]
                             + pc[kp] - pa[kp] + pd[kp] - pb[kp]) * qInvDy;
        const double gradPz = (-pc[k]  - pa[k]  - pd[k]  - pb[k]
                             +  pc[kp] + pa[kp] + pd[kp] + pb[kp]) * qInvDz;

        const double resU = ax[k] - gradPx;
        const double resV = ay[k] - gradPy;
        const double resW = az[k] - gradPz;
        const double resSq = resU*resU + resV*resV + resW*resW;
        outSq[k]   = resSq;
        outNorm[k] = std::sqrt(resSq);
    }
}
