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
