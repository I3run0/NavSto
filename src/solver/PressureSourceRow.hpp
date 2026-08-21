#pragma once
// =============================================================================
//  PressureSourceRow.hpp — one k-row of buildPressureSource, for both host
//  backends.
//
//  Lifted out of the kernel because of how it compiles, not to share code:
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
