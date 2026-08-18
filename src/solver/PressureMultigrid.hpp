#pragma once
// =============================================================================
//  PressureMultigrid.hpp — two-level correction scheme for the pressure solve.
//
//  Gauss-Seidel kills high-frequency error in a sweep or two and then crawls:
//  measured on the production shape, interior residual falls 3.7x over 500
//  sweeps and is still moving. A coarse-grid correction is the standard answer
//  — the smooth error GS cannot see is well represented on a grid half as fine
//  and cheap to solve there.
//
//  Deliberately two-level, not a full V-cycle: it answers whether the coarse
//  correction beats plain sweeping at all before anyone builds a hierarchy.
//
//  STATUS: IT DOES NOT. This does not converge and is worse than the Gauss-
//  Seidel it replaces -- 1 cycle leaves interior RMS 6.07e-3 where 5 GS sweeps
//  reach 2.81e-3, and more cycles make it worse. cfg.pressureSolver defaults to
//  GaussSeidel and should stay there. Kept, behind that flag, because the
//  failure is informative and a second attempt should start here rather than
//  from scratch.
//
//  Two causes were found and fixed and did NOT rescue it: the coarse problem is
//  singular under homogeneous Neumann (fixed by projecting rhs and the
//  correction to zero mean, which turned outright divergence into stagnation),
//  and the fine gauge drifts when a correction moves the pinned reference node
//  (fixed by anchoring the correction there; no measurable effect).
//
//  What is left, in the order worth trying:
//    1. The transfer pair is not variational. Full-weighting restriction's
//       transpose is TRILINEAR prolongation, not the piecewise-constant
//       injection below, so the coarse correction is not a Galerkin correction
//       and there is no theory saying it must converge. Most likely culprit.
//    2. The coarse operator is the plain Laplacian while the fine operator
//       carries corner source scaling and inline mirroring, so the two solve
//       different problems near the boundary.
//    3. mgBuild contracts the coarse span inward (jLow rounds up, jHigh rounds
//       down). On a ramped mask like RoundedCorner that leaves a band along the
//       whole boundary with no coarse cell above it, and mgProlongAdd then
//       clamps into range and applies a correction belonging to a different
//       column there.
// =============================================================================

#include "Geometry.hpp"
#include "MultigridWorkspace.hpp"
#include "SimState.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

/// Geometry is fixed once initSimulation() has run, so this runs once.
inline void mgBuild(const SimState& s, MultigridWorkspace& mg)
{
    const auto& cfg = s.cfg;
    mg.nx = cfg.numCellsX / 2;
    mg.ny = cfg.numCellsY / 2;
    mg.nz = cfg.numCellsZ / 2;

    // Coarse span per coarse column, contracted inward from the fine spans.
    //
    // Expanding it instead -- covering every fine cell, rounding outward -- was
    // tried and diverges hard (interior RMS 2.4e8 after 500 cycles). Coarse
    // cells with no active fine child restrict to rhs = 0 yet stay coupled to
    // the active region through the Laplacian, so they are unconstrained
    // unknowns that grow and feed back. Contraction leaves a thin uncovered
    // band along a ramped mask, which is a real but far smaller error.
    mg.jLow.assign(mg.nx + 2, 0);
    mg.jHigh.assign(mg.nx + 2, 0);
    for (int ic = 0; ic <= mg.nx; ++ic) {
        const int i = std::min(2 * ic, cfg.numCellsX);
        mg.jLow[ic]  = (s.jLow[i] + 1) / 2;   // round up
        mg.jHigh[ic] = s.jHigh[i] / 2;        // round down
        if (mg.jHigh[ic] < mg.jLow[ic]) mg.jHigh[ic] = mg.jLow[ic];
    }

    mg.corr.resize(GridSize{mg.nx + 2, mg.ny + 2, mg.nz + 2});
    mg.rhs .resize(GridSize{mg.nx + 2, mg.ny + 2, mg.nz + 2});
    mg.fineRes.resize(s.g);
    mg.built = true;
}

// ── Pieces of one correction pass ────────────────────────────────────────────

/// Re-mirror the Neumann ghost layer against the CURRENT pressure field.
///
/// The smoother writes each row's ghosts before it updates that row, so after
/// the last sweep every ghost is one update stale. The residual stencil reads
/// those ghosts, so it has to see them refreshed or it measures a lag rather
/// than an error. Same rule set as the smoother, extracted.
inline void mgMirrorGhosts(SimState& s)
{
    const auto& cfg = s.cfg;
    const bool solid = (cfg.lateralCondition == LateralBC::SolidWall);
    const int nZ = cfg.numCellsZ;

    for (int i = 1; i <= cfg.numCellsX; ++i) {
        int jLoopS, jLoopN; mirrorJRange(s, i, jLoopS, jLoopN);
        for (int j = jLoopS; j <= jLoopN; ++j) {
            const bool mirrorW = (i == 1) || (i == s.iLow[j] + 1);
            const bool mirrorE = (i == cfg.numCellsX) || (i == s.iHigh[j]);
            const bool mirrorS = (j == jLoopS);
            const bool mirrorN = (j == jLoopN);
            for (int k = 1; k <= nZ; ++k) {
                if (mirrorW) s.press(i-1, j, k) = s.press(i, j, k);
                if (mirrorE) s.press(i+1, j, k) = s.press(i, j, k);
                if (mirrorS) s.press(i, j-1, k) = s.press(i, j, k);
                if (mirrorN) s.press(i, j+1, k) = s.press(i, j, k);
                if (solid) {
                    if (k == 1)  s.press(i, j, k-1) = s.press(i, j, k);
                    if (k == nZ) s.press(i, j, k+1) = s.press(i, j, k);
                }
            }
        }
    }
}

/// r = S - lap(p) on the fine grid, over the cells the smoother owns.
///
/// Note what is deliberately NOT here. Setting pNew == p in the smoother's
/// update gives lap(p) = f*S, where the corner correction makes f = 2 on a
/// corner cell and f = 4 when a SolidWall k face is also involved -- so the
/// algebraically correct right-hand side is f*S, and the coarse grid's plain
/// Laplacian was always the right OPERATOR. Feeding f*S here was tried and is
/// measurably worse (5 cycles of 2,16,2: 1.65e-3 with f = 1 against 2.56e-3
/// with the true f). The reason is a multigrid one rather than an algebraic
/// one: f*S puts a large, sharply localised residual on corner cells, which is
/// exactly the high-frequency content a coarse grid cannot represent, so
/// restricting it injects error instead of removing it. Damping the boundary
/// residual is standard practice; f = 1 is the crude version of it.
///
/// Bounds follow the smoother's own (mirrorJRange), not activeJRange, and the
/// pinned reference cell carries no equation.
inline void mgFineResidual(SimState& s, GridField<>& res)
{
    const auto& cfg = s.cfg;
    const double cX = 1.0 / s.cellSizeXsq, cY = 1.0 / s.cellSizeYsq, cZ = 1.0 / s.cellSizeZsq;
    const bool solid = (cfg.lateralCondition == LateralBC::SolidWall);
    const int nZ = cfg.numCellsZ;
    const int iRef = cfg.numCellsX;
    const int jRef = (s.jHigh[cfg.numCellsX] + s.jLow[cfg.numCellsX]) / 2;
    const int kRef = (nZ + 1) / 2;

    mgMirrorGhosts(s);
    res.fill(0.0);

    for (int i = 1; i <= cfg.numCellsX; ++i) {
        int jLoopS, jLoopN; mirrorJRange(s, i, jLoopS, jLoopN);
        const bool iEdge = (i == 1 || i == cfg.numCellsX);
        for (int j = jLoopS; j <= jLoopN; ++j) {
            const bool corner = iEdge && (j == s.jLow[i] + 1 || j == s.jHigh[i]);
            for (int k = 1; k <= nZ; ++k) {
                if (i == iRef && j == jRef && k == kRef) continue;  // pinned
                int km = k-1, kp = k+1;
                if (!solid) { if (k == 1) km = nZ; if (k == nZ) kp = 1; }
                const double lap =
                      cX*(s.press(i+1,j,k) - 2.0*s.press(i,j,k) + s.press(i-1,j,k))
                    + cY*(s.press(i,j+1,k) - 2.0*s.press(i,j,k) + s.press(i,j-1,k))
                    + cZ*(s.press(i,j,kp)  - 2.0*s.press(i,j,k) + s.press(i,j,km));
                (void)corner;   // see the note above on why f stays 1
                const double f = 1.0;
                res(i,j,k) = f * s.pressureSource(i,j,k) - lap;
            }
        }
    }
}

/// Full-weighting restriction: each coarse cell averages its eight fine
/// children. Children outside the fine active span contribute nothing and are
/// not counted, so a coarse cell on the mask edge still gets a sane mean.
inline void mgRestrict(const SimState& s, MultigridWorkspace& mg)
{
    const auto& cfg = s.cfg;
    mg.rhs.fill(0.0);
    mg.corr.fill(0.0);

    for (int ic = 1; ic <= mg.nx; ++ic)
        for (int jc = mg.jLow[ic] + 1; jc <= mg.jHigh[ic]; ++jc)
            for (int kc = 1; kc <= mg.nz; ++kc) {
                double sum = 0.0; int n = 0;
                for (int di = 0; di < 2; ++di)
                    for (int dj = 0; dj < 2; ++dj)
                        for (int dk = 0; dk < 2; ++dk) {
                            const int i = 2*ic - di, j = 2*jc - dj, k = 2*kc - dk;
                            if (i < 1 || i > cfg.numCellsX) continue;
                            int jS, jN; activeJRange(s, i, jS, jN);
                            if (j < jS || j > jN) continue;
                            if (k < 1 || k > cfg.numCellsZ) continue;
                            sum += mg.fineRes(i,j,k); ++n;
                        }
                mg.rhs(ic,jc,kc) = n ? sum / n : 0.0;
            }
}

/// Mean of a coarse field over the active cells.
inline double mgCoarseMean(const MultigridWorkspace& mg, const GridField<>& f)
{
    double sum = 0.0; long long n = 0;
    for (int i = 1; i <= mg.nx; ++i)
        for (int j = mg.jLow[i] + 1; j <= mg.jHigh[i]; ++j)
            for (int k = 1; k <= mg.nz; ++k) { sum += f(i,j,k); ++n; }
    return n ? sum / static_cast<double>(n) : 0.0;
}

inline void mgCoarseShift(const MultigridWorkspace& mg, GridField<>& f, double d)
{
    for (int i = 1; i <= mg.nx; ++i)
        for (int j = mg.jLow[i] + 1; j <= mg.jHigh[i]; ++j)
            for (int k = 1; k <= mg.nz; ++k) f(i,j,k) -= d;
}

/// Gauss-Seidel on the coarse correction equation lap(e) = rhs, homogeneous
/// Neumann by mirroring, e = 0 initially. The coarse operator is the plain
/// Laplacian: a coarse grid only has to represent the smooth error well, it
/// does not have to reproduce the fine solver's boundary treatment.
///
/// The null space has to be handled explicitly, and getting this wrong makes
/// the whole scheme DIVERGE rather than converge slowly. Homogeneous Neumann on
/// every face leaves the coarse operator singular: a constant is in its kernel.
/// So the right-hand side is projected to zero mean first (otherwise the
/// problem is unsolvable and the constant mode grows without bound), and the
/// correction is projected to zero mean afterwards (otherwise it carries an
/// arbitrary constant onto a fine field whose reference node is pinned, and the
/// next smoothing sweep tears that one node back and manufactures residual).
inline void mgCoarseSolve(const SimState& s, MultigridWorkspace& mg, int sweeps)
{
    const auto& cfg = s.cfg;
    const double hx = 2.0 * cfg.cellSizeX, hy = 2.0 * cfg.cellSizeY, hz = 2.0 * cfg.cellSizeZ;
    const double cX = 1.0/(hx*hx), cY = 1.0/(hy*hy), cZ = 1.0/(hz*hz);
    const double invDiag = 0.5 / (cX + cY + cZ);
    const bool solid = (cfg.lateralCondition == LateralBC::SolidWall);
    // Over-relaxation is for the fine smoother; on a singular coarse problem it
    // amplifies the very mode being projected out. Plain Gauss-Seidel here.
    const double omega = 1.0;

    mgCoarseShift(mg, mg.rhs, mgCoarseMean(mg, mg.rhs));   // enforce solvability

    for (int sweep = 0; sweep < sweeps; ++sweep)
        for (int i = 1; i <= mg.nx; ++i) {
            const int jS = mg.jLow[i] + 1, jN = mg.jHigh[i];
            for (int j = jS; j <= jN; ++j)
                for (int k = 1; k <= mg.nz; ++k) {
                    int km = k-1, kp = k+1;
                    if (i == 1)      mg.corr(i-1,j,k) = mg.corr(i,j,k);
                    if (i == mg.nx)  mg.corr(i+1,j,k) = mg.corr(i,j,k);
                    if (j == jS)     mg.corr(i,j-1,k) = mg.corr(i,j,k);
                    if (j == jN)     mg.corr(i,j+1,k) = mg.corr(i,j,k);
                    if (solid) {
                        if (k == 1)     mg.corr(i,j,km) = mg.corr(i,j,k);
                        if (k == mg.nz) mg.corr(i,j,kp) = mg.corr(i,j,k);
                    } else {
                        if (k == 1)     km = mg.nz;
                        if (k == mg.nz) kp = 1;
                    }
                    const double eNew = (cX*(mg.corr(i+1,j,k) + mg.corr(i-1,j,k))
                                       + cY*(mg.corr(i,j+1,k) + mg.corr(i,j-1,k))
                                       + cZ*(mg.corr(i,j,kp)  + mg.corr(i,j,km))
                                       - mg.rhs(i,j,k)) * invDiag;
                    mg.corr(i,j,k) += omega * (eNew - mg.corr(i,j,k));
                }
        }

    // Anchor the correction at the coarse cell holding the fine grid's pinned
    // reference node, not at the mean. The fine smoother pins press(iRef) to
    // whatever it held on entry; a correction that moves that node shifts the
    // gauge every cycle, and the next sweep tears the one pinned node back
    // against a field that has drifted around it.
    const int iRefC = std::min((cfg.numCellsX + 1) / 2, mg.nx);
    int jRefC = ((s.jHigh[cfg.numCellsX] + s.jLow[cfg.numCellsX]) / 2 + 1) / 2;
    jRefC = std::max(mg.jLow[iRefC] + 1, std::min(jRefC, mg.jHigh[iRefC]));
    const int kRefC = std::min(((cfg.numCellsZ + 1) / 2 + 1) / 2, mg.nz);
    mgCoarseShift(mg, mg.corr, mg.corr(iRefC, jRefC, kRefC));
}

/// Trilinear prolongation.
///
/// Cell-centred coarsening puts coarse cell ic over fine cells 2ic-1 and 2ic,
/// so a fine cell sits either side of its parent's centre and interpolates
/// 3/4 from that parent and 1/4 from the neighbour it leans toward. In 3-D that
/// is the product of the three 1-D weights over eight coarse cells.
///
/// Replaces piecewise-constant injection, whose jumps at every coarse-cell
/// boundary inject high-frequency error that one post-smoothing sweep cannot
/// remove.
inline void mgProlongAdd(SimState& s, const MultigridWorkspace& mg)
{
    const auto& cfg = s.cfg;
    const bool periodic = (cfg.lateralCondition != LateralBC::SolidWall);

    for (int i = 1; i <= cfg.numCellsX; ++i) {
        int jS, jN; activeJRange(s, i, jS, jN);

        int icA = std::min(std::max((i + 1) / 2, 1), mg.nx);
        int icB = (i & 1) ? icA - 1 : icA + 1;
        icB = std::min(std::max(icB, 1), mg.nx);

        for (int j = jS; j <= jN; ++j) {
            const int jcRaw = (j + 1) / 2;
            const int jcNbr = (j & 1) ? jcRaw - 1 : jcRaw + 1;
            // Each coarse column has its own active span, so clamp per column.
            auto clampJ = [&](int ic, int jc) {
                return std::min(std::max(jc, mg.jLow[ic] + 1), mg.jHigh[ic]);
            };
            const int jcA_iA = clampJ(icA, jcRaw), jcB_iA = clampJ(icA, jcNbr);
            const int jcA_iB = clampJ(icB, jcRaw), jcB_iB = clampJ(icB, jcNbr);

            for (int k = 1; k <= cfg.numCellsZ; ++k) {
                int kcA = (k + 1) / 2;
                int kcB = (k & 1) ? kcA - 1 : kcA + 1;
                if (periodic) {                       // z wraps
                    if (kcA < 1)     kcA += mg.nz;
                    if (kcA > mg.nz) kcA -= mg.nz;
                    if (kcB < 1)     kcB += mg.nz;
                    if (kcB > mg.nz) kcB -= mg.nz;
                } else {
                    kcA = std::min(std::max(kcA, 1), mg.nz);
                    kcB = std::min(std::max(kcB, 1), mg.nz);
                }

                const double e =
                    0.75*0.75*0.75 * mg.corr(icA, jcA_iA, kcA)
                  + 0.75*0.75*0.25 * mg.corr(icA, jcA_iA, kcB)
                  + 0.75*0.25*0.75 * mg.corr(icA, jcB_iA, kcA)
                  + 0.75*0.25*0.25 * mg.corr(icA, jcB_iA, kcB)
                  + 0.25*0.75*0.75 * mg.corr(icB, jcA_iB, kcA)
                  + 0.25*0.75*0.25 * mg.corr(icB, jcA_iB, kcB)
                  + 0.25*0.25*0.75 * mg.corr(icB, jcB_iB, kcA)
                  + 0.25*0.25*0.25 * mg.corr(icB, jcB_iB, kcB);
                s.press(i,j,k) += e;
            }
        }
    }
}
