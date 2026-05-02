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
#include "Logger.hpp"

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
//  Effective 1/Re accounting for the hyper-viscous sponge layer.
// ---------------------------------------------------------------------------
double effectiveInvRe(const SimState& s, int i)
{
    if (s.cfg.hyperViscousStart == 0 || i <= s.cfg.numCellsX - s.cfg.hyperViscousStart)
        return 1.0 / s.cfg.reynoldsNumber;

    const double hpi    = 2.0 * std::atan(1.0);  // π/2
    const int IIorig    = s.cfg.numCellsX - 5 * s.cfg.hyperViscousStart / 8;
    const int IIamp     = 3 * s.cfg.hyperViscousStart / 8;
    const double zReOrig = 0.5 * (1.0/s.cfg.hyperViscousRe + 1.0/s.cfg.reynoldsNumber);
    const double zReAmp  = 0.5 * (-1.0/s.cfg.hyperViscousRe + 1.0/s.cfg.reynoldsNumber);

    if (i < s.cfg.numCellsX - s.cfg.hyperViscousStart / 4)
        return zReOrig - zReAmp * std::sin(((i - IIorig) / (double)IIamp) * hpi);
    return 1.0 / s.cfg.hyperViscousRe;
}

// ---------------------------------------------------------------------------
//  Fully-developed parabolic (Poiseuille) x-profile at plane i = planeI.
//  Returns the required pressure gradient in presGradX_out.
// ---------------------------------------------------------------------------
void developedProfileX(SimState& s, int planeI,
                       double& presGradX_out, double& meanVelocity)
{
    const int    numJSpan   = s.jHigh[planeI] - s.jLow[planeI];
    const double spanHeight = numJSpan * s.cfg.cellSizeY;
    meanVelocity = 1.5 / spanHeight;

    for (int span = 0; span <= numJSpan; ++span) {
        const int    j     = span + s.jLow[planeI];
        const double yNorm = static_cast<double>(span) / numJSpan;
        const double uVal  = -4.0 * (yNorm - 1.0) * yNorm * meanVelocity;
        for (int k = 0; k <= s.cfg.numCellsZ; ++k)
            s.velX(planeI, j, k) = uVal;
    }

    presGradX_out = -12.0 * meanVelocity
                  / (spanHeight * spanHeight * s.cfg.reynoldsNumber);

    if (s.cfg.lateralCondition == LateralBC::SolidWall) {
        // Enforce no-slip on z walls and refine with Gauss-Seidel
        for (int j = s.jLow[planeI]+1; j <= s.jHigh[planeI]-1; ++j) {
            s.velX(planeI, j, 0)                    = 0.0;
            s.velX(planeI, j, s.cfg.numCellsZ)      = 0.0;
        }
        for (int k = 1; k <= s.numCellsZm1; ++k) {
            s.velX(planeI, s.jLow [planeI], k) = 0.0;
            s.velX(planeI, s.jHigh[planeI], k) = 0.0;
        }

        const double coeffY    = 1.0 / s.cellSizeYsq;
        const double coeffZ    = 1.0 / s.cellSizeZsq;
        const double diagCoeff = 2.0 * (coeffY + coeffZ);
        const double invDiag   = 1.0 / diagCoeff;
        const double omega     = 1.85;
        int    iter    = 0;
        double maxResid;

        do {
            maxResid = 0.0;
            ++iter;
            for (int j = s.jLow[planeI]+1; j <= s.jHigh[planeI]-1; ++j)
                for (int k = 1; k <= s.numCellsZm1; ++k) {
                    const double rhs =
                        coeffY * (s.velX(planeI, j+1, k) + s.velX(planeI, j-1, k))
                      + coeffZ * (s.velX(planeI, j, k+1) + s.velX(planeI, j, k-1))
                      - presGradX_out;
                    const double resid = std::abs(rhs - diagCoeff * s.velX(planeI, j, k));
                    maxResid = std::max(maxResid, resid);
                    const double uNew = rhs * invDiag;
                    s.velX(planeI, j, k) += omega * (uNew - s.velX(planeI, j, k));
                }
        } while (maxResid >= 1e-8 && iter < 100000);

        // Rescale so cell-averaged velocity equals exactly 1
        double integral = 0.0;
        for (int j = s.jLow[planeI]+1; j <= s.jHigh[planeI]-1; ++j)
            for (int k = 1; k <= s.numCellsZm1; ++k)
                integral += s.velX(planeI, j, k);
        integral /= (numJSpan * s.cfg.numCellsZ);
        const double scale = 1.0 / integral;
        presGradX_out *= scale;
        for (int j = s.jLow[planeI]+1; j <= s.jHigh[planeI]-1; ++j)
            for (int k = 1; k <= s.numCellsZm1; ++k)
                s.velX(planeI, j, k) *= scale;
    }
}

// ---------------------------------------------------------------------------
//  Fully-developed y-direction profile (used at curved-geometry inlets).
// ---------------------------------------------------------------------------
void developedProfileY(SimState& s, int planeJ,
                       double& presGradY_out, double& meanVelocity)
{
    const int    numISpan   = s.iHigh[planeJ] - s.iLow[planeJ];
    const double spanWidth  = numISpan * s.cfg.cellSizeX;
    meanVelocity = 1.5 / spanWidth;

    for (int span = 0; span <= numISpan; ++span) {
        const int    i     = span + s.iLow[planeJ];
        const double xNorm = static_cast<double>(span) / numISpan;
        const double vVal  = -4.0 * (xNorm - 1.0) * xNorm * meanVelocity;
        for (int k = 0; k <= s.cfg.numCellsZ; ++k)
            s.velY(i, planeJ, k) = vVal;
    }
    presGradY_out = -12.0 * meanVelocity
                  / (spanWidth * spanWidth * s.cfg.reynoldsNumber);
}

// ---------------------------------------------------------------------------
//  Hagen-Poiseuille + Bernoulli initial pressure estimate.
// ---------------------------------------------------------------------------
void buildInitialPressure(SimState& s)
{
    const int    jSpan0  = s.jHigh[0] - s.jLow[0];
    const double height0 = jSpan0 * s.cfg.cellSizeY;
    double dpDx0  = -12.0 * s.cfg.cellSizeX / (height0 * height0 * height0 * s.cfg.reynoldsNumber);
    double uMean0 = 1.0 / height0;
    s.press(0, 0, 0) = 0.0;

    for (int i = 1; i <= s.degreeIndex1; ++i) {
        const int im = i - 1;
        const double localHeight = (i != s.degreeIndex2)
            ? (s.jHigh[i]  - s.jLow[i])  * s.cfg.cellSizeY
            : (s.jHigh[im] - s.jLow[im]) * s.cfg.cellSizeY;
        const double uMean   = 1.0 / localHeight;
        const double zReloc  = effectiveInvRe(s, i);
        const double dpDx1   = -12.0 * s.cfg.cellSizeX * zReloc / (localHeight * localHeight);
        const double dpDx    = 0.5 * (dpDx0 + dpDx1)
                             + 0.5 * (uMean0*uMean0 - uMean*uMean);
        dpDx0  = dpDx1;
        uMean0 = uMean;

        s.press(i, 0, 0) = s.press(im, 0, 0) + dpDx;
        const int jStart = (i != s.degreeIndex2) ? s.jLow[i]+1 : s.jLow[im]+1;
        for (int j = jStart; j <= s.jHigh[i]; ++j)
            for (int k = 1; k <= s.cfg.numCellsZ; ++k)
                s.press(i, j, k) = s.press(i, 0, 0);
    }
    for (int i = s.degreeIndex1+1; i <= s.cfg.numCellsX; ++i) {
        s.press(i, 0, 0) = s.press(s.degreeIndex1, 0, 0);
        for (int j = s.jLow[i]+1; j <= s.jHigh[i]; ++j)
            for (int k = 1; k <= s.cfg.numCellsZ; ++k)
                s.press(i, j, k) = s.press(i, 0, 0);
    }
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

} // anonymous namespace

// ════════════════════════════════════════════════════════════════════════════
//  PUBLIC API
// ════════════════════════════════════════════════════════════════════════════

// ---------------------------------------------------------------------------
//  initSimulation — geometry + initial conditions
// ---------------------------------------------------------------------------
void initSimulation(SimState& s)
{
    const auto& cfg = s.cfg;

    s.numCellsXm1  = cfg.numCellsX - 1;
    s.numCellsYm1  = cfg.numCellsY - 1;
    s.numCellsZm1  = cfg.numCellsZ - 1;
    s.cellSizeXsq  = cfg.cellSizeX * cfg.cellSizeX;
    s.cellSizeYsq  = cfg.cellSizeY * cfg.cellSizeY;
    s.cellSizeZsq  = cfg.cellSizeZ * cfg.cellSizeZ;

    // Zero all fields
    s.velX.fill(0.0);   s.velY.fill(0.0);   s.velZ.fill(0.0);
    s.press.fill(0.0);
    s.accelX.fill(0.0); s.accelY.fill(0.0); s.accelZ.fill(0.0);
    s.pressureSource.fill(0.0);

    // ── Geometry: fill jLow/jHigh and iLow/iHigh ────────────────────────────
    switch (cfg.geometryShape) {
    case GeometryShape::AbruptExpansion:
        s.degreeIndexY = cfg.numCellsY / 2;
        s.degreeIndex1 = cfg.numCellsX / 12;
        s.degreeIndex2 = cfg.numCellsX + 2;
        for (int i = 0; i <= s.degreeIndex1; ++i)  { s.jLow[i] = s.degreeIndexY; s.jHigh[i] = cfg.numCellsY; }
        for (int i = s.degreeIndex1+1; i <= cfg.numCellsX; ++i) { s.jLow[i] = 0; s.jHigh[i] = cfg.numCellsY; }
        for (int j = 0; j <= s.degreeIndexY; ++j)  s.iLow[j] = s.degreeIndex1;
        for (int j = s.degreeIndexY+1; j <= cfg.numCellsY; ++j) s.iLow[j] = 0;
        for (int j = 0; j <= cfg.numCellsY; ++j)   s.iHigh[j] = cfg.numCellsX;
        break;

    case GeometryShape::AbruptContraction:
        s.jLowInitial  = cfg.numCellsY / 2;
        s.jHighFinal   = cfg.numCellsY;
        s.degreeIndex1 = -1;
        s.degreeIndex2 = 2 * cfg.baseUnit;
        for (int i = 0; i < s.degreeIndex2;  ++i)  { s.jLow[i] = 0;              s.jHigh[i] = s.jHighFinal; }
        for (int i = s.degreeIndex2; i <= cfg.numCellsX; ++i) { s.jLow[i] = s.jLowInitial; s.jHigh[i] = s.jHighFinal; }
        for (int j = 0; j <= cfg.numCellsY; ++j)   { s.iLow[j] = 0; s.iHigh[j] = cfg.numCellsX; }
        break;

    case GeometryShape::SharpCorner:
        s.degreeIndex1 = 0;
        s.degreeIndex2 = cfg.baseUnit;
        s.degreeIndexY = cfg.numCellsY - cfg.baseUnit;
        for (int i = 0; i < s.degreeIndex2;  ++i)  { s.jLow[i] = 0; s.jHigh[i] = cfg.numCellsY; }
        for (int i = s.degreeIndex2; i <= cfg.numCellsX; ++i) { s.jLow[i] = s.degreeIndexY; s.jHigh[i] = cfg.numCellsY; }
        for (int j = 0; j <= cfg.numCellsY; ++j)   s.iLow[j] = 0;
        for (int j = 0; j <= s.degreeIndexY; ++j)  s.iHigh[j] = s.degreeIndex2;
        for (int j = s.degreeIndexY+1; j <= cfg.numCellsY; ++j) s.iHigh[j] = cfg.numCellsX;
        break;

    case GeometryShape::RoundedCorner: {
        s.degreeIndex1 = 0;
        s.degreeIndex2 = cfg.baseUnit;
        s.degreeIndexY = cfg.numCellsY - cfg.baseUnit;
        const int jRmp1loc = cfg.numCellsY - 5*cfg.baseUnit/4;
        s.rampIndexX1  = cfg.numCellsY - jRmp1loc;
        s.rampIndexY1  = jRmp1loc;
        int jRmp2, iRmp2;
        if (s.rampIndexY1 < cfg.numCellsY - 6*cfg.baseUnit/10) {
            jRmp2 = s.rampIndexY1 - 4*cfg.baseUnit/10;
            iRmp2 = s.degreeIndex2 + s.degreeIndexY - jRmp2;
        } else { jRmp2 = -2; iRmp2 = cfg.numCellsX + 2; }
        s.rampIndexY2 = jRmp2;
        s.rampIndexX2 = iRmp2;
        for (int i = 0; i < s.degreeIndex2; ++i) s.jLow[i] = 0;
        for (int i = s.degreeIndex2; i <= iRmp2; ++i) s.jLow[i] = jRmp2 + i - s.degreeIndex2;
        for (int i = iRmp2; i <= cfg.numCellsX; ++i) s.jLow[i] = s.degreeIndexY;
        for (int i = 0; i <= s.rampIndexX1; ++i) s.jHigh[i] = s.rampIndexY1 + i;
        for (int i = s.rampIndexX1+1; i <= cfg.numCellsX; ++i) s.jHigh[i] = cfg.numCellsY;
        for (int j = 0; j <= s.rampIndexY1; ++j) s.iLow[j] = 0;
        for (int j = s.rampIndexY1+1; j <= cfg.numCellsY; ++j) s.iLow[j] = j - s.rampIndexY1;
        for (int j = 0; j <= jRmp2; ++j) s.iHigh[j] = s.degreeIndex2;
        for (int j = jRmp2+1; j <= s.degreeIndexY; ++j) s.iHigh[j] = s.degreeIndex2 + j - jRmp2;
        for (int j = s.degreeIndexY+1; j <= cfg.numCellsY; ++j) s.iHigh[j] = cfg.numCellsX;
        break;
    }

    default:
        // Straight channel fallback — full domain active
        for (int i = 0; i <= cfg.numCellsX; ++i) { s.jLow[i] = 0; s.jHigh[i] = cfg.numCellsY; }
        for (int j = 0; j <= cfg.numCellsY; ++j) { s.iLow[j] = 0; s.iHigh[j] = cfg.numCellsX; }
        s.degreeIndex1 = cfg.numCellsX + 1;
        s.degreeIndex2 = cfg.numCellsX + 2;
        break;
    }

    // ── Initial velocity field ───────────────────────────────────────────────
    double velMeanInit = 0.0;
    if (cfg.initialProfile == InitialProfile::InletProfile) {
        developedProfileX(s, 0, s.initialPressureGradX, velMeanInit);
        for (int i = 1; i <= cfg.numCellsX; ++i)
            for (int j = s.jLow[0]; j <= s.jHigh[0]; ++j)
                for (int k = 0; k <= cfg.numCellsZ; ++k) {
                    s.velX(i, j, k) = s.velX(0, j, k);
                    s.velY(i, j, k) = 0.0;
                }
        s.uMaxAtInlet = 1.5;
        s.vMaxAtInlet = 0.15;
    } else {
        // Potential flow initialisation
        if (cfg.geometryType == GeometryType::Axial) {
            for (int i = 1; i <= s.numCellsXm1; ++i)
                for (int k = 0; k <= cfg.numCellsZ; ++k) {
                    s.velX(i, s.jLow[i],  k) = 0.0;  s.velY(i, s.jLow[i],  k) = 0.0;
                    s.velX(i, s.jHigh[i], k) = 0.0;  s.velY(i, s.jHigh[i], k) = 0.0;
                }
            developedProfileX(s, 0,             s.initialPressureGradX, velMeanInit);
            developedProfileX(s, cfg.numCellsX, s.initialPressureGradX, velMeanInit);
        } else {
            for (int j = 0; j <= cfg.numCellsY; ++j) {
                s.velX(s.iLow[j],  j, 0) = 0.0;
                s.velY(s.iLow[j],  j, 0) = 0.0;
                s.velY(s.iHigh[j], j, 0) = 0.0;
            }
            developedProfileY(s, 0,             s.initialPressureGradY, velMeanInit);
            developedProfileX(s, cfg.numCellsX, s.initialPressureGradX, velMeanInit);
        }
        s.uMaxAtInlet = velMeanInit * 1.5;
        s.vMaxAtInlet = velMeanInit * 0.15;
    }

    buildInitialPressure(s);

    // ── Adaptive initial time step (CFL) ────────────────────────────────────
    const double reForDiff = (cfg.hyperViscousStart == 0) ? cfg.reynoldsNumber : cfg.hyperViscousRe;
    const double dtViscous  = 0.5 * reForDiff
                            / (1.0/s.cellSizeXsq + 1.0/s.cellSizeYsq + 1.0/s.cellSizeZsq);
    const double dtAdvective = std::min(cfg.cellSizeX / (s.uMaxAtInlet + 1e-30),
                                        cfg.cellSizeY / (s.vMaxAtInlet + 1e-30));
    s.timeStepSize = 0.35 * std::min(dtViscous, dtAdvective);

    // ── Count active fluid cells (for RMS norms) ────────────────────────────
    s.numActiveCells = 0;
    for (int i = 1; i <= cfg.numCellsX; ++i)
        s.numActiveCells += s.jHigh[i] - s.jLow[i];
    s.numActiveCells *= cfg.numCellsZ;

    LOG_INFO("Initialisation complete.  Active cells: ", s.numActiveCells,
             "  dt0=", s.timeStepSize);
}

// ---------------------------------------------------------------------------
//  computeAccelerations — UNIFAES advective + viscous accelerations
// ---------------------------------------------------------------------------
void computeAccelerations(SimState& s)
{
    const auto& cfg = s.cfg;
    const bool periodic = (cfg.lateralCondition == LateralBC::Periodic);
    const int KKfim = periodic ? cfg.numCellsZ : s.numCellsZm1;

    s.accelX.fill(0.0);
    s.accelY.fill(0.0);
    s.accelZ.fill(0.0);

    for (int i = 1; i <= s.numCellsXm1; ++i) {
        const double localRe = 1.0 / effectiveInvRe(s, i);

        for (int j1 = s.jLow[i]; j1 <= s.jHigh[i]; ++j1) {
            const int j2 = j1;
            // ── x-direction UNIFAES ──────────────────────────────────────────
            {
                const int im = i - 1, ip = i + 1;
                double uFace = 0.5 * (s.velX(i, j1, 1) + s.velX(im, j1, 1));
                double DPe   = localRe * uFace * cfg.cellSizeX;
                double pip, cE, cW;
                computeExponentialWeights(localRe, DPe, pip, cE, cW);

                // Accumulate x-accelerations for nodes (i-1,j1) and (i,j1)
                for (int k = 1; k <= KKfim; ++k) {
                    uFace  = 0.5 * (s.velX(i, j1, k) + s.velX(im, j1, k));
                    DPe    = localRe * uFace * cfg.cellSizeX;
                    computeExponentialWeights(localRe, DPe, pip, cE, cW);

                    const double qsi = computeQsi(DPe, pip, 0.5);
                    // Upwind-biased flux from x-faces
                    s.accelX(im, j1, k) -= (cE * (s.velX(i,  j1, k) - s.velX(im, j1, k)));
                    s.accelX(i,  j1, k) += (cW * (s.velX(ip, j1, k) - s.velX(i,  j1, k)));
                    s.accelY(im, j1, k) -= (cE * (s.velY(i,  j1, k) - s.velY(im, j1, k)));
                    s.accelY(i,  j1, k) += (cW * (s.velY(ip, j1, k) - s.velY(i,  j1, k)));
                    s.accelZ(im, j1, k) -= (cE * (s.velZ(i,  j1, k) - s.velZ(im, j1, k)));
                    s.accelZ(i,  j1, k) += (cW * (s.velZ(ip, j1, k) - s.velZ(i,  j1, k)));
                    (void)qsi;  // cross-term: included in y/z sweeps below
                }
            }

            // ── y-direction UNIFAES ──────────────────────────────────────────
            {
                const int jm2 = j2 - 1, jp2 = j2 + 1;
                for (int k = 1; k <= KKfim; ++k) {
                    const double vFace = 0.5 * (s.velY(i, j2, k) + s.velY(i, jm2, k));
                    const double DPe   = localRe * vFace * cfg.cellSizeY;
                    double pip, cE, cW;
                    computeExponentialWeights(localRe, DPe, pip, cE, cW);
                    s.accelX(i, jm2, k) -= cE * (s.velX(i, j2,  k) - s.velX(i, jm2, k));
                    s.accelX(i, j2,  k) += cW * (s.velX(i, jp2, k) - s.velX(i, j2,  k));
                    s.accelY(i, jm2, k) -= cE * (s.velY(i, j2,  k) - s.velY(i, jm2, k));
                    s.accelY(i, j2,  k) += cW * (s.velY(i, jp2, k) - s.velY(i, j2,  k));
                    s.accelZ(i, jm2, k) -= cE * (s.velZ(i, j2,  k) - s.velZ(i, jm2, k));
                    s.accelZ(i, j2,  k) += cW * (s.velZ(i, jp2, k) - s.velZ(i, j2,  k));
                }
            }

            // ── z-direction UNIFAES ──────────────────────────────────────────
            for (int k = (periodic ? 1 : 1); k <= KKfim; ++k) {
                const int km = (periodic && k == 1) ? cfg.numCellsZ : k - 1;
                const double wFace = 0.5 * (s.velZ(i, j2, k) + s.velZ(i, j2, km));
                const double DPe   = localRe * wFace * cfg.cellSizeZ;
                double pip, cE, cW;
                computeExponentialWeights(localRe, DPe, pip, cE, cW);
                const int kp = (periodic && k == cfg.numCellsZ) ? 1 : k + 1;
                s.accelX(i, j2, km) -= cE * (s.velX(i, j2, k)  - s.velX(i, j2, km));
                s.accelX(i, j2, k)  += cW * (s.velX(i, j2, kp) - s.velX(i, j2, k));
                s.accelY(i, j2, km) -= cE * (s.velY(i, j2, k)  - s.velY(i, j2, km));
                s.accelY(i, j2, k)  += cW * (s.velY(i, j2, kp) - s.velY(i, j2, k));
                s.accelZ(i, j2, km) -= cE * (s.velZ(i, j2, k)  - s.velZ(i, j2, km));
                s.accelZ(i, j2, k)  += cW * (s.velZ(i, j2, kp) - s.velZ(i, j2, k));
            }
        }
    }

    // Periodic BC in z: copy KK → 0
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

    for (int sweep = 0; sweep < cfg.numPressureIter; ++sweep) {
        for (int i = 1; i <= cfg.numCellsX; ++i) {
            const int im = i-1, ip = i+1;
            int jLoopS, jLoopN;
            if      (s.jLow[im]  >= s.jLow[i])  jLoopS = s.jLow[i]+1;
            else                                  jLoopS = s.jLow[i];
            if      (s.jHigh[im] <= s.jHigh[i]) jLoopN = s.jHigh[i];
            else                                  jLoopN = s.jHigh[i]+1;
            if (i == s.degreeIndex2) { jLoopS = s.jLow[im]+1; jLoopN = s.jHigh[im]; }

            for (int k = 1; k <= cfg.numCellsZ; ++k) {
                int km = k-1, kp = k+1;
                for (int j = jLoopS; j <= jLoopN; ++j) {
                    const int jm = j-1, jp = j+1;

                    // Neumann ghost-cell mirroring
                    if (i == 1 || i == s.iLow[j]+1)             s.press(im, j, k) = s.press(i, j, k);
                    if (i == cfg.numCellsX || i == s.iHigh[j])  s.press(ip, j, k) = s.press(i, j, k);
                    if (j == jLoopS)                             s.press(i, jm, k) = s.press(i, j, k);
                    if (j == jLoopN)                             s.press(i, jp, k) = s.press(i, j, k);
                    if (cfg.lateralCondition == LateralBC::SolidWall) {
                        if (k == 1)            s.press(i, j, km) = s.press(i, j, k);
                        if (k == cfg.numCellsZ) s.press(i, j, kp) = s.press(i, j, k);
                    } else {
                        if (k == 1)            km = cfg.numCellsZ;
                        if (k == cfg.numCellsZ) kp = 1;
                    }

                    if (i == iRef && j == jRef && k == kRef) {
                        s.press(i, j, k) = pRef;
                    } else {
                        double pNew = (cY*(s.press(i,jp,k) + s.press(i,jm,k))
                                    + cX*(s.press(ip,j,k) + s.press(im,j,k))
                                    + cZ*(s.press(i,j,kp) + s.press(i,j,km))
                                    - s.pressureSource(i,j,k)) * invDiag;
                        // Corner correction
                        if ((i==1 || i==cfg.numCellsX) && (j==s.jLow[i]+1 || j==s.jHigh[i])) {
                            pNew -= s.pressureSource(i,j,k) * invDiag;
                            if (cfg.lateralCondition == LateralBC::SolidWall && (k==1 || k==cfg.numCellsZ))
                                pNew -= 2.0 * s.pressureSource(i,j,k) * invDiag;
                        }
                        s.press(i, j, k) = pNew;
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
