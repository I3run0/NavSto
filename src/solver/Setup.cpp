// =============================================================================
//  Setup.cpp — geometry construction and initial conditions.
//
//  Shared by every backend, including CUDA (host-only, runs once before time
//  marching). Setup the backends must agree on for their numbers to be
//  comparable; the per-step kernels are what differ.
// =============================================================================

#include "Physics.hpp"
#include "Logger.hpp"
#include "ViscosityModel.hpp"

#include <cmath>
#include <algorithm>
#include <stdexcept>

// ─── Internal helpers ────────────────────────────────────────────────────────

namespace {

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
    meanVelocity = 1.0 / spanWidth;   // matches PerfilDesenvolvidoHoriz in NavSto_dynamic

    for (int span = 0; span <= numISpan; ++span) {
        const int    i     = span + s.iLow[planeJ];
        const double xNorm = static_cast<double>(span) / numISpan;
        const double vVal  = -6.0 * (xNorm - 1.0) * xNorm * meanVelocity;  // -6 not -4
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
    // SharpCorner/RoundedCorner derive several index bounds directly from
    // baseUnit (e.g. degreeIndex2 = baseUnit, rampIndexX1 ~= 1.25*baseUnit);
    // if baseUnit isn't kept well below the grid size, those bounds exceed
    // jLow/jHigh's allocated range (numCellsX+1) and corrupt heap memory.
    // The original Pascal convention keeps baseUnit == numCellsY/2, so
    // requiring baseUnit <= numCellsY/2 and <= numCellsX/2 stays safely
    // inside that convention (verified against every checked-in .cfg).
    if (cfg.geometryShape == GeometryShape::SharpCorner ||
        cfg.geometryShape == GeometryShape::RoundedCorner) {
        if (cfg.baseUnit < 1 || cfg.baseUnit > cfg.numCellsX / 2 ||
            cfg.baseUnit > cfg.numCellsY / 2) {
            throw std::invalid_argument(
                "baseUnit=" + std::to_string(cfg.baseUnit) +
                " is incompatible with grid " + std::to_string(cfg.numCellsX) +
                "x" + std::to_string(cfg.numCellsY) + " for this geometryShape "
                "(need 1 <= baseUnit <= min(numCellsX, numCellsY)/2)");
        }
    }

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

    case GeometryShape::Straight:
    default:
        // Straight channel — full domain active at every x (no expansion,
        // contraction, or corner). Used for analytical validation (plane
        // Poiseuille flow has a closed-form solution only when the channel
        // is straight end-to-end) and as the fallback for shapes not yet
        // implemented (OpenCavity, GradualExpansion, etc).
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
        s.vMaxAtInlet = velMeanInit * 0.1 * 1.5;   // vmax = 0.1*umax (matches NavSto_dynamic)
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
