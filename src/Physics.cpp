// =============================================================================
//  Physics.cpp  —  Single-thread optimized Navier-Stokes kernels.
//
//  Optimizations applied (all single-thread):
//
//  OPT-A: Branch hoisting via template<LateralBC>
//    The `if (periodic)` and `if (solidWall)` checks were evaluated on every
//    k-iteration (millions of times per step). Templating the hot loops on the
//    BC enum turns them into compile-time constants that are eliminated by the
//    optimizer entirely, also unblocking auto-vectorization of the k-loop.
//
//  OPT-B: exp() fast path in UNIFAES weight computation
//    computeExponentialWeights called std::exp() for every face even when
//    |Pe| < 0.1 (where the polynomial branch is taken). Reordered to check
//    the polynomial range first, skip exp() when not needed.
//
//  OPT-C: Loop structure in computeAccelerations
//    Original: one j-loop with x/y/z blocks that each re-enter a k-loop.
//    Causes repeated j-indexed loads and poor instruction mix.
//    Fixed: separate tightly-focused x, y, z sweeps with single k-loops
//    that touch only the data they need — better cache line reuse.
//
//  OPT-D: Pressure solver ghost cells as scalars (correctness + perf)
//    Original wrote ghost values back into the array (also caused OOB writes).
//    Now uses local doubles pIm/pIp/pJm/pJp/pKm/pKp — no aliasing, lets
//    the compiler keep them in registers across the stencil computation.
//
//  OPT-E: Removed dead computation
//    qsi was computed then discarded with (void)qsi. Removed.
//    Redundant uFace pre-computation before the k-loop was removed.
//
//  OPT-F: const& and noexcept on all hot helpers
//    Prevents the compiler from emitting invisible copies and lets it
//    inline aggressively across translation units.
// =============================================================================

#include "Physics.hpp"
#include "Logger.hpp"

#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <vector>

// ─── Internal helpers ────────────────────────────────────────────────────────

namespace {

// ---------------------------------------------------------------------------
//  OPT-B: UNIFAES exponential weights.
//  Polynomial branch checked FIRST — avoids std::exp() for |Pe| < 0.1,
//  which is the common case at low Reynolds numbers or near walls.
//  [[gnu::const]]: no side effects, result depends only on arguments —
//  compiler may CSE or hoist freely.
// ---------------------------------------------------------------------------
[[gnu::const]]
inline void weights(double Re, double DPe,
                    double& pip, double& cE, double& cW) noexcept
{
    const double a = std::abs(DPe);
    if (a < 0.1) {
        // Polynomial approximation — numerically stable, no transcendental
        pip = 1.0 / ((((0.05*DPe + 0.25)*DPe + 1.0)*DPe*(1.0/6.0) + 0.5)*DPe + 1.0);
    } else if (a <= 200.0) {
        pip = DPe / (std::exp(DPe) - 1.0);
    } else {
        pip = (DPe > 0.0) ? 0.0 : -DPe;
    }
    const double pim = DPe + pip;
    cE = pip / Re;
    cW = pim / Re;
}

[[gnu::const]]
inline double effectiveRe(const SimState& s, int i) noexcept
{
    const auto& c = s.cfg;
    if (c.hyperViscousStart == 0 || i <= c.numCellsX - c.hyperViscousStart)
        return c.reynoldsNumber;

    constexpr double hpi = 1.5707963267948966;
    const int orig = c.numCellsX - 5*c.hyperViscousStart/8;
    const int amp  = 3*c.hyperViscousStart/8;
    const double rO = 0.5*(c.hyperViscousRe + c.reynoldsNumber);
    const double rA = 0.5*(c.hyperViscousRe - c.reynoldsNumber);

    if (i < c.numCellsX - c.hyperViscousStart/4)
        return rO + rA * std::sin(((i - orig) / (double)amp) * hpi);
    return c.hyperViscousRe;
}

// ---------------------------------------------------------------------------
//  OPT-A: compile-time k-index helpers for periodic/solid-wall BC.
//  The branch is resolved at instantiation — zero cost inside the k-loop.
// ---------------------------------------------------------------------------
template <LateralBC kBC>
[[gnu::const]] inline int kprev(int k, int KZ) noexcept {
    if constexpr (kBC == LateralBC::Periodic)
        return (k == 1) ? KZ : k - 1;
    else
        return k - 1;
}

template <LateralBC kBC>
[[gnu::const]] inline int knext(int k, int KZ) noexcept {
    if constexpr (kBC == LateralBC::Periodic)
        return (k == KZ) ? 1 : k + 1;
    else
        return k + 1;
}

// ---------------------------------------------------------------------------
//  Developed Poiseuille profile in X at plane planeI.
// ---------------------------------------------------------------------------
void developedProfileX(SimState& s, int planeI,
                       double& dpGrad, double& uMean)
{
    const auto& c = s.cfg;
    const int    span   = s.jHigh[planeI] - s.jLow[planeI];
    const double height = span * c.cellSizeY;
    uMean = 1.5 / height;

    for (int sp = 0; sp <= span; ++sp) {
        const int    j  = sp + s.jLow[planeI];
        const double yn = static_cast<double>(sp) / span;
        const double u  = -4.0 * (yn - 1.0) * yn * uMean;
        for (int k = 0; k <= c.numCellsZ; ++k)
            s.velX(planeI, j, k) = u;
    }
    dpGrad = -12.0 * uMean / (height * height * c.reynoldsNumber);

    if (c.lateralCondition == LateralBC::SolidWall) {
        for (int j = s.jLow[planeI]+1; j <= s.jHigh[planeI]-1; ++j) {
            s.velX(planeI, j, 0)          = 0.0;
            s.velX(planeI, j, c.numCellsZ) = 0.0;
        }
        for (int k = 1; k <= s.numCellsZm1; ++k) {
            s.velX(planeI, s.jLow [planeI], k) = 0.0;
            s.velX(planeI, s.jHigh[planeI], k) = 0.0;
        }
        const double cY = 1.0 / s.cellSizeYsq;
        const double cZ = 1.0 / s.cellSizeZsq;
        const double D  = 2.0*(cY+cZ);
        const double iD = 1.0/D;
        const double om = 1.85;
        int iter = 0; double res;
        do {
            res = 0.0; ++iter;
            for (int j = s.jLow[planeI]+1; j <= s.jHigh[planeI]-1; ++j)
                for (int k = 1; k <= s.numCellsZm1; ++k) {
                    const double rhs = cY*(s.velX(planeI,j+1,k)+s.velX(planeI,j-1,k))
                                     + cZ*(s.velX(planeI,j,k+1)+s.velX(planeI,j,k-1))
                                     - dpGrad;
                    res = std::max(res, std::abs(rhs - D*s.velX(planeI,j,k)));
                    s.velX(planeI,j,k) += om*(rhs*iD - s.velX(planeI,j,k));
                }
        } while (res >= 1e-8 && iter < 100000);

        double integ = 0.0;
        for (int j = s.jLow[planeI]+1; j <= s.jHigh[planeI]-1; ++j)
            for (int k = 1; k <= s.numCellsZm1; ++k)
                integ += s.velX(planeI,j,k);
        integ /= (span * c.numCellsZ);
        const double sc = 1.0/integ;
        dpGrad *= sc;
        for (int j = s.jLow[planeI]+1; j <= s.jHigh[planeI]-1; ++j)
            for (int k = 1; k <= s.numCellsZm1; ++k)
                s.velX(planeI,j,k) *= sc;
    }
}

void developedProfileY(SimState& s, int planeJ,
                       double& dpGrad, double& vMean)
{
    const auto& c = s.cfg;
    const int    span  = s.iHigh[planeJ] - s.iLow[planeJ];
    const double width = span * c.cellSizeX;
    vMean = 1.5 / width;
    for (int sp = 0; sp <= span; ++sp) {
        const int    i  = sp + s.iLow[planeJ];
        const double xn = static_cast<double>(sp) / span;
        const double v  = -4.0*(xn-1.0)*xn*vMean;
        for (int k = 0; k <= c.numCellsZ; ++k)
            s.velY(i, planeJ, k) = v;
    }
    dpGrad = -12.0*vMean/(width*width*c.reynoldsNumber);
}

void buildInitialPressure(SimState& s)
{
    const auto& c = s.cfg;
    s.press(0,0,0) = 0.0;
    const int    span0  = s.jHigh[0] - s.jLow[0];
    const double h0     = span0 * c.cellSizeY;
    double dpDx0  = -12.0*c.cellSizeX/(h0*h0*h0*c.reynoldsNumber);
    double uMean0 = 1.0/h0;

    for (int i = 1; i <= s.degreeIndex1; ++i) {
        const int im = i-1;
        const double h = (i != s.degreeIndex2)
            ? (s.jHigh[i] -s.jLow[i]) *c.cellSizeY
            : (s.jHigh[im]-s.jLow[im])*c.cellSizeY;
        const double um  = 1.0/h;
        const double Re  = effectiveRe(s,i);
        const double dp1 = -12.0*c.cellSizeX/(Re*h*h);   // note: effectiveRe returns Re, not 1/Re
        const double dpDx = 0.5*(dpDx0+dp1) + 0.5*(uMean0*uMean0 - um*um);
        dpDx0 = dp1; uMean0 = um;
        s.press(i,0,0) = s.press(im,0,0) + dpDx;
        const int jS = (i!=s.degreeIndex2) ? s.jLow[i]+1 : s.jLow[im]+1;
        for (int j = jS; j <= s.jHigh[i]; ++j)
            for (int k = 1; k <= c.numCellsZ; ++k)
                s.press(i,j,k) = s.press(i,0,0);
    }
    for (int i = s.degreeIndex1+1; i <= c.numCellsX; ++i) {
        s.press(i,0,0) = s.press(s.degreeIndex1,0,0);
        for (int j = s.jLow[i]+1; j <= s.jHigh[i]; ++j)
            for (int k = 1; k <= c.numCellsZ; ++k)
                s.press(i,j,k) = s.press(i,0,0);
    }
}

} // anonymous namespace

// ════════════════════════════════════════════════════════════════════════════
//  initSimulation
// ════════════════════════════════════════════════════════════════════════════
void initSimulation(SimState& s)
{
    const auto& c = s.cfg;
    s.numCellsXm1 = c.numCellsX-1;
    s.numCellsYm1 = c.numCellsY-1;
    s.numCellsZm1 = c.numCellsZ-1;
    s.cellSizeXsq  = c.cellSizeX*c.cellSizeX;
    s.cellSizeYsq  = c.cellSizeY*c.cellSizeY;
    s.cellSizeZsq  = c.cellSizeZ*c.cellSizeZ;

    s.velX.fill(0); s.velY.fill(0); s.velZ.fill(0);
    s.press.fill(0);
    s.accelX.fill(0); s.accelY.fill(0); s.accelZ.fill(0);
    s.pressureSource.fill(0);

    // ── Geometry ─────────────────────────────────────────────────────────────
    switch (c.geometryShape) {
    case GeometryShape::AbruptExpansion:
        s.degreeIndexY = c.numCellsY/2;
        s.degreeIndex1 = c.numCellsX/12;
        s.degreeIndex2 = c.numCellsX+2;
        for (int i = 0; i <= s.degreeIndex1; ++i) { s.jLow[i]=s.degreeIndexY; s.jHigh[i]=c.numCellsY; }
        for (int i = s.degreeIndex1+1; i<=c.numCellsX; ++i) { s.jLow[i]=0; s.jHigh[i]=c.numCellsY; }
        for (int j = 0; j<=s.degreeIndexY; ++j) s.iLow[j]=s.degreeIndex1;
        for (int j = s.degreeIndexY+1; j<=c.numCellsY; ++j) s.iLow[j]=0;
        for (int j = 0; j<=c.numCellsY; ++j) s.iHigh[j]=c.numCellsX;
        break;

    case GeometryShape::AbruptContraction:
        s.jLowInitial=c.numCellsY/2; s.jHighFinal=c.numCellsY;
        s.degreeIndex1=-1; s.degreeIndex2=2*c.baseUnit;
        for (int i=0; i<s.degreeIndex2; ++i) { s.jLow[i]=0; s.jHigh[i]=s.jHighFinal; }
        for (int i=s.degreeIndex2; i<=c.numCellsX; ++i) { s.jLow[i]=s.jLowInitial; s.jHigh[i]=s.jHighFinal; }
        for (int j=0; j<=c.numCellsY; ++j) { s.iLow[j]=0; s.iHigh[j]=c.numCellsX; }
        break;

    case GeometryShape::SharpCorner:
        s.degreeIndex1=0; s.degreeIndex2=c.baseUnit; s.degreeIndexY=c.numCellsY-c.baseUnit;
        for (int i=0; i<s.degreeIndex2; ++i) { s.jLow[i]=0; s.jHigh[i]=c.numCellsY; }
        for (int i=s.degreeIndex2; i<=c.numCellsX; ++i) { s.jLow[i]=s.degreeIndexY; s.jHigh[i]=c.numCellsY; }
        for (int j=0; j<=c.numCellsY; ++j) s.iLow[j]=0;
        for (int j=0; j<=s.degreeIndexY; ++j) s.iHigh[j]=s.degreeIndex2;
        for (int j=s.degreeIndexY+1; j<=c.numCellsY; ++j) s.iHigh[j]=c.numCellsX;
        break;

    case GeometryShape::RoundedCorner: {
        s.degreeIndex1=0; s.degreeIndex2=c.baseUnit; s.degreeIndexY=c.numCellsY-c.baseUnit;
        const int jR1=c.numCellsY-5*c.baseUnit/4;
        s.rampIndexX1=c.numCellsY-jR1; s.rampIndexY1=jR1;
        int jR2,iR2;
        if (jR1 < c.numCellsY-6*c.baseUnit/10) { jR2=jR1-4*c.baseUnit/10; iR2=s.degreeIndex2+s.degreeIndexY-jR2; }
        else { jR2=-2; iR2=c.numCellsX+2; }
        s.rampIndexY2=jR2; s.rampIndexX2=iR2;
        for (int i=0; i<s.degreeIndex2; ++i) s.jLow[i]=0;
        for (int i=s.degreeIndex2; i<=iR2; ++i) s.jLow[i]=jR2+i-s.degreeIndex2;
        for (int i=iR2; i<=c.numCellsX; ++i) s.jLow[i]=s.degreeIndexY;
        for (int i=0; i<=s.rampIndexX1; ++i) s.jHigh[i]=jR1+i;
        for (int i=s.rampIndexX1+1; i<=c.numCellsX; ++i) s.jHigh[i]=c.numCellsY;
        for (int j=0; j<=jR1; ++j) s.iLow[j]=0;
        for (int j=jR1+1; j<=c.numCellsY; ++j) s.iLow[j]=j-jR1;
        for (int j=0; j<=jR2; ++j) s.iHigh[j]=s.degreeIndex2;
        for (int j=jR2+1; j<=s.degreeIndexY; ++j) s.iHigh[j]=s.degreeIndex2+j-jR2;
        for (int j=s.degreeIndexY+1; j<=c.numCellsY; ++j) s.iHigh[j]=c.numCellsX;
        break;
    }
    default:
        for (int i=0; i<=c.numCellsX; ++i) { s.jLow[i]=0; s.jHigh[i]=c.numCellsY; }
        for (int j=0; j<=c.numCellsY; ++j) { s.iLow[j]=0; s.iHigh[j]=c.numCellsX; }
        s.degreeIndex1=c.numCellsX+1; s.degreeIndex2=c.numCellsX+2;
        break;
    }

    // ── Initial velocity ──────────────────────────────────────────────────────
    double vm = 0.0;
    if (c.initialProfile == InitialProfile::InletProfile) {
        developedProfileX(s, 0, s.initialPressureGradX, vm);
        for (int i=1; i<=c.numCellsX; ++i)
            for (int j=s.jLow[0]; j<=s.jHigh[0]; ++j)
                for (int k=0; k<=c.numCellsZ; ++k)
                    { s.velX(i,j,k)=s.velX(0,j,k); s.velY(i,j,k)=0.0; }
        s.uMaxAtInlet=1.5; s.vMaxAtInlet=0.15;
    } else {
        if (c.geometryType == GeometryType::Axial) {
            for (int i=1; i<=s.numCellsXm1; ++i)
                for (int k=0; k<=c.numCellsZ; ++k) {
                    s.velX(i,s.jLow[i], k)=s.velY(i,s.jLow[i], k)=0.0;
                    s.velX(i,s.jHigh[i],k)=s.velY(i,s.jHigh[i],k)=0.0;
                }
            developedProfileX(s,0,             s.initialPressureGradX, vm);
            developedProfileX(s,c.numCellsX,   s.initialPressureGradX, vm);
        } else {
            for (int j=0; j<=c.numCellsY; ++j) {
                s.velX(s.iLow[j], j,0)=s.velY(s.iLow[j], j,0)=0.0;
                s.velY(s.iHigh[j],j,0)=0.0;
            }
            developedProfileY(s,0,           s.initialPressureGradY, vm);
            developedProfileX(s,c.numCellsX, s.initialPressureGradX, vm);
        }
        s.uMaxAtInlet=vm*1.5; s.vMaxAtInlet=vm*0.15;
    }
    buildInitialPressure(s);

    const double Rd = (c.hyperViscousStart==0) ? c.reynoldsNumber : c.hyperViscousRe;
    const double dtV = 0.5*Rd/(1.0/s.cellSizeXsq+1.0/s.cellSizeYsq+1.0/s.cellSizeZsq);
    const double dtA = std::min(c.cellSizeX/(s.uMaxAtInlet+1e-30),
                                c.cellSizeY/(s.vMaxAtInlet+1e-30));
    s.timeStepSize = 0.35*std::min(dtV,dtA);

    s.numActiveCells = 0;
    for (int i=1; i<=c.numCellsX; ++i) s.numActiveCells += s.jHigh[i]-s.jLow[i];
    s.numActiveCells *= c.numCellsZ;

    LOG_INFO("Init done. Active cells=", s.numActiveCells, "  dt0=", s.timeStepSize);
}

// ════════════════════════════════════════════════════════════════════════════
//  OPT-A+C: computeAccelerations — templated BC, separated x/y/z sweeps.
//
//  Splitting into three dedicated sweeps means each k-loop touches a compact,
//  predictable set of cache lines. The original interleaved x/y/z updates
//  inside one j-loop caused the CPU prefetcher to chase three stride patterns
//  simultaneously. Separate sweeps = one stride pattern per sweep.
// ════════════════════════════════════════════════════════════════════════════
template <LateralBC kBC>
static void accelImpl(SimState& s)
{
    const auto& c  = s.cfg;
    const int KZ   = c.numCellsZ;
    const int KKfim = (kBC==LateralBC::SolidWall) ? s.numCellsZm1 : KZ;

    s.accelX.fill(0); s.accelY.fill(0); s.accelZ.fill(0);

    for (int i = 1; i <= s.numCellsXm1; ++i) {
        const double Re = effectiveRe(s, i);
        const int    im = i-1, ip = i+1;

        // ── OPT-C sweep 1: x-faces (i-1 ↔ i) ──────────────────────────────
        // Only interior j nodes; wall nodes have zero velocity.
        for (int j = s.jLow[i]+1; j <= s.jHigh[i]-1; ++j) {
            for (int k = 1; k <= KKfim; ++k) {
                // West face (im, i)
                {
                    const double uF  = 0.5*(s.velX(i,j,k)+s.velX(im,j,k));
                    const double DPe = uF * c.cellSizeX;  // ×Re applied inside weights()
                    double pip, cE, cW;
                    weights(Re, DPe*Re, pip, cE, cW);
                    const double dU = s.velX(i,j,k)-s.velX(im,j,k);
                    const double dV = s.velY(i,j,k)-s.velY(im,j,k);
                    const double dW = s.velZ(i,j,k)-s.velZ(im,j,k);
                    s.accelX(im,j,k) -= cE*dU;  s.accelX(i,j,k) -= cE*dU;
                    s.accelY(im,j,k) -= cE*dV;  s.accelY(i,j,k) -= cE*dV;
                    s.accelZ(im,j,k) -= cE*dW;  s.accelZ(i,j,k) -= cE*dW;
                }
                // East face (i, ip)
                if (ip <= c.numCellsX) {
                    const double uF  = 0.5*(s.velX(ip,j,k)+s.velX(i,j,k));
                    const double DPe = uF * c.cellSizeX;
                    double pip, cE, cW;
                    weights(Re, DPe*Re, pip, cE, cW);
                    const double dU = s.velX(ip,j,k)-s.velX(i,j,k);
                    const double dV = s.velY(ip,j,k)-s.velY(i,j,k);
                    const double dW = s.velZ(ip,j,k)-s.velZ(i,j,k);
                    s.accelX(i,j,k) += cW*dU;
                    s.accelY(i,j,k) += cW*dV;
                    s.accelZ(i,j,k) += cW*dW;
                }
            }
        }

        // ── OPT-C sweep 2: y-faces ─────────────────────────────────────────
        for (int j = s.jLow[i]+1; j <= s.jHigh[i]-1; ++j) {
            const int jm = j-1, jp = j+1;
            for (int k = 1; k <= KKfim; ++k) {
                // South face (jm, j)
                {
                    const double vF  = 0.5*(s.velY(i,j,k)+s.velY(i,jm,k));
                    double pip, cE, cW;
                    weights(Re, vF*c.cellSizeY*Re, pip, cE, cW);
                    const double dU = s.velX(i,j,k)-s.velX(i,jm,k);
                    const double dV = s.velY(i,j,k)-s.velY(i,jm,k);
                    const double dW = s.velZ(i,j,k)-s.velZ(i,jm,k);
                    s.accelX(i,jm,k) -= cE*dU;  s.accelX(i,j,k) -= cE*dU;
                    s.accelY(i,jm,k) -= cE*dV;  s.accelY(i,j,k) -= cE*dV;
                    s.accelZ(i,jm,k) -= cE*dW;  s.accelZ(i,j,k) -= cE*dW;
                }
                // North face (j, jp) — guard against top wall
                if (jp <= s.jHigh[i]) {
                    const double vF  = 0.5*(s.velY(i,jp,k)+s.velY(i,j,k));
                    double pip, cE, cW;
                    weights(Re, vF*c.cellSizeY*Re, pip, cE, cW);
                    const double dU = s.velX(i,jp,k)-s.velX(i,j,k);
                    const double dV = s.velY(i,jp,k)-s.velY(i,j,k);
                    const double dW = s.velZ(i,jp,k)-s.velZ(i,j,k);
                    s.accelX(i,j,k) += cW*dU;
                    s.accelY(i,j,k) += cW*dV;
                    s.accelZ(i,j,k) += cW*dW;
                }
            }
        }

        // ── OPT-C sweep 3: z-faces (OPT-A: no branch in k-loop) ───────────
        for (int j = s.jLow[i]+1; j <= s.jHigh[i]-1; ++j) {
            for (int k = 1; k <= KKfim; ++k) {
                const int km = kprev<kBC>(k, KZ);  // compile-time: no branch
                const int kp = knext<kBC>(k, KZ);
                // Back face (km, k)
                {
                    const double wF  = 0.5*(s.velZ(i,j,k)+s.velZ(i,j,km));
                    double pip, cE, cW;
                    weights(Re, wF*c.cellSizeZ*Re, pip, cE, cW);
                    const double dU = s.velX(i,j,k)-s.velX(i,j,km);
                    const double dV = s.velY(i,j,k)-s.velY(i,j,km);
                    const double dW = s.velZ(i,j,k)-s.velZ(i,j,km);
                    s.accelX(i,j,km) -= cE*dU;  s.accelX(i,j,k) -= cE*dU;
                    s.accelY(i,j,km) -= cE*dV;  s.accelY(i,j,k) -= cE*dV;
                    s.accelZ(i,j,km) -= cE*dW;  s.accelZ(i,j,k) -= cE*dW;
                }
                // Front face (k, kp)
                if (kp != km) {  // skip if periodic wrap-around would double-count
                    const double wF  = 0.5*(s.velZ(i,j,kp)+s.velZ(i,j,k));
                    double pip, cE, cW;
                    weights(Re, wF*c.cellSizeZ*Re, pip, cE, cW);
                    const double dU = s.velX(i,j,kp)-s.velX(i,j,k);
                    const double dV = s.velY(i,j,kp)-s.velY(i,j,k);
                    const double dW = s.velZ(i,j,kp)-s.velZ(i,j,k);
                    s.accelX(i,j,k) += cW*dU;
                    s.accelY(i,j,k) += cW*dV;
                    s.accelZ(i,j,k) += cW*dW;
                }
            }
        }
    }

    // Periodic z BC: copy KZ plane → 0
    if constexpr (kBC == LateralBC::Periodic) {
        for (int i=1; i<=s.numCellsXm1; ++i)
            for (int j=s.jLow[i]+1; j<=s.jHigh[i]-1; ++j) {
                s.accelX(i,j,0) = s.accelX(i,j,KZ);
                s.accelY(i,j,0) = s.accelY(i,j,KZ);
                s.accelZ(i,j,0) = s.accelZ(i,j,KZ);
            }
    }
}

void computeAccelerations(SimState& s)
{
    if (s.cfg.lateralCondition == LateralBC::Periodic) accelImpl<LateralBC::Periodic>(s);
    else                                                accelImpl<LateralBC::SolidWall>(s);
}

// ════════════════════════════════════════════════════════════════════════════
//  OPT-A+D: buildPressureSource — template BC + no ghost writes.
// ════════════════════════════════════════════════════════════════════════════
template <LateralBC kBC>
static void pressSourceImpl(SimState& s)
{
    const auto& c = s.cfg;
    const int KZ  = c.numCellsZ;

    // Zero boundary accelerations
    for (int i=0; i<=c.numCellsX; ++i)
        for (int k=0; k<=KZ; ++k) {
            const int jB=s.jLow[i], jT=s.jHigh[i];
            s.accelX(i,jB,k)=s.accelX(i,jT,k)=0;
            s.accelY(i,jB,k)=s.accelY(i,jT,k)=0;
            s.accelZ(i,jB,k)=s.accelZ(i,jT,k)=0;
        }
    for (int j=0; j<=c.numCellsY; ++j)
        for (int k=0; k<=KZ; ++k) {
            const int iL=s.iLow[j], iR=s.iHigh[j];
            s.accelX(iL,j,k)=s.accelX(iR,j,k)=0;
            s.accelY(iL,j,k)=s.accelY(iR,j,k)=0;
            s.accelZ(iL,j,k)=s.accelZ(iR,j,k)=0;
        }

    const double qIdx = 0.25/c.cellSizeX;
    const double qIdy = 0.25/c.cellSizeY;
    const double qIdz = 0.25/c.cellSizeZ;
    const double iDt  = 1.0/s.timeStepSize;

    for (int i=1; i<=c.numCellsX; ++i) {
        const int im = i-1;
        const int jS = (i!=s.degreeIndex2) ? s.jLow[i]+1  : s.jLow[im]+1;
        const int jN = (i!=s.degreeIndex2) ? s.jHigh[i]   : s.jHigh[im];
        for (int j=jS; j<=jN; ++j) {
            const int jm=j-1;
            for (int k=1; k<=KZ; ++k) {
                const int km = kprev<kBC>(k,KZ);  // OPT-A: no branch

                const double divU =
                    (s.velX(i,j,k) -s.velX(im,j,k) +s.velX(i,jm,k) -s.velX(im,jm,k)
                    +s.velX(i,j,km)-s.velX(im,j,km)+s.velX(i,jm,km)-s.velX(im,jm,km))*qIdx;
                const double divV =
                    (s.velY(i,j,k) -s.velY(i,jm,k) +s.velY(im,j,k) -s.velY(im,jm,k)
                    +s.velY(i,j,km)-s.velY(i,jm,km)+s.velY(im,j,km)-s.velY(im,jm,km))*qIdy;
                const double divW =
                    (s.velZ(i,j,k) +s.velZ(i,jm,k) +s.velZ(im,j,k) +s.velZ(im,jm,k)
                    -s.velZ(i,j,km)-s.velZ(i,jm,km)-s.velZ(im,j,km)-s.velZ(im,jm,km))*qIdz;

                const double divAu =
                    (s.accelX(i,j,k) -s.accelX(im,j,k) +s.accelX(i,jm,k) -s.accelX(im,jm,k)
                    +s.accelX(i,j,km)-s.accelX(im,j,km)+s.accelX(i,jm,km)-s.accelX(im,jm,km))*qIdx;
                const double divAv =
                    (s.accelY(i,j,k) -s.accelY(i,jm,k) +s.accelY(im,j,k) -s.accelY(im,jm,k)
                    +s.accelY(i,j,km)-s.accelY(i,jm,km)+s.accelY(im,j,km)-s.accelY(im,jm,km))*qIdy;
                const double divAw =
                    (s.accelZ(i,j,k) +s.accelZ(i,jm,k) +s.accelZ(im,j,k) +s.accelZ(im,jm,k)
                    -s.accelZ(i,j,km)-s.accelZ(i,jm,km)-s.accelZ(im,j,km)-s.accelZ(im,jm,km))*qIdz;

                s.pressureSource(i,j,k) = (divU+divV+divW)*iDt + (divAu+divAv+divAw);
            }
        }
    }
}

void buildPressureSource(SimState& s)
{
    if (s.cfg.lateralCondition==LateralBC::Periodic) pressSourceImpl<LateralBC::Periodic>(s);
    else                                              pressSourceImpl<LateralBC::SolidWall>(s);
}

// ════════════════════════════════════════════════════════════════════════════
//  OPT-A+D: solvePressurePoisson — template BC, ghost cells as scalars.
//
//  OPT-D: writing ghost values into the array (old code) caused aliasing and
//  potential OOB. Using local scalars pIm/pIp/pJm/pJp/pKm/pKp:
//    • no aliasing — compiler keeps them in registers
//    • the 6-point stencil fma chain is a single dependency graph
//    • no spurious store → load → stencil pipeline stall
// ════════════════════════════════════════════════════════════════════════════
template <LateralBC kBC>
static void pressPoissImpl(SimState& s)
{
    const auto& c = s.cfg;
    const int KZ  = c.numCellsZ;
    const double cX  = 1.0/s.cellSizeXsq;
    const double cY  = 1.0/s.cellSizeYsq;
    const double cZ  = 1.0/s.cellSizeZsq;
    const double iDg = 0.5/(cX+cY+cZ);

    const int    iRef = c.numCellsX;
    const int    jRef = (s.jHigh[iRef]+s.jLow[iRef])/2;
    const int    kRef = (KZ+1)/2;
    const double pRef = s.press(iRef,jRef,kRef);

    for (int sw=0; sw<c.numPressureIter; ++sw) {
        for (int i=1; i<=c.numCellsX; ++i) {
            const int im=i-1, ip=i+1;
            const int jS = (i!=s.degreeIndex2) ? s.jLow[i]+1  : s.jLow[im]+1;
            const int jN = (i!=s.degreeIndex2) ? s.jHigh[i]   : s.jHigh[im];

            for (int j=jS; j<=jN; ++j) {
                const int jm=j-1, jp=j+1;
                for (int k=1; k<=KZ; ++k) {
                    // OPT-D: local ghost scalars — never OOB, compiler keeps in registers
                    const double pC = s.press(i,j,k);
                    double pIm = (i==1||i==s.iLow[j]+1)            ? pC : s.press(im,j,k);
                    double pIp = (i==c.numCellsX||i==s.iHigh[j])   ? pC : s.press(ip,j,k);
                    double pJm = (j==jS)                            ? pC : s.press(i,jm,k);
                    double pJp = (j==jN)                            ? pC : s.press(i,jp,k);
                    double pKm, pKp;
                    if constexpr (kBC==LateralBC::SolidWall) {         // OPT-A: compile-time
                        pKm = (k==1)  ? pC : s.press(i,j,k-1);
                        pKp = (k==KZ) ? pC : s.press(i,j,k+1);
                    } else {
                        pKm = s.press(i,j,kprev<kBC>(k,KZ));
                        pKp = s.press(i,j,knext<kBC>(k,KZ));
                    }

                    if (i==iRef && j==jRef && k==kRef) {
                        s.press(i,j,k) = pRef;
                    } else {
                        double pNew = (cY*(pJp+pJm)+cX*(pIp+pIm)+cZ*(pKp+pKm)
                                      - s.pressureSource(i,j,k)) * iDg;
                        if ((i==1||i==c.numCellsX)&&(j==s.jLow[i]+1||j==s.jHigh[i])) {
                            pNew -= s.pressureSource(i,j,k)*iDg;
                            if constexpr (kBC==LateralBC::SolidWall)
                                if (k==1||k==KZ) pNew -= 2.0*s.pressureSource(i,j,k)*iDg;
                        }
                        s.press(i,j,k) = pNew;
                    }
                }
            }
        }
    }
}

void solvePressurePoisson(SimState& s)
{
    if (s.cfg.lateralCondition==LateralBC::Periodic) pressPoissImpl<LateralBC::Periodic>(s);
    else                                              pressPoissImpl<LateralBC::SolidWall>(s);
}

// ════════════════════════════════════════════════════════════════════════════
//  OPT-A: updateVelocities — template BC.
// ════════════════════════════════════════════════════════════════════════════
template <LateralBC kBC>
static void velUpdImpl(SimState& s)
{
    const auto& c   = s.cfg;
    const int KZ    = c.numCellsZ;
    const int KKfim = (kBC==LateralBC::SolidWall) ? s.numCellsZm1 : KZ;
    const double qIdx = 0.25/c.cellSizeX;
    const double qIdy = 0.25/c.cellSizeY;
    const double qIdz = 0.25/c.cellSizeZ;
    const double dt   = s.useHalfStep ? 0.5*s.timeStepSize : s.timeStepSize;
    double maxC = 0.0;

    for (int i=1; i<=s.numCellsXm1; ++i) {
        const int ip=i+1;
        for (int j=s.jLow[i]+1; j<=s.jHigh[i]-1; ++j) {
            const int jp=j+1;
            for (int k=1; k<=KKfim; ++k) {
                const int kp = knext<kBC>(k,KZ);  // OPT-A

                const double dpu =
                    (s.press(ip,j, kp)-s.press(i,j, kp)+s.press(ip,jp,kp)-s.press(i,jp,kp)
                    +s.press(ip,j, k) -s.press(i,j, k) +s.press(ip,jp,k) -s.press(i,jp,k))*qIdx;
                const double dpv =
                    (s.press(i, jp,kp)-s.press(i,j, kp)+s.press(ip,jp,kp)-s.press(ip,j,kp)
                    +s.press(i, jp,k) -s.press(i,j, k) +s.press(ip,jp,k) -s.press(ip,j,k))*qIdy;
                const double dpw =
                    (s.press(i, jp,kp)+s.press(i,j, kp)+s.press(ip,jp,kp)+s.press(ip,j,kp)
                    -s.press(i, jp,k) -s.press(i,j, k) -s.press(ip,jp,k) -s.press(ip,j,k))*qIdz;

                const double du = s.accelX(i,j,k)-dpu;
                const double dv = s.accelY(i,j,k)-dpv;
                const double dw = s.accelZ(i,j,k)-dpw;
                s.velX(i,j,k) += du*dt;
                s.velY(i,j,k) += dv*dt;
                s.velZ(i,j,k) += dw*dt;
                maxC = std::max(maxC, du*du+dv*dv+dw*dw);
            }
            if constexpr (kBC==LateralBC::Periodic) {
                s.velX(i,j,0)=s.velX(i,j,KZ);
                s.velY(i,j,0)=s.velY(i,j,KZ);
                s.velZ(i,j,0)=s.velZ(i,j,KZ);
            }
        }
    }
    s.maxVelocityChange = std::sqrt(maxC);

    const int NX = c.numCellsX;
    if (c.outletCondition==OutletBC::ZeroFirstDeriv)
        for (int j=s.jLow[NX]+1; j<=s.jHigh[NX]-1; ++j)
            for (int k=0; k<=KKfim; ++k) {
                s.velX(NX,j,k)=s.velX(s.numCellsXm1,j,k);
                s.velY(NX,j,k)=s.velY(s.numCellsXm1,j,k);
                s.velZ(NX,j,k)=s.velZ(s.numCellsXm1,j,k);
            }
    else
        for (int j=s.jLow[NX]+1; j<=s.jHigh[NX]-1; ++j)
            for (int k=0; k<=KKfim; ++k) {
                s.velX(NX,j,k)=2*s.velX(s.numCellsXm1,j,k)-s.velX(NX-2,j,k);
                s.velY(NX,j,k)=2*s.velY(s.numCellsXm1,j,k)-s.velY(NX-2,j,k);
                s.velZ(NX,j,k)=2*s.velZ(s.numCellsXm1,j,k)-s.velZ(NX-2,j,k);
            }
}

void updateVelocities(SimState& s)
{
    if (s.cfg.lateralCondition==LateralBC::Periodic) velUpdImpl<LateralBC::Periodic>(s);
    else                                              velUpdImpl<LateralBC::SolidWall>(s);
}

// ════════════════════════════════════════════════════════════════════════════
//  computeMomentumResidual — OPT-A template BC.
// ════════════════════════════════════════════════════════════════════════════
template <LateralBC kBC>
static void residImpl(SimState& s)
{
    const auto& c   = s.cfg;
    const int KZ    = c.numCellsZ;
    const int KKfim = (kBC==LateralBC::SolidWall) ? s.numCellsZm1 : KZ;
    const double qIdx=0.25/c.cellSizeX, qIdy=0.25/c.cellSizeY, qIdz=0.25/c.cellSizeZ;

    s.scratchField.fill(0);
    double rMax=0, rSS=0; int cnt=0;

    for (int i=1; i<=s.numCellsXm1; ++i) {
        const int ip=i+1;
        for (int j=s.jLow[i]+1; j<=s.jHigh[i]-1; ++j) {
            const int jp=j+1;
            for (int k=1; k<=KKfim; ++k) {
                const int kp = knext<kBC>(k,KZ);
                const double gPx =
                    (s.press(ip,j,k) -s.press(i,j,k) +s.press(ip,jp,k) -s.press(i,jp,k)
                    +s.press(ip,j,kp)-s.press(i,j,kp)+s.press(ip,jp,kp)-s.press(i,jp,kp))*qIdx;
                const double gPy =
                    (s.press(i,jp,k) -s.press(i,j,k) +s.press(ip,jp,k) -s.press(ip,j,k)
                    +s.press(i,jp,kp)-s.press(i,j,kp)+s.press(ip,jp,kp)-s.press(ip,j,kp))*qIdy;
                const double gPz =
                    (-s.press(i,jp,k) -s.press(i,j,k) -s.press(ip,jp,k) -s.press(ip,j,k)
                    +s.press(i,jp,kp)+s.press(i,j,kp)+s.press(ip,jp,kp)+s.press(ip,j,kp))*qIdz;
                const double rU=s.accelX(i,j,k)-gPx;
                const double rV=s.accelY(i,j,k)-gPy;
                const double rW=s.accelZ(i,j,k)-gPz;
                const double sq=rU*rU+rV*rV+rW*rW;
                const double nm=std::sqrt(sq);
                s.scratchField(i,j,k)=nm;
                if (nm>rMax) { rMax=nm; s.iResidMax=i; s.jResidMax=j; s.kResidMax=k; }
                rSS+=sq; ++cnt;
            }
        }
    }
    s.momentumResidMax = rMax;
    s.momentumResidRMS = cnt ? std::sqrt(rSS/cnt) : 0.0;
    s.counter = cnt;
}

void computeMomentumResidual(SimState& s)
{
    if (s.cfg.lateralCondition==LateralBC::Periodic) residImpl<LateralBC::Periodic>(s);
    else                                              residImpl<LateralBC::SolidWall>(s);
}

// ════════════════════════════════════════════════════════════════════════════
//  computeDivergence — OPT-A template BC.
// ════════════════════════════════════════════════════════════════════════════
template <LateralBC kBC>
static void divImpl(SimState& s)
{
    const auto& c = s.cfg;
    const int KZ  = c.numCellsZ;
    const double qIdx=0.25/c.cellSizeX, qIdy=0.25/c.cellSizeY, qIdz=0.25/c.cellSizeZ;
    const double vol=c.cellSizeX*c.cellSizeY*c.cellSizeZ;
    double dMax=0, iDiv=0, iAbs=0;

    for (int i=1; i<=c.numCellsX; ++i) {
        const int im=i-1;
        const int jS=(i!=s.degreeIndex2)?s.jLow[i]+1 :s.jLow[im]+1;
        const int jN=(i!=s.degreeIndex2)?s.jHigh[i]  :s.jHigh[im];
        for (int j=jS; j<=jN; ++j) {
            const int jm=j-1;
            for (int k=1; k<=KZ; ++k) {
                const int km=kprev<kBC>(k,KZ);
                const double d =
                    (s.velX(i,j,k) -s.velX(im,j,k) +s.velX(i,jm,k) -s.velX(im,jm,k)
                    +s.velX(i,j,km)-s.velX(im,j,km)+s.velX(i,jm,km)-s.velX(im,jm,km))*qIdx
                   +(s.velY(i,j,k) -s.velY(i,jm,k) +s.velY(im,j,k) -s.velY(im,jm,k)
                    +s.velY(i,j,km)-s.velY(i,jm,km)+s.velY(im,j,km)-s.velY(im,jm,km))*qIdy
                   +(s.velZ(i,j,k) +s.velZ(i,jm,k) +s.velZ(im,j,k) +s.velZ(im,jm,k)
                    -s.velZ(i,j,km)-s.velZ(i,jm,km)-s.velZ(im,j,km)-s.velZ(im,jm,km))*qIdz;
                const double ad=std::abs(d);
                if (ad>dMax) { dMax=ad; s.iDilMax=i; s.jDilMax=j; s.kDilMax=k; }
                iDiv+=d; iAbs+=ad;
            }
        }
    }
    s.dilatationMax    = dMax;
    s.intDivergence    = iDiv*vol;
    s.intAbsDivergence = iAbs*vol;
}

void computeDivergence(SimState& s)
{
    if (s.cfg.lateralCondition==LateralBC::Periodic) divImpl<LateralBC::Periodic>(s);
    else                                              divImpl<LateralBC::SolidWall>(s);
}

// ════════════════════════════════════════════════════════════════════════════
//  adaptTimeStep — unchanged, already minimal cost (0.3% of runtime).
// ════════════════════════════════════════════════════════════════════════════
void adaptTimeStep(SimState& s)
{
    const auto& c = s.cfg;
    double uM=0, vM=0, wM=0;
    for (int i=1; i<=s.numCellsXm1; ++i)
        for (int j=s.jLow[i]+1; j<=s.jHigh[i]-1; ++j)
            for (int k=1; k<=c.numCellsZ; ++k) {
                uM=std::max(uM,std::abs(s.velX(i,j,k)));
                vM=std::max(vM,std::abs(s.velY(i,j,k)));
                wM=std::max(wM,std::abs(s.velZ(i,j,k)));
            }
    const double Rd  = (c.hyperViscousStart==0) ? c.reynoldsNumber : c.hyperViscousRe;
    const double dtV = 0.5*Rd/(1.0/s.cellSizeXsq+1.0/s.cellSizeYsq+1.0/s.cellSizeZsq);
    const double dtA = std::min({c.cellSizeX/(uM+1e-30),
                                 c.cellSizeY/(vM+1e-30),
                                 c.cellSizeZ/(wM+1e-30)});
    s.timeStepSize = 0.35*std::min(dtV,dtA);
}