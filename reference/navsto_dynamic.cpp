// =============================================================================
// NavSto_dynamic.cpp
// Numerical Solution of the Navier-Stokes Equations
// Incompressible Newtonian 3D Flow
// Semi-staggered grid, UNIFAES scheme for advection-viscous terms,
// Poisson pressure equation with momentum interpolation
//
// Original Pascal code by J.R. Figueiredo, 2016–2019
// Translated to C++ (direct translation, V1 – no optimizations)
// Rewritten to use fully dynamically allocated arrays (std::vector).
// =============================================================================

#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <string>
#include <cassert>
#include <vector>

// =============================================================================
// Dynamic 3-D array helper
//
// All 3-D field arrays are stored as flat std::vector<double> of length
// (dimI * dimJ * dimK).  The inline accessor F3(v,i,j,k) maps (i,j,k) to
// the correct flat index.
// =============================================================================

// Global dimension extents used by the accessor macro.
// These are set once in main() before any array is used.
static int g_dimI = 0;   // II + 1
static int g_dimJ = 0;   // JJ + 1
static int g_dimK = 0;   // KK + 1

// Flat-index accessor for a 3-D array stored as vector<double>.
// Dimensions are (g_dimI x g_dimJ x g_dimK).
inline double& F3(std::vector<double>& v, int i, int j, int k)
{
    return v[static_cast<std::size_t>(i) * g_dimJ * g_dimK
           + static_cast<std::size_t>(j) * g_dimK
           + k];
}

// Flat-index accessor for a 2-D array stored as vector<double>.
// Dimensions are (g_dimI x g_dimJ).
inline double& F2(std::vector<double>& v, int i, int j)
{
    return v[static_cast<std::size_t>(i) * g_dimJ + j];
}

// Helper: allocate a 3-D field of zeros with the current global dimensions.
inline std::vector<double> alloc3D()
{
    return std::vector<double>(
        static_cast<std::size_t>(g_dimI) * g_dimJ * g_dimK, 0.0);
}

// Helper: allocate a 2-D field of zeros with the current global dimensions.
inline std::vector<double> alloc2D()
{
    return std::vector<double>(
        static_cast<std::size_t>(g_dimI) * g_dimJ, 0.0);
}

// =============================================================================
// Dynamic 1-D vector helpers
//
// VetMax was double[-1..NNmax], stored as double[NNmax+2] with index offset +1.
// We keep the same offset convention: element at logical index idx is stored
// at physical index idx+1.
// =============================================================================

// Offset accessor for a VetMax-style vector (logical index range -1..N).
inline double& VM(std::vector<double>& v, int idx) { return v[idx + 1]; }

// =============================================================================
// String type (exactly 3 chars + null)
// =============================================================================
using Nome = char[4];

// =============================================================================
// Global simulation parameters
// =============================================================================

// --- geometry / problem strings ---
Nome  Geometria, CondSaida, CondLater, PerfilInicial, TipoGeometria, half;

// --- integer indices ---
int II, JJ, KK, IIm, JJm, KKm, NN;
int nt, ntmax, ntmostra, kmostra, itp, jfim_var;
int IIdeg1, IIdeg2, JJdeg, JJinicial, JJultimo, IIrmp1, IIrmp2, JJrmp1, JJrmp2;
int celulas, IIhiperviscoso;
int niter, FatEx, ix, jx, kx;
int iresidmax, jresidmax, kresidmax;
int iDilmax, jDilmax, kDilmax;
int KKZ2, um, zero_var;
int im_g, ip_g, jm_g, jp_g, km_g, kp_g, jp2, jm2, jp3, NNZ2, DNN;
int jjiniloop, jjfimloop, contador, nciclos;

// --- real parameters ---
double dx, dy, dz, dxq, dyq, dzq, dt, T;
double Alt, Cmp, Lrg, Re;
double dmax, Resid, ResDil, DilMax, ResidRMS, ResidMax;
double Eps;
double IntDil, IntAbsDil;
double wx, wy, wz, uu, vv, ww;
double pi1mdx, dpi1mdy, dpi1mdz, dtpr;
double cipmjk, cijpmk, cijkpm, sijk;
double Redt, dtZ6, dtZ24, DecaiHiper;
double dpdxinicial, dpdyinicial, dpdxfinal, aux_g;
double ReHiper, fluxo, deltapAnt, VelMedAnt;
double hZdx, hZdy, C1, C2, C3, C4;
double umax_g, vmax_g;

// --- index arrays (dynamic) ---
std::vector<int> IIini, IIfim, JJini, JJfim;

// --- 3-D field arrays (dynamic) ---
std::vector<double> u, v, w, p, Au, Av, Aw, s, crng;

// --- output files ---
std::ofstream saida, saida2, saida3;

// =============================================================================
// Helper
// =============================================================================
inline double pascal_pi() { return 4.0 * std::atan(1.0); }

// =============================================================================
// Forward declarations
// =============================================================================
void INIT();
void CoCOEF();
void cv2Fonte();
void PRESSAOGaussSiedel();
void CovRESIDUO();
void CovDILATACAO();
void CovVELOCDD();
void CovRESULTADOS();
void DeltaT();

// =============================================================================
// UNIFAES helper functions
// =============================================================================

static double qsi_func(double DPe, double pip, double xeZdx)
{
    if (std::abs(DPe) < 0.01)
        return DPe * (1.0 - DPe * DPe / 60.0) / 12.0 + xeZdx - 0.5;
    else
        return (pip - 1.0) / DPe + xeZdx;
}

static void PiReg(double Reloc, double DPe, double& pip, double& cip, double& cim)
{
    double pim;
    if (std::abs(DPe) < 0.1)
        pip = 1.0 / ((((0.05 * DPe + 0.25) * DPe + 1.0) * DPe / 6.0 + 0.5) * DPe + 1.0);
    else if (std::abs(DPe) <= 200.0)
        pip = DPe / (std::exp(DPe) - 1.0);
    else if (DPe > 200.0)
        pip = 0.0;
    else
        pip = -DPe;

    pim = DPe + pip;
    cip = pip / Reloc;
    cim = pim / Reloc;
}

// =============================================================================
// PROCEDURE PerfilDesenvolvido
// =============================================================================
static void PerfilDesenvolvido(int Idesenvolvido, double& dpdxinicial_loc, double& VelMed)
{
    int jvarredura = JJfim[Idesenvolvido] - JJini[Idesenvolvido];
    double deltavarredura = jvarredura * dy;

    VelMed = 1.5 / deltavarredura;

    for (int jesp = 0; jesp <= jvarredura; ++jesp) {
        int j = jesp + JJini[Idesenvolvido];
        double jdyZH = (double)jesp / jvarredura;
        F3(u, Idesenvolvido, j, 0) = -4.0 * (jdyZH - 1.0) * jdyZH * VelMed;
        for (int k = 1; k <= KK; ++k)
            F3(u, Idesenvolvido, j, k) = F3(u, Idesenvolvido, j, 0);
    }

    dpdxinicial_loc = -12.0 * VelMed / (deltavarredura * deltavarredura * Re);

    if (std::string(CondLater) == "dir") {
        for (int j = JJini[Idesenvolvido] + 1; j <= JJfim[Idesenvolvido] - 1; ++j) {
            F3(u, Idesenvolvido, j, 0)  = 0.0;
            F3(u, Idesenvolvido, j, KK) = 0.0;
        }
        for (int k = 1; k <= KKm; ++k) {
            F3(u, Idesenvolvido, JJini[Idesenvolvido], k) = 0.0;
            F3(u, Idesenvolvido, JJfim[Idesenvolvido], k) = 0.0;
        }

        double cjpmk  = 1.0 / dyq;
        double cjkpm  = 1.0 / dzq;
        double cij    = 2.0 * (cjpmk + cjkpm);
        double Zcij   = 1.0 / cij;
        double relax  = 1.85;
        int    iter   = 0;

        double maxresid;
        do {
            maxresid = 0.0;
            ++iter;
            for (int j = JJini[Idesenvolvido] + 1; j <= JJfim[Idesenvolvido] - 1; ++j) {
                int jm = j - 1, jp = j + 1;
                for (int k = 1; k <= KKm; ++k) {
                    int km = k - 1, kp = k + 1;
                    double aux = cjpmk * (F3(u, Idesenvolvido, jp, k) + F3(u, Idesenvolvido, jm, k))
                               + cjkpm * (F3(u, Idesenvolvido, j, kp) + F3(u, Idesenvolvido, j, km))
                               - dpdxinicial_loc;
                    double resid = std::abs(aux - cij * F3(u, 0, j, k));
                    if (resid > maxresid) maxresid = resid;
                    double u0 = aux * Zcij;
                    F3(u, Idesenvolvido, j, k) += relax * (u0 - F3(u, Idesenvolvido, j, k));
                }
            }
        } while (maxresid >= 0.00000001 && iter < 100000);

        double integral = 0.0;
        for (int j = JJini[Idesenvolvido] + 1; j <= JJfim[Idesenvolvido] - 1; ++j)
            for (int k = 1; k <= KKm; ++k)
                integral += F3(u, Idesenvolvido, j, k);
        integral /= (jvarredura * KK);
        double correcao = 1.0 / integral;
        dpdxinicial_loc *= correcao;
        for (int j = JJini[Idesenvolvido] + 1; j <= JJfim[Idesenvolvido] - 1; ++j)
            for (int k = 1; k <= KKm; ++k)
                F3(u, Idesenvolvido, j, k) *= correcao;
    }
}

// =============================================================================
// PROCEDURE PerfilDesenvolvidoHoriz
// =============================================================================
static void PerfilDesenvolvidoHoriz(int Jdesenvolvido, double& dpdyinicial_loc, double& VelMed)
{
    int ivarredura = IIfim[Jdesenvolvido] - IIini[Jdesenvolvido];
    double deltavarredura = ivarredura * dx;

    VelMed = 1.0 / deltavarredura;

    for (int iesp = 0; iesp <= ivarredura; ++iesp) {
        int i = iesp + IIini[Jdesenvolvido];
        double idxZH = (double)iesp / ivarredura;
        F3(v, i, Jdesenvolvido, 0) = -6.0 * (idxZH - 1.0) * idxZH * VelMed;
        for (int k = 1; k <= KK; ++k)
            F3(v, i, Jdesenvolvido, k) = F3(v, i, Jdesenvolvido, 0);
    }

    dpdyinicial_loc = -12.0 * VelMed / (deltavarredura * deltavarredura * Re);

    if (std::string(CondLater) == "dir") {
        for (int i = IIini[Jdesenvolvido]; i <= IIfim[Jdesenvolvido]; ++i) {
            F3(v, i, Jdesenvolvido, 0)  = 0.0;
            F3(v, i, Jdesenvolvido, KK) = 0.0;
        }
        for (int k = 1; k <= KKm; ++k) {
            F3(v, IIini[Jdesenvolvido], Jdesenvolvido, k) = 0.0;
            F3(v, IIfim[Jdesenvolvido], Jdesenvolvido, k) = 0.0;
        }

        double cipmk  = 1.0 / dxq;
        double cikpm  = 1.0 / dzq;
        double cij    = 2.0 * (cipmk + cikpm);
        double Zcij   = 1.0 / cij;
        double relax  = 1.85;
        int    iter   = 0;

        double maxresid;
        do {
            maxresid = 0.0;
            ++iter;
            for (int i = IIini[Jdesenvolvido] + 1; i <= IIfim[Jdesenvolvido] - 1; ++i) {
                int im = i - 1, ip = i + 1;
                for (int k = 1; k <= KKm; ++k) {
                    int km = k - 1, kp = k + 1;
                    double aux = cipmk * (F3(v, ip, Jdesenvolvido, k) + F3(v, im, Jdesenvolvido, k))
                               + cikpm * (F3(v, i, Jdesenvolvido, kp) + F3(v, i, Jdesenvolvido, km))
                               - dpdyinicial_loc;
                    double resid = std::abs(aux - cij * F3(v, i, Jdesenvolvido, k));
                    if (resid > maxresid) maxresid = resid;
                    double v0 = aux * Zcij;
                    F3(v, i, Jdesenvolvido, k) += relax * (v0 - F3(v, i, Jdesenvolvido, k));
                }
            }
        } while (maxresid >= 0.00000001 && iter < 100000);

        double integral = 0.0;
        for (int i = IIini[Jdesenvolvido] + 1; i <= IIfim[Jdesenvolvido] - 1; ++i)
            for (int k = 1; k <= KKm; ++k)
                integral += F3(v, i, Jdesenvolvido, k);
        integral /= (ivarredura * KK);
        double correcao = 1.0 / integral;
        dpdyinicial_loc *= correcao;
        for (int i = IIini[Jdesenvolvido] + 1; i <= IIfim[Jdesenvolvido] - 1; ++i)
            for (int k = 1; k <= KKm; ++k)
                F3(v, i, Jdesenvolvido, k) *= correcao;
    }
}

// =============================================================================
// PROCEDURE PerfilLongitudinalInicial
// =============================================================================
static void PerfilLongitudinalInicial(double& umax_loc, double& vmax_loc)
{
    double loc_dxq = dx * dx;
    double loc_dyq = dy * dy;
    int KKref = 0;
    IIm = II - 1;

    // --- Potential stream function boundary + initial conditions ---
    if (std::string(TipoGeometria) == "Axl") {
        for (int i = 0; i <= II; ++i) {
            int jvarredura = JJfim[i] - JJini[i];
            for (int jesp = 0; jesp <= jvarredura; ++jesp) {
                int j = jesp + JJini[i];
                double jdyZH = (double)jesp / jvarredura;
                F3(crng, i, j, KKref) = jdyZH;
            }
        }
    }

    if (std::string(TipoGeometria) == "Crv") {
        for (int i = 0; i <= IIdeg2; ++i) {
            F3(crng, i, JJini[i], kmostra) = (double)(IIdeg2 - i) / IIdeg2;
            F3(crng, i, JJfim[i], kmostra) = 1.0;
        }
        for (int i = IIdeg2 + 1; i <= II; ++i) {
            F3(crng, i, JJini[i], kmostra) = 0.0;
            F3(crng, i, JJfim[i], kmostra) = 1.0;
        }
        for (int j = 0; j <= JJ; ++j)
            F3(crng, IIini[j], j, kmostra) = 1.0;
        for (int j = 0; j <= JJdeg; ++j)
            F3(crng, IIfim[j], j, kmostra) = 0.0;
        for (int j = JJdeg + 1; j <= JJ; ++j)
            F3(crng, IIfim[j], j, kmostra) = (double)(j - JJdeg) / (JJ - JJdeg);

        for (int i = 1; i <= II - 1; ++i) {
            for (int j = JJini[i] + 1; j <= JJfim[i] - 1; ++j) {
                double aa = 1.0 / (i - IIini[j]);
                double bb = 1.0 / (IIfim[j] - i);
                double cc = 1.0 / (j - JJini[i]);
                double dd = 1.0 / (JJfim[i] - j);
                F3(crng, i, j, KKref) = (aa * F3(crng, IIini[j], j, KKref)
                                       + bb * F3(crng, IIfim[j], j, KKref)
                                       + cc * F3(crng, i, JJini[i], KKref)
                                       + dd * F3(crng, i, JJfim[i], KKref)) / (aa + bb + cc + dd);
            }
        }
    }

    // --- Solve for potential stream function (Gauss-Seidel) ---
    double cipmj = 1.0 / loc_dxq;
    double cijpm = 1.0 / loc_dyq;
    double cij   = 2.0 * (cipmj + cijpm);
    double Zcij  = 1.0 / cij;
    double relax = 1.8;
    int    itera = 0;

    double maxresid;
    do {
        maxresid = 0.0;
        ++itera;
        for (int i = 1; i <= IIm; ++i) {
            int im = i - 1, ip = i + 1;
            for (int j = JJini[i] + 1; j <= JJfim[i] - 1; ++j) {
                int jm = j - 1, jp = j + 1;
                double aux = cipmj * (F3(crng, ip, j, KKref) + F3(crng, im, j, KKref))
                           + cijpm * (F3(crng, i, jp, KKref) + F3(crng, i, jm, KKref));
                double resid = std::abs(aux - cij * F3(crng, i, j, KKref));
                if (resid > maxresid) maxresid = resid;
                double crng0 = aux * Zcij;
                F3(crng, i, j, KKref) += relax * (crng0 - F3(crng, i, j, KKref));
            }
        }
    } while (maxresid >= 0.0000001 && itera < 100000);

    // --- Adapt to rotational field ---
    for (int i = 0; i <= II; ++i)
        for (int j = JJini[i]; j <= JJfim[i]; ++j) {
            double c = F3(crng, i, j, KKref);
            F3(crng, i, j, KKref) = c * c * (3.0 - 2.0 * c);
        }

    // --- Derive u from stream function (cubic polynomial interpolation) ---
    umax_loc = 0.0;
    for (int i = 0; i <= IIm; ++i) {
        int jsrt = JJini[i];
        while (true) {
            int j   = jsrt;
            int jp  = jsrt + 1;
            int jpp = jsrt + 2;
            int jp3 = jsrt + 3;
            double w0 = F3(crng, i, j,   KKref);
            double w1 = F3(crng, i, jp,  KKref);
            double w2 = F3(crng, i, jpp, KKref);
            double w3 = F3(crng, i, jp3, KKref);
            double a  = (2*w3 - 9*w2 + 18*w1 - 11*w0) / (6*dy);
            double b  = (-w3 + 4*w2 - 5*w1 + 2*w0) / (2*loc_dyq);
            double c2 = (w3 - 3*w2 + 3*w1 - w0) / (6*loc_dyq*dy);
            F3(u, i, jp,  KKref) = (3*c2*dy + 2*b)*dy + a;
            F3(u, i, jpp, KKref) = (12*c2*dy + 4*b)*dy + a;
            jsrt = jpp;
            if (jsrt == JJfim[i] - 2) jsrt = jp;
            if (std::abs(F3(u, i, jp,  KKref)) > umax_loc) umax_loc = std::abs(F3(u, i, jp,  KKref));
            if (std::abs(F3(u, i, jpp, KKref)) > umax_loc) umax_loc = std::abs(F3(u, i, jpp, KKref));
            if (jp3 == JJfim[i]) break;
        }
    }

    // --- Derive v from stream function ---
    vmax_loc = 0.0;
    for (int j = 0; j <= JJ; ++j) {
        int isrt = IIini[j];
        while (true) {
            int i   = isrt;
            int ip  = isrt + 1;
            int ipp = isrt + 2;
            int ip3 = isrt + 3;
            double w0 = F3(crng, i,   j, KKref);
            double w1 = F3(crng, ip,  j, KKref);
            double w2 = F3(crng, ipp, j, KKref);
            double w3 = F3(crng, ip3, j, KKref);
            double a  = (2*w3 - 9*w2 + 18*w1 - 11*w0) / (6*dx);
            double b  = (-w3 + 4*w2 - 5*w1 + 2*w0) / (2*loc_dxq);
            double c2 = (w3 - 3*w2 + 3*w1 - w0) / (6*loc_dxq*dx);
            F3(v, ip,  j, KKref) = -(3*c2*dx + 2*b)*dx - a;
            F3(v, ipp, j, KKref) = -(12*c2*dx + 4*b)*dx - a;
            isrt = ipp;
            if (isrt == IIfim[j] - 2) isrt = ip;
            if (std::abs(F3(v, ip,  j, KKref)) > vmax_loc) vmax_loc = std::abs(F3(v, ip,  j, KKref));
            if (std::abs(F3(v, ipp, j, KKref)) > vmax_loc) vmax_loc = std::abs(F3(v, ipp, j, KKref));
            if (ip3 == IIfim[j]) break;
        }
    }

    // Copy boundary
    int i = II;
    for (int j = JJini[i]; j <= JJfim[i]; ++j) {
        int im_loc = i - 1;
        F3(u, i, j, KKref) = F3(u, im_loc, j, KKref);
        F3(v, i, j, KKref) = F3(v, im_loc, j, KKref);
    }

    // Clear stream function array
    for (int i2 = 0; i2 <= II; ++i2)
        for (int j = JJini[i2]; j <= JJfim[i2]; ++j)
            F3(crng, i2, j, KKref) = 0.0;

    // Generalise to multiple z planes
    if (std::string(CondLater) == "prd") {
        for (int k = 1; k <= KK; ++k)
            for (int i2 = 0; i2 <= II; ++i2)
                for (int j = JJini[i2]; j <= JJfim[i2]; ++j) {
                    F3(u, i2, j, k) = F3(u, i2, j, KKref);
                    F3(v, i2, j, k) = F3(v, i2, j, KKref);
                }
    } else if (std::string(TipoGeometria) == "Axl") {
        for (int k = 1; k <= KKm; ++k) {
            double flux = 0.0;
            for (int j = JJini[0] + 1; j <= JJfim[0] - 1; ++j)
                flux += F3(u, 0, j, KKref);
            flux *= dy;
            for (int i2 = 0; i2 <= II; ++i2)
                for (int j = JJini[i2]; j <= JJfim[i2]; ++j) {
                    F3(u, i2, j, k) = F3(u, i2, j, KKref) * flux;
                    F3(v, i2, j, k) = F3(v, i2, j, KKref) * flux;
                }
        }
    } else {
        for (int k = 1; k <= KKm; ++k) {
            double flux = 0.0;
            for (int i2 = IIini[0] + 1; i2 <= IIfim[0] - 1; ++i2)
                flux += F3(v, i2, 0, KKref);
            flux *= dx;
            for (int i2 = 0; i2 <= II; ++i2)
                for (int j = JJini[i2]; j <= JJfim[i2]; ++j) {
                    F3(u, i2, j, k) = F3(u, i2, j, KKref) * flux;
                    F3(v, i2, j, k) = F3(v, i2, j, KKref) * flux;
                }
        }
    }
}

// =============================================================================
// PROCEDURE PressaoInicial
// =============================================================================
static void PressaoInicial()
{
    int    jvarredura0    = JJfim[0] - JJini[0];
    double jvarredura0dy  = jvarredura0 * dy;
    double deltap0        = -12.0 * dx / (jvarredura0dy * jvarredura0dy * jvarredura0dy * Re);
    double VelMed0        = 1.0 / jvarredura0dy;
    double pressao_loc    = 0.0;
    F3(p, 0, 0, 0) = pressao_loc;

    double hpi    = 2.0 * std::atan(1.0);
    int    IIorig = II - 5 * IIhiperviscoso / 8;
    int    IIamp  = 3  * IIhiperviscoso / 8;
    double ZReOrig = 0.5 * (1.0/ReHiper + 1.0/Re);
    double ZReAmp  = 0.5 * (-1.0/ReHiper + 1.0/Re);

    for (int i = 1; i <= IIdeg1; ++i) {
        int im_loc = i - 1;
        double jvarredurady;
        if (i != IIdeg2) jvarredurady = (JJfim[i] - JJini[i]) * dy;
        else             jvarredurady = (JJfim[im_loc] - JJini[im_loc]) * dy;
        double VelMed = 1.0 / jvarredurady;

        double ZReloc;
        if (i <= II - IIhiperviscoso)        ZReloc = 1.0/Re;
        else if (i < II - IIhiperviscoso/4)  ZReloc = ZReOrig - ZReAmp * std::sin(((i - IIorig) / (double)IIamp) * hpi);
        else                                  ZReloc = 1.0/ReHiper;

        double deltap1 = -12.0 * dx * ZReloc / (jvarredurady * jvarredurady);
        double deltap  = 0.5 * (deltap0 + deltap1);
        deltap += 0.5 * (VelMed0 * VelMed0 - VelMed * VelMed);
        deltap0 = deltap1;
        VelMed0 = VelMed;
        F3(p, i, 0, 0) = F3(p, im_loc, 0, 0) + deltap;

        int JJstrt = (i != IIdeg2) ? JJini[i] + 1 : JJini[im_loc] + 1;
        for (int j = JJstrt; j <= JJfim[i]; ++j)
            for (int k = 1; k <= KK; ++k)
                F3(p, i, j, k) = F3(p, i, 0, 0);
    }

    for (int i = IIdeg1 + 1; i <= II; ++i) {
        F3(p, i, 0, 0) = F3(p, IIdeg1, 0, 0);
        for (int j = JJini[i] + 1; j <= JJfim[i]; ++j)
            for (int k = 1; k <= KK; ++k)
                F3(p, i, j, k) = F3(p, i, 0, 0);
    }
}

// =============================================================================
// PROCEDURE INIT
// =============================================================================
void INIT()
{
    IIm = II - 1;  JJm = JJ - 1;  KKm = KK - 1;
    dxq = dx * dx; dyq = dy * dy; dzq = dz * dz;

    // Zero all field arrays
    for (int i = 0; i <= II; ++i)
        for (int j = 0; j <= JJ; ++j)
            for (int k = 0; k <= KK; ++k) {
                F3(u,  i, j, k) = 0.0;  F3(v,  i, j, k) = 0.0;  F3(w,  i, j, k) = 0.0;
                F3(p,  i, j, k) = 0.0;  F3(s,  i, j, k) = 0.0;
                F3(Au, i, j, k) = 0.0;  F3(Av, i, j, k) = 0.0;  F3(Aw, i, j, k) = 0.0;
            }

    // --- Geometry setup ---
    if (std::string(Geometria) == "Eab") {
        JJdeg  = JJ / 2;
        IIdeg1 = II / 12;   IIdeg2 = II + 2;   IIrmp1 = -2;   IIrmp2 = II + 2;
        for (int i = 0; i <= IIdeg1; ++i)         { JJini[i] = JJdeg; JJfim[i] = JJ; }
        for (int i = IIdeg1+1; i <= II; ++i)       { JJini[i] = 0;    JJfim[i] = JJ; }
        for (int j = 0; j <= JJdeg; ++j)            IIini[j] = IIdeg1;
        for (int j = JJdeg+1; j <= JJ; ++j)         IIini[j] = 0;
        for (int j = 0; j <= JJ; ++j)               IIfim[j] = II;
    }

    if (std::string(Geometria) == "Cab") {
        JJinicial = JJ / 2;  JJultimo = JJ;
        IIdeg1 = -1;   IIdeg2 = 2 * NN;   IIrmp1 = -2;   IIrmp2 = II + 2;
        for (int i = 0; i < IIdeg2; ++i)            { JJini[i] = 0;         JJfim[i] = JJultimo; }
        for (int i = IIdeg2; i <= II; ++i)           { JJini[i] = JJinicial; JJfim[i] = JJultimo; }
        for (int j = 0; j <= JJ; ++j)                IIini[j] = 0;
        for (int j = 0; j <= JJinicial; ++j)         IIfim[j] = IIdeg2;
        for (int j = JJinicial+1; j <= JJultimo; ++j) IIfim[j] = II;
    }

    if (std::string(Geometria) == "CVD") {
        JJinicial = JJ / 2;    JJultimo = JJ;
        IIdeg1 = NN / 2;       IIdeg2 = 3 * NN / 2;   IIrmp1 = -2;   IIrmp2 = II + 2;
        for (int i = 0; i <= IIdeg1; ++i)                  { JJini[i] = JJinicial; JJfim[i] = JJultimo; }
        for (int i = IIdeg1+1; i < IIdeg2; ++i)            { JJini[i] = 0;         JJfim[i] = JJultimo; }
        for (int i = IIdeg2; i <= II; ++i)                  { JJini[i] = JJinicial; JJfim[i] = JJultimo; }
        for (int j = 0; j <= JJinicial; ++j)               { IIini[j] = IIdeg1;    IIfim[j] = IIdeg2;   }
        for (int j = JJinicial+1; j <= JJultimo; ++j)      { IIini[j] = 0;         IIfim[j] = II;        }
    }

    if (std::string(Geometria) == "Exp") {
        IIdeg1 = -2;   IIdeg2 = II + 2;
        int FatExp   = 3;
        JJinicial    = (FatExp - 1) * JJ / (2 * FatExp);
        JJultimo     = (FatExp + 1) * JJ / (2 * FatExp);
        IIrmp1       = 2 * NN / 3;
        IIrmp2       = IIrmp1 + JJinicial;
        for (int i = 0; i <= IIrmp1; ++i)           { JJini[i] = JJinicial; JJfim[i] = JJultimo;          }
        for (int i = IIrmp1+1; i < IIrmp2; ++i)     { JJini[i] = IIrmp2-i; JJfim[i] = JJultimo+i-IIrmp1; }
        for (int i = IIrmp2; i <= II; ++i)           { JJini[i] = 0;        JJfim[i] = JJ;                 }
        for (int j = 0; j <= JJinicial; ++j)          IIini[j] = IIrmp1 - j;
        for (int j = JJinicial+1; j < JJultimo; ++j)  IIini[j] = 0;
        for (int j = JJultimo; j <= JJ; ++j)           IIini[j] = IIrmp1 + j - JJultimo;
        for (int j = 0; j <= JJ; ++j)                  IIfim[j] = II;
    }

    if (std::string(Geometria) == "Eun") {
        IIdeg1 = -2;   IIdeg2 = II + 2;
        int FatExp = 3;
        JJinicial  = JJ - JJ / FatExp;
        JJultimo   = JJ;
        IIrmp1     = NN;
        IIrmp2     = IIrmp1 + JJinicial;
        for (int i = 0; i <= IIrmp1; ++i)               { JJini[i] = JJinicial;       JJfim[i] = JJultimo; }
        for (int i = IIrmp1+1; i < IIrmp2; ++i)         { JJini[i] = JJinicial-(i-IIrmp1); JJfim[i] = JJultimo; }
        for (int i = IIrmp2; i <= II; ++i)               { JJini[i] = 0;               JJfim[i] = JJ;       }
        for (int j = 0; j <= JJinicial; ++j)              IIini[j] = IIrmp1 - j + JJinicial;
        for (int j = JJinicial+1; j < JJultimo; ++j)      IIini[j] = 0;
        for (int j = 0; j <= JJ; ++j)                     IIfim[j] = II;
    }

    if (std::string(Geometria) == "Cnt") {
        IIdeg1 = -2;   IIdeg2 = II + 2;
        int FatCnt = 3;
        JJinicial  = 0;   JJultimo = JJ;
        IIrmp1     = 2 * NN / 3;
        IIrmp2     = IIrmp1 + (FatCnt - 1) * JJ / (2 * FatCnt);
        for (int i = 0; i <= IIrmp1; ++i)     { JJini[i] = JJinicial; JJfim[i] = JJultimo; }
        for (int i = IIrmp1+1; i <= IIrmp2; ++i) {
            JJini[i] = JJini[i-1] + 1;
            JJfim[i] = JJfim[i-1] - 1;
        }
        JJinicial = (FatCnt - 1) * JJ / (2 * FatCnt);
        JJultimo  = (FatCnt + 1) * JJ / (2 * FatCnt);
        for (int i = IIrmp2+1; i <= II; ++i)  { JJini[i] = JJinicial; JJfim[i] = JJultimo; }
        for (int j = 0; j <= JJ; ++j)           IIini[j] = 0;
        for (int j = 0; j <= JJinicial; ++j)    IIfim[j] = IIrmp1 + j;
        for (int j = JJinicial+1; j < JJultimo; ++j) IIfim[j] = II;
        for (int j = JJultimo; j <= JJ; ++j)    IIfim[j] = IIrmp1 + JJ - j;
    }

    if (std::string(Geometria) == "Cun") {
        IIdeg1 = -2;   IIdeg2 = II + 2;
        int FatCnt = 3;
        JJinicial  = JJ - JJ / FatCnt;
        IIrmp1     = NN;
        IIrmp2     = IIrmp1 + JJinicial;
        for (int i = 0; i <= II; ++i)                    JJfim[i] = JJ;
        for (int i = 0; i <= IIrmp1; ++i)               JJini[i] = 0;
        for (int i = IIrmp1+1; i < IIrmp2; ++i)         JJini[i] = i - IIrmp1;
        for (int i = IIrmp2; i <= II; ++i)               JJini[i] = JJinicial;
        for (int j = 0; j <= JJ; ++j)                    IIini[j] = 0;
        for (int j = 0; j <= JJinicial; ++j)             IIfim[j] = IIrmp1 + j;
        for (int j = JJinicial+1; j <= JJ; ++j)          IIfim[j] = II;
    }

    if (std::string(Geometria) == "Cvv") {
        IIdeg1 = 0;   IIdeg2 = NN;   JJdeg = JJ - NN;
        for (int i = 0; i < IIdeg2; ++i)              { JJini[i] = 0;    JJfim[i] = JJ; }
        for (int i = IIdeg2; i <= II; ++i)             { JJini[i] = JJdeg; JJfim[i] = JJ; }
        for (int j = 0; j <= JJ; ++j)                   IIini[j] = 0;
        for (int j = 0; j <= JJdeg; ++j)                IIfim[j] = IIdeg2;
        for (int j = JJdeg+1; j <= JJ; ++j)             IIfim[j] = II;
    }

    if (std::string(Geometria) == "Cam") {
        IIdeg1 = 0;   IIdeg2 = NN;   JJdeg = JJ - NN;
        int JJrmp1_loc = JJ - 5 * NN / 4;
        IIrmp1 = JJ - JJrmp1_loc;
        JJrmp1 = JJrmp1_loc;

        int JJrmp2, IIrmp2_loc;
        if (JJrmp1 < JJ - 6 * NN / 10) {
            JJrmp2 = JJrmp1 - 4 * NN / 10;
            IIrmp2_loc = IIdeg2 + JJdeg - JJrmp2;
        } else {
            JJrmp2 = -2;
            IIrmp2_loc = II + 2;
        }
        IIrmp2 = IIrmp2_loc;

        for (int i = 0; i < IIdeg2; ++i)               JJini[i] = 0;
        for (int i = IIdeg2; i <= IIrmp2; ++i)         JJini[i] = JJrmp2 + i - IIdeg2;
        for (int i = IIrmp2; i <= II; ++i)              JJini[i] = JJdeg;
        for (int i = 0; i <= IIrmp1; ++i)               JJfim[i] = JJrmp1 + i;
        for (int i = IIrmp1+1; i <= II; ++i)            JJfim[i] = JJ;
        for (int j = 0; j <= JJrmp1; ++j)               IIini[j] = 0;
        for (int j = JJrmp1+1; j <= JJ; ++j)            IIini[j] = j - JJrmp1;
        for (int j = 0; j <= JJrmp2; ++j)               IIfim[j] = IIdeg2;
        for (int j = JJrmp2+1; j <= JJdeg; ++j)         IIfim[j] = IIdeg2 + j - JJrmp2;
        for (int j = JJdeg+1; j <= JJ; ++j)             IIfim[j] = II;
    }

    // --- Initial velocity profiles ---
    double VelMed_loc;
    if (std::string(PerfilInicial) == "Ent") {
        int Idesenvolvido = 0;
        PerfilDesenvolvido(Idesenvolvido, dpdxinicial, VelMed_loc);
        for (int i = 1; i <= II; ++i)
            for (int j = JJini[0]; j <= JJfim[0]; ++j)
                for (int k = 0; k <= KK; ++k) {
                    F3(u, i, j, k) = F3(u, 0, j, k);
                    F3(v, i, j, k) = 0.0;
                }
        umax_g = 1.5;   vmax_g = 0.1 * umax_g;
    } else {
        // PerfilInicial == "LCP"
        if (std::string(TipoGeometria) == "Axl") {
            for (int i = 1; i <= IIm; ++i)
                for (int k = 0; k <= KK; ++k) {
                    F3(u, i, JJini[i], k) = 0.0;   F3(v, i, JJini[i], k) = 0.0;
                    F3(u, i, JJfim[i], k) = 0.0;   F3(v, i, JJfim[i], k) = 0.0;
                }
            int Idesenvolvido = 0;
            PerfilDesenvolvido(Idesenvolvido, dpdxinicial, VelMed_loc);
            Idesenvolvido = II;
            PerfilDesenvolvido(Idesenvolvido, dpdxinicial, VelMed_loc);
        } else {
            for (int i = 1; i <= IIm; ++i)
                for (int k = 0; k <= KK; ++k) {
                    F3(u, i, JJfim[i], k) = 0.0;   F3(v, i, JJfim[i], k) = 0.0;
                    F3(u, i, JJini[i], k) = 0.0;
                }
            for (int i = IIdeg2; i <= IIm; ++i)
                for (int k = 0; k <= KK; ++k)
                    F3(v, i, JJini[i], k) = 0.0;
            for (int j = 0; j <= JJ; ++j) {
                F3(u, IIini[j], j, 0) = 0.0;   F3(v, IIini[j], j, 0) = 0.0;
                F3(v, IIfim[j], j, 0) = 0.0;
            }
            for (int j = 0; j <= JJdeg; ++j) F3(u, IIfim[j], j, 0) = 0.0;

            int Jdesenvolvido = 0;
            PerfilDesenvolvidoHoriz(Jdesenvolvido, dpdyinicial, VelMed_loc);
            int Idesenvolvido = II;
            PerfilDesenvolvido(Idesenvolvido, dpdxinicial, VelMed_loc);
        }

        PerfilLongitudinalInicial(umax_g, vmax_g);
    }

    PressaoInicial();

    // Re-compute auxiliary parameters
    IIm = II - 1;   JJm = JJ - 1;   KKm = KK - 1;
    dxq = dx*dx;    dyq = dy*dy;    dzq = dz*dz;

    // Time step
    double dtcond = (IIhiperviscoso == 0)
        ? 0.5 * Re / (1.0/dxq + 1.0/dyq + 1.0/dzq)
        : 0.5 * ReHiper / (1.0/dxq + 1.0/dyq + 1.0/dzq);
    double dtadvc = (dx/umax_g < dy/vmax_g) ? dx/umax_g : dy/vmax_g;
    dt = (dtcond < dtadvc) ? dtcond : dtadvc;
    dt *= 0.35;

    // Cell count
    celulas = 0;
    for (int i = 1; i <= II; ++i) celulas += JJfim[i] - JJini[i];
    celulas *= KK;

    // Print header
    saida << "\n   H A R W E L \n";
    saida << "ALGORITMO PARA SOLUCAO NUMERICA DAS EQUACOES DE NAVIER-STOKES\n";
    saida << " ESCOAMENTO NEWTONIANO INCOMPRESSIVEL TRIDIMENSIONAL\n\n";
    saida << "MALHA SEMI-DESLOCADA COM ESPACAMENTO CARTESIANO REGULAR\n";
    saida << "ESQUEMA UNIFAES PARA TERMOS ADVECTIVOS-VISCOSOS\n";
    saida << "EQUACAO DE PRESSAO DE POISSON COM INTERPOLACAO DE MOMENTUM\n\n";
    saida << " PARAMETROS DE ENTRADA\n";
    saida << "  Geometria =";
    if      (std::string(Geometria)=="Exp") saida << " Expansao gradual\n";
    else if (std::string(Geometria)=="Cnt") saida << " Contracao gradual\n";
    else if (std::string(Geometria)=="Eun") saida << " Expansao gradual unilateral\n";
    else if (std::string(Geometria)=="Cun") saida << " Contracao gradual unilateral\n";
    else if (std::string(Geometria)=="Eab") saida << " Expansao abrupta\n";
    else if (std::string(Geometria)=="Cab") saida << " Contracao abrupta\n";
    else if (std::string(Geometria)=="CVD") saida << " Cavidade aberta\n";
    else if (std::string(Geometria)=="Cvv") saida << " Canto vivo\n";
    else if (std::string(Geometria)=="Cam") saida << " Canto amortecido\n";
    saida << "  Condicao lateral =";
    if (std::string(CondLater)=="prd") saida << " Condicoes laterais periodicas\n";
    if (std::string(CondLater)=="dir") saida << " Condicoes laterais de Dirichlet para velocidades (parede)\n";
    saida << "  Re = "      << Re       << "\n";
    saida << "  II = " << II << "   JJ = " << JJ << "   KK = " << KK << "\n";
    saida << "  numero de celulas = " << celulas << "\n";
    saida << "  dt = " << dt << "\n";
    saida << "  Eps = " << Eps << "\n\n";
}

// =============================================================================
// PROCEDURE CoCOEF  (UNIFAES advection-viscous coefficients)
// =============================================================================
void CoCOEF()
{
    int KKfim = (std::string(CondLater) == "dir") ? KKm : KK;
    double hpi = 2.0 * std::atan(1.0);
    int IIorig  = II - 5 * IIhiperviscoso / 8;
    int IIamp   = 3  * IIhiperviscoso / 8;
    double ZReOrig = 0.5 * (1.0/ReHiper + 1.0/Re);
    double ZReAmp  = 0.5 * (-1.0/ReHiper + 1.0/Re);

    // Temporary coefficient vectors (dynamic).
    // Logical index range -1..N. Physical size N+2, offset +1.
    // We use the maximum of II, JJ, KK to ensure enough space.
    int maxDim = std::max({II, JJ, KK});
    int vetSize = maxDim + 2;
    std::vector<double> ppie_v(vetSize, 0.0), ppiw_v(vetSize, 0.0);
    std::vector<double> ppin_v(vetSize, 0.0), ppis_v(vetSize, 0.0);
    std::vector<double> ppiu_v(vetSize, 0.0), ppid_v(vetSize, 0.0);
    std::vector<double> qsie_v(vetSize, 0.0), qsin_v(vetSize, 0.0), qsiu_v(vetSize, 0.0);
    std::vector<double> Ku_v(vetSize, 0.0),   Kv_v(vetSize, 0.0),   Kw_v(vetSize, 0.0);

    // Convenience references using the VM offset accessor
    auto& ppie = ppie_v; auto& ppiw = ppiw_v;
    auto& ppin = ppin_v; auto& ppis = ppis_v;
    auto& ppiu = ppiu_v; auto& ppid = ppid_v;
    auto& qsie = qsie_v; auto& qsin = qsin_v; auto& qsiu = qsiu_v;
    auto& Ku = Ku_v; auto& Kv = Kv_v; auto& Kw = Kw_v;

    // ============ Direction X ============
    double Zdxq = 1.0 / dxq;
    for (int j = 1; j <= JJm; ++j) {
        int icmc = IIini[j];
        int ifim = IIfim[j];
        for (int k = 1; k <= KKfim; ++k) {
            for (int i = icmc; i <= ifim - 1; ++i) {
                double ZReloc;
                if (i <= II - IIhiperviscoso)       ZReloc = 1.0/Re;
                else if (i <= II - IIhiperviscoso/4) ZReloc = ZReOrig - ZReAmp * std::sin(((i + 0.5 - IIorig) / IIamp) * hpi);
                else                                  ZReloc = 1.0/ReHiper;
                double Reloc = 1.0 / ZReloc;
                int ip = i + 1;
                double Reudx = 0.5 * Reloc * dx * (F3(u, ip, j, k) + F3(u, i, j, k));
                double pip_val, cip_val, cim_val;
                PiReg(Reloc, Reudx, pip_val, cip_val, cim_val);
                VM(ppie, i+1) = cip_val;
                VM(ppiw, ip+1) = cim_val;
                VM(qsie, i+1) = qsi_func(Reudx, pip_val, 0.5);
            }
            for (int i = IIini[j]+1; i <= IIfim[j]-1; ++i) {
                int ip = i+1, im = i-1;
                F3(Au, i, j, k) = (VM(ppie,i+1)*(F3(u,ip,j,k)-F3(u,i,j,k)) + VM(ppiw,i+1)*(F3(u,im,j,k)-F3(u,i,j,k))) * Zdxq;
                F3(Av, i, j, k) = (VM(ppie,i+1)*(F3(v,ip,j,k)-F3(v,i,j,k)) + VM(ppiw,i+1)*(F3(v,im,j,k)-F3(v,i,j,k))) * Zdxq;
                F3(Aw, i, j, k) = (VM(ppie,i+1)*(F3(w,ip,j,k)-F3(w,i,j,k)) + VM(ppiw,i+1)*(F3(w,im,j,k)-F3(w,i,j,k))) * Zdxq;
            }
            for (int i = icmc+1; i <= ifim-1; ++i) {
                double ZReloc;
                if (i <= II - IIhiperviscoso)       ZReloc = 1.0/Re;
                else if (i <  II - IIhiperviscoso/4) ZReloc = ZReOrig - ZReAmp * std::sin(((i - IIorig) / (double)IIamp) * hpi);
                else                                  ZReloc = 1.0/ReHiper;
                double Reloc = 1.0 / ZReloc;
                int ip = i+1, im = i-1;
                double Reu = Reloc * F3(u, i, j, k) * dx;
                double pip_val, cie, ciw;
                PiReg(Reloc, Reu, pip_val, cie, ciw);
                cie *= Zdxq;  ciw *= Zdxq;
                VM(Ku, i+1) = cie*(F3(u,i,j,k)-F3(u,ip,j,k)) + ciw*(F3(u,i,j,k)-F3(u,im,j,k));
                VM(Kv, i+1) = cie*(F3(v,i,j,k)-F3(v,ip,j,k)) + ciw*(F3(v,i,j,k)-F3(v,im,j,k));
                VM(Kw, i+1) = cie*(F3(w,i,j,k)-F3(w,ip,j,k)) + ciw*(F3(w,i,j,k)-F3(w,im,j,k));
            }
            VM(Ku, icmc+1) = 2.0*VM(Ku, icmc+2) - VM(Ku, icmc+3);
            VM(Kv, icmc+1) = 2.0*VM(Kv, icmc+2) - VM(Kv, icmc+3);
            VM(Kw, icmc+1) = 2.0*VM(Kw, icmc+2) - VM(Kw, icmc+3);
            VM(Ku, ifim+1) = 2.0*VM(Ku, ifim)   - VM(Ku, ifim-1);
            VM(Kv, ifim+1) = 2.0*VM(Kv, ifim)   - VM(Kv, ifim-1);
            VM(Kw, ifim+1) = 2.0*VM(Kw, ifim)   - VM(Kw, ifim-1);
            for (int i = icmc; i <= ifim-1; ++i) {
                int ip = i+1;
                VM(Ku, i+1) = 0.5*(VM(Ku, i+1) + VM(Ku, ip+1));
                VM(Kv, i+1) = 0.5*(VM(Kv, i+1) + VM(Kv, ip+1));
                VM(Kw, i+1) = 0.5*(VM(Kw, i+1) + VM(Kw, ip+1));
            }
            for (int i = icmc+1; i <= ifim-1; ++i) {
                int im = i-1;
                F3(Au, i, j, k) -= (VM(Ku,i+1)*VM(qsie,i+1) - VM(Ku,im+1)*VM(qsie,im+1));
                F3(Av, i, j, k) -= (VM(Kv,i+1)*VM(qsie,i+1) - VM(Kv,im+1)*VM(qsie,im+1));
                F3(Aw, i, j, k) -= (VM(Kw,i+1)*VM(qsie,i+1) - VM(Kw,im+1)*VM(qsie,im+1));
            }
        }
    }

    // ============ Direction Y ============
    double Zdyq = 1.0 / dyq;
    for (int i = 1; i <= IIm; ++i) {
        int jcmc = JJini[i];
        int jtrm = JJfim[i];
        double ZReloc;
        if (i <= II - IIhiperviscoso)       ZReloc = 1.0/Re;
        else if (i <  II - IIhiperviscoso/4) ZReloc = ZReOrig - ZReAmp * std::sin(((i - IIorig) / (double)IIamp) * hpi);
        else                                  ZReloc = 1.0/ReHiper;
        double Reloc = 1.0 / ZReloc;

        for (int k = 1; k <= KKfim; ++k) {
            for (int j = jcmc; j <= jtrm-1; ++j) {
                int jp = j+1;
                double Revdy = 0.5 * Reloc * dy * (F3(v,i,j,k) + F3(v,i,jp,k));
                double pip_val, cin, cis;
                PiReg(Reloc, Revdy, pip_val, cin, cis);
                VM(ppin, j+1) = cin;
                VM(ppis, jp+1) = cis;
                VM(qsin, j+1) = qsi_func(Revdy, pip_val, 0.5);
            }
            for (int j = jcmc+1; j <= jtrm-1; ++j) {
                int jp = j+1, jm = j-1;
                F3(Au, i, j, k) += (VM(ppin,j+1)*(F3(u,i,jp,k)-F3(u,i,j,k)) + VM(ppis,j+1)*(F3(u,i,jm,k)-F3(u,i,j,k))) * Zdyq;
                F3(Av, i, j, k) += (VM(ppin,j+1)*(F3(v,i,jp,k)-F3(v,i,j,k)) + VM(ppis,j+1)*(F3(v,i,jm,k)-F3(v,i,j,k))) * Zdyq;
                F3(Aw, i, j, k) += (VM(ppin,j+1)*(F3(w,i,jp,k)-F3(w,i,j,k)) + VM(ppis,j+1)*(F3(w,i,jm,k)-F3(w,i,j,k))) * Zdyq;
            }
            for (int j = jcmc+1; j <= jtrm-1; ++j) {
                int jp = j+1, jm = j-1;
                double Rev = Reloc * F3(v, i, j, k) * dy;
                double pip_val, cin, cis;
                PiReg(Reloc, Rev, pip_val, cin, cis);
                cin *= Zdyq;  cis *= Zdyq;
                VM(Ku, j+1) = cin*(F3(u,i,j,k)-F3(u,i,jp,k)) + cis*(F3(u,i,j,k)-F3(u,i,jm,k));
                VM(Kv, j+1) = cin*(F3(v,i,j,k)-F3(v,i,jp,k)) + cis*(F3(v,i,j,k)-F3(v,i,jm,k));
                VM(Kw, j+1) = cin*(F3(w,i,j,k)-F3(w,i,jp,k)) + cis*(F3(w,i,j,k)-F3(w,i,jm,k));
            }
            VM(Ku, jcmc+1) = 2.0*VM(Ku, jcmc+2) - VM(Ku, jcmc+3);
            VM(Ku, jtrm+1) = 2.0*VM(Ku, jtrm)   - VM(Ku, jtrm-1);
            VM(Kv, jcmc+1) = 2.0*VM(Kv, jcmc+2) - VM(Kv, jcmc+3);
            VM(Kv, jtrm+1) = 2.0*VM(Kv, jtrm)   - VM(Kv, jtrm-1);
            VM(Kw, jcmc+1) = 2.0*VM(Kw, jcmc+2) - VM(Kw, jcmc+3);
            VM(Kw, jtrm+1) = 2.0*VM(Kw, jtrm)   - VM(Kw, jtrm-1);
            for (int j = jcmc; j <= jtrm-1; ++j) {
                int jp = j+1;
                VM(Ku, j+1) = 0.5*(VM(Ku, j+1) + VM(Ku, jp+1));
                VM(Kv, j+1) = 0.5*(VM(Kv, j+1) + VM(Kv, jp+1));
                VM(Kw, j+1) = 0.5*(VM(Kw, j+1) + VM(Kw, jp+1));
            }
            for (int j = jcmc+1; j <= jtrm-1; ++j) {
                int jm = j-1;
                F3(Au, i, j, k) -= (VM(Ku,j+1)*VM(qsin,j+1) - VM(Ku,jm+1)*VM(qsin,jm+1));
                F3(Av, i, j, k) -= (VM(Kv,j+1)*VM(qsin,j+1) - VM(Kv,jm+1)*VM(qsin,jm+1));
                F3(Aw, i, j, k) -= (VM(Kw,j+1)*VM(qsin,j+1) - VM(Kw,jm+1)*VM(qsin,jm+1));
            }
        }

        // ============ Direction Z ============
        double Zdzq = 1.0 / dzq;
        for (int j = JJini[i]+1; j <= JJfim[i]-1; ++j) {
            if (std::string(CondLater) == "prd") F3(w, i, j, 0) = F3(w, i, j, KK);

            for (int k = 0; k <= KKfim; ++k) {
                int kp = (k < KK) ? k+1 : 1;
                double Rewdz = 0.5 * Reloc * dz * (F3(w,i,j,k) + F3(w,i,j,kp));
                double pip_val, ciu, cid;
                PiReg(Reloc, Rewdz, pip_val, ciu, cid);
                VM(ppiu, k+1) = ciu;
                VM(ppid, kp+1) = cid;
                VM(qsiu, k+1) = qsi_func(Rewdz, pip_val, 0.5);
            }
            for (int k = 1; k <= KKfim; ++k) {
                int kp = k+1, km = k-1;
                if (std::string(CondLater) == "prd") {
                    if (k == KK) kp = 1;
                    if (k == 1)  km = KK;
                }
                F3(Au, i, j, k) += (VM(ppiu,k+1)*(F3(u,i,j,kp)-F3(u,i,j,k)) + VM(ppid,k+1)*(F3(u,i,j,km)-F3(u,i,j,k))) * Zdzq;
                F3(Av, i, j, k) += (VM(ppiu,k+1)*(F3(v,i,j,kp)-F3(v,i,j,k)) + VM(ppid,k+1)*(F3(v,i,j,km)-F3(v,i,j,k))) * Zdzq;
                F3(Aw, i, j, k) += (VM(ppiu,k+1)*(F3(w,i,j,kp)-F3(w,i,j,k)) + VM(ppid,k+1)*(F3(w,i,j,km)-F3(w,i,j,k))) * Zdzq;
            }
            for (int k = 1; k <= KKfim; ++k) {
                int kp = k+1, km = k-1;
                if (std::string(CondLater) == "prd") {
                    if (k == KK) kp = 1;
                    if (k == 1)  km = KK;
                }
                double Rew = Reloc * F3(w, i, j, k) * dz;
                double pip_val, ciu, cid;
                PiReg(Reloc, Rew, pip_val, ciu, cid);
                ciu *= Zdzq;  cid *= Zdzq;
                VM(Ku, k+1) = ciu*(F3(u,i,j,k)-F3(u,i,j,kp)) + cid*(F3(u,i,j,k)-F3(u,i,j,km));
                VM(Kv, k+1) = ciu*(F3(v,i,j,k)-F3(v,i,j,kp)) + cid*(F3(v,i,j,k)-F3(v,i,j,km));
                VM(Kw, k+1) = ciu*(F3(w,i,j,k)-F3(w,i,j,kp)) + cid*(F3(w,i,j,k)-F3(w,i,j,km));
            }
            if (std::string(CondLater) == "dir") {
                VM(Ku, 0+1) = 2.0*VM(Ku,1+1) - VM(Ku,2+1);   VM(Ku, KK+1) = 2.0*VM(Ku,KKm+1) - VM(Ku,KK-1);
                VM(Kv, 0+1) = 2.0*VM(Kv,1+1) - VM(Kv,2+1);   VM(Kv, KK+1) = 2.0*VM(Kv,KKm+1) - VM(Kv,KK-1);
                VM(Kw, 0+1) = 2.0*VM(Kw,1+1) - VM(Kw,2+1);   VM(Kw, KK+1) = 2.0*VM(Kw,KKm+1) - VM(Kw,KK-1);
            }
            for (int k = 0; k <= KKfim; ++k) {
                int kp = (k < KK) ? k+1 : 1;
                VM(Ku, k+1) = 0.5*(VM(Ku, k+1) + VM(Ku, kp+1));
                VM(Kv, k+1) = 0.5*(VM(Kv, k+1) + VM(Kv, kp+1));
                VM(Kw, k+1) = 0.5*(VM(Kw, k+1) + VM(Kw, kp+1));
            }
            for (int k = 1; k <= KKfim; ++k) {
                int km = k-1;
                F3(Au, i, j, k) -= (VM(Ku,k+1)*VM(qsiu,k+1) - VM(Ku,km+1)*VM(qsiu,km+1));
                F3(Av, i, j, k) -= (VM(Kv,k+1)*VM(qsiu,k+1) - VM(Kv,km+1)*VM(qsiu,km+1));
                F3(Aw, i, j, k) -= (VM(Kw,k+1)*VM(qsiu,k+1) - VM(Kw,km+1)*VM(qsiu,km+1));
            }
        }
    }

    // Periodic z copy
    if (std::string(CondLater) == "prd") {
        for (int i = 1; i <= IIm; ++i)
            for (int j = JJini[i]+1; j <= JJfim[i]-1; ++j) {
                F3(Au, i, j, 0) = F3(Au, i, j, KK);
                F3(Av, i, j, 0) = F3(Av, i, j, KK);
                F3(Aw, i, j, 0) = F3(Aw, i, j, KK);
            }
    }
}

// =============================================================================
// PROCEDURE cv2Fonte  (source term for pressure equation)
// =============================================================================
void cv2Fonte()
{
    // Zero wall values of A
    for (int i = 0; i <= II; ++i) {
        int jcmc = JJini[i], jtrm = JJfim[i];
        for (int k = 0; k <= KK; ++k) {
            F3(Au, i, jcmc, k)=0.0;  F3(Au, i, jtrm, k)=0.0;
            F3(Av, i, jcmc, k)=0.0;  F3(Av, i, jtrm, k)=0.0;
            F3(Aw, i, jcmc, k)=0.0;  F3(Aw, i, jtrm, k)=0.0;
        }
    }
    for (int j = 0; j <= JJ; ++j) {
        int icmc = IIini[j], itrm = IIfim[j];
        for (int k = 0; k <= KK; ++k) {
            F3(Au, icmc, j, k)=0.0;  F3(Au, itrm, j, k)=0.0;
            F3(Av, icmc, j, k)=0.0;  F3(Av, itrm, j, k)=0.0;
            F3(Aw, icmc, j, k)=0.0;  F3(Aw, itrm, j, k)=0.0;
        }
    }

    double qZdx = 0.25/dx, qZdy = 0.25/dy, qZdz = 0.25/dz;
    double Zdt  = 1.0/dt;

    for (int i = 1; i <= II; ++i) {
        int im = i-1;
        int jcmc, jtrm_loc;
        if (i != IIdeg2) { jcmc = JJini[i]+1;    jtrm_loc = JJfim[i];   }
        else              { jcmc = JJini[im]+1;   jtrm_loc = JJfim[im];  }

        for (int j = jcmc; j <= jtrm_loc; ++j) {
            int jm = j-1;
            for (int k = 1; k <= KK; ++k) {
                int km = k-1;
                if (std::string(CondLater)=="prd" && k==1) km = KK;

                double dudx = F3(u,i,j,k)-F3(u,im,j,k)+F3(u,i,jm,k)-F3(u,im,jm,k)
                             +F3(u,i,j,km)-F3(u,im,j,km);
                dudx = (dudx + F3(u,i,jm,km) - F3(u,im,jm,km)) * qZdx;

                double dvdy = F3(v,i,j,k)-F3(v,i,jm,k)+F3(v,im,j,k)-F3(v,im,jm,k)
                             +F3(v,i,j,km)-F3(v,i,jm,km);
                dvdy = (dvdy + F3(v,im,j,km) - F3(v,im,jm,km)) * qZdy;

                double dwdz = F3(w,i,j,k)+F3(w,i,jm,k)+F3(w,im,j,k)+F3(w,im,jm,k)
                             -F3(w,i,j,km)-F3(w,i,jm,km);
                dwdz = (dwdz - F3(w,im,j,km) - F3(w,im,jm,km)) * qZdz;

                double dilij = (dudx + dvdy + dwdz) * Zdt;

                dudx = F3(Au,i,j,k)-F3(Au,im,j,k)+F3(Au,i,jm,k)-F3(Au,im,jm,k)
                      +F3(Au,i,j,km)-F3(Au,im,j,km);
                dudx = (dudx + F3(Au,i,jm,km) - F3(Au,im,jm,km)) * qZdx;

                dvdy = F3(Av,i,j,k)-F3(Av,i,jm,k)+F3(Av,im,j,k)-F3(Av,im,jm,k)
                      +F3(Av,i,j,km)-F3(Av,i,jm,km);
                dvdy = (dvdy + F3(Av,im,j,km) - F3(Av,im,jm,km)) * qZdy;

                dwdz = F3(Aw,i,j,k)+F3(Aw,i,jm,k)+F3(Aw,im,j,k)+F3(Aw,im,jm,k)
                      -F3(Aw,i,j,km)-F3(Aw,i,jm,km);
                dwdz = (dwdz - F3(Aw,im,j,km) - F3(Aw,im,jm,km)) * qZdz;

                double sij = dudx + dvdy + dwdz;
                F3(s, i, j, k) = dilij + sij;
            }
        }
    }
}

// =============================================================================
// PROCEDURE PRESSAOGaussSiedel  (Gauss-Seidel pressure solver)
// =============================================================================
void PRESSAOGaussSiedel()
{
    double cijpmk  = 1.0/dyq;
    double cipmjk  = 1.0/dxq;
    double cijkpm  = 1.0/dzq;
    double ZcijkSmt = 0.5 / (cipmjk + cijpmk + cijkpm);

    int i_ref = II;
    int j_ref = (JJfim[II] + JJini[II]) / 2;
    int k_ref = (KK + 1) / 2;
    double pref = F3(p, i_ref, j_ref, k_ref);

    for (int nitp = 1; nitp <= 5; ++nitp) {
        for (int i = 1; i <= II; ++i) {
            int im = i-1, ip = i+1;
            int jjiniloop_loc, jjfimloop_loc;

            if      (JJini[im] == JJini[i]) jjiniloop_loc = JJini[i] + 1;
            else if (JJini[im] >  JJini[i]) jjiniloop_loc = JJini[i] + 1;
            else                            jjiniloop_loc = JJini[i];

            if      (JJfim[im] == JJfim[i]) jjfimloop_loc = JJfim[i];
            else if (JJfim[im] <  JJfim[i]) jjfimloop_loc = JJfim[i];
            else                            jjfimloop_loc = JJfim[i] + 1;

            if (i == IIdeg2) {
                jjiniloop_loc = JJini[im] + 1;
                jjfimloop_loc = JJfim[im];
            }

            for (int k = 1; k <= KK; ++k) {
                int km = k-1, kp = k+1;
                for (int j = jjiniloop_loc; j <= jjfimloop_loc; ++j) {
                    int jp = j+1, jm = j-1;

                    if (i == 1 || i == IIini[j]+1) F3(p, im, j, k) = F3(p, i, j, k);
                    if (i == II || i == IIfim[j])  F3(p, ip, j, k) = F3(p, i, j, k);
                    if (j == jjiniloop_loc) F3(p, i, jm, k) = F3(p, i, j, k);
                    if (j == jjfimloop_loc) F3(p, i, jp, k) = F3(p, i, j, k);

                    if (std::string(CondLater) == "dir") {
                        if (k == 1)  F3(p, i, j, km) = F3(p, i, j, k);
                        if (k == KK) F3(p, i, j, kp) = F3(p, i, j, k);
                    } else {
                        if (k == 1)  km = KK;
                        if (k == KK) kp = 1;
                    }

                    if (i == II && j == j_ref && k == k_ref) {
                        F3(p, i, j, k) = pref;
                    } else {
                        double pijk = (cijpmk*(F3(p,i,jp,k)+F3(p,i,jm,k))
                                     + cipmjk*(F3(p,ip,j,k)+F3(p,im,j,k))
                                     + cijkpm*(F3(p,i,j,kp)+F3(p,i,j,km))
                                     - F3(s,i,j,k)) * ZcijkSmt;

                        if ((i==1||i==II) && (j==JJini[i]+1||j==JJfim[i])) {
                            pijk -= F3(s,i,j,k) * ZcijkSmt;
                            if (std::string(CondLater)=="dir" && (k==1||k==KK))
                                pijk -= 2.0 * F3(s,i,j,k) * ZcijkSmt;
                        }
                        F3(p, i, j, k) = pijk;
                    }
                }
            }
        }
    }
}

// =============================================================================
// PROCEDURE CovRESIDUO
// =============================================================================
void CovRESIDUO()
{
    for (int i = 0; i <= II; ++i)
        for (int j = 0; j <= JJ; ++j)
            for (int k = 0; k <= KK; ++k)
                F3(crng, i, j, k) = 0.0;

    double qZdx = 0.25/dx, qZdy = 0.25/dy, qZdz = 0.25/dz;
    int KKfim = (std::string(CondLater)=="dir") ? KKm : KK;

    ResidMax  = 0.0;
    contador  = 0;
    ResidRMS  = 0.0;

    for (int i = 1; i <= IIm; ++i) {
        int ip = i+1;
        for (int j = JJini[i]+1; j <= JJfim[i]-1; ++j) {
            int jp = j+1;
            for (int k = 1; k <= KKfim; ++k) {
                int kp = (k < KK) ? k+1 : 1;

                double Residu = F3(p,ip,j,k)-F3(p,i,j,k)+F3(p,ip,jp,k)-F3(p,i,jp,k)
                               +F3(p,ip,j,kp)-F3(p,i,j,kp);
                Residu = -(Residu + F3(p,ip,jp,kp) - F3(p,i,jp,kp)) * qZdx + F3(Au,i,j,k);

                double Residv = F3(p,i,jp,k)-F3(p,i,j,k)+F3(p,ip,jp,k)-F3(p,ip,j,k)
                               +F3(p,i,jp,kp)-F3(p,i,j,kp);
                Residv = -(Residv + F3(p,ip,jp,kp) - F3(p,ip,j,kp)) * qZdy + F3(Av,i,j,k);

                double Residw = -F3(p,i,jp,k)-F3(p,i,j,k)-F3(p,ip,jp,k)-F3(p,ip,j,k)
                               +F3(p,i,jp,kp)+F3(p,i,j,kp);
                Residw = -(Residw + F3(p,ip,jp,kp) + F3(p,ip,j,kp)) * qZdz + F3(Aw,i,j,k);

                double Residqq = Residu*Residu + Residv*Residv + Residw*Residw;
                double Residq  = std::sqrt(Residqq);
                F3(crng, i, j, k) = Residq;

                if (Residq > ResidMax) {
                    ResidMax  = Residq;
                    iresidmax = i;  jresidmax = j;  kresidmax = k;
                }
                ResidRMS += Residqq;
                ++contador;
            }
        }
    }
    ResidRMS = std::sqrt(ResidRMS / contador);
}

// =============================================================================
// PROCEDURE CovDILATACAO
// =============================================================================
void CovDILATACAO()
{
    double qZdx = 0.25/dx, qZdy = 0.25/dy, qZdz = 0.25/dz;

    for (int i = 0; i <= II; ++i)
        for (int j = 0; j <= JJ; ++j)
            for (int k = 0; k <= KK; ++k)
                F3(crng, i, j, k) = 0.0;

    DilMax = 0.0;  IntDil = 0.0;  IntAbsDil = 0.0;

    for (int i = 1; i <= II; ++i) {
        int im = i-1;
        int jcmc_loc, jfim_loc;
        if (i != IIdeg2) { jcmc_loc = JJini[i]+1;    jfim_loc = JJfim[i];   }
        else              { jcmc_loc = JJini[i-1]+1;  jfim_loc = JJfim[i-1]; }

        for (int j = jcmc_loc; j <= jfim_loc; ++j) {
            int jm = j-1;
            for (int k = 1; k <= KK; ++k) {
                int km = (std::string(CondLater)=="prd" && k==1) ? KK : k-1;

                double dudx = F3(u,i,j,k)-F3(u,im,j,k)+F3(u,i,jm,k)-F3(u,im,jm,k)
                             +F3(u,i,j,km)-F3(u,im,j,km);
                dudx = (dudx + F3(u,i,jm,km) - F3(u,im,jm,km)) * qZdx;

                double dvdy = F3(v,i,j,k)-F3(v,i,jm,k)+F3(v,im,j,k)-F3(v,im,jm,k)
                             +F3(v,i,j,km)-F3(v,i,jm,km);
                dvdy = (dvdy + F3(v,im,j,km) - F3(v,im,jm,km)) * qZdy;

                double dwdz = F3(w,i,j,k)+F3(w,i,jm,k)+F3(w,im,j,k)+F3(w,im,jm,k)
                             -F3(w,i,j,km)-F3(w,i,jm,km);
                dwdz = (dwdz - F3(w,im,j,km) - F3(w,im,jm,km)) * qZdz;

                double Dil = dudx + dvdy + dwdz;
                F3(crng, i, j, k) = Dil;
                IntDil += Dil;
                Dil = std::abs(Dil);
                IntAbsDil += Dil;
                if (Dil > DilMax) {
                    DilMax = Dil;
                    iDilmax = i;  jDilmax = j;  kDilmax = k;
                }
            }
        }
    }
    IntDil    *= dx*dy*dz;
    IntAbsDil *= dx*dy*dz;
}

// =============================================================================
// PROCEDURE CovVELOCDD  (velocity update)
// =============================================================================
void CovVELOCDD()
{
    double qZdx = 0.25/dx, qZdy = 0.25/dy, qZdz = 0.25/dz;
    double hdt  = (std::string(half)=="sim") ? 0.5*dt : dt;
    int KKfim   = (std::string(CondLater)=="dir") ? KKm : KK;

    dmax = 0.0;

    for (int i = 1; i <= IIm; ++i) {
        int ip = i+1;
        for (int j = JJini[i]+1; j <= JJfim[i]-1; ++j) {
            int jp = j+1;
            for (int k = 1; k <= KKfim; ++k) {
                int kp = (k < KK) ? k+1 : 1;

                double du = F3(p,ip,j,kp)-F3(p,i,j,kp)+F3(p,ip,jp,kp)-F3(p,i,jp,kp)
                           +F3(p,ip,j,k)-F3(p,i,j,k);
                du = -(du + F3(p,ip,jp,k) - F3(p,i,jp,k)) * qZdx + F3(Au,i,j,k);

                double dv = F3(p,i,jp,kp)-F3(p,i,j,kp)+F3(p,ip,jp,kp)-F3(p,ip,j,kp)
                           +F3(p,i,jp,k)-F3(p,i,j,k);
                dv = -(dv + F3(p,ip,jp,k) - F3(p,ip,j,k)) * qZdy + F3(Av,i,j,k);

                double dw = F3(p,i,jp,kp)+F3(p,i,j,kp)+F3(p,ip,jp,kp)+F3(p,ip,j,kp)
                           -F3(p,i,jp,k)-F3(p,i,j,k);
                dw = -(dw - F3(p,ip,jp,k) - F3(p,ip,j,k)) * qZdz + F3(Aw,i,j,k);

                F3(u, i, j, k) += du * hdt;
                F3(v, i, j, k) += dv * hdt;
                F3(w, i, j, k) += dw * hdt;

                double modduvw = std::sqrt(du*du + dv*dv + dw*dw);
                if (modduvw > dmax) dmax = modduvw;
            }

            if (std::string(CondLater) == "prd") {
                F3(u, i, j, 0) = F3(u, i, j, KK);
                F3(v, i, j, 0) = F3(v, i, j, KK);
                F3(w, i, j, 0) = F3(w, i, j, KK);
            }
        }
    }

    if (std::string(CondSaida) == "1d0") {
        for (int j = JJini[II]+1; j <= JJfim[II]-1; ++j)
            for (int k = 0; k <= KKfim; ++k) {
                F3(u, II, j, k) = F3(u, IIm, j, k);
                F3(v, II, j, k) = F3(v, IIm, j, k);
                F3(w, II, j, k) = F3(w, IIm, j, k);
            }
    }
    if (std::string(CondSaida) == "2d0") {
        for (int j = JJini[II]+1; j <= JJfim[II]-1; ++j)
            for (int k = 0; k <= KKfim; ++k) {
                F3(u, II, j, k) = 2.0*F3(u, IIm, j, k) - F3(u, II-2, j, k);
                F3(v, II, j, k) = 2.0*F3(v, IIm, j, k) - F3(v, II-2, j, k);
                F3(w, II, j, k) = 2.0*F3(w, IIm, j, k) - F3(w, II-2, j, k);
            }
    }
}

// =============================================================================
// PROCEDURE DeltaT
// =============================================================================
void DeltaT()
{
    double umax_loc = 0.0, vmax_loc = 0.0, wmax_loc = 0.0;

    for (int i = 1; i <= IIm; ++i)
        for (int j = JJini[i]+1; j <= JJfim[i]-1; ++j)
            for (int k = 1; k <= KK; ++k) {
                if (std::abs(F3(u,i,j,k)) > umax_loc) umax_loc = std::abs(F3(u,i,j,k));
                if (std::abs(F3(v,i,j,k)) > vmax_loc) vmax_loc = std::abs(F3(u,i,j,k));  // mirrors Pascal bug
                if (std::abs(F3(w,i,j,k)) > wmax_loc) wmax_loc = std::abs(F3(u,i,j,k));  // mirrors Pascal bug
            }

    double dtcond = (IIhiperviscoso == 0)
        ? 0.5 * Re    / (1.0/dxq + 1.0/dyq + 1.0/dzq)
        : 0.5 * ReHiper / (1.0/dxq + 1.0/dyq + 1.0/dzq);

    double dtadvc = (dx/umax_loc < dy/vmax_loc) ? dx/umax_loc : dy/vmax_loc;
    if (dz/wmax_loc < dtadvc) dtadvc = dz/wmax_loc;

    dt = (dtcond < dtadvc) ? dtcond : dtadvc;
    dt *= 0.35;
}

// =============================================================================
// PROCEDURE MOSTRE  (block-column formatted 2D field printer)
// =============================================================================
static void MOSTRE(const char* xxx, std::vector<double>& apresentacao,
                   int kmostra_loc,
                   int Iinicio, int Jinicio, int Itermina, int Jtermina)
{
    const int larg  = 1800;
    const int comp  = 10;
    const int mant  = 4;
    const int icent = comp - 3;
    const int ncol  = (larg - 6) / (comp + 1);

    int IIif  = Itermina - Iinicio + 1;
    int nbloc = IIif / ncol;
    if (IIif % ncol > 0) ++nbloc;

    saida << xxx << "  kmostra=  " << kmostra_loc << "\n";

    int inic = Iinicio;
    for (int n = 1; n <= nbloc; ++n) {
        int fim = inic + ncol - 1;
        if (Itermina < fim) fim = Itermina;

        saida << "\n";
        saida << "j\\i";
        for (int i = inic; i <= fim; ++i) {
            saida.width(icent); saida << i;
            saida << "    ";
        }
        saida << "i/j\n";

        for (int j = Jtermina; j >= Jinicio; --j) {
            saida.width(2); saida << j;
            for (int i = inic; i <= fim; ++i) {
                bool inDomain = (j >= JJini[i] && j <= JJfim[i]);
                bool isDeg1   = (i == IIdeg1 && IIini[j] == IIdeg1);
                bool isDeg2   = (i == IIdeg2 && IIfim[j] == IIdeg2);
                if (inDomain || isDeg1 || isDeg2) {
                    saida << " ";
                    saida.width(comp);
                    saida.precision(mant);
                    saida << std::fixed << F2(apresentacao, i, j);
                } else {
                    saida << "     ----- ";
                }
            }
            saida << "  "; saida.width(2); saida << j; saida << "\n";
        }
        inic += ncol;
    }
    saida << std::defaultfloat;
}

// =============================================================================
// PROCEDURE MostraPerfiseFluxosI
// =============================================================================
static void MostraPerfiseFluxosI(int imostra)
{
    int KKfim = (std::string(CondLater) == "prd") ? KK : KKm;
    int k     = (KK + 1) / 2;
    int i     = imostra;

    saida << " i= " << i << "\n";
    saida << " j =     u[i,j,k]      v[i,j,k]      w[i,j,k]      p[i,j,k]\n";

    if (JJ % 60 != 0) {
        for (int j = JJini[i]; j <= JJfim[i]; ++j)
            if ((j * 60) % JJ == 0)
                saida << std::fixed << std::setprecision(4)
                      << std::setw(10) << F3(u, i, j, k) << "\n";
    } else {
        double Z60  = 1.0 / 60.0;
        int    j60  = JJini[i] * 60 / JJ;
        for (int j = JJini[i]; j <= JJfim[i]; ++j) {
            int    jp      = j + 1;
            double y       = (double)j  / JJ;
            double yp      = (double)jp / JJ;
            double j60Z60  = j60 * Z60;
            if (y <= j60Z60 && yp > j60Z60) {
                double uintrpld = F3(u, i, j, k)
                    + (F3(u, i, jp, k) - F3(u, i, j, k)) * (j60Z60 - y) / dy;
                saida << std::fixed << std::setprecision(4)
                      << std::setw(10) << uintrpld << "\n";
                ++j60;
            }
        }
    }

    double flux = 0.0;
    for (int j = JJini[i]; j <= JJfim[i]; ++j)
        for (int k2 = 1; k2 <= KKfim; ++k2)
            flux += F3(u, i, j, k2) * dy * dz;
    saida << " i= " << i << "     fluxo= " << flux << "\n\n";
    saida << std::defaultfloat;
}

// =============================================================================
// PROCEDURE MostraPerfiseFluxosJ
// =============================================================================
static void MostraPerfiseFluxosJ(int jmostra)
{
    int KKfim = (std::string(CondLater) == "prd") ? KK : KKm;
    int k     = (KK + 1) / 2;
    int j     = jmostra;

    saida << " j= " << j << "\n";
    saida << " i =     u[i,j,k]      v[i,j,k]      w[i,j,k]      p[i,j,k]\n";

    for (int i = IIini[j]; i <= IIfim[j]; ++i)
        saida << std::setw(4) << i
              << "    " << std::fixed << std::setprecision(4)
              << std::setw(10) << F3(u, i, j, k)
              << "    " << std::setw(10) << F3(v, i, j, k)
              << "    " << std::setw(10) << F3(w, i, j, k)
              << "    " << std::setw(10) << F3(p, i, j, k) << "\n";

    double flux = 0.0;
    for (int i = IIini[j]; i <= IIfim[j]; ++i)
        for (int k2 = 1; k2 <= KKfim; ++k2)
            flux += F3(v, i, j, k2) * dx * dz;
    saida << " j= " << j << "     fluxo= " << flux << "\n\n";
    saida << std::defaultfloat;
}

// =============================================================================
// PROCEDURE CovRESULTADOS
// =============================================================================
void CovRESULTADOS()
{
    std::vector<double> apresentacao = alloc2D();

    int zero = 0, um_loc = 1;
    kmostra = (1 + KK) / 2;

    // Reset crng
    for (int i = 0; i <= II; ++i)
        for (int j = 0; j <= JJ; ++j)
            for (int k = 0; k <= KK; ++k)
                F3(crng, i, j, k) = 0.0;

    saida << "\n   nt =   " << std::setw(5) << nt
          << "   T =  " << std::fixed << std::setprecision(2) << T << "\n";
    saida << std::defaultfloat;

    std::cout << "\n PLANOS LONGITUDINAIS \n";
    saida     << " PLANOS LONGITUDINAIS \n";

    // --- u field ---
    for (int i = 0; i <= II; ++i)
        for (int j = 0; j <= JJ; ++j)
            F2(apresentacao, i, j) = F3(u, i, j, kmostra);
    MOSTRE(" u ", apresentacao, kmostra, zero, zero, II, JJ);

    // --- v field ---
    for (int i = 0; i <= II; ++i)
        for (int j = 0; j <= JJ; ++j)
            F2(apresentacao, i, j) = F3(v, i, j, kmostra);
    MOSTRE(" v ", apresentacao, kmostra, zero, zero, II, JJ);

    // --- w field ---
    for (int i = 0; i <= II; ++i)
        for (int j = 0; j <= JJ; ++j)
            F2(apresentacao, i, j) = F3(w, i, j, kmostra);
    MOSTRE(" w ", apresentacao, kmostra, zero, zero, II, JJ);

    // --- p field ---
    for (int i = 1; i <= II; ++i)
        for (int j = 1; j <= JJ; ++j)
            F2(apresentacao, i, j) = F3(p, i, j, kmostra);
    MOSTRE(" p ", apresentacao, kmostra, um_loc, um_loc, II, JJ);

    // --- source term s ---
    cv2Fonte();
    for (int i = 1; i <= II; ++i)
        for (int j = 1; j <= JJ; ++j)
            F2(apresentacao, i, j) = F3(s, i, j, kmostra);
    MOSTRE(" s ", apresentacao, kmostra, um_loc, um_loc, II, JJ);

    // --- Au, Av, Aw ---
    for (int i = 1; i <= II; ++i)
        for (int j = 1; j <= JJ; ++j)
            F2(apresentacao, i, j) = F3(Au, i, j, kmostra);
    MOSTRE("Au ", apresentacao, kmostra, um_loc, um_loc, II, JJ);

    for (int i = 0; i <= II; ++i)
        for (int j = 0; j <= JJ; ++j)
            F2(apresentacao, i, j) = F3(Av, i, j, kmostra);
    MOSTRE(" Av", apresentacao, kmostra, zero, zero, II, JJ);

    for (int i = 0; i <= II; ++i)
        for (int j = 0; j <= JJ; ++j)
            F2(apresentacao, i, j) = F3(Aw, i, j, kmostra);
    MOSTRE(" Aw", apresentacao, kmostra, zero, zero, II, JJ);

    // --- Momentum residue ---
    CovRESIDUO();
    saida << " ResidMax=  " << ResidMax
          << " ResidRMS=  " << ResidRMS << "\n";
    saida << " iresidmax=  " << iresidmax
          << " jresidmax=  " << jresidmax
          << " kresidmax=  " << kresidmax << "\n";
    if (kresidmax > 0 && kresidmax <= KK) kmostra = kresidmax;
    for (int i = 1; i <= II; ++i)
        for (int j = 1; j <= JJ; ++j)
            F2(apresentacao, i, j) = F3(crng, i, j, kmostra);
    MOSTRE("res", apresentacao, kmostra, um_loc, um_loc, II, JJ);

    // --- Dilatation ---
    CovDILATACAO();
    saida << " Dilmax= "    << DilMax
          << " IntDil=  "   << IntDil
          << " IntAbsDil=  " << IntAbsDil << "\n";
    saida << " iDilmax= " << iDilmax
          << " jDilmax=  " << jDilmax
          << " kDilmax=  " << kDilmax << "\n";
    if (kDilmax > 0 && kDilmax <= KK) kmostra = kDilmax;
    for (int i = 1; i <= II; ++i)
        for (int j = 1; j <= JJ; ++j)
            F2(apresentacao, i, j) = F3(crng, i, j, kmostra);
    MOSTRE("di2", apresentacao, kDilmax, um_loc, um_loc, II, JJ);

    // --- Profile sections ---
    if (std::string(TipoGeometria) == "Axl") {
        int imostra = 0;
        MostraPerfiseFluxosI(imostra);
        if (IIdeg1 > 0) {
            imostra = IIdeg1;
            MostraPerfiseFluxosI(imostra);
        }
        if (IIdeg1 > 0 && IIdeg2 < II) {
            imostra = (IIdeg1 + IIdeg2) / 2;
            MostraPerfiseFluxosI(imostra);
        }
    } else {
        int jmostra = 0;
        MostraPerfiseFluxosJ(jmostra);
        if (JJdeg < JJ) {
            jmostra = JJdeg;
            MostraPerfiseFluxosJ(jmostra);
        }
    }

    if (IIdeg2 < II) {
        int imostra = IIdeg2;
        MostraPerfiseFluxosI(imostra);
        imostra = (IIdeg2 + II) / 2;
        MostraPerfiseFluxosI(imostra);
    }

    if (IIrmp1 > 0) {
        int imostra = IIrmp1;
        MostraPerfiseFluxosI(imostra);
    }
    if (IIrmp1 > 0 && IIrmp2 < II) {
        int imostra = (IIrmp1 + IIrmp2) / 2;
        MostraPerfiseFluxosI(imostra);
    }
    if (IIrmp2 < II) {
        int imostra = IIrmp2;
        MostraPerfiseFluxosI(imostra);
        imostra = (IIrmp2 + II) / 2;
        MostraPerfiseFluxosI(imostra);
    }

    int imostra = II;
    MostraPerfiseFluxosI(imostra);

    // --- Wall derivatives ---
    int k = (KK + 1) / 2;
    saida << " Derivadas nas superficies solidas no plano k= " << k << "\n";

    // derivUini  (lower wall, j = JJini[i])
    saida << "  derivUini\n";
    for (int i = 0; i <= II; ++i) {
        int j   = JJini[i];
        int jp  = j + 1, jp2 = j + 2;
        if (jp2 <= JJ) {
            double derivUini = (4.0*F3(u,i,jp,k) - F3(u,i,jp2,k)) / (2.0*dy);
            if ((i * 60) % NN == 0)
                saida << std::fixed << std::setprecision(6) << derivUini << "\n";
        }
    }

    // derivUfim  (upper wall, j = JJfim[i])
    saida << "  derivUfim\n";
    for (int i = 0; i <= II; ++i) {
        int j   = JJfim[i];
        int jm  = j - 1, jm2 = j - 2;
        if (jm2 >= 0) {
            double derivUfim = (-4.0*F3(u,i,jm,k) + F3(u,i,jm2,k)) / (2.0*dy);
            if ((i * 60) % NN == 0)
                saida << std::fixed << std::setprecision(6) << derivUfim << "\n";
        }
    }

    // Vertical plane derivatives at IIdeg1
    if (IIdeg1 > 0 && IIdeg1 < II) {
        int i   = IIdeg1;
        int ip  = i + 1, ip2 = i + 2;
        if (ip2 <= II) {
            saida << " Plano vertical  i=IIdeg1=" << std::setw(3) << i << "\n";
            for (int j = 1; j <= JJ; ++j) {
                if (F3(u, i, j, k) == 0.0) {
                    double derivVdeg1 = (4.0*F3(v,ip,j,k) - F3(v,ip2,j,k)) / (2.0*dx);
                    if ((i * 60) % NN == 0)
                        saida << std::fixed << std::setprecision(6) << derivVdeg1 << "\n";
                }
            }
        }
    }

    // Vertical plane derivatives at IIdeg2
    if (IIdeg2 > 1 && IIdeg2 <= II) {
        int i   = IIdeg2;
        int im  = i - 1, im2 = i - 2;
        if (im2 >= 0) {
            saida << " Plano vertical   i=IIdeg2=" << std::setw(3) << i << "\n";
            for (int j = 1; j <= JJ; ++j) {
                if (F3(u, i, j, k) == 0.0) {
                    double derivVdeg2 = (-4.0*F3(v,im,j,k) + F3(v,im2,j,k)) / (2.0*dx);
                    if ((i * 60) % NN == 0)
                        saida << std::fixed << std::setprecision(6) << derivVdeg2 << "\n";
                }
            }
        }
    }

    saida << std::defaultfloat;
}

// =============================================================================
// MAIN
// =============================================================================
int main()
{
    saida.open("harwelEabRe1000m120p0.txt");
    if (!saida.is_open()) {
        std::cerr << "Cannot open output file.\n";
        return 1;
    }

    // ---  Problem definition parameters ---
    strncpy(TipoGeometria, "Crv", 4);
    strncpy(Geometria,     "Cam", 4);
    strncpy(PerfilInicial, "LCP", 4);
    strncpy(CondSaida,     "1d0", 4);
    strncpy(CondLater,     "prd", 4);

    Re      = 100.0;
    ReHiper = 20.0;

    Cmp = 6.0;   Alt = 2.0;   Lrg = 1.0;

    NN = 20;
    II = 6 * NN;   JJ = 2 * NN;   KK = NN;
    dx = Cmp / II; dy = Alt / JJ;  dz = Lrg / KK;

    Eps      = 0.000001;
    ntmax    = 10000;
    ntmostra = 1000000;
    IIhiperviscoso = 0;

    itp = 0;

    T = 0.0;  nt = 0;

    // -------------------------------------------------------------------------
    // Fully dynamic allocation of all arrays
    // -------------------------------------------------------------------------
    g_dimI = II + 1;
    g_dimJ = JJ + 1;
    g_dimK = KK + 1;

    // 1-D index arrays
    int maxIdx = std::max(II, JJ);
    IIini.assign(maxIdx + 1, 0);
    IIfim.assign(maxIdx + 1, 0);
    JJini.assign(maxIdx + 1, 0);
    JJfim.assign(maxIdx + 1, 0);

    // 3-D field arrays
    u    = alloc3D();
    v    = alloc3D();
    w    = alloc3D();
    p    = alloc3D();
    Au   = alloc3D();
    Av   = alloc3D();
    Aw   = alloc3D();
    s    = alloc3D();
    crng = alloc3D();
    // -------------------------------------------------------------------------

    // --- Initialise ---
    INIT();
    CoCOEF();
    CovRESULTADOS();

    // =========================================================================
    // PERMANENT REGIMEN (itp == 0)
    // =========================================================================
    if (itp == 0) {
        strncpy(half, "nao", 4);

        do {
            DeltaT();
            ++nt;  T += dt;

            cv2Fonte();
            PRESSAOGaussSiedel();
            CovVELOCDD();
            CoCOEF();
            CovRESIDUO();
            CovDILATACAO();

            std::cout << " nt=" << std::setw(4) << nt
                      << "  T=" << std::fixed << std::setprecision(4) << T
                      << "   ResidMax = " << ResidMax
                      << "   DilMax = "   << DilMax  << "\n";
            saida << " nt=" << std::setw(4) << nt
                  << "  T=" << std::fixed << std::setprecision(4) << T
                  << "   ResidMax = " << ResidMax
                  << "   ResidRMS = " << ResidRMS
                  << "   DilMax = "   << DilMax
                  << "   IntDil = "   << IntDil
                  << "   IntAbsDil = " << IntAbsDil << "\n";
            saida << std::defaultfloat;

            if (nt == ntmostra)
                CovRESULTADOS();

        } while (ResidMax >= Eps && nt < ntmax);
    }

    // =========================================================================
    // TRANSITORY REGIMEN (itp == 1)
    // =========================================================================
    else {
        std::vector<double> u0 = alloc3D(), v0 = alloc3D(), w0 = alloc3D();
        std::vector<double> Ku = alloc3D(), Kv = alloc3D(), Kw = alloc3D();

        do {
            DeltaT();
            ++nt;  T += dt;

            for (int i = 0; i <= II; ++i)
                for (int j = 0; j <= JJ; ++j)
                    for (int k = 0; k <= KK; ++k) {
                        F3(u0, i, j, k) = F3(u, i, j, k);
                        F3(v0, i, j, k) = F3(v, i, j, k);
                        F3(w0, i, j, k) = F3(w, i, j, k);
                    }

            for (int i = 0; i <= II; ++i)
                for (int j = 0; j <= JJ; ++j)
                    for (int k = 0; k <= KK; ++k)
                        F3(Ku,i,j,k) = F3(Kv,i,j,k) = F3(Kw,i,j,k) = 0.0;

            const double weights[4] = {1.0, 2.0, 2.0, 1.0};

            for (int rk = 1; rk <= 4; ++rk) {
                for (int i = 0; i <= II; ++i)
                    for (int j = 0; j <= JJ; ++j)
                        for (int k = 0; k <= KK; ++k) {
                            F3(u, i, j, k) = F3(u0, i, j, k);
                            F3(v, i, j, k) = F3(v0, i, j, k);
                            F3(w, i, j, k) = F3(w0, i, j, k);
                        }

                if (rk <= 2) strncpy(half, "sim", 4);
                else         strncpy(half, "nao", 4);

                cv2Fonte();
                PRESSAOGaussSiedel();
                CovVELOCDD();

                for (int i = 1; i <= IIm; ++i)
                    for (int j = JJini[i]+1; j <= JJfim[i]-1; ++j)
                        for (int k = 1; k <= KK; ++k) {
                            F3(Ku,i,j,k) += weights[rk-1] * (F3(u,i,j,k) - F3(u0,i,j,k));
                            F3(Kv,i,j,k) += weights[rk-1] * (F3(v,i,j,k) - F3(v0,i,j,k));
                            F3(Kw,i,j,k) += weights[rk-1] * (F3(w,i,j,k) - F3(w0,i,j,k));
                        }

                CoCOEF();
            }

            for (int i = 1; i <= IIm; ++i)
                for (int j = JJini[i]+1; j <= JJfim[i]-1; ++j)
                    for (int k = 1; k <= KK; ++k) {
                        F3(u, i, j, k) = F3(u0, i, j, k) + F3(Ku, i, j, k) / 6.0;
                        F3(v, i, j, k) = F3(v0, i, j, k) + F3(Kv, i, j, k) / 6.0;
                        F3(w, i, j, k) = F3(w0, i, j, k) + F3(Kw, i, j, k) / 6.0;
                    }

            strncpy(half, "nao", 4);
            int KKfim = (std::string(CondLater)=="dir") ? KKm : KK;
            if (std::string(CondSaida) == "1d0")
                for (int j = JJini[II]+1; j <= JJfim[II]-1; ++j)
                    for (int k = 0; k <= KKfim; ++k) {
                        F3(u, II, j, k) = F3(u, IIm, j, k);
                        F3(v, II, j, k) = F3(v, IIm, j, k);
                        F3(w, II, j, k) = F3(w, IIm, j, k);
                    }
            if (std::string(CondLater) == "prd")
                for (int i = 1; i <= IIm; ++i)
                    for (int j = JJini[i]+1; j <= JJfim[i]-1; ++j) {
                        F3(u, i, j, 0) = F3(u, i, j, KK);
                        F3(v, i, j, 0) = F3(v, i, j, KK);
                        F3(w, i, j, 0) = F3(w, i, j, KK);
                    }

            CoCOEF();
            CovRESIDUO();
            CovDILATACAO();

            std::cout << " nt=" << std::setw(4) << nt
                      << "  T=" << std::fixed << std::setprecision(4) << T
                      << "   ResidMax = " << ResidMax
                      << "   DilMax = "   << DilMax  << "\n";
            saida << " nt=" << std::setw(4) << nt
                  << "  T=" << std::fixed << std::setprecision(4) << T
                  << "   ResidMax = " << ResidMax
                  << "   ResidRMS = " << ResidRMS
                  << "   DilMax = "   << DilMax
                  << "   IntDil = "   << IntDil
                  << "   IntAbsDil = " << IntAbsDil << "\n";
            saida << std::defaultfloat;

            if (nt == ntmostra)
                CovRESULTADOS();

        } while (ResidMax >= Eps && nt < ntmax);
    }

    CovRESULTADOS();
    saida.close();

    return 0;
}

