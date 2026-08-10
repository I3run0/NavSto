// =============================================================================
//  HostBackend.cpp — Backend.hpp for any backend that computes in host memory.
//
//  Not a backend itself: serial and openmp both link this, since neither has
//  anything to upload, download, or launch. A backend whose fields do not live
//  in host memory writes its own (see src/backends/cuda/CudaBackend.cu).
// =============================================================================

#include "Backend.hpp"

const char* backendName() { return kBackendName; }

void backendStartup(SimState& s)
{
    if (s.cfg.flowType == FlowType::RK4Transient)
        s.ext.rk4.allocate(s.g);
}

void backendShutdown(SimState&) {}

/// Fields are already host memory — the driver's snapshot path needs nothing.
void syncFieldsToHost(SimState&) {}

// ---------------------------------------------------------------------------
//  RK4 bookkeeping
//
//  Save/restore cover the whole padded grid; accumulate/combine cover the
//  active domain only, matching updateVelocities' bounds.
// ---------------------------------------------------------------------------
namespace {
int lastActiveK(const SimState& s)
{
    return (s.cfg.lateralCondition == LateralBC::SolidWall)
         ? s.numCellsZm1 : s.cfg.numCellsZ;
}
}  // namespace

void rk4Save(SimState& s)
{
    auto& w = s.ext.rk4;
    for (int i = 0; i <= s.cfg.numCellsX; ++i)
        for (int j = 0; j <= s.cfg.numCellsY; ++j)
            for (int k = 0; k <= s.cfg.numCellsZ; ++k) {
                w.velX0(i,j,k) = s.velX(i,j,k);
                w.velY0(i,j,k) = s.velY(i,j,k);
                w.velZ0(i,j,k) = s.velZ(i,j,k);
                w.Ku(i,j,k)    = 0.0;
                w.Kv(i,j,k)    = 0.0;
                w.Kw(i,j,k)    = 0.0;
            }
}

void rk4Restore(SimState& s)
{
    const auto& w = s.ext.rk4;
    for (int i = 0; i <= s.cfg.numCellsX; ++i)
        for (int j = 0; j <= s.cfg.numCellsY; ++j)
            for (int k = 0; k <= s.cfg.numCellsZ; ++k) {
                s.velX(i,j,k) = w.velX0(i,j,k);
                s.velY(i,j,k) = w.velY0(i,j,k);
                s.velZ(i,j,k) = w.velZ0(i,j,k);
            }
}

void rk4Accumulate(SimState& s, double weight)
{
    auto& w = s.ext.rk4;
    const int KKfim = lastActiveK(s);
    for (int i = 1; i <= s.numCellsXm1; ++i)
        for (int j = s.jLow[i]+1; j <= s.jHigh[i]-1; ++j)
            for (int k = 1; k <= KKfim; ++k) {
                w.Ku(i,j,k) += weight * (s.velX(i,j,k) - w.velX0(i,j,k));
                w.Kv(i,j,k) += weight * (s.velY(i,j,k) - w.velY0(i,j,k));
                w.Kw(i,j,k) += weight * (s.velZ(i,j,k) - w.velZ0(i,j,k));
            }
}

void rk4Combine(SimState& s)
{
    const auto& w = s.ext.rk4;
    const int KKfim = lastActiveK(s);
    for (int i = 1; i <= s.numCellsXm1; ++i)
        for (int j = s.jLow[i]+1; j <= s.jHigh[i]-1; ++j)
            for (int k = 1; k <= KKfim; ++k) {
                s.velX(i,j,k) = w.velX0(i,j,k) + w.Ku(i,j,k) / 6.0;
                s.velY(i,j,k) = w.velY0(i,j,k) + w.Kv(i,j,k) / 6.0;
                s.velZ(i,j,k) = w.velZ0(i,j,k) + w.Kw(i,j,k) / 6.0;
            }
}

void rk4ApplyFinalBCs(SimState& s)
{
    const int NX = s.cfg.numCellsX;
    const int KKfim = lastActiveK(s);

    if (s.cfg.outletCondition == OutletBC::ZeroFirstDeriv)
        for (int j = s.jLow[NX]+1; j <= s.jHigh[NX]-1; ++j)
            for (int k = 0; k <= KKfim; ++k) {
                s.velX(NX,j,k) = s.velX(s.numCellsXm1,j,k);
                s.velY(NX,j,k) = s.velY(s.numCellsXm1,j,k);
                s.velZ(NX,j,k) = s.velZ(s.numCellsXm1,j,k);
            }

    if (s.cfg.lateralCondition == LateralBC::Periodic)
        for (int i = 1; i <= s.numCellsXm1; ++i)
            for (int j = s.jLow[i]+1; j <= s.jHigh[i]-1; ++j) {
                s.velX(i,j,0) = s.velX(i,j,s.cfg.numCellsZ);
                s.velY(i,j,0) = s.velY(i,j,s.cfg.numCellsZ);
                s.velZ(i,j,0) = s.velZ(i,j,s.cfg.numCellsZ);
            }
}
