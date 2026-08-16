#pragma once
// =============================================================================
//  Physics.hpp — the operators every backend must define.
//
//  Each takes SimState& and nothing else: SimState already carries this
//  backend's storage type and private working memory (`s.ext`), both picked in
//  its own BackendConfig.hpp, so retuning a backend changes no signature here.
//  Backend.hpp covers what the driver needs around these.
// =============================================================================

#include "SimState.hpp"

// ── Initialisation ──────────────────────────────────────────────────────────

/// Set up geometry arrays (jLow/jHigh/iLow/iHigh), fill initial velocity and
/// pressure fields, and compute the first stable time step.
void initSimulation(SimState& s);

// ── Per-step operators ──────────────────────────────────────────────────────

/// Recompute UNIFAES advective + viscous accelerations (accelX/Y/Z).
/// Uses this backend's coefficient scratch (`s.ext`).
void computeAccelerations(SimState& s);

/// Build the RHS of the pressure Poisson equation: S = ∇·u/dt + ∇·A.
void buildPressureSource(SimState& s);

/// Solve ∇²p = S iteratively. Parallel backends drive the sweep from a
/// precomputed index list in `s.ext`; the serial Gauss-Seidel implementation
/// has a loop-carried dependency and keeps no such list at all.
void solvePressurePoisson(SimState& s);

/// Projection step: u_new = u_old + dt*(A - ∇p).
void updateVelocities(SimState& s);

// ── Diagnostics ─────────────────────────────────────────────────────────────

/// Compute L∞ and L² norms of the momentum residual.
void computeMomentumResidual(SimState& s);

/// Compute the velocity divergence (mass conservation check).
void computeDivergence(SimState& s);

/// Adapt the time step to satisfy the CFL stability criterion.
void adaptTimeStep(SimState& s);
