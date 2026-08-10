#pragma once
// =============================================================================
//  Physics.hpp  —  Forward declarations for all physics routines.
//
//  All functions take `SimState&` as their primary argument and operate
//  exclusively through it.  This makes dependencies explicit and allows
//  future refactoring into a class hierarchy without touching call sites.
// =============================================================================

#include "SimState.hpp"

// ── UNIFAES helpers (internal linkage in Physics.cpp) ──────────────────────
// Not declared here; used only inside Physics.cpp.

// Every function below takes SimState& and nothing else. SimState already
// carries this backend's private working memory as `s.ext`, and its fields
// already use this backend's storage type -- both chosen in its own
// BackendConfig.hpp, so a backend can be retuned without touching a single
// signature here. See SimState.hpp.

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
