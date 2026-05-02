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

// ── Initialisation ──────────────────────────────────────────────────────────

/// Set up geometry arrays (jLow/jHigh/iLow/iHigh), fill initial velocity and
/// pressure fields, and compute the first stable time step.
void initSimulation(SimState& s);

// ── Per-step operators ──────────────────────────────────────────────────────

/// Recompute UNIFAES advective + viscous accelerations (accelX/Y/Z).
void computeAccelerations(SimState& s);

/// Build the RHS of the pressure Poisson equation: S = ∇·u/dt + ∇·A.
void buildPressureSource(SimState& s);

/// Solve ∇²p = S by Gauss-Seidel iteration.
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
