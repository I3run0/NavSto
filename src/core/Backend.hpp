#pragma once
// =============================================================================
//  Backend.hpp — what a backend must supply beyond the Physics.hpp operators.
//
//  Physics.hpp fixes WHAT is computed; this fixes what the shared driver
//  (src/app/Driver.cpp) needs around it. Adding a backend means implementing
//  both, plus a BackendConfig.hpp, and adding a target in CMakeLists.txt.
// =============================================================================

#include "SimState.hpp"

// ── Program lifecycle ───────────────────────────────────────────────────────

/// Short name for the startup banner, e.g. "serial". From kBackendName.
const char* backendName();

/// Called once after initSimulation(), before the first step. Builds whatever
/// the per-step loop needs that geometry had to exist for.
void backendStartup(SimState& s);

/// Called once on every exit path, including the error paths.
void backendShutdown(SimState& s);

/// Makes s's Fields readable on the host. The driver calls this before every
/// VTK write; backends that already compute in host memory do nothing.
void syncFieldsToHost(SimState& s);

// ── RK4 bookkeeping ─────────────────────────────────────────────────────────
//  The driver owns the stage sequence and the weights; these move the data.
//  Only called when cfg.flowType == RK4Transient.

/// Snapshot u_n into the workspace and zero the K accumulators.
void rk4Save(SimState& s);

/// Restore u_n from the snapshot, at the top of each stage.
void rk4Restore(SimState& s);

/// K += weight * (u - u_n), over the active domain.
void rk4Accumulate(SimState& s, double weight);

/// u_{n+1} = u_n + K/6, over the active domain.
void rk4Combine(SimState& s);

/// Re-apply the outlet and lateral velocity BCs after rk4Combine().
void rk4ApplyFinalBCs(SimState& s);

// ── For a decomposed (multi-process) backend ────────────────────────────────
//
//  SimConfig::numCells* are LOCAL and globalNumCells*/origin* describe the full
//  domain: set those in backendStartup() before initSimulation() and geometry
//  slices itself to this rank. The residual/divergence reductions then become
//  collectives, and syncFieldsToHost() is where a gather belongs. Still
//  unsolved: the driver logs and writes VTK from every rank.
