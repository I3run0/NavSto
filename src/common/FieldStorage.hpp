#pragma once
// =============================================================================
//  FieldStorage.hpp — contract a backend's `Field` type must satisfy.
//
//  Shared code touches fields only through these members, so a backend can
//  change layout, padding or page-faulting behind them. Checked by the
//  static_assert in SimState.hpp, which fails naming the type rather than
//  spilling instantiation errors into whichever shared file used it first.
// =============================================================================

#include "GridField.hpp"

#include <concepts>
#include <cstddef>

/// Everything shared code needs from a field. Checked once per backend by the
/// static_assert in SimState.hpp.
template <class F>
concept FieldStorage = requires(F f, const F cf, GridSize g, double v, int i) {
    // Construction. Default for SimState's members; from GridSize for the
    // driver's RK4 working storage.
    requires std::default_initializable<F>;
    { F(g) }               -> std::same_as<F>;

    // Allocation. Separate from construction so a backend can control page
    // faulting (see NUMA note above).
    { f.resize(g) }        -> std::same_as<void>;
    { f.fill(v) }          -> std::same_as<void>;

    // Element access — the only way shared code ever reads or writes a field.
    { f(i, i, i) }         -> std::same_as<double&>;
    { cf(i, i, i) }        -> std::same_as<const double&>;

    // Bulk access. Must be contiguous doubles; see the note above.
    { cf.data().data() }   -> std::convertible_to<const double*>;
    { cf.data().size() }   -> std::convertible_to<std::size_t>;
    { cf.gridSize() }      -> std::convertible_to<GridSize>;
};

// The stock storage satisfies its own contract. Checked here so a change to
// GridField that breaks the contract is caught in GridField's own terms,
// rather than in whichever backend next tries to use it.
static_assert(FieldStorage<GridField<>>,
              "GridField no longer satisfies FieldStorage — see FieldStorage.hpp");
