#pragma once
// =============================================================================
//  GridField.hpp  —  Strongly-typed, bounds-checked 3-D/2-D field wrapper.
//
//  Replaces the raw #define F3 macro with a zero-overhead inline accessor.
//  In Release builds (-DNDEBUG) the bounds check is compiled away.
// =============================================================================

#include <vector>
#include <stdexcept>
#include <string>
#include <cstddef>

/// Allocated extents, which are numCells + 2 per axis, not + 1: indices run
/// 0..numCells, plus one ghost layer the pressure solve mirrors Neumann values
/// into. See SimState::allocateFields().
struct GridSize {
    int sI = 0;   ///< numCellsX + 2
    int sJ = 0;   ///< numCellsY + 2
    int sK = 0;   ///< numCellsZ + 2
};

// ---------------------------------------------------------------------------
//  GridField<T> — owning wrapper around a flat std::vector<T> that exposes
//  (i, j, k) indexing.  Constructed from a GridSize; resize() re-uses storage.
// ---------------------------------------------------------------------------
template <typename T = double>
class GridField {
public:
    GridField() = default;

    explicit GridField(const GridSize& g, T init = T{})
        : g_(g), data_(static_cast<std::size_t>(g.sI) * g.sJ * g.sK, init) {}

    void resize(const GridSize& g, T init = T{}) {
        g_ = g;
        data_.assign(static_cast<std::size_t>(g.sI) * g.sJ * g.sK, init);
    }

    void fill(T v) { std::fill(data_.begin(), data_.end(), v); }

    // ── Accessor ─────────────────────────────────────────────────────────────
    [[nodiscard]] T& at(int i, int j, int k) {
#ifndef NDEBUG
        boundsCheck(i, j, k);
#endif
        return data_[idx(i, j, k)];
    }
    [[nodiscard]] const T& at(int i, int j, int k) const {
#ifndef NDEBUG
        boundsCheck(i, j, k);
#endif
        return data_[idx(i, j, k)];
    }

    // Shorthand operator() mirrors at() — same cost after inlining.
    T& operator()(int i, int j, int k)       { return at(i, j, k); }
    const T& operator()(int i, int j, int k) const { return at(i, j, k); }

    // Raw storage access (for I/O and inter-op).
    [[nodiscard]] const std::vector<T>& data() const { return data_; }
    [[nodiscard]] std::vector<T>&       data()       { return data_; }

    [[nodiscard]] const GridSize& gridSize() const { return g_; }

private:
    GridSize       g_;
    std::vector<T> data_;

    [[nodiscard]] std::size_t idx(int i, int j, int k) const noexcept {
        return (static_cast<std::size_t>(i) * g_.sJ + j) * g_.sK + k;
    }

    void boundsCheck(int i, int j, int k) const {
        if (i < 0 || i >= g_.sI || j < 0 || j >= g_.sJ || k < 0 || k >= g_.sK) {
            throw std::out_of_range(
                "GridField out-of-range: (" + std::to_string(i) + "," +
                std::to_string(j) + "," + std::to_string(k) +
                ") in grid (" + std::to_string(g_.sI) + "x" +
                std::to_string(g_.sJ) + "x" + std::to_string(g_.sK) + ")"
            );
        }
    }
};
