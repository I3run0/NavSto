#pragma once
// =============================================================================
//  FirstTouchField.hpp — GridField with NUMA-aware first touch. OpenMP only.
//
//  Identical storage and indexing; only differs in who faults the pages in.
//  std::vector fills from the calling thread, so every page lands on one NUMA
//  node; here each thread writes the range it will later compute on.
//
//  Compiled and contract-checked but NOT currently selected — measured inside
//  the run-to-run variance on this single-socket machine. See BackendConfig.hpp.
// =============================================================================

#include "FieldStorage.hpp"
#include "GridField.hpp"

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

template <typename T = double>
class FirstTouchFieldT {
public:
    FirstTouchFieldT() = default;

    explicit FirstTouchFieldT(const GridSize& g, T init = T{}) { resize(g, init); }

    /// Allocates without value-initialising, then writes every element from
    /// the thread that owns that index range under the same static schedule
    /// the solver's own loops use — so the page ends up on that thread's node.
    void resize(const GridSize& g, T init = T{}) {
        g_ = g;
        const std::size_t n =
            static_cast<std::size_t>(g.sI) * static_cast<std::size_t>(g.sJ) *
            static_cast<std::size_t>(g.sK);

        // resize() alone would value-initialise serially, defeating the point;
        // the parallel loop below is what actually faults the pages in.
        data_.clear();
        data_.resize(n);

        T* p = data_.data();
        const long long total = static_cast<long long>(n);
        #pragma omp parallel for schedule(static)
        for (long long i = 0; i < total; ++i) p[i] = init;
    }

    void fill(T v) {
        T* p = data_.data();
        const long long total = static_cast<long long>(data_.size());
        #pragma omp parallel for schedule(static)
        for (long long i = 0; i < total; ++i) p[i] = v;
    }

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

    T& operator()(int i, int j, int k)             { return at(i, j, k); }
    const T& operator()(int i, int j, int k) const { return at(i, j, k); }

    [[nodiscard]] const std::vector<T>& data() const { return data_; }
    [[nodiscard]] std::vector<T>&       data()       { return data_; }
    [[nodiscard]] const GridSize& gridSize() const   { return g_; }

private:
    GridSize       g_;
    std::vector<T> data_;

    // Identical to GridField's — same layout, only the faulting differs.
    [[nodiscard]] std::size_t idx(int i, int j, int k) const noexcept {
        return (static_cast<std::size_t>(i) * g_.sJ + j) * g_.sK + k;
    }

    void boundsCheck(int i, int j, int k) const {
        if (i < 0 || i >= g_.sI || j < 0 || j >= g_.sJ || k < 0 || k >= g_.sK) {
            throw std::out_of_range(
                "FirstTouchField out-of-range: (" + std::to_string(i) + "," +
                std::to_string(j) + "," + std::to_string(k) + ")");
        }
    }
};

using FirstTouchField = FirstTouchFieldT<double>;

// Compiled but not selected, so this is what stops it drifting out of contract.
static_assert(FieldStorage<FirstTouchField>,
              "FirstTouchField no longer satisfies FieldStorage "
              "— see src/common/FieldStorage.hpp");
