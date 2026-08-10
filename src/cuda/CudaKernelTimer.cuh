#pragma once
// =============================================================================
//  CudaKernelTimer.cuh — GPU half of the per-kernel profiling.
//
//  Reports into the same KernelProfile as the CPU timer. Uses cudaEvents
//  because launches are async: a host clock would time the launch, not the
//  work. The stop-event sync that makes attribution possible also serializes
//  work the GPU would overlap, so profiled totals are not comparable.
//
//  Events are created once per kernel slot and reused.
// =============================================================================

#include "KernelTimers.hpp"

#ifdef NAVSOLVER_PROFILE

#include <cuda_runtime.h>

#include <cstddef>
#include <utility>

class ScopedCudaKernelTimer {
public:
    explicit ScopedCudaKernelTimer(Kernel k) : k_(k) {
        cudaEventRecord(slot(k_).first);
    }

    ~ScopedCudaKernelTimer() {
        auto& e = slot(k_);
        cudaEventRecord(e.second);
        cudaEventSynchronize(e.second);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, e.first, e.second);
        KernelProfile::instance().add(k_, static_cast<double>(ms) / 1000.0);
    }

    ScopedCudaKernelTimer(const ScopedCudaKernelTimer&) = delete;
    ScopedCudaKernelTimer& operator=(const ScopedCudaKernelTimer&) = delete;

private:
    using EventPair = std::pair<cudaEvent_t, cudaEvent_t>;

    /// One reused start/stop pair per kernel. Safe because the destructor
    /// synchronizes before the next call to the same kernel can record again.
    static EventPair& slot(Kernel k) {
        static EventPair events[static_cast<std::size_t>(Kernel::COUNT)];
        static bool created = [] {
            for (auto& e : events) {
                cudaEventCreate(&e.first);
                cudaEventCreate(&e.second);
            }
            return true;
        }();
        (void)created;
        return events[static_cast<std::size_t>(k)];
    }

    Kernel k_;
};

#define NAVSOLVER_TIME_CUDA(kernel, call)                      \
    do {                                                       \
        ScopedCudaKernelTimer _navsolver_ct(Kernel::kernel);    \
        call;                                                  \
    } while (0)

#else   // !NAVSOLVER_PROFILE — expands to the bare call

#define NAVSOLVER_TIME_CUDA(kernel, call) \
    do {                                  \
        call;                             \
    } while (0)

#endif  // NAVSOLVER_PROFILE
