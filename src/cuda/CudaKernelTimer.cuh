#pragma once
// =============================================================================
//  CudaKernelTimer.cuh — device-side half of the per-kernel profiling.
//
//  Reports into the same KernelProfile registry as the CPU timer, so a GPU run
//  and a CPU run produce the same table and the same <runName>_kernels.csv.
//  Only the measurement differs, and it has to:
//
//  ── Why not steady_clock ───────────────────────────────────────────────────
//  Kernel launches are asynchronous. Wrapping computeAccelerationsCuda() in a
//  host wall-clock would time the LAUNCH -- a few microseconds of queueing --
//  and attribute none of the actual GPU work. cudaEvent markers are recorded
//  in-stream, so they bracket the work itself.
//
//  ── What this costs ────────────────────────────────────────────────────────
//  cudaEventSynchronize() in the destructor. That is unavoidable for
//  per-kernel attribution -- the elapsed time isn't readable until the stop
//  event has actually happened -- but it drains the stream after every
//  operator, serializing work the GPU would otherwise overlap across the step
//  loop. So, as KernelTimers.hpp says: the SHARES are the deliverable, the
//  TOTAL is not. Take end-to-end GPU timings from an uninstrumented build.
//
//  Events are created once per kernel slot and reused. Creating and destroying
//  a pair per call would add microseconds to every measurement -- material
//  against operators that run in single-digit milliseconds.
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
