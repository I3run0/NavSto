#pragma once
// =============================================================================
//  KernelTimers.hpp — attributes wall time to each per-step operator.
//
//  Enabled by -DNAVSOLVER_PROFILE; otherwise NAVSOLVER_TIME(k, call) expands
//  to the bare call and nothing here is instantiated.
//
//  Profiled TOTALS are not comparable to a normal build (the CUDA timer
//  synchronizes per kernel — see CudaKernelTimer.cuh). The per-kernel SHARES
//  are the point; take end-to-end timings from an uninstrumented build.
// =============================================================================

#include <cstddef>

#ifdef NAVSOLVER_PROFILE
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <ostream>
#include <sstream>
#include <string>
#endif

// ---------------------------------------------------------------------------
//  The seven per-step operators, in the order Driver.cpp calls them.
// ---------------------------------------------------------------------------
enum class Kernel {
    Accel,        ///< computeAccelerations
    PressSource,  ///< buildPressureSource
    PressSolve,   ///< solvePressurePoisson
    UpdateVel,    ///< updateVelocities
    Residual,     ///< computeMomentumResidual
    Divergence,   ///< computeDivergence
    TimeStep,     ///< adaptTimeStep
    COUNT
};

inline const char* kernelName(Kernel k) {
    switch (k) {
        case Kernel::Accel:       return "computeAccelerations";
        case Kernel::PressSource: return "buildPressureSource";
        case Kernel::PressSolve:  return "solvePressurePoisson";
        case Kernel::UpdateVel:   return "updateVelocities";
        case Kernel::Residual:    return "computeMomentumResidual";
        case Kernel::Divergence:  return "computeDivergence";
        case Kernel::TimeStep:    return "adaptTimeStep";
        default:                  return "unknown";
    }
}

#ifdef NAVSOLVER_PROFILE

// ---------------------------------------------------------------------------
//  Accumulator. Backend-agnostic on purpose: it only adds seconds, so the CPU
//  (steady_clock) and CUDA (cudaEvent) timers both report into the same table.
// ---------------------------------------------------------------------------
class KernelProfile {
public:
    static KernelProfile& instance() {
        static KernelProfile p;
        return p;
    }

    void add(Kernel k, double seconds) {
        const std::size_t i = static_cast<std::size_t>(k);
        seconds_[i] += seconds;
        calls_[i]   += 1;
    }

    [[nodiscard]] double totalSeconds() const {
        double t = 0.0;
        for (std::size_t i = 0; i < static_cast<std::size_t>(Kernel::COUNT); ++i) t += seconds_[i];
        return t;
    }

    /// Human-readable table, slowest first — the ordering a person tuning a
    /// backend actually wants.
    void report(std::ostream& os) const {
        const double total = totalSeconds();
        std::size_t order[static_cast<std::size_t>(Kernel::COUNT)];
        for (std::size_t i = 0; i < static_cast<std::size_t>(Kernel::COUNT); ++i) order[i] = i;
        for (std::size_t a = 0; a + 1 < static_cast<std::size_t>(Kernel::COUNT); ++a)
            for (std::size_t b = a + 1; b < static_cast<std::size_t>(Kernel::COUNT); ++b)
                if (seconds_[order[b]] > seconds_[order[a]]) std::swap(order[a], order[b]);

        os << "\n  per-kernel time (profiling build — see KernelTimers.hpp)\n"
           << "  " << std::left << std::setw(26) << "kernel"
           << std::right << std::setw(10) << "calls"
           << std::setw(12) << "total_s"
           << std::setw(12) << "mean_ms"
           << std::setw(9)  << "share\n";

        for (std::size_t n = 0; n < static_cast<std::size_t>(Kernel::COUNT); ++n) {
            const std::size_t i = order[n];
            if (calls_[i] == 0) continue;
            const double meanMs = 1000.0 * seconds_[i] / static_cast<double>(calls_[i]);
            const double share  = (total > 0.0) ? 100.0 * seconds_[i] / total : 0.0;
            os << "  " << std::left << std::setw(26) << kernelName(static_cast<Kernel>(i))
               << std::right << std::setw(10) << calls_[i]
               << std::setw(12) << std::fixed << std::setprecision(4) << seconds_[i]
               << std::setw(12) << std::setprecision(3) << meanMs
               << std::setw(8)  << std::setprecision(1) << share << "%\n";
        }
        os << "  " << std::left << std::setw(26) << "TOTAL measured"
           << std::right << std::setw(10) << ""
           << std::setw(12) << std::fixed << std::setprecision(4) << total << "\n";
    }

    /// Machine-readable sibling of report(), for diffing two tuning runs.
    void writeCsv(const std::filesystem::path& path) const {
        std::ofstream f(path);
        if (!f) return;
        const double total = totalSeconds();
        f << "kernel,calls,total_seconds,mean_ms,share_percent\n";
        for (std::size_t i = 0; i < static_cast<std::size_t>(Kernel::COUNT); ++i) {
            if (calls_[i] == 0) continue;
            const double meanMs = 1000.0 * seconds_[i] / static_cast<double>(calls_[i]);
            const double share  = (total > 0.0) ? 100.0 * seconds_[i] / total : 0.0;
            f << kernelName(static_cast<Kernel>(i)) << ',' << calls_[i] << ','
              << std::setprecision(9) << seconds_[i] << ','
              << std::setprecision(6) << meanMs << ',' << share << '\n';
        }
    }

private:
    double      seconds_[static_cast<std::size_t>(Kernel::COUNT)] = {};
    long long   calls_  [static_cast<std::size_t>(Kernel::COUNT)] = {};
};

/// Host-side scope timer. CUDA uses its own event-based one; see the header
/// comment for why wall-clock cannot work there.
class ScopedKernelTimer {
public:
    explicit ScopedKernelTimer(Kernel k)
        : k_(k), t0_(std::chrono::steady_clock::now()) {}

    ~ScopedKernelTimer() {
        const std::chrono::duration<double> dt = std::chrono::steady_clock::now() - t0_;
        KernelProfile::instance().add(k_, dt.count());
    }

    ScopedKernelTimer(const ScopedKernelTimer&) = delete;
    ScopedKernelTimer& operator=(const ScopedKernelTimer&) = delete;

private:
    Kernel k_;
    std::chrono::steady_clock::time_point t0_;
};

#define NAVSOLVER_TIME(kernel, call)                       \
    do {                                                   \
        ScopedKernelTimer _navsolver_t(Kernel::kernel);     \
        call;                                              \
    } while (0)

#define NAVSOLVER_PROFILE_ENABLED 1

#else   // !NAVSOLVER_PROFILE — expands to the bare call, nothing added

#define NAVSOLVER_TIME(kernel, call) \
    do {                             \
        call;                        \
    } while (0)

#define NAVSOLVER_PROFILE_ENABLED 0

#endif  // NAVSOLVER_PROFILE
