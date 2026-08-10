// peak_exp_throughput.cpp — empirical std::exp() throughput microbenchmark.
//
// computeExponentialWeights() (src/backends/serial/Physics.cpp) is ~13% of profiled
// wall time over 60M+ calls per 100 steps, dominated by std::exp() in its
// dominant branch (Bernstein-Crank formula, |DPe| in [0.1, 200]). glibc's
// scalar exp() is not auto-vectorized here (no SVML/-ffast-math), so this
// is a distinct roofline ceiling from FMA peak, not the same one.
//
// Compile:
//   g++ -O3 -march=native -funroll-loops -o peak_exp_throughput peak_exp_throughput.cpp
//   g++ -O3 -march=native -funroll-loops -fopenmp -DUSE_OPENMP -o peak_exp_throughput_omp peak_exp_throughput.cpp

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>

#ifdef USE_OPENMP
#include <omp.h>
#endif

constexpr long ITERS = 200'000'000L;

// Returns exp() calls per second for one thread.
double runExpKernel(double seed) {
    double x = 0.01 + seed;   // stay inside the |DPe|<=200 branch computeExponentialWeights actually takes
    double sink = 0.0;

    auto t0 = std::chrono::steady_clock::now();
    for (long it = 0; it < ITERS; ++it) {
        // Vary the input every iteration (not a constant) so the compiler
        // can't hoist/CSE the exp() call out of the loop.
        x += 1e-7;
        if (x > 100.0) x = 0.01;
        sink += std::exp(x);
    }
    auto t1 = std::chrono::steady_clock::now();

    if (sink == -1.0) std::fprintf(stderr, "unlikely\n");  // keeps sink live
    double seconds = std::chrono::duration<double>(t1 - t0).count();
    return static_cast<double>(ITERS) / seconds;
}

int main() {
#ifdef USE_OPENMP
    int nthreads = omp_get_max_threads();
    double total = 0.0;
    #pragma omp parallel reduction(+:total)
    {
        total += runExpKernel(omp_get_thread_num() * 0.1);
    }
    std::printf("threads=%d exp_calls_per_sec=%.3e\n", nthreads, total);
#else
    double rate = runExpKernel(0.0);
    std::printf("threads=1 exp_calls_per_sec=%.3e\n", rate);
#endif
    return 0;
}
