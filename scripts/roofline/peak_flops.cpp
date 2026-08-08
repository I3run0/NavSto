// peak_flops.cpp — empirical peak GFLOP/s microbenchmark.
//
// 16 independent FMA accumulator chains (independent, not just wide) so the
// CPU's FMA pipeline stays saturated despite multi-cycle FMA latency —  a
// single dependent chain (acc = acc*a+b repeated) would bottleneck on
// latency, not throughput, and badly undercount peak.
//
// Compile with the project's own Release flags for an apples-to-apples
// ceiling:
//   g++ -O3 -march=native -funroll-loops -o peak_flops peak_flops.cpp
//   g++ -O3 -march=native -funroll-loops -fopenmp -DUSE_OPENMP -o peak_flops_omp peak_flops.cpp
//
// Verify actual vectorization (don't just trust the flag):
//   g++ -O3 -march=native -funroll-loops -fopt-info-vec-optimized -c peak_flops.cpp 2>&1 | grep vectorized
//   objdump -d peak_flops | grep -c vfmadd

#include <chrono>
#include <cstdio>
#include <cstdlib>

#ifdef USE_OPENMP
#include <omp.h>
#endif

constexpr int NUM_CHAINS = 16;
constexpr long ITERS = 2'000'000'000L;

// Runs NUM_CHAINS independent FMA chains for ITERS iterations each.
// Returns GFLOP/s (2 FLOPs per FMA * NUM_CHAINS * ITERS).
double runFlopKernel(double a, double b) {
    double acc[NUM_CHAINS];
    for (int l = 0; l < NUM_CHAINS; ++l) acc[l] = a + l;

    auto t0 = std::chrono::steady_clock::now();
    for (long it = 0; it < ITERS; ++it) {
        #pragma GCC unroll 16
        for (int l = 0; l < NUM_CHAINS; ++l) {
            acc[l] = acc[l] * a + b;
        }
    }
    auto t1 = std::chrono::steady_clock::now();

    // Prevent dead-code elimination of the accumulators.
    double sink = 0.0;
    for (int l = 0; l < NUM_CHAINS; ++l) sink += acc[l];
    if (sink == 123456789.0) std::fprintf(stderr, "unlikely\n");  // never true, keeps sink live

    double seconds = std::chrono::duration<double>(t1 - t0).count();
    double flops = 2.0 * NUM_CHAINS * static_cast<double>(ITERS);
    return flops / seconds / 1e9;
}

int main(int argc, char** argv) {
    // a, b read from argv (not compile-time constant) so the compiler can't
    // fold the recurrence into a closed form.
    double a = (argc > 1) ? std::atof(argv[1]) : 1.0000000001;
    double b = (argc > 2) ? std::atof(argv[2]) : 0.9999999999;

#ifdef USE_OPENMP
    int nthreads = omp_get_max_threads();
    double total = 0.0;
    #pragma omp parallel reduction(+:total)
    {
        total += runFlopKernel(a + omp_get_thread_num() * 1e-6, b);
    }
    std::printf("threads=%d total_gflops=%.2f\n", nthreads, total);
#else
    double gflops = runFlopKernel(a, b);
    std::printf("threads=1 total_gflops=%.2f\n", gflops);
#endif
    return 0;
}
