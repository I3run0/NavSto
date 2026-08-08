// peak_bandwidth.cpp — empirical peak memory-bandwidth microbenchmark
// (STREAM-triad: a[i] = b[i] + scalar*c[i]).
//
// Arrays sized well beyond this CPU's L3 (12 MiB total, confirmed via
// `lscpu`) so the measurement reflects real DRAM bandwidth, not
// cache-resident reuse.
//
// Compile:
//   g++ -O3 -march=native -funroll-loops -o peak_bandwidth peak_bandwidth.cpp
//   g++ -O3 -march=native -funroll-loops -fopenmp -DUSE_OPENMP -o peak_bandwidth_omp peak_bandwidth.cpp

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

#ifdef USE_OPENMP
#include <omp.h>
#endif

// 8,000,000 doubles/array * 3 arrays * 8 bytes = 192 MiB total working set
// — well beyond the 12 MiB L3, so this cannot be served from cache.
constexpr long N = 8'000'000L;
constexpr int REPS = 50;

int main() {
    std::vector<double> a(N), b(N), c(N);
    for (long i = 0; i < N; ++i) { b[i] = 1.0; c[i] = 2.0; }
    double scalar = 3.0;

    double* pa = a.data();
    const double* pb = b.data();
    const double* pc = c.data();

    auto t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < REPS; ++r) {
#ifdef USE_OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (long i = 0; i < N; ++i) {
            pa[i] = pb[i] + scalar * pc[i];
        }
        // Vary scalar slightly each rep so the compiler can't hoist/CSE
        // across repetitions.
        scalar += 1e-9;
    }
    auto t1 = std::chrono::steady_clock::now();

    // Prevent dead-code elimination.
    double sink = 0.0;
    for (long i = 0; i < N; i += N / 8) sink += pa[i];
    if (sink == -1.0) std::fprintf(stderr, "unlikely\n");

    double seconds = std::chrono::duration<double>(t1 - t0).count();
    // 2 reads + 1 write, 8 bytes/double, N elements, REPS repetitions.
    double bytes = 3.0 * 8.0 * static_cast<double>(N) * REPS;
    double gbps = bytes / seconds / 1e9;

#ifdef USE_OPENMP
    std::printf("threads=%d bandwidth_gbps=%.2f\n", omp_get_max_threads(), gbps);
#else
    std::printf("threads=1 bandwidth_gbps=%.2f\n", gbps);
#endif
    return 0;
}
