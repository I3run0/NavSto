// peak_division.cpp — empirical double-division throughput on this machine.
//
// docs/roofline.md's 2026-08-28 correction found computeAccelerations to be
// division-throughput bound, not memory- or exp()-bound, so a division ceiling
// is what a lower bound on that kernel's runtime has to be built from. The
// FMA and bandwidth ceilings say nothing about it: `divsd` and `vdivpd` run on
// a separate, non-pipelined divider.
//
// Two ceilings, because which one applies depends entirely on whether the
// enclosing loop vectorises:
//   scalar  — divsd, one double per division, the pre-vectorisation shape
//   packed  — vdivpd on 256-bit vectors, four doubles per division
//
// 16 independent chains for the same reason peak_flops.cpp uses them: the
// divider's latency is far longer than its throughput interval, so a single
// dependent chain measures latency and undercounts the ceiling several-fold.
//
//   g++ -O3 -march=native -funroll-loops -o peak_division peak_division.cpp
//
// Verify the packed loop really is packed (do not trust the flag):
//   objdump -d peak_division | grep -c vdivpd

#include <chrono>
#include <cstdio>

constexpr int  NUM_CHAINS = 16;
constexpr long ITERS      = 200'000'000L;

// One array-based division loop, compiled twice. Array-based because that is
// the shape the kernel presents (KernelRows.hpp's weightDivideRow); the only
// difference between the two ceilings is whether the compiler is allowed to
// pack it, which is exactly the difference the 2026-08-28 correction is about.
//
// A previous version of this benchmark used per-accumulator recurrences for the
// scalar arm. GCC collapsed the outer loop and reported 5.6e7 Gdiv/s -- kept as
// a warning: always read the disassembly of a division microbenchmark back.
static inline double divide_loop(const double* __restrict a,
                                 const double* __restrict b,
                                 double* __restrict out, int n) {
    for (int i = 0; i < n; ++i) out[i] = a[i] / b[i];
    return out[0];
}

__attribute__((optimize("no-tree-vectorize")))
double runScalar(int n, double* a, double* b, double* out, long reps) {
    auto t0 = std::chrono::steady_clock::now();
    for (long r = 0; r < reps; ++r) {
        divide_loop(a, b, out, n);
        a[0] += 1e-13;                     // block hoisting the whole call
    }
    auto t1 = std::chrono::steady_clock::now();
    double s = std::chrono::duration<double>(t1 - t0).count();
    return (double)reps * n / s / 1e9;
}

double runPacked(int n, double* a, double* b, double* out, long reps) {
    auto t0 = std::chrono::steady_clock::now();
    for (long r = 0; r < reps; ++r) {
        divide_loop(a, b, out, n);
        a[0] += 1e-13;
    }
    auto t1 = std::chrono::steady_clock::now();
    double s = std::chrono::duration<double>(t1 - t0).count();
    return (double)reps * n / s / 1e9;
}

int main() {
    // n small enough to stay L1-resident: this measures the divider, not memory.
    constexpr int  N    = 1024;
    constexpr long REPS = 4'000'000L;
    static double a[N], b[N], out[N];
    for (int i = 0; i < N; ++i) { a[i] = 3.7 + i * 1e-3; b[i] = 1.0 + (i % 97) * 0.01; }

    double scalar = runScalar(N, a, b, out, REPS);
    for (int i = 0; i < N; ++i) a[i] = 3.7 + i * 1e-3;
    double packed = runPacked(N, a, b, out, REPS);

    std::printf("scalar divisions : %8.3f Gdiv/s  (%.3f ns per division)\n",
                scalar, 1.0 / scalar);
    std::printf("packed divisions : %8.3f Gdiv/s  (%.3f ns per division)\n",
                packed, 1.0 / packed);
    std::printf("packed speedup   : %8.2fx\n", packed / scalar);
    return 0;
}
