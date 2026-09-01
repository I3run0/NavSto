// peak_bandwidth_cuda.cu — empirical device memory-bandwidth microbenchmark.
//
// STREAM-triad (a[i] = b[i] + s*c[i]) in both float and double, so the
// precision question can be answered against a measured ceiling rather than
// a spec sheet. Arrays are sized far beyond the L2 so this is HBM/GDDR
// traffic, not cache reuse. Reports one line per type.
//
// Compile:
//   nvcc -O3 -arch=sm_86 -o peak_bandwidth_cuda peak_bandwidth_cuda.cu
#include <cstdio>
#include <cuda_runtime.h>

template <typename T>
__global__ void triad(T* __restrict a, const T* __restrict b,
                      const T* __restrict c, T s, long n)
{
    long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    long stride = (long)gridDim.x * blockDim.x;
    for (; i < n; i += stride) a[i] = b[i] + s * c[i];
}

template <typename T>
static void run(const char* name, long bytesTarget)
{
    const long n = bytesTarget / (3 * (long)sizeof(T));
    T *a, *b, *c;
    if (cudaMalloc(&a, n*sizeof(T)) != cudaSuccess) { printf("%s alloc failed\n", name); return; }
    cudaMalloc(&b, n*sizeof(T)); cudaMalloc(&c, n*sizeof(T));
    cudaMemset(b, 1, n*sizeof(T)); cudaMemset(c, 1, n*sizeof(T));

    int block = 256, grid = 0, dev = 0;
    cudaDeviceProp p; cudaGetDevice(&dev); cudaGetDeviceProperties(&p, dev);
    grid = p.multiProcessorCount * 32;

    for (int w = 0; w < 3; ++w) triad<T><<<grid, block>>>(a, b, c, (T)3, n);
    cudaDeviceSynchronize();

    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    float best = 1e30f;
    for (int r = 0; r < 10; ++r) {
        cudaEventRecord(e0);
        triad<T><<<grid, block>>>(a, b, c, (T)3, n);
        cudaEventRecord(e1); cudaEventSynchronize(e1);
        float ms; cudaEventElapsedTime(&ms, e0, e1);
        if (ms < best) best = ms;
    }
    // two reads + one write per element
    double gb = 3.0 * n * sizeof(T) / 1e9;
    printf("type=%-6s elems=%ld  bandwidth_gbps=%.1f\n", name, n, gb / (best / 1e3));
    cudaFree(a); cudaFree(b); cudaFree(c);
}

int main()
{
    int dev = 0; cudaDeviceProp p;
    cudaGetDevice(&dev); cudaGetDeviceProperties(&p, dev);
    printf("device=%s  SMs=%d  L2=%d KB  memClk=%d MHz  busWidth=%d bit  theoretical_gbps=%.1f\n",
           p.name, p.multiProcessorCount, p.l2CacheSize/1024, p.memoryClockRate/1000,
           p.memoryBusWidth, 2.0 * p.memoryClockRate * 1e3 * (p.memoryBusWidth/8) / 1e9);
    run<float>("float", 768L<<20);
    run<double>("double", 768L<<20);
    return 0;
}
