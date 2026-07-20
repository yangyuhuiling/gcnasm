#include <opus/hip_minimal.hpp>

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "gemm_defs.h"

template<typename Traits>
__global__ void gemm_a16w16_quad_subtile_kernel(opus_gemm_kargs kargs);

template<typename Traits>
__global__ void gemm_a16w16_mono_tile_kernel(opus_gemm_kargs kargs);

#define CHECK_HIP(call)                                                                                 \
    do {                                                                                                \
        hipError_t status_ = call;                                                                      \
        if (status_ != hipSuccess) {                                                                    \
            std::fprintf(stderr, "HIP error (%s:%d): %s\n", __FILE__, __LINE__, hipGetErrorString(status_)); \
            std::exit(1);                                                                               \
        }                                                                                               \
    } while (0)

template<typename Traits, typename Kernel>
static void run_one(const char* label, Kernel kernel, const opus_gemm_kargs& kargs, int warmup, int iters) {
    const int num_tiles_m = ceil_div(kargs.m, Traits::B_M);
    const int num_tiles_n = ceil_div(kargs.n, Traits::B_N);
    dim3 grid(num_tiles_m * num_tiles_n, 1, kargs.batch);
    dim3 block(Traits::BLOCK_SIZE);

    auto launch = [&]() {
        kernel<<<grid, block>>>(kargs);
        CHECK_HIP(hipGetLastError());
    };

    for (int i = 0; i < warmup; ++i) launch();
    CHECK_HIP(hipDeviceSynchronize());

    hipEvent_t start, stop;
    CHECK_HIP(hipEventCreate(&start));
    CHECK_HIP(hipEventCreate(&stop));
    CHECK_HIP(hipEventRecord(start));
    for (int i = 0; i < iters; ++i) launch();
    CHECK_HIP(hipEventRecord(stop));
    CHECK_HIP(hipEventSynchronize(stop));

    float total_ms = 0.0f;
    CHECK_HIP(hipEventElapsedTime(&total_ms, start, stop));
    CHECK_HIP(hipEventDestroy(start));
    CHECK_HIP(hipEventDestroy(stop));

    const double avg_ms = static_cast<double>(total_ms) / iters;
    const double flops = 2.0 * double(kargs.m) * double(kargs.n) * double(kargs.k) * double(kargs.batch);
    const double tflops = flops / (avg_ms * 1.0e9);
    std::printf("  %-22s tile=%dx%dx%d grid=%dx%d avg=%8.4f ms  %8.2f TFLOP/s\n",
                label, Traits::B_M, Traits::B_N, Traits::B_K, grid.x, grid.z, avg_ms, tflops);
}

int main(int argc, char** argv) {
    int M = 4096;
    int N = 4096;
    int K = 4096;
    int batch = 1;
    int warmup = 10;
    int iters = 50;

    for (int i = 1; i < argc; ++i) {
        if ((std::strcmp(argv[i], "-m") == 0 || std::strcmp(argv[i], "--m") == 0) && i + 1 < argc) M = std::atoi(argv[++i]);
        else if ((std::strcmp(argv[i], "-n") == 0 || std::strcmp(argv[i], "--n") == 0) && i + 1 < argc) N = std::atoi(argv[++i]);
        else if ((std::strcmp(argv[i], "-k") == 0 || std::strcmp(argv[i], "--k") == 0) && i + 1 < argc) K = std::atoi(argv[++i]);
        else if ((std::strcmp(argv[i], "-b") == 0 || std::strcmp(argv[i], "--b") == 0) && i + 1 < argc) batch = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) warmup = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--iters") == 0 && i + 1 < argc) iters = std::atoi(argv[++i]);
    }

    bf16_t* dev_a = nullptr;
    bf16_t* dev_b = nullptr;
    bf16_t* dev_c = nullptr;
    const size_t a_elems = static_cast<size_t>(batch) * M * K;
    const size_t b_elems = static_cast<size_t>(batch) * N * K;
    const size_t c_elems = static_cast<size_t>(batch) * M * N;
    CHECK_HIP(hipMalloc(&dev_a, a_elems * sizeof(bf16_t)));
    CHECK_HIP(hipMalloc(&dev_b, b_elems * sizeof(bf16_t)));
    CHECK_HIP(hipMalloc(&dev_c, c_elems * sizeof(bf16_t)));
    CHECK_HIP(hipMemset(dev_a, 1, a_elems * sizeof(bf16_t)));
    CHECK_HIP(hipMemset(dev_b, 2, b_elems * sizeof(bf16_t)));
    CHECK_HIP(hipMemset(dev_c, 0, c_elems * sizeof(bf16_t)));

    opus_gemm_kargs kargs{};
    kargs.ptr_a = dev_a;
    kargs.ptr_b = dev_b;
    kargs.ptr_c = dev_c;
    kargs.m = M;
    kargs.n = N;
    kargs.k = K;
    kargs.batch = batch;
    kargs.stride_a = K;
    kargs.stride_b = K;
    kargs.stride_c = N;
    kargs.stride_a_batch = M * K;
    kargs.stride_b_batch = N * K;
    kargs.stride_c_batch = M * N;

    std::printf("BF16 GEMM bench-only: M=%d N=%d K=%d batch=%d warmup=%d iters=%d\n", M, N, K, batch, warmup, iters);
    using TraitsQuad = opus_gemm_traits<512, 256, 256, 64, bf16_t, bf16_t, bf16_t, float>;
    using TraitsMono = opus_gemm_traits<512, 192, 256, 64, bf16_t, bf16_t, bf16_t, float>;
    run_one<TraitsQuad>("quad_subtile", gemm_a16w16_quad_subtile_kernel<TraitsQuad>, kargs, warmup, iters);
    run_one<TraitsMono>("mono_tile", gemm_a16w16_mono_tile_kernel<TraitsMono>, kargs, warmup, iters);

    CHECK_HIP(hipFree(dev_a));
    CHECK_HIP(hipFree(dev_b));
    CHECK_HIP(hipFree(dev_c));
    return 0;
}
