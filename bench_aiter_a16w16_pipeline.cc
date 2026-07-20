#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <type_traits>

#include "opus_gemm/include/gfx950/opus_gemm_pipeline_a16w16_gfx950.cuh"

using bf16_t = __bf16;
using fp32_t = float;

#define CHECK_HIP(call)                                                                                   \
    do {                                                                                                  \
        hipError_t status_ = call;                                                                        \
        if (status_ != hipSuccess) {                                                                      \
            std::fprintf(stderr, "HIP error (%s:%d): %s\n", __FILE__, __LINE__, hipGetErrorString(status_)); \
            std::exit(1);                                                                                 \
        }                                                                                                 \
    } while (0)

template<int BLOCK_SIZE, int B_M, int B_N, int B_K, int T_N, bool HAS_OOB = true>
using A16W16Traits = opus_gemm_a16w16_traits_gfx950<
    BLOCK_SIZE,
    opus::seq<B_M, B_N, B_K>,
    opus::tuple<bf16_t, bf16_t, bf16_t, fp32_t>,
    opus::seq<8, 8, 4>,
    opus::seq<2, T_N, 1>,
    opus::seq<16, 16, 32>,
    false,
    void,
    HAS_OOB>;

template<typename Traits>
static bool shape_supported(int m, int n, int k) {
    if constexpr (Traits::HAS_OOB) {
        return k % Traits::B_K == 0 && ((k / Traits::B_K) % 2) == 0;
    } else {
        return m % Traits::B_M == 0 && n % Traits::B_N == 0 &&
               k % Traits::B_K == 0 && ((k / Traits::B_K) % 2) == 0;
    }
}

template<typename Traits>
static void bench_one(const char* label, const opus_gemm_noscale_kargs_gfx950& kargs,
                      int warmup, int iters) {
    if (!shape_supported<Traits>(kargs.m, kargs.n, kargs.k)) {
        std::printf("  %-16s unsupported shape\n", label);
        return;
    }

    const int num_tiles_m = (kargs.m + Traits::B_M - 1) / Traits::B_M;
    const int num_tiles_n = (kargs.n + Traits::B_N - 1) / Traits::B_N;
    const int total_wgs = num_tiles_m * num_tiles_n;
    dim3 grid(total_wgs, 1, kargs.batch);
    dim3 block(Traits::BLOCK_SIZE);

    auto launch = [&]() {
        gemm_a16w16_kernel<Traits><<<grid, block>>>(kargs);
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
    const double flops = 2.0 * double(kargs.batch) * double(kargs.m) * double(kargs.n) * double(kargs.k);
    const double tflops = flops / (avg_ms * 1.0e9);
    const int waves = Traits::BLOCK_SIZE / 64;
    std::printf("  %-16s %dw tile=%dx%dx%d grid=%d avg=%8.4f ms  %8.2f TFLOP/s\n",
                label, waves, Traits::B_M, Traits::B_N, Traits::B_K, total_wgs, avg_ms, tflops);
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
        else if ((std::strcmp(argv[i], "-b") == 0 || std::strcmp(argv[i], "--batch") == 0) && i + 1 < argc) batch = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) warmup = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--iters") == 0 && i + 1 < argc) iters = std::atoi(argv[++i]);
    }

    bf16_t* d_a = nullptr;
    bf16_t* d_b = nullptr;
    bf16_t* d_c = nullptr;
    const size_t a_elems = static_cast<size_t>(batch) * M * K;
    const size_t b_elems = static_cast<size_t>(batch) * N * K;
    const size_t c_elems = static_cast<size_t>(batch) * M * N;
    CHECK_HIP(hipMalloc(&d_a, a_elems * sizeof(bf16_t)));
    CHECK_HIP(hipMalloc(&d_b, b_elems * sizeof(bf16_t)));
    CHECK_HIP(hipMalloc(&d_c, c_elems * sizeof(bf16_t)));
    CHECK_HIP(hipMemset(d_a, 1, a_elems * sizeof(bf16_t)));
    CHECK_HIP(hipMemset(d_b, 2, b_elems * sizeof(bf16_t)));
    CHECK_HIP(hipMemset(d_c, 0, c_elems * sizeof(bf16_t)));

    opus_gemm_noscale_kargs_gfx950 kargs{};
    kargs.ptr_a = d_a;
    kargs.ptr_b = d_b;
    kargs.ptr_c = d_c;
    kargs.ptr_bias = nullptr;
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
    kargs.stride_bias_batch = 0;

    std::printf("aiter gfx950 a16w16 split-barrier pipeline: M=%d N=%d K=%d batch=%d warmup=%d iters=%d\n",
                M, N, K, batch, warmup, iters);
    bench_one<A16W16Traits<256, 128, 256, 32, 2>>("kid4", kargs, warmup, iters);
    bench_one<A16W16Traits<256, 256, 128, 32, 2>>("kid5", kargs, warmup, iters);
    bench_one<A16W16Traits<512, 128, 128, 64, 4>>("kid6", kargs, warmup, iters);
    bench_one<A16W16Traits<512, 256, 128, 64, 4>>("kid7", kargs, warmup, iters);
    bench_one<A16W16Traits<512, 128, 256, 64, 4>>("kid8", kargs, warmup, iters);
    bench_one<A16W16Traits<512, 256, 256, 64, 4>>("kid9", kargs, warmup, iters);

    CHECK_HIP(hipFree(d_a));
    CHECK_HIP(hipFree(d_b));
    CHECK_HIP(hipFree(d_c));
    return 0;
}
