// 2-slot LDS ping-pong variant of matrix_core_tcopy_move_1p1c_2wgCluster.cc.
//
// Goal: verify that producer can overlap TDM issue for step ks+1 with
//       consumer reading step ks, by alternating LDS slots.
//
// LDS layout (one cluster WG):
//   slot 0:  A region [0,                 slot_bytes_A)
//            B region [slot_bytes_A,      slot_pair_bytes)
//   slot 1:  A region [slot_pair_bytes,                slot_pair_bytes + slot_bytes_A)
//            B region [slot_pair_bytes + slot_bytes_A, 2 * slot_pair_bytes)
//
//   slot_bytes_A   = Block_M * (Block_K + 8) * sizeof(fp16_t)   = 8704
//   slot_bytes_B   = slot_bytes_A                                = 8704
//   slot_pair_bytes= slot_bytes_A + slot_bytes_B                 = 17408
//
// Producer pipeline (per producer wave, 2-deep):
//   issue step 0  → slot 0           (prologue)
//   if K_STEPS > 1:
//     win.move(Block_K, 0, +slot_pair_bytes); issue step 1 → slot 1   (prefetch)
//   for ks in 0..K_STEPS:
//     s_wait_tensorcnt(ks < K_STEPS-1 ? 1 : 0)   // step ks completed
//     signal nbar_1                              // slot (ks&1) ready
//     wait   nbar_2                              // slot (ks&1) freed by consumer
//     if ks + 2 < K_STEPS:
//       dlds = (ks & 1) ? +slot_pair_bytes : -slot_pair_bytes
//       win.move(Block_K, 0, dlds)               // alternate slot
//       win.load_to_lds()                        // prefetch step ks+2 → slot (ks&1)
//
// Consumer pipeline (per consumer wave):
//   for ks in 0..K_STEPS:
//     wait nbar_1
//     ds_read from base + (ks&1)*slot_pair_bytes + per-tile offset
//     WMMA accumulate
//     signal nbar_2
//
// s_wait_tensorcnt semantics: count is the maximum number of inflight TDM
// operations allowed to remain. wait(0) drains all; wait(1) keeps the most
// recent prefetch inflight.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <random>
#include <iostream>
#include <memory>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <cassert>
#include <algorithm>
#include <stdio.h>
#include <omp.h>
#include "half.hpp"

#include "opus/opus.hpp"
#include "reg_access_uitls.h"
// tcopy_desc / tcopy_window now in opus.hpp
#include "named_barrier.hpp"

#define CHECK_HIP(call)                                                                                   \
    do {                                                                                                  \
        hipError_t status_ = call;                                                                        \
        if (status_ != hipSuccess) {                                                                      \
            fprintf(stderr, "HIP error (%s:%d): %s\n", __FILE__, __LINE__, hipGetErrorString(status_));   \
            exit(1);                                                                                      \
        }                                                                                                 \
    } while(0)

#define CHECK_HIP_KERNEL_LAUNCH() CHECK_HIP(hipGetLastError())

using fp32_t  = float;
using float16 = half_float::half;

using namespace opus;

using int32x4_t = int32_t __attribute__((ext_vector_type(4)));
using int32x8_t = int32_t __attribute__((ext_vector_type(8)));

__device__ __forceinline__ int waveid_in_workgroup()
{
    int wave_id;
    asm volatile("s_bfe_u32 %0, ttmp8, 0x50019" : "=s"(wave_id));
    return wave_id;
}

__global__ __launch_bounds__(256, 2) __cluster_dims__(2, 1, 1) void
wmma_kernel_kmove_2slot(const void* __restrict__ ptr_a,
                        const void* __restrict__ ptr_b,
                        void* __restrict__ ptr_c,
                        int stride_a,
                        int stride_b,
                        int stride_c)
{
    using opus::operator""_I;

    constexpr int Block_K = 128;
    constexpr int Block_M = 32;
    constexpr int Block_N = 32;
    constexpr int32_t consumerSubWarpNum = 4;
    DECLARE_NAMED_BARRIERS();

    const int      wave_id              = waveid_in_workgroup();
    const int      sub_consumer_wave_id = wave_id % 4;
    const uint32_t cluster_workgroup_id_x = __builtin_amdgcn_cluster_workgroup_id_x();
    const int32_t  c_cluster_offset_elems = static_cast<int32_t>(cluster_workgroup_id_x) * Block_N;

    const int K_STEPS = (stride_a + Block_K - 1) / Block_K;

    // ── LDS layout ─────────────────────────────────────────────────────────
    constexpr int slot_bytes_A    = Block_M * (Block_K + 8) * static_cast<int>(sizeof(fp16_t));
    constexpr int slot_bytes_B    = slot_bytes_A;
    constexpr int slot_pair_bytes = slot_bytes_A + slot_bytes_B;
    constexpr int num_slots       = 2;

    __shared__ char Smem[num_slots * slot_pair_bytes];
    const uintptr_t smembase = reinterpret_cast<uintptr_t>(Smem);

    using NoSelectedWgs = opus::seq<>;
    using WinB = tcopy_window<fp16_t,
                             Block_K, 16, 0, 0, 0,
                             1, 0, 0, 0, 1,
                             0, 0, 0,
                             0,
                             1, 5, 3,
                             NoSelectedWgs>;

    using SelectedWgs = opus::seq<0, 1>;
    using WinA = tcopy_window<fp16_t,
                             Block_K, 16, 0, 0, 0,
                             1, 0, 0, 0, 1,
                             0, 0, 0,
                             2,
                             1, 5, 3,
                             SelectedWgs>;

    // dlds(ks): alternating LDS slot delta for the 2-deep ping-pong.
    //   ks even → next slot is 0 (we were at 1) → -slot_pair_bytes
    //   ks odd  → next slot is 1 (we were at 0) → +slot_pair_bytes
    // Shared by both producer A and producer B.
    const auto dlds = [&](int ks) -> intptr_t {
        return (ks & 1) ? +static_cast<intptr_t>(slot_pair_bytes)
                        : -static_cast<intptr_t>(slot_pair_bytes);
    };

    // ────────────────────────────── producers ──────────────────────────────
    if (wave_id < 4) {
        if (wave_id < 2) {
            // asm volatile(";LOAD A START\n\t");
            __builtin_amdgcn_s_barrier_signal(-1);
            __builtin_amdgcn_s_barrier_wait(-1);

            WinA win_a;
            // Initial LDS offset in slot 0: per-wave row block.
            const uintptr_t wave_lds_off =
                static_cast<uintptr_t>(wave_id) * 16 * (Block_K + 8) * sizeof(fp16_t);

            win_a.make(
                /*lds_base       */ smembase,
                /*global_base    */ ptr_a,
                /*lds_offset_byt */ wave_lds_off,
                /*tensor_dim0    */ static_cast<uint32_t>(stride_a),
                /*tensor_dim1    */ static_cast<uint32_t>(Block_M - wave_id * 16),
                /*stride0        */ static_cast<uint64_t>(stride_a),
                /*origin0        */ 0,
                /*origin1        */ static_cast<uint32_t>(wave_id * 16));

            // Prologue: issue step 0 → slot 0
            win_a.load_to_lds();

            // Prefetch: issue step 1 → slot 1
            if (K_STEPS > 1) {
                win_a.move(Block_K, 0_I, 0_I, 0_I, 0_I, /*lds=*/+slot_pair_bytes);
                win_a.load_to_lds();
            }

            for (int ks = 0; ks < K_STEPS; ++ks) {
                if (ks < K_STEPS - 1) {
                    __builtin_amdgcn_s_wait_tensorcnt(1);   // step ks done, step ks+1 still inflight
                } else {
                    __builtin_amdgcn_s_wait_tensorcnt(0);   // last step: drain
                }
                s_barrier_join_ptr(&__nbar_1);
                __builtin_amdgcn_s_barrier_signal(1);

                s_barrier_join_ptr(&__nbar_2);
                __builtin_amdgcn_s_barrier_wait(2);

                // Prefetch step ks+2 if any.
                if (ks + 2 < K_STEPS) {
                    win_a.move(Block_K, 0_I, 0_I, 0_I, 0_I, /*lds=*/dlds(ks));
                    win_a.load_to_lds();
                }
            }
            // asm volatile(";LOAD A DONE\n\t");
        }
        else {
            // asm volatile(";LOAD B START\n\t");
            __builtin_amdgcn_s_barrier_signal(-1);
            __builtin_amdgcn_s_barrier_wait(-1);

            WinB win_b;
            const uintptr_t wave_lds_off =
                static_cast<uintptr_t>(wave_id - 2) * 16 * (Block_K + 8) * sizeof(fp16_t);

            // B's lds_base is shifted past A's region within the SAME slot.
            // slot_pair_bytes increment then naturally also advances B's slot.
            // Global wave-row offset goes to origin1:
            //   row = cluster_workgroup_id_x * Block_N + (wave_id - 2) * 16
            const uint32_t b_origin1 = static_cast<uint32_t>(
                cluster_workgroup_id_x * Block_N + (wave_id - 2) * 16);

            win_b.make(
                /*lds_base       */ smembase + slot_bytes_A,
                /*global_base    */ ptr_b,
                /*lds_offset_byt */ wave_lds_off,
                /*tensor_dim0    */ static_cast<uint32_t>(stride_b),
                /*tensor_dim1    */ static_cast<uint32_t>(Block_N - (wave_id - 2) * 16),
                /*stride0        */ static_cast<uint64_t>(stride_b),
                /*origin0        */ 0,
                /*origin1        */ b_origin1);

            win_b.load_to_lds();

            if (K_STEPS > 1) {
                win_b.move(Block_K, 0_I, 0_I, 0_I, 0_I, /*lds=*/+slot_pair_bytes);
                win_b.load_to_lds();
            }

            for (int ks = 0; ks < K_STEPS; ++ks) {
                if (ks < K_STEPS - 1) {
                    __builtin_amdgcn_s_wait_tensorcnt(1);
                } else {
                    __builtin_amdgcn_s_wait_tensorcnt(0);
                }
                s_barrier_join_ptr(&__nbar_1);
                __builtin_amdgcn_s_barrier_signal(1);

                s_barrier_join_ptr(&__nbar_2);
                __builtin_amdgcn_s_barrier_wait(2);

                if (ks + 2 < K_STEPS) {
                    win_b.move(Block_K, 0_I, 0_I, 0_I, 0_I, /*lds=*/dlds(ks));
                    win_b.load_to_lds();
                }
            }
            // asm volatile(";LOAD B DONE\n\t");
        }
    }
    // ────────────────────────────── consumers ──────────────────────────────
    else {
        // asm volatile(";CONSUMER START\n\t");
        s_barrier_init_ptr(&__nbar_1, 4);
        s_barrier_init_ptr(&__nbar_2, 4);
        __builtin_amdgcn_s_barrier_signal(-1);
        __builtin_amdgcn_s_barrier_wait(-1);

        constexpr int32_t AKSldPack    = 16 / static_cast<int32_t>(sizeof(fp16_t));
        constexpr int32_t AKSldLane    = 16 / AKSldPack;
        constexpr int32_t AMSldLane    = opus::get_warp_size() / AKSldLane;
        constexpr int32_t AMSldRepeat  = Block_M / (AMSldLane * consumerSubWarpNum / 2);
        constexpr int32_t AKSldRepeat  = Block_K / (AKSldPack * AKSldLane);
        static_assert(AKSldLane * AMSldLane == opus::get_warp_size(), "A sld lane product");

        constexpr int32_t SMemKPitch = Block_K + 8;

        auto block_sld_shape_a  = opus::make_tuple(opus::number<AMSldRepeat>{},
                                                   opus::number<consumerSubWarpNum / 2>{},
                                                   opus::number<AKSldRepeat>{},
                                                   opus::number<AKSldLane>{},
                                                   opus::number<AMSldLane>{},
                                                   opus::number<AKSldPack>{});
        auto block_sld_stride_a = opus::make_tuple(AMSldLane * SMemKPitch * consumerSubWarpNum / 2,
                                                   AMSldLane * SMemKPitch,
                                                   AKSldPack * AKSldLane,
                                                   AKSldPack,
                                                   SMemKPitch,
                                                   1_I);
        auto block_sld_win_a    = opus::make_layout<0>(block_sld_shape_a, block_sld_stride_a);

        constexpr int32_t BSldKPack   = 16 / static_cast<int32_t>(sizeof(fp16_t));
        constexpr int32_t BSldKLane   = 16 / BSldKPack;
        constexpr int32_t BSldNLane   = opus::get_warp_size() / BSldKLane;
        constexpr int32_t BSldNRepeat = Block_N / (BSldNLane * consumerSubWarpNum / 2);
        constexpr int32_t BSldKRepeat = Block_K / (BSldKPack * BSldKLane);
        static_assert(BSldKLane * BSldNLane == opus::get_warp_size(), "B sld lane product");

        auto block_sld_shape_b  = opus::make_tuple(opus::number<BSldNRepeat>{},
                                                   opus::number<consumerSubWarpNum / 2>{},
                                                   opus::number<BSldKRepeat>{},
                                                   opus::number<BSldKLane>{},
                                                   opus::number<BSldNLane>{},
                                                   opus::number<BSldKPack>{});
        auto block_sld_stride_b = opus::make_tuple(BSldNLane * SMemKPitch * consumerSubWarpNum / 2,
                                                   BSldNLane * SMemKPitch,
                                                   BSldKPack * BSldKLane,
                                                   BSldKPack,
                                                   SMemKPitch,
                                                   1_I);
        auto block_sld_win_b    = opus::make_layout<0>(block_sld_shape_b, block_sld_stride_b);

        const int32_t sub_consumer_wave_m = sub_consumer_wave_id / 2;
        const int32_t sub_consumer_wave_n = sub_consumer_wave_id % 2;
        const int32_t lane_id             = opus::lane_id();
        const int32_t a_lane_m            = lane_id / AMSldLane;
        const int32_t a_lane_n            = lane_id % AMSldLane;
        const int32_t b_lane_m            = lane_id / BSldNLane;
        const int32_t b_lane_n            = lane_id % BSldNLane;

        fp16x8_t v_c = {.0f};

        constexpr int KtileElems  = 32;
        static_assert(Block_K % KtileElems == 0, "Block_K must be multiple of 32");
        constexpr int K_WmmaTiles = Block_K / KtileElems;

        for (int ks = 0; ks < K_STEPS; ++ks) {
            s_barrier_join_ptr(&__nbar_1);
            __builtin_amdgcn_s_barrier_wait(1);

            const int32_t slot_offset = (ks & 1) * slot_pair_bytes;

            #pragma unroll
            for (int kt = 0; kt < K_WmmaTiles; ++kt) {
                const int32_t kr0 = 2 * kt;
                const int32_t kr1 = kr0 + 1;

                const int32_t a_sld_os0 =
                    slot_offset
                    + block_sld_win_a(0_I, sub_consumer_wave_m, kr0, a_lane_m, a_lane_n, 0_I)
                      * static_cast<int32_t>(sizeof(fp16_t));
                const int32_t a_sld_os1 =
                    slot_offset
                    + block_sld_win_a(0_I, sub_consumer_wave_m, kr1, a_lane_m, a_lane_n, 0_I)
                      * static_cast<int32_t>(sizeof(fp16_t));
                const int32_t b_sld_os0 =
                    slot_offset + slot_bytes_A
                    + block_sld_win_b(0_I, sub_consumer_wave_n, kr0, b_lane_m, b_lane_n, 0_I)
                      * static_cast<int32_t>(sizeof(fp16_t));
                const int32_t b_sld_os1 =
                    slot_offset + slot_bytes_A
                    + block_sld_win_b(0_I, sub_consumer_wave_n, kr1, b_lane_m, b_lane_n, 0_I)
                      * static_cast<int32_t>(sizeof(fp16_t));

                fp16x8_t sld_a0, sld_a1, sld_b0, sld_b1;
                asm volatile(
                    "ds_read_b128 %[a0], %[a_os0]\n\t"
                    "ds_read_b128 %[a1], %[a_os1]\n\t"
                    "ds_read_b128 %[b0], %[b_os0]\n\t"
                    "ds_read_b128 %[b1], %[b_os1]\n\t"
                    : [a0]"=v"(sld_a0), [a1]"=v"(sld_a1),
                      [b0]"=v"(sld_b0), [b1]"=v"(sld_b1)
                    : [a_os0]"v"(a_sld_os0), [a_os1]"v"(a_sld_os1),
                      [b_os0]"v"(b_sld_os0), [b_os1]"v"(b_sld_os1)
                    : "memory");
                asm volatile("" : : "v"(a_sld_os0), "v"(a_sld_os1),
                                    "v"(b_sld_os0), "v"(b_sld_os1) : "memory");
                asm volatile("s_wait_dscnt(0)" ::: "memory");

                reg_utils::Fp16x16Packer convertA = __builtin_bit_cast(
                    reg_utils::Fp16x16Packer, opus::array<fp16x8_t, 2>{sld_a0, sld_a1});
                reg_utils::Fp16x16Packer convertB = __builtin_bit_cast(
                    reg_utils::Fp16x16Packer, opus::array<fp16x8_t, 2>{sld_b0, sld_b1});

                __builtin_amdgcn_sched_barrier(0);
                v_c = __builtin_amdgcn_wmma_f16_16x16x32_f16(
                    0, convertB.vec16, 0, convertA.vec16, 0, v_c, false, false);
            }

            s_barrier_join_ptr(&__nbar_2);
            __builtin_amdgcn_s_barrier_signal(2);
        }

        // ── C store (unchanged) ──
        constexpr int32_t CGstNPack = 8;
        constexpr int32_t CGstNLane = 2;
        constexpr int32_t CGstMLane = 16;

        auto block_gmem_gst_shape_c  = opus::make_tuple(
            opus::number<consumerSubWarpNum / 2>{},
            opus::number<consumerSubWarpNum / 2>{},
            opus::number<CGstNLane>{},
            opus::number<CGstMLane>{},
            opus::number<CGstNPack>{});
        auto block_gmem_gst_stride_c = opus::make_tuple(
            CGstMLane * stride_c,
            CGstNPack * CGstNLane,
            CGstNPack,
            stride_c,
            1_I);
        auto block_gmem_gst_win_c = opus::make_layout<0>(block_gmem_gst_shape_c, block_gmem_gst_stride_c);

        int32_t c_offset_elem = block_gmem_gst_win_c(
            sub_consumer_wave_m,
            sub_consumer_wave_n,
            lane_id / 16,
            lane_id % 16,
            0_I) + c_cluster_offset_elems;

        *(reinterpret_cast<fp16x8_t*>(reinterpret_cast<fp16_t*>(ptr_c) + c_offset_elem)) = v_c;
        asm volatile(";CONSUMER DONE\n\t");
    }
}

// ─────────────────────────────── host harness ───────────────────────────────
template<typename T>
void rand_vector_2d(T* ptr, int m, int n, int ld, float min_val = 0.0f, float max_val = 1.0f) {
    #pragma omp parallel
    {
        std::random_device rd;
        std::mt19937 gen(rd() + omp_get_thread_num());
        std::uniform_real_distribution<float> dis(min_val, max_val);
        #pragma omp for collapse(2)
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                ptr[i * ld + j] = static_cast<T>(dis(gen));
            }
        }
    }
}

bool valid_vector(const float* ref, const float16* result, int n, float threshold) {
    int errors = 0;
    for (int i = 0; i < n; i++) {
        float diff = std::abs(ref[i] - static_cast<float>(result[i]));
        if (diff > threshold) {
            if (errors < 32) {
                printf("Error at %d: ref=%.6f, result=%.6f, diff=%.6f\n",
                       i, ref[i], static_cast<float>(result[i]), diff);
            }
            errors++;
            if (errors >= 1024) break;
        }
    }
    return errors == 0;
}

void gemm_ref(const float* a, const float* b, float* c, int m, int n, int k, int lda, int ldb, int ldc) {
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int p = 0; p < k; p++) {
                sum += a[i * lda + p] * b[j * ldb + p];
            }
            c[i * ldc + j] = sum;
        }
    }
}

void benchmark_kernel(float16* dev_a, float16* dev_b, float16* dev_c,
                      int lda, int ldb, int ldc, int m, int n, int k,
                      int warmup = 5, int iterations = 20) {
    for (int i = 0; i < warmup; ++i) {
        wmma_kernel_kmove_2slot<<<dim3(2, 1, 1), 256>>>(dev_a, dev_b, dev_c, lda, ldb, ldc);
        CHECK_HIP_KERNEL_LAUNCH();
    }

    hipEvent_t start, stop;
    CHECK_HIP(hipEventCreate(&start));
    CHECK_HIP(hipEventCreate(&stop));
    CHECK_HIP(hipDeviceSynchronize());
    CHECK_HIP(hipEventRecord(start));
    for (int i = 0; i < iterations; ++i) {
        wmma_kernel_kmove_2slot<<<dim3(2, 1, 1), 256>>>(dev_a, dev_b, dev_c, lda, ldb, ldc);
        CHECK_HIP_KERNEL_LAUNCH();
    }
    CHECK_HIP(hipEventRecord(stop));
    CHECK_HIP(hipEventSynchronize(stop));

    float total_ms = 0;
    CHECK_HIP(hipEventElapsedTime(&total_ms, start, stop));
    CHECK_HIP(hipEventDestroy(start));
    CHECK_HIP(hipEventDestroy(stop));

    const float avg_ms = total_ms / iterations;
    const float tflops = 2.0f * m * n * k / 1.0e9f / avg_ms;
    const int k_steps = (k + 127) / 128;
    printf("Kernel Performance: K=%d (%d steps, 2-slot), avg_time=%.4f ms, %.2f TFlops\n",
           k, k_steps, avg_ms, tflops);
}

int main(int argc, char** argv) {
    int m = 32;
    int n = 64;
    int k = 256;

    for (int i = 1; i < argc; ++i) {
        const char* arg = argv[i];
        if ((std::strcmp(arg, "-m") == 0 || std::strcmp(arg, "--m") == 0) && i + 1 < argc) {
            m = std::atoi(argv[++i]);
        } else if ((std::strcmp(arg, "-n") == 0 || std::strcmp(arg, "--n") == 0) && i + 1 < argc) {
            n = std::atoi(argv[++i]);
        } else if ((std::strcmp(arg, "-k") == 0 || std::strcmp(arg, "--k") == 0) && i + 1 < argc) {
            k = std::atoi(argv[++i]);
        }
    }

    if (m != 32 || n != 64 || k <= 0) {
        fprintf(stderr,
                "This kernel is specialized for m=32, n=64, k > 0. "
                "Got m=%d n=%d k=%d.\n", m, n, k);
        return 1;
    }

    int lda = k;
    int ldb = k;
    int ldc = n;

    auto host_a = std::make_unique<float[]>(lda * m);
    auto host_b = std::make_unique<float[]>(ldb * n);
    auto host_c = std::make_unique<float[]>(ldc * m);
    auto fp16_a = std::make_unique<float16[]>(lda * m);
    auto fp16_b = std::make_unique<float16[]>(ldb * n);
    auto fp16_c = std::make_unique<float16[]>(ldc * m);

    rand_vector_2d(host_a.get(), m, k, lda, 0.0f, 1.0f);
    rand_vector_2d(host_b.get(), n, k, ldb, -0.5f, 0.5f);

    for (int i = 0; i < lda * m; i++) fp16_a[i] = __float2half_rn(host_a[i]);
    for (int i = 0; i < ldb * n; i++) fp16_b[i] = __float2half_rn(host_b[i]);

    float16 *dev_a, *dev_b, *dev_c;
    CHECK_HIP(hipMalloc(&dev_a, lda * m * sizeof(float16)));
    CHECK_HIP(hipMalloc(&dev_b, ldb * n * sizeof(float16)));
    CHECK_HIP(hipMalloc(&dev_c, ldc * m * sizeof(float16)));

    CHECK_HIP(hipMemcpy(dev_a, fp16_a.get(), lda * m * sizeof(float16), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(dev_b, fp16_b.get(), ldb * n * sizeof(float16), hipMemcpyHostToDevice));

    const int k_steps = (k + 127) / 128;
    printf("m:%d, n:%d, k:%d (K_STEPS=%d, last_tile=%d, 2-slot), lda:%d, ldb:%d, ldc:%d\n",
           m, n, k, k_steps,
           (k - (k_steps - 1) * 128),
           lda, ldb, ldc);

    gemm_ref(host_a.get(), host_b.get(), host_c.get(), m, n, k, lda, ldb, ldc);

    wmma_kernel_kmove_2slot<<<dim3(2, 1, 1), 256>>>(dev_a, dev_b, dev_c, lda, ldb, ldc);
    CHECK_HIP_KERNEL_LAUNCH();

    CHECK_HIP(hipMemcpy(fp16_c.get(), dev_c, ldc * m * sizeof(float16), hipMemcpyDeviceToHost));

    float threshold = std::max(1e-2f, 5e-5f * static_cast<float>(k));
    bool valid = valid_vector(host_c.get(), fp16_c.get(), m * n, threshold);
    printf("[32x64xK=%d, Tcopy K-move 2-slot, 2WG Cluster] %s\n",
           k, valid ? "VALID" : "FAIL");

    printf("\n");
    benchmark_kernel(dev_a, dev_b, dev_c, lda, ldb, ldc, m, n, k);

    CHECK_HIP(hipFree(dev_a));
    CHECK_HIP(hipFree(dev_b));
    CHECK_HIP(hipFree(dev_c));

    return valid ? 0 : 1;
}
