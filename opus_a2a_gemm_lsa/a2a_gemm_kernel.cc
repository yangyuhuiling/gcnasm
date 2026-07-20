#include <opus/hip_minimal.hpp>
#include "gemm_defs.h"

#ifndef __HIP_DEVICE_COMPILE__
template<typename Traits, int Mode>
__global__ void a2a_gemm_lsa_kernel(opus_a2a_gemm_kargs kargs) {}
template __global__ void a2a_gemm_lsa_kernel<
    opus_gemm_traits<512, 256, 256, 64, bf16_t, bf16_t, bf16_t, float>, 0>(opus_a2a_gemm_kargs);
template __global__ void a2a_gemm_lsa_kernel<
    opus_gemm_traits<512, 256, 256, 64, bf16_t, bf16_t, bf16_t, float>, 1>(opus_a2a_gemm_kargs);
#else
#include "a2a_gemm_kernel_template.hpp"
template __global__ void a2a_gemm_lsa_kernel<
    opus_gemm_traits<512, 256, 256, 64, bf16_t, bf16_t, bf16_t, float>, 0>(opus_a2a_gemm_kargs);
template __global__ void a2a_gemm_lsa_kernel<
    opus_gemm_traits<512, 256, 256, 64, bf16_t, bf16_t, bf16_t, float>, 1>(opus_a2a_gemm_kargs);
#endif
