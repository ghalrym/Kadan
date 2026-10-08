#include <cuda_runtime.h>
#include "fp8_kernel.cuh"
#include "fp8_numeric.hpp"

namespace {
__device__ float bf16_weight(float x) { unsigned b=__float_as_uint(x);b+=0x7fffu+((b>>16)&1u);return __uint_as_float(b&0xffff0000u); }

__device__ __forceinline__ float warp_sum(float value) {
    for (int delta = 16; delta != 0; delta /= 2)
        value += __shfl_down_sync(0xffffffffU, value, delta);
    return value;
}
// Original scalar CUDA projection: four warps per row, no dense weight scratch.
// Finite bytes/scales and exact decoded-weight range were checked at admission.
template<bool BF16>
__global__ void fp8_matvec(const std::uint8_t* weights, const float* scales,
                          bool row_scales, const float* input, float* output,
                          unsigned* status, std::size_t columns) {
    const std::size_t row = blockIdx.x;
    const float scale = scales[row_scales ? row : 0];
    float sum = 0;
    for (std::size_t col = threadIdx.x; col < columns; col += blockDim.x) {
        float weight = __fmul_rn(kadan::cuda::detail::finite_e4m3fn(weights[row * columns + col]), scale);
        if constexpr(BF16) weight=bf16_weight(weight);
        if (!isfinite(weight)) atomicOr(status,1U);
        sum = fmaf(weight, input[col], sum);
    }
    sum = warp_sum(sum);
    __shared__ float partial[4];
    const unsigned lane = threadIdx.x & 31;
    const unsigned warp = threadIdx.x >> 5;
    if (lane == 0) partial[warp] = sum;
    __syncthreads();
    if (warp == 0) {
        sum = warp_sum(lane < 4 ? partial[lane] : 0.0F);
        if (lane == 0) {
            if (!isfinite(sum)) atomicOr(status, 2U);
            output[row] = sum;
        }
    }
}
}
cudaError_t old_launch_fp8(const std::uint8_t* weights, const float* scales,
                            bool row_scales, const float* input, float* output,
                            unsigned* status, std::size_t rows, std::size_t columns) {
    fp8_matvec<false><<<static_cast<unsigned>(rows), 128, 0, cudaStreamLegacy>>>(
        weights, scales, row_scales, input, output, status, columns);
    return cudaGetLastError();
}

cudaError_t old_launch_fp8_bf16(const std::uint8_t* weights, const float* scales,
                            bool row_scales, const float* input, float* output,
                            unsigned* status, std::size_t rows, std::size_t columns) {
    fp8_matvec<true><<<static_cast<unsigned>(rows), 128, 0, cudaStreamLegacy>>>(
        weights, scales, row_scales, input, output, status, columns);
    return cudaGetLastError();
}
