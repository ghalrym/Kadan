#include <cuda_runtime.h>
#include "fp8_kernel.cuh"
#include "fp8_numeric.hpp"

namespace {
__device__ __forceinline__ float warp_sum(float value) {
    for (int delta = 16; delta != 0; delta /= 2)
        value += __shfl_down_sync(0xffffffffU, value, delta);
    return value;
}
// Original scalar CUDA projection: four warps per row, no dense weight scratch.
// Finite bytes/scales and exact decoded-weight range were checked at admission.
__global__ void fp8_matvec(const std::uint8_t* weights, const float* scales,
                          bool row_scales, const float* input, float* output,
                          unsigned* status, std::size_t columns) {
    const std::size_t row = blockIdx.x;
    const float scale = scales[row_scales ? row : 0];
    float sum = 0;
    for (std::size_t col = threadIdx.x; col < columns; col += blockDim.x) {
        const float weight = __fmul_rn(kadan::cuda::detail::finite_e4m3fn(weights[row * columns + col]), scale);
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
cudaError_t kadan_launch_fp8(const std::uint8_t* weights, const float* scales,
                            bool row_scales, const float* input, float* output,
                            unsigned* status, std::size_t rows, std::size_t columns) {
    fp8_matvec<<<static_cast<unsigned>(rows), 128, 0, cudaStreamLegacy>>>(
        weights, scales, row_scales, input, output, status, columns);
    return cudaGetLastError();
}
