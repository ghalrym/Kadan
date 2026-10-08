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
// Four rows share a block, each retaining its original four-warp reduction.
// Decode once into a block-local table (one per row for row scales). Only
// accessed entries affect status; unused overflowing entries must not flag.
// Finite bytes/scales and exact decoded-weight range were checked at admission.
template<bool BF16>
__global__ void fp8_matvec(const std::uint8_t* weights, const float* scales,
                          bool row_scales, const float* input, float* output,
                          unsigned* status, std::size_t columns, std::size_t rows) {
    const unsigned group=threadIdx.x/128,tid=threadIdx.x%128;
    const std::size_t row = blockIdx.x*4+group;
    const float scale = row_scales?(row<rows?scales[row]:1.f):scales[0];
    __shared__ float tables[4][256];float* table=tables[row_scales?group:0];
    const unsigned start=row_scales?tid:threadIdx.x,step=row_scales?128:blockDim.x;
    for(unsigned code=start;code<256;code+=step){float value=__fmul_rn(kadan::cuda::detail::finite_e4m3fn(code),scale);if constexpr(BF16)value=bf16_weight(value);table[code]=value;}
    __syncthreads();
    float sum = 0;bool invalid=false;
    for (std::size_t col = tid; row<rows && col < columns; col += 128) {
        float weight=table[weights[row*columns+col]];
        invalid|=!isfinite(weight);
        sum = fmaf(weight, input[col], sum);
    }
    if(invalid)atomicOr(status,1U);
    sum = warp_sum(sum);
    __shared__ float partial[16];
    const unsigned lane = tid & 31;
    const unsigned warp = tid >> 5;
    if (lane == 0) partial[group*4+warp] = sum;
    __syncthreads();
    if (warp == 0) {
        sum = warp_sum(lane < 4 ? partial[group*4+lane] : 0.0F);
        if (lane == 0 && row<rows) {
            if (!isfinite(sum)) atomicOr(status, 2U);
            output[row] = sum;
        }
    }
}
}
cudaError_t kadan_launch_fp8(const std::uint8_t* weights, const float* scales,
                            bool row_scales, const float* input, float* output,
                            unsigned* status, std::size_t rows, std::size_t columns) {
    fp8_matvec<false><<<static_cast<unsigned>((rows+3)/4), 512, 0, cudaStreamLegacy>>>(
        weights, scales, row_scales, input, output, status, columns, rows);
    return cudaGetLastError();
}

cudaError_t kadan_launch_fp8_bf16(const std::uint8_t* weights, const float* scales,
                            bool row_scales, const float* input, float* output,
                            unsigned* status, std::size_t rows, std::size_t columns) {
    fp8_matvec<true><<<static_cast<unsigned>((rows+3)/4), 512, 0, cudaStreamLegacy>>>(
        weights, scales, row_scales, input, output, status, columns, rows);
    return cudaGetLastError();
}
