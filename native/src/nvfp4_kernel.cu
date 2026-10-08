#include "nvfp4_kernel.cuh"
#include "nvfp4_numeric.hpp"
#include <cuda_runtime.h>

namespace {
__device__ float bf16_weight(float x) { unsigned b=__float_as_uint(x);b+=0x7fffu+((b>>16)&1u);return __uint_as_float(b&0xffff0000u); }

__device__ float bf16_activation(float value,unsigned* status){
    if(!isfinite(value)){atomicOr(status,1U);value=0;}
    value=bf16_weight(value);
    if(!isfinite(value)){atomicOr(status,1U);value=0;}
    return value;
}

__device__ __forceinline__ float fp4(unsigned code) {
    const unsigned exponent = (code >> 1) & 3;
    const float magnitude = exponent == 0 ? float(code & 1) * 0.5F
        : float(2 + (code & 1)) * float(1U << (exponent - 1)) * 0.5F;
    return code & 8 ? -magnitude : magnitude;
}
__device__ __forceinline__ float positive_fp8(unsigned code) {
    // Host validation admits only finite nonnegative E4M3FN block scales.
    const unsigned exponent = code >> 3;
    const unsigned fraction = code & 7;
    return exponent == 0 ? float(fraction) * 0.001953125F
        : __uint_as_float(((exponent + 120U) << 23) | (fraction << 20));
}
__device__ __forceinline__ float warp_sum(float value) {
    for (int delta = 16; delta != 0; delta /= 2)
        value += __shfl_down_sync(0xffffffffU, value, delta);
    return value;
}
// Original scalar CUDA implementation. Four warps cooperate on one row; each
// lane handles packed byte pairs, with in-register dequantization and FP32 FMA.
// Padding/tails do not mask warp participation. No dense weight intermediates.
template<bool BF16, bool Bounded, bool Accumulate=false, bool Pair=false>
__global__ void nvfp4_matvec(const std::uint8_t* weights, const std::uint8_t* scales,
                            float global, const float* input, float* output,
                            unsigned* status, std::size_t columns, Nvfp4Accumulation accumulation={}, Nvfp4Pair pair={}) {
    if constexpr(Pair) if(blockIdx.y){weights=pair.weights;scales=pair.scales;global=pair.global;output=pair.output;}
    const std::size_t row = blockIdx.x;
    const std::size_t packed_columns = columns / 2;
    const auto* row_weights = weights + row * packed_columns;
    const auto* row_scales = scales + row * (columns / 16);
    float sum = 0;
    bool invalid = false;
    for (std::size_t byte = threadIdx.x; byte < packed_columns; byte += blockDim.x) {
        const unsigned packed = row_weights[byte];
        const float scale = positive_fp8(row_scales[byte / 8]);
        // FP4 * E4M3 is exact in FP32. Explicit round-to-nearest global multiply
        // matches the CPU decoded-weight rounding before the dot product.
        const float local_low = fp4(packed & 15) * scale;
        const float local_high = fp4(packed >> 4) * scale;
        if constexpr(!Bounded)
            invalid |= kadan::cuda::detail::weight_overflows(local_low, global)
                    || kadan::cuda::detail::weight_overflows(local_high, global);
        float low = __fmul_rn(local_low, global);
        float high = __fmul_rn(local_high, global);
        if constexpr(BF16) { low=bf16_weight(low);high=bf16_weight(high); }
        invalid |= !isfinite(low) || !isfinite(high);
        sum = fmaf(low, input[byte * 2], sum);
        sum = fmaf(high, input[byte * 2 + 1], sum);
    }
    if (invalid) atomicOr(status, 1U);
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
            if constexpr(Accumulate){
                float weight=0;for(std::size_t rank=0;rank<accumulation.top_k;++rank)
                    if(accumulation.selected[rank]==accumulation.expert)weight=accumulation.weights[rank];
                const float down=bf16_activation(sum,status);
                const float weighted=bf16_activation(__fmul_rn(down,weight),status);
                accumulation.accumulator[row]=bf16_activation(__fadd_rn(accumulation.accumulator[row],weighted),status);
            }
        }
    }
}
}
cudaError_t kadan_launch_nvfp4(const std::uint8_t* weights, const std::uint8_t* scales,
                             float global, const float* input, float* output,
                             unsigned* status, std::size_t rows, std::size_t columns) {
    if(kadan::cuda::detail::bounded_nvfp4_global(global))
        nvfp4_matvec<false,true><<<static_cast<unsigned>(rows),128,0,cudaStreamLegacy>>>(weights,scales,global,input,output,status,columns);
    else
        nvfp4_matvec<false,false><<<static_cast<unsigned>(rows),128,0,cudaStreamLegacy>>>(weights,scales,global,input,output,status,columns);
    return cudaGetLastError();
}

cudaError_t kadan_launch_nvfp4_bf16(const std::uint8_t* weights, const std::uint8_t* scales,
                             float global, const float* input, float* output,
                             unsigned* status, std::size_t rows, std::size_t columns) {
    if(kadan::cuda::detail::bounded_nvfp4_global(global))
        nvfp4_matvec<true,true><<<static_cast<unsigned>(rows),128,0,cudaStreamLegacy>>>(weights,scales,global,input,output,status,columns);
    else
        nvfp4_matvec<true,false><<<static_cast<unsigned>(rows),128,0,cudaStreamLegacy>>>(weights,scales,global,input,output,status,columns);
    return cudaGetLastError();
}

cudaError_t kadan_launch_nvfp4_pair(const std::uint8_t* weights,const std::uint8_t* scales,float global,const float* input,float* output,unsigned* status,std::size_t rows,std::size_t columns,bool bf16,Nvfp4Pair pair){
    const bool bounded=kadan::cuda::detail::bounded_nvfp4_global(global)&&kadan::cuda::detail::bounded_nvfp4_global(pair.global);
    if(bf16&&bounded) nvfp4_matvec<true,true,false,true><<<dim3(static_cast<unsigned>(rows),2),128,0,cudaStreamLegacy>>>(weights,scales,global,input,output,status,columns,{},pair);
    else if(bf16) nvfp4_matvec<true,false,false,true><<<dim3(static_cast<unsigned>(rows),2),128,0,cudaStreamLegacy>>>(weights,scales,global,input,output,status,columns,{},pair);
    else if(bounded) nvfp4_matvec<false,true,false,true><<<dim3(static_cast<unsigned>(rows),2),128,0,cudaStreamLegacy>>>(weights,scales,global,input,output,status,columns,{},pair);
    else nvfp4_matvec<false,false,false,true><<<dim3(static_cast<unsigned>(rows),2),128,0,cudaStreamLegacy>>>(weights,scales,global,input,output,status,columns,{},pair);
    return cudaGetLastError();
}

cudaError_t kadan_launch_nvfp4_accumulate(const std::uint8_t* weights,const std::uint8_t* scales,float global,const float* input,float* output,unsigned* status,std::size_t rows,std::size_t columns,bool bf16,Nvfp4Accumulation accumulation){
    const bool bounded=kadan::cuda::detail::bounded_nvfp4_global(global);
    if(bf16&&bounded) nvfp4_matvec<true,true,true,false><<<static_cast<unsigned>(rows),128,0,cudaStreamLegacy>>>(weights,scales,global,input,output,status,columns,accumulation,{});
    else if(bf16) nvfp4_matvec<true,false,true,false><<<static_cast<unsigned>(rows),128,0,cudaStreamLegacy>>>(weights,scales,global,input,output,status,columns,accumulation,{});
    else if(bounded) nvfp4_matvec<false,true,true,false><<<static_cast<unsigned>(rows),128,0,cudaStreamLegacy>>>(weights,scales,global,input,output,status,columns,accumulation,{});
    else nvfp4_matvec<false,false,true,false><<<static_cast<unsigned>(rows),128,0,cudaStreamLegacy>>>(weights,scales,global,input,output,status,columns,accumulation,{});
    return cudaGetLastError();
}
