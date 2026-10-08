#pragma once
#include <cuda_runtime.h>
#include <cstddef>
namespace kadan::cuda::detail {
// IEEE addition cannot recover from Inf/NaN without explicit sanitization.
// A finite final sum therefore proves every intermediate sum was finite. Keep
// exactly the original add order; replay only failures to preserve reset-to-zero
// behavior and diagnostic outputs, as well as the accumulated numeric flag.
__device__ inline float ordered_finite_sum(const float* values,std::size_t n,unsigned* status){
    float sum=0;for(std::size_t j=0;j<n;++j)sum=__fadd_rn(sum,values[j]);
    if(isfinite(sum))return sum;
    sum=0;for(std::size_t j=0;j<n;++j){sum=__fadd_rn(sum,values[j]);if(!isfinite(sum)){atomicOr(status,1u);sum=0;}}
    return sum;
}
}
