#pragma once
#include "kadan/linear_attention.hpp"
#include <cuda_runtime_api.h>
namespace kadan::cuda::detail {
struct LinearBuffers {
    const std::uint16_t *input_norm,*conv_weight,*a_weight,*b_weight,*a_log,*dt_bias,*output_norm;
    float *normalized,*qkv,*z,*a,*b,*core,*gated,*projected;
    std::uint16_t* convolution;float* recurrent;unsigned* status;
};
cudaError_t linear_normalize(linear::Config c,LinearBuffers b,const float* input);
cudaError_t linear_core(linear::Config c,LinearBuffers b);
cudaError_t linear_residual(linear::Config c,LinearBuffers b,const float* input,float* output);
} // namespace kadan::cuda::detail
