#pragma once
#include "kadan/full_attention.hpp"
#include <cuda_runtime_api.h>
namespace kadan::cuda::detail {
struct FullBuffers {
    const std::uint16_t *input_norm,*query_norm,*key_norm;
    const float* frequencies;
    float *normalized,*qg,*key,*value,*query,*gate,*probabilities,*core,*gated,*projected;
    std::uint16_t *keys,*values;unsigned* status;
};
cudaError_t full_normalize(full::Config,FullBuffers,const float*);
// Five launches: prepare Q/K/G, append KV, scores, serial softmax, values/gate.
cudaError_t full_prepare(full::Config,FullBuffers,std::size_t);
cudaError_t full_core(full::Config,FullBuffers,std::size_t position);
cudaError_t full_residual(full::Config,FullBuffers,const float*,float*);
} // namespace kadan::cuda::detail
