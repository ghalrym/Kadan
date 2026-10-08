#pragma once
#include "kadan/moe.hpp"
#include <cuda_runtime_api.h>
namespace kadan::cuda::detail {
struct MoeBuffers {
    const std::uint16_t *router,*shared_gate;
    float *logits,*probabilities,*top_weights,*gate,*up,*activation,*down,*accumulator,*shared,*result,*shared_factor;
    unsigned *selected,*status;
};
cudaError_t moe_route(moe::Config,MoeBuffers,const float*);
cudaError_t moe_activate(std::size_t,MoeBuffers);
cudaError_t moe_accumulate(moe::Config,MoeBuffers,unsigned expert);
cudaError_t moe_finish(moe::Config,MoeBuffers,float*);
} // namespace kadan::cuda::detail
