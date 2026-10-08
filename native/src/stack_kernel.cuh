#pragma once
#include <cuda_runtime_api.h>
#include <cstddef>
#include <cstdint>
namespace kadan::cuda::detail {
cudaError_t stack_embedding(std::size_t,const std::uint16_t*,unsigned,float*);
cudaError_t stack_select(std::size_t,float*,unsigned*,unsigned*);
}
