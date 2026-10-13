#pragma once
#include <cstddef>
#include <cstdint>

#include <cuda_runtime_api.h>

namespace kadan::cuda::detail {
cudaError_t
decoder_norm(std::size_t, float, const std::uint16_t*, const float*, float*, unsigned*);
cudaError_t
decoder_residual(std::size_t, const float*, const float*, float*, unsigned*);
}  // namespace kadan::cuda::detail
