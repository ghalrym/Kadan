#pragma once
#include <cstddef>
#include <cstdint>

#include <cuda_runtime_api.h>

// Batch-one GPT-OSS decoder launchers on the legacy default stream. Pointers
// may be device memory or mapped host memory; the owner bounds every buffer.
namespace kadan::cuda::detail::gpt_oss {
cudaError_t embed(const std::uint16_t* table, unsigned token, float* x);
// FP32 RMSNorm with a BF16 weight, eps 1e-5.
cudaError_t norm(const float* x, const std::uint16_t* weight, float* y);
// y = W x + bias, or y += W x + bias. Row-major BF16 W; cols % 8 == 0, <= 4096.
cudaError_t linear(
    const std::uint16_t* weight,
    const std::uint16_t* bias,
    const float* x,
    float* y,
    std::size_t rows,
    std::size_t cols,
    bool accumulate
);
// Half-split YaRN rotary on the 64 query and 8 key heads of a fused QKV row.
cudaError_t
rotary(float* qkv, const float* inverse_frequency, float scaling, std::size_t position);
// Rounds K/V to BF16 into one cache slot.
cudaError_t
store(const float* qkv, std::uint16_t* keys, std::uint16_t* values, std::size_t slot);
// Grouped-query attention over count cached keys with per-head sink logits.
cudaError_t attention(
    const float* qkv,
    const std::uint16_t* keys,
    const std::uint16_t* values,
    const std::uint16_t* sinks,
    std::size_t count,
    float* partial,
    float* output
);
// Top-4 experts (lower id breaks ties) and FP32 softmax over their logits.
cudaError_t route(
    const float* logits,
    std::size_t experts,
    unsigned* indices,
    float* weights,
    unsigned* status
);
// Selected experts' clamped SwiGLU, then their routed sum added to residual.
// experts[e] addresses [gate_up blocks][scales][down blocks][scales].
cudaError_t experts(
    const std::uint8_t* const* experts,
    const std::uint16_t* gate_up_bias,
    const std::uint16_t* down_bias,
    const unsigned* indices,
    const float* weights,
    const float* x,
    float* activation,
    float* residual
);
}  // namespace kadan::cuda::detail::gpt_oss
