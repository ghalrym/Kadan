#pragma once
#include <ATen/ATen.h>

namespace kadan::image::detail {
// Comfy Kitchen v0.2.37 (Apache-2.0), commit
// be003b7c23c5b01328657955b8bc5d3f073d868e, CUDA ops/adaln.cu:
// normalize and apply (1 + scale) in FP32, then cast once. This composes
// ATen operations independently; it does not reproduce the CUDA reduction tree.
struct AtenModulation {
    at::Tensor attention_scale, attention_gate, mlp_scale, mlp_gate;

    explicit AtenModulation(const at::Tensor& projected) {
        const auto width = projected.size(-1) / 4;
        attention_scale = projected.slice(-1, 0, width).to(at::kFloat).add(1);
        attention_gate = projected.slice(-1, width, 2 * width).tanh();
        mlp_scale = projected.slice(-1, 2 * width, 3 * width).to(at::kFloat).add(1);
        mlp_gate = projected.slice(-1, 3 * width).tanh();
    }
};

inline at::Tensor
aten_modulated_norm(const at::Tensor& x, const at::Tensor& scale, std::int64_t prefix) {
    auto normalized = at::layer_norm(x.to(at::kFloat), {x.size(-1)}, {}, {}, 1e-6);
    normalized.slice(0, prefix).mul_(scale[0]);
    if (prefix)
        normalized.slice(0, 0, prefix).mul_(scale[1]);
    return normalized.to(x.scalar_type());
}

// The caller owns x. addcmul performs the residual product and addition with
// one output rounding, rather than materializing a rounded BF16 product.
inline at::Tensor aten_gated_residual(
    at::Tensor x, const at::Tensor& delta, const at::Tensor& gate, std::int64_t prefix
) {
    x.slice(0, prefix).addcmul_(delta.slice(0, prefix), gate[0]);
    if (prefix)
        x.slice(0, 0, prefix).addcmul_(delta.slice(0, 0, prefix), gate[1]);
    return x;
}
}  // namespace kadan::image::detail
