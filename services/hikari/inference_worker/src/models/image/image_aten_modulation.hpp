#pragma once
#include <ATen/ATen.h>

namespace kadan::image::detail {
// FLUX.2 shared AdaLN modulation: each parameter set is (shift, scale, gate).
// Normalize and apply (1 + scale) * norm(x) + shift in FP32, then cast once.
struct Flux2ModulationSet {
    at::Tensor shift, scale, gate;
};

struct Flux2DoubleModulation {
    Flux2ModulationSet attention, mlp;

    explicit Flux2DoubleModulation(const at::Tensor& projected) {
        auto chunks = projected.unsqueeze(1).chunk(6, -1);
        attention = {
            chunks[0].to(at::kFloat),
            chunks[1].to(at::kFloat).add(1),
            chunks[2]
        };
        mlp = {
            chunks[3].to(at::kFloat),
            chunks[4].to(at::kFloat).add(1),
            chunks[5]
        };
    }
};

struct Flux2SingleModulation {
    Flux2ModulationSet parallel;

    explicit Flux2SingleModulation(const at::Tensor& projected) {
        auto chunks = projected.unsqueeze(1).chunk(3, -1);
        parallel = {
            chunks[0].to(at::kFloat),
            chunks[1].to(at::kFloat).add(1),
            chunks[2]
        };
    }
};

inline at::Tensor flux2_modulated_norm(
    const at::Tensor& x, const at::Tensor& shift, const at::Tensor& scale
) {
    auto normalized = at::layer_norm(x.to(at::kFloat), {x.size(-1)}, {}, {}, 1e-6);
    return normalized.mul_(scale).add_(shift).to(x.scalar_type());
}

inline at::Tensor
flux2_gated_residual(at::Tensor x, const at::Tensor& delta, const at::Tensor& gate) {
    return x.addcmul_(delta, gate);
}
}  // namespace kadan::image::detail
