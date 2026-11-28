// CPU arithmetic primitives only: no model, checkpoint, forward pass or CUDA.
#include "../src/models/image/image_aten_modulation.hpp"

#include <iostream>
#include <source_location>

namespace {
void check(bool value, std::source_location at = std::source_location::current()) {
    if (!value)
        throw std::runtime_error(
            "image_aten_modulation_test:" + std::to_string(at.line())
        );
}

void double_and_single_modulation() {
    using namespace kadan::image::detail;
    const std::vector<float> row_vals{-3.f, -1.f, 1.f, 3.f};
    std::vector<std::uint16_t> x(2 * 3 * 4);
    for (std::size_t i = 0; i < x.size(); ++i)
        x[i] = float_to_bf16(row_vals[i % 4]);

    std::vector<std::uint16_t> projected(24, float_to_bf16(0.f));
    for (std::size_t i = 0; i < 4; ++i) {
        projected[i] = float_to_bf16(0.25f);        // shift_msa
        projected[4 + i] = float_to_bf16(0.5f);     // scale_msa -> 1.5
        projected[8 + i] = float_to_bf16(0.5f);     // gate_msa
        projected[12 + i] = float_to_bf16(-0.25f);  // shift_mlp
        projected[16 + i] = float_to_bf16(1.0f);    // scale_mlp -> 2.0
        projected[20 + i] = float_to_bf16(-0.5f);   // gate_mlp
    }

    Flux2CpuDoubleModulation dmod(projected);
    check(dmod.attention.shift[0] == 0.25f);
    check(dmod.attention.scale[0] == 1.5f);
    check(bf16_to_float(dmod.attention.gate[0]) == 0.5f);
    check(dmod.mlp.shift[0] == -0.25f);
    check(dmod.mlp.scale[0] == 2.0f);
    check(bf16_to_float(dmod.mlp.gate[0]) == -0.5f);

    const auto normed = flux2_modulated_norm_cpu(
        x, 6, 4, dmod.attention.shift, dmod.attention.scale
    );
    check(normed.size() == x.size());
    for (std::size_t i = 0; i < 12; ++i)
        check(normed[i] == normed[12 + i]);

    std::vector<std::uint16_t> single_proj(12, float_to_bf16(0.f));
    for (std::size_t i = 0; i < 4; ++i) {
        single_proj[i] = float_to_bf16(0.1f);
        single_proj[4 + i] = float_to_bf16(0.2f);
        single_proj[8 + i] = float_to_bf16(0.5f);
    }
    Flux2CpuSingleModulation smod(single_proj);
    std::vector<std::uint16_t> base(2 * 3 * 4, float_to_bf16(1.0f));
    std::vector<std::uint16_t> delta(2 * 3 * 4, float_to_bf16(2.0f));
    flux2_gated_residual_cpu(base, delta, smod.parallel.gate);
    check(bf16_to_float(base[0]) == 2.0f);
}
}  // namespace

int main() {
    double_and_single_modulation();
    std::cout
        << "image modulation arithmetic units passed (CPU only, no inference)\n";
}
