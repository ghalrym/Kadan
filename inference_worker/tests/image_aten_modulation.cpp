// CPU arithmetic primitives only: no model, checkpoint, forward pass or CUDA.
#include "../src/models/image/image_aten_modulation.hpp"

#include <array>
#include <iostream>
#include <source_location>

namespace {
void check(bool value, std::source_location at = std::source_location::current()) {
    if (!value)
        throw std::runtime_error(
            "image_aten_modulation_test:" + std::to_string(at.line())
        );
}

void normalization_rounding_and_rows() {
    using namespace kadan::image::detail;
    auto x = at::tensor({-3.f, -1.f, 1.f, 3.f}).to(at::kBFloat16).repeat({2, 1});
    auto projected = at::zeros({2, 16}, at::kBFloat16);
    projected[0].slice(0, 0, 4).fill_(.003f);
    projected[1].slice(0, 0, 4).fill_(.067f);
    projected.slice(1, 8, 12).fill_(.13f);
    projected.slice(1, 4, 8).fill_(.5f);
    projected.slice(1, 12).fill_(-.5f);
    AtenModulation mod(projected);
    check(mod.attention_scale.scalar_type() == at::kFloat);
    check(mod.attention_scale[0][0].item<float>() == 1.0030059814453125f);
    check(mod.mlp_scale[0][0].item<float>() == 1.1298828125f);
    check(mod.attention_gate[0][0].item<float>() == .462890625f);
    check(mod.mlp_gate[0][0].item<float>() == -.462890625f);
    // Golden BF16 words for FP32 norm/scale with one final cast. The old path
    // differs on the two inner target values and all four prefix values.
    const std::array<std::uint16_t, 8> expected{
        49079, 48884, 16116, 16311, 49068, 48870, 16102, 16300
    };
    auto result = aten_modulated_norm(x, mod.attention_scale, 1).contiguous();
    check(result.scalar_type() == at::kBFloat16);
    const auto* words = result.const_data_ptr<at::BFloat16>();
    for (std::size_t i = 0; i < expected.size(); ++i)
        check(words[i].x == expected[i]);
    auto target_only = aten_modulated_norm(x.slice(0, 1), mod.attention_scale, 0);
    check(at::equal(target_only, result.slice(0, 1)));
    auto constant = at::full({1, 4}, 3.f, at::kBFloat16);
    check(
        aten_modulated_norm(constant, mod.attention_scale, 0).eq(0).all().item<bool>()
    );
    check(x[0][0].item<float>() == -3.f);  // Norm leaves the residual input intact.
}

void residual_rounding_and_rows() {
    using kadan::image::detail::aten_gated_residual;
    auto x = at::ones({3, 1}, at::kBFloat16);
    auto delta = at::full({3, 1}, 1.1f, at::kBFloat16);
    auto gate = at::tensor({.00355f, -.00355f}).to(at::kBFloat16).reshape({2, 1});
    auto result = aten_gated_residual(x, delta, gate, 1);
    check(result.data_ptr() == x.data_ptr());
    check(result[0][0].item<float>() == .99609375f);
    check(result[1][0].item<float>() == 1.0078125f);
    check(result[2][0].item<float>() == 1.0078125f);
    // The previous separately rounded product yields 1.0 on the target rows.
    auto target = at::ones({1, 1}, at::kBFloat16);
    check(
        aten_gated_residual(target, delta.slice(0, 1, 2), gate, 0)[0][0].item<float>()
        == 1.0078125f
    );
}
}  // namespace

int main() {
    normalization_rounding_and_rows();
    residual_rounding_and_rows();
    std::cout
        << "image ATen modulation arithmetic units passed (CPU only, no inference)\n";
}
