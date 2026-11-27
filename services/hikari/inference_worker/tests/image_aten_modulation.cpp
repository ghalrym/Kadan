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
    auto x = at::tensor({-3.f, -1.f, 1.f, 3.f})
                 .to(at::kBFloat16)
                 .reshape({1, 1, 4})
                 .repeat({2, 3, 1});
    auto projected = at::zeros({1, 24}, at::kBFloat16);
    projected.slice(1, 0, 4).fill_(0.25f);    // shift_msa
    projected.slice(1, 4, 8).fill_(0.5f);     // scale_msa -> 1.5
    projected.slice(1, 8, 12).fill_(0.5f);    // gate_msa
    projected.slice(1, 12, 16).fill_(-0.25f); // shift_mlp
    projected.slice(1, 16, 20).fill_(1.0f);   // scale_mlp -> 2.0
    projected.slice(1, 20, 24).fill_(-0.5f);  // gate_mlp

    Flux2DoubleModulation dmod(projected);
    check(dmod.attention.scale.scalar_type() == at::kFloat);
    check(dmod.attention.shift[0][0][0].item<float>() == 0.25f);
    check(dmod.attention.scale[0][0][0].item<float>() == 1.5f);
    check(dmod.attention.gate[0][0][0].item<float>() == 0.5f);
    check(dmod.mlp.shift[0][0][0].item<float>() == -0.25f);
    check(dmod.mlp.scale[0][0][0].item<float>() == 2.0f);
    check(dmod.mlp.gate[0][0][0].item<float>() == -0.5f);

    auto normed = flux2_modulated_norm(x, dmod.attention.shift, dmod.attention.scale);
    check(normed.scalar_type() == at::kBFloat16);
    check(at::equal(normed[0], normed[1]));

    auto single_proj = at::zeros({1, 12}, at::kBFloat16);
    single_proj.slice(1, 0, 4).fill_(0.1f);
    single_proj.slice(1, 4, 8).fill_(0.2f);
    single_proj.slice(1, 8, 12).fill_(0.5f);
    Flux2SingleModulation smod(single_proj);
    auto base = at::ones({2, 3, 4}, at::kBFloat16);
    auto delta = at::full({2, 3, 4}, 2.0f, at::kBFloat16);
    auto updated = flux2_gated_residual(base, delta, smod.parallel.gate);
    check(updated.data_ptr() == base.data_ptr());
    check(updated[0][0][0].item<float>() == 2.0f);
}
}  // namespace

int main() {
    double_and_single_modulation();
    std::cout
        << "image ATen modulation arithmetic units passed (CPU only, no inference)\n";
}
