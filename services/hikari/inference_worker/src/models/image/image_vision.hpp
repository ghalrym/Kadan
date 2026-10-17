#pragma once
#include "image_aten_stage.hpp"
#include "kadan/image_input.hpp"

namespace kadan::image::detail {
struct VisionCondition {
    Tensor embeddings;
    std::array<Tensor, 3> deepstack;
    std::size_t grid_h = 0, grid_w = 0;
};

inline void validate_vision(const Inventory& inventory) {
    auto linear = [&](const std::string& p, std::uint64_t in, std::uint64_t out) {
        inventory.shape(p + ".weight", {out, in});
        inventory.shape(p + ".bias", {out});
    };
    auto norm = [&](const std::string& p, std::uint64_t width) {
        inventory.shape(p + ".weight", {width});
        inventory.shape(p + ".bias", {width});
    };
    inventory.shape("patch_embed.proj.weight", {1152, 3, 2, 16, 16});
    inventory.shape("patch_embed.proj.bias", {1152});
    inventory.shape("pos_embed.weight", {2304, 1152});
    for (int i = 0; i < 27; ++i) {
        const auto p = "blocks." + std::to_string(i) + ".";
        norm(p + "norm1", 1152);
        norm(p + "norm2", 1152);
        linear(p + "attn.qkv", 1152, 3456);
        linear(p + "attn.proj", 1152, 1152);
        linear(p + "mlp.linear_fc1", 1152, 4304);
        linear(p + "mlp.linear_fc2", 4304, 1152);
    }
    for (int i = -1; i < 3; ++i) {
        const auto p = i < 0 ? std::string("merger")
                             : "deepstack_merger_list." + std::to_string(i);
        norm(p + ".norm", i < 0 ? 1152 : 4608);
        linear(p + ".linear_fc1", 4608, 4608);
        linear(p + ".linear_fc2", 4608, 4096);
    }
}

inline Tensor vision_linear(Stage& stage, const std::string& p, const Tensor& x) {
    return at::linear(
        x,
        stage.weight(p + ".weight").to(x.scalar_type()),
        stage.weight(p + ".bias").to(x.scalar_type())
    );
}

inline Tensor vision_norm(Stage& stage, const std::string& p, const Tensor& x) {
    return at::layer_norm(
        x,
        {x.size(-1)},
        stage.weight(p + ".weight").to(x.scalar_type()),
        stage.weight(p + ".bias").to(x.scalar_type()),
        1e-6
    );
}

inline Tensor vision_merge(Stage& stage, const Tensor& x, int deepstack = -1) {
    const auto p = deepstack < 0 ? std::string("merger")
                                 : "deepstack_merger_list." + std::to_string(deepstack);
    auto normalized =
        vision_norm(stage, p + ".norm", deepstack < 0 ? x : x.reshape({-1, 4608}))
            .reshape({-1, 4608});
    return vision_linear(
        stage,
        p + ".linear_fc2",
        at::gelu(vision_linear(stage, p + ".linear_fc1", normalized), "none")
    );
}

// Single-image Qwen3-VL path: patch groups are ordered block-row, block-column,
// inner-row, inner-column so that the merger sees contiguous 2x2 patches.
inline VisionCondition encode_vision(
    Stage& stage,
    std::span<const float> source,
    std::size_t width,
    std::size_t height,
    const std::atomic_bool& cancel,
    const Generator::Hook& hook
) {
    need(
        width && height && width % 32 == 0 && height % 32 == 0,
        "image_reference_dimensions"
    );
    need(source.size() == width * height * 4, "image_reference_pixels");
    const auto gh = std::int64_t(height / 16), gw = std::int64_t(width / 16);
    const auto patches = gh * gw;
    const auto options =
        at::TensorOptions().device(at::kCUDA, stage.device).dtype(at::kFloat);
    auto rgba = at::from_blob(
                    const_cast<float*>(source.data()),
                    {4, std::int64_t(height), std::int64_t(width)},
                    at::kFloat
    )
                    .to(options.device());
    auto alpha = (rgba.select(0, 3) + 1.f) * .5f;
    // The official PIL path composites alpha over white at 8-bit precision;
    // processor mean/std=.5 maps the RGB copy back to [-1,1].
    auto rgb = (rgba.slice(0, 0, 3) * alpha + (1.f - alpha));
    rgb = ((rgb + 1.f) * 127.5f).round().clamp(0.f, 255.f) / 127.5f - 1.f;
    auto pixel_patches = rgb.reshape({3, gh / 2, 2, 16, gw / 2, 2, 16})
                             .permute({1, 4, 2, 5, 0, 3, 6})
                             .unsqueeze(5)
                             .expand({gh / 2, gw / 2, 2, 2, 3, 2, 16, 16})
                             .reshape({patches, 1536});
    auto x = at::linear(
        pixel_patches,
        stage.weight("patch_embed.proj.weight").to(at::kFloat).reshape({1152, 1536}),
        stage.weight("patch_embed.proj.bias").to(at::kFloat)
    );
    pixel_patches = Tensor();
    rgb = Tensor();
    alpha = Tensor();
    rgba = Tensor();

    auto row = at::arange(gh, options)
                   .reshape({gh / 2, 2, 1, 1})
                   .expand({gh / 2, 2, gw / 2, 2})
                   .permute({0, 2, 1, 3})
                   .reshape({patches});
    auto col = at::arange(gw, options)
                   .reshape({1, 1, gw / 2, 2})
                   .expand({gh / 2, 2, gw / 2, 2})
                   .permute({0, 2, 1, 3})
                   .reshape({patches});
    {
        // Learned 48x48 positions use align-corners bilinear interpolation.
        auto y = row * (47.f / float(gh - 1));
        auto z = col * (47.f / float(gw - 1));
        auto y0 = y.floor(), z0 = z.floor(), y1 = (y0 + 1).clamp_max(47),
             z1 = (z0 + 1).clamp_max(47);
        auto dy = y - y0, dz = z - z0;
        auto positions = stage.weight("pos_embed.weight").to(at::kFloat);
        auto lookup = [&](const Tensor& yy, const Tensor& zz) {
            return at::embedding(positions, (yy * 48 + zz).to(at::kLong));
        };
        x = x + lookup(y0, z0) * ((1 - dy) * (1 - dz)).unsqueeze(1)
            + lookup(y0, z1) * ((1 - dy) * dz).unsqueeze(1)
            + lookup(y1, z0) * (dy * (1 - dz)).unsqueeze(1)
            + lookup(y1, z1) * (dy * dz).unsqueeze(1);
    }
    auto frequency = at::exp(at::arange(0, 36, 2, options) * (-std::log(10000.) / 36));
    auto angle =
        at::cat({row.unsqueeze(1) * frequency, col.unsqueeze(1) * frequency}, 1)
            .reshape({1, 1, patches, 36});
    auto cosine = angle.cos(), sine = angle.sin();
    auto rotate = [&](const Tensor& input) {
        auto a = input.slice(-1, 0, 36), b = input.slice(-1, 36, 72);
        return at::cat({a * cosine - b * sine, b * cosine + a * sine}, -1);
    };
    VisionCondition condition;
    condition.grid_h = height / 32;
    condition.grid_w = width / 32;
    for (int layer = 0; layer < 27; ++layer) {
        stop(cancel);
        const auto p = "blocks." + std::to_string(layer) + ".";
        auto qkv =
            vision_linear(stage, p + "attn.qkv", vision_norm(stage, p + "norm1", x))
                .reshape({patches, 3, 16, 72})
                .permute({1, 2, 0, 3});
        auto q = rotate(qkv.select(0, 0).unsqueeze(0));
        auto k = rotate(qkv.select(0, 1).unsqueeze(0));
        auto v = qkv.select(0, 2).unsqueeze(0);
        auto attention = at::scaled_dot_product_attention(q, k, v, {}, 0., false)
                             .squeeze(0)
                             .transpose(0, 1)
                             .reshape({patches, 1152});
        x = x + vision_linear(stage, p + "attn.proj", attention);
        auto mlp = vision_linear(
            stage, p + "mlp.linear_fc1", vision_norm(stage, p + "norm2", x)
        );
        x = x + vision_linear(stage, p + "mlp.linear_fc2", at::gelu(mlp, "tanh"));
        if (layer == 8 || layer == 16 || layer == 24) {
            const auto index = layer / 8 - 1;
            condition.deepstack[index] = cpu_float(vision_merge(stage, x, index));
        }
        if (hook)
            hook("vision_layer", layer);
    }
    stop(cancel);
    condition.embeddings = cpu_float(vision_merge(stage, x));
    return condition;
}
}  // namespace kadan::image::detail
