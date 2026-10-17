#include "kadan/image_aten.hpp"

#include <ATen/ATen.h>
#include <ATen/Context.h>
#include <ATen/cuda/CUDAContextLight.h>
#include <algorithm>
#include <c10/core/InferenceMode.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAGuard.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <map>
#include <optional>

#include "image_aten_modulation.hpp"
#include "image_aten_stage.hpp"
#include "image_vision.hpp"
#include "kadan/checkpoint.hpp"
#include "kadan/image_aten_plan.hpp"
#include "kadan/image_checkpoint_config.hpp"
#include "kadan/image_edit.hpp"
#include "kadan/image_input.hpp"
#include "kadan/image_reference.hpp"
#include "kadan/image_schedule.hpp"
#include "kadan/text_tokenizer.hpp"

namespace kadan::image {
namespace {
using namespace detail;

Tensor rms_heads(Stage& stage, const std::string& name, const Tensor& x) {
    return stage.rms(name, x);
}

Tensor heads(const Tensor& x, std::int64_t count) {
    return x.reshape({1, x.size(0), count, 128}).transpose(1, 2);
}

Tensor unheads(const Tensor& x) {
    return x.transpose(1, 2).reshape({x.size(2), -1});
}

Tensor text_rope(const Tensor& x, const Tensor& cosine, const Tensor& sine) {
    auto a = x.slice(-1, 0, 64), b = x.slice(-1, 64, 128);
    return at::cat({a * cosine - b * sine, b * cosine + a * sine}, -1);
}

Tensor encode_text(
    Stage& stage,
    const ImagePrompt& prompt,
    const VisionCondition* vision,
    const std::atomic_bool& cancel,
    const Generator::Hook& hook
) {
    auto options =
        at::TensorOptions().device(at::kCUDA, stage.device).dtype(at::kFloat);
    std::vector<std::int64_t> ids(
        prompt.tokens.begin(), prompt.tokens.begin() + prompt.count
    );
    auto indices = at::from_blob(ids.data(), {std::int64_t(ids.size())}, at::kLong)
                       .to(options.device());
    auto x = at::embedding(stage.weight("embed_tokens.weight"), indices).to(at::kFloat);
    need(bool(vision) == bool(prompt.image_count), "image_reference_condition");
    if (vision) {
        need(
            vision->grid_h == prompt.grid_h && vision->grid_w == prompt.grid_w
                && vision->embeddings.size(0) == std::int64_t(prompt.image_count),
            "image_reference_visual_shape"
        );
        x.slice(0, prompt.image_start, prompt.image_start + prompt.image_count)
            .copy_(vision->embeddings.to(options.device()));
    }
    const auto position_ids = image_text_positions(prompt);
    auto frequencies =
        at::exp(at::arange(64, options) * (-2 * std::log(5000000.) / 128));
    std::array<Tensor, 3> angles;
    for (int axis = 0; axis < 3; ++axis) {
        auto positions = at::from_blob(
                             const_cast<std::int64_t*>(position_ids[axis].data()),
                             {std::int64_t(prompt.count)},
                             at::kLong
        )
                             .to(options)
                             .view({1, 1, -1, 1});
        angles[axis] = positions * frequencies;
    }
    auto angle = angles[0];
    // Qwen3-VL interleaves 24 temporal,20 height,20 width frequencies.
    angle.slice(-1, 1, 60, 3).copy_(angles[1].slice(-1, 1, 60, 3));
    angle.slice(-1, 2, 60, 3).copy_(angles[2].slice(-1, 2, 60, 3));
    auto cosine = angle.cos(), sine = angle.sin();
    for (int layer = 0; layer < 36; ++layer) {
        stop(cancel);
        auto p = "layers." + std::to_string(layer) + ".";
        auto normalized = stage.rms(p + "input_layernorm", x);
        auto q = rms_heads(
            stage,
            p + "self_attn.q_norm",
            heads(stage.linear(p + "self_attn.q_proj", normalized), 32)
        );
        auto k = rms_heads(
            stage,
            p + "self_attn.k_norm",
            heads(stage.linear(p + "self_attn.k_proj", normalized), 8)
        );
        auto v = heads(stage.linear(p + "self_attn.v_proj", normalized), 8);
        q = text_rope(q, cosine, sine);
        k = text_rope(k, cosine, sine);
        auto mix = at::scaled_dot_product_attention(q, k, v, {}, 0., true, {}, true);
        x = x + stage.linear(p + "self_attn.o_proj", unheads(mix));
        normalized = stage.rms(p + "post_attention_layernorm", x);
        auto gate = at::silu(stage.linear(p + "mlp.gate_proj", normalized));
        auto up = stage.linear(p + "mlp.up_proj", normalized);
        x = x + stage.linear(p + "mlp.down_proj", gate * up);
        if (vision && layer < 3) {
            auto visual =
                x.slice(0, prompt.image_start, prompt.image_start + prompt.image_count);
            visual.add_(vision->deepstack[layer].to(options.device()));
        }
        if (hook)
            hook("text_layer", layer);
    }
    stop(cancel);
    return cpu_float(x);
}

struct Rotary {
    Tensor cosine, sine;
};

Rotary
image_rotary(const ImageReferenceLayout& layout, const at::TensorOptions& options) {
    auto positions = at::from_blob(
                         const_cast<std::int64_t*>(layout.positions.front().data()),
                         {std::int64_t(layout.positions.size()), 3},
                         at::kLong
    )
                         .to(options);
    const std::array<int, 3> widths{16, 56, 56};
    std::vector<Tensor> angles;
    for (int axis = 0; axis < 3; ++axis) {
        auto frequency = at::exp(
            at::arange(0, widths[axis], 2, options) * (-std::log(10000.) / widths[axis])
        );
        angles.push_back(positions.select(1, axis).unsqueeze(1) * frequency);
    }
    auto angle = at::cat(angles, 1).unsqueeze(0).unsqueeze(0);
    return {angle.cos(), angle.sin()};
}

Tensor rotate_image(const Tensor& input, const Rotary& rotary) {
    auto pairs = input.to(at::kFloat).reshape({1, 32, input.size(2), 64, 2});
    auto a = pairs.select(-1, 0), b = pairs.select(-1, 1);
    return at::stack(
               {a * rotary.cosine - b * rotary.sine,
                b * rotary.cosine + a * rotary.sine},
               -1
    )
        .reshape_as(input)
        .to(input.scalar_type());
}

Tensor denoise(
    Stage& stage,
    Tensor latent_cpu,
    const Tensor& source_cpu,
    const Tensor& condition_cpu,
    const ImagePrompt& prompt,
    std::int64_t h,
    std::int64_t w,
    const ImageRequest& request,
    const ImageCheckpointConfig& config,
    const std::atomic_bool& cancel,
    const Generator::Hook& hook
) {
    auto options =
        at::TensorOptions().device(at::kCUDA, stage.device).dtype(at::kFloat);
    auto latent = latent_cpu.to(options.device());
    auto times = config.sampling_schedule(h * w, request.steps);
    if (!request.source.empty() && !request.reference)
        times = edit_schedule(times, request.strength);
    Tensor source, noise, mask, inverse_mask;
    if (source_cpu.defined() && !request.reference) {
        source = source_cpu.to(options.device());
        noise = latent;
        latent = source * (1.f - times.sigma[0]) + noise * times.sigma[0];
        if (!request.mask.empty()) {
            // Match the native pipeline's area average, including soft masks.
            auto mask_cpu = at::empty({h * w, 1}, at::kFloat);
            auto* values = mask_cpu.data_ptr<float>();
            for (std::int64_t y = 0; y < h; ++y) {
                stop(cancel);
                for (std::int64_t x = 0; x < w; ++x) {
                    float sum = 0;
                    for (int dy = 0; dy < 16; ++dy)
                        for (int dx = 0; dx < 16; ++dx)
                            sum +=
                                request
                                    .mask[(y * 16 + dy) * request.width + x * 16 + dx];
                    values[y * w + x] = sum / 256.f;
                }
            }
            mask = mask_cpu.to(options.device());
            inverse_mask = 1.f - mask;
        }
    }
    auto condition = condition_cpu.to(options.device(), at::kBFloat16);
    auto projected = stage.linear(
        "txt_in.out_layer",
        at::gelu(
            stage.linear(
                "txt_in.in_layer", stage.rms("txt_in.text_norm", condition, true)
            ),
            "tanh"
        )
    );
    const auto layout = image_reference_layout(
        condition.size(0),
        request.reference ? std::int64_t(prompt.image_start - prompt.prefix) : 0,
        request.reference ? prompt.image_count : 0,
        request.source_height / 16,
        request.source_width / 16,
        h,
        w
    );
    if (request.reference) {
        const auto start = std::int64_t(prompt.image_start - prompt.prefix);
        projected = at::cat(
            {projected.slice(0, 0, start),
             stage.linear("img_in", source_cpu.to(options.device(), at::kBFloat16)),
             projected.slice(0, start + prompt.image_count)},
            0
        );
    }
    const auto prefix_tokens = layout.prefix;
    auto all_rotary = image_rotary(layout, options);
    Rotary image_only{
        all_rotary.cosine.slice(2, prefix_tokens),
        all_rotary.sine.slice(2, prefix_tokens)
    };
    std::array<Tensor, 32> prefix_k, prefix_v;
    auto frequency = at::exp(at::arange(128, options) * (-std::log(10000.) / 128));
    for (std::size_t step = 0; step < times.steps; ++step) {
        stop(cancel);
        // Match the reference BF16 rounding at the model timestep boundary.
        const auto sigma =
            (at::scalar_tensor(times.sigma[step] * 1000, options).to(at::kBFloat16)
             / 1000)
                .to(at::kBFloat16)
                .item<float>();
        auto angle = at::tensor({sigma * 1000, 0.f}, options).unsqueeze(1) * frequency;
        auto embedding = at::cat({angle.cos(), angle.sin()}, 1).to(at::kBFloat16);
        auto time = stage.linear(
            "time_text_embed.timestep_embedder.linear_2",
            at::silu(
                stage.linear("time_text_embed.timestep_embedder.linear_1", embedding)
            )
        );
        auto activated_time = at::silu(time);
        detail::AtenModulation modulation(stage.linear("modulation.1", activated_time));
        auto image = stage.linear("img_in", latent.to(at::kBFloat16));
        Tensor x = step ? image : at::cat({projected, image}, 0);
        const auto prefix = step ? 0 : prefix_tokens;
        const auto& rotary = step ? image_only : all_rotary;
        if (hook)
            hook("conditioning", step);
        for (int layer = 0; layer < 32; ++layer) {
            stop(cancel);
            if (hook)
                hook("block_started", layer);
            auto p = "transformer_blocks." + std::to_string(layer) + ".";
            auto n = detail::aten_modulated_norm(x, modulation.attention_scale, prefix);
            auto q = rotate_image(
                stage.rms(
                    p + "attn.norm_q", heads(stage.linear(p + "attn.to_q", n), 32)
                ),
                rotary
            );
            auto k = rotate_image(
                stage.rms(
                    p + "attn.norm_k", heads(stage.linear(p + "attn.to_k", n), 32)
                ),
                rotary
            );
            auto v = heads(stage.linear(p + "attn.to_v", n), 32);
            Tensor mix;
            if (!step) {
                prefix_k[layer] = k.slice(2, 0, prefix_tokens).clone();
                prefix_v[layer] = v.slice(2, 0, prefix_tokens).clone();
                std::vector<Tensor> attention;
                for (const auto& segment : layout.segments) {
                    Tensor mask;
                    if (segment.text && segment.start) {
                        mask = at::arange(segment.end, options).unsqueeze(0)
                               <= at::arange(segment.start, segment.end, options)
                                      .unsqueeze(1);
                    }
                    attention.push_back(
                        at::scaled_dot_product_attention(
                            q.slice(2, segment.start, segment.end),
                            k.slice(2, 0, segment.end),
                            v.slice(2, 0, segment.end),
                            mask.defined() ? std::optional<Tensor>(mask) : std::nullopt,
                            0.,
                            segment.text && segment.start == 0
                        )
                    );
                }
                attention.push_back(
                    at::scaled_dot_product_attention(q.slice(2, prefix_tokens), k, v)
                );
                mix = at::cat(attention, 2);
            } else {
                mix = at::scaled_dot_product_attention(
                    q,
                    at::cat({prefix_k[layer], k}, 2),
                    at::cat({prefix_v[layer], v}, 2)
                );
            }
            x = detail::aten_gated_residual(
                x,
                stage.linear(p + "attn.to_out.0", unheads(mix)),
                modulation.attention_gate,
                prefix
            );
            n = detail::aten_modulated_norm(x, modulation.mlp_scale, prefix);
            auto gate = at::silu(stage.linear(p + "img_mlp.gate_layer", n));
            auto up = stage.linear(p + "img_mlp.proj", n);
            x = detail::aten_gated_residual(
                x,
                stage.linear(p + "img_mlp.out", gate * up),
                modulation.mlp_gate,
                prefix
            );
            if (hook)
                hook("block", layer);
        }
        auto scale = stage.linear("norm_out.linear", activated_time.slice(0, 0, 1))
                         .to(at::kFloat)
                         .add(1);
        auto normalized = detail::aten_modulated_norm(x.slice(0, prefix), scale, 0);
        auto velocity = stage.linear("proj_out", normalized).to(at::kFloat);
        latent = latent + velocity * (times.sigma[step + 1] - times.sigma[step]);
        if (mask.defined()) {
            const auto next_sigma = times.sigma[step + 1];
            latent =
                mask * latent
                + inverse_mask * (source * (1.f - next_sigma) + noise * next_sigma);
        }
        if (hook)
            hook("step_completed", step + 1);
    }
    stop(cancel);
    return cpu_float(latent);
}

Tensor vae_norm(Stage& stage, const std::string& p, const Tensor& x, bool activate) {
    auto denominator = x.square().sum(1, true).sqrt().clamp_min(1e-12);
    auto gamma = stage.weight(p + ".gamma").reshape({1, x.size(1), 1, 1});
    // Division owns this output; reuse it without mutating the residual input.
    auto y = x / denominator;
    y.mul_(std::sqrt(double(x.size(1)))).mul_(gamma);
    if (activate)
        at::silu_(y);
    return y;
}

Tensor convolution(
    Stage& stage, const std::string& p, const Tensor& x, bool downsample = false
) {
    const auto weight = stage.weight(p + ".weight");
    if (downsample)
        return at::conv2d(
            at::constant_pad_nd(x, {0, 1, 0, 1}, 0),
            weight,
            stage.weight(p + ".bias"),
            {2, 2},
            at::IntArrayRef{0, 0}
        );
    return at::conv2d(
        x,
        weight,
        stage.weight(p + ".bias"),
        {1, 1},
        {weight.size(2) / 2, weight.size(3) / 2}
    );
}

Tensor residual(
    Stage& stage, const std::string& p, const Tensor& x, const std::atomic_bool& cancel
) {
    stop(cancel);
    auto y = convolution(stage, p + ".conv1", vae_norm(stage, p + ".norm1", x, true));
    stop(cancel);
    y = convolution(stage, p + ".conv2", vae_norm(stage, p + ".norm2", y, true));
    return y.add_(
        x.size(1) == y.size(1) ? x : convolution(stage, p + ".conv_shortcut", x)
    );
}

Tensor vae_attention(
    Stage& stage, const std::string& p, const Tensor& x, const std::atomic_bool& cancel
) {
    auto qkv = convolution(stage, p + ".to_qkv", vae_norm(stage, p + ".norm", x, false))
                   .flatten(2);
    auto c = x.size(1), points = qkv.size(2);
    auto q = qkv.slice(1, 0, c).transpose(1, 2).unsqueeze(1);
    auto k = qkv.slice(1, c, 2 * c).transpose(1, 2).unsqueeze(1);
    auto v = qkv.slice(1, 2 * c, 3 * c).transpose(1, 2).unsqueeze(1);
    // The VAE has one wide head. Query tiling bounds the math backend's
    // scratch when this shape cannot use Flash/efficient SDPA.
    auto mix = at::empty_like(q);
    for (std::int64_t row = 0; row < points; row += 256) {
        stop(cancel);
        auto end = std::min(row + 256, points);
        mix.slice(2, row, end)
            .copy_(at::scaled_dot_product_attention(q.slice(2, row, end), k, v));
    }
    return x
           + convolution(
               stage, p + ".proj", mix.squeeze(1).transpose(1, 2).reshape_as(x)
           );
}

Tensor encode_vae_blocks(
    Stage& stage, Tensor x, const std::atomic_bool& cancel, const Generator::Hook& hook
) {
    x = convolution(stage, "encoder.conv_in", x);
    for (int i = 0; i < 5; ++i) {
        stop(cancel);
        const auto p = "encoder.down_blocks." + std::to_string(i);
        auto original = x;
        for (int j = 0; j < 2; ++j)
            x = residual(stage, p + ".resnets." + std::to_string(j), x, cancel);
        if (i < 4)
            x = convolution(stage, p + ".downsampler.resample.1", x, true);
        // AvgDown3D orders [channel,time,dy,dx] before averaging channel groups.
        // A single real frame follows temporal zeros; the first chunk never
        // executes the checkpoint's time_conv. This also adds block4's identity.
        const auto temporal = i > 0 && i < 4 ? 2 : 1;
        const auto spatial = i < 4 ? 2 : 1;
        const auto group = original.size(1) * temporal * spatial * spatial / x.size(1);
        auto shortcut = original.unsqueeze(2);
        if (temporal == 2)
            shortcut = at::constant_pad_nd(shortcut, {0, 0, 0, 0, 1, 0}, 0);
        shortcut = shortcut.reshape(
            {1, original.size(1), 1, temporal, x.size(2), spatial, x.size(3), spatial}
        );
        original = Tensor{};
        shortcut = shortcut.permute({0, 1, 3, 5, 7, 2, 4, 6})
                       .reshape({1, x.size(1), group, 1, x.size(2), x.size(3)})
                       .mean(2)
                       .squeeze(2);
        x.add_(shortcut);
        if (hook)
            hook("vae_down_block", i);
    }
    return x;
}

Tensor encode_source(
    Stage& stage,
    const ImageRequest& request,
    const ImageCheckpointConfig& config,
    const std::atomic_bool& cancel,
    const Generator::Hook& hook
) {
    const auto device = at::Device(at::kCUDA, stage.device);
    auto x = at::from_blob(
                 const_cast<float*>(request.source.data()),
                 {1, 4, std::int64_t(request.height), std::int64_t(request.width)},
                 at::kFloat
    )
                 .to(device);
    if (stage.tiled_vae()) {
        const auto h = x.size(2) / 16, w = x.size(3) / 16;
        auto encoded = at::empty({1, 768, h, w}, x.options());
        for (std::int64_t y = 0; y < h; y += vae_tile_core) {
            const auto rows = vae_tile_axis(y, h, vae_encode_halo);
            for (std::int64_t column = 0; column < w; column += vae_tile_core) {
                stop(cancel);
                const auto cols = vae_tile_axis(column, w, vae_encode_halo);
                auto tile = encode_vae_blocks(
                    stage,
                    x.slice(2, rows.input_begin * 16, rows.input_end * 16)
                        .slice(3, cols.input_begin * 16, cols.input_end * 16)
                        .contiguous(),
                    cancel,
                    hook
                );
                encoded.slice(2, rows.begin, rows.end)
                    .slice(3, cols.begin, cols.end)
                    .copy_(tile.slice(
                                   2,
                                   rows.crop_begin(),
                                   rows.crop_begin() + rows.end - rows.begin
                    )
                               .slice(
                                   3,
                                   cols.crop_begin(),
                                   cols.crop_begin() + cols.end - cols.begin
                               ));
            }
        }
        x = std::move(encoded);
    } else {
        x = encode_vae_blocks(stage, x, cancel, hook);
    }
    x = residual(stage, "encoder.mid_block.resnets.0", x, cancel);
    x = vae_attention(stage, "encoder.mid_block.attentions.0", x, cancel);
    x = residual(stage, "encoder.mid_block.resnets.1", x, cancel);
    x = convolution(
        stage, "encoder.conv_out", vae_norm(stage, "encoder.norm_out", x, true)
    );
    x = convolution(stage, "quant_conv", x).slice(1, 0, 64);
    auto mean =
        at::from_blob(const_cast<float*>(config.mean.data()), {1, 64, 1, 1}, at::kFloat)
            .to(device);
    auto stddev =
        at::from_blob(
            const_cast<float*>(config.stddev.data()), {1, 64, 1, 1}, at::kFloat
        )
            .to(device);
    stop(cancel);
    return cpu_float(((x - mean) / stddev).flatten(2).squeeze(0).transpose(0, 1));
}

Tensor decode_vae_blocks(
    Stage& stage, Tensor x, const std::atomic_bool& cancel, const Generator::Hook& hook
) {
    for (int i = 0; i < 5; ++i) {
        stop(cancel);
        const auto p = "decoder.up_blocks." + std::to_string(i);
        auto original = i < 4 ? x : Tensor{};
        for (int j = 0; j < 3; ++j)
            x = residual(stage, p + ".resnets." + std::to_string(j), x, cancel);
        if (i < 4) {
            x = at::upsample_nearest2d(x, {x.size(2) * 2, x.size(3) * 2}, {}, {});
            x = convolution(stage, p + ".upsampler.resample.1", x);
            auto factor = i < 3 ? 2 : 1;
            auto repeats = x.size(1) * factor * 4 / original.size(1);
            need(
                repeats > 0 && x.size(1) * factor * 4 % original.size(1) == 0,
                "image_aten_vae_shortcut"
            );
            const auto original_h = original.size(2), original_w = original.size(3);
            auto shortcut = original.repeat_interleave(repeats, 1);
            // The reshape below copies a permuted view. Retire the source first
            // so the peak is x + repeated shortcut + reshaped shortcut.
            original = Tensor{};
            shortcut =
                shortcut.reshape({1, x.size(1), factor, 2, 2, original_h, original_w})
                    .select(2, factor - 1)
                    .permute({0, 1, 4, 2, 5, 3})
                    .reshape_as(x);
            x.add_(shortcut);
        }
        if (hook)
            hook("vae_up_block", i);
    }
    x = convolution(
        stage, "decoder.conv_out", vae_norm(stage, "decoder.norm_out", x, true)
    );
    stop(cancel);
    return x.clamp(-1, 1);
}

Tensor decode(
    Stage& stage,
    Tensor latent,
    std::int64_t h,
    std::int64_t w,
    const ImageCheckpointConfig& config,
    const std::atomic_bool& cancel,
    const Generator::Hook& hook
) {
    auto options =
        at::TensorOptions().device(at::kCUDA, stage.device).dtype(at::kFloat);
    auto mean =
        at::from_blob(const_cast<float*>(config.mean.data()), {1, 64, 1, 1}, at::kFloat)
            .to(options.device());
    auto stddev =
        at::from_blob(
            const_cast<float*>(config.stddev.data()), {1, 64, 1, 1}, at::kFloat
        )
            .to(options.device());
    auto x = latent.to(options.device()).transpose(0, 1).reshape({1, 64, h, w}) * stddev
             + mean;
    x = convolution(stage, "post_quant_conv", x);
    x = convolution(stage, "decoder.conv_in", x);
    x = residual(stage, "decoder.mid_block.resnets.0", x, cancel);
    x = vae_attention(stage, "decoder.mid_block.attentions.0", x, cancel);
    x = residual(stage, "decoder.mid_block.resnets.1", x, cancel);
    if (!stage.tiled_vae())
        return cpu_float(decode_vae_blocks(stage, x, cancel, hook));
    auto output = at::empty({1, 4, h * 16, w * 16}, at::kFloat);
    for (std::int64_t y = 0; y < h; y += vae_tile_core) {
        const auto rows = vae_tile_axis(y, h, vae_decode_halo);
        for (std::int64_t column = 0; column < w; column += vae_tile_core) {
            stop(cancel);
            const auto cols = vae_tile_axis(column, w, vae_decode_halo);
            auto tile = decode_vae_blocks(
                stage,
                x.slice(2, rows.input_begin, rows.input_end)
                    .slice(3, cols.input_begin, cols.input_end)
                    .contiguous(),
                cancel,
                hook
            );
            auto pixels = tile.slice(
                                  2,
                                  rows.crop_begin() * 16,
                                  (rows.crop_begin() + rows.end - rows.begin) * 16
            )
                              .slice(
                                  3,
                                  cols.crop_begin() * 16,
                                  (cols.crop_begin() + cols.end - cols.begin) * 16
                              );
            output.slice(2, rows.begin * 16, rows.end * 16)
                .slice(3, cols.begin * 16, cols.end * 16)
                .copy_(cpu_float(pixels));
        }
    }
    return output;
}

// Backend flags are process-wide, so restore them before another workload runs.
struct ArithmeticScope {
    at::Context& context = at::globalContext();
    bool blas = context.allowTF32CuBLAS(), cudnn = context.allowTF32CuDNN();
    at::CuBLASReductionOption reduction = context.allowBF16ReductionCuBLAS();

    ArithmeticScope() {
        context.setAllowTF32CuBLAS(false);
        context.setAllowTF32CuDNN(false);
        context.setAllowBF16ReductionCuBLAS(false);
    }

    ~ArithmeticScope() {
        context.setAllowTF32CuBLAS(blas);
        context.setAllowTF32CuDNN(cudnn);
        context.setAllowBF16ReductionCuBLAS(
            reduction == at::CuBLASReductionOption::AllowReducedPrecisionWithSplitK,
            reduction
                != at::CuBLASReductionOption::DisallowReducedPrecisionDisallowSplitK
        );
    }
};

void stage_timing(
    const char* phase, const char* stage, int device, Clock::time_point started
) {
    auto seconds = std::chrono::duration<double>(Clock::now() - started).count();
    std::fprintf(
        stderr,
        "KADAN_ATEN_IMAGE_TIMING "
        "{\"phase\":\"%s\",\"stage\":\"%s\",\"device\":%d,\"seconds\":%.9f}\n",
        phase,
        stage,
        device,
        seconds
    );
    std::fflush(stderr);
}
}  // namespace

struct AtenGenerator::Impl {
    // Parser/tokenizer allocations are accounted for even while GPU stages idle.
    Resources& resources;
    Handle metadata;
    text::Tokenizer tokenizer;
    ImageCheckpointConfig config;
    std::string root;
    std::unique_ptr<Inventory> text, denoiser, vae, vae_encoder, vision;

    explicit Impl(const std::shared_ptr<Resources>& r)
        : resources(*r), metadata(0), tokenizer(r, Workload::image) {
        auto bytes = r->snapshot().capacity;
        std::fill(bytes.begin(), bytes.end(), 0);
        bytes[0] = 192 * mib;
        metadata = r->reserve(Workload::image, bytes);
    }

    ~Impl() {
        vision.reset();
        vae_encoder.reset();
        vae.reset();
        denoiser.reset();
        text.reset();
        resources.released(metadata);
    }

    Inventory& encoder(const std::atomic_bool& cancel) {
        if (!vae_encoder) {
            const std::array<std::string, 1> files{
                "diffusion_pytorch_model.safetensors"
            };
            auto inventory = std::make_unique<Inventory>(
                root + "/vae", files, AtenImageStage::vae_encoder, cancel
            );
            inventory->validate_vae();
            vae_encoder = std::move(inventory);
        }
        return *vae_encoder;
    }

    Inventory& visual_encoder(const std::atomic_bool& cancel) {
        if (!vision) {
            auto inventory = std::make_unique<Inventory>(
                root + "/text_encoder",
                config.text_files,
                AtenImageStage::vision,
                cancel
            );
            validate_vision(*inventory);
            vision = std::move(inventory);
        }
        return *vision;
    }
};

AtenGenerator::AtenGenerator(
    std::shared_ptr<Resources> resources, std::vector<int> devices
)
    : resources_(std::move(resources)), devices_(std::move(devices)) {
    need(bool(resources_) && !devices_.empty(), "image_aten_resources");
}

AtenGenerator::~AtenGenerator() = default;

void AtenGenerator::load(
    const std::string& root, const std::atomic_bool& cancel, const Generator::Hook& hook
) {
    need(!model_, "image_generator_loaded");
    auto model = std::make_unique<Impl>(resources_);
    model->root = root;
    model->config = validate_checkpoint_config(root);
    model->tokenizer.load(root + "/processor/tokenizer.json", cancel);
    const std::array<std::string, 2> denoiser_files{
        "diffusion_pytorch_model-00001-of-00002.safetensors",
        "diffusion_pytorch_model-00002-of-00002.safetensors"
    };
    const std::array<std::string, 1> vae_files{"diffusion_pytorch_model.safetensors"};
    model->text = std::make_unique<Inventory>(
        root + "/text_encoder", model->config.text_files, AtenImageStage::text, cancel
    );
    model->denoiser = std::make_unique<Inventory>(
        root + "/transformer", denoiser_files, AtenImageStage::denoiser, cancel
    );
    model->vae = std::make_unique<Inventory>(
        root + "/vae", vae_files, AtenImageStage::vae, cancel
    );
    model->text->validate_text();
    model->denoiser->validate_denoiser();
    model->vae->validate_vae();
    stop(cancel);
    model_ = std::move(model);
    if (hook)
        hook("aten_checkpoint_loaded", 0);
}

void AtenGenerator::unload() {
    model_.reset();
}

void AtenGenerator::prepare(
    const ImageRequest& request, const std::atomic_bool& cancel
) {
    need(bool(model_), "image_generator_not_loaded");
    Generator::validate(request);
    stop(cancel);
    if (!request.source.empty() && !request.reference && request.strength == 0)
        return;
    auto& m = *model_;
    auto prompt = prepare_image_prompt(
        m.tokenizer,
        request.prompt,
        cancel,
        request.reference ? request.source_width : 0,
        request.reference ? request.source_height : 0
    );
    const auto points = (request.height / 16) * (request.width / 16);
    const auto reference_points =
        request.reference ? (request.source_height / 16) * (request.source_width / 16)
                          : 0;
    std::vector<const Inventory*> inventories{
        m.text.get(), m.denoiser.get(), m.vae.get()
    };
    if (!request.source.empty()
        && (request.reference || request.strength < 1 || !request.mask.empty()))
        inventories.push_back(&m.encoder(cancel));
    if (request.reference)
        inventories.push_back(&m.visual_encoder(cancel));
    // All required stage plans must fit before the FIFO owner commits to
    // running. Components execute sequentially, so their GPU budgets do not sum.
    for (const auto* inventory : inventories) {
        stop(cancel);
        const auto tokens = inventory->stage == AtenImageStage::text
                                ? prompt.count
                                : prompt.count - prompt.prefix - prompt.image_count;
        const auto stage_points =
            inventory->stage == AtenImageStage::vae_encoder && request.reference
                ? reference_points
                : points;
        const auto workspace = aten_stage_workspace(
            inventory->stage,
            stage_points,
            tokens,
            !request.source.empty() && !request.reference,
            reference_points,
            prompt.image_count
        );
        (void)aten_stage_plan(
            stage_snapshot(*resources_, devices_),
            devices_,
            inventory->bytes,
            inventory->largest,
            workspace,
            aten_tiled_vae_workspace(inventory->stage, stage_points)
        );
    }
}

void AtenGenerator::generate(
    const ImageRequest& request,
    std::span<float> output,
    const std::atomic_bool& cancel,
    const Generator::Hook& hook
) {
    need(bool(model_), "image_generator_not_loaded");
    Generator::validate(request);
    need(output.size() == request.width * request.height * 4, "image_output_shape");
    stop(cancel);
    if (!request.source.empty() && !request.reference && request.strength == 0) {
        std::copy(request.source.begin(), request.source.end(), output.begin());
        return;
    }
    c10::InferenceMode inference;
    ArithmeticScope arithmetic;
    auto& m = *model_;
    auto prompt = prepare_image_prompt(
        m.tokenizer,
        request.prompt,
        cancel,
        request.reference ? request.source_width : 0,
        request.reference ? request.source_height : 0
    );
    const auto count = prompt.count, drop = prompt.prefix;
    const auto h = request.height / 16, w = request.width / 16, points = h * w;
    const auto reference_points =
        request.reference ? (request.source_height / 16) * (request.source_width / 16)
                          : 0;
    Tensor source;
    if (!request.source.empty()
        && (request.reference || request.strength < 1 || !request.mask.empty())) {
        auto source_request = request;
        if (request.reference) {
            source_request.width = request.source_width;
            source_request.height = request.source_height;
        }
        const auto source_points =
            (source_request.height / 16) * (source_request.width / 16);
        auto started = Clock::now();
        Stage stage(
            *resources_,
            devices_,
            m.encoder(cancel),
            aten_stage_workspace(AtenImageStage::vae_encoder, source_points, 0),
            cancel,
            aten_tiled_vae_workspace(AtenImageStage::vae_encoder, source_points)
        );
        stage_timing("load", "vae_encoder", stage.device, started);
        c10::cuda::CUDAGuard guard(stage.device);
        started = Clock::now();
        source = encode_source(stage, source_request, m.config, cancel, hook);
        stage.sync();
        stage_timing("generation", "vae_encoder", stage.device, started);
        started = Clock::now();
        stage.close();
        stage_timing("unload", "vae_encoder", stage.device, started);
    }
    std::optional<VisionCondition> vision;
    if (request.reference) {
        auto started = Clock::now();
        Stage stage(
            *resources_,
            devices_,
            m.visual_encoder(cancel),
            aten_stage_workspace(
                AtenImageStage::vision,
                reference_points,
                count,
                false,
                reference_points,
                prompt.image_count
            ),
            cancel
        );
        stage_timing("load", "vision", stage.device, started);
        c10::cuda::CUDAGuard guard(stage.device);
        started = Clock::now();
        vision = encode_vision(
            stage,
            request.source,
            request.source_width,
            request.source_height,
            cancel,
            hook
        );
        stage.sync();
        stage_timing("generation", "vision", stage.device, started);
        started = Clock::now();
        stage.close();
        stage_timing("unload", "vision", stage.device, started);
    }
    Tensor condition;
    {
        auto started = Clock::now();
        Stage stage(
            *resources_,
            devices_,
            *m.text,
            aten_stage_workspace(AtenImageStage::text, points, count),
            cancel
        );
        stage_timing("load", "text", stage.device, started);
        c10::cuda::CUDAGuard guard(stage.device);
        started = Clock::now();
        condition =
            encode_text(stage, prompt, vision ? &*vision : nullptr, cancel, hook)
                .slice(0, drop)
                .contiguous();
        stage.sync();
        stage_timing("generation", "text", stage.device, started);
        started = Clock::now();
        stage.close();
        stage_timing("unload", "text", stage.device, started);
    }
    vision.reset();
    // Preserve Kadan's seed/noise contract. A shared replay tensor is needed for
    // elementwise comparison to PyTorch CPU CHW noise in the reference harness.
    auto latent = at::empty({std::int64_t(points), 64}, at::kFloat);
    seed_image_latent(
        {latent.data_ptr<float>(), std::size_t(latent.numel())}, request.seed, cancel
    );
    {
        auto started = Clock::now();
        Stage stage(
            *resources_,
            devices_,
            *m.denoiser,
            aten_stage_workspace(
                AtenImageStage::denoiser,
                points,
                count - drop - prompt.image_count,
                !request.source.empty() && !request.reference,
                reference_points
            ),
            cancel
        );
        stage_timing("load", "denoiser", stage.device, started);
        c10::cuda::CUDAGuard guard(stage.device);
        started = Clock::now();
        latent = denoise(
            stage,
            latent,
            source,
            condition,
            prompt,
            h,
            w,
            request,
            m.config,
            cancel,
            hook
        );
        stage.sync();
        stage_timing("generation", "denoiser", stage.device, started);
        started = Clock::now();
        stage.close();
        stage_timing("unload", "denoiser", stage.device, started);
    }
    {
        auto started = Clock::now();
        Stage stage(
            *resources_,
            devices_,
            *m.vae,
            aten_stage_workspace(AtenImageStage::vae, points, count - drop),
            cancel,
            aten_tiled_vae_workspace(AtenImageStage::vae, points)
        );
        stage_timing("load", "vae", stage.device, started);
        c10::cuda::CUDAGuard guard(stage.device);
        started = Clock::now();
        auto decoded = decode(stage, latent, h, w, m.config, cancel, hook);
        stage.sync();
        stage_timing("generation", "vae", stage.device, started);
        std::copy_n(decoded.data_ptr<float>(), output.size(), output.data());
        started = Clock::now();
        stage.close();
        stage_timing("unload", "vae", stage.device, started);
    }
    if (!request.mask.empty())
        composite_image_edit(output, request.source, request.mask);
    stop(cancel);
}
}  // namespace kadan::image
