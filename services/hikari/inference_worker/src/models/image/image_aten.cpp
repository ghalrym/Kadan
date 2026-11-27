#include "kadan/image_aten.hpp"

#include <ATen/ATen.h>
#include <ATen/Context.h>
#include <ATen/cuda/CUDAContextLight.h>
#include <algorithm>
#include <array>
#include <c10/core/InferenceMode.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAGuard.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <map>
#include <mutex>
#include <optional>
#include <thread>

#include "image_aten_modulation.hpp"
#include "image_aten_stage.hpp"
#include "kadan/checkpoint.hpp"
#include "kadan/checkpoint_cache.hpp"
#include "kadan/cuda_phase_metrics.hpp"
#include "kadan/image_aten_performance.hpp"
#include "kadan/image_aten_plan.hpp"
#include "kadan/image_checkpoint_config.hpp"
#include "kadan/image_edit.hpp"
#include "kadan/image_input.hpp"
#include "kadan/image_reference.hpp"
#include "kadan/image_schedule.hpp"
#include "kadan/image_weight_cache.hpp"
#include "kadan/text_tokenizer.hpp"

namespace kadan::image {
namespace {
using namespace detail;

Tensor heads(const Tensor& x, std::int64_t count) {
    return x.reshape({x.dim() == 3 ? x.size(0) : 1, x.size(-2), count, 128})
        .transpose(1, 2);
}

Tensor unheads(const Tensor& x) {
    return x.transpose(1, 2).reshape({x.size(0), x.size(2), -1});
}

Tensor text_rope(const Tensor& x, const Tensor& cosine, const Tensor& sine) {
    auto a = x.slice(-1, 0, 64), b = x.slice(-1, 64, 128);
    return at::cat({a * cosine - b * sine, b * cosine + a * sine}, -1);
}

Tensor encode_text(
    Stage& stage,
    const ImagePrompt& prompt,
    const std::atomic_bool& cancel,
    const Generator::Hook& hook
) {
    auto options =
        at::TensorOptions().device(at::kCUDA, stage.device).dtype(at::kFloat);
    constexpr std::int64_t seq_len = std::int64_t(image_prompt_length);
    std::vector<std::int64_t> ids(prompt.tokens.begin(), prompt.tokens.end());
    auto indices = at::from_blob(ids.data(), {seq_len}, at::kLong).to(options.device());
    auto x = at::embedding(stage.weight("embed_tokens.weight"), indices).to(at::kFloat);

    // Causal + right-padding mask matching HuggingFace Qwen3 attention_mask.
    auto query_idx = at::arange(seq_len, at::kLong).unsqueeze(1);
    auto key_idx = at::arange(seq_len, at::kLong).unsqueeze(0);
    auto active = std::int64_t(prompt.active_count);
    auto mask_cpu = (key_idx <= query_idx) & (key_idx < active);
    auto attn_mask = mask_cpu.unsqueeze(0).unsqueeze(0).to(options.device());

    auto positions = at::arange(seq_len, options).view({1, 1, seq_len, 1});
    auto frequencies =
        at::exp(at::arange(64, options) * (-2.0 * std::log(1000000.0) / 128.0));
    auto angle = positions * frequencies;
    auto cosine = angle.cos(), sine = angle.sin();

    // FLUX.2 Klein 4B stacks hidden_states[9], hidden_states[18], hidden_states[27]
    // (0-indexed: after layers 8, 17, 26) along the channel dimension -> 7680.
    std::array<Tensor, 3> tapped;
    for (int layer = 0; layer < 27; ++layer) {
        stop(cancel);
        auto p = "layers." + std::to_string(layer) + ".";
        auto normalized = stage.rms(p + "input_layernorm", x);
        auto q = stage.rms(
            p + "self_attn.q_norm",
            heads(stage.linear(p + "self_attn.q_proj", normalized), 32)
        );
        auto k = stage.rms(
            p + "self_attn.k_norm",
            heads(stage.linear(p + "self_attn.k_proj", normalized), 8)
        );
        auto v = heads(stage.linear(p + "self_attn.v_proj", normalized), 8);
        q = text_rope(q, cosine, sine);
        k = text_rope(k, cosine, sine);
        auto mix = at::scaled_dot_product_attention(
            q, k, v, attn_mask, 0.0, false, {}, true
        );
        x = x + stage.linear(p + "self_attn.o_proj", unheads(mix).squeeze(0));
        normalized = stage.rms(p + "post_attention_layernorm", x);
        auto gate = at::silu(stage.linear(p + "mlp.gate_proj", normalized));
        auto up = stage.linear(p + "mlp.up_proj", normalized);
        x = x + stage.linear(p + "mlp.down_proj", gate * up);
        if (layer == 8)
            tapped[0] = x;
        else if (layer == 17)
            tapped[1] = x;
        else if (layer == 26)
            tapped[2] = x;
        if (hook)
            hook("text_layer", layer);
    }
    stop(cancel);
    return cpu_float(at::cat({tapped[0], tapped[1], tapped[2]}, -1));
}

struct Rotary {
    Tensor cosine, sine;
};

Rotary flux2_rotary(
    const std::vector<std::array<std::int64_t, 4>>& coordinates,
    const at::TensorOptions& float_options
) {
    auto double_options = float_options.dtype(at::kDouble);
    auto positions = at::from_blob(
                         const_cast<std::int64_t*>(coordinates.front().data()),
                         {std::int64_t(coordinates.size()), 4},
                         at::kLong
    )
                         .to(double_options);
    constexpr std::array<int, 4> widths{32, 32, 32, 32};
    std::vector<Tensor> angles;
    angles.reserve(4);
    for (int axis = 0; axis < 4; ++axis) {
        auto frequency = at::exp(
            at::arange(0, widths[axis], 2, double_options)
            * (-std::log(2000.0) / double(widths[axis]))
        );
        angles.push_back(positions.select(1, axis).unsqueeze(1) * frequency);
    }
    auto angle = at::cat(angles, 1).unsqueeze(0).unsqueeze(0);
    return {angle.cos().to(at::kFloat), angle.sin().to(at::kFloat)};
}

Tensor rotate_flux2(const Tensor& input, const Rotary& rotary) {
    auto pairs =
        input.to(at::kFloat).reshape({input.size(0), 24, input.size(2), 64, 2});
    auto a = pairs.select(-1, 0), b = pairs.select(-1, 1);
    return at::stack(
               {a * rotary.cosine - b * rotary.sine,
                b * rotary.cosine + a * rotary.sine},
               -1
    )
        .reshape_as(input)
        .to(input.scalar_type());
}

Tensor flux2_swiglu(const Tensor& x) {
    const auto half = x.size(-1) / 2;
    return at::silu(x.slice(-1, 0, half)) * x.slice(-1, half);
}

std::vector<ImageGrid> reference_grids(const ImageRequest& request) {
    std::vector<ImageGrid> grids;
    if (!request.reference)
        return grids;
    grids.push_back(
        {std::int64_t(request.source_height / 16), std::int64_t(request.source_width / 16)}
    );
    for (const auto& extra : request.references)
        grids.push_back(
            {std::int64_t(extra.height / 16), std::int64_t(extra.width / 16)}
        );
    return grids;
}

Bytes total_reference_points(const ImageRequest& request) {
    Bytes total = 0;
    for (const auto& grid : reference_grids(request))
        total += Bytes(grid.height) * Bytes(grid.width);
    return total;
}

Tensor denoise_batch(
    Stage& stage,
    Tensor latent_cpu,
    const Tensor& source_cpu,
    const std::vector<Tensor>& reference_latents_cpu,
    const std::vector<Tensor>& conditions_cpu,
    std::int64_t h,
    std::int64_t w,
    std::span<const ImageRequest> requests,
    const ImageCheckpointConfig& config,
    const std::atomic_bool& cancel,
    const Generator::Hook& hook
) {
    const auto& request = requests.front();
    const auto batch = std::int64_t(requests.size());
    const auto target_tokens = h * w;
    auto options =
        at::TensorOptions().device(at::kCUDA, stage.device).dtype(at::kFloat);
    auto latent = latent_cpu.to(options.device());
    auto times = config.sampling_schedule(std::size_t(target_tokens), request.steps);
    if (!request.source.empty() && !request.reference)
        times = edit_schedule(times, request.strength);

    Tensor source, noise, mask, inverse_mask;
    if (source_cpu.defined() && !request.reference) {
        source = source_cpu.to(options.device());
        noise = latent;
        latent = source * (1.f - times.sigma[0]) + noise * times.sigma[0];
        if (!request.mask.empty()) {
            auto mask_cpu = at::empty({target_tokens, 1}, at::kFloat);
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

    Tensor ref_latents;
    if (request.reference && !reference_latents_cpu.empty()) {
        std::vector<Tensor> gpu_refs;
        gpu_refs.reserve(reference_latents_cpu.size());
        for (const auto& r : reference_latents_cpu)
            gpu_refs.push_back(r.to(options.device(), at::kBFloat16));
        ref_latents =
            at::cat(gpu_refs, 0).unsqueeze(0).expand({batch, -1, 128}).contiguous();
    }

    const auto grids = reference_grids(request);
    const auto layout =
        flux2_position_layout(std::int64_t(image_prompt_length), h, w, grids);
    const auto rotary = flux2_rotary(layout.positions, options);
    const auto text_tokens = layout.text_tokens;

    std::vector<Tensor> projected_texts;
    projected_texts.reserve(requests.size());
    for (std::int64_t i = 0; i < batch; ++i) {
        auto cond = conditions_cpu[i].to(options.device(), at::kBFloat16);
        projected_texts.push_back(stage.linear("context_embedder", cond));
    }
    auto base_txt = at::stack(projected_texts, 0);
    projected_texts.clear();

    auto time_freq = at::exp(at::arange(128, options) * (-std::log(10000.0) / 128.0));

    for (std::size_t step = 0; step < times.steps; ++step) {
        stop(cancel);
        const float sigma = times.sigma[step];
        // Diffusers passes sigma = t/1000 and Flux2Transformer2DModel multiplies by 1000.
        auto angle =
            at::scalar_tensor(double(sigma) * 1000.0, options).unsqueeze(0).unsqueeze(1)
            * time_freq;
        auto time_proj = at::cat({angle.cos(), angle.sin()}, 1).to(at::kBFloat16);
        auto temb = stage.linear(
            "time_guidance_embed.timestep_embedder.linear_2",
            at::silu(stage.linear(
                "time_guidance_embed.timestep_embedder.linear_1", time_proj
            ))
        );
        auto activated_temb = at::silu(temb);
        Flux2DoubleModulation mod_img(
            stage.linear("double_stream_modulation_img.linear", activated_temb)
        );
        Flux2DoubleModulation mod_txt(
            stage.linear("double_stream_modulation_txt.linear", activated_temb)
        );
        Flux2SingleModulation mod_single(
            stage.linear("single_stream_modulation.linear", activated_temb)
        );

        auto img_input = ref_latents.defined()
                             ? at::cat({latent.to(at::kBFloat16), ref_latents}, 1)
                             : latent.to(at::kBFloat16);
        auto hidden_states = stage.linear("x_embedder", img_input);
        auto encoder_hidden_states = base_txt;

        if (hook)
            hook("conditioning", step);

        for (int layer = 0; layer < 5; ++layer) {
            stop(cancel);
            if (hook)
                hook("block_started", layer);
            auto p = "transformer_blocks." + std::to_string(layer) + ".";
            auto norm_img = flux2_modulated_norm(
                hidden_states, mod_img.attention.shift, mod_img.attention.scale
            );
            auto norm_txt = flux2_modulated_norm(
                encoder_hidden_states, mod_txt.attention.shift, mod_txt.attention.scale
            );
            auto img_q = stage.rms(
                p + "attn.norm_q", heads(stage.linear(p + "attn.to_q", norm_img), 24)
            );
            auto img_k = stage.rms(
                p + "attn.norm_k", heads(stage.linear(p + "attn.to_k", norm_img), 24)
            );
            auto img_v = heads(stage.linear(p + "attn.to_v", norm_img), 24);

            auto txt_q = stage.rms(
                p + "attn.norm_added_q",
                heads(stage.linear(p + "attn.add_q_proj", norm_txt), 24)
            );
            auto txt_k = stage.rms(
                p + "attn.norm_added_k",
                heads(stage.linear(p + "attn.add_k_proj", norm_txt), 24)
            );
            auto txt_v = heads(stage.linear(p + "attn.add_v_proj", norm_txt), 24);

            auto q = rotate_flux2(at::cat({txt_q, img_q}, 2), rotary);
            auto k = rotate_flux2(at::cat({txt_k, img_k}, 2), rotary);
            auto v = at::cat({txt_v, img_v}, 2);
            auto mix = unheads(at::scaled_dot_product_attention(q, k, v));

            auto txt_attn = stage.linear(
                p + "attn.to_add_out", mix.slice(1, 0, text_tokens)
            );
            auto img_attn = stage.linear(
                p + "attn.to_out.0", mix.slice(1, text_tokens)
            );

            hidden_states = flux2_gated_residual(
                hidden_states, img_attn, mod_img.attention.gate
            );
            norm_img = flux2_modulated_norm(
                hidden_states, mod_img.mlp.shift, mod_img.mlp.scale
            );
            auto img_ff = stage.linear(
                p + "ff.linear_out",
                flux2_swiglu(stage.linear(p + "ff.linear_in", norm_img))
            );
            hidden_states = flux2_gated_residual(
                hidden_states, img_ff, mod_img.mlp.gate
            );

            encoder_hidden_states = flux2_gated_residual(
                encoder_hidden_states, txt_attn, mod_txt.attention.gate
            );
            norm_txt = flux2_modulated_norm(
                encoder_hidden_states, mod_txt.mlp.shift, mod_txt.mlp.scale
            );
            auto txt_ff = stage.linear(
                p + "ff_context.linear_out",
                flux2_swiglu(stage.linear(p + "ff_context.linear_in", norm_txt))
            );
            encoder_hidden_states = flux2_gated_residual(
                encoder_hidden_states, txt_ff, mod_txt.mlp.gate
            );

            if (hook)
                hook("block", layer);
        }

        auto x = at::cat({encoder_hidden_states, hidden_states}, 1);
        for (int layer = 0; layer < 20; ++layer) {
            stop(cancel);
            if (hook)
                hook("block_started", 5 + layer);
            auto p = "single_transformer_blocks." + std::to_string(layer) + ".attn.";
            auto n = flux2_modulated_norm(
                x, mod_single.parallel.shift, mod_single.parallel.scale
            );
            auto proj = stage.linear(p + "to_qkv_mlp_proj", n);
            auto qkv = proj.slice(-1, 0, 9216);
            auto mlp_in = proj.slice(-1, 9216);

            auto q = rotate_flux2(
                stage.rms(p + "norm_q", heads(qkv.slice(-1, 0, 3072), 24)), rotary
            );
            auto k = rotate_flux2(
                stage.rms(p + "norm_k", heads(qkv.slice(-1, 3072, 6144), 24)), rotary
            );
            auto v = heads(qkv.slice(-1, 6144, 9216), 24);
            auto attn_out = unheads(at::scaled_dot_product_attention(q, k, v));
            auto mlp_out = flux2_swiglu(mlp_in);
            auto out = stage.linear(p + "to_out", at::cat({attn_out, mlp_out}, -1));
            x = flux2_gated_residual(x, out, mod_single.parallel.gate);
            if (hook)
                hook("block", 5 + layer);
        }

        auto target_x = x.slice(1, text_tokens, text_tokens + target_tokens);
        // AdaLayerNormContinuous splits into (scale, shift).
        auto out_mod = stage.linear("norm_out.linear", activated_temb).unsqueeze(1);
        auto out_scale = out_mod.slice(-1, 0, 3072).to(at::kFloat).add(1);
        auto out_shift = out_mod.slice(-1, 3072).to(at::kFloat);
        auto normalized = flux2_modulated_norm(target_x, out_shift, out_scale);
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

Tensor vae_group_norm(Stage& stage, const std::string& p, const Tensor& x, bool activate) {
    auto y = at::group_norm(
        x, 32, stage.weight(p + ".weight"), stage.weight(p + ".bias"), 1e-6
    );
    if (activate)
        at::silu_(y);
    return y;
}

Tensor vae_conv(
    Stage& stage, const std::string& p, const Tensor& x, bool downsample = false
) {
    const auto weight = stage.weight(p + ".weight");
    const auto bias = stage.weight(p + ".bias");
    if (downsample)
        return at::conv2d(
            at::constant_pad_nd(x, {0, 1, 0, 1}, 0),
            weight,
            bias,
            {2, 2},
            at::IntArrayRef{0, 0}
        );
    return at::conv2d(
        x, weight, bias, {1, 1}, {weight.size(2) / 2, weight.size(3) / 2}
    );
}

Tensor vae_residual(
    Stage& stage, const std::string& p, const Tensor& x, const std::atomic_bool& cancel
) {
    stop(cancel);
    auto y = vae_conv(stage, p + ".conv1", vae_group_norm(stage, p + ".norm1", x, true));
    stop(cancel);
    y = vae_conv(stage, p + ".conv2", vae_group_norm(stage, p + ".norm2", y, true));
    return y.add_(
        x.size(1) == y.size(1) ? x : vae_conv(stage, p + ".conv_shortcut", x)
    );
}

Tensor vae_attention(
    Stage& stage, const std::string& p, const Tensor& x, const std::atomic_bool& cancel
) {
    auto h = vae_group_norm(stage, p + ".group_norm", x, false);
    const auto c = x.size(1);
    const auto points = x.size(2) * x.size(3);
    auto seq = h.flatten(2).transpose(1, 2);  // [1, points, C]
    auto q = stage.linear_bias(p + ".to_q", seq).unsqueeze(1);  // [1, 1, points, C]
    auto k = stage.linear_bias(p + ".to_k", seq).unsqueeze(1);
    auto v = stage.linear_bias(p + ".to_v", seq).unsqueeze(1);
    auto mix = at::empty_like(q);
    for (std::int64_t row = 0; row < points; row += 256) {
        stop(cancel);
        auto end = std::min(row + 256, points);
        mix.slice(2, row, end)
            .copy_(at::scaled_dot_product_attention(q.slice(2, row, end), k, v));
    }
    auto out = stage.linear_bias(p + ".to_out.0", mix.squeeze(1));
    return x + out.transpose(1, 2).reshape({1, c, x.size(2), x.size(3)});
}

Tensor encode_vae_blocks(
    Stage& stage, Tensor x, const std::atomic_bool& cancel, const Generator::Hook& hook
) {
    x = vae_conv(stage, "encoder.conv_in", x);
    for (int i = 0; i < 4; ++i) {
        stop(cancel);
        const auto p = "encoder.down_blocks." + std::to_string(i);
        for (int j = 0; j < 2; ++j)
            x = vae_residual(stage, p + ".resnets." + std::to_string(j), x, cancel);
        if (i < 3)
            x = vae_conv(stage, p + ".downsamplers.0.conv", x, true);
        if (hook)
            hook("vae_down_block", i);
    }
    return x;
}

Tensor encode_pixels(
    Stage& stage,
    std::span<const float> rgba_pixels,
    std::size_t width,
    std::size_t height,
    const ImageCheckpointConfig& config,
    const std::atomic_bool& cancel,
    const Generator::Hook& hook
) {
    const auto device = at::Device(at::kCUDA, stage.device);
    auto x = at::from_blob(
                 const_cast<float*>(rgba_pixels.data()),
                 {1, 4, std::int64_t(height), std::int64_t(width)},
                 at::kFloat
    )
                 .slice(1, 0, 3)
                 .contiguous()
                 .to(device);
    const auto h = std::int64_t(height / 16), w = std::int64_t(width / 16);
    if (stage.tiled_vae()) {
        auto encoded = at::empty({1, 512, h * 2, w * 2}, x.options());
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
                encoded.slice(2, rows.begin * 2, rows.end * 2)
                    .slice(3, cols.begin * 2, cols.end * 2)
                    .copy_(tile.slice(
                                   2,
                                   rows.crop_begin() * 2,
                                   (rows.crop_begin() + rows.end - rows.begin) * 2
                    )
                               .slice(
                                   3,
                                   cols.crop_begin() * 2,
                                   (cols.crop_begin() + cols.end - cols.begin) * 2
                               ));
            }
        }
        x = std::move(encoded);
    } else {
        x = encode_vae_blocks(stage, x, cancel, hook);
    }
    x = vae_residual(stage, "encoder.mid_block.resnets.0", x, cancel);
    x = vae_attention(stage, "encoder.mid_block.attentions.0", x, cancel);
    x = vae_residual(stage, "encoder.mid_block.resnets.1", x, cancel);
    x = vae_conv(
        stage, "encoder.conv_out", vae_group_norm(stage, "encoder.conv_norm_out", x, true)
    );
    x = vae_conv(stage, "quant_conv", x).slice(1, 0, 32);
    // 2x2 patchify: [1, 32, h*2, w*2] -> [1, 128, h, w]
    x = x.reshape({1, 32, h, 2, w, 2})
            .permute({0, 1, 3, 5, 2, 4})
            .reshape({1, 128, h, w});
    auto bn_mean = stage.weight("bn.running_mean").reshape({1, 128, 1, 1});
    auto bn_std =
        (stage.weight("bn.running_var").reshape({1, 128, 1, 1}) + config.batch_norm_eps)
            .sqrt();
    stop(cancel);
    return cpu_float(((x - bn_mean) / bn_std).flatten(2).squeeze(0).transpose(0, 1));
}

Tensor decode_vae_blocks(
    Stage& stage, Tensor x, const std::atomic_bool& cancel, const Generator::Hook& hook
) {
    for (int i = 0; i < 4; ++i) {
        stop(cancel);
        const auto p = "decoder.up_blocks." + std::to_string(i);
        for (int j = 0; j < 3; ++j)
            x = vae_residual(stage, p + ".resnets." + std::to_string(j), x, cancel);
        if (i < 3) {
            x = at::upsample_nearest2d(x, {x.size(2) * 2, x.size(3) * 2}, {}, {});
            x = vae_conv(stage, p + ".upsamplers.0.conv", x);
        }
        if (hook)
            hook("vae_up_block", i);
    }
    x = vae_conv(
        stage, "decoder.conv_out", vae_group_norm(stage, "decoder.conv_norm_out", x, true)
    );
    stop(cancel);
    auto rgb = x.clamp(-1, 1);
    auto alpha = at::ones({1, 1, rgb.size(2), rgb.size(3)}, rgb.options());
    return at::cat({rgb, alpha}, 1);
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
    const auto device = at::Device(at::kCUDA, stage.device);
    auto bn_mean = stage.weight("bn.running_mean").reshape({1, 128, 1, 1});
    auto bn_std =
        (stage.weight("bn.running_var").reshape({1, 128, 1, 1}) + config.batch_norm_eps)
            .sqrt();
    auto x = latent.to(device).transpose(0, 1).reshape({1, 128, h, w}) * bn_std
             + bn_mean;
    // 2x2 unpatchify: [1, 128, h, w] -> [1, 32, h*2, w*2]
    x = x.reshape({1, 32, 2, 2, h, w})
            .permute({0, 1, 4, 2, 5, 3})
            .reshape({1, 32, h * 2, w * 2});
    x = vae_conv(stage, "post_quant_conv", x);
    x = vae_conv(stage, "decoder.conv_in", x);
    x = vae_residual(stage, "decoder.mid_block.resnets.0", x, cancel);
    x = vae_attention(stage, "decoder.mid_block.attentions.0", x, cancel);
    x = vae_residual(stage, "decoder.mid_block.resnets.1", x, cancel);
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
                x.slice(2, rows.input_begin * 2, rows.input_end * 2)
                    .slice(3, cols.input_begin * 2, cols.input_end * 2)
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
    Resources& resources;
    Handle metadata;
    text::Tokenizer tokenizer;
    ImageCheckpointConfig config;
    std::string root;
    std::unique_ptr<Inventory> text, denoiser, vae, vae_encoder;
    std::shared_ptr<checkpoint::ReadCache> weight_cache;
    DeviceWeightCache device_cache;
    AtenDenoiserPerformance performance;
    bool cache_populated = false;

    AtenDenoiserWorkload
    workload(Bytes points, Bytes tokens, Bytes reference, std::size_t steps) const {
        bool retained = false;
        for (std::size_t device = 0; device + 1 < resources.snapshot().capacity.size();
             ++device)
            retained = retained || device_cache.contains(*denoiser, int(device));
        return {points, tokens, reference, steps, cache_populated, retained};
    }

    std::vector<AtenDenoiserLane> denoiser_plan(
        const Snapshot& snapshot,
        const std::vector<int>& devices,
        Bytes points,
        Bytes tokens,
        Bytes reference,
        std::size_t images,
        std::size_t steps
    ) const {
        const auto parallel = aten_denoiser_plan(
            device_cache.reclaimable_snapshot(snapshot),
            devices,
            DeviceWeightCache::weight_bytes(*denoiser),
            denoiser->largest,
            points,
            tokens,
            reference,
            images
        );
        std::vector<AtenDenoiserLane> single;
        try {
            single.push_back(
                {0,
                 images,
                 device_cache.plan(
                     snapshot,
                     devices,
                     *denoiser,
                     aten_batch_workspace(points, tokens, reference, images)
                 )}
            );
        } catch (const ResourceExhausted&) {
        }
        const auto cost = [&](const std::vector<AtenDenoiserLane>& lanes) {
            AtenDenoiserCost estimate;
            estimate.load_seconds.reserve(lanes.size());
            for (const auto& lane : lanes) {
                estimate.load_seconds.push_back(
                    device_cache.contains(*denoiser, lane.stage.device)
                        ? std::optional<double>(0)
                        : device_cache.load_cost(*denoiser, lane.stage.device)
                );
                estimate.deferred_reload_seconds +=
                    device_cache.eviction_cost(snapshot, *denoiser, lane.stage);
            }
            return estimate;
        };
        std::vector<std::vector<AtenDenoiserLane>> candidates;
        candidates.reserve(parallel.size() + 1);
        candidates.push_back(std::move(single));
        std::vector<int> subset;
        subset.reserve(parallel.size());
        for (const auto& lane : parallel) {
            subset.push_back(lane.stage.device);
            if (subset.size() < 2)
                continue;
            try {
                candidates.push_back(aten_denoiser_plan(
                    device_cache.reclaimable_snapshot(snapshot),
                    subset,
                    DeviceWeightCache::weight_bytes(*denoiser),
                    denoiser->largest,
                    points,
                    tokens,
                    reference,
                    images
                ));
            } catch (const ResourceExhausted&) {
            }
        }
        if (parallel.size() < 2)
            candidates.push_back(parallel);
        std::vector<AtenDenoiserCost> costs;
        costs.reserve(candidates.size());
        for (const auto& candidate : candidates)
            costs.push_back(cost(candidate));
        return performance.choose_candidates(
            workload(points, tokens, reference, steps), candidates, costs
        );
    }

    auto inventories() {
        return std::array<Inventory*, 4>{
            text.get(), denoiser.get(), vae.get(), vae_encoder.get()
        };
    }

    void cache_reads(const std::atomic_bool* cancel) {
        for (auto* inventory : inventories())
            if (inventory)
                for (auto& shard : inventory->shards)
                    shard->cache_reads(cancel ? weight_cache : nullptr, cancel);
    }

    void prepare_cache(const std::shared_ptr<Resources>& owner, std::size_t lanes) {
        if (weight_cache)
            return;
        Bytes stored = 0, staging = 0;
        for (const auto* inventory : inventories()) {
            if (!inventory)
                continue;
            for (const auto& [name, weight] : inventory->weights) {
                if (weight.info.bytes > std::numeric_limits<Bytes>::max() - stored)
                    throw std::overflow_error("image_weight_cache_inventory");
                stored += weight.info.bytes;
            }
            const auto copies =
                inventory->stage == AtenImageStage::denoiser ? lanes : 1;
            staging = std::max(staging, copies * (2 * inventory->largest + 256 * mib));
        }
        const auto snapshot = resources.snapshot();
        const auto bytes = image_weight_cache_bytes(
            snapshot.capacity[0] - snapshot.used[0], staging, stored
        );
        if (bytes)
            weight_cache =
                std::make_shared<checkpoint::ReadCache>(owner, bytes, Workload::image);
    }

    struct CacheReads {
        Impl& model;
        checkpoint::ReadCache::Stats before;

        CacheReads(Impl& m, const std::atomic_bool& cancel) : model(m) {
            if (model.weight_cache)
                before = model.weight_cache->stats();
            model.cache_reads(&cancel);
        }

        ~CacheReads() {
            model.cache_reads(nullptr);
            if (!model.weight_cache)
                return;
            const auto after = model.weight_cache->stats();
            std::fprintf(
                stderr,
                "KADAN_ATEN_IMAGE_CACHE {\"location\":\"host\","
                "\"reserved_bytes\":%llu,\"allocated_bytes\":%zu,"
                "\"retained_before_bytes\":%zu,\"entries\":%zu,"
                "\"hits\":%llu,\"misses\":%llu,\"source_bytes\":%llu}\n",
                static_cast<unsigned long long>(model.weight_cache->reserved_bytes()),
                after.allocated,
                before.allocated,
                after.entries,
                static_cast<unsigned long long>(after.hits - before.hits),
                static_cast<unsigned long long>(after.misses - before.misses),
                static_cast<unsigned long long>(
                    after.source_bytes - before.source_bytes
                )
            );
            std::fflush(stderr);
        }
    };

    explicit Impl(const std::shared_ptr<Resources>& r)
        : resources(*r), metadata(0), tokenizer(r, Workload::image), device_cache(*r) {
        auto bytes = r->snapshot().capacity;
        std::fill(bytes.begin(), bytes.end(), 0);
        bytes[0] = 192 * mib;
        metadata = r->reserve(Workload::image, bytes);
    }

    ~Impl() {
        try {
            device_cache.clear();
        } catch (...) {
        }
        vae_encoder.reset();
        vae.reset();
        denoiser.reset();
        text.reset();
        weight_cache.reset();
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
    model->tokenizer.load(root + "/tokenizer/tokenizer.json", cancel);
    const std::array<std::string, 1> vae_files{"diffusion_pytorch_model.safetensors"};
    model->text = std::make_unique<Inventory>(
        root + "/text_encoder", model->config.text_files, AtenImageStage::text, cancel
    );
    model->denoiser = std::make_unique<Inventory>(
        root + "/transformer",
        model->config.denoiser_files,
        AtenImageStage::denoiser,
        cancel
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
    if (model_)
        model_->device_cache.clear();
    model_.reset();
}

ImageDeviceCacheMemory AtenGenerator::device_cache_memory() const {
    ImageDeviceCacheMemory memory;
    memory.allocated.resize(resources_->snapshot().capacity.size());
    memory.reserved.resize(memory.allocated.size());
    if (model_)
        for (int device : devices_) {
            memory.allocated.at(device + 1) =
                model_->device_cache.allocated_bytes(device);
            memory.reserved.at(device + 1) = model_->device_cache.device_bytes(device);
        }
    return memory;
}

void AtenGenerator::prepare(
    const ImageRequest& request, const std::atomic_bool& cancel
) {
    Generator::validate(request);
    (void)admission_plan(request, 1, cancel);
}

ImageBatchPlan AtenGenerator::admission_plan(
    const ImageRequest& request, std::size_t images, const std::atomic_bool& cancel
) {
    need(bool(model_), "image_generator_not_loaded");
    stop(cancel);
    auto snapshot = stage_snapshot(*resources_, devices_);
    ImageBatchPlan result{Footprint(snapshot.capacity.size()), {}};
    auto& footprint = result.memory;
    if (!request.source.empty() && !request.reference && request.strength == 0)
        return result;
    auto& m = *model_;
    (void)prepare_image_prompt(m.tokenizer, request.prompt, cancel);
    const auto points = (request.height / 16) * (request.width / 16);
    const auto reference_points = total_reference_points(request);
    Bytes max_encoder_points = points;
    if (request.reference) {
        max_encoder_points = 0;
        for (const auto& grid : reference_grids(request))
            max_encoder_points =
                std::max(max_encoder_points, Bytes(grid.height) * Bytes(grid.width));
    }
    std::vector<const Inventory*> inventories{
        m.text.get(), m.denoiser.get(), m.vae.get()
    };
    if (request.reference
        || (!request.source.empty() && (request.strength < 1 || !request.mask.empty())))
        inventories.push_back(&m.encoder(cancel));
    for (const auto* inventory : inventories) {
        stop(cancel);
        const auto tokens = Bytes(image_prompt_length);
        const auto stage_points =
            inventory->stage == AtenImageStage::vae_encoder ? max_encoder_points : points;
        Footprint stage(footprint.size());
        if (inventory->stage == AtenImageStage::denoiser
            && (request.source.empty() || request.reference)) {
            const auto lanes = m.denoiser_plan(
                snapshot,
                devices_,
                points,
                tokens,
                reference_points,
                images,
                request.steps
            );
            result.cost.denoise_seconds = m.performance.denoise_seconds(
                m.workload(points, tokens, reference_points, request.steps), lanes
            );
            for (const auto& lane : lanes) {
                result.cost.streamed_weights =
                    result.cost.streamed_weights || lane.stage.stream_weights;
                if (!lane.stage.stream_weights
                    && !m.device_cache.contains(*inventory, lane.stage.device))
                    result.cost.loading_seconds +=
                        m.device_cache.load_cost(*inventory, lane.stage.device)
                            .value_or(0);
                result.cost.loading_seconds +=
                    m.device_cache.eviction_cost(snapshot, *inventory, lane.stage);
                stage[0] += lane.stage.host_bytes;
                stage[lane.stage.device + 1] += lane.stage.device_bytes;
            }
        } else {
            const auto plan = m.device_cache.plan(
                snapshot,
                devices_,
                *inventory,
                aten_stage_workspace(
                    inventory->stage,
                    stage_points,
                    tokens,
                    !request.source.empty() && !request.reference,
                    reference_points
                ),
                aten_tiled_vae_workspace(inventory->stage, stage_points)
            );
            result.cost.streamed_weights =
                result.cost.streamed_weights || plan.stream_weights;
            if (!plan.stream_weights
                && !m.device_cache.contains(*inventory, plan.device))
                result.cost.loading_seconds +=
                    m.device_cache.load_cost(*inventory, plan.device).value_or(0);
            result.cost.loading_seconds +=
                m.device_cache.eviction_cost(snapshot, *inventory, plan);
            stage[0] = plan.host_bytes;
            stage[plan.device + 1] = plan.device_bytes;
        }
        for (std::size_t i = 0; i < footprint.size(); ++i)
            footprint[i] = std::max(footprint[i], stage[i]);
    }
    if (result.cost.streamed_weights)
        result.cost.denoise_seconds.reset();
    return result;
}

void AtenGenerator::prepare_batch(
    std::span<const ImageRequest> requests, const std::atomic_bool& cancel
) {
    need(!requests.empty(), "image_batch_empty");
    for (const auto& request : requests) {
        need(
            requests.size() == 1 || Generator::compatible(requests.front(), request),
            "image_batch_incompatible"
        );
        prepare(request, cancel);
    }
    if (requests.size() == 1)
        return;
    const auto& request = requests.front();
    const auto points = request.height / 16 * (request.width / 16);
    const auto reference = total_reference_points(request);
    (void)model_->denoiser_plan(
        stage_snapshot(*resources_, devices_),
        devices_,
        points,
        image_prompt_length,
        reference,
        requests.size(),
        request.steps
    );
}

std::vector<ImageBatchPlan> AtenGenerator::admissible_batch_plans(
    const ImageRequest& request, std::size_t limit, const std::atomic_bool& cancel
) {
    stop(cancel);
    if ((!request.source.empty() && !request.reference) || !request.mask.empty())
        limit = std::min<std::size_t>(limit, 1);
    std::vector<ImageBatchPlan> plans;
    plans.reserve(limit);
    for (std::size_t count = 1; count <= limit; ++count) {
        try {
            plans.push_back(admission_plan(request, count, cancel));
        } catch (const ResourceExhausted&) {
            break;
        }
    }
    return plans;
}

void AtenGenerator::generate(
    const ImageRequest& request,
    std::span<float> output,
    const std::atomic_bool& cancel,
    const Generator::Hook& hook
) {
    generate_batch(
        {&request, 1},
        {&output, 1},
        cancel,
        [&](std::size_t, const char* event, std::size_t value) {
            if (hook)
                hook(event, value);
        }
    );
}

void AtenGenerator::generate_batch(
    std::span<const ImageRequest> requests,
    std::span<const std::span<float>> outputs,
    const std::atomic_bool& cancel,
    const Generator::BatchHook& batch_hook
) {
    need(bool(model_), "image_generator_not_loaded");
    need(!requests.empty() && requests.size() == outputs.size(), "image_batch_shape");
    const auto& request = requests.front();
    for (std::size_t i = 0; i < requests.size(); ++i) {
        Generator::validate(requests[i]);
        need(
            outputs[i].size() == requests[i].width * requests[i].height * 4,
            "image_output_shape"
        );
        need(
            requests.size() == 1 || Generator::compatible(request, requests[i]),
            "image_batch_incompatible"
        );
    }
    stop(cancel);
    measurement_ = {};
    measurement_.images = requests.size();
    measurement_.peak_allocated.resize(resources_->snapshot().capacity.size());
    measurement_.peak_reserved.resize(measurement_.peak_allocated.size());
    measurement_.baseline_allocated.resize(measurement_.peak_allocated.size());
    measurement_.baseline_reserved.resize(measurement_.peak_allocated.size());
    measurement_.additional_image.resize(measurement_.peak_allocated.size());
    measurement_.baseline_allocated[0] = resources_->snapshot().used[0];
    measurement_.baseline_reserved[0] = measurement_.baseline_allocated[0];
    at::globalContext().lazyInitDevice(c10::DeviceType::CUDA);
    for (const auto device : devices_) {
        const auto stats = c10::cuda::CUDACachingAllocator::getDeviceStats(device);
        measurement_.baseline_allocated[device + 1] = stats.allocated_bytes[0].current;
        measurement_.baseline_reserved[device + 1] = stats.reserved_bytes[0].current;
    }
    const auto indexed_hook = [&](std::size_t index) -> Generator::Hook {
        return [&, index](const char* event, std::size_t value) {
            if (batch_hook)
                batch_hook(index, event, value);
        };
    };
    const Generator::Hook hook = [&](const char* event, std::size_t value) {
        for (std::size_t i = 0; i < requests.size(); ++i)
            if (batch_hook)
                batch_hook(i, event, value);
    };
    if (!request.source.empty() && !request.reference && request.strength == 0) {
        std::copy(
            request.source.begin(), request.source.end(), outputs.front().begin()
        );
        return;
    }
    c10::InferenceMode inference;
    ArithmeticScope arithmetic;
    auto& m = *model_;
    std::vector<ImagePrompt> prompts;
    prompts.reserve(requests.size());
    for (const auto& item : requests)
        prompts.push_back(prepare_image_prompt(m.tokenizer, item.prompt, cancel));

    const auto h = request.height / 16, w = request.width / 16, points = h * w;
    const auto reference_points = total_reference_points(request);
    const Bytes text_tokens = image_prompt_length;

    if (!request.source.empty()
        && (request.reference || request.strength < 1 || !request.mask.empty()))
        (void)m.encoder(cancel);
    m.prepare_cache(resources_, std::min(requests.size(), devices_.size()));
    Impl::CacheReads cache_reads(m, cancel);

    Tensor source;
    std::vector<Tensor> reference_latents;
    if (!request.source.empty()
        && (request.reference || request.strength < 1 || !request.mask.empty())) {
        Bytes max_source_points =
            request.reference ? (request.source_height / 16) * (request.source_width / 16)
                              : points;
        if (request.reference) {
            for (const auto& extra : request.references)
                max_source_points = std::max(
                    max_source_points, Bytes(extra.height / 16) * Bytes(extra.width / 16)
                );
        }
        auto started = Clock::now();
        Stage stage(
            *resources_,
            m.device_cache,
            devices_,
            m.encoder(cancel),
            aten_stage_workspace(AtenImageStage::vae_encoder, max_source_points, 0),
            cancel,
            aten_tiled_vae_workspace(AtenImageStage::vae_encoder, max_source_points),
            &measurement_
        );
        stage_timing("load", "vae_encoder", stage.device, started);
        c10::cuda::CUDAGuard guard(stage.device);
        started = Clock::now();
        const auto first_w = request.reference ? request.source_width : request.width;
        const auto first_h = request.reference ? request.source_height : request.height;
        auto first_encoded = encode_pixels(
            stage, request.source, first_w, first_h, m.config, cancel, hook
        );
        if (request.reference) {
            reference_latents.push_back(std::move(first_encoded));
            for (const auto& extra : request.references) {
                reference_latents.push_back(encode_pixels(
                    stage, extra.pixels, extra.width, extra.height, m.config, cancel, hook
                ));
            }
        } else {
            source = std::move(first_encoded);
        }
        stage.sync();
        stage_timing("generation", "vae_encoder", stage.device, started);
        started = Clock::now();
        stage.close();
        stage_timing("unload", "vae_encoder", stage.device, started);
    }

    std::vector<Tensor> conditions;
    conditions.reserve(requests.size());
    {
        auto started = Clock::now();
        Stage stage(
            *resources_,
            m.device_cache,
            devices_,
            *m.text,
            aten_stage_workspace(AtenImageStage::text, points, text_tokens),
            cancel,
            0,
            &measurement_
        );
        stage_timing("load", "text", stage.device, started);
        c10::cuda::CUDAGuard guard(stage.device);
        started = Clock::now();
        for (std::size_t i = 0; i < requests.size(); ++i)
            conditions.push_back(
                encode_text(stage, prompts[i], cancel, indexed_hook(i))
            );
        stage.sync();
        stage_timing("generation", "text", stage.device, started);
        started = Clock::now();
        stage.close();
        stage_timing("unload", "text", stage.device, started);
    }

    auto latent = at::empty(
        {std::int64_t(requests.size()), std::int64_t(points), 128}, at::kFloat
    );
    for (std::size_t i = 0; i < requests.size(); ++i)
        seed_image_latent(
            {latent[i].data_ptr<float>(), points * 128}, requests[i].seed, cancel
        );

    std::optional<AtenDenoiserWorkload> completed_workload;
    std::vector<AtenDenoiserMeasurement> completed_timings;
    {
        const auto snapshot = stage_snapshot(*resources_, devices_);
        const auto editing = !request.source.empty() && !request.reference;
        const auto lanes =
            editing
                ? std::vector<AtenDenoiserLane>{{0, 1, m.device_cache.plan(snapshot, devices_, *m.denoiser, aten_stage_workspace(AtenImageStage::denoiser, points, text_tokens, true))}}
                : m.denoiser_plan(
                      snapshot,
                      devices_,
                      points,
                      text_tokens,
                      reference_points,
                      requests.size(),
                      request.steps
                  );
        const auto workload =
            m.workload(points, text_tokens, reference_points, request.steps);
        std::vector<AtenDenoiserMeasurement> timings(lanes.size());
        std::vector<std::unique_ptr<Stage>> stages;
        stages.reserve(lanes.size());
        for (const auto& lane : lanes) {
            const auto started = Clock::now();
            stages.push_back(
                std::make_unique<Stage>(
                    *resources_,
                    m.device_cache,
                    *m.denoiser,
                    lane.stage,
                    cancel,
                    &measurement_,
                    stages.empty() ? nullptr : stages.front().get()
                )
            );
            timings[stages.size() - 1] = {
                lane.stage.device,
                lane.images,
                stages.size() > 1,
                std::chrono::duration<double>(Clock::now() - started).count(),
                0
            };
            stage_timing("load", "denoiser", lane.stage.device, started);
            constexpr Bytes fixed = 2ULL * 1024 * 1024 * 1024;
            const auto workspace =
                editing ? aten_stage_workspace(
                              AtenImageStage::denoiser, points, text_tokens, true
                          )
                        : aten_batch_workspace(
                              points, text_tokens, reference_points, lane.images
                          );
            measurement_.additional_image[lane.stage.device + 1] =
                (workspace - fixed) / lane.images;
        }
        std::mutex progress_mutex;
        std::vector<Tensor> results(lanes.size());
        std::vector<std::exception_ptr> failures(lanes.size());
        const auto run_lane = [&](std::size_t index) {
            const auto& lane = lanes[index];
            auto& stage = *stages[index];
            const auto started = Clock::now();
            bool completed = false;
            try {
                c10::InferenceMode lane_inference;
                c10::cuda::CUDAGuard guard(stage.device);
                const std::vector<Tensor> lane_conditions(
                    conditions.begin() + lane.begin,
                    conditions.begin() + lane.begin + lane.images
                );
                results[index] = denoise_batch(
                    stage,
                    latent.slice(0, lane.begin, lane.begin + lane.images),
                    source,
                    reference_latents,
                    lane_conditions,
                    std::int64_t(h),
                    std::int64_t(w),
                    requests.subspan(lane.begin, lane.images),
                    m.config,
                    cancel,
                    [&](const char* event, std::size_t value) {
                        std::lock_guard lock(progress_mutex);
                        for (std::size_t i = lane.begin; i < lane.begin + lane.images;
                             ++i)
                            if (batch_hook)
                                batch_hook(i, event, value);
                    }
                );
                stage.sync();
                completed = true;
            } catch (...) {
                failures[index] = std::current_exception();
            }
            const auto finished = Clock::now();
            timings[index].denoise_seconds =
                std::chrono::duration<double>(finished - started).count();
            std::fprintf(
                stderr,
                "KADAN_ATEN_IMAGE_TIMING "
                "{\"phase\":\"generation\",\"stage\":\"denoiser\",\"device\":%d,"
                "\"images\":%zu,\"begin\":%zu,\"started_seconds\":%.9f,"
                "\"finished_seconds\":%.9f,\"seconds\":%.9f,\"completed\":%s}\n",
                stage.device,
                lane.images,
                lane.begin,
                std::chrono::duration<double>(started.time_since_epoch()).count(),
                std::chrono::duration<double>(finished.time_since_epoch()).count(),
                std::chrono::duration<double>(finished - started).count(),
                completed ? "true" : "false"
            );
            std::fflush(stderr);
        };
        {
            std::vector<std::jthread> threads;
            threads.reserve(lanes.size());
            for (std::size_t i = 0; i < lanes.size(); ++i)
                threads.emplace_back([&, i, phase = serving::cuda_phase] {
                    serving::CudaPhaseScope tracing(phase);
                    run_lane(i);
                });
        }
        for (auto& stage : stages) {
            const auto started = Clock::now();
            stage->close();
            stage_timing("unload", "denoiser", stage->device, started);
        }
        for (const auto& failure : failures)
            if (failure)
                std::rethrow_exception(failure);
        for (const auto& timing : timings)
            measurement_.denoise_seconds =
                std::max(measurement_.denoise_seconds, timing.denoise_seconds);
        if (!editing && std::none_of(lanes.begin(), lanes.end(), [](const auto& lane) {
                return lane.stage.stream_weights;
            })) {
            completed_workload = workload;
            completed_timings = std::move(timings);
        }
        latent = at::cat(results, 0);
    }

    {
        auto started = Clock::now();
        Stage stage(
            *resources_,
            m.device_cache,
            devices_,
            *m.vae,
            aten_stage_workspace(AtenImageStage::vae, points, text_tokens),
            cancel,
            aten_tiled_vae_workspace(AtenImageStage::vae, points),
            &measurement_
        );
        stage_timing("load", "vae", stage.device, started);
        c10::cuda::CUDAGuard guard(stage.device);
        started = Clock::now();
        for (std::size_t i = 0; i < requests.size(); ++i) {
            auto decoded = decode(
                stage,
                latent[i],
                std::int64_t(h),
                std::int64_t(w),
                m.config,
                cancel,
                indexed_hook(i)
            );
            stage.sync();
            std::copy_n(
                decoded.data_ptr<float>(), outputs[i].size(), outputs[i].data()
            );
        }
        stage_timing("generation", "vae", stage.device, started);
        started = Clock::now();
        stage.close();
        stage_timing("unload", "vae", stage.device, started);
    }

    if (!request.mask.empty())
        composite_image_edit(outputs.front(), request.source, request.mask);
    stop(cancel);
    if (completed_workload)
        m.performance.observe(*completed_workload, completed_timings);
    m.cache_populated = m.weight_cache && m.weight_cache->stats().entries != 0;
}
}  // namespace kadan::image
