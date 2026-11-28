#include "kadan/image_aten.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <limits>
#include <map>
#include <mutex>
#include <optional>
#include <thread>
#include <vector>

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

struct Rotary {
    Tensor cosine, sine;
};

Rotary qwen3_rotary(int device, std::int64_t seq_len) {
    std::vector<float> cos_host(std::size_t(seq_len) * 64);
    std::vector<float> sin_host(std::size_t(seq_len) * 64);
    const double scale = -2.0 * std::log(1000000.0) / 128.0;
    std::array<double, 64> freqs{};
    for (int i = 0; i < 64; ++i)
        freqs[std::size_t(i)] = std::exp(double(i) * scale);
    for (std::int64_t s = 0; s < seq_len; ++s) {
        for (int i = 0; i < 64; ++i) {
            const double angle = double(s) * freqs[std::size_t(i)];
            const std::size_t idx = std::size_t(s) * 64 + std::size_t(i);
            cos_host[idx] = float(std::cos(angle));
            sin_host[idx] = float(std::sin(angle));
        }
    }
    return {
        Tensor::from_cpu_float(device, {seq_len, 64}, cos_host, gpu::Dtype::fp32),
        Tensor::from_cpu_float(device, {seq_len, 64}, sin_host, gpu::Dtype::fp32)
    };
}

HostTensor encode_text(
    Stage& stage,
    const ImagePrompt& prompt,
    const std::atomic_bool& cancel,
    const Generator::Hook& hook
) {
    constexpr std::int64_t seq_len = std::int64_t(image_prompt_length);
    std::vector<std::int64_t> ids(prompt.tokens.begin(), prompt.tokens.end());
    auto indices = Tensor::from_cpu_int64(stage.device, {seq_len}, ids);
    const auto embed_w = stage.weight("embed_tokens.weight");
    auto x = gpu::embedding_bf16_to_fp32(embed_w.get(), indices);
    const auto rotary = qwen3_rotary(stage.device, seq_len);
    const auto active = std::int64_t(prompt.active_count);

    // FLUX.2 Klein 4B stacks hidden_states[9], hidden_states[18], hidden_states[27]
    // (0-indexed: after layers 8, 17, 26) along the channel dimension -> 7680.
    std::array<Tensor, 3> tapped;
    for (int layer = 0; layer < 27; ++layer) {
        stop(cancel);
        auto p = "layers." + std::to_string(layer) + ".";
        auto normalized = stage.rms(p + "input_layernorm", x);
        auto q = gpu::text_rope_fp32(
            stage.rms(
                p + "self_attn.q_norm",
                gpu::transpose_to_heads(stage.linear(p + "self_attn.q_proj", normalized), 32, 128)
            ),
            rotary.cosine,
            rotary.sine
        );
        auto k = gpu::text_rope_fp32(
            stage.rms(
                p + "self_attn.k_norm",
                gpu::transpose_to_heads(stage.linear(p + "self_attn.k_proj", normalized), 8, 128)
            ),
            rotary.cosine,
            rotary.sine
        );
        auto v = gpu::transpose_to_heads(
            stage.linear(p + "self_attn.v_proj", normalized), 8, 128
        );
        auto mix = gpu::unheads(gpu::sdpa_qwen3_fp32(q, k, v, active));
        mix.reshape_inplace({seq_len, 4096});
        auto o_proj = stage.linear(p + "self_attn.o_proj", mix);
        gpu::add_inplace_fp32(x, o_proj);

        normalized = stage.rms(p + "post_attention_layernorm", x);
        auto gate = stage.linear(p + "mlp.gate_proj", normalized);
        auto up = stage.linear(p + "mlp.up_proj", normalized);
        auto mlp_mid = gpu::silu_mul_fp32(gate, up);
        auto mlp_out = stage.linear(p + "mlp.down_proj", mlp_mid);
        gpu::add_inplace_fp32(x, mlp_out);

        if (layer == 8)
            tapped[0] = x.clone();
        else if (layer == 17)
            tapped[1] = x.clone();
        else if (layer == 26)
            tapped[2] = x.clone();
        if (hook)
            hook("text_layer", layer);
    }
    stop(cancel);
    return cpu_float(gpu::concat_last_3_fp32(tapped[0], tapped[1], tapped[2]));
}

Rotary flux2_rotary(
    int device, const std::vector<std::array<std::int64_t, 4>>& coordinates
) {
    const std::size_t seq = coordinates.size();
    std::vector<float> cos_host(seq * 64);
    std::vector<float> sin_host(seq * 64);
    constexpr std::array<int, 4> widths{32, 32, 32, 32};
    std::array<double, 16> freqs{};
    for (int i = 0; i < 16; ++i)
        freqs[std::size_t(i)] =
            std::exp(double(i * 2) * (-std::log(2000.0) / double(widths[0])));
    for (std::size_t s = 0; s < seq; ++s) {
        for (int axis = 0; axis < 4; ++axis) {
            const double pos = double(coordinates[s][std::size_t(axis)]);
            for (int i = 0; i < 16; ++i) {
                const double angle = pos * freqs[std::size_t(i)];
                const std::size_t idx = s * 64 + std::size_t(axis * 16 + i);
                cos_host[idx] = float(std::cos(angle));
                sin_host[idx] = float(std::sin(angle));
            }
        }
    }
    return {
        Tensor::from_cpu_float(device, {std::int64_t(seq), 64}, cos_host, gpu::Dtype::fp32),
        Tensor::from_cpu_float(device, {std::int64_t(seq), 64}, sin_host, gpu::Dtype::fp32)
    };
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

HostTensor denoise_batch(
    Stage& stage,
    const HostTensor& latent_cpu,
    const HostTensor& source_cpu,
    const std::vector<HostTensor>& reference_latents_cpu,
    const std::vector<HostTensor>& conditions_cpu,
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
    auto latent = latent_cpu.to_device(stage.device, gpu::Dtype::fp32);
    auto times = config.sampling_schedule(std::size_t(target_tokens), request.steps);
    if (!request.source.empty() && !request.reference)
        times = edit_schedule(times, request.strength);

    Tensor source, noise, mask;
    if (source_cpu.defined() && !request.reference) {
        source = source_cpu.to_device(stage.device, gpu::Dtype::fp32);
        source.reshape_inplace({1, target_tokens, 128});
        noise = latent.clone();
        gpu::mix_source_noise_fp32(latent, source, noise, times.sigma[0]);
        if (!request.mask.empty()) {
            std::vector<float> mask_host(static_cast<std::size_t>(target_tokens), 0.0f);
            for (std::int64_t y = 0; y < h; ++y) {
                stop(cancel);
                for (std::int64_t x = 0; x < w; ++x) {
                    float sum = 0.0f;
                    for (int dy = 0; dy < 16; ++dy)
                        for (int dx = 0; dx < 16; ++dx)
                            sum += request.mask
                                       [std::size_t(y * 16 + dy) * request.width
                                        + std::size_t(x * 16 + dx)];
                    mask_host[std::size_t(y * w + x)] = sum / 256.0f;
                }
            }
            mask = Tensor::from_cpu_float(
                stage.device, {target_tokens, 1}, mask_host, gpu::Dtype::fp32
            );
        }
    }

    Tensor ref_latents;
    if (request.reference && !reference_latents_cpu.empty()) {
        std::int64_t total_ref = 0;
        std::vector<float> flat_refs;
        for (const auto& r : reference_latents_cpu) {
            total_ref += r.size(0);
            flat_refs.insert(flat_refs.end(), r.data.begin(), r.data.end());
        }
        auto single_ref = Tensor::from_cpu_float(
            stage.device, {total_ref, 128}, flat_refs, gpu::Dtype::bf16
        );
        ref_latents = gpu::broadcast_batch_bf16(single_ref, batch);
    }

    const auto grids = reference_grids(request);
    const auto layout =
        flux2_position_layout(std::int64_t(image_prompt_length), h, w, grids);
    const auto rotary = flux2_rotary(stage.device, layout.positions);
    const auto text_tokens = layout.text_tokens;

    std::vector<Tensor> projected_texts;
    projected_texts.reserve(requests.size());
    for (std::int64_t i = 0; i < batch; ++i) {
        auto cond = conditions_cpu[std::size_t(i)].to_device(
            stage.device, gpu::Dtype::bf16
        );
        projected_texts.push_back(stage.linear("context_embedder", cond));
    }
    auto base_txt = gpu::stack_batch_bf16(projected_texts);
    projected_texts.clear();

    for (std::size_t step = 0; step < times.steps; ++step) {
        stop(cancel);
        const float sigma = times.sigma[step];
        auto time_proj = gpu::flux2_timestep_proj_bf16(stage.device, sigma);
        auto temb = stage.linear(
            "time_guidance_embed.timestep_embedder.linear_2",
            gpu::silu_bf16(stage.linear(
                "time_guidance_embed.timestep_embedder.linear_1", time_proj
            ))
        );
        auto activated_temb = gpu::silu_bf16(temb);
        auto mod_img = gpu::split_double_modulation(
            stage.linear("double_stream_modulation_img.linear", activated_temb)
        );
        auto mod_txt = gpu::split_double_modulation(
            stage.linear("double_stream_modulation_txt.linear", activated_temb)
        );
        auto mod_single = gpu::split_single_modulation(
            stage.linear("single_stream_modulation.linear", activated_temb)
        );

        auto latent_bf16 = latent.to_dtype(gpu::Dtype::bf16);
        auto img_input = ref_latents.defined()
                             ? gpu::concat_axis1_bf16(latent_bf16, ref_latents)
                             : std::move(latent_bf16);
        auto hidden_states = stage.linear("x_embedder", img_input);
        auto encoder_hidden_states = base_txt.clone();

        if (hook)
            hook("conditioning", step);

        for (int layer = 0; layer < 5; ++layer) {
            stop(cancel);
            if (hook)
                hook("block_started", layer);
            auto p = "transformer_blocks." + std::to_string(layer) + ".";
            auto norm_img = gpu::flux2_modulated_norm_bf16(
                hidden_states, mod_img.attention.shift, mod_img.attention.scale
            );
            auto norm_txt = gpu::flux2_modulated_norm_bf16(
                encoder_hidden_states, mod_txt.attention.shift, mod_txt.attention.scale
            );

            auto img_q = stage.rms(
                p + "attn.norm_q",
                gpu::transpose_to_heads(stage.linear(p + "attn.to_q", norm_img), 24, 128)
            );
            auto img_k = stage.rms(
                p + "attn.norm_k",
                gpu::transpose_to_heads(stage.linear(p + "attn.to_k", norm_img), 24, 128)
            );
            auto img_v =
                gpu::transpose_to_heads(stage.linear(p + "attn.to_v", norm_img), 24, 128);

            auto txt_q = stage.rms(
                p + "attn.norm_added_q",
                gpu::transpose_to_heads(stage.linear(p + "attn.add_q_proj", norm_txt), 24, 128)
            );
            auto txt_k = stage.rms(
                p + "attn.norm_added_k",
                gpu::transpose_to_heads(stage.linear(p + "attn.add_k_proj", norm_txt), 24, 128)
            );
            auto txt_v = gpu::transpose_to_heads(
                stage.linear(p + "attn.add_v_proj", norm_txt), 24, 128
            );

            auto q = gpu::flux2_rope_bf16(
                gpu::concat_axis2_bf16(txt_q, img_q), rotary.cosine, rotary.sine
            );
            auto k = gpu::flux2_rope_bf16(
                gpu::concat_axis2_bf16(txt_k, img_k), rotary.cosine, rotary.sine
            );
            auto v = gpu::concat_axis2_bf16(txt_v, img_v);
            auto mix = gpu::unheads(gpu::sdpa_flux2_bf16(q, k, v));

            auto txt_attn = stage.linear(
                p + "attn.to_add_out", gpu::slice_axis1_bf16(mix, 0, text_tokens)
            );
            auto img_attn = stage.linear(
                p + "attn.to_out.0",
                gpu::slice_axis1_bf16(mix, text_tokens, mix.size(1) - text_tokens)
            );

            gpu::flux2_gated_residual_bf16(
                hidden_states, img_attn, mod_img.attention.gate
            );
            norm_img = gpu::flux2_modulated_norm_bf16(
                hidden_states, mod_img.mlp.shift, mod_img.mlp.scale
            );
            auto img_ff = stage.linear(
                p + "ff.linear_out",
                gpu::flux2_swiglu_bf16(stage.linear(p + "ff.linear_in", norm_img))
            );
            gpu::flux2_gated_residual_bf16(hidden_states, img_ff, mod_img.mlp.gate);

            gpu::flux2_gated_residual_bf16(
                encoder_hidden_states, txt_attn, mod_txt.attention.gate
            );
            norm_txt = gpu::flux2_modulated_norm_bf16(
                encoder_hidden_states, mod_txt.mlp.shift, mod_txt.mlp.scale
            );
            auto txt_ff = stage.linear(
                p + "ff_context.linear_out",
                gpu::flux2_swiglu_bf16(stage.linear(p + "ff_context.linear_in", norm_txt))
            );
            gpu::flux2_gated_residual_bf16(
                encoder_hidden_states, txt_ff, mod_txt.mlp.gate
            );

            if (hook)
                hook("block", layer);
        }

        auto x = gpu::concat_axis1_bf16(encoder_hidden_states, hidden_states);
        encoder_hidden_states.reset();
        hidden_states.reset();

        for (int layer = 0; layer < 20; ++layer) {
            stop(cancel);
            if (hook)
                hook("block_started", 5 + layer);
            auto p = "single_transformer_blocks." + std::to_string(layer) + ".attn.";
            auto n = gpu::flux2_modulated_norm_bf16(
                x, mod_single.shift, mod_single.scale
            );
            auto proj = stage.linear(p + "to_qkv_mlp_proj", n);
            auto q = gpu::flux2_rope_bf16(
                stage.rms(
                    p + "norm_q",
                    gpu::transpose_to_heads(gpu::slice_last_bf16(proj, 0, 3072), 24, 128)
                ),
                rotary.cosine,
                rotary.sine
            );
            auto k = gpu::flux2_rope_bf16(
                stage.rms(
                    p + "norm_k",
                    gpu::transpose_to_heads(gpu::slice_last_bf16(proj, 3072, 3072), 24, 128)
                ),
                rotary.cosine,
                rotary.sine
            );
            auto v = gpu::transpose_to_heads(gpu::slice_last_bf16(proj, 6144, 3072), 24, 128);
            auto mlp_in = gpu::slice_last_bf16(proj, 9216, proj.size(-1) - 9216);
            proj.reset();

            auto attn_out = gpu::unheads(gpu::sdpa_flux2_bf16(q, k, v));
            q.reset();
            k.reset();
            v.reset();
            auto mlp_out = gpu::flux2_swiglu_bf16(mlp_in);
            mlp_in.reset();
            auto out =
                stage.linear(p + "to_out", gpu::concat_last_bf16(attn_out, mlp_out));
            gpu::flux2_gated_residual_bf16(x, out, mod_single.gate);
            if (hook)
                hook("block", 5 + layer);
        }

        auto target_x = gpu::slice_axis1_bf16(x, text_tokens, target_tokens);
        x.reset();
        auto [out_scale, out_shift] =
            gpu::split_out_modulation(stage.linear("norm_out.linear", activated_temb));
        auto normalized =
            gpu::flux2_modulated_norm_bf16(target_x, out_shift, out_scale);
        auto velocity = stage.linear("proj_out", normalized).to_dtype(gpu::Dtype::fp32);
        gpu::euler_step_fp32(
            latent, velocity, times.sigma[step + 1] - times.sigma[step]
        );
        if (mask.defined()) {
            gpu::apply_latent_mask_fp32(
                latent, source, noise, mask, times.sigma[step + 1]
            );
        }
        if (hook)
            hook("step_completed", step + 1);
    }
    stop(cancel);
    return cpu_float(latent);
}

Tensor vae_group_norm(
    Stage& stage, const std::string& p, const Tensor& x, bool activate
) {
    const auto w = stage.weight(p + ".weight");
    const auto b = stage.weight(p + ".bias");
    return gpu::group_norm_32_fp32(x, w.get(), b.get(), activate, 1e-6f);
}

Tensor vae_conv(
    Stage& stage, const std::string& p, const Tensor& x, bool downsample = false
) {
    const auto w = stage.weight(p + ".weight");
    const auto b = stage.weight(p + ".bias");
    return gpu::conv2d_fp32(x, w.get(), b.get(), downsample);
}

Tensor vae_residual(
    Stage& stage, const std::string& p, const Tensor& x, const std::atomic_bool& cancel
) {
    stop(cancel);
    auto y = vae_conv(stage, p + ".conv1", vae_group_norm(stage, p + ".norm1", x, true));
    stop(cancel);
    y = vae_conv(stage, p + ".conv2", vae_group_norm(stage, p + ".norm2", y, true));
    if (x.size(1) == y.size(1)) {
        gpu::add_inplace_fp32(y, x);
    } else {
        auto shortcut = vae_conv(stage, p + ".conv_shortcut", x);
        gpu::add_inplace_fp32(y, shortcut);
    }
    return y;
}

Tensor vae_attention(
    Stage& stage, const std::string& p, const Tensor& x, const std::atomic_bool& cancel
) {
    auto h = vae_group_norm(stage, p + ".group_norm", x, false);
    const auto c = x.size(1);
    const auto gh = x.size(2);
    const auto gw = x.size(3);
    auto seq = gpu::nchw_to_seq_fp32(h);
    h.reset();
    auto q = stage.linear_bias(p + ".to_q", seq);
    auto k = stage.linear_bias(p + ".to_k", seq);
    auto v = stage.linear_bias(p + ".to_v", seq);
    seq.reset();
    auto mix = gpu::vae_sdpa_fp32(q, k, v, cancel);
    q.reset();
    k.reset();
    v.reset();
    auto out = gpu::seq_to_nchw_fp32(stage.linear_bias(p + ".to_out.0", mix), c, gh, gw);
    gpu::add_inplace_fp32(out, x);
    return out;
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

HostTensor encode_pixels(
    Stage& stage,
    std::span<const float> rgba_pixels,
    std::size_t width,
    std::size_t height,
    const ImageCheckpointConfig& config,
    const std::atomic_bool& cancel,
    const Generator::Hook& hook
) {
    // rgba_pixels is planar [1, 4, H, W]; first 3*H*W floats are RGB.
    auto x = Tensor::from_cpu_float(
        stage.device,
        {1, 3, std::int64_t(height), std::int64_t(width)},
        rgba_pixels.first(3 * height * width),
        gpu::Dtype::fp32
    );
    const auto h = std::int64_t(height / 16), w = std::int64_t(width / 16);
    if (stage.tiled_vae()) {
        Tensor encoded(stage.device, gpu::Dtype::fp32, {1, 512, h * 2, w * 2});
        for (std::int64_t y = 0; y < h; y += vae_tile_core) {
            const auto rows = vae_tile_axis(y, h, vae_encode_halo);
            for (std::int64_t column = 0; column < w; column += vae_tile_core) {
                stop(cancel);
                const auto cols = vae_tile_axis(column, w, vae_encode_halo);
                auto tile_in = gpu::crop_2d_fp32(
                    x,
                    rows.input_begin * 16,
                    (rows.input_end - rows.input_begin) * 16,
                    cols.input_begin * 16,
                    (cols.input_end - cols.input_begin) * 16
                );
                auto tile = encode_vae_blocks(stage, std::move(tile_in), cancel, hook);
                gpu::paste_2d_fp32(
                    encoded,
                    tile,
                    rows.begin * 2,
                    cols.begin * 2,
                    rows.crop_begin() * 2,
                    cols.crop_begin() * 2,
                    (rows.end - rows.begin) * 2,
                    (cols.end - cols.begin) * 2
                );
            }
        }
        x = std::move(encoded);
    } else {
        x = encode_vae_blocks(stage, std::move(x), cancel, hook);
    }
    x = vae_residual(stage, "encoder.mid_block.resnets.0", x, cancel);
    x = vae_attention(stage, "encoder.mid_block.attentions.0", x, cancel);
    x = vae_residual(stage, "encoder.mid_block.resnets.1", x, cancel);
    x = vae_conv(
        stage, "encoder.conv_out", vae_group_norm(stage, "encoder.conv_norm_out", x, true)
    );
    x = gpu::slice_channel_fp32(vae_conv(stage, "quant_conv", x), 0, 32);
    const auto bn_mean = stage.weight("bn.running_mean");
    const auto bn_var = stage.weight("bn.running_var");
    auto normalized = gpu::vae_patchify_bn_norm_fp32(
        x, bn_mean.get(), bn_var.get(), config.batch_norm_eps
    );
    stop(cancel);
    return cpu_float(normalized);
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
            x = gpu::upsample_nearest2d_fp32(x);
            x = vae_conv(stage, p + ".upsamplers.0.conv", x);
        }
        if (hook)
            hook("vae_up_block", i);
    }
    x = vae_conv(
        stage, "decoder.conv_out", vae_group_norm(stage, "decoder.conv_norm_out", x, true)
    );
    stop(cancel);
    return gpu::vae_clamp_rgba_fp32(x);
}

HostTensor decode(
    Stage& stage,
    const HostTensor& latent,
    std::int64_t h,
    std::int64_t w,
    const ImageCheckpointConfig& config,
    const std::atomic_bool& cancel,
    const Generator::Hook& hook
) {
    const auto bn_mean = stage.weight("bn.running_mean");
    const auto bn_var = stage.weight("bn.running_var");
    auto lat_gpu = latent.to_device(stage.device, gpu::Dtype::fp32);
    auto x = gpu::vae_bn_denorm_unpatchify_fp32(
        lat_gpu, h, w, bn_mean.get(), bn_var.get(), config.batch_norm_eps
    );
    lat_gpu.reset();
    x = vae_conv(stage, "post_quant_conv", x);
    x = vae_conv(stage, "decoder.conv_in", x);
    x = vae_residual(stage, "decoder.mid_block.resnets.0", x, cancel);
    x = vae_attention(stage, "decoder.mid_block.attentions.0", x, cancel);
    x = vae_residual(stage, "decoder.mid_block.resnets.1", x, cancel);
    if (!stage.tiled_vae())
        return cpu_float(decode_vae_blocks(stage, std::move(x), cancel, hook));

    const std::int64_t full_h = h * 16;
    const std::int64_t full_w = w * 16;
    HostTensor output{{1, 4, full_h, full_w}, std::vector<float>(std::size_t(4 * full_h * full_w))};
    for (std::int64_t y = 0; y < h; y += vae_tile_core) {
        const auto rows = vae_tile_axis(y, h, vae_decode_halo);
        for (std::int64_t column = 0; column < w; column += vae_tile_core) {
            stop(cancel);
            const auto cols = vae_tile_axis(column, w, vae_decode_halo);
            auto tile_in = gpu::crop_2d_fp32(
                x,
                rows.input_begin * 2,
                (rows.input_end - rows.input_begin) * 2,
                cols.input_begin * 2,
                (cols.input_end - cols.input_begin) * 2
            );
            auto tile_gpu = decode_vae_blocks(stage, std::move(tile_in), cancel, hook);
            auto tile_crop = gpu::crop_2d_fp32(
                tile_gpu,
                rows.crop_begin() * 16,
                (rows.end - rows.begin) * 16,
                cols.crop_begin() * 16,
                (cols.end - cols.begin) * 16
            );
            const auto tile_host = cpu_float(tile_crop);
            const std::size_t tile_h = std::size_t((rows.end - rows.begin) * 16);
            const std::size_t tile_w = std::size_t((cols.end - cols.begin) * 16);
            const std::size_t dst_y0 = std::size_t(rows.begin * 16);
            const std::size_t dst_x0 = std::size_t(cols.begin * 16);
            for (std::size_t c = 0; c < 4; ++c) {
                for (std::size_t dy = 0; dy < tile_h; ++dy) {
                    const float* src_row =
                        &tile_host.data[(c * tile_h + dy) * tile_w];
                    float* dst_row =
                        &output.data[(c * std::size_t(full_h) + (dst_y0 + dy)) * std::size_t(full_w) + dst_x0];
                    std::copy_n(src_row, tile_w, dst_row);
                }
            }
        }
    }
    return output;
}

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
    for (int device : devices_)
        gpu::destroy_device_handles(device);
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
    for (const auto device : devices_) {
        gpu::init_device(device);
        const auto stats = gpu::device_stats(device);
        measurement_.baseline_allocated[device + 1] = stats.allocated;
        measurement_.baseline_reserved[device + 1] = stats.allocated;
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

    HostTensor source;
    std::vector<HostTensor> reference_latents;
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
        gpu::DeviceGuard guard(stage.device);
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

    std::vector<HostTensor> conditions;
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
        gpu::DeviceGuard guard(stage.device);
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

    std::vector<HostTensor> seeded_latents(requests.size());
    for (std::size_t i = 0; i < requests.size(); ++i) {
        seeded_latents[i].shape = {std::int64_t(points), 128};
        seeded_latents[i].data.resize(points * 128);
        seed_image_latent(seeded_latents[i].data, requests[i].seed, cancel);
    }

    std::vector<HostTensor> denoised_latents(requests.size());
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
        std::vector<HostTensor> results(lanes.size());
        std::vector<std::exception_ptr> failures(lanes.size());
        const auto run_lane = [&](std::size_t index) {
            const auto& lane = lanes[index];
            auto& stage = *stages[index];
            const auto started = Clock::now();
            bool completed = false;
            try {
                gpu::DeviceGuard guard(stage.device);
                HostTensor lane_latent;
                lane_latent.shape = {std::int64_t(lane.images), std::int64_t(points), 128};
                lane_latent.data.reserve(lane.images * points * 128);
                for (std::size_t i = lane.begin; i < lane.begin + lane.images; ++i)
                    lane_latent.data.insert(
                        lane_latent.data.end(),
                        seeded_latents[i].data.begin(),
                        seeded_latents[i].data.end()
                    );
                const std::vector<HostTensor> lane_conditions(
                    conditions.begin() + std::ptrdiff_t(lane.begin),
                    conditions.begin() + std::ptrdiff_t(lane.begin + lane.images)
                );
                results[index] = denoise_batch(
                    stage,
                    lane_latent,
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
        const std::size_t per_img = points * 128;
        for (std::size_t lane_idx = 0; lane_idx < lanes.size(); ++lane_idx) {
            const auto& lane = lanes[lane_idx];
            const auto& batch_res = results[lane_idx];
            for (std::size_t local_i = 0; local_i < lane.images; ++local_i) {
                auto& dst = denoised_latents[lane.begin + local_i];
                dst.shape = {std::int64_t(points), 128};
                const auto* src_ptr = batch_res.data.data() + local_i * per_img;
                dst.data.assign(src_ptr, src_ptr + per_img);
            }
        }
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
        gpu::DeviceGuard guard(stage.device);
        started = Clock::now();
        for (std::size_t i = 0; i < requests.size(); ++i) {
            auto decoded = decode(
                stage,
                denoised_latents[i],
                std::int64_t(h),
                std::int64_t(w),
                m.config,
                cancel,
                indexed_hook(i)
            );
            stage.sync();
            std::copy_n(decoded.data.data(), outputs[i].size(), outputs[i].data());
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
