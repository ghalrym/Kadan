// SPDX-License-Identifier: Apache-2.0
// Portions adapted from Transformers:
// Copyright 2025 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
// Native C++ adaptation of the original Fish Audio Drama 3 reference graph.
// Graph/key contracts: SGLang-Omni fishaudio_drama3/{audio_decoder,decoder_layers,
// text_config,sampling}.py at be34dd84ed9e0971f72fe5e3d9ba8245cdad780a (Apache-2.0),
// and Transformers 5.19.0 models/qwen3_5_moe/modeling_qwen3_5_moe.py
// (Apache-2.0). Recurrent updates, sampling and tower ordering adapt those
// reference algorithms; storage, admission and projection plumbing are native.
// Source and license references:
// https://github.com/sgl-project/sglang-omni/tree/be34dd84ed9e0971f72fe5e3d9ba8245cdad780a/sglang_omni/models/fishaudio_drama3
// https://github.com/sgl-project/sglang-omni/blob/be34dd84ed9e0971f72fe5e3d9ba8245cdad780a/LICENSE
// https://github.com/huggingface/transformers/blob/v5.19.0/src/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py
// https://github.com/huggingface/transformers/blob/v5.19.0/LICENSE
// This implementation uses Hikari's synchronous bounded DenseCompute; no serving
// runtime is loaded.
#include "kadan/drama_model.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <numeric>
#include <random>
#include <stdexcept>

#include <nlohmann/json.hpp>

#include "drama_weights.hpp"
#include "kadan/layer_math.hpp"

namespace kadan::drama {
namespace {
constexpr std::size_t hidden = 2048, head_dim = 256, heads = 16, kv_heads = 2,
                      query_width = heads * head_dim, kv_width = kv_heads * head_dim;
constexpr float epsilon = 1e-6f;
using Frame = std::array<std::uint32_t, 10>;
using Json = nlohmann::json;

void equal(const Json& object, const char* key, const Json& value) {
    require(
        object.contains(key) && object.at(key) == value,
        "drama_configuration_unsupported"
    );
}

float sigmoid(float value) {
    return 1.f / (1.f + std::exp(-value));
}

void add(Vector& destination, std::span<const float> source, float scale = 1) {
    require(destination.size() == source.size(), "drama_residual_shape");
    for (std::size_t i = 0; i < source.size(); ++i)
        destination[i] += scale * source[i];
}

void normalize(Vector& x, std::span<const float> weights, bool one_plus = true) {
    require(!weights.empty() && x.size() % weights.size() == 0, "drama_norm_shape");
    math::rms_norm(
        {x.size() / weights.size(),
         weights.size(),
         epsilon,
         one_plus ? math::NormScale::one_plus : math::NormScale::direct},
        x,
        weights,
        x
    );
}

// Backbone RoPE is split-half; the codebook tower uses adjacent pairs. Only the
// first 64 coordinates rotate, with the checkpoint's common 10,000,000 base.
void rotary(Vector& values, std::size_t position, bool interleaved) {
    static const auto frequencies = [] {
        std::array<float, 32> result{};
        math::inverse_frequencies(1e7, result);
        return result;
    }();
    if (!interleaved) {
        math::rope(
            {values.size() / head_dim, head_dim, 64, 4096},
            position,
            frequencies,
            values,
            values
        );
        return;
    }
    // The shared layer_math RoPE deliberately accepts split-half layout only.
    // Tower checkpoints pair adjacent coordinates, so retain that distinct map.
    for (std::size_t base = 0; base < values.size(); base += head_dim)
        for (std::size_t i = 0; i < frequencies.size(); ++i) {
            const float angle = float(position) * frequencies[i];
            const float cosine = std::cos(angle), sine = std::sin(angle);
            const auto first = base + 2 * i, second = first + 1;
            const float a = values[first], b = values[second];
            values[first] = a * cosine - b * sine;
            values[second] = a * sine + b * cosine;
        }
}

struct AttentionState {
    Vector keys, values;

    explicit AttentionState(std::size_t context)
        : keys(context * kv_width), values(keys.size()) {
    }
};

struct DeltaState {
    Vector convolution = Vector(3 * 8192);
    // HF ordering [value head, key coordinate, value coordinate].
    Vector recurrent = Vector(32 * 128 * 128);
};

std::uint32_t sample(
    std::span<const float> logits, const Sampling& options, std::mt19937_64& random
) {
    std::vector<std::size_t> order(logits.size());
    std::iota(order.begin(), order.end(), 0);
    for (float v : logits)
        require(std::isfinite(v), "drama_sampling_nonfinite");
    std::stable_sort(order.begin(), order.end(), [&](auto a, auto b) {
        return logits[a] > logits[b];
    });
    const auto inverse_temperature = 1.f / std::max(options.temperature, 1e-5f);
    std::vector<double> probability(logits.size());
    double sum = 0;
    for (std::size_t i = 0; i < order.size(); ++i) {
        probability[i] = std::exp(
            double((logits[order[i]] - logits[order[0]]) * inverse_temperature)
        );
        sum += probability[i];
    }
    double cumulative = 0;
    for (std::size_t i = 0; i < order.size(); ++i) {
        cumulative += probability[i] / sum;
        // Upstream uses inclusive cumulative mass, always preserving rank zero.
        if (i && (i >= options.top_k || cumulative > options.top_p))
            probability[i] = 0;
    }
    std::discrete_distribution<std::size_t> choose(
        probability.begin(), probability.end()
    );
    return static_cast<std::uint32_t>(order[choose(random)]);
}
}  // namespace

ModelInfo inspect(const std::filesystem::path& directory, std::size_t context) {
    require(context > 0 && context <= 4096, "drama_context_limit");
    const auto path = directory / "config.json";
    require(
        std::filesystem::is_regular_file(path)
            && std::filesystem::file_size(path) <= 65536,
        "drama_config_missing_or_too_large"
    );
    std::ifstream input(path);
    require(bool(input), "drama_config_read");
    const auto config = Json::parse(input);
    equal(config, "model_type", "fish_omni");
    equal(config, "fish_schema_version", 1);
    equal(config, "speech_template_version", "reference-bracket-v1");
    equal(config, "semantic_start_token_id", 248099);
    equal(config, "semantic_end_token_id", 252194);
    equal(config, "eos_token_id", 248046);
    require(
        config.value("audio_encoder_config", Json()) == nullptr,
        "drama_audio_encoder_unsupported"
    );
    require(
        !config.contains("extra_samplable_token_ids")
            || config["extra_samplable_token_ids"].empty(),
        "drama_extra_token_unsupported"
    );
    const auto& text = config.at("text_config");
    equal(text, "model_type", "fish_omni_text");
    for (const auto& [key, value] : std::initializer_list<std::pair<const char*, int>>{
             {"hidden_size", 2048},
             {"num_hidden_layers", 40},
             {"num_attention_heads", 16},
             {"num_key_value_heads", 2},
             {"head_dim", 256},
             {"vocab_size", 252288},
             {"linear_conv_kernel_dim", 4},
             {"linear_key_head_dim", 128},
             {"linear_value_head_dim", 128},
             {"linear_num_key_heads", 16},
             {"linear_num_value_heads", 32},
             {"num_experts", 256},
             {"num_experts_per_tok", 8},
             {"moe_intermediate_size", 512},
             {"shared_expert_intermediate_size", 512}
         })
        equal(text, key, value);
    equal(text, "rms_norm_eps", 1e-6);
    equal(text, "hidden_act", "silu");
    equal(text, "tie_word_embeddings", false);
    equal(text, "rotary_interleaved", false);
    equal(text, "use_qk_norm", true);
    equal(text, "attn_output_gate", true);
    equal(text, "layernorm_zero_centered_gamma", true);
    equal(text, "shared_expert_gate", true);
    equal(text, "attention_bias", false);
    require(
        text.value("attention_dropout", 0.0) == 0.0
            && text.value("norm_topk_prob", true),
        "drama_backbone_options_unsupported"
    );
    require(
        text.at("max_position_embeddings").get<std::size_t>() >= context,
        "drama_model_context_limit"
    );
    equal(text.at("rope_parameters"), "rope_type", "default");
    equal(text.at("rope_parameters"), "rope_theta", 1e7);
    equal(text.at("rope_parameters"), "partial_rotary_factor", 0.25);
    require(
        !text.at("rope_parameters").contains("mrope_section"), "drama_mrope_unsupported"
    );
    require(
        text.at("layer_types").size() == 40 && text.at("mlp_layer_types").size() == 40,
        "drama_layer_count"
    );
    for (std::size_t i = 0; i < 40; ++i) {
        require(
            text.at("layer_types")[i]
                == ((i + 1) % 4 ? "linear_attention" : "full_attention"),
            "drama_layer_schedule"
        );
        require(text.at("mlp_layer_types")[i] == "moe", "drama_mlp_schedule");
    }
    const auto& decoder = config.at("audio_decoder_config");
    equal(decoder, "model_type", "fish_qwen3_audio_decoder");
    equal(decoder, "decoder_type", "codebook_mtp_shared");
    for (const auto& [key, value] : std::initializer_list<std::pair<const char*, int>>{
             {"dim", 2048},
             {"text_dim", 2048},
             {"n_head", 16},
             {"n_local_heads", 2},
             {"head_dim", 256},
             {"n_layer", 4},
             {"shared_enrichment_layers", 4},
             {"intermediate_size", 6144},
             {"num_codebooks", 10},
             {"vocab_size", 4096},
             {"max_seq_len", 11},
             {"vq_backbone_codebooks", 1}
         })
        equal(decoder, key, value);
    equal(decoder, "norm_eps", 1e-6);
    equal(decoder, "rope_base", 1e7);
    equal(decoder, "rotary_percent", 0.25);
    equal(decoder, "layernorm_zero_centered_gamma", true);
    equal(decoder, "attention_output_gate", true);
    equal(decoder, "vq_feature_scope", "all_vq");
    equal(decoder, "vq_feature_placement", "backbone_partial");
    equal(decoder.at("mtp_spec"), "attention_variant", "full_attention");
    for (const auto* key :
         {"use_moe", "mtp_use_swa", "attention_qkv_bias", "attention_o_bias"})
        require(!decoder.value(key, false), "drama_decoder_options_unsupported");
    require(decoder.value("dropout", 0.0) == 0.0, "drama_decoder_dropout_unsupported");
    for (const auto* key :
         {"window_size", "window_attn_skip_freq", "window_rotary_base"})
        require(
            decoder.value(key, Json()) == nullptr, "drama_decoder_window_unsupported"
        );
    ModelInfo info;
    info.context_limit = context;
    if (config.contains("quantization_config")
        && !config["quantization_config"].is_null()) {
        const auto& quantization = config["quantization_config"];
        const auto method = quantization.value("quant_method", std::string());
        require(method != "drama3-int4", "drama_int4_gptq_marlin_unsupported");
        require(method == "bsgemm-moe", "drama_quantization_unsupported");
        equal(quantization, "weight_block_size", Json::array({128, 128}));
        equal(quantization, "activation_scheme", "dynamic");
        info.fp8_experts = true;
    }
    // Metadata parser quota (128 MiB), bounded tensor-name/index tables, stdlib
    // allocation overhead and small live vectors are covered by 256 MiB. The
    // exact recurrent/conv and full-attention state is admitted separately.
    info.host_bytes = (256ULL << 20)
                      + 30ULL * (3 * 8192 + 32 * 128 * 128) * sizeof(float)
                      + 23ULL * 2 * context * kv_width * sizeof(float);
    return info;
}

struct Generator::Impl {
    std::shared_ptr<Resources> resources;
    std::shared_ptr<DenseCompute> compute;
    ModelInfo metadata;
    std::unique_ptr<Weights> weights;
    std::vector<AttentionState> attention;
    std::vector<DeltaState> delta;
    Handle handle = 0;
    bool resident = false;

    Impl(
        std::shared_ptr<Resources> r,
        std::shared_ptr<DenseCompute> c,
        std::size_t context
    )
        : resources(std::move(r)), compute(std::move(c)) {
        require(bool(resources) && bool(compute), "drama_worker_compute_required");
        require(context > 0 && context <= 4096, "drama_context_limit");
        metadata.context_limit = context;
    }

    ~Impl() {
        try {
            if (weights)
                weights->release(*compute);
            weights.reset();
            attention.clear();
            delta.clear();
            if (handle) {
                if (resident)
                    resources->begin_eviction(handle);
                resources->released(handle);
            }
        } catch (...) {
            // Preserve the reservation if physical cleanup failed. The worker's
            // owning failure path must recover its compute context/ledger.
        }
    }

    Vector norm(
        Vector x,
        const std::string& name,
        const std::atomic_bool& cancel,
        bool one_plus = true
    ) {
        normalize(x, weights->read(name, cancel), one_plus);
        return x;
    }

    Vector project(
        const std::string& name,
        std::span<const float> x,
        const std::atomic_bool& cancel
    ) {
        return weights->project(*compute, name, x, cancel);
    }

    Vector attend(
        Vector q,
        Vector k,
        const Vector& v,
        const Vector& gate,
        AttentionState& state,
        std::size_t position,
        bool tower,
        const std::string& prefix,
        const std::atomic_bool& cancel
    ) {
        normalize(q, weights->read(prefix + "q_norm.weight", cancel));
        normalize(k, weights->read(prefix + "k_norm.weight", cancel));
        rotary(q, position, tower);
        rotary(k, position, tower);
        std::copy(k.begin(), k.end(), state.keys.begin() + position * kv_width);
        std::copy(v.begin(), v.end(), state.values.begin() + position * kv_width);
        Vector out(query_width);
        const auto key_view =
            std::span<const float>(state.keys).first((position + 1) * kv_width);
        const auto value_view =
            std::span<const float>(state.values).first(key_view.size());
        if (!compute->attend(
                q,
                key_view,
                value_view,
                1,
                position + 1,
                heads,
                kv_heads,
                head_dim,
                1,
                0,
                position,
                out,
                cancel
            )) {
            Vector scores(position + 1);
            for (std::size_t h = 0; h < heads; ++h) {
                check_cancel(cancel);
                const auto kh = h / (heads / kv_heads);
                for (std::size_t t = 0; t <= position; ++t) {
                    double sum = 0;
                    for (std::size_t d = 0; d < head_dim; ++d)
                        sum += double(q[h * head_dim + d])
                               * state.keys[t * kv_width + kh * head_dim + d];
                    scores[t] = static_cast<float>(sum / 16.0);
                }
                const auto maximum = *std::max_element(scores.begin(), scores.end());
                double denominator = 0;
                for (auto& score : scores) {
                    score = std::exp(score - maximum);
                    denominator += score;
                }
                for (std::size_t d = 0; d < head_dim; ++d) {
                    double value = 0;
                    for (std::size_t t = 0; t <= position; ++t)
                        value += double(scores[t])
                                 * state.values[t * kv_width + kh * head_dim + d];
                    out[h * head_dim + d] = static_cast<float>(value / denominator);
                }
            }
        }
        math::gate(math::Gate::sigmoid, out, gate, out);
        return out;
    }

    Vector full_attention(
        const Vector& x,
        std::size_t layer,
        std::size_t position,
        const std::atomic_bool& cancel
    ) {
        const auto p =
            "model.language_model.layers." + std::to_string(layer) + ".self_attn.";
        const auto combined = project(p + "q_proj.weight", x, cancel);
        Vector q(query_width), gate(query_width);
        for (std::size_t h = 0; h < heads; ++h) {
            std::copy_n(
                combined.begin() + h * 2 * head_dim, head_dim, q.begin() + h * head_dim
            );
            std::copy_n(
                combined.begin() + h * 2 * head_dim + head_dim,
                head_dim,
                gate.begin() + h * head_dim
            );
        }
        auto k = project(p + "k_proj.weight", x, cancel);
        const auto v = project(p + "v_proj.weight", x, cancel);
        const auto out = attend(
            std::move(q),
            std::move(k),
            v,
            gate,
            attention[layer / 4],
            position,
            false,
            p,
            cancel
        );
        return project(p + "o_proj.weight", out, cancel);
    }

    Vector linear_attention(
        const Vector& x, std::size_t layer, const std::atomic_bool& cancel
    ) {
        const auto p =
            "model.language_model.layers." + std::to_string(layer) + ".linear_attn.";
        auto& state = delta[layer - layer / 4];
        const auto qkv = project(p + "in_proj_qkv.weight", x, cancel);
        const auto z = project(p + "in_proj_z.weight", x, cancel);
        const auto a = project(p + "in_proj_a.weight", x, cancel);
        const auto b = project(p + "in_proj_b.weight", x, cancel);
        const auto kernel = weights->read(p + "conv1d.weight", cancel);
        const auto dt = weights->read(p + "dt_bias", cancel);
        const auto a_log = weights->read(p + "A_log", cancel);
        const auto scale = weights->read(p + "norm.weight", cancel);
        Vector mixed(8192), out(4096);
        for (std::size_t c = 0; c < mixed.size(); ++c) {
            float sum = qkv[c] * kernel[c * 4 + 3];
            for (std::size_t t = 0; t < 3; ++t)
                sum += state.convolution[t * 8192 + c] * kernel[c * 4 + t];
            mixed[c] = sum * sigmoid(sum);
        }
        std::move(
            state.convolution.begin() + 8192,
            state.convolution.end(),
            state.convolution.begin()
        );
        std::copy(qkv.begin(), qkv.end(), state.convolution.end() - 8192);
        for (std::size_t h = 0; h < 32; ++h) {
            check_cancel(cancel);
            // HF repeat_interleave, unlike the tiled GGUF conversion layout.
            const auto kh = h / 2;
            const auto q_start = kh * 128, k_start = 2048 + kh * 128;
            double q_sq = epsilon, k_sq = epsilon;
            for (std::size_t d = 0; d < 128; ++d) {
                q_sq += double(mixed[q_start + d]) * mixed[q_start + d];
                k_sq += double(mixed[k_start + d]) * mixed[k_start + d];
            }
            const float q_scale = 1.f / std::sqrt(float(q_sq) * 128.f);
            const float k_scale = 1.f / std::sqrt(float(k_sq));
            const float alpha = a[h] + dt[h];
            const float softplus = alpha > 20 ? alpha : std::log1p(std::exp(alpha));
            const float decay = std::exp(-std::exp(a_log[h]) * softplus);
            const float beta = sigmoid(b[h]);
            auto matrix = std::span(state.recurrent).subspan(h * 128 * 128, 128 * 128);
            for (auto& cell : matrix)
                cell *= decay;
            std::array<float, 128> update{};
            for (std::size_t v = 0; v < 128; ++v) {
                double predicted = 0;
                for (std::size_t k = 0; k < 128; ++k)
                    predicted +=
                        double(matrix[k * 128 + v]) * mixed[k_start + k] * k_scale;
                update[v] = (mixed[4096 + h * 128 + v] - float(predicted)) * beta;
            }
            for (std::size_t k = 0; k < 128; ++k)
                for (std::size_t v = 0; v < 128; ++v)
                    matrix[k * 128 + v] += mixed[k_start + k] * k_scale * update[v];
            for (std::size_t v = 0; v < 128; ++v) {
                double value = 0;
                for (std::size_t k = 0; k < 128; ++k)
                    value += double(matrix[k * 128 + v]) * mixed[q_start + k] * q_scale;
                out[h * 128 + v] = static_cast<float>(value);
            }
        }
        normalize(out, scale, false);
        math::gate(math::Gate::silu, out, z, out);
        return project(p + "out_proj.weight", out, cancel);
    }

    Vector feedforward(
        const Vector& x,
        const std::string& p,
        const std::atomic_bool& cancel,
        bool tower = false
    ) {
        auto gate = project(p + (tower ? "w1.weight" : "gate_proj.weight"), x, cancel);
        const auto up =
            project(p + (tower ? "w3.weight" : "up_proj.weight"), x, cancel);
        math::gate(math::Gate::silu, up, gate, gate);
        return project(p + (tower ? "w2.weight" : "down_proj.weight"), gate, cancel);
    }

    Vector mixture(const Vector& x, std::size_t layer, const std::atomic_bool& cancel) {
        const auto p = "model.language_model.layers." + std::to_string(layer) + ".mlp.";
        auto logits = project(p + "gate.weight", x, cancel);
        std::array<std::size_t, 256> order;
        std::iota(order.begin(), order.end(), 0);
        std::partial_sort(
            order.begin(), order.begin() + 8, order.end(), [&](auto a, auto b) {
                return logits[a] > logits[b];
            }
        );
        float denominator = 0;
        std::array<float, 8> probability{};
        for (std::size_t i = 0; i < 8; ++i) {
            probability[i] = std::exp(logits[order[i]] - logits[order[0]]);
            denominator += probability[i];
        }
        Vector output(hidden);
        for (std::size_t i = 0; i < 8; ++i) {
            check_cancel(cancel);
            const auto expert =
                feedforward(x, p + "experts." + std::to_string(order[i]) + ".", cancel);
            add(output, expert, probability[i] / denominator);
        }
        const auto shared = feedforward(x, p + "shared_expert.", cancel);
        const auto gate = project(p + "shared_expert_gate.weight", x, cancel);
        add(output, shared, sigmoid(gate[0]));
        return output;
    }

    Vector backbone(
        std::uint32_t token,
        const Frame& frame,
        std::size_t position,
        const std::atomic_bool& cancel
    ) {
        auto x =
            weights->row("model.language_model.embed_tokens.weight", token, cancel);
        if (token >= metadata.semantic_begin && token <= metadata.semantic_end) {
            const auto embedded = weights->row(
                "audio_decoder.codebook_embeddings.weight", 4096 + frame[1], cancel
            );
            add(x, embedded);
            for (auto& value : x)
                value *= 1.f / std::sqrt(2.f);
        }
        for (std::size_t layer = 0; layer < 40; ++layer) {
            check_cancel(cancel);
            const auto p = "model.language_model.layers." + std::to_string(layer) + ".";
            const auto normalized = norm(x, p + "input_layernorm.weight", cancel);
            const auto attended =
                (layer + 1) % 4 ? linear_attention(normalized, layer, cancel)
                                : full_attention(normalized, layer, position, cancel);
            add(x, attended);
            const auto projected = mixture(
                norm(x, p + "post_attention_layernorm.weight", cancel), layer, cancel
            );
            add(x, projected);
        }
        return norm(std::move(x), "model.language_model.norm.weight", cancel);
    }

    Vector tower_layer(
        Vector x,
        std::size_t layer,
        std::uint32_t code,
        bool semantic,
        const Vector& shared_input,
        std::size_t position,
        const std::atomic_bool& cancel
    ) {
        const bool shared = layer < 4;
        const auto p = std::string(
                           shared ? "audio_decoder_mtp_shared.layers."
                                  : "audio_decoder_mtp.layers."
                       )
                       + std::to_string(shared ? layer : layer - 4) + ".";
        Vector embeddings(hidden);
        if (semantic)
            embeddings = weights->row(
                p
                    + (shared ? "semantic_embedding.weight"
                              : "codebook_embedding.weight"),
                code,
                cancel
            );
        if (layer == 0)
            add(embeddings, shared_input);
        embeddings = norm(std::move(embeddings), p + "enorm.weight", cancel);
        x = norm(std::move(x), p + "hnorm.weight", cancel);
        embeddings.insert(embeddings.end(), x.begin(), x.end());
        auto mixed = project(p + "eh_proj.weight", embeddings, cancel);
        const auto input =
            norm(mixed, p + "transformer_layer.attention_norm.weight", cancel);
        const auto a = p + "transformer_layer.attention.";
        const auto qkv = project(a + "wqkv.weight", input, cancel);
        // Saved tower rows are [all Q | all gates | all K | all V].
        Vector q(qkv.begin(), qkv.begin() + query_width);
        Vector gate(qkv.begin() + query_width, qkv.begin() + 2 * query_width);
        Vector k(
            qkv.begin() + 2 * query_width, qkv.begin() + 2 * query_width + kv_width
        );
        Vector v(qkv.begin() + 2 * query_width + kv_width, qkv.end());
        const auto attended = attend(
            std::move(q),
            std::move(k),
            v,
            gate,
            attention[10 + layer],
            position,
            true,
            a,
            cancel
        );
        add(mixed, project(a + "wo.weight", attended, cancel));
        const auto feed = feedforward(
            norm(mixed, p + "transformer_layer.ffn_norm.weight", cancel),
            p + "transformer_layer.feed_forward.",
            cancel,
            true
        );
        add(mixed, feed);
        return norm(std::move(mixed), p + "final_layernorm.weight", cancel);
    }

    Frame tower(
        Vector x,
        const Frame& previous,
        bool previous_semantic,
        std::uint32_t next,
        std::size_t position,
        bool sampling,
        const Sampling& options,
        std::mt19937_64& random,
        const std::atomic_bool& cancel
    ) {
        const bool semantic =
            next >= metadata.semantic_begin && next <= metadata.semantic_end;
        Frame frame{};
        frame[0] = semantic ? next - metadata.semantic_begin : 0;
        Vector extra(hidden);
        if (previous_semantic)
            for (std::size_t c = 2; c < 10; ++c)
                add(extra,
                    weights->row(
                        "audio_decoder.codebook_embeddings.weight",
                        c * 4096 + previous[c],
                        cancel
                    ));
        for (std::size_t layer = 0; layer < 13; ++layer) {
            check_cancel(cancel);
            x = tower_layer(
                std::move(x),
                layer,
                layer < 4 ? frame[0] : frame[layer - 4],
                semantic,
                extra,
                position,
                cancel
            );
            if (sampling && layer >= 4) {
                const auto logits = project(
                    "audio_decoder_mtp.heads." + std::to_string(layer - 4) + ".weight",
                    x,
                    cancel
                );
                frame[layer - 3] = sample(logits, options, random);
            }
        }
        return frame;
    }
};

Generator::Generator(
    std::shared_ptr<Resources> resources,
    std::shared_ptr<DenseCompute> compute,
    std::size_t context_limit
)
    : impl_(
          std::make_unique<Impl>(
              std::move(resources), std::move(compute), context_limit
          )
      ) {
}

Generator::~Generator() = default;

void Generator::load(
    const std::filesystem::path& directory, const std::atomic_bool& cancel
) {
    auto& s = *impl_;
    require(!s.handle, "drama_already_loaded");
    require(
        s.compute->supports_weight_sources(), "drama_streamed_projection_unsupported"
    );
    check_cancel(cancel);
    s.metadata = inspect(directory, s.metadata.context_limit);
    auto footprint = s.resources->snapshot().capacity;
    std::fill(footprint.begin(), footprint.end(), 0);
    footprint[0] = s.metadata.host_bytes;
    s.handle = s.resources->reserve(Workload::tts, std::move(footprint));
    // Each projection source admits its own device tiles; only headers are read
    // here. Planning freezes a capacity-weighted split across owned GPUs, without
    // requiring the full decoded expert inventory to fit in device memory.
    s.weights = std::make_unique<Weights>(directory, s.metadata, cancel);
    s.weights->validate(cancel);
    check_cancel(cancel);
    s.compute->prepare_weights(s.weights->execution_bytes());
    check_cancel(cancel);
    for (std::size_t i = 0; i < 23; ++i)
        s.attention.emplace_back(s.metadata.context_limit);
    s.delta.resize(30);
    s.resources->loaded(s.handle);
    s.resident = true;
}

const ModelInfo& Generator::info() const {
    return impl_->metadata;
}

bool Generator::generate(
    std::span<const std::uint32_t> prompt,
    std::size_t max_frames,
    const std::atomic_bool& cancel,
    const std::function<void(std::span<const std::uint32_t>)>& on_frame,
    Sampling sampling
) {
    auto& s = *impl_;
    require(s.resident, "drama_not_loaded");
    require(
        !prompt.empty() && max_frames && prompt.size() <= s.metadata.context_limit
            && max_frames <= s.metadata.context_limit - prompt.size() + 1,
        "drama_context_exceeded"
    );
    require(bool(on_frame), "drama_frame_callback_required");
    require(
        std::isfinite(sampling.temperature) && sampling.temperature >= 0
            && std::isfinite(sampling.top_p) && sampling.top_p > 0
            && sampling.top_p <= 1 && sampling.top_k > 0 && sampling.top_k <= 4096,
        "drama_sampling_options"
    );
    for (auto token : prompt)
        require(
            token < s.metadata.vocabulary
                && !(
                    token >= s.metadata.semantic_begin
                    && token <= s.metadata.semantic_end
                ),
            "drama_text_prompt_required_no_reference_frames"
        );
    for (auto& state : s.delta) {
        std::fill(state.convolution.begin(), state.convolution.end(), 0);
        std::fill(state.recurrent.begin(), state.recurrent.end(), 0);
    }
    // Attention reads only the prefix overwritten during this request.
    std::mt19937_64 random(sampling.seed);
    Frame frame{};
    Vector hidden_states;
    std::size_t position = 0;
    for (; position < prompt.size(); ++position) {
        check_cancel(cancel);
        hidden_states = s.backbone(prompt[position], frame, position, cancel);
        if (position + 1 < prompt.size())
            s.tower(
                hidden_states,
                frame,
                false,
                prompt[position + 1],
                position,
                false,
                sampling,
                random,
                cancel
            );
    }
    --position;
    std::array<std::uint32_t, 16> history;
    history.fill(UINT32_MAX);
    bool previous_semantic = false;
    for (std::size_t produced = 0; produced < max_frames; ++produced) {
        check_cancel(cancel);
        auto logits = s.weights->project(
            *s.compute,
            "lm_head.weight",
            hidden_states,
            cancel,
            s.metadata.semantic_begin,
            4096
        );
        const auto eos = s.weights->project(
            *s.compute, "lm_head.weight", hidden_states, cancel, s.metadata.eos, 1
        );
        // EOS sorts before the semantic band in the upstream candidate table.
        logits.insert(logits.begin(), eos[0]);
        auto selected = sample(logits, sampling, random);
        const auto alternate =
            sample(logits, Sampling{1.f, 0.95f, 30, sampling.seed}, random);
        auto token =
            selected ? s.metadata.semantic_begin + selected - 1 : s.metadata.eos;
        if (selected
            && std::find(history.begin(), history.end(), token) != history.end()) {
            selected = alternate;
            token =
                selected ? s.metadata.semantic_begin + selected - 1 : s.metadata.eos;
        }
        if (token == s.metadata.eos)
            return true;
        frame = s.tower(
            std::move(hidden_states),
            frame,
            previous_semantic,
            token,
            position,
            true,
            sampling,
            random,
            cancel
        );
        check_cancel(cancel);
        on_frame(frame);
        history[produced % history.size()] = token;
        if (produced + 1 == max_frames)
            return false;
        ++position;
        hidden_states = s.backbone(token, frame, position, cancel);
        previous_semantic = true;
    }
    return false;
}
}  // namespace kadan::drama
