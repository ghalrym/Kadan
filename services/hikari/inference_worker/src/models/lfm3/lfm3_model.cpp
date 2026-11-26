#include "kadan/lfm3.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>

#include "kadan/float_weights.hpp"

namespace kadan::lfm3 {
namespace {
void check(bool ok, const char* message) {
    if (!ok)
        throw std::runtime_error(message);
}

void check_cancel(const std::atomic_bool& cancel) {
    if (cancel.load(std::memory_order_relaxed))
        throw std::runtime_error("lfm3_cancelled");
}

double sigmoid_f64(double x) {
    const double e = std::exp(-std::abs(x));
    return x >= 0.0 ? 1.0 / (1.0 + e) : e / (1.0 + e);
}

double silu_f64(double x) {
    return x * sigmoid_f64(x);
}

std::uint64_t splitmix64(std::uint64_t& state) {
    std::uint64_t z = (state += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

double uniform_unit(std::uint64_t& state) {
    return static_cast<double>(splitmix64(state) >> 11) * (1.0 / 9007199254740992.0);
}

LayerKind parse_layer_kind(const std::string& name) {
    if (name == "conv")
        return LayerKind::conv;
    if (name == "full_attention")
        return LayerKind::full_attention;
    if (name == "sliding_attention")
        return LayerKind::sliding_attention;
    throw std::runtime_error("lfm3_unsupported_layer_type: " + name);
}

RopeParams parse_rope(const Json& params, const char* key, RopeParams fallback) {
    if (!params.is_object())
        return fallback;
    const Json* entry = nullptr;
    if (params.contains(key) && params.at(key).is_object())
        entry = &params.at(key);
    else if (params.contains("rope_theta"))
        entry = &params;
    if (!entry)
        return fallback;
    RopeParams out = fallback;
    out.theta = entry->value("rope_theta", fallback.theta);
    out.partial_rotary_factor =
        entry->value("partial_rotary_factor", fallback.partial_rotary_factor);
    return out;
}

bool has_tensor(const checkpoint::Shard& shard, std::string_view name) {
    try {
        (void)shard.tensor(name);
        return true;
    } catch (...) {
        return false;
    }
}

void project_linear(
    Resources& resources,
    Workload workload,
    DenseCompute* compute,
    const checkpoint::Shard& shard,
    std::string_view weight_name,
    std::span<const float> bias,
    std::span<const float> input,
    std::size_t in_dim,
    std::size_t out_dim,
    std::span<float> output,
    const std::atomic_bool& cancel
) {
    checkpoint::project_source(
        resources,
        workload,
        compute,
        shard,
        weight_name,
        bias,
        input,
        in_dim,
        out_dim,
        output,
        cancel
    );
}

// CPU GQA attention supporting causal, sliding window, bidirectional, and d1 set-mask.
void cpu_gqa_attention(
    std::span<const float> q,  // [q_rows, heads * head_dim]
    std::span<const float> k,  // [kv_len, kv_heads * head_dim]
    std::span<const float> v,  // [kv_len, kv_heads * head_dim]
    std::size_t q_rows,
    std::size_t kv_len,
    std::size_t start_pos,
    std::size_t heads,
    std::size_t kv_heads,
    std::size_t head_dim,
    bool causal,
    std::size_t window,
    const SetAttentionMask* set_mask,
    std::span<float> out,
    const std::atomic_bool& cancel
) {
    const std::size_t group = heads / kv_heads;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    std::vector<float> scores(kv_len);
    for (std::size_t qi = 0; qi < q_rows; ++qi) {
        if ((qi & 15) == 0)
            check_cancel(cancel);
        const std::size_t q_abs = start_pos + qi;
        for (std::size_t h = 0; h < heads; ++h) {
            const std::size_t kh = h / group;
            const float* qp = q.data() + (qi * heads + h) * head_dim;
            float max_score = -std::numeric_limits<float>::infinity();
            std::size_t k_begin = 0;
            std::size_t k_end = kv_len;
            if (!set_mask) {
                if (causal)
                    k_end = std::min(kv_len, q_abs + 1);
                if (window > 0 && q_abs + 1 > window)
                    k_begin = std::max(k_begin, q_abs + 1 - window);
            }
            for (std::size_t ki = k_begin; ki < k_end; ++ki) {
                if (set_mask) {
                    const bool q_in_state = q_abs < set_mask->state_len;
                    const bool k_in_state = ki < set_mask->state_len;
                    const bool same_q =
                        !set_mask->question_groups.empty()
                        && set_mask->question_groups[q_abs]
                               == set_mask->question_groups[ki];
                    if (!q_in_state && !k_in_state && !same_q) {
                        scores[ki] = -std::numeric_limits<float>::infinity();
                        continue;
                    }
                }
                const float* kp = k.data() + (ki * kv_heads + kh) * head_dim;
                double dot = 0.0;
                for (std::size_t d = 0; d < head_dim; ++d)
                    dot += static_cast<double>(qp[d]) * static_cast<double>(kp[d]);
                const float s = static_cast<float>(dot) * scale;
                scores[ki] = s;
                if (s > max_score)
                    max_score = s;
            }
            double sum = 0.0;
            for (std::size_t ki = k_begin; ki < k_end; ++ki) {
                if (std::isfinite(scores[ki])) {
                    const double e = std::exp(static_cast<double>(scores[ki] - max_score));
                    scores[ki] = static_cast<float>(e);
                    sum += e;
                } else {
                    scores[ki] = 0.0f;
                }
            }
            const float inv = sum > 0.0 ? static_cast<float>(1.0 / sum) : 0.0f;
            float* op = out.data() + (qi * heads + h) * head_dim;
            std::fill(op, op + head_dim, 0.0f);
            for (std::size_t ki = k_begin; ki < k_end; ++ki) {
                const float w = scores[ki] * inv;
                if (w == 0.0f)
                    continue;
                const float* vp = v.data() + (ki * kv_heads + kh) * head_dim;
                for (std::size_t d = 0; d < head_dim; ++d)
                    op[d] += w * vp[d];
            }
        }
    }
}
}  // namespace

BackboneConfig BackboneConfig::parse(const Json& root) {
    const Json& cfg = root.contains("text_config") && root.at("text_config").is_object()
                          ? root.at("text_config")
                          : root;
    BackboneConfig c;
    c.vocab_size = cfg.value("vocab_size", c.vocab_size);
    c.hidden_size = cfg.at("hidden_size").get<std::size_t>();
    c.intermediate_size = cfg.at("intermediate_size").get<std::size_t>();
    if (cfg.value("block_auto_adjust_ff_dim", false)) {
        const double mult = cfg.value("block_ffn_dim_multiplier", 1.0);
        const std::size_t m = cfg.value("block_multiple_of", std::size_t(256));
        std::size_t adjusted = static_cast<std::size_t>(
            static_cast<double>(2 * c.intermediate_size) / 3.0 * mult
        );
        if (m > 0)
            adjusted = ((adjusted + m - 1) / m) * m;
        c.intermediate_size = adjusted;
    }
    c.num_hidden_layers = cfg.at("num_hidden_layers").get<std::size_t>();
    c.num_attention_heads = cfg.at("num_attention_heads").get<std::size_t>();
    c.num_key_value_heads = cfg.at("num_key_value_heads").get<std::size_t>();
    c.head_dim = cfg.value(
        "head_dim",
        c.num_attention_heads ? c.hidden_size / c.num_attention_heads : std::size_t(128)
    );
    c.max_position_embeddings =
        cfg.value("max_position_embeddings", c.max_position_embeddings);
    c.conv_l_cache = cfg.value("conv_L_cache", c.conv_l_cache);
    if (cfg.contains("sliding_window") && !cfg.at("sliding_window").is_null())
        c.sliding_window = cfg.at("sliding_window").get<std::size_t>();
    else
        c.sliding_window = 0;
    c.norm_eps = cfg.value("norm_eps", c.norm_eps);
    c.conv_bias = cfg.value("conv_bias", c.conv_bias);
    c.use_sandwich_norm = cfg.value(
        "use_sandwich_norm", cfg.value("model_type", std::string("lfm3")) == "lfm3"
    );
    c.tie_word_embeddings = cfg.value("tie_word_embeddings", c.tie_word_embeddings);
    c.is_causal = cfg.value("is_causal", true);
    const auto opt_token_id = [&](const char* key) -> std::optional<std::uint32_t> {
        if (!cfg.contains(key) || cfg.at(key).is_null())
            return std::nullopt;
        const auto& val = cfg.at(key);
        if (val.is_number_integer() && val.get<std::int64_t>() >= 0)
            return static_cast<std::uint32_t>(val.get<std::int64_t>());
        if (val.is_array() && !val.empty() && val.at(0).is_number_integer()
            && val.at(0).get<std::int64_t>() >= 0)
            return static_cast<std::uint32_t>(val.at(0).get<std::int64_t>());
        return std::nullopt;
    };
    c.bos_token_id = opt_token_id("bos_token_id");
    c.eos_token_id = opt_token_id("eos_token_id");
    c.pad_token_id = opt_token_id("pad_token_id");
    c.layer_types.clear();
    if (cfg.contains("layer_types") && cfg.at("layer_types").is_array()) {
        for (const auto& entry : cfg.at("layer_types"))
            c.layer_types.push_back(parse_layer_kind(entry.get<std::string>()));
    } else {
        c.layer_types.assign(c.num_hidden_layers, LayerKind::full_attention);
    }
    check(c.layer_types.size() == c.num_hidden_layers, "lfm3_layer_types_mismatch");
    if (cfg.contains("rope_parameters")) {
        c.full_rope =
            parse_rope(cfg.at("rope_parameters"), "full_attention", {1000000.0, 1.0});
        c.sliding_rope =
            parse_rope(cfg.at("rope_parameters"), "sliding_attention", {10000.0, 1.0});
    }
    check(
        c.hidden_size > 0 && c.num_attention_heads > 0 && c.num_key_value_heads > 0
            && c.num_attention_heads % c.num_key_value_heads == 0 && c.head_dim > 0
            && c.head_dim % 2 == 0,
        "lfm3_invalid_config"
    );
    return c;
}

BackboneConfig BackboneConfig::liquid_codec_decoder_config(
    std::size_t encoder_dim, std::size_t decoder_window
) {
    BackboneConfig c;
    c.vocab_size = 1;
    c.hidden_size = encoder_dim;
    c.intermediate_size = 2304;
    c.num_hidden_layers = 8;
    c.num_attention_heads = 8;
    c.num_key_value_heads = 2;
    c.head_dim = 64;
    c.max_position_embeddings = 131072;
    c.norm_eps = 1e-5f;
    c.conv_l_cache = 7;
    c.conv_bias = false;
    c.use_sandwich_norm = true;
    c.tie_word_embeddings = false;
    c.is_causal = true;
    c.sliding_window = decoder_window;
    const LayerKind attn_kind =
        decoder_window ? LayerKind::sliding_attention : LayerKind::full_attention;
    c.layer_types.resize(8, LayerKind::conv);
    c.layer_types[2] = attn_kind;
    c.layer_types[4] = attn_kind;
    c.layer_types[6] = attn_kind;
    c.full_rope = {1000000.0, 1.0};
    c.sliding_rope = {1000000.0, 1.0};
    return c;
}

void BackboneState::reset(const BackboneConfig& config) {
    layers.clear();
    layers.resize(config.num_hidden_layers);
    for (std::size_t i = 0; i < config.num_hidden_layers; ++i) {
        if (config.layer_types[i] == LayerKind::conv && config.conv_l_cache > 1) {
            layers[i].conv_history.assign(
                config.hidden_size * (config.conv_l_cache - 1), 0.0f
            );
        }
    }
    position = 0;
}

void rms_norm_lfm3(
    std::span<const float> input,
    std::span<const float> weight,
    std::size_t rows,
    std::size_t width,
    float eps,
    std::span<float> output
) {
    check(
        width > 0 && input.size() == rows * width && weight.size() == width
            && output.size() == rows * width,
        "lfm3_rmsnorm_shape"
    );
    for (std::size_t r = 0; r < rows; ++r) {
        const float* in = input.data() + r * width;
        float* out = output.data() + r * width;
        double sum_sq = 0.0;
        for (std::size_t c = 0; c < width; ++c) {
            const double v = static_cast<double>(in[c]);
            sum_sq += v * v;
        }
        const double inv =
            1.0 / std::sqrt(sum_sq / static_cast<double>(width) + static_cast<double>(eps));
        for (std::size_t c = 0; c < width; ++c)
            out[c] = static_cast<float>(static_cast<double>(in[c]) * inv) * weight[c];
    }
}

void layer_norm_1d(
    std::span<const float> input,
    std::span<const float> weight,
    std::span<const float> bias,
    std::size_t rows,
    std::size_t width,
    float eps,
    std::span<float> output
) {
    check(
        width > 0 && input.size() == rows * width && weight.size() == width
            && (bias.empty() || bias.size() == width) && output.size() == rows * width,
        "lfm3_layernorm_shape"
    );
    for (std::size_t r = 0; r < rows; ++r) {
        const float* in = input.data() + r * width;
        float* out = output.data() + r * width;
        double mean = 0.0;
        for (std::size_t c = 0; c < width; ++c)
            mean += static_cast<double>(in[c]);
        mean /= static_cast<double>(width);
        double var = 0.0;
        for (std::size_t c = 0; c < width; ++c) {
            const double d = static_cast<double>(in[c]) - mean;
            var += d * d;
        }
        const double inv =
            1.0 / std::sqrt(var / static_cast<double>(width) + static_cast<double>(eps));
        for (std::size_t c = 0; c < width; ++c) {
            const double norm = (static_cast<double>(in[c]) - mean) * inv;
            const double b = bias.empty() ? 0.0 : static_cast<double>(bias[c]);
            out[c] = static_cast<float>(norm * static_cast<double>(weight[c]) + b);
        }
    }
}

void compute_partial_rope_inv_freq(
    double theta, double partial_rotary_factor, std::size_t head_dim, std::span<float> out
) {
    check(
        head_dim > 0 && head_dim % 2 == 0 && out.size() == head_dim / 2 && theta > 1.0
            && partial_rotary_factor > 0.0 && partial_rotary_factor <= 1.0,
        "lfm3_rope_params"
    );
    const std::size_t rope_angles =
        static_cast<std::size_t>(partial_rotary_factor * static_cast<double>(head_dim))
        / 2;
    for (std::size_t i = 0; i < out.size(); ++i) {
        if (i < rope_angles) {
            const double exponent =
                static_cast<double>(2 * i) / static_cast<double>(head_dim);
            out[i] = static_cast<float>(1.0 / std::pow(theta, exponent));
        } else {
            out[i] = 0.0f;
        }
    }
}

void apply_rope_lfm3(
    std::span<float> tensor,
    std::size_t rows,
    std::size_t heads,
    std::size_t head_dim,
    std::size_t start_position,
    std::span<const float> inv_freq
) {
    const std::size_t half = head_dim / 2;
    check(
        tensor.size() == rows * heads * head_dim && inv_freq.size() == half,
        "lfm3_rope_shape"
    );
    for (std::size_t r = 0; r < rows; ++r) {
        const float pos = static_cast<float>(start_position + r);
        for (std::size_t h = 0; h < heads; ++h) {
            float* v = tensor.data() + (r * heads + h) * head_dim;
            for (std::size_t i = 0; i < half; ++i) {
                const float angle = pos * inv_freq[i];
                const float c = std::cos(angle);
                const float s = std::sin(angle);
                const float x1 = v[i];
                const float x2 = v[i + half];
                v[i] = x1 * c - x2 * s;
                v[i + half] = x2 * c + x1 * s;
            }
        }
    }
}

void depthwise_causal_conv1d(
    std::span<const float> input,
    std::span<const float> weight,
    std::span<const float> bias,
    std::size_t rows,
    std::size_t channels,
    std::size_t kernel_size,
    std::span<float> history,
    std::span<float> output
) {
    check(
        kernel_size >= 1 && input.size() == rows * channels
            && weight.size() == channels * kernel_size
            && (bias.empty() || bias.size() == channels)
            && output.size() == rows * channels,
        "lfm3_conv_shape"
    );
    const std::size_t hist_len = kernel_size - 1;
    const bool has_history = !history.empty();
    if (has_history)
        check(history.size() == channels * hist_len, "lfm3_conv_history_shape");
    for (std::size_t r = 0; r < rows; ++r) {
        for (std::size_t c = 0; c < channels; ++c) {
            double acc = bias.empty() ? 0.0 : static_cast<double>(bias[c]);
            const float* wc = weight.data() + c * kernel_size;
            for (std::size_t k = 0; k < kernel_size; ++k) {
                const std::ptrdiff_t src_row =
                    static_cast<std::ptrdiff_t>(r) + static_cast<std::ptrdiff_t>(k)
                    - static_cast<std::ptrdiff_t>(hist_len);
                float val = 0.0f;
                if (src_row >= 0) {
                    val = input[static_cast<std::size_t>(src_row) * channels + c];
                } else if (has_history) {
                    const std::size_t hist_idx =
                        static_cast<std::size_t>(src_row + static_cast<std::ptrdiff_t>(hist_len));
                    val = history[c * hist_len + hist_idx];
                }
                acc += static_cast<double>(val) * static_cast<double>(wc[k]);
            }
            output[r * channels + c] = static_cast<float>(acc);
        }
    }
    if (has_history && hist_len > 0) {
        std::vector<float> updated(hist_len);
        for (std::size_t c = 0; c < channels; ++c) {
            float* hc = history.data() + c * hist_len;
            if (rows >= hist_len) {
                for (std::size_t k = 0; k < hist_len; ++k)
                    updated[k] = input[(rows - hist_len + k) * channels + c];
            } else {
                const std::size_t keep = hist_len - rows;
                for (std::size_t k = 0; k < keep; ++k)
                    updated[k] = hc[k + rows];
                for (std::size_t k = 0; k < rows; ++k)
                    updated[keep + k] = input[k * channels + c];
            }
            std::copy(updated.begin(), updated.end(), hc);
        }
    }
}

std::uint32_t sample_logits(
    std::span<float> logits,
    const SamplingConfig& config,
    std::span<const std::uint32_t> history,
    std::uint64_t& rng_state
) {
    check(!logits.empty(), "lfm3_empty_logits");
    if (config.repetition_penalty != 1.0f && !history.empty()) {
        const std::size_t start =
            (config.repetition_window > 0 && history.size() > config.repetition_window)
                ? history.size() - config.repetition_window
                : 0;
        std::vector<std::uint32_t> seen(history.begin() + start, history.end());
        std::sort(seen.begin(), seen.end());
        seen.erase(std::unique(seen.begin(), seen.end()), seen.end());
        for (std::uint32_t id : seen) {
            if (id < logits.size() && std::isfinite(logits[id])) {
                logits[id] = logits[id] < 0.0f ? logits[id] * config.repetition_penalty
                                               : logits[id] / config.repetition_penalty;
            }
        }
    }
    for (std::uint32_t id : config.suppress_ids) {
        if (id < logits.size())
            logits[id] = -std::numeric_limits<float>::infinity();
    }
    if (!config.do_sample || config.temperature <= 0.0f) {
        std::size_t best = 0;
        float best_val = -std::numeric_limits<float>::infinity();
        for (std::size_t i = 0; i < logits.size(); ++i) {
            if (logits[i] > best_val) {
                best_val = logits[i];
                best = i;
            }
        }
        return static_cast<std::uint32_t>(best);
    }

    const float inv_temp = 1.0f / config.temperature;
    std::vector<std::pair<float, std::uint32_t>> candidates;
    candidates.reserve(logits.size());
    for (std::size_t i = 0; i < logits.size(); ++i) {
        if (std::isfinite(logits[i]))
            candidates.emplace_back(logits[i] * inv_temp, static_cast<std::uint32_t>(i));
    }
    check(!candidates.empty(), "lfm3_all_logits_masked");
    const std::size_t k =
        (config.top_k > 0 && config.top_k < candidates.size()) ? config.top_k
                                                               : candidates.size();
    std::partial_sort(
        candidates.begin(),
        candidates.begin() + static_cast<std::ptrdiff_t>(k),
        candidates.end(),
        [](const auto& a, const auto& b) { return a.first > b.first; }
    );
    candidates.resize(k);

    const float max_s = candidates.front().first;
    std::vector<double> probs(candidates.size());
    double total = 0.0;
    for (std::size_t i = 0; i < candidates.size(); ++i) {
        probs[i] = std::exp(static_cast<double>(candidates[i].first - max_s));
        total += probs[i];
    }
    for (double& p : probs)
        p /= total;

    std::size_t kept = probs.size();
    if (config.top_p > 0.0f && config.top_p < 1.0f) {
        double cumulative = 0.0;
        for (std::size_t i = 0; i < probs.size(); ++i) {
            cumulative += probs[i];
            if (cumulative >= static_cast<double>(config.top_p)) {
                kept = i + 1;
                break;
            }
        }
    }
    double kept_sum = 0.0;
    for (std::size_t i = 0; i < kept; ++i)
        kept_sum += probs[i];
    const double u = uniform_unit(rng_state) * kept_sum;
    double acc = 0.0;
    for (std::size_t i = 0; i < kept; ++i) {
        acc += probs[i];
        if (u <= acc)
            return candidates[i].second;
    }
    return candidates[kept - 1].second;
}

Backbone::Backbone(
    BackboneConfig config,
    std::shared_ptr<checkpoint::Shard> shard,
    std::string weight_prefix,
    std::shared_ptr<Resources> resources,
    Workload workload,
    std::shared_ptr<DenseCompute> compute,
    bool load_embeddings,
    bool load_lm_head
)
    : config_(std::move(config)), shard_(std::move(shard)),
      prefix_(std::move(weight_prefix)), resources_(std::move(resources)),
      workload_(workload), compute_(std::move(compute)),
      has_embeddings_(load_embeddings), has_lm_head_(load_lm_head) {
    check(bool(shard_) && bool(resources_), "lfm3_backbone_dependencies");
    full_inv_freq_.resize(config_.head_dim / 2);
    sliding_inv_freq_.resize(config_.head_dim / 2);
    compute_partial_rope_inv_freq(
        config_.full_rope.theta,
        config_.full_rope.partial_rotary_factor,
        config_.head_dim,
        full_inv_freq_
    );
    compute_partial_rope_inv_freq(
        config_.sliding_rope.theta,
        config_.sliding_rope.partial_rotary_factor,
        config_.head_dim,
        sliding_inv_freq_
    );
    if (has_lm_head_) {
        if (!config_.tie_word_embeddings && has_tensor(*shard_, "lm_head.weight"))
            lm_head_name_ = "lm_head.weight";
        else
            lm_head_name_ = prefix_ + "embed_tokens.weight";
    }
}

Backbone::~Backbone() = default;

void Backbone::embed_tokens(
    std::span<const std::uint32_t> tokens,
    std::span<float> out,
    const std::atomic_bool& cancel
) const {
    check(has_embeddings_, "lfm3_no_embeddings");
    check(out.size() == tokens.size() * config_.hidden_size, "lfm3_embed_shape");
    const std::string name = prefix_ + "embed_tokens.weight";
    for (std::size_t i = 0; i < tokens.size(); ++i) {
        if ((i & 63) == 0)
            check_cancel(cancel);
        check(tokens[i] < config_.vocab_size, "lfm3_token_out_of_range");
        shard_->read_float_tensor(
            name,
            static_cast<std::size_t>(tokens[i]) * config_.hidden_size,
            out.subspan(i * config_.hidden_size, config_.hidden_size),
            cancel
        );
    }
}

void Backbone::forward(
    std::span<const float> inputs_embeds,
    std::size_t rows,
    std::span<float> output,
    BackboneState* state,
    const SetAttentionMask* set_mask,
    const std::atomic_bool& cancel,
    std::size_t prefill_chunk_size
) const {
    const std::size_t H = config_.hidden_size;
    check(
        rows > 0 && inputs_embeds.size() == rows * H && output.size() == rows * H,
        "lfm3_forward_shape"
    );
    if (state)
        check(
            state->layers.size() == config_.num_hidden_layers, "lfm3_state_layers"
        );

    const std::size_t chunk_cap =
        (set_mask || !config_.is_causal || prefill_chunk_size == 0)
            ? rows
            : std::min(rows, prefill_chunk_size);

    std::vector<float> hidden(chunk_cap * H);
    std::vector<float> normed(chunk_cap * H);
    std::vector<float> op_out(chunk_cap * H);
    std::vector<float> ffn_gate(chunk_cap * config_.intermediate_size);
    std::vector<float> ffn_up(chunk_cap * config_.intermediate_size);
    std::vector<float> ffn_out(chunk_cap * H);

    const std::size_t q_dim = config_.num_attention_heads * config_.head_dim;
    const std::size_t kv_dim = config_.num_key_value_heads * config_.head_dim;

    for (std::size_t offset = 0; offset < rows; offset += chunk_cap) {
        check_cancel(cancel);
        const std::size_t n = std::min(chunk_cap, rows - offset);
        const std::size_t start_pos = state ? state->position : offset;
        std::copy_n(inputs_embeds.data() + offset * H, n * H, hidden.data());

        for (std::size_t layer = 0; layer < config_.num_hidden_layers; ++layer) {
            check_cancel(cancel);
            const std::string lp = prefix_ + "layers." + std::to_string(layer) + ".";
            const LayerKind kind = config_.layer_types[layer];

            auto op_norm_w = checkpoint::float_slice(
                *resources_, workload_, *shard_, lp + "operator_norm.weight", 0, H, cancel
            );
            rms_norm_lfm3(
                std::span(hidden.data(), n * H),
                op_norm_w,
                n,
                H,
                config_.norm_eps,
                std::span(normed.data(), n * H)
            );

            if (kind == LayerKind::conv) {
                std::vector<float> bcx(n * 3 * H);
                std::vector<float> in_bias;
                if (config_.conv_bias && has_tensor(*shard_, lp + "conv.in_proj.bias")) {
                    auto b = checkpoint::float_slice(
                        *resources_,
                        workload_,
                        *shard_,
                        lp + "conv.in_proj.bias",
                        0,
                        3 * H,
                        cancel
                    );
                    in_bias.assign(b.begin(), b.end());
                }
                project_linear(
                    *resources_,
                    workload_,
                    compute_.get(),
                    *shard_,
                    lp + "conv.in_proj.weight",
                    in_bias,
                    std::span(normed.data(), n * H),
                    H,
                    3 * H,
                    bcx,
                    cancel
                );
                std::vector<float> gated(n * H);
                std::vector<float> c_gate(n * H);
                for (std::size_t r = 0; r < n; ++r) {
                    const float* row = bcx.data() + r * 3 * H;
                    for (std::size_t c = 0; c < H; ++c) {
                        gated[r * H + c] = row[c] * row[2 * H + c];
                        c_gate[r * H + c] = row[H + c];
                    }
                }
                auto conv_w = checkpoint::float_slice(
                    *resources_,
                    workload_,
                    *shard_,
                    lp + "conv.conv.weight",
                    0,
                    H * config_.conv_l_cache,
                    cancel
                );
                std::vector<float> conv_bias;
                if (config_.conv_bias && has_tensor(*shard_, lp + "conv.conv.bias")) {
                    auto b = checkpoint::float_slice(
                        *resources_,
                        workload_,
                        *shard_,
                        lp + "conv.conv.bias",
                        0,
                        H,
                        cancel
                    );
                    conv_bias.assign(b.begin(), b.end());
                }
                std::vector<float> conv_out(n * H);
                std::span<float> hist =
                    state ? std::span<float>(state->layers[layer].conv_history)
                          : std::span<float>{};
                depthwise_causal_conv1d(
                    gated,
                    conv_w,
                    conv_bias,
                    n,
                    H,
                    config_.conv_l_cache,
                    hist,
                    conv_out
                );
                for (std::size_t i = 0; i < n * H; ++i)
                    conv_out[i] *= c_gate[i];
                std::vector<float> out_bias;
                if (config_.conv_bias && has_tensor(*shard_, lp + "conv.out_proj.bias")) {
                    auto b = checkpoint::float_slice(
                        *resources_,
                        workload_,
                        *shard_,
                        lp + "conv.out_proj.bias",
                        0,
                        H,
                        cancel
                    );
                    out_bias.assign(b.begin(), b.end());
                }
                project_linear(
                    *resources_,
                    workload_,
                    compute_.get(),
                    *shard_,
                    lp + "conv.out_proj.weight",
                    out_bias,
                    conv_out,
                    H,
                    H,
                    std::span(op_out.data(), n * H),
                    cancel
                );
            } else {
                const bool sliding = (kind == LayerKind::sliding_attention);
                const std::size_t window = sliding ? config_.sliding_window : 0;
                const std::string ap = lp + "self_attn.";
                std::vector<float> q(n * q_dim);
                std::vector<float> k(n * kv_dim);
                std::vector<float> v(n * kv_dim);
                project_linear(
                    *resources_,
                    workload_,
                    compute_.get(),
                    *shard_,
                    ap + "q_proj.weight",
                    {},
                    std::span(normed.data(), n * H),
                    H,
                    q_dim,
                    q,
                    cancel
                );
                project_linear(
                    *resources_,
                    workload_,
                    compute_.get(),
                    *shard_,
                    ap + "k_proj.weight",
                    {},
                    std::span(normed.data(), n * H),
                    H,
                    kv_dim,
                    k,
                    cancel
                );
                project_linear(
                    *resources_,
                    workload_,
                    compute_.get(),
                    *shard_,
                    ap + "v_proj.weight",
                    {},
                    std::span(normed.data(), n * H),
                    H,
                    kv_dim,
                    v,
                    cancel
                );
                if (has_tensor(*shard_, ap + "q_norm.weight")) {
                    auto qn = checkpoint::float_slice(
                        *resources_,
                        workload_,
                        *shard_,
                        ap + "q_norm.weight",
                        0,
                        config_.head_dim,
                        cancel
                    );
                    rms_norm_lfm3(
                        q,
                        qn,
                        n * config_.num_attention_heads,
                        config_.head_dim,
                        config_.norm_eps,
                        q
                    );
                } else if (has_tensor(*shard_, ap + "q_layernorm.weight")) {
                    auto qn = checkpoint::float_slice(
                        *resources_,
                        workload_,
                        *shard_,
                        ap + "q_layernorm.weight",
                        0,
                        config_.head_dim,
                        cancel
                    );
                    rms_norm_lfm3(
                        q,
                        qn,
                        n * config_.num_attention_heads,
                        config_.head_dim,
                        config_.norm_eps,
                        q
                    );
                }
                if (has_tensor(*shard_, ap + "k_norm.weight")) {
                    auto kn = checkpoint::float_slice(
                        *resources_,
                        workload_,
                        *shard_,
                        ap + "k_norm.weight",
                        0,
                        config_.head_dim,
                        cancel
                    );
                    rms_norm_lfm3(
                        k,
                        kn,
                        n * config_.num_key_value_heads,
                        config_.head_dim,
                        config_.norm_eps,
                        k
                    );
                } else if (has_tensor(*shard_, ap + "k_layernorm.weight")) {
                    auto kn = checkpoint::float_slice(
                        *resources_,
                        workload_,
                        *shard_,
                        ap + "k_layernorm.weight",
                        0,
                        config_.head_dim,
                        cancel
                    );
                    rms_norm_lfm3(
                        k,
                        kn,
                        n * config_.num_key_value_heads,
                        config_.head_dim,
                        config_.norm_eps,
                        k
                    );
                }
                const auto& inv_freq = sliding ? sliding_inv_freq_ : full_inv_freq_;
                apply_rope_lfm3(
                    q, n, config_.num_attention_heads, config_.head_dim, start_pos, inv_freq
                );
                apply_rope_lfm3(
                    k, n, config_.num_key_value_heads, config_.head_dim, start_pos, inv_freq
                );

                std::span<const float> k_all = k;
                std::span<const float> v_all = v;
                std::size_t kv_len = n;
                if (state) {
                    auto& ls = state->layers[layer];
                    ls.keys.insert(ls.keys.end(), k.begin(), k.end());
                    ls.values.insert(ls.values.end(), v.begin(), v.end());
                    ls.kv_len += n;
                    k_all = ls.keys;
                    v_all = ls.values;
                    kv_len = ls.kv_len;
                }

                std::vector<float> attn_out(n * q_dim);
                const SetAttentionMask* active_set =
                    (!sliding && set_mask) ? set_mask : nullptr;
                bool gpu_done = false;
                if (compute_ && !active_set) {
                    const std::size_t cuda_window =
                        (window > 0 && window < kv_len) ? window : 0;
                    gpu_done = compute_->attend(
                        q,
                        k_all,
                        v_all,
                        n,
                        kv_len,
                        config_.num_attention_heads,
                        config_.num_key_value_heads,
                        config_.head_dim,
                        config_.is_causal ? n : 0,
                        cuda_window,
                        start_pos,
                        attn_out,
                        cancel
                    );
                }
                if (!gpu_done) {
                    cpu_gqa_attention(
                        q,
                        k_all,
                        v_all,
                        n,
                        kv_len,
                        start_pos,
                        config_.num_attention_heads,
                        config_.num_key_value_heads,
                        config_.head_dim,
                        config_.is_causal,
                        window,
                        active_set,
                        attn_out,
                        cancel
                    );
                }

                if (has_tensor(*shard_, ap + "gate_proj.weight")) {
                    std::vector<float> gate_logits(n * config_.num_attention_heads);
                    project_linear(
                        *resources_,
                        workload_,
                        compute_.get(),
                        *shard_,
                        ap + "gate_proj.weight",
                        {},
                        std::span(normed.data(), n * H),
                        H,
                        config_.num_attention_heads,
                        gate_logits,
                        cancel
                    );
                    for (std::size_t r = 0; r < n; ++r) {
                        for (std::size_t h = 0; h < config_.num_attention_heads; ++h) {
                            const float g = static_cast<float>(
                                sigmoid_f64(gate_logits[r * config_.num_attention_heads + h])
                            );
                            float* hp =
                                attn_out.data()
                                + (r * config_.num_attention_heads + h) * config_.head_dim;
                            for (std::size_t d = 0; d < config_.head_dim; ++d)
                                hp[d] *= g;
                        }
                    }
                }

                const std::string o_name = has_tensor(*shard_, ap + "o_proj.weight")
                                               ? (ap + "o_proj.weight")
                                               : (ap + "out_proj.weight");
                project_linear(
                    *resources_,
                    workload_,
                    compute_.get(),
                    *shard_,
                    o_name,
                    {},
                    attn_out,
                    q_dim,
                    H,
                    std::span(op_out.data(), n * H),
                    cancel
                );
            }

            if (config_.use_sandwich_norm
                && has_tensor(*shard_, lp + "post_operator_norm.weight")) {
                auto post_w = checkpoint::float_slice(
                    *resources_,
                    workload_,
                    *shard_,
                    lp + "post_operator_norm.weight",
                    0,
                    H,
                    cancel
                );
                rms_norm_lfm3(
                    std::span(op_out.data(), n * H),
                    post_w,
                    n,
                    H,
                    config_.norm_eps,
                    std::span(op_out.data(), n * H)
                );
            }
            for (std::size_t i = 0; i < n * H; ++i)
                hidden[i] += op_out[i];

            auto ffn_norm_w = checkpoint::float_slice(
                *resources_, workload_, *shard_, lp + "ffn_norm.weight", 0, H, cancel
            );
            rms_norm_lfm3(
                std::span(hidden.data(), n * H),
                ffn_norm_w,
                n,
                H,
                config_.norm_eps,
                std::span(normed.data(), n * H)
            );
            const std::string fp = lp + "feed_forward.";
            project_linear(
                *resources_,
                workload_,
                compute_.get(),
                *shard_,
                fp + "w1.weight",
                {},
                std::span(normed.data(), n * H),
                H,
                config_.intermediate_size,
                std::span(ffn_gate.data(), n * config_.intermediate_size),
                cancel
            );
            project_linear(
                *resources_,
                workload_,
                compute_.get(),
                *shard_,
                fp + "w3.weight",
                {},
                std::span(normed.data(), n * H),
                H,
                config_.intermediate_size,
                std::span(ffn_up.data(), n * config_.intermediate_size),
                cancel
            );
            for (std::size_t i = 0; i < n * config_.intermediate_size; ++i) {
                ffn_gate[i] =
                    static_cast<float>(silu_f64(ffn_gate[i])) * ffn_up[i];
            }
            project_linear(
                *resources_,
                workload_,
                compute_.get(),
                *shard_,
                fp + "w2.weight",
                {},
                std::span(ffn_gate.data(), n * config_.intermediate_size),
                config_.intermediate_size,
                H,
                std::span(ffn_out.data(), n * H),
                cancel
            );
            if (config_.use_sandwich_norm
                && has_tensor(*shard_, lp + "post_ffn_norm.weight")) {
                auto post_w = checkpoint::float_slice(
                    *resources_,
                    workload_,
                    *shard_,
                    lp + "post_ffn_norm.weight",
                    0,
                    H,
                    cancel
                );
                rms_norm_lfm3(
                    std::span(ffn_out.data(), n * H),
                    post_w,
                    n,
                    H,
                    config_.norm_eps,
                    std::span(ffn_out.data(), n * H)
                );
            }
            for (std::size_t i = 0; i < n * H; ++i)
                hidden[i] += ffn_out[i];
        }

        auto final_w = checkpoint::float_slice(
            *resources_, workload_, *shard_, prefix_ + "embedding_norm.weight", 0, H, cancel
        );
        rms_norm_lfm3(
            std::span(hidden.data(), n * H),
            final_w,
            n,
            H,
            config_.norm_eps,
            output.subspan(offset * H, n * H)
        );
        if (state)
            state->position += n;
    }
}

void Backbone::lm_head(
    std::span<const float> hidden,
    std::size_t rows,
    std::span<float> logits,
    const std::atomic_bool& cancel
) const {
    check(has_lm_head_, "lfm3_no_lm_head");
    check(
        hidden.size() == rows * config_.hidden_size
            && logits.size() == rows * config_.vocab_size,
        "lfm3_lm_head_shape"
    );
    project_linear(
        *resources_,
        workload_,
        compute_.get(),
        *shard_,
        lm_head_name_,
        {},
        hidden,
        config_.hidden_size,
        config_.vocab_size,
        logits,
        cancel
    );
}

}  // namespace kadan::lfm3
