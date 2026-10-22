// Qwen3.8 Flash Next (GGUF qwen4exp): llama.cpp src/models/qwen4exp.cpp and
// llama-memory-hybrid-idx.cpp at 6420cf5dd588c68e48657a0bbd82bb3bd5ac7209.
// Hyper-connections replace layer norms, full-attention layers use QSA top-block
// selection, and one linear layer adds n-gram hash embeddings. Text only; the
// MTP head and vision tower are not executed.
#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>
#include <string>
#include <variant>

#include "gguf_model.hpp"
#include "qwen_layers.hpp"

namespace kadan::text {
namespace {
constexpr std::size_t hidden = 2560, layers = 48, streams = 4, wide = streams * hidden,
                      low_rank = 320, experts = 512, top_k = 10, expert_ff = 640,
                      vocabulary_size = 248320, context_length = 262144,
                      index_heads = 4, index_dim = 128, index_budget = 2048, pool = 4,
                      ple_layer = 1, ngram = 3, ple_per_gram = 8,
                      ple_heads = (ngram - 1) * ple_per_gram, ple_dim = 160,
                      ple_kernel = 4, ple_history = (ple_kernel - 1) * ngram,
                      ple_table_rows = 320001536;
constexpr unsigned ple_eos = 248044;
constexpr float eps = 1e-6f;
const DeltaNetSpec delta_spec{hidden, 16, 48, 128, 4, eps, true};
const AttentionSpec attention_spec{hidden, 24, 2, 256, 64, 1e7f, eps};
const MoeSpec moe_spec{hidden, experts, top_k, expert_ff, expert_ff};

bool full_attention(std::size_t layer) {
    return (layer + 1) % 4 == 0;
}

std::string prefix(std::size_t layer) {
    return "blk." + std::to_string(layer) + ".";
}

template <class Metadata> void validate(const Metadata& m) {
    need(m.string("general.architecture") == "qwen4exp", "qwen4exp_architecture");
    for (const auto& [key, expected] :
         std::initializer_list<std::pair<const char*, std::uint64_t>>{
             {"block_count", layers},
             {"context_length", context_length},
             {"embedding_length", hidden},
             {"attention.head_count", attention_spec.heads},
             {"attention.head_count_kv", attention_spec.kv_heads},
             {"attention.key_length", attention_spec.dim},
             {"attention.value_length", attention_spec.dim},
             {"rope.dimension_count", attention_spec.rotary},
             {"expert_count", experts},
             {"expert_used_count", top_k},
             {"expert_feed_forward_length", expert_ff},
             {"expert_shared_feed_forward_length", expert_ff},
             {"ssm.conv_kernel", delta_spec.conv},
             {"ssm.state_size", delta_spec.dim},
             {"ssm.group_count", delta_spec.key_heads},
             {"ssm.time_step_rank", delta_spec.value_heads},
             {"ssm.inner_size", delta_spec.value_heads * delta_spec.dim},
             {"full_attention_interval", 4},
             {"hyper_connection.count", streams},
             {"hyper_connection.low_rank", low_rank},
             {"attention.indexer.head_count", index_heads},
             {"attention.indexer.key_length", index_dim},
             {"attention.indexer.top_k", index_budget},
             {"ple.ngram_size", ngram},
             {"ple.heads_per_ngram", ple_per_gram},
             {"ple.conv_kernel", ple_kernel},
             {"ple.eos_token_id", ple_eos},
             {"embedding_length_per_layer_input", ple_dim}
         })
        need(
            m.integer(std::string("qwen4exp.") + key) == expected, "qwen4exp_dimensions"
        );
    need(
        std::abs(m.number("qwen4exp.rope.freq_base") - 1e7) < 1
            && std::abs(m.number("qwen4exp.attention.layer_norm_rms_epsilon") - eps)
                   < 1e-12,
        "qwen4exp_arithmetic"
    );
    const auto sections = m.integers("qwen4exp.rope.dimension_sections");
    need(
        sections.size() == 4 && sections[0] == 11 && sections[1] == 11
            && sections[2] == 10 && sections[3] == 0,
        "qwen4exp_rope_sections"
    );
    const auto ratios = m.integers("qwen4exp.attention.compress_ratios");
    need(ratios.size() == layers, "qwen4exp_compression");
    for (std::size_t i = 0; i < layers; ++i)
        need(
            ratios[i] == (full_attention(i) ? std::int64_t(pool) : 0),
            "qwen4exp_compression"
        );
    const auto ple = m.integers("qwen4exp.ple.layers");
    need(ple.size() == 1 && ple[0] == std::int64_t(ple_layer), "qwen4exp_ple_layers");
    need(
        m.integers("qwen4exp.ple.layer_multipliers").size() == ngram
            && m.integers("qwen4exp.ple.head_offsets").size() == ple_heads
            && m.integers("qwen4exp.ple.head_vocab_sizes").size() == ple_heads,
        "qwen4exp_ple_hash"
    );
    // Absent keys select llama.cpp's defaults: no routed weight scale, no MTP,
    // and the interval-derived recurrent schedule.
    need(
        !m.has("qwen4exp.expert_weights_scale")
            && !m.has("qwen4exp.nextn_predict_layers")
            && !m.has("qwen4exp.attention.recurrent_layers")
            && m.string("tokenizer.ggml.model") == "gpt2"
            && m.string("tokenizer.ggml.pre") == "qwen35"
            && m.integer("tokenizer.ggml.eos_token_id") == 248046,
        "qwen4exp_contract"
    );
}

// One hyper-connection mixer: grouped RMSNorm with folded (1 + w) gains, a
// low-rank sigmoid gate and the stream mean.
struct Mixer {
    std::string down, up;
    Vec norm, inject;
};

struct QsaAttention {
    AttentionLayer attention;
    Vec index_q_norm, index_k_norm, raw_keys, pooled;

    QsaAttention(const GgufWeights& w, const std::string& p, std::size_t context)
        : attention(w, p, attention_spec, context),
          index_q_norm(w.read(p + "indexer.q_norm.weight")),
          index_k_norm(w.read(p + "indexer.k_norm.weight")),
          raw_keys(context * index_dim), pooled(context / pool * index_dim) {
    }
};

struct Layer {
    Mixer attention_mixer, ffn_mixer;
    MoeLayer moe;
    std::variant<DeltaNetLayer, QsaAttention> attention;
};

class Qwen4Exp final : public Model {
public:
    Qwen4Exp(const GgufWeights& weights, DenseCompute& compute, std::size_t context)
        : w_(weights), compute_(compute), context_(context),
          head_{
              "output_hc_down.weight",
              "output_hc_up.weight",
              weights.read("output_hc_norm.weight"),
              {}
          } {
        need(context && context <= context_length, "qwen4exp_context");
        const auto& gguf = weights.gguf();
        for (const auto value : gguf.integers("qwen4exp.ple.layer_multipliers"))
            multipliers_.push_back(std::uint64_t(value));
        const auto offsets = gguf.integers("qwen4exp.ple.head_offsets");
        const auto sizes = gguf.integers("qwen4exp.ple.head_vocab_sizes");
        for (std::size_t h = 0; h < ple_heads; ++h) {
            need(
                sizes[h] > 0 && offsets[h] >= 0
                    && offsets[h] + sizes[h] <= std::int64_t(ple_table_rows),
                "qwen4exp_ple_range"
            );
            offsets_[h] = std::uint64_t(offsets[h]);
            sizes_[h] = std::uint64_t(sizes[h]);
        }
        for (std::size_t i = 0; i < layers; ++i) {
            const auto p = prefix(i);
            layers_.push_back(
                {mixer(p + "hc_attn_"),
                 mixer(p + "hc_ffn_"),
                 MoeLayer(w_, p, moe_spec),
                 full_attention(i)
                     ? std::variant<DeltaNetLayer, QsaAttention>(
                           std::in_place_type<QsaAttention>, w_, p, context
                       )
                     : std::variant<DeltaNetLayer, QsaAttention>(
                           std::in_place_type<DeltaNetLayer>, w_, p, delta_spec
                       )}
            );
        }
        const auto p = prefix(ple_layer);
        ple_norm_key_ = w_.read(p + "ple_norm_key.weight");
        ple_norm_query_ = w_.read(p + "ple_norm_query.weight");
        ple_norm_conv_ = w_.read(p + "ple_norm_conv.weight");
        ple_conv_ = w_.read(p + "ple_conv1d.weight");
        ple_state_.resize(ple_history * wide);
    }

    std::size_t vocabulary() const override {
        return vocabulary_size;
    }

    std::size_t device_state_bytes() const override {
        return 12 * AttentionLayer::state_bytes(attention_spec, context_)
               + 36 * DeltaNetLayer::device_bytes(delta_spec);
    }

    bool eos(unsigned token) const override {
        return token == 248046 || token == 248044;
    }

    void reset() override {
        for (auto& layer : layers_)
            if (auto* delta = std::get_if<DeltaNetLayer>(&layer.attention))
                delta->reset();
        std::fill(ple_state_.begin(), ple_state_.end(), 0.f);
        recent_.clear();
    }

    ModelState take_state() override {
        ModelState state;
        state.layers.reserve(layers_.size());
        for (auto& layer : layers_) {
            if (auto* delta = std::get_if<DeltaNetLayer>(&layer.attention))
                state.layers.emplace_back(delta->take_state(compute_));
            else {
                auto& attention = std::get<QsaAttention>(layer.attention);
                state.layers.emplace_back(
                    QsaState{
                        attention.attention.take_state(),
                        std::move(attention.raw_keys),
                        std::move(attention.pooled)
                    }
                );
            }
        }
        state.ple = std::move(ple_state_);
        state.recent = std::move(recent_);
        return state;
    }

    void restore_state(ModelState state) override {
        need(state.layers.size() == layers_.size(), "gguf_text_snapshot_layers");
        for (std::size_t i = 0; i < layers_.size(); ++i) {
            auto& layer = layers_[i];
            if (auto* delta = std::get_if<DeltaNetLayer>(&layer.attention))
                delta->restore_state(
                    std::get<RecurrentState>(std::move(state.layers[i]))
                );
            else {
                auto& attention = std::get<QsaAttention>(layer.attention);
                auto saved = std::get<QsaState>(std::move(state.layers[i]));
                need(
                    saved.raw_keys.size() == context_ * index_dim
                        && saved.pooled.size() == context_ / pool * index_dim,
                    "gguf_text_qsa_snapshot_shape"
                );
                attention.attention.restore_state(std::move(saved.attention));
                attention.raw_keys.swap(saved.raw_keys);
                attention.pooled.swap(saved.pooled);
            }
        }
        need(
            state.ple.size() == ple_history * wide && state.recent.size() <= ngram - 1,
            "gguf_text_ple_snapshot_shape"
        );
        ple_state_.swap(state.ple);
        recent_.swap(state.recent);
    }

    Vec forward(
        std::span<const unsigned> tokens,
        std::size_t position,
        bool logits,
        const Progress& progress
    ) override {
        const auto rows = tokens.size();
        need(rows && position + rows <= context_, "qwen4exp_context_exceeded");
        need(
            position == 0
                || recent_.size() == std::min<std::size_t>(position, ngram - 1),
            "qwen4exp_history"
        );
        Vec embedded(rows * hidden);
        for (std::size_t row = 0; row < rows; ++row) {
            need(tokens[row] < vocabulary_size, "qwen4exp_token");
            w_.rows(
                "token_embd.weight",
                tokens[row],
                std::span(embedded).subspan(row * hidden, hidden)
            );
        }
        // The wide residual starts as identical copies of the embedding.
        Vec residual(rows * wide);
        for (std::size_t row = 0; row < rows; ++row)
            for (std::size_t s = 0; s < streams; ++s)
                std::copy_n(
                    embedded.begin() + row * hidden,
                    hidden,
                    residual.begin() + row * wide + s * hidden
                );
        for (std::size_t i = 0; i < layers; ++i) {
            auto& layer = layers_[i];
            if (i == ple_layer)
                ple(tokens, residual);
            Vec inject;
            auto x = mix(layer.attention_mixer, residual, &inject);
            auto* attention = std::get_if<QsaAttention>(&layer.attention);
            const auto mixed =
                attention
                    ? attend_layer(i, *attention, x, position)
                    : std::get<DeltaNetLayer>(layer.attention).forward(compute_, w_, x);
            combine(residual, mixed, inject);
            x = mix(layer.ffn_mixer, residual, &inject);
            combine(residual, layer.moe.forward(compute_, w_, x), inject);
            if (progress)
                progress("layer", i + 1);
        }
        for (const auto token : tokens) {
            recent_.push_back(token);
            if (recent_.size() > ngram - 1)
                recent_.erase(recent_.begin());
        }
        if (!logits)
            return {};
        Vec last(residual.end() - wide, residual.end());
        return w_.project(compute_, "output.weight", mix(head_, last, nullptr));
    }

private:
    const GgufWeights& w_;
    DenseCompute& compute_;
    std::size_t context_;
    Mixer head_;
    std::vector<Layer> layers_;
    std::vector<std::uint64_t> multipliers_;
    std::array<std::uint64_t, ple_heads> offsets_{}, sizes_{};
    Vec ple_norm_key_, ple_norm_query_, ple_norm_conv_, ple_conv_, ple_state_;
    std::vector<unsigned> recent_;

    Mixer mixer(const std::string& p) const {
        return {
            p + "down.weight",
            p + "up.weight",
            w_.read(p + "norm.weight"),
            w_.read(p + "inject.weight")
        };
    }

    // RMSNorm over each stream, scaled by that stream's slice of the gains.
    static void grouped_norm(std::span<float> x, std::span<const float> gains) {
        rms_norm(x, hidden, eps);
        for (std::size_t i = 0; i < x.size(); ++i)
            x[i] *= gains[i % wide];
    }

    Vec mix(const Mixer& m, std::span<const float> residual, Vec* inject) const {
        const auto rows = residual.size() / wide;
        Vec normalized(residual.begin(), residual.end());
        grouped_norm(normalized, m.norm);
        auto low = w_.project(compute_, m.down, normalized);
        for (auto& v : low)
            v = silu(v / float(streams));
        const auto gate = w_.project(compute_, m.up, low);
        Vec mixed(rows * hidden);
        for (std::size_t row = 0; row < rows; ++row)
            for (std::size_t s = 0; s < streams; ++s)
                for (std::size_t d = 0; d < hidden; ++d) {
                    const auto at = row * wide + s * hidden + d;
                    mixed[row * hidden + d] += normalized[at] * sigmoid(gate[at]);
                }
        for (auto& v : mixed)
            v /= float(streams);
        if (inject) {
            inject->assign(rows * streams, 0.f);
            for (std::size_t row = 0; row < rows; ++row)
                for (std::size_t s = 0; s < streams; ++s) {
                    float sum = 0;
                    for (std::size_t c = 0; c < wide; ++c)
                        sum += m.inject[s * wide + c] * normalized[row * wide + c];
                    (*inject)[row * streams + s] = sum;
                }
        }
        return mixed;
    }

    // 2 * sigmoid centres the scatter weights on one.
    static void combine(
        Vec& residual, std::span<const float> block, std::span<const float> inject
    ) {
        const auto rows = block.size() / hidden;
        for (std::size_t row = 0; row < rows; ++row)
            for (std::size_t s = 0; s < streams; ++s) {
                const float weight =
                    2 * sigmoid(inject[row * streams + s] / float(streams));
                for (std::size_t d = 0; d < hidden; ++d)
                    residual[row * wide + s * hidden + d] +=
                        block[row * hidden + d] * weight;
            }
    }

    // N-gram hash rows for one token; an EOS or a missing predecessor cuts the
    // window, and the token's own EOS does not cut its context.
    std::array<std::uint64_t, ple_heads>
    ple_rows(unsigned token, std::span<const unsigned> before) const {
        std::array<std::uint64_t, ngram> context{token};
        bool cut = false;
        for (std::size_t s = 1; s < ngram; ++s) {
            const bool present = s <= before.size();
            const unsigned t = present ? before[before.size() - s] : ple_eos;
            cut = cut || !present || t == ple_eos;
            context[s] = cut ? ple_eos : t;
        }
        std::array<std::uint64_t, ple_heads> rows{};
        for (std::size_t n = 2; n <= ngram; ++n) {
            std::uint64_t mixed = context[0] * multipliers_[0];
            for (std::size_t j = 1; j < n; ++j)
                mixed ^= context[j] * multipliers_[j];
            for (std::size_t g = 0; g < ple_per_gram; ++g) {
                const auto h = (n - 2) * ple_per_gram + g;
                rows[h] = mixed % sizes_[h] + offsets_[h];
            }
        }
        return rows;
    }

    void ple(std::span<const unsigned> tokens, Vec& residual) {
        const auto rows = tokens.size();
        const auto p = prefix(ple_layer);
        std::vector<unsigned> history = recent_;
        Vec embedding(rows * ple_heads * ple_dim);
        for (std::size_t row = 0; row < rows; ++row) {
            const auto indices = ple_rows(tokens[row], history);
            for (std::size_t h = 0; h < ple_heads; ++h)
                w_.rows(
                    "per_layer_token_embd.weight",
                    indices[h],
                    std::span(embedding).subspan(
                        (row * ple_heads + h) * ple_dim, ple_dim
                    )
                );
            history.push_back(tokens[row]);
        }
        auto key = w_.project(compute_, p + "ple_key.weight", embedding);
        const auto value = w_.project(compute_, p + "ple_value.weight", embedding);
        grouped_norm(key, ple_norm_key_);
        Vec query(residual);
        grouped_norm(query, ple_norm_query_);
        Vec gated(rows * wide);
        for (std::size_t row = 0; row < rows; ++row)
            for (std::size_t s = 0; s < streams; ++s) {
                float dot = 0;
                for (std::size_t d = 0; d < hidden; ++d) {
                    const auto at = row * wide + s * hidden + d;
                    dot += key[at] * query[at];
                }
                dot /= std::sqrt(float(hidden));
                const float sign = float((dot > 0) - (dot < 0));
                const float gate =
                    sigmoid(sign * std::sqrt(std::max(std::abs(dot), 1e-6f)));
                for (std::size_t d = 0; d < hidden; ++d)
                    gated[row * wide + s * hidden + d] = value[row * hidden + d] * gate;
            }
        Vec normalized(gated);
        grouped_norm(normalized, ple_norm_conv_);
        // Depthwise causal convolution dilated by the n-gram size:
        // out[c, t] = sum_k w[k, c] * x[c, t - (K - 1 - k) * n].
        for (std::size_t row = 0; row < rows; ++row) {
            w_.check_cancel();
            const auto* x = normalized.data() + row * wide;
            for (std::size_t c = 0; c < wide; ++c) {
                float sum = x[c] * ple_conv_[c * ple_kernel + ple_kernel - 1];
                for (std::size_t k = 0; k + 1 < ple_kernel; ++k) {
                    const auto back = (ple_kernel - 1 - k) * ngram;
                    sum += ple_state_[(ple_history - back) * wide + c]
                           * ple_conv_[c * ple_kernel + k];
                }
                residual[row * wide + c] += gated[row * wide + c] + silu(sum);
            }
            std::move(ple_state_.begin() + wide, ple_state_.end(), ple_state_.begin());
            std::copy(x, x + wide, ple_state_.end() - wide);
        }
    }

    // QSA: blocks of `pool` positions scored by one pooled key, the mean of its
    // raw indexer keys normed and rotated to its first member. Up to
    // index_budget / pool visible blocks every position is attended.
    Vec attend_layer(
        std::size_t layer,
        QsaAttention& state,
        std::span<const float> x,
        std::size_t position
    ) {
        const auto rows = x.size() / hidden;
        const auto p = prefix(layer);
        auto& raw = state.raw_keys;
        auto& pooled = state.pooled;
        const auto keys = w_.project(compute_, p + "indexer.k_proj.weight", x);
        std::copy(keys.begin(), keys.end(), raw.begin() + position * index_dim);
        for (std::size_t at = position; at < position + rows; ++at) {
            if ((at + 1) % pool)
                continue;
            const auto block = (at + 1) / pool - 1;
            auto mean = std::span(pooled).subspan(block * index_dim, index_dim);
            std::fill(mean.begin(), mean.end(), 0.f);
            for (std::size_t m = 0; m < pool; ++m)
                for (std::size_t d = 0; d < index_dim; ++d)
                    mean[d] += raw[(block * pool + m) * index_dim + d];
            for (auto& v : mean)
                v /= float(pool);
            rms_norm(mean, state.index_k_norm, eps);
            rope(
                mean,
                index_dim,
                attention_spec.rotary,
                block * pool,
                attention_spec.base
            );
        }
        std::vector<std::vector<std::size_t>> sparse(rows);
        const auto budget = index_budget / pool;
        if ((position + rows) / pool > budget) {
            auto q = w_.project(compute_, p + "indexer.q_proj.weight", x);
            rms_norm(q, state.index_q_norm, eps);
            for (std::size_t row = 0; row < rows; ++row) {
                const auto at = position + row, visible = (at + 1) / pool;
                if (visible <= budget)
                    continue;
                auto query = std::span(q).subspan(
                    row * index_heads * index_dim, index_heads * index_dim
                );
                rope(query, index_dim, attention_spec.rotary, at, attention_spec.base);
                Vec score(visible, 0.f);
                for (std::size_t b = 0; b < visible; ++b)
                    for (std::size_t h = 0; h < index_heads; ++h) {
                        float dot = 0;
                        for (std::size_t d = 0; d < index_dim; ++d)
                            dot += query[h * index_dim + d] * pooled[b * index_dim + d];
                        score[b] += std::max(dot, 0.f) / std::sqrt(float(index_dim));
                    }
                std::vector<std::size_t> order(visible);
                std::iota(order.begin(), order.end(), 0);
                std::partial_sort(
                    order.begin(),
                    order.begin() + budget,
                    order.end(),
                    [&](auto a, auto b) { return score[a] > score[b]; }
                );
                auto& selected = sparse[row];
                for (std::size_t i = 0; i < budget; ++i)
                    for (std::size_t m = 0; m < pool; ++m)
                        selected.push_back(order[i] * pool + m);
                for (std::size_t k = 0; k < (at + 1) % pool; ++k)
                    selected.push_back(at - k);
            }
        }
        return state.attention.forward(compute_, w_, x, position, &sparse);
    }
};
}  // namespace

void validate_qwen4exp(const checkpoint::GgufMetadata& m) {
    validate(m);
}

void validate_qwen4exp(const checkpoint::GgufCheckpoint& m) {
    validate(m);
}

void bind_qwen4exp(GgufWeights& w) {
    validate_qwen4exp(w.gguf());
    w.matrix("token_embd.weight", vocabulary_size, hidden);
    w.matrix("output.weight", vocabulary_size, hidden);
    w.vector("output_hc_norm.weight", {wide});
    w.matrix("output_hc_down.weight", low_rank, wide);
    w.matrix("output_hc_up.weight", wide, low_rank);
    w.table("per_layer_token_embd.weight", ple_table_rows, ple_dim);
    for (std::size_t i = 0; i < layers; ++i) {
        const auto p = prefix(i);
        for (const auto* tag : {"hc_attn_", "hc_ffn_"}) {
            const auto m = p + tag;
            w.vector(m + "norm.weight", {wide});
            w.matrix(m + "down.weight", low_rank, wide);
            w.matrix(m + "up.weight", wide, low_rank);
            w.vector(m + "inject.weight", {wide, streams});
        }
        if (full_attention(i)) {
            bind_attention(w, p, attention_spec);
            w.matrix(p + "indexer.q_proj.weight", index_heads * index_dim, hidden);
            w.matrix(p + "indexer.k_proj.weight", index_dim, hidden);
            w.vector(p + "indexer.q_norm.weight", {index_dim});
            w.vector(p + "indexer.k_norm.weight", {index_dim});
        } else
            bind_delta_net(w, p, delta_spec);
        if (i == ple_layer) {
            w.matrix(p + "ple_key.weight", wide, hidden);
            w.matrix(p + "ple_value.weight", hidden, hidden);
            w.vector(p + "ple_norm_key.weight", {wide});
            w.vector(p + "ple_norm_query.weight", {wide});
            w.vector(p + "ple_norm_conv.weight", {wide});
            w.vector(p + "ple_conv1d.weight", {ple_kernel, wide});
        }
        bind_moe(w, p, moe_spec);
    }
    w.finish();
}

std::size_t qwen4exp_state_bytes(std::size_t context) {
    return 36 * DeltaNetLayer::state_bytes(delta_spec)
           + 12
                 * (AttentionLayer::state_bytes(attention_spec, context)
                    + (context + context / pool) * index_dim * sizeof(float))
           + ple_history * wide * sizeof(float);
}

std::unique_ptr<Model>
qwen4exp_model(const GgufWeights& w, DenseCompute& compute, std::size_t context) {
    return std::make_unique<Qwen4Exp>(w, compute, context);
}
}  // namespace kadan::text
