// GLM-5.3 Flash (GGUF glm5-next): llama.cpp src/models/glm5-next.cpp at
// 6420cf5dd588c68e48657a0bbd82bb3bd5ac7209. mHC streams, KDA linear layers and
// nope-only MLA with the k-pool sparse indexer; MLA is absorbed so only the
// 512-wide latent is cached. Text only; the MTP block is not executed.
#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>
#include <string>

#include "gguf_model.hpp"

namespace kadan::text {
namespace {
constexpr std::size_t hidden = 4096, layers = 45, blocks = 46, streams = 4,
                      wide = streams * hidden, mixes = (2 + streams) * streams,
                      sinkhorn = 20, heads = 64, kda_dim = 128, inner = heads * kda_dim,
                      conv = 4, q_rank = 1536, latent = 512, nope = 256,
                      value_dim = 256, experts = 288, top_k = 8, dense_layers = 3,
                      ff = 12288, expert_ff = 2048, index_heads = 32, index_dim = 128,
                      pool = 4, index_budget = 2048, vocabulary_size = 154880,
                      context_length = 1048576;
constexpr float eps = 1e-5f, layer_norm_eps = 1e-6f, hc_eps = 1e-6f, l2_eps = 1e-6f,
                gate_bound = -5, routed_scale = 2.5f, clamp = 10;

bool sparse_attention(std::size_t layer) {
    return layer % 4 == 3;
}

std::string prefix(std::size_t layer) {
    return "blk." + std::to_string(layer) + ".";
}

template <class Metadata> void validate(const Metadata& m) {
    need(m.string("general.architecture") == "glm5-next", "glm_architecture");
    for (const auto& [key, expected] :
         std::initializer_list<std::pair<const char*, std::uint64_t>>{
             {"block_count", blocks},
             {"context_length", context_length},
             {"embedding_length", hidden},
             {"feed_forward_length", ff},
             {"attention.head_count", heads},
             {"expert_count", experts},
             {"expert_used_count", top_k},
             {"expert_group_count", 1},
             {"expert_group_used_count", 1},
             {"expert_gating_func", 2},
             {"vocab_size", vocabulary_size},
             {"attention.q_lora_rank", q_rank},
             {"attention.kv_lora_rank", latent},
             {"rope.dimension_count", 0},
             {"attention.key_length", latent},
             {"attention.value_length", latent},
             {"attention.key_length_mla", nope},
             {"attention.value_length_mla", value_dim},
             {"ssm.conv_kernel", conv},
             {"kda.head_dim", kda_dim},
             {"attention.indexer.head_count", index_heads},
             {"attention.indexer.key_length", index_dim},
             {"attention.indexer.top_k", index_budget},
             {"attention.indexer.kpool", pool},
             {"hyper_connection.count", streams},
             {"hyper_connection.sinkhorn_iterations", sinkhorn},
             {"expert_feed_forward_length", expert_ff},
             {"expert_shared_feed_forward_length", expert_ff},
             {"expert_shared_count", 1},
             {"leading_dense_block_count", dense_layers},
             {"nextn_predict_layers", 1}
         })
        need(m.integer(std::string("glm5-next.") + key) == expected, "glm_dimensions");
    for (const auto& [key, expected] :
         std::initializer_list<std::pair<const char*, double>>{
             {"attention.layer_norm_rms_epsilon", eps},
             {"attention.layer_norm_epsilon", layer_norm_eps},
             {"kda.gate_lower_bound", gate_bound},
             {"hyper_connection.epsilon", hc_eps},
             {"expert_weights_scale", routed_scale}
         })
        need(
            std::abs(m.number(std::string("glm5-next.") + key) - expected)
                <= std::abs(expected) * 1e-6,
            "glm_arithmetic"
        );
    need(m.boolean("glm5-next.expert_weights_norm"), "glm_routing");
    const auto kv = m.integers("glm5-next.attention.head_count_kv");
    need(kv.size() == blocks, "glm_layer_schedule");
    for (std::size_t i = 0; i < blocks; ++i)
        need(
            kv[i] == ((i == layers || sparse_attention(i)) ? 1 : 0),
            "glm_layer_schedule"
        );
    for (const auto* key :
         {"glm5-next.swiglu_clamp_exp", "glm5-next.swiglu_clamp_shexp"}) {
        const auto limits = m.numbers(key);
        need(
            limits.size() == blocks
                && std::all_of(
                    limits.begin(), limits.end(), [](double v) { return v == clamp; }
                ),
            "glm_swiglu_clamp"
        );
    }
    need(
        !m.has("glm5-next.attention.indexer.types")
            && m.string("tokenizer.ggml.model") == "gpt2"
            && m.string("tokenizer.ggml.pre") == "glm4"
            && m.integer("tokenizer.ggml.eos_token_id") == 154820
            && m.integer("tokenizer.ggml.eot_token_id") == 154827
            && m.integer("tokenizer.ggml.eom_token_id") == 154829,
        "glm_contract"
    );
}

float clamped_swiglu(float gate, float up) {
    return silu(std::min(gate, clamp)) * std::clamp(up, -clamp, clamp);
}

struct Mixing {
    std::array<float, streams> pre, post;
    std::array<float, streams * streams> comb;
};

class Glm final : public Model {
public:
    Glm(const GgufWeights& weights, DenseCompute& compute, std::size_t context)
        : w_(weights), compute_(compute), context_(context),
          output_norm_(weights.read("output_norm.weight")) {
        need(context && context <= context_length, "glm_context");
        layers_.resize(layers);
        for (std::size_t i = 0; i < layers; ++i) {
            const auto p = prefix(i);
            auto& l = layers_[i];
            for (std::size_t m = 0; m < 2; ++m) {
                const std::string tag = m ? "hc_ffn_" : "hc_attn_";
                l.hc_base[m] = w_.read(p + tag + "base.weight");
                l.hc_scale[m] = w_.read(p + tag + "scale.weight");
            }
            l.attn_norm = w_.read(p + "attn_norm.weight");
            l.ffn_norm = w_.read(p + "ffn_norm.weight");
            if (i >= dense_layers)
                l.bias = w_.read(p + "exp_probs_b.bias");
            if (sparse_attention(i)) {
                l.q_norm = w_.read(p + "attn_q_a_norm.weight");
                l.kv_norm = w_.read(p + "attn_kv_a_norm.weight");
                l.k_b = w_.decode(p + "attn_k_b.weight");
                l.v_b = w_.decode(p + "attn_v_b.weight");
                l.index_scale = w_.read(p + "indexer.k_norm.weight");
                l.index_bias = w_.read(p + "indexer.k_norm.bias");
                l.ape = w_.read(p + "indexer_compressor_ape.weight");
                l.latents.resize(context * latent);
                l.index_keys.resize(context * index_dim);
                l.index_gates.resize(context * index_dim);
                l.pooled.resize(context / pool * index_dim);
            } else {
                for (std::size_t c = 0; c < 3; ++c)
                    l.conv_weight[c] =
                        w_.read(p + "ssm_conv1d_" + "qkv"[c] + ".weight");
                l.dt = w_.read(p + "ssm_dt.bias");
                l.a = w_.read(p + "ssm_a");
                l.o_norm = w_.read(p + "ssm_norm.weight");
                l.conv.resize(3 * (conv - 1) * inner);
                l.state.resize(heads * kda_dim * kda_dim);
            }
        }
    }

    std::size_t vocabulary() const override {
        return vocabulary_size;
    }

    bool eos(unsigned token) const override {
        return token == 154820 || token == 154827 || token == 154829;
    }

    void reset() override {
        for (auto& l : layers_) {
            std::fill(l.conv.begin(), l.conv.end(), 0.f);
            std::fill(l.state.begin(), l.state.end(), 0.f);
        }
    }

    Vec forward(
        std::span<const unsigned> tokens,
        std::size_t position,
        bool logits,
        const Progress& progress
    ) override {
        const auto rows = tokens.size();
        need(rows && position + rows <= context_, "glm_context_exceeded");
        Vec x(rows * hidden), residual(rows * wide);
        for (std::size_t row = 0; row < rows; ++row) {
            need(tokens[row] < vocabulary_size, "glm_token");
            w_.rows(
                "token_embd.weight",
                tokens[row],
                std::span(x).subspan(row * hidden, hidden)
            );
            for (std::size_t s = 0; s < streams; ++s)
                std::copy_n(
                    x.begin() + row * hidden,
                    hidden,
                    residual.begin() + row * wide + s * hidden
                );
        }
        for (std::size_t i = 0; i < layers; ++i) {
            const auto p = prefix(i);
            auto& l = layers_[i];
            std::vector<Mixing> mixing;
            auto h = pre(i, 0, residual, mixing);
            rms_norm(h, l.attn_norm, eps);
            post(
                residual, sparse_attention(i) ? dsa(i, h, position) : kda(i, h), mixing
            );
            h = pre(i, 1, residual, mixing);
            rms_norm(h, l.ffn_norm, eps);
            post(residual, i < dense_layers ? dense(p, h) : moe(i, h), mixing);
            if (progress)
                progress("layer", i + 1);
        }
        if (!logits)
            return {};
        // Collapse the streams by their mean before the output norm.
        Vec last(hidden);
        const auto* row = residual.data() + (rows - 1) * wide;
        for (std::size_t d = 0; d < hidden; ++d) {
            float sum = 0;
            for (std::size_t s = 0; s < streams; ++s)
                sum += row[s * hidden + d];
            last[d] = sum / float(streams);
        }
        rms_norm(last, output_norm_, eps);
        return w_.project(compute_, "output.weight", last);
    }

private:
    struct Layer {
        std::array<Vec, 2> hc_base, hc_scale;
        std::array<Vec, 3> conv_weight;
        Vec attn_norm, ffn_norm, bias, q_norm, kv_norm, k_b, v_b, index_scale,
            index_bias, ape, dt, a, o_norm, conv, state, latents, index_keys,
            index_gates, pooled;
    };

    const GgufWeights& w_;
    DenseCompute& compute_;
    std::size_t context_;
    Vec output_norm_;
    std::vector<Layer> layers_;

    // mHC pre-mix: sigmoid stream weights, 2 * sigmoid post weights and a
    // Sinkhorn-normalized [src][dst] combination, as build_hc_pre.
    Vec
    pre(std::size_t layer,
        std::size_t which,
        std::span<const float> residual,
        std::vector<Mixing>& mixing) {
        const auto rows = residual.size() / wide;
        const auto& l = layers_[layer];
        const auto& base = l.hc_base[which];
        const auto& scale = l.hc_scale[which];
        Vec flat(residual.begin(), residual.end());
        rms_norm(flat, wide, eps);
        const auto coefficients = w_.project(
            compute_,
            prefix(layer) + (which ? "hc_ffn_fn.weight" : "hc_attn_fn.weight"),
            flat
        );
        mixing.assign(rows, {});
        Vec input(rows * hidden);
        for (std::size_t row = 0; row < rows; ++row) {
            const auto* c = coefficients.data() + row * mixes;
            auto& m = mixing[row];
            for (std::size_t s = 0; s < streams; ++s) {
                m.pre[s] = sigmoid(c[s] * scale[0] + base[s]) + hc_eps;
                m.post[s] = 2 * sigmoid(c[streams + s] * scale[1] + base[streams + s]);
            }
            auto& comb = m.comb;
            for (std::size_t src = 0; src < streams; ++src) {
                float maximum = -INFINITY;
                for (std::size_t dst = 0; dst < streams; ++dst) {
                    const auto k = src * streams + dst;
                    comb[k] = c[2 * streams + k] * scale[2] + base[2 * streams + k];
                    maximum = std::max(maximum, comb[k]);
                }
                float total = 0;
                for (std::size_t dst = 0; dst < streams; ++dst)
                    total += comb[src * streams + dst] =
                        std::exp(comb[src * streams + dst] - maximum);
                for (std::size_t dst = 0; dst < streams; ++dst)
                    comb[src * streams + dst] =
                        comb[src * streams + dst] / total + hc_eps;
            }
            const auto normalize = [&](bool columns) {
                for (std::size_t a = 0; a < streams; ++a) {
                    float total = hc_eps;
                    for (std::size_t b = 0; b < streams; ++b)
                        total += comb[columns ? b * streams + a : a * streams + b];
                    for (std::size_t b = 0; b < streams; ++b)
                        comb[columns ? b * streams + a : a * streams + b] /= total;
                }
            };
            normalize(true);
            for (std::size_t iteration = 1; iteration < sinkhorn; ++iteration) {
                normalize(false);
                normalize(true);
            }
            for (std::size_t s = 0; s < streams; ++s)
                for (std::size_t d = 0; d < hidden; ++d)
                    input[row * hidden + d] +=
                        m.pre[s] * residual[row * wide + s * hidden + d];
        }
        return input;
    }

    static void post(
        Vec& residual, std::span<const float> block, const std::vector<Mixing>& mixing
    ) {
        const auto rows = block.size() / hidden;
        Vec next(residual.size());
        for (std::size_t row = 0; row < rows; ++row) {
            const auto& m = mixing[row];
            for (std::size_t dst = 0; dst < streams; ++dst)
                for (std::size_t d = 0; d < hidden; ++d) {
                    float value = block[row * hidden + d] * m.post[dst];
                    for (std::size_t src = 0; src < streams; ++src)
                        value += residual[row * wide + src * hidden + d]
                                 * m.comb[src * streams + dst];
                    next[row * wide + dst * hidden + d] = value;
                }
        }
        residual.swap(next);
    }

    Vec kda(std::size_t layer, std::span<const float> h) {
        const auto rows = h.size() / hidden;
        const auto p = prefix(layer);
        auto& l = layers_[layer];
        std::array<Vec, 3> projected{
            w_.project(compute_, p + "attn_q.weight", h),
            w_.project(compute_, p + "attn_k.weight", h),
            w_.project(compute_, p + "attn_v.weight", h)
        };
        const auto f = w_.project(
            compute_,
            p + "ssm_f_b.weight",
            w_.project(compute_, p + "ssm_f_a.weight", h)
        );
        const auto g = w_.project(
            compute_,
            p + "ssm_g_b.weight",
            w_.project(compute_, p + "ssm_g_a.weight", h)
        );
        const auto beta = w_.project(compute_, p + "ssm_beta.weight", h);
        const float scale = 1 / std::sqrt(float(kda_dim));
        Vec output(rows * inner), query(kda_dim), decay(kda_dim), head(kda_dim);
        std::array<Vec, 3> mixed;
        for (auto& v : mixed)
            v.resize(inner);
        for (std::size_t row = 0; row < rows; ++row) {
            w_.check_cancel();
            for (std::size_t which = 0; which < 3; ++which) {
                const auto* x = projected[which].data() + row * inner;
                auto* history = l.conv.data() + which * (conv - 1) * inner;
                const auto& kernel = l.conv_weight[which];
                for (std::size_t c = 0; c < inner; ++c) {
                    float sum = x[c] * kernel[c * conv + conv - 1];
                    for (std::size_t k = 0; k + 1 < conv; ++k)
                        sum += history[k * inner + c] * kernel[c * conv + k];
                    mixed[which][c] = silu(sum);
                }
                std::move(history + inner, history + (conv - 1) * inner, history);
                std::copy(x, x + inner, history + (conv - 2) * inner);
            }
            l2_norm(mixed[0], kda_dim, l2_eps);
            l2_norm(mixed[1], kda_dim, l2_eps);
            for (std::size_t hd = 0; hd < heads; ++hd) {
                for (std::size_t d = 0; d < kda_dim; ++d) {
                    const auto at = hd * kda_dim + d;
                    query[d] = mixed[0][at] * scale;
                    // ssm_a holds -exp(A_log): g = bound * sigmoid(-(f + dt) * a).
                    decay[d] = std::exp(
                        gate_bound
                        * sigmoid(-(f[row * inner + at] + l.dt[at]) * l.a[hd])
                    );
                }
                delta_rule(
                    std::span(l.state).subspan(
                        hd * kda_dim * kda_dim, kda_dim * kda_dim
                    ),
                    query,
                    std::span<const float>(mixed[1]).subspan(hd * kda_dim, kda_dim),
                    std::span<const float>(mixed[2]).subspan(hd * kda_dim, kda_dim),
                    decay,
                    sigmoid(beta[row * heads + hd]),
                    head
                );
                rms_norm(head, l.o_norm, eps);
                for (std::size_t d = 0; d < kda_dim; ++d)
                    output[row * inner + hd * kda_dim + d] =
                        head[d] * sigmoid(g[row * inner + hd * kda_dim + d]);
            }
        }
        return w_.project(compute_, p + "attn_output.weight", output);
    }

    Vec dsa(std::size_t layer, std::span<const float> h, std::size_t position) {
        const auto rows = h.size() / hidden;
        const auto p = prefix(layer);
        auto& l = layers_[layer];
        auto qr = w_.project(compute_, p + "attn_q_a.weight", h);
        rms_norm(qr, l.q_norm, eps);
        const auto q = w_.project(compute_, p + "attn_q_b.weight", qr);
        auto kv = w_.project(compute_, p + "attn_kv_a_mqa.weight", h);
        rms_norm(kv, l.kv_norm, eps);
        std::copy(kv.begin(), kv.end(), l.latents.begin() + position * latent);
        // Indexer keys use LayerNorm with bias; pool gates select within a block.
        auto keys = w_.project(compute_, p + "indexer.attn_k.weight", h);
        for (std::size_t row = 0; row < rows; ++row) {
            auto key = std::span(keys).subspan(row * index_dim, index_dim);
            const float mean =
                std::accumulate(key.begin(), key.end(), 0.f) / float(index_dim);
            float variance = 0;
            for (auto v : key)
                variance += (v - mean) * (v - mean);
            const float factor =
                1 / std::sqrt(variance / float(index_dim) + layer_norm_eps);
            for (std::size_t d = 0; d < index_dim; ++d)
                key[d] = (key[d] - mean) * factor * l.index_scale[d] + l.index_bias[d];
        }
        const auto gates =
            w_.project(compute_, p + "indexer_compressor_gate.weight", h);
        std::copy(
            keys.begin(), keys.end(), l.index_keys.begin() + position * index_dim
        );
        std::copy(
            gates.begin(), gates.end(), l.index_gates.begin() + position * index_dim
        );
        for (std::size_t at = position; at < position + rows; ++at) {
            if ((at + 1) % pool)
                continue;
            const auto block = (at + 1) / pool - 1;
            for (std::size_t d = 0; d < index_dim; ++d) {
                std::array<float, pool> logit;
                for (std::size_t m = 0; m < pool; ++m)
                    logit[m] = l.index_gates[(block * pool + m) * index_dim + d]
                               + l.ape[m * index_dim + d];
                const float maximum = *std::max_element(logit.begin(), logit.end());
                float total = 0;
                for (auto& v : logit)
                    total += v = std::exp(v - maximum);
                float value = 0;
                for (std::size_t m = 0; m < pool; ++m)
                    value += logit[m] / total
                             * l.index_keys[(block * pool + m) * index_dim + d];
                l.pooled[block * index_dim + d] = value;
            }
        }
        // Absorb wk_b: each head's query becomes a latent-space query.
        Vec absorbed(rows * heads * latent);
        for (std::size_t row = 0; row < rows; ++row)
            for (std::size_t hd = 0; hd < heads; ++hd) {
                const auto* qh = q.data() + (row * heads + hd) * nope;
                const auto* kb = l.k_b.data() + hd * latent * nope;
                auto* out = absorbed.data() + (row * heads + hd) * latent;
                for (std::size_t r = 0; r < latent; ++r) {
                    float sum = 0;
                    for (std::size_t n = 0; n < nope; ++n)
                        sum += kb[r * nope + n] * qh[n];
                    out[r] = sum;
                }
            }
        const float scale = 1 / std::sqrt(float(nope));
        Vec attended(rows * heads * latent);
        const auto budget = index_budget / pool;
        if ((position + rows) / pool <= budget)
            attend(
                compute_,
                absorbed,
                l.latents,
                l.latents,
                rows,
                position + rows,
                heads,
                1,
                latent,
                position,
                scale,
                attended,
                w_.cancel()
            );
        else {
            auto iq = w_.project(compute_, p + "indexer.attn_q_b.weight", qr);
            auto weights = w_.project(compute_, p + "indexer.proj.weight", h);
            for (auto& v : weights)
                v /= std::sqrt(float(index_dim * index_heads));
            for (std::size_t row = 0; row < rows; ++row) {
                const auto at = position + row, visible = (at + 1) / pool;
                std::vector<std::size_t> selected;
                if (visible <= budget)
                    for (std::size_t t = 0; t <= at; ++t)
                        selected.push_back(t);
                else {
                    Vec score(visible, 0.f);
                    for (std::size_t b = 0; b < visible; ++b)
                        for (std::size_t hd = 0; hd < index_heads; ++hd) {
                            float dot = 0;
                            for (std::size_t d = 0; d < index_dim; ++d)
                                dot += iq[(row * index_heads + hd) * index_dim + d]
                                       * l.pooled[b * index_dim + d];
                            score[b] +=
                                std::max(dot, 0.f) * weights[row * index_heads + hd];
                        }
                    std::vector<std::size_t> order(visible);
                    std::iota(order.begin(), order.end(), 0);
                    std::partial_sort(
                        order.begin(),
                        order.begin() + budget,
                        order.end(),
                        [&](auto a, auto b) { return score[a] > score[b]; }
                    );
                    for (std::size_t i = 0; i < budget; ++i)
                        for (std::size_t m = 0; m < pool; ++m)
                            selected.push_back(order[i] * pool + m);
                    for (std::size_t k = 0; k < (at + 1) % pool; ++k)
                        selected.push_back(at - k);
                }
                Vec gathered(selected.size() * latent);
                for (std::size_t i = 0; i < selected.size(); ++i)
                    std::copy_n(
                        l.latents.begin() + selected[i] * latent,
                        latent,
                        gathered.begin() + i * latent
                    );
                attend(
                    compute_,
                    std::span<const float>(absorbed).subspan(
                        row * heads * latent, heads * latent
                    ),
                    gathered,
                    gathered,
                    1,
                    selected.size(),
                    heads,
                    1,
                    latent,
                    selected.size() - 1,
                    scale,
                    std::span(attended).subspan(row * heads * latent, heads * latent),
                    w_.cancel()
                );
            }
        }
        // Expand each head's attended latent through wv_b.
        Vec output(rows * heads * value_dim);
        for (std::size_t row = 0; row < rows; ++row)
            for (std::size_t hd = 0; hd < heads; ++hd) {
                const auto* lat = attended.data() + (row * heads + hd) * latent;
                const auto* vb = l.v_b.data() + hd * value_dim * latent;
                for (std::size_t n = 0; n < value_dim; ++n) {
                    float sum = 0;
                    for (std::size_t r = 0; r < latent; ++r)
                        sum += vb[n * latent + r] * lat[r];
                    output[(row * heads + hd) * value_dim + n] = sum;
                }
            }
        return w_.project(compute_, p + "attn_output.weight", output);
    }

    Vec feed_forward(
        const std::string& gate_name,
        const std::string& up_name,
        const std::string& down_name,
        std::span<const float> h,
        std::size_t expert = 0
    ) {
        auto gate = w_.project(compute_, gate_name, h, expert);
        const auto up = w_.project(compute_, up_name, h, expert);
        for (std::size_t j = 0; j < gate.size(); ++j)
            gate[j] = clamped_swiglu(gate[j], up[j]);
        return w_.project(compute_, down_name, gate, expert);
    }

    Vec dense(const std::string& p, std::span<const float> h) {
        return feed_forward(
            p + "ffn_gate.weight", p + "ffn_up.weight", p + "ffn_down.weight", h
        );
    }

    // Sigmoid router with selection bias; selected probabilities are
    // renormalized and scaled, then summed in selection order.
    Vec moe(std::size_t layer, std::span<const float> h) {
        const auto rows = h.size() / hidden;
        const auto p = prefix(layer);
        const auto& bias = layers_[layer].bias;
        const auto logits = w_.project(compute_, p + "ffn_gate_inp.weight", h);
        std::vector<std::array<std::size_t, top_k>> chosen(rows);
        std::vector<std::array<float, top_k>> weight(rows);
        for (std::size_t row = 0; row < rows; ++row) {
            std::array<float, experts> probability;
            for (std::size_t e = 0; e < experts; ++e)
                probability[e] = sigmoid(logits[row * experts + e]);
            std::array<std::size_t, experts> order;
            std::iota(order.begin(), order.end(), 0);
            std::partial_sort(
                order.begin(), order.begin() + top_k, order.end(), [&](auto a, auto b) {
                    return probability[a] + bias[a] > probability[b] + bias[b];
                }
            );
            float sum = 0;
            for (std::size_t i = 0; i < top_k; ++i) {
                chosen[row][i] = order[i];
                sum += weight[row][i] = probability[order[i]];
            }
            sum = std::max(sum, 6.103515625e-5f);
            for (auto& v : weight[row])
                v = v / sum * routed_scale;
        }
        Vec contribution(rows * top_k * hidden);
        for (std::size_t e = 0; e < experts; ++e) {
            std::vector<std::pair<std::size_t, std::size_t>> routed;
            for (std::size_t row = 0; row < rows; ++row)
                for (std::size_t i = 0; i < top_k; ++i)
                    if (chosen[row][i] == e)
                        routed.emplace_back(row, i);
            if (routed.empty())
                continue;
            Vec gathered(routed.size() * hidden);
            for (std::size_t r = 0; r < routed.size(); ++r)
                std::copy_n(
                    h.begin() + routed[r].first * hidden,
                    hidden,
                    gathered.begin() + r * hidden
                );
            const auto down = feed_forward(
                p + "ffn_gate_exps.weight",
                p + "ffn_up_exps.weight",
                p + "ffn_down_exps.weight",
                gathered,
                e
            );
            for (std::size_t r = 0; r < routed.size(); ++r) {
                const auto [row, slot] = routed[r];
                for (std::size_t d = 0; d < hidden; ++d)
                    contribution[(row * top_k + slot) * hidden + d] =
                        down[r * hidden + d] * weight[row][slot];
            }
        }
        const auto shared = feed_forward(
            p + "ffn_gate_shexp.weight",
            p + "ffn_up_shexp.weight",
            p + "ffn_down_shexp.weight",
            h
        );
        Vec output(rows * hidden);
        for (std::size_t row = 0; row < rows; ++row)
            for (std::size_t d = 0; d < hidden; ++d) {
                float sum = 0;
                for (std::size_t i = 0; i < top_k; ++i)
                    sum += contribution[(row * top_k + i) * hidden + d];
                output[row * hidden + d] = sum + shared[row * hidden + d];
            }
        return output;
    }
};
}  // namespace

void validate_glm(const checkpoint::GgufMetadata& m) {
    validate(m);
}

void validate_glm(const checkpoint::GgufCheckpoint& m) {
    validate(m);
}

void bind_glm(GgufWeights& w) {
    validate_glm(w.gguf());
    w.matrix("token_embd.weight", vocabulary_size, hidden);
    w.matrix("output.weight", vocabulary_size, hidden);
    w.vector("output_norm.weight", {hidden});
    w.skip(prefix(layers));
    for (std::size_t i = 0; i < layers; ++i) {
        const auto p = prefix(i);
        for (const auto* tag : {"hc_attn_", "hc_ffn_"}) {
            w.matrix(p + tag + "fn.weight", mixes, wide);
            w.vector(p + tag + "base.weight", {mixes});
            w.vector(p + tag + "scale.weight", {3});
        }
        w.vector(p + "attn_norm.weight", {hidden});
        w.vector(p + "ffn_norm.weight", {hidden});
        if (sparse_attention(i)) {
            w.matrix(p + "attn_q_a.weight", q_rank, hidden);
            w.vector(p + "attn_q_a_norm.weight", {q_rank});
            w.matrix(p + "attn_q_b.weight", heads * nope, q_rank);
            w.matrix(p + "attn_kv_a_mqa.weight", latent, hidden);
            w.vector(p + "attn_kv_a_norm.weight", {latent});
            w.table(p + "attn_k_b.weight", latent, nope, heads);
            w.table(p + "attn_v_b.weight", value_dim, latent, heads);
            w.matrix(p + "attn_output.weight", hidden, heads * value_dim);
            w.matrix(p + "indexer.attn_q_b.weight", index_heads * index_dim, q_rank);
            w.matrix(p + "indexer.attn_k.weight", index_dim, hidden);
            w.vector(p + "indexer.k_norm.weight", {index_dim});
            w.vector(p + "indexer.k_norm.bias", {index_dim});
            w.matrix(p + "indexer.proj.weight", index_heads, hidden);
            w.matrix(p + "indexer_compressor_gate.weight", index_dim, hidden);
            w.vector(p + "indexer_compressor_ape.weight", {index_dim, pool});
        } else {
            for (const auto* tag : {"q", "k", "v"}) {
                w.matrix(p + "attn_" + tag + ".weight", inner, hidden);
                w.vector(p + "ssm_conv1d_" + tag + ".weight", {conv, 1, inner});
            }
            for (const auto* tag : {"f", "g"}) {
                w.matrix(p + "ssm_" + tag + "_a.weight", kda_dim, hidden);
                w.matrix(p + "ssm_" + tag + "_b.weight", inner, kda_dim);
            }
            w.vector(p + "ssm_dt.bias", {inner});
            w.vector(p + "ssm_a", {heads});
            w.matrix(p + "ssm_beta.weight", heads, hidden);
            w.vector(p + "ssm_norm.weight", {kda_dim});
            w.matrix(p + "attn_output.weight", hidden, inner);
        }
        if (i < dense_layers) {
            w.matrix(p + "ffn_gate.weight", ff, hidden);
            w.matrix(p + "ffn_up.weight", ff, hidden);
            w.matrix(p + "ffn_down.weight", hidden, ff);
        } else {
            w.matrix(p + "ffn_gate_inp.weight", experts, hidden);
            w.vector(p + "exp_probs_b.bias", {experts});
            w.matrix(p + "ffn_gate_exps.weight", expert_ff, hidden, experts);
            w.matrix(p + "ffn_up_exps.weight", expert_ff, hidden, experts);
            w.matrix(p + "ffn_down_exps.weight", hidden, expert_ff, experts);
            w.matrix(p + "ffn_gate_shexp.weight", expert_ff, hidden);
            w.matrix(p + "ffn_up_shexp.weight", expert_ff, hidden);
            w.matrix(p + "ffn_down_shexp.weight", hidden, expert_ff);
        }
    }
    w.finish();
}

std::size_t glm_state_bytes(std::size_t context) {
    const std::size_t linear = 34, sparse = 11;
    return sizeof(float)
           * (linear * (3 * (conv - 1) * inner + heads * kda_dim * kda_dim)
              + sparse
                    * (context * (latent + 2 * index_dim) + context / pool * index_dim
                       + 2 * heads * latent * nope));
}

std::unique_ptr<Model>
glm_model(const GgufWeights& w, DenseCompute& compute, std::size_t context) {
    return std::make_unique<Glm>(w, compute, context);
}
}  // namespace kadan::text
