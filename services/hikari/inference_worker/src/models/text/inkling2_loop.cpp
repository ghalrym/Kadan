#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "gguf_model.hpp"
#include "kadan/inkling_variant.hpp"

namespace kadan::text {
namespace {
constexpr std::size_t n_phys = 24;
constexpr std::size_t n_embd = 2304;
constexpr std::size_t n_heads = 24;
constexpr std::size_t n_kv_heads = 8;
constexpr std::size_t head_dim = 128;
constexpr std::size_t n_embd_k_gqa = n_kv_heads * head_dim;  // 1024
constexpr std::size_t n_embd_q = n_heads * head_dim;         // 3072
constexpr std::size_t n_rel = 16;
constexpr std::size_t n_rel_extent = 1024;
constexpr std::size_t n_swa_window = 1024;
constexpr std::size_t n_block_size = 6;
constexpr std::size_t n_conv = 4;
constexpr std::size_t n_conv_hist = n_conv - 1;  // 3
constexpr std::size_t n_conv_channels = 2 * n_embd_k_gqa + 2 * n_embd;  // 6656
constexpr std::size_t n_vocab_padded = 201024;
constexpr std::size_t n_vocab_unpadded = 200058;
constexpr std::size_t n_experts = 128;
constexpr std::size_t n_experts_used = 8;
constexpr std::size_t n_ff_exp = 1152;
constexpr std::size_t loop_start = 6;
constexpr std::size_t loop_end = 18;
constexpr std::size_t loop_core_len = loop_end - loop_start;  // 12
constexpr std::size_t loop_passes = 3;
constexpr std::size_t loop_gate_rank = 64;
constexpr std::size_t loop_ctrl_count = 4;  // q, k, v, ffn
constexpr std::size_t n_slots = n_phys + (loop_passes - 1) * loop_core_len;  // 48
constexpr unsigned eos_token = 200006;

constexpr float rms_eps = 1e-6f;
constexpr float qk_clip_threshold = 22.627416997969522f;  // 16 * sqrt(2)
constexpr float rel_clip_threshold = 8.0f;
constexpr float log_scale_floor = 8192.0f;
constexpr float log_scale_alpha = 0.1f;
constexpr float expert_weights_scale = 8.0f;
constexpr float expert_weights_norm_eps = 1e-20f;
constexpr float situ_beta = 4.0f;
constexpr float situ_linear_beta = 25.0f;
constexpr float final_logit_softcap = 30.0f;

constexpr bool is_swa_phys(std::size_t ip) {
    return (ip % n_block_size) != (n_block_size - 1);
}

constexpr std::size_t slot_phys(std::size_t slot) {
    return slot < n_phys ? slot : loop_start + (slot - n_phys) % loop_core_len;
}

constexpr std::size_t slot_kv_capacity(std::size_t slot, std::size_t context) {
    return is_swa_phys(slot_phys(slot)) ? std::min(context, n_swa_window) : context;
}

std::string blk(std::size_t i, const char* suffix) {
    return "blk." + std::to_string(i) + "." + suffix;
}

constexpr std::array<const char*, loop_ctrl_count> ctrl_names = {
    "q", "k", "v", "ffn"
};

bool close_to(double actual, double expected, double tol) {
    return std::isfinite(actual) && std::abs(actual - expected) <= tol;
}

template <class Metadata> void validate_inkling2_metadata(const Metadata& m) {
    need(m.string("general.architecture") == "inkling2_loop", "gguf_text_architecture");
    need(m.integer("inkling2_loop.block_count") == n_phys, "gguf_text_metadata");
    need(m.integer("inkling2_loop.context_length") == 1048576, "gguf_text_metadata");
    need(m.integer("inkling2_loop.embedding_length") == n_embd, "gguf_text_metadata");

    const auto heads = m.integers("inkling2_loop.attention.head_count");
    const auto kv_heads = m.integers("inkling2_loop.attention.head_count_kv");
    need(heads.size() == n_phys && kv_heads.size() == n_phys, "gguf_text_metadata");
    for (std::size_t i = 0; i < n_phys; ++i)
        need(
            heads[i] == std::int64_t(n_heads) && kv_heads[i] == std::int64_t(n_kv_heads),
            "gguf_text_metadata"
        );

    need(
        m.integer("inkling2_loop.attention.key_length") == head_dim
            && m.integer("inkling2_loop.attention.value_length") == head_dim,
        "gguf_text_metadata"
    );
    need(
        close_to(
            m.number("inkling2_loop.attention.layer_norm_rms_epsilon"), rms_eps, 1e-8
        ),
        "gguf_text_metadata"
    );
    need(
        close_to(
            m.number("inkling2_loop.attention.qk_clip"), qk_clip_threshold, 1e-3
        ),
        "gguf_text_metadata"
    );
    need(
        m.integer("inkling2_loop.attention.relative_length") == n_rel
            && m.integer("inkling2_loop.attention.relative_extent") == n_rel_extent
            && close_to(
                m.number("inkling2_loop.attention.relative_clip"),
                rel_clip_threshold,
                1e-4
            ),
        "gguf_text_metadata"
    );
    need(
        m.integer("inkling2_loop.attention.log_scale_floor") == 8192
            && close_to(
                m.number("inkling2_loop.attention.log_scale_alpha"),
                log_scale_alpha,
                1e-4
            ),
        "gguf_text_metadata"
    );
    need(
        m.integer("inkling2_loop.attention.sliding_window") == n_swa_window,
        "gguf_text_metadata"
    );
    const auto swa = m.booleans("inkling2_loop.attention.sliding_window_pattern");
    need(swa.size() == n_phys, "gguf_text_metadata");
    for (std::size_t i = 0; i < n_phys; ++i)
        need(swa[i] == is_swa_phys(i), "gguf_text_metadata");

    need(m.integer("inkling2_loop.attn_res.block_size") == n_block_size, "gguf_text_metadata");
    need(m.integer("inkling2_loop.shortconv.l_cache") == n_conv, "gguf_text_metadata");
    need(
        m.integer("inkling2_loop.vocab_size") == n_vocab_padded
            && m.integer("inkling2_loop.vocab_size_unpadded") == n_vocab_unpadded,
        "gguf_text_metadata"
    );
    need(
        m.integer("inkling2_loop.expert_count") == n_experts
            && m.integer("inkling2_loop.expert_used_count") == n_experts_used
            && m.integer("inkling2_loop.expert_feed_forward_length") == n_ff_exp
            && close_to(
                m.number("inkling2_loop.expert_weights_scale"),
                expert_weights_scale,
                1e-4
            ),
        "gguf_text_metadata"
    );
    need(
        close_to(m.number("inkling2_loop.activation.situ_beta"), situ_beta, 1e-4)
            && close_to(
                m.number("inkling2_loop.activation.situ_linear_beta"),
                situ_linear_beta,
                1e-4
            )
            && close_to(
                m.number("inkling2_loop.final_logit_softcapping"),
                final_logit_softcap,
                1e-4
            ),
        "gguf_text_metadata"
    );
    need(
        m.integer("inkling2_loop.loop.start") == loop_start
            && m.integer("inkling2_loop.loop.end") == loop_end
            && m.integer("inkling2_loop.loop.count") == loop_passes
            && m.integer("inkling2_loop.loop.gate_rank") == loop_gate_rank,
        "gguf_text_metadata"
    );
    if (m.has("inkling2_loop.loop.varying"))
        need(!m.boolean("inkling2_loop.loop.varying"), "gguf_text_metadata");
    if (m.has("tokenizer.ggml.eos_token_id"))
        need(m.integer("tokenizer.ggml.eos_token_id") == eos_token, "gguf_text_metadata");
}

void bind_small_2d_or_1d(
    GgufWeights& w, const std::string& name, std::size_t rows, std::size_t cols
) {
    const auto& t = w.gguf().tensor(name);
    if (rows == 1 && t.rank == 1)
        w.vector(name, {cols});
    else
        w.table(name, rows, cols);
}

Vec load_small_2d_or_1d(const GgufWeights& w, const std::string& name) {
    if (w.gguf().tensor(name).rank == 1)
        return w.read(name);
    return w.decode(name);
}

struct ConvWeights {
    std::array<Vec, n_conv> tap;  // tap[k][c] = weight[c * n_conv + k] (t-3, t-2, t-1, t)
};

ConvWeights transpose_conv(const Vec& raw, std::size_t channels) {
    ConvWeights out;
    for (std::size_t k = 0; k < n_conv; ++k)
        out.tap[k].resize(channels);
    for (std::size_t c = 0; c < channels; ++c)
        for (std::size_t k = 0; k < n_conv; ++k)
            out.tap[k][c] = raw[c * n_conv + k];
    return out;
}

void rms_norm_rows(
    const GgufWeights& weights,
    std::span<float> x,
    std::span<const float> weight,
    std::size_t rows,
    float eps
) {
    const std::size_t width = weight.size();
    if (rows < 8) {
        rms_norm(x, weight, eps);
        return;
    }
    weights.parallel(rows, [&](std::size_t r) {
        rms_norm(x.subspan(r * width, width), weight, eps);
    });
}

void rms_norm_rows(
    const GgufWeights& weights,
    std::span<float> x,
    std::size_t width,
    std::size_t rows,
    float eps
) {
    if (rows < 8) {
        rms_norm(x, width, eps);
        return;
    }
    weights.parallel(rows, [&](std::size_t r) {
        rms_norm(x.subspan(r * width, width), width, eps);
    });
}

// Per-row optional scaling followed by L2 norm clipping to `threshold`:
//   y = (x * scale) * (threshold / max(||x * scale||_2, threshold))
void clip_heads(
    const GgufWeights& weights,
    std::span<float> x,
    std::size_t rows,
    std::size_t heads,
    std::size_t dim,
    const Vec* scale,
    float threshold
) {
    const auto do_row = [&](std::size_t r) {
        for (std::size_t h = 0; h < heads; ++h) {
            float* ptr = x.data() + (r * heads + h) * dim;
            double sum_sq = 0.0;
            if (scale) {
                const float* s = scale->data() + h * dim;
                for (std::size_t d = 0; d < dim; ++d) {
                    const float v = ptr[d] * s[d];
                    ptr[d] = v;
                    sum_sq += double(v) * double(v);
                }
            } else {
                for (std::size_t d = 0; d < dim; ++d)
                    sum_sq += double(ptr[d]) * double(ptr[d]);
            }
            const float norm = std::sqrt(float(sum_sq));
            if (norm > threshold) {
                const float factor = threshold / norm;
                for (std::size_t d = 0; d < dim; ++d)
                    ptr[d] *= factor;
            }
        }
    };
    if (rows >= 8)
        weights.parallel(rows, do_row);
    else
        for (std::size_t r = 0; r < rows; ++r)
            do_row(r);
}

// Block attention residuals:
// Stack [blocks..., cur], apply unweighted RMSNorm(1e-6) to each state, compute
// softmax((S @ query) / sqrt(n_embd)) over the state dimension, and return the
// weighted sum of the normalized states for each query row.
// Output layout: n_queries vectors of [rows * n_embd].
std::vector<Vec> mix_states(
    const GgufWeights& weights,
    const std::vector<const Vec*>& blocks,
    const Vec& cur,
    const Vec& query,
    std::size_t n_queries,
    std::size_t rows
) {
    const std::size_t n_states = blocks.size() + 1;
    std::vector<Vec> outputs(n_queries, Vec(rows * n_embd, 0.0f));
    const float inv_sqrt_d = 1.0f / std::sqrt(float(n_embd));

    const auto do_row = [&](std::size_t r) {
        thread_local Vec norm_states;
        thread_local Vec scores_buf;
        norm_states.resize(n_states * n_embd);
        scores_buf.resize(n_states);

        for (std::size_t s = 0; s < n_states; ++s) {
            const float* src =
                (s < blocks.size() ? blocks[s]->data() : cur.data()) + r * n_embd;
            float* dst = norm_states.data() + s * n_embd;
            double sum_sq = 0.0;
            for (std::size_t d = 0; d < n_embd; ++d)
                sum_sq += double(src[d]) * double(src[d]);
            const float inv = 1.0f / std::sqrt(float(sum_sq / double(n_embd)) + rms_eps);
            for (std::size_t d = 0; d < n_embd; ++d)
                dst[d] = src[d] * inv;
        }
        for (std::size_t q = 0; q < n_queries; ++q) {
            const float* qv = query.data() + q * n_embd;
            float max_score = -INFINITY;
            for (std::size_t s = 0; s < n_states; ++s) {
                const float* sv = norm_states.data() + s * n_embd;
                float dot = 0.0f;
                for (std::size_t d = 0; d < n_embd; ++d)
                    dot += sv[d] * qv[d];
                scores_buf[s] = dot * inv_sqrt_d;
                max_score = std::max(max_score, scores_buf[s]);
            }
            double denom = 0.0;
            for (std::size_t s = 0; s < n_states; ++s) {
                scores_buf[s] = std::exp(scores_buf[s] - max_score);
                denom += double(scores_buf[s]);
            }
            const float inv_denom = float(1.0 / denom);
            float* out = outputs[q].data() + r * n_embd;
            {
                const float w0 = scores_buf[0] * inv_denom;
                const float* sv0 = norm_states.data();
                for (std::size_t d = 0; d < n_embd; ++d)
                    out[d] = w0 * sv0[d];
            }
            for (std::size_t s = 1; s < n_states; ++s) {
                const float w = scores_buf[s] * inv_denom;
                const float* sv = norm_states.data() + s * n_embd;
                for (std::size_t d = 0; d < n_embd; ++d)
                    out[d] += w * sv[d];
            }
        }
    };

    if (rows >= 8)
        weights.parallel(rows, do_row);
    else
        for (std::size_t r = 0; r < rows; ++r)
            do_row(r);
    return outputs;
}

// Causal short convolution with residual and optional input-dependent per-head taps.
// `conv_state` holds [n_conv_channels, n_conv_hist] floats (oldest..newest).
// `channel_offset` selects the slice [channel_offset .. channel_offset + channels).
// `weight.tap[k]` holds [channels] floats for lag k=0..3 (oldest..newest: t-3..t).
// `dynamic` (if non-null) holds [rows, dyn_heads, n_conv] floats with lag 0=t .. 3=t-3.
Vec convolve(
    const GgufWeights& weights,
    const Vec& cur,
    std::size_t rows,
    std::size_t channels,
    const ConvWeights& weight,
    const Vec* dynamic,
    std::size_t dyn_heads,
    std::span<float> conv_state,
    std::size_t channel_offset
) {
    Vec out(rows * channels);
    const float scale =
        1.0f / std::sqrt(float((dynamic ? 2 : 1) * n_conv + 1));
    const float* w0 = weight.tap[0].data();
    const float* w1 = weight.tap[1].data();
    const float* w2 = weight.tap[2].data();
    const float* w3 = weight.tap[3].data();

    // Fast path for single-token decode: no temporary history buffer, no thread wakeup.
    if (rows == 1) {
        const float* x = cur.data();
        float* dst = out.data();
        float* state_base = conv_state.data() + channel_offset * n_conv_hist;
        if (!dynamic) {
            for (std::size_t c = 0; c < channels; ++c) {
                float* hist = state_base + c * n_conv_hist;
                const float h0 = hist[0], h1 = hist[1], h2 = hist[2];
                const float x_cur = x[c];
                const float acc = h0 * w0[c] + h1 * w1[c] + h2 * w2[c] + x_cur * w3[c];
                dst[c] = (x_cur + acc) * scale;
                hist[0] = h1;
                hist[1] = h2;
                hist[2] = x_cur;
            }
        } else {
            const std::size_t head_width = channels / dyn_heads;
            for (std::size_t dh = 0; dh < dyn_heads; ++dh) {
                const float* taps = dynamic->data() + dh * n_conv;
                const float t0 = taps[0], t1 = taps[1], t2 = taps[2], t3 = taps[3];
                const std::size_t c0 = dh * head_width;
                const std::size_t c1 = c0 + head_width;
                for (std::size_t c = c0; c < c1; ++c) {
                    float* hist = state_base + c * n_conv_hist;
                    const float h0 = hist[0], h1 = hist[1], h2 = hist[2];
                    const float x_cur = x[c];
                    const float acc = h0 * (w0[c] + t3) + h1 * (w1[c] + t2)
                                      + h2 * (w2[c] + t1) + x_cur * (w3[c] + t0);
                    dst[c] = (x_cur + acc) * scale;
                    hist[0] = h1;
                    hist[1] = h2;
                    hist[2] = x_cur;
                }
            }
        }
        return out;
    }

    // Multi-row path: unpack 3 history rows so every row reads contiguous memory.
    std::array<Vec, n_conv_hist> hist_rows;
    for (std::size_t k = 0; k < n_conv_hist; ++k)
        hist_rows[k].resize(channels);
    float* state_base = conv_state.data() + channel_offset * n_conv_hist;
    for (std::size_t c = 0; c < channels; ++c) {
        const float* hist = state_base + c * n_conv_hist;
        hist_rows[0][c] = hist[0];
        hist_rows[1][c] = hist[1];
        hist_rows[2][c] = hist[2];
    }

    const std::size_t head_width = dyn_heads ? channels / dyn_heads : 0;
    const auto do_row = [&](std::size_t r) {
        const float* x0 = r >= 3 ? cur.data() + (r - 3) * channels : hist_rows[r].data();
        const float* x1 =
            r >= 2 ? cur.data() + (r - 2) * channels : hist_rows[r + 1].data();
        const float* x2 =
            r >= 1 ? cur.data() + (r - 1) * channels : hist_rows[r + 2].data();
        const float* x3 = cur.data() + r * channels;
        float* dst = out.data() + r * channels;
        if (!dynamic) {
            for (std::size_t c = 0; c < channels; ++c) {
                const float acc =
                    x0[c] * w0[c] + x1[c] * w1[c] + x2[c] * w2[c] + x3[c] * w3[c];
                dst[c] = (x3[c] + acc) * scale;
            }
        } else {
            for (std::size_t dh = 0; dh < dyn_heads; ++dh) {
                const float* taps = dynamic->data() + (r * dyn_heads + dh) * n_conv;
                const float t0 = taps[0], t1 = taps[1], t2 = taps[2], t3 = taps[3];
                const std::size_t c0 = dh * head_width;
                const std::size_t c1 = c0 + head_width;
                for (std::size_t c = c0; c < c1; ++c) {
                    const float acc = x0[c] * (w0[c] + t3) + x1[c] * (w1[c] + t2)
                                      + x2[c] * (w2[c] + t1) + x3[c] * (w3[c] + t0);
                    dst[c] = (x3[c] + acc) * scale;
                }
            }
        }
    };

    if (rows >= 8)
        weights.parallel(rows, do_row);
    else
        for (std::size_t r = 0; r < rows; ++r)
            do_row(r);

    // Commit the trailing 3 rows back into conv_state.
    const float* n0 = rows >= 3 ? cur.data() + (rows - 3) * channels : hist_rows[rows].data();
    const float* n1 =
        rows >= 2 ? cur.data() + (rows - 2) * channels : hist_rows[rows + 1].data();
    const float* n2 = cur.data() + (rows - 1) * channels;
    for (std::size_t c = 0; c < channels; ++c) {
        float* hist = state_base + c * n_conv_hist;
        hist[0] = n0[c];
        hist[1] = n1[c];
        hist[2] = n2[c];
    }
    return out;
}

struct PhysLayer {
    bool is_swa = false;
    Vec attn_norm;        // SWA only
    Vec attn_q_inp_norm;  // Full only
    Vec attn_k_inp_norm;  // Full only
    Vec attn_v_inp_norm;  // Full only

    Vec attn_q_clip;      // [n_heads, head_dim]
    Vec attn_rel_bias;    // [n_rel_extent, n_rel]
    Vec attn_rel_clip;    // [n_heads, n_rel]

    ConvWeights attn_k_conv;     // 4 x [n_embd_k_gqa]
    ConvWeights attn_v_conv;     // 4 x [n_embd_k_gqa]
    ConvWeights attn_post_conv;  // 4 x [n_embd]
    ConvWeights ffn_post_conv;   // 4 x [n_embd]

    bool has_attn_res = false;
    Vec attn_res_query;   // [1 or 3, n_embd]
    Vec ffn_res_query;    // [n_embd]
    Vec ffn_norm;         // [n_embd]
    bool has_exp_bias = false;
    Vec ffn_exp_probs_b;  // [n_experts]

    // Tensor names for device/Strata projections
    std::string wq, wk, wv, wo, wgate, wrel, wsink;
    std::string wk_conv_proj, wv_conv_proj;
    std::string wffn_router, wffn_gate_up, wffn_down;
};

struct LoopLayer {
    Vec msg_query;  // [loop_ctrl_count=4, n_embd]
    std::array<std::string, loop_ctrl_count> ctrl_down;
    std::array<std::string, loop_ctrl_count> ctrl_gain;
    std::array<std::string, loop_ctrl_count> ctrl_forget;
};

class Inkling2LoopModel final : public Model {
public:
    Inkling2LoopModel(
        const GgufWeights& weights,
        DenseCompute& compute,
        std::size_t context,
        std::uint32_t initial_variant_id = 0
    )
        : weights_(weights), compute_(compute), context_(context),
          active_variant_id_(initial_variant_id) {
        tok_norm_ = weights_.read("token_embd_norm.weight");
        out_norm_ = weights_.read("output_norm.weight");
        out_res_query_ = weights_.read("output_res_query.weight");

        layers_.resize(n_phys);
        for (std::size_t i = 0; i < n_phys; ++i) {
            auto& l = layers_[i];
            l.is_swa = is_swa_phys(i);
            l.wq = blk(i, "attn_q.weight");
            l.wk = blk(i, "attn_k.weight");
            l.wv = blk(i, "attn_v.weight");
            l.wo = blk(i, "attn_output.weight");
            l.wgate = blk(i, "attn_gate.weight");
            l.wrel = blk(i, "attn_rel.weight");
            l.wsink = blk(i, "attn_sink_proj.weight");
            l.wffn_router = blk(i, "ffn_gate_inp.weight");
            l.wffn_gate_up = blk(i, "ffn_gate_up_exps.weight");
            l.wffn_down = blk(i, "ffn_down_exps.weight");

            if (l.is_swa) {
                l.attn_norm = weights_.read(blk(i, "attn_norm.weight"));
            } else {
                l.attn_q_inp_norm = weights_.read(blk(i, "attn_q_inp_norm.weight"));
                l.attn_k_inp_norm = weights_.read(blk(i, "attn_k_inp_norm.weight"));
                l.attn_v_inp_norm = weights_.read(blk(i, "attn_v_inp_norm.weight"));
                l.wk_conv_proj = blk(i, "attn_k_conv_proj.weight");
                l.wv_conv_proj = blk(i, "attn_v_conv_proj.weight");
            }

            l.attn_q_clip = weights_.decode(blk(i, "attn_q_clip.weight"));
            l.attn_rel_bias = weights_.decode(blk(i, "attn_rel_bias.weight"));
            l.attn_rel_clip = weights_.decode(blk(i, "attn_rel_clip.weight"));
            l.attn_k_conv =
                transpose_conv(weights_.decode(blk(i, "attn_k_conv.weight")), n_embd_k_gqa);
            l.attn_v_conv =
                transpose_conv(weights_.decode(blk(i, "attn_v_conv.weight")), n_embd_k_gqa);
            l.attn_post_conv =
                transpose_conv(weights_.decode(blk(i, "attn_post_conv.weight")), n_embd);
            l.ffn_post_conv =
                transpose_conv(weights_.decode(blk(i, "ffn_post_conv.weight")), n_embd);

            if (i > 0) {
                l.has_attn_res = true;
                l.attn_res_query =
                    load_small_2d_or_1d(weights_, blk(i, "attn_res_query.weight"));
            }
            l.ffn_res_query = weights_.read(blk(i, "ffn_res_query.weight"));
            l.ffn_norm = weights_.read(blk(i, "ffn_norm.weight"));

            const auto bias_name = blk(i, "exp_probs_b.bias");
            l.has_exp_bias = has_tensor(weights_.gguf(), bias_name);
            if (l.has_exp_bias)
                l.ffn_exp_probs_b = weights_.read(bias_name);
        }

        loop_layers_.resize(loop_core_len);
        for (std::size_t ip = loop_start; ip < loop_end; ++ip) {
            auto& ll = loop_layers_[ip - loop_start];
            ll.msg_query = weights_.decode(blk(ip, "loop_msg_query.weight"));
            for (std::size_t c = 0; c < loop_ctrl_count; ++c) {
                const std::string prefix = std::string("loop_") + ctrl_names[c] + "_ctrl_";
                ll.ctrl_down[c] = blk(ip, (prefix + "down.weight").c_str());
                ll.ctrl_gain[c] = blk(ip, (prefix + "gain.weight").c_str());
                ll.ctrl_forget[c] = blk(ip, (prefix + "forget.weight").c_str());
            }
        }

        init_state();
    }

    std::size_t vocabulary() const override {
        return n_vocab_unpadded;
    }

    bool eos(unsigned token) const override {
        return token == eos_token;
    }

    std::size_t device_state_bytes() const override {
        return inkling2_loop_device_state_bytes(context_);
    }

    void reset() override {
        if (slots_.size() != n_slots) {
            init_state();
        } else {
            for (auto& slot : slots_) {
                std::fill(slot.attention.keys.begin(), slot.attention.keys.end(), 0.0f);
                std::fill(slot.attention.values.begin(), slot.attention.values.end(), 0.0f);
                std::fill(slot.conv.begin(), slot.conv.end(), 0.0f);
            }
        }
        if (gpu_)
            gpu_->reset();
        gpu_valid_tokens_ = 0;
        host_state_stale_ = false;
        gpu_needs_upload_ = false;
    }

    ModelState take_state() override {
        if (host_state_stale_ && gpu_) {
            gpu_->sync_to_host(slots_, gpu_valid_tokens_, weights_.cancel());
            host_state_stale_ = false;
        }
        ModelState out;
        out.layers.reserve(n_slots);
        for (auto& slot : slots_)
            out.layers.emplace_back(std::move(slot));
        slots_.clear();
        return out;
    }

    ModelState snapshot_state(std::size_t tokens, std::size_t context) override {
        if (host_state_stale_ && gpu_) {
            gpu_->sync_to_host(slots_, gpu_valid_tokens_, weights_.cancel());
            host_state_stale_ = false;
        }
        ModelState compacted;
        compacted.layers.reserve(n_slots);
        for (const auto& s : slots_) {
            InklingLayerState copy;
            copy.sliding_window = s.sliding_window;
            copy.conv = s.conv;
            if (s.sliding_window || tokens >= context) {
                copy.attention.keys = s.attention.keys;
                copy.attention.values = s.attention.values;
            } else {
                const std::size_t width = s.attention.keys.size() / context;
                const std::size_t keep = tokens * width;
                copy.attention.keys.assign(
                    s.attention.keys.begin(),
                    s.attention.keys.begin() + std::ptrdiff_t(keep)
                );
                copy.attention.values.assign(
                    s.attention.values.begin(),
                    s.attention.values.begin() + std::ptrdiff_t(keep)
                );
            }
            compacted.layers.emplace_back(std::move(copy));
        }
        return compacted;
    }

    void restore_state(ModelState state) override {
        need(state.layers.size() == n_slots, "gguf_text_state_layers");
        slots_.clear();
        slots_.reserve(n_slots);
        for (std::size_t s = 0; s < n_slots; ++s) {
            auto& entry = std::get<InklingLayerState>(state.layers[s]);
            const auto cap = slot_kv_capacity(s, context_);
            need(
                entry.sliding_window == is_swa_phys(slot_phys(s))
                    && entry.attention.keys.size() == cap * n_embd_k_gqa
                    && entry.attention.values.size() == cap * n_embd_k_gqa
                    && entry.conv.size() == n_conv_hist * n_conv_channels,
                "gguf_text_state_shape"
            );
            slots_.push_back(std::move(entry));
        }
        host_state_stale_ = false;
        gpu_needs_upload_ = true;
    }

    void save_gpu_boundary(
        unsigned slot_idx, std::size_t tokens, std::uint32_t variant_id
    ) override {
        if (!gpu_ || gpu_needs_upload_ || gpu_valid_tokens_ != tokens || !tokens
            || variant_id != active_variant_id_)
            return;
        gpu_->save_boundary(slot_idx, tokens, variant_id, weights_.cancel());
    }

    bool try_rewind_gpu_boundary(
        unsigned slot_idx, std::size_t boundary_tokens, std::uint32_t variant_id
    ) override {
        if (!gpu_ || gpu_needs_upload_ || !boundary_tokens
            || gpu_valid_tokens_ < boundary_tokens || variant_id != active_variant_id_)
            return false;
        if (!gpu_->restore_boundary(
                slot_idx, boundary_tokens, variant_id, weights_.cancel()
            ))
            return false;
        gpu_valid_tokens_ = boundary_tokens;
        host_state_stale_ = true;
        gpu_needs_upload_ = false;
        return true;
    }

    bool has_gpu_executor() const override {
        return bool(gpu_);
    }

    bool switch_variant(
        std::uint32_t target_variant_id,
        const checkpoint::GgufCheckpoint& target_gguf,
        const std::atomic_bool& cancel
    ) override {
        if (gpu_) {
            std::vector<InklingVariantTensor> tensors;
            tensors.reserve(
                (n_phys - kInklingVariantAttnOutputStart)
                + (n_phys - kInklingVariantFfnDownStart) * n_experts
            );
            auto make_desc = [&](const std::string& name,
                                 std::size_t in,
                                 std::size_t out,
                                 std::size_t expert,
                                 std::size_t experts) -> InklingMatrixDesc {
                const auto& tensor = target_gguf.tensor(name);
                const Bytes bytes = tensor.bytes / experts;
                const std::size_t row_bytes = bytes / out;
                auto source = [&target_gguf, name, expert, &cancel](
                                  std::size_t first, std::span<std::byte> to
                              ) {
                    target_gguf.read_packed_rows(name, expert, first, to, cancel);
                };
                return InklingMatrixDesc{
                    target_gguf.identity(name, expert),
                    tensor.type == 0 ? std::nullopt : gguf::encoding(tensor.type),
                    in,
                    out,
                    row_bytes,
                    std::move(source)
                };
            };
            for (std::size_t i = kInklingVariantAttnOutputStart; i < n_phys; ++i) {
                tensors.push_back(
                    InklingVariantTensor{
                        i,
                        InklingVariantTensor::Role::attn_output,
                        0,
                        make_desc(layers_[i].wo, n_embd_q, n_embd, 0, 1)
                    }
                );
            }
            for (std::size_t i = kInklingVariantFfnDownStart; i < n_phys; ++i) {
                for (std::size_t e = 0; e < n_experts; ++e) {
                    tensors.push_back(
                        InklingVariantTensor{
                            i,
                            InklingVariantTensor::Role::ffn_down_expert,
                            e,
                            make_desc(
                                layers_[i].wffn_down, n_ff_exp, n_embd, e, n_experts
                            )
                        }
                    );
                }
            }
            if (!gpu_->swap_variant(target_variant_id, tensors, cancel))
                return false;
            for (auto& slot : slots_) {
                std::fill(slot.attention.keys.begin(), slot.attention.keys.end(), 0.0f);
                std::fill(
                    slot.attention.values.begin(), slot.attention.values.end(), 0.0f
                );
                std::fill(slot.conv.begin(), slot.conv.end(), 0.0f);
            }
            gpu_valid_tokens_ = 0;
            host_state_stale_ = false;
            gpu_needs_upload_ = false;
            active_variant_id_ = target_variant_id;
            return true;
        }
        reset();
        active_variant_id_ = target_variant_id;
        return true;
    }

    std::optional<unsigned> forward_argmax(
        std::span<const unsigned> tokens,
        std::size_t position,
        const Progress& progress,
        std::span<const ImageSpan> images
    ) override {
        const std::size_t rows = tokens.size();
        need(rows > 0 && position + rows <= context_, "gguf_text_forward_range");
        if (!gpu_tried_) {
            gpu_tried_ = true;
            gpu_ = try_create_gpu_executor();
        }
        if (!gpu_)
            return std::nullopt;
        Vec cur = embed_rows(tokens, position, images);
        weights_.check_unchanged();
        if (position == 0) {
            gpu_->reset();
            gpu_valid_tokens_ = 0;
            gpu_needs_upload_ = false;
        } else if (gpu_needs_upload_ || gpu_valid_tokens_ != position) {
            gpu_->sync_from_host(slots_, position, weights_.cancel());
            gpu_valid_tokens_ = position;
            gpu_needs_upload_ = false;
        }
        Vec unused_logits;
        unsigned chosen = 0;
        if (gpu_->forward(
                cur, rows, position, true, unused_logits, progress, weights_.cancel(), &chosen
            )) {
            weights_.check_unchanged();
            gpu_valid_tokens_ = position + rows;
            host_state_stale_ = true;
            return chosen;
        }
        if (host_state_stale_) {
            gpu_->sync_to_host(slots_, gpu_valid_tokens_, weights_.cancel());
            host_state_stale_ = false;
        }
        gpu_.reset();
        return std::nullopt;
    }

    Vec forward(
        std::span<const unsigned> tokens,
        std::size_t position,
        bool logits,
        const Progress& progress,
        std::span<const ImageSpan> images
    ) override {
        const std::size_t rows = tokens.size();
        need(rows > 0 && position + rows <= context_, "gguf_text_forward_range");

        // 1. Token embeddings (+ token_embd_norm) with optional raw image embeddings
        //    (image embeddings explicitly skip token_embd_norm, per inkling2.cpp).
        Vec cur = embed_rows(tokens, position, images);

        // Fast path: 100% GPU-resident 48-slot forward pass (+ decode CUDA Graph)
        // when the single state GPU holds all packed weights and FP16 KV state.
        if (!gpu_tried_) {
            gpu_tried_ = true;
            gpu_ = try_create_gpu_executor();
        }
        if (gpu_) {
            weights_.check_unchanged();
            if (position == 0) {
                gpu_->reset();
                gpu_valid_tokens_ = 0;
                gpu_needs_upload_ = false;
            } else if (gpu_needs_upload_ || gpu_valid_tokens_ != position) {
                gpu_->sync_from_host(slots_, position, weights_.cancel());
                gpu_valid_tokens_ = position;
                gpu_needs_upload_ = false;
            }
            Vec gpu_logits;
            if (gpu_->forward(
                    cur, rows, position, logits, gpu_logits, progress, weights_.cancel()
                )) {
                weights_.check_unchanged();
                gpu_valid_tokens_ = position + rows;
                host_state_stale_ = true;
                return gpu_logits;
            }
            // Device could not grow state; sync back to host and fall back cleanly to Strata.
            if (host_state_stale_) {
                gpu_->sync_to_host(slots_, gpu_valid_tokens_, weights_.cancel());
                host_state_stale_ = false;
            }
            gpu_.reset();
        }

        // Precompute full-attention log-scale tau(pos) for this chunk.
        Vec tau(rows);
        for (std::size_t r = 0; r < rows; ++r) {
            const float pos1 = float(position + r + 1);
            tau[r] = 1.0f + log_scale_alpha * std::log(std::max(1.0f, pos1 / log_scale_floor));
        }

        // Block residual stream history.
        std::vector<Vec> block_storage;
        block_storage.reserve(16);
        auto block_ptrs = [](const std::vector<Vec>& storage) {
            std::vector<const Vec*> ptrs;
            ptrs.reserve(storage.size());
            for (const auto& b : storage)
                ptrs.push_back(&b);
            return ptrs;
        };

        std::size_t visit = 0;

        // 2. Prelude: physical layers 0..5 (slots 0..5)
        for (std::size_t ip = 0; ip < loop_start; ++ip) {
            cur = run_layer(ip, ip, cur, block_storage, nullptr, nullptr, position, rows, tau);
            if (progress)
                progress("layer", ++visit);
        }

        // 3. Looped core: passes 0, 1, 2 over physical layers 6..17
        const Vec prelude = cur;
        const std::vector<Vec> prelude_blocks = block_storage;
        for (std::size_t pass = 0; pass < loop_passes; ++pass) {
            std::vector<Vec> bank;
            Vec bank_out;
            if (pass > 0) {
                bank.assign(
                    block_storage.begin() + std::ptrdiff_t(prelude_blocks.size() + 1),
                    block_storage.end()
                );
                bank_out = std::move(cur);
                cur = prelude;
                block_storage = prelude_blocks;
            }
            const auto bank_refs = block_ptrs(bank);
            for (std::size_t ip = loop_start; ip < loop_end; ++ip) {
                const std::size_t slot =
                    pass == 0 ? ip
                              : n_phys + (pass - 1) * loop_core_len + (ip - loop_start);
                if (pass == 0) {
                    cur = run_layer(
                        slot, ip, cur, block_storage, nullptr, nullptr, position, rows, tau
                    );
                } else {
                    const auto& loop = loop_layers_[ip - loop_start];
                    auto messages = mix_states(
                        weights_, bank_refs, bank_out, loop.msg_query, loop_ctrl_count, rows
                    );
                    cur = run_layer(
                        slot, ip, cur, block_storage, &messages, &loop, position, rows, tau
                    );
                }
                if (progress)
                    progress("layer", ++visit);
            }
        }

        // 4. Coda: physical layers 18..23 (slots 18..23)
        for (std::size_t ip = loop_end; ip < n_phys; ++ip) {
            cur = run_layer(ip, ip, cur, block_storage, nullptr, nullptr, position, rows, tau);
            if (progress)
                progress("layer", ++visit);
        }

        if (!logits)
            return {};

        // 5. Final output head (last row only):
        //    mix(blocks, cur, output_res_query) -> output_norm -> output.weight
        //    -> softcap 30*tanh(x/30) -> slice [0..n_vocab_unpadded)
        std::vector<Vec> last_blocks;
        last_blocks.reserve(block_storage.size());
        for (const auto& b : block_storage)
            last_blocks.emplace_back(
                b.begin() + std::ptrdiff_t((rows - 1) * n_embd), b.end()
            );
        Vec last_cur(
            cur.begin() + std::ptrdiff_t((rows - 1) * n_embd), cur.end()
        );
        auto mixed_out = mix_states(
            weights_, block_ptrs(last_blocks), last_cur, out_res_query_, 1, 1
        );
        Vec hidden = std::move(mixed_out[0]);
        rms_norm(hidden, out_norm_, rms_eps);
        Vec raw_logits = weights_.project(compute_, "output.weight", hidden);
        need(raw_logits.size() == n_vocab_padded, "gguf_text_logits_shape");
        Vec result(n_vocab_unpadded);
        const float inv_cap = 1.0f / final_logit_softcap;
        for (std::size_t i = 0; i < n_vocab_unpadded; ++i)
            result[i] = final_logit_softcap * std::tanh(raw_logits[i] * inv_cap);
        return result;
    }

private:
    Vec embed_rows(
        std::span<const unsigned> tokens,
        std::size_t position,
        std::span<const ImageSpan> images
    ) const {
        const std::size_t rows = tokens.size();
        Vec cur(rows * n_embd);
        std::vector<bool> is_image_row(rows, false);
        for (const auto& img : images) {
            need(
                img.embeddings.size() == img.tokens * n_embd,
                "gguf_text_image_span_shape"
            );
            const std::size_t img_begin = img.offset;
            const std::size_t img_end = img.offset + img.tokens;
            const std::size_t ov_begin = std::max(position, img_begin);
            const std::size_t ov_end = std::min(position + rows, img_end);
            for (std::size_t p = ov_begin; p < ov_end; ++p) {
                const std::size_t r = p - position;
                const std::size_t src_row = p - img_begin;
                std::memcpy(
                    cur.data() + r * n_embd,
                    img.embeddings.data() + src_row * n_embd,
                    n_embd * sizeof(float)
                );
                is_image_row[r] = true;
            }
        }
        for (std::size_t r = 0; r < rows; ++r) {
            if (is_image_row[r])
                continue;
            need(tokens[r] < n_vocab_unpadded, "gguf_text_token_range");
            auto row_span = std::span(cur).subspan(r * n_embd, n_embd);
            weights_.rows("token_embd.weight", tokens[r], row_span);
            rms_norm(row_span, tok_norm_, rms_eps);
        }
        return cur;
    }

    static bool has_tensor(
        const checkpoint::GgufCheckpoint& gguf, std::string_view name
    ) {
        for (std::size_t i = 0; i < gguf.tensor_count(); ++i)
            if (gguf.tensor_at(i).name == name)
                return true;
        return false;
    }

    void init_state() {
        slots_.clear();
        slots_.reserve(n_slots);
        for (std::size_t s = 0; s < n_slots; ++s) {
            InklingLayerState st;
            st.sliding_window = is_swa_phys(slot_phys(s));
            const auto cap = slot_kv_capacity(s, context_);
            st.attention.keys.assign(cap * n_embd_k_gqa, 0.0f);
            st.attention.values.assign(cap * n_embd_k_gqa, 0.0f);
            st.conv.assign(n_conv_hist * n_conv_channels, 0.0f);
            slots_.push_back(std::move(st));
        }
    }

    // Cross-pass gated control for one or more controllers batched on device.
    // inputs[k] = (1 - forget_k) * inputs[k] + gain_k * messages[k]
    void apply_controls(
        std::span<Vec> targets,
        std::span<const Vec> messages,
        const LoopLayer& loop,
        std::size_t ctrl_base,
        std::size_t rows
    ) {
        const std::size_t count = targets.size();
        std::vector<Vec> concats(count);
        std::vector<GgufWeights::Projection> down_jobs;
        down_jobs.reserve(count);

        for (std::size_t k = 0; k < count; ++k) {
            auto& cat = concats[k];
            cat.resize(rows * 2 * n_embd);
            const auto& a_src = targets[k];
            const auto& b_src = messages[k];
            const auto pack_row = [&](std::size_t r) {
                float* dst_a = cat.data() + r * 2 * n_embd;
                float* dst_b = dst_a + n_embd;
                std::memcpy(dst_a, a_src.data() + r * n_embd, n_embd * sizeof(float));
                std::memcpy(dst_b, b_src.data() + r * n_embd, n_embd * sizeof(float));
                rms_norm(std::span(dst_a, n_embd), n_embd, rms_eps);
                rms_norm(std::span(dst_b, n_embd), n_embd, rms_eps);
            };
            if (rows >= 8)
                weights_.parallel(rows, pack_row);
            else
                for (std::size_t r = 0; r < rows; ++r)
                    pack_row(r);
            down_jobs.push_back({loop.ctrl_down[ctrl_base + k], cat, 0});
        }
        auto gates = weights_.project(compute_, down_jobs);

        std::vector<GgufWeights::Projection> up_jobs;
        up_jobs.reserve(count * 2);
        for (std::size_t k = 0; k < count; ++k) {
            up_jobs.push_back({loop.ctrl_gain[ctrl_base + k], gates[k], 0});
            up_jobs.push_back({loop.ctrl_forget[ctrl_base + k], gates[k], 0});
        }
        auto up = weights_.project(compute_, up_jobs);

        for (std::size_t k = 0; k < count; ++k) {
            const auto& g = up[2 * k];
            const auto& f = up[2 * k + 1];
            const auto& msg = messages[k];
            auto& dst = targets[k];
            const auto blend_row = [&](std::size_t r) {
                const std::size_t base = r * n_embd;
                for (std::size_t d = 0; d < n_embd; ++d) {
                    const float gain = std::tanh(g[base + d]);
                    const float forget = std::tanh(f[base + d]);
                    dst[base + d] =
                        (1.0f - forget) * dst[base + d] + gain * msg[base + d];
                }
            };
            if (rows >= 8)
                weights_.parallel(rows, blend_row);
            else
                for (std::size_t r = 0; r < rows; ++r)
                    blend_row(r);
        }
    }

    Vec run_layer(
        std::size_t slot,
        std::size_t ip,
        const Vec& cur_in,
        std::vector<Vec>& block_storage,
        const std::vector<Vec>* messages,
        const LoopLayer* loop,
        std::size_t position,
        std::size_t rows,
        const Vec& tau
    ) {
        weights_.check_cancel();
        const auto& l = layers_[ip];
        auto& st = slots_[slot];

        std::vector<const Vec*> block_refs;
        block_refs.reserve(block_storage.size());
        for (const auto& b : block_storage)
            block_refs.push_back(&b);

        // Attention inputs: SWA mixes 1 query; full-attention mixes 3 (Q, K, V).
        // When SWA has no loop controls, Q/K/V share a single normalized buffer so
        // dense_gguf_batch deduplicates the input upload.
        const bool shared_swa_input = l.is_swa && !messages;
        std::array<Vec, 3> inputs;
        if (l.has_attn_res) {
            const std::size_t nq = l.is_swa ? 1 : 3;
            auto mixed =
                mix_states(weights_, block_refs, cur_in, l.attn_res_query, nq, rows);
            if (shared_swa_input) {
                inputs[0] = std::move(mixed[0]);
            } else if (l.is_swa) {
                inputs[0] = std::move(mixed[0]);
                inputs[1] = inputs[0];
                inputs[2] = inputs[0];
            } else {
                inputs[0] = std::move(mixed[0]);
                inputs[1] = std::move(mixed[1]);
                inputs[2] = std::move(mixed[2]);
            }
        } else if (shared_swa_input) {
            inputs[0] = cur_in;
        } else {
            inputs[0] = cur_in;
            inputs[1] = cur_in;
            inputs[2] = cur_in;
        }

        if (messages) {
            apply_controls(
                std::span(inputs),
                std::span<const Vec>(messages->data(), 3),
                *loop,
                0,
                rows
            );
        }

        if (shared_swa_input) {
            rms_norm_rows(weights_, inputs[0], l.attn_norm, rows, rms_eps);
        } else if (l.is_swa) {
            for (std::size_t j = 0; j < 3; ++j)
                rms_norm_rows(weights_, inputs[j], l.attn_norm, rows, rms_eps);
        } else {
            rms_norm_rows(weights_, inputs[0], l.attn_q_inp_norm, rows, rms_eps);
            rms_norm_rows(weights_, inputs[1], l.attn_k_inp_norm, rows, rms_eps);
            rms_norm_rows(weights_, inputs[2], l.attn_v_inp_norm, rows, rms_eps);
        }

        const std::span<const float> in_q = inputs[0];
        const std::span<const float> in_k = shared_swa_input ? std::span<const float>(inputs[0]) : std::span<const float>(inputs[1]);
        const std::span<const float> in_v = shared_swa_input ? std::span<const float>(inputs[0]) : std::span<const float>(inputs[2]);

        // Batched Q, K, V, R, sink, gate (+ dynamic conv taps on full-attention layers)
        std::vector<GgufWeights::Projection> attn_jobs;
        attn_jobs.reserve(l.is_swa ? 6 : 8);
        attn_jobs.push_back({l.wq, in_q, 0});
        attn_jobs.push_back({l.wk, in_k, 0});
        attn_jobs.push_back({l.wv, in_v, 0});
        attn_jobs.push_back({l.wrel, in_q, 0});
        attn_jobs.push_back({l.wsink, in_q, 0});
        attn_jobs.push_back({l.wgate, in_q, 0});
        if (!l.is_swa) {
            attn_jobs.push_back({l.wk_conv_proj, in_k, 0});
            attn_jobs.push_back({l.wv_conv_proj, in_v, 0});
        }
        auto attn_proj = weights_.project(compute_, attn_jobs);

        Vec q = std::move(attn_proj[0]);
        Vec k_raw = std::move(attn_proj[1]);
        Vec v_raw = std::move(attn_proj[2]);
        Vec r_cur = std::move(attn_proj[3]);
        Vec sinks = std::move(attn_proj[4]);
        Vec gate = std::move(attn_proj[5]);

        const auto conv_started = std::chrono::steady_clock::now();
        Vec k = convolve(
            weights_,
            k_raw,
            rows,
            n_embd_k_gqa,
            l.attn_k_conv,
            l.is_swa ? nullptr : &attn_proj[6],
            l.is_swa ? 0 : n_kv_heads,
            st.conv,
            0
        );
        Vec v = convolve(
            weights_,
            v_raw,
            rows,
            n_embd_k_gqa,
            l.attn_v_conv,
            l.is_swa ? nullptr : &attn_proj[7],
            l.is_swa ? 0 : n_kv_heads,
            st.conv,
            n_embd_k_gqa
        );
        weights_.add_recurrent(
            std::chrono::duration<double>(std::chrono::steady_clock::now() - conv_started)
                .count()
        );

        clip_heads(weights_, q, rows, n_heads, head_dim, &l.attn_q_clip, qk_clip_threshold);
        clip_heads(weights_, k, rows, n_kv_heads, head_dim, nullptr, qk_clip_threshold);
        rms_norm_rows(weights_, v, head_dim, rows * n_kv_heads, rms_eps);
        clip_heads(weights_, r_cur, rows, n_heads, n_rel, &l.attn_rel_clip, rel_clip_threshold);

        if (!l.is_swa) {
            const auto scale_row = [&](std::size_t r) {
                const float t = tau[r];
                float* qp = q.data() + r * n_embd_q;
                for (std::size_t i = 0; i < n_embd_q; ++i)
                    qp[i] *= t;
                float* rp = r_cur.data() + r * n_heads * n_rel;
                for (std::size_t i = 0; i < n_heads * n_rel; ++i)
                    rp[i] *= t;
            };
            if (rows >= 8)
                weights_.parallel(rows, scale_row);
            else
                for (std::size_t r = 0; r < rows; ++r)
                    scale_row(r);
        }

        // Attention with relative-position bias + learned virtual sink logit.
        const auto attn_started = std::chrono::steady_clock::now();
        Vec attn_out(rows * n_embd_q, 0.0f);
        const float qk_scale = 1.0f / std::sqrt(float(head_dim));
        const std::size_t group = n_heads / n_kv_heads;

        if (l.is_swa) {
            const std::size_t swa_win = slot_kv_capacity(slot, context_);
            DenseCompute::InklingAttention req{
                slot,
                ip,
                q,
                r_cur,
                sinks,
                l.attn_rel_bias,
                st.attention.keys,
                st.attention.values,
                k,
                v,
                rows,
                position,
                n_heads,
                n_kv_heads,
                head_dim,
                n_rel,
                n_rel_extent,
                swa_win,
                qk_scale,
                attn_out
            };
            if (!compute_.attend_inkling_cached(req, weights_.cancel())) {
                // Host fallback reads directly from host ring + new chunk (zero 8 MiB copy).
                weights_.parallel(rows * n_heads, [&](std::size_t work) {
                    const std::size_t r = work / n_heads;
                    const std::size_t h = work % n_heads;
                    const std::size_t kv_h = h / group;
                    const std::size_t pos_q = position + r;
                    const std::size_t pos_k_min =
                        pos_q + 1 > swa_win ? pos_q + 1 - swa_win : 0;
                    const std::size_t count = pos_q - pos_k_min + 1;

                    const float* qv = q.data() + (r * n_heads + h) * head_dim;
                    const float* rv = r_cur.data() + (r * n_heads + h) * n_rel;
                    const float sink_logit = sinks[r * n_heads + h];

                    thread_local Vec scores;
                    scores.resize(count);
                    float max_score = sink_logit;
                    for (std::size_t j = 0; j < count; ++j) {
                        const std::size_t pos_k = pos_k_min + j;
                        const std::size_t dist = pos_q - pos_k;
                        const float* kv =
                            pos_k < position
                                ? st.attention.keys.data()
                                      + ((pos_k % swa_win) * n_kv_heads + kv_h) * head_dim
                                : k.data() + ((pos_k - position) * n_kv_heads + kv_h) * head_dim;
                        float dot = 0.0f;
                        for (std::size_t d = 0; d < head_dim; ++d)
                            dot += qv[d] * kv[d];
                        const float* rb = l.attn_rel_bias.data() + dist * n_rel;
                        float rel_dot = 0.0f;
                        for (std::size_t d = 0; d < n_rel; ++d)
                            rel_dot += rv[d] * rb[d];
                        const float sc = dot * qk_scale + rel_dot;
                        scores[j] = sc;
                        max_score = std::max(max_score, sc);
                    }

                    double denom = std::exp(double(sink_logit - max_score));
                    for (std::size_t j = 0; j < count; ++j) {
                        scores[j] = std::exp(scores[j] - max_score);
                        denom += double(scores[j]);
                    }
                    const float inv_denom = float(1.0 / denom);
                    float* out = attn_out.data() + (r * n_heads + h) * head_dim;
                    for (std::size_t j = 0; j < count; ++j) {
                        const float p = scores[j] * inv_denom;
                        const std::size_t pos_k = pos_k_min + j;
                        const float* vv =
                            pos_k < position
                                ? st.attention.values.data()
                                      + ((pos_k % swa_win) * n_kv_heads + kv_h) * head_dim
                                : v.data() + ((pos_k - position) * n_kv_heads + kv_h) * head_dim;
                        for (std::size_t d = 0; d < head_dim; ++d)
                            out[d] += p * vv[d];
                    }
                });
            }

            // Commit the last min(rows, swa_win) entries into host ring state.
            const std::size_t commit_begin = rows > swa_win ? rows - swa_win : 0;
            for (std::size_t r = commit_begin; r < rows; ++r) {
                const std::size_t ring_idx = (position + r) % swa_win;
                std::memcpy(
                    st.attention.keys.data() + ring_idx * n_embd_k_gqa,
                    k.data() + r * n_embd_k_gqa,
                    n_embd_k_gqa * sizeof(float)
                );
                std::memcpy(
                    st.attention.values.data() + ring_idx * n_embd_k_gqa,
                    v.data() + r * n_embd_k_gqa,
                    n_embd_k_gqa * sizeof(float)
                );
            }
        } else {
            // Full attention: write directly into full-context KV cache.
            std::memcpy(
                st.attention.keys.data() + position * n_embd_k_gqa,
                k.data(),
                rows * n_embd_k_gqa * sizeof(float)
            );
            std::memcpy(
                st.attention.values.data() + position * n_embd_k_gqa,
                v.data(),
                rows * n_embd_k_gqa * sizeof(float)
            );

            DenseCompute::InklingAttention req{
                slot,
                ip,
                q,
                r_cur,
                sinks,
                l.attn_rel_bias,
                st.attention.keys,
                st.attention.values,
                k,
                v,
                rows,
                position,
                n_heads,
                n_kv_heads,
                head_dim,
                n_rel,
                n_rel_extent,
                0,
                qk_scale,
                attn_out
            };
            if (!compute_.attend_inkling_cached(req, weights_.cancel())) {
                weights_.parallel(rows * n_heads, [&](std::size_t work) {
                    const std::size_t r = work / n_heads;
                    const std::size_t h = work % n_heads;
                    const std::size_t kv_h = h / group;
                    const std::size_t pos_q = position + r;
                    const std::size_t visible = pos_q + 1;

                    const float* qv = q.data() + (r * n_heads + h) * head_dim;
                    const float* rv = r_cur.data() + (r * n_heads + h) * n_rel;
                    const float sink_logit = sinks[r * n_heads + h];

                    thread_local Vec scores;
                    scores.resize(visible);
                    float max_score = sink_logit;
                    for (std::size_t t = 0; t < visible; ++t) {
                        const float* kv =
                            st.attention.keys.data() + (t * n_kv_heads + kv_h) * head_dim;
                        float dot = 0.0f;
                        for (std::size_t d = 0; d < head_dim; ++d)
                            dot += qv[d] * kv[d];
                        float sc = dot * qk_scale;
                        const std::size_t dist = pos_q - t;
                        if (dist < n_rel_extent) {
                            const float* rb = l.attn_rel_bias.data() + dist * n_rel;
                            float rel_dot = 0.0f;
                            for (std::size_t d = 0; d < n_rel; ++d)
                                rel_dot += rv[d] * rb[d];
                            sc += rel_dot;
                        }
                        scores[t] = sc;
                        max_score = std::max(max_score, sc);
                    }

                    double denom = std::exp(double(sink_logit - max_score));
                    for (std::size_t t = 0; t < visible; ++t) {
                        scores[t] = std::exp(scores[t] - max_score);
                        denom += double(scores[t]);
                    }
                    const float inv_denom = float(1.0 / denom);
                    float* out = attn_out.data() + (r * n_heads + h) * head_dim;
                    for (std::size_t t = 0; t < visible; ++t) {
                        const float p = scores[t] * inv_denom;
                        const float* vv =
                            st.attention.values.data() + (t * n_kv_heads + kv_h) * head_dim;
                        for (std::size_t d = 0; d < head_dim; ++d)
                            out[d] += p * vv[d];
                    }
                });
            }
        }
        weights_.add_attention(
            std::chrono::duration<double>(std::chrono::steady_clock::now() - attn_started)
                .count()
        );

        // Sigmoid output gate -> attn_output projection -> attn_post_conv
        {
            const auto gate_row = [&](std::size_t r) {
                float* ao = attn_out.data() + r * n_embd_q;
                const float* gp = gate.data() + r * n_embd_q;
                for (std::size_t d = 0; d < n_embd_q; ++d)
                    ao[d] *= sigmoid(gp[d]);
            };
            if (rows >= 8)
                weights_.parallel(rows, gate_row);
            else
                for (std::size_t r = 0; r < rows; ++r)
                    gate_row(r);
        }
        Vec wo_out = weights_.project(compute_, l.wo, attn_out);
        Vec attn_final = convolve(
            weights_,
            wo_out,
            rows,
            n_embd,
            l.attn_post_conv,
            nullptr,
            0,
            st.conv,
            2 * n_embd_k_gqa
        );

        // Block residual stream update:
        // First layer of each 6-layer block pushes inpSA to blocks and starts a fresh stream.
        Vec ffn_inp;
        if (ip % n_block_size == 0) {
            block_storage.push_back(cur_in);
            ffn_inp = std::move(attn_final);
        } else {
            ffn_inp.resize(rows * n_embd);
            const auto add_row = [&](std::size_t r) {
                const float* a = cur_in.data() + r * n_embd;
                const float* b = attn_final.data() + r * n_embd;
                float* dst = ffn_inp.data() + r * n_embd;
                for (std::size_t d = 0; d < n_embd; ++d)
                    dst[d] = a[d] + b[d];
            };
            if (rows >= 8)
                weights_.parallel(rows, add_row);
            else
                for (std::size_t r = 0; r < rows; ++r)
                    add_row(r);
        }

        // MoE FFN input: mix(blocks, ffn_inp, ffn_res_query) -> optional loop ctrl -> ffn_norm
        block_refs.clear();
        for (const auto& b : block_storage)
            block_refs.push_back(&b);
        auto ffn_mixed = mix_states(weights_, block_refs, ffn_inp, l.ffn_res_query, 1, rows);
        Vec ffn_h = std::move(ffn_mixed[0]);
        if (messages) {
            std::array<Vec, 1> target{std::move(ffn_h)};
            apply_controls(
                std::span(target),
                std::span<const Vec>(messages->data() + 3, 1),
                *loop,
                3,
                rows
            );
            ffn_h = std::move(target[0]);
        }
        rms_norm_rows(weights_, ffn_h, l.ffn_norm, rows, rms_eps);

        // MoE routing: sigmoid(h @ ffn_gate_inp), top-8 by (probs + bias),
        // normalize selected probs by (sum + 1e-20), scale by 8.0.
        Vec router_logits = weights_.project(compute_, l.wffn_router, ffn_h);
        struct RowChoice {
            std::size_t expert;
            float weight;
        };
        std::vector<std::array<RowChoice, n_experts_used>> row_choices(rows);
        {
            const auto route_row = [&](std::size_t r) {
                const float* lr = router_logits.data() + r * n_experts;
                std::array<float, n_experts> probs;
                std::array<float, n_experts> sel_scores;
                std::array<std::size_t, n_experts> idx;
                for (std::size_t e = 0; e < n_experts; ++e) {
                    probs[e] = sigmoid(lr[e]);
                    sel_scores[e] =
                        probs[e] + (l.has_exp_bias ? l.ffn_exp_probs_b[e] : 0.0f);
                    idx[e] = e;
                }
                std::partial_sort(
                    idx.begin(),
                    idx.begin() + n_experts_used,
                    idx.end(),
                    [&](std::size_t a, std::size_t b) {
                        return sel_scores[a] > sel_scores[b];
                    }
                );
                double sum = 0.0;
                for (std::size_t k = 0; k < n_experts_used; ++k)
                    sum += double(probs[idx[k]]);
                const float inv =
                    expert_weights_scale / float(sum + double(expert_weights_norm_eps));
                for (std::size_t k = 0; k < n_experts_used; ++k) {
                    const std::size_t e = idx[k];
                    row_choices[r][k] = {e, probs[e] * inv};
                }
            };
            if (rows >= 16)
                weights_.parallel(rows, route_row);
            else
                for (std::size_t r = 0; r < rows; ++r)
                    route_row(r);
        }

        struct Route {
            std::size_t row;
            float weight;
        };
        struct ScatterRef {
            std::size_t active_idx;
            std::size_t slot_in_expert;
            float weight;
        };
        std::array<std::vector<Route>, n_experts> assignments;
        std::array<std::size_t, n_experts> expert_to_active;
        expert_to_active.fill(n_experts);
        std::vector<std::array<ScatterRef, n_experts_used>> row_scatter(rows);

        for (std::size_t r = 0; r < rows; ++r) {
            for (std::size_t k = 0; k < n_experts_used; ++k) {
                const auto& ch = row_choices[r][k];
                const std::size_t pos = assignments[ch.expert].size();
                assignments[ch.expert].push_back({r, ch.weight});
                row_scatter[r][k] = {ch.expert, pos, ch.weight};
            }
        }

        // Pack active experts for batched Strata projection (VRAM + RAM-CPU + Disk-stream).
        std::vector<std::size_t> active_experts;
        active_experts.reserve(n_experts);
        for (std::size_t e = 0; e < n_experts; ++e) {
            if (assignments[e].empty())
                continue;
            expert_to_active[e] = active_experts.size();
            active_experts.push_back(e);
        }

        std::vector<Vec> expert_inputs;
        std::vector<GgufWeights::Projection> gate_up_jobs;
        gate_up_jobs.reserve(active_experts.size());
        if (rows == 1) {
            // Single decode row: all 8 experts share ffn_h so dense_gguf_batch uploads once.
            for (std::size_t i = 0; i < active_experts.size(); ++i)
                gate_up_jobs.push_back({l.wffn_gate_up, ffn_h, active_experts[i]});
        } else {
            expert_inputs.resize(active_experts.size());
            weights_.parallel(active_experts.size(), [&](std::size_t i) {
                const auto& routes = assignments[active_experts[i]];
                auto& inp = expert_inputs[i];
                inp.resize(routes.size() * n_embd);
                for (std::size_t j = 0; j < routes.size(); ++j)
                    std::memcpy(
                        inp.data() + j * n_embd,
                        ffn_h.data() + routes[j].row * n_embd,
                        n_embd * sizeof(float)
                    );
            });
            for (std::size_t i = 0; i < active_experts.size(); ++i)
                gate_up_jobs.push_back({l.wffn_gate_up, expert_inputs[i], active_experts[i]});
        }
        auto gate_up_outs = weights_.project(compute_, gate_up_jobs);

        // SiTU activation:
        //   act = (situ_beta * tanh(gate / situ_beta) * sigmoid(gate))
        //       * (situ_linear_beta * tanh(up / situ_linear_beta))
        std::vector<Vec> act_outs(active_experts.size());
        weights_.parallel(active_experts.size(), [&](std::size_t i) {
            const std::size_t m = assignments[active_experts[i]].size();
            const auto& gu = gate_up_outs[i];
            auto& act = act_outs[i];
            act.resize(m * n_ff_exp);
            const float inv_beta = 1.0f / situ_beta;
            const float inv_lb = 1.0f / situ_linear_beta;
            for (std::size_t r = 0; r < m; ++r) {
                const float* gate_p = gu.data() + r * (2 * n_ff_exp);
                const float* up_p = gate_p + n_ff_exp;
                float* dst = act.data() + r * n_ff_exp;
                for (std::size_t d = 0; d < n_ff_exp; ++d) {
                    const float g = gate_p[d];
                    const float u = up_p[d];
                    const float g_act =
                        situ_beta * std::tanh(g * inv_beta) * sigmoid(g);
                    const float u_act =
                        situ_linear_beta * std::tanh(u * inv_lb);
                    dst[d] = g_act * u_act;
                }
            }
        });

        std::vector<GgufWeights::Projection> down_jobs;
        down_jobs.reserve(active_experts.size());
        for (std::size_t i = 0; i < active_experts.size(); ++i)
            down_jobs.push_back({l.wffn_down, act_outs[i], active_experts[i]});
        auto down_outs = weights_.project(compute_, down_jobs);

        Vec moe_out(rows * n_embd);
        {
            const auto scatter_row = [&](std::size_t r) {
                float* dst = moe_out.data() + r * n_embd;
                {
                    const auto& ref0 = row_scatter[r][0];
                    const float w0 = ref0.weight;
                    const float* src0 =
                        down_outs[expert_to_active[ref0.active_idx]].data()
                        + ref0.slot_in_expert * n_embd;
                    for (std::size_t d = 0; d < n_embd; ++d)
                        dst[d] = w0 * src0[d];
                }
                for (std::size_t k = 1; k < n_experts_used; ++k) {
                    const auto& ref = row_scatter[r][k];
                    const float w = ref.weight;
                    const float* src =
                        down_outs[expert_to_active[ref.active_idx]].data()
                        + ref.slot_in_expert * n_embd;
                    for (std::size_t d = 0; d < n_embd; ++d)
                        dst[d] += w * src[d];
                }
            };
            if (rows >= 8)
                weights_.parallel(rows, scatter_row);
            else
                for (std::size_t r = 0; r < rows; ++r)
                    scatter_row(r);
        }

        // ffn_post_conv + residual add
        Vec ffn_final = convolve(
            weights_,
            moe_out,
            rows,
            n_embd,
            l.ffn_post_conv,
            nullptr,
            0,
            st.conv,
            2 * n_embd_k_gqa + n_embd
        );
        Vec out(rows * n_embd);
        {
            const auto add_row = [&](std::size_t r) {
                const float* a = ffn_inp.data() + r * n_embd;
                const float* b = ffn_final.data() + r * n_embd;
                float* dst = out.data() + r * n_embd;
                for (std::size_t d = 0; d < n_embd; ++d)
                    dst[d] = a[d] + b[d];
            };
            if (rows >= 8)
                weights_.parallel(rows, add_row);
            else
                for (std::size_t r = 0; r < rows; ++r)
                    add_row(r);
        }
        return out;
    }

    std::unique_ptr<InklingGpuExecutor> try_create_gpu_executor() {
        InklingGpuModelDesc desc;
        desc.context = context_;
        desc.out_norm = out_norm_;
        desc.out_res_query = out_res_query_;
        desc.output_weight = weights_.describe_matrix("output.weight");
        std::size_t total_bytes = n_vocab_unpadded * desc.output_weight.row_bytes;

        auto bind_mat = [&](const std::string& name, std::size_t expert = 0) {
            auto m = weights_.describe_matrix(name, expert);
            total_bytes += m.out * m.row_bytes;
            return m;
        };

        desc.layers.resize(n_phys);
        for (std::size_t i = 0; i < n_phys; ++i) {
            const auto& src = layers_[i];
            auto& dst = desc.layers[i];
            dst.is_swa = src.is_swa;
            dst.has_attn_res = src.has_attn_res;
            dst.has_exp_bias = src.has_exp_bias;
            dst.attn_norm = src.attn_norm;
            dst.attn_q_inp_norm = src.attn_q_inp_norm;
            dst.attn_k_inp_norm = src.attn_k_inp_norm;
            dst.attn_v_inp_norm = src.attn_v_inp_norm;
            dst.attn_q_clip = src.attn_q_clip;
            dst.attn_rel_bias = src.attn_rel_bias;
            dst.attn_rel_clip = src.attn_rel_clip;
            for (std::size_t k = 0; k < n_conv; ++k) {
                dst.attn_k_conv[k] = src.attn_k_conv.tap[k];
                dst.attn_v_conv[k] = src.attn_v_conv.tap[k];
                dst.attn_post_conv[k] = src.attn_post_conv.tap[k];
                dst.ffn_post_conv[k] = src.ffn_post_conv.tap[k];
            }
            dst.attn_res_query = src.attn_res_query;
            dst.ffn_res_query = src.ffn_res_query;
            dst.ffn_norm = src.ffn_norm;
            dst.ffn_exp_probs_b = src.ffn_exp_probs_b;

            dst.wq = bind_mat(src.wq);
            dst.wk = bind_mat(src.wk);
            dst.wv = bind_mat(src.wv);
            dst.wo = bind_mat(src.wo);
            dst.wgate = bind_mat(src.wgate);
            dst.wrel = bind_mat(src.wrel);
            dst.wsink = bind_mat(src.wsink);
            if (!src.is_swa) {
                dst.wk_conv_proj = bind_mat(src.wk_conv_proj);
                dst.wv_conv_proj = bind_mat(src.wv_conv_proj);
            }
            dst.wffn_router = bind_mat(src.wffn_router);
            for (std::size_t e = 0; e < n_experts; ++e) {
                dst.wffn_gate_up[e] = bind_mat(src.wffn_gate_up, e);
                dst.wffn_down[e] = bind_mat(src.wffn_down, e);
            }
        }
        desc.loop_layers.resize(loop_core_len);
        for (std::size_t i = 0; i < loop_core_len; ++i) {
            const auto& src = loop_layers_[i];
            auto& dst = desc.loop_layers[i];
            dst.msg_query = src.msg_query;
            for (std::size_t c = 0; c < loop_ctrl_count; ++c) {
                dst.ctrl_down[c] = bind_mat(src.ctrl_down[c]);
                dst.ctrl_gain[c] = bind_mat(src.ctrl_gain[c]);
                dst.ctrl_forget[c] = bind_mat(src.ctrl_forget[c]);
            }
        }
        desc.total_matrix_bytes = total_bytes;
        desc.initial_variant_id = active_variant_id_;
        return compute_.create_inkling_gpu(std::move(desc), weights_.cancel());
    }

    const GgufWeights& weights_;
    DenseCompute& compute_;
    std::size_t context_ = 0;
    std::uint32_t active_variant_id_ = 0;

    Vec tok_norm_, out_norm_, out_res_query_;
    std::vector<PhysLayer> layers_;
    std::vector<LoopLayer> loop_layers_;
    std::vector<InklingLayerState> slots_;

    bool gpu_tried_ = false;
    bool host_state_stale_ = false;
    bool gpu_needs_upload_ = false;
    std::size_t gpu_valid_tokens_ = 0;
    std::unique_ptr<InklingGpuExecutor> gpu_;
};
}  // namespace

void validate_inkling2_loop(const checkpoint::GgufMetadata& m) {
    validate_inkling2_metadata(m);
}

void validate_inkling2_loop(const checkpoint::GgufCheckpoint& c) {
    validate_inkling2_metadata(c);
}

void bind_inkling2_loop(GgufWeights& w) {
    validate_inkling2_loop(w.gguf());
    w.table("token_embd.weight", n_vocab_padded, n_embd);
    w.vector("token_embd_norm.weight", {n_embd});
    w.matrix("output.weight", n_vocab_padded, n_embd);
    w.vector("output_norm.weight", {n_embd});
    w.vector("output_res_query.weight", {n_embd});

    for (std::size_t i = 0; i < n_phys; ++i) {
        const bool swa = is_swa_phys(i);
        w.matrix(blk(i, "attn_q.weight"), n_embd_q, n_embd);
        w.matrix(blk(i, "attn_k.weight"), n_embd_k_gqa, n_embd);
        w.matrix(blk(i, "attn_v.weight"), n_embd_k_gqa, n_embd);
        w.matrix(blk(i, "attn_output.weight"), n_embd, n_embd_q);
        w.matrix(blk(i, "attn_gate.weight"), n_embd_q, n_embd);
        w.matrix(blk(i, "attn_rel.weight"), n_rel * n_heads, n_embd);
        w.matrix(blk(i, "attn_sink_proj.weight"), n_heads, n_embd);

        w.table(blk(i, "attn_q_clip.weight"), n_heads, head_dim);
        w.table(blk(i, "attn_rel_bias.weight"), n_rel_extent, n_rel);
        w.table(blk(i, "attn_rel_clip.weight"), n_heads, n_rel);

        w.table(blk(i, "attn_k_conv.weight"), n_embd_k_gqa, n_conv);
        w.table(blk(i, "attn_v_conv.weight"), n_embd_k_gqa, n_conv);
        w.table(blk(i, "attn_post_conv.weight"), n_embd, n_conv);
        w.table(blk(i, "ffn_post_conv.weight"), n_embd, n_conv);

        if (swa) {
            w.vector(blk(i, "attn_norm.weight"), {n_embd});
        } else {
            w.vector(blk(i, "attn_q_inp_norm.weight"), {n_embd});
            w.vector(blk(i, "attn_k_inp_norm.weight"), {n_embd});
            w.vector(blk(i, "attn_v_inp_norm.weight"), {n_embd});
            w.matrix(blk(i, "attn_k_conv_proj.weight"), n_conv * n_kv_heads, n_embd);
            w.matrix(blk(i, "attn_v_conv_proj.weight"), n_conv * n_kv_heads, n_embd);
        }

        if (i > 0)
            bind_small_2d_or_1d(w, blk(i, "attn_res_query.weight"), swa ? 1 : 3, n_embd);
        w.vector(blk(i, "ffn_res_query.weight"), {n_embd});
        w.vector(blk(i, "ffn_norm.weight"), {n_embd});
        w.matrix(blk(i, "ffn_gate_inp.weight"), n_experts, n_embd);

        const auto bias_name = blk(i, "exp_probs_b.bias");
        for (std::size_t t = 0; t < w.gguf().tensor_count(); ++t) {
            if (w.gguf().tensor_at(t).name == bias_name) {
                w.vector(bias_name, {n_experts});
                break;
            }
        }
        w.matrix(blk(i, "ffn_gate_up_exps.weight"), 2 * n_ff_exp, n_embd, n_experts);
        w.matrix(blk(i, "ffn_down_exps.weight"), n_embd, n_ff_exp, n_experts);
    }

    for (std::size_t ip = loop_start; ip < loop_end; ++ip) {
        w.table(blk(ip, "loop_msg_query.weight"), loop_ctrl_count, n_embd);
        for (std::size_t c = 0; c < loop_ctrl_count; ++c) {
            const std::string prefix = std::string("loop_") + ctrl_names[c] + "_ctrl_";
            w.matrix(blk(ip, (prefix + "down.weight").c_str()), loop_gate_rank, 2 * n_embd);
            w.matrix(blk(ip, (prefix + "gain.weight").c_str()), n_embd, loop_gate_rank);
            w.matrix(blk(ip, (prefix + "forget.weight").c_str()), n_embd, loop_gate_rank);
        }
    }
    w.finish();
}

std::size_t inkling2_loop_state_bytes(std::size_t context) {
    std::size_t floats = 0;
    for (std::size_t s = 0; s < n_slots; ++s) {
        const auto cap = slot_kv_capacity(s, context);
        floats += 2 * cap * n_embd_k_gqa + n_conv_hist * n_conv_channels;
    }
    return floats * sizeof(float);
}

std::size_t inkling2_loop_device_state_bytes(std::size_t context) {
    // Mirrored KV caches stay in FP16 (2 bytes/value); short-conv state,
    // relative-position bias tables and small resident layer params stay F32.
    // Includes 2 bounded GPU boundary snapshots (40 FP16 SWA windows + 48 conv
    // states each) for zero-reupload prefix rewind across turns.
    std::size_t total = 0;
    for (std::size_t s = 0; s < n_slots; ++s) {
        const std::size_t cap = is_swa_phys(slot_phys(s))
                                    ? 2 * slot_kv_capacity(s, context)
                                    : context;
        total += 2 * cap * n_embd_k_gqa * sizeof(std::uint16_t);
        total += n_conv_hist * n_conv_channels * sizeof(float);
    }
    total += InklingGeometry::n_gpu_boundary_slots * inkling_gpu_boundary_bytes(context);
    total += n_phys * n_rel_extent * n_rel * sizeof(float);
    total += 16ULL * 1024 * 1024;  // small layer params, pointer tables & step state
    return total;
}

std::unique_ptr<Model> inkling2_loop_model(
    const GgufWeights& weights,
    DenseCompute& compute,
    std::size_t context,
    std::uint32_t initial_variant_id
) {
    return std::make_unique<Inkling2LoopModel>(
        weights, compute, context, initial_variant_id
    );
}
}  // namespace kadan::text
