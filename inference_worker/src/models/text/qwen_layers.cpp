// Gated delta net and gated attention: llama.cpp src/models/qwen35.cpp,
// qwen4exp.cpp and delta-net-base.cpp at 6420cf5dd588c68e48657a0bbd82bb3bd5ac7209.
// The converter tiles V heads, so value head h reads key head h % key_heads.
#include "qwen_layers.hpp"

#include <algorithm>
#include <cmath>

namespace kadan::text {
void bind_delta_net(GgufWeights& w, const std::string& p, const DeltaNetSpec& s) {
    const auto inner = s.value_heads * s.dim;
    w.matrix(p + "attn_qkv.weight", s.channels(), s.hidden);
    w.matrix(p + "attn_gate.weight", inner, s.hidden);
    w.vector(p + "ssm_conv1d.weight", {s.conv, s.channels()});
    w.vector(p + "ssm_dt.bias", {s.value_heads});
    w.vector(p + "ssm_a", {s.value_heads});
    w.matrix(p + "ssm_beta.weight", s.value_heads, s.hidden);
    w.matrix(p + "ssm_alpha.weight", s.value_heads, s.hidden);
    w.vector(p + "ssm_norm.weight", {s.dim});
    w.matrix(p + "ssm_out.weight", s.hidden, inner);
}

DeltaNetLayer::DeltaNetLayer(
    const GgufWeights& w, std::string prefix, const DeltaNetSpec& spec
)
    : spec_(spec), prefix_(std::move(prefix)),
      conv_weight_(w.read(prefix_ + "ssm_conv1d.weight")),
      dt_(w.read(prefix_ + "ssm_dt.bias")), a_(w.read(prefix_ + "ssm_a")),
      norm_(w.read(prefix_ + "ssm_norm.weight")),
      conv_((spec.conv - 1) * spec.channels()),
      state_(spec.value_heads * spec.dim * spec.dim) {
}

std::size_t DeltaNetLayer::state_bytes(const DeltaNetSpec& s) {
    return ((s.conv - 1) * s.channels() + s.value_heads * s.dim * s.dim)
           * sizeof(float);
}

void DeltaNetLayer::reset() {
    std::fill(conv_.begin(), conv_.end(), 0.f);
    std::fill(state_.begin(), state_.end(), 0.f);
}

Vec DeltaNetLayer::forward(
    DenseCompute& compute, const GgufWeights& w, std::span<const float> input
) {
    const auto& s = spec_;
    const auto rows = input.size() / s.hidden, channels = s.channels(),
               inner = s.value_heads * s.dim, history = s.conv - 1;
    const auto qkv = w.project(compute, prefix_ + "attn_qkv.weight", input);
    const auto z = w.project(compute, prefix_ + "attn_gate.weight", input);
    const auto alpha = w.project(compute, prefix_ + "ssm_alpha.weight", input);
    const auto beta = w.project(compute, prefix_ + "ssm_beta.weight", input);
    const float scale = 1 / std::sqrt(float(s.dim));
    Vec mixed(channels), output(rows * inner), query(s.dim), decay(1), head(s.dim);
    for (std::size_t row = 0; row < rows; ++row) {
        w.check_cancel();
        const auto* x = qkv.data() + row * channels;
        // Depthwise causal convolution over the raw history, oldest first.
        for (std::size_t c = 0; c < channels; ++c) {
            const auto* kernel = conv_weight_.data() + c * s.conv;
            float sum = 0;
            for (std::size_t k = 0; k < history; ++k)
                sum += conv_[k * channels + c] * kernel[k];
            sum += x[c] * kernel[history];
            mixed[c] = silu(sum);
        }
        std::move(conv_.begin() + channels, conv_.end(), conv_.begin());
        std::copy(x, x + channels, conv_.end() - channels);
        auto q = std::span(mixed).first(s.key_heads * s.dim);
        auto k = std::span(mixed).subspan(s.key_heads * s.dim, s.key_heads * s.dim);
        const auto v = std::span(mixed).subspan(2 * s.key_heads * s.dim);
        l2_norm(q, s.dim, s.eps);
        l2_norm(k, s.dim, s.eps);
        for (std::size_t h = 0; h < s.value_heads; ++h) {
            const auto kh = h % s.key_heads;
            decay[0] =
                std::exp(softplus(alpha[row * s.value_heads + h] + dt_[h]) * a_[h]);
            for (std::size_t d = 0; d < s.dim; ++d)
                query[d] = q[kh * s.dim + d] * scale;
            delta_rule(
                std::span(state_).subspan(h * s.dim * s.dim, s.dim * s.dim),
                query,
                k.subspan(kh * s.dim, s.dim),
                v.subspan(h * s.dim, s.dim),
                decay,
                sigmoid(beta[row * s.value_heads + h]),
                head
            );
            rms_norm(head, norm_, s.eps);
            for (std::size_t d = 0; d < s.dim; ++d) {
                const float gate = z[row * inner + h * s.dim + d];
                output[row * inner + h * s.dim + d] =
                    head[d] * (s.sigmoid_gate ? sigmoid(gate) : silu(gate));
            }
        }
    }
    return w.project(compute, prefix_ + "ssm_out.weight", output);
}

void bind_attention(GgufWeights& w, const std::string& p, const AttentionSpec& s) {
    w.matrix(p + "attn_q.weight", 2 * s.heads * s.dim, s.hidden);
    w.matrix(p + "attn_k.weight", s.kv_heads * s.dim, s.hidden);
    w.matrix(p + "attn_v.weight", s.kv_heads * s.dim, s.hidden);
    w.vector(p + "attn_q_norm.weight", {s.dim});
    w.vector(p + "attn_k_norm.weight", {s.dim});
    w.matrix(p + "attn_output.weight", s.hidden, s.heads * s.dim);
}

AttentionLayer::AttentionLayer(
    const GgufWeights& w,
    std::string prefix,
    const AttentionSpec& spec,
    std::size_t context
)
    : spec_(spec), prefix_(std::move(prefix)), context_(context),
      q_norm_(w.read(prefix_ + "attn_q_norm.weight")),
      k_norm_(w.read(prefix_ + "attn_k_norm.weight")),
      keys_(context * spec.kv_heads * spec.dim), values_(keys_.size()) {
}

std::size_t AttentionLayer::state_bytes(const AttentionSpec& s, std::size_t context) {
    return 2 * context * s.kv_heads * s.dim * sizeof(float);
}

Vec AttentionLayer::forward(
    DenseCompute& compute,
    const GgufWeights& w,
    std::span<const float> input,
    std::size_t position,
    const std::vector<std::vector<std::size_t>>* sparse
) {
    const auto& s = spec_;
    const auto rows = input.size() / s.hidden, width = s.heads * s.dim,
               kv_width = s.kv_heads * s.dim;
    need(position + rows <= context_, "gguf_text_context_exceeded");
    need(!sparse || sparse->size() == rows, "gguf_text_sparse_rows");
    const auto joint = w.project(compute, prefix_ + "attn_q.weight", input);
    auto k = w.project(compute, prefix_ + "attn_k.weight", input);
    const auto v = w.project(compute, prefix_ + "attn_v.weight", input);
    Vec q(rows * width), gate(rows * width);
    for (std::size_t row = 0; row < rows; ++row)
        for (std::size_t h = 0; h < s.heads; ++h) {
            const auto* from = joint.data() + row * 2 * width + h * 2 * s.dim;
            std::copy(from, from + s.dim, q.begin() + row * width + h * s.dim);
            std::copy(
                from + s.dim, from + 2 * s.dim, gate.begin() + row * width + h * s.dim
            );
        }
    rms_norm(q, q_norm_, s.eps);
    rms_norm(k, k_norm_, s.eps);
    for (std::size_t row = 0; row < rows; ++row) {
        rope(
            std::span(q).subspan(row * width, width),
            s.dim,
            s.rotary,
            position + row,
            s.base
        );
        rope(
            std::span(k).subspan(row * kv_width, kv_width),
            s.dim,
            s.rotary,
            position + row,
            s.base
        );
    }
    std::copy(k.begin(), k.end(), keys_.begin() + position * kv_width);
    std::copy(v.begin(), v.end(), values_.begin() + position * kv_width);
    Vec attended(rows * width);
    const float scale = 1 / std::sqrt(float(s.dim));
    const bool dense =
        !sparse
        || std::all_of(sparse->begin(), sparse->end(), [](const auto& selected) {
               return selected.empty();
           });
    if (dense)
        attend(
            compute,
            q,
            keys_,
            values_,
            rows,
            position + rows,
            s.heads,
            s.kv_heads,
            s.dim,
            position,
            scale,
            attended,
            w.cancel()
        );
    else
        for (std::size_t row = 0; row < rows; ++row) {
            auto selected = (*sparse)[row];
            if (selected.empty())
                for (std::size_t t = 0; t <= position + row; ++t)
                    selected.push_back(t);
            Vec gathered_k(selected.size() * kv_width), gathered_v(gathered_k.size());
            for (std::size_t i = 0; i < selected.size(); ++i) {
                need(selected[i] <= position + row, "gguf_text_sparse_future");
                std::copy_n(
                    keys_.begin() + selected[i] * kv_width,
                    kv_width,
                    gathered_k.begin() + i * kv_width
                );
                std::copy_n(
                    values_.begin() + selected[i] * kv_width,
                    kv_width,
                    gathered_v.begin() + i * kv_width
                );
            }
            // One query attending every gathered key: its causal offset is last.
            attend(
                compute,
                std::span<const float>(q).subspan(row * width, width),
                gathered_k,
                gathered_v,
                1,
                selected.size(),
                s.heads,
                s.kv_heads,
                s.dim,
                selected.size() - 1,
                scale,
                std::span(attended).subspan(row * width, width),
                w.cancel()
            );
        }
    for (std::size_t i = 0; i < attended.size(); ++i)
        attended[i] *= sigmoid(gate[i]);
    return w.project(compute, prefix_ + "attn_output.weight", attended);
}
}  // namespace kadan::text
