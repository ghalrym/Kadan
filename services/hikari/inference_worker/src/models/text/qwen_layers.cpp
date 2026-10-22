// Gated delta net, gated attention and routed experts: llama.cpp
// src/models/qwen35moe.cpp, qwen4exp.cpp, delta-net-base.cpp and
// llama-graph.cpp build_moe_ffn at 6420cf5dd588c68e48657a0bbd82bb3bd5ac7209.
// The converter tiles V heads, so value head h reads key head h % key_heads.
#include "qwen_layers.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <numeric>

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

std::size_t DeltaNetLayer::device_bytes(const DeltaNetSpec& s) {
    return state_bytes(s)
           + (s.channels() * s.conv + 2 * s.value_heads + s.dim) * sizeof(float);
}

void DeltaNetLayer::reset() {
    std::fill(conv_.begin(), conv_.end(), 0.f);
    std::fill(state_.begin(), state_.end(), 0.f);
    mode_ = Mode::unknown;
}

RecurrentState DeltaNetLayer::take_state(DenseCompute& compute) {
    if (mode_ == Mode::device)
        compute.copy_recurrent_state(
            std::uint64_t(reinterpret_cast<std::uintptr_t>(this)), state_, conv_
        );
    return {std::move(conv_), std::move(state_), mode_ != Mode::host};
}

void DeltaNetLayer::restore_state(RecurrentState state) {
    need(
        state.convolution.size() == (spec_.conv - 1) * spec_.channels()
            && state.recurrent.size() == spec_.value_heads * spec_.dim * spec_.dim,
        "gguf_text_recurrent_snapshot_shape"
    );
    conv_.swap(state.convolution);
    state_.swap(state.recurrent);
    mode_ = state.device ? Mode::restored : Mode::host;
}

Vec DeltaNetLayer::forward(
    DenseCompute& compute, const GgufWeights& w, std::span<const float> input
) {
    const auto& s = spec_;
    const auto rows = input.size() / s.hidden, channels = s.channels(),
               inner = s.value_heads * s.dim, history = s.conv - 1;
    const std::array<GgufWeights::Projection, 4> requests{
        {{prefix_ + "attn_qkv.weight", input},
         {prefix_ + "attn_gate.weight", input},
         {prefix_ + "ssm_alpha.weight", input},
         {prefix_ + "ssm_beta.weight", input}}
    };
    const auto projected = w.project(compute, requests);
    const auto &qkv = projected[0], &z = projected[1], &alpha = projected[2],
               &beta = projected[3];
    const float scale = 1 / std::sqrt(float(s.dim));
    need(s.dim <= 256, "gguf_text_delta_shape");
    const auto started = std::chrono::steady_clock::now();
    Vec mixed(channels), output(rows * inner);
    if (mode_ != Mode::host) {
        const bool fresh = mode_ == Mode::unknown || mode_ == Mode::restored;
        const DenseCompute::DeltaStep step{
            std::uint64_t(reinterpret_cast<std::uintptr_t>(this)),
            fresh,
            qkv,
            z,
            alpha,
            beta,
            conv_weight_,
            dt_,
            a_,
            norm_,
            rows,
            s.key_heads,
            s.value_heads,
            s.dim,
            s.conv,
            s.eps,
            s.sigmoid_gate,
            output,
            mode_ == Mode::restored ? std::span<const float>(state_)
                                    : std::span<const float>{},
            mode_ == Mode::restored ? std::span<const float>(conv_)
                                    : std::span<const float>{}
        };
        if (compute.delta_step(step, w.cancel())) {
            mode_ = Mode::device;
            w.add_recurrent(
                std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - started
                )
                    .count()
            );
            return w.project(compute, prefix_ + "ssm_out.weight", output);
        }
        need(fresh, "gguf_text_recurrent_device_lost");
        mode_ = Mode::host;
    }
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
        // Heads own disjoint state and output slices.
        w.parallel(s.value_heads, [&](std::size_t h) {
            std::array<float, 256> query_storage, head_storage;
            const auto query = std::span(query_storage).first(s.dim);
            const auto head = std::span(head_storage).first(s.dim);
            const auto kh = h % s.key_heads;
            const float decay =
                std::exp(softplus(alpha[row * s.value_heads + h] + dt_[h]) * a_[h]);
            for (std::size_t d = 0; d < s.dim; ++d)
                query[d] = q[kh * s.dim + d] * scale;
            delta_rule(
                std::span(state_).subspan(h * s.dim * s.dim, s.dim * s.dim),
                query,
                k.subspan(kh * s.dim, s.dim),
                v.subspan(h * s.dim, s.dim),
                std::span(&decay, 1),
                sigmoid(beta[row * s.value_heads + h]),
                head
            );
            rms_norm(head, norm_, s.eps);
            for (std::size_t d = 0; d < s.dim; ++d) {
                const float gate = z[row * inner + h * s.dim + d];
                output[row * inner + h * s.dim + d] =
                    head[d] * (s.sigmoid_gate ? sigmoid(gate) : silu(gate));
            }
        });
    }
    w.add_recurrent(
        std::chrono::duration<double>(std::chrono::steady_clock::now() - started)
            .count()
    );
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
    const std::array<GgufWeights::Projection, 3> requests{
        {{prefix_ + "attn_q.weight", input},
         {prefix_ + "attn_k.weight", input},
         {prefix_ + "attn_v.weight", input}}
    };
    auto projected = w.project(compute, requests);
    const auto& joint = projected[0];
    auto& k = projected[1];
    const auto& v = projected[2];
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
    const auto started = std::chrono::steady_clock::now();
    // The device mirrors this layer's KV cache; the host copy stays complete.
    const auto cached = [&](std::span<const std::uint32_t> chosen,
                            std::span<const std::uint32_t> offsets) {
        return compute.attend_cached(
            std::uint64_t(reinterpret_cast<std::uintptr_t>(this)),
            q,
            std::span<const float>(keys_).first((position + rows) * kv_width),
            std::span<const float>(values_).first((position + rows) * kv_width),
            rows,
            position,
            s.heads,
            s.kv_heads,
            s.dim,
            scale,
            attended,
            w.cancel(),
            chosen,
            offsets
        );
    };
    if (dense) {
        if (!cached({}, {}))
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
    } else {
        // Every row's positions; an empty selection is every earlier one.
        std::vector<std::uint32_t> chosen, offsets{0};
        for (std::size_t row = 0; row < rows; ++row) {
            const auto& selected = (*sparse)[row];
            if (selected.empty())
                for (std::size_t t = 0; t <= position + row; ++t)
                    chosen.push_back(std::uint32_t(t));
            for (const auto t : selected) {
                need(t <= position + row, "gguf_text_sparse_future");
                chosen.push_back(std::uint32_t(t));
            }
            offsets.push_back(std::uint32_t(chosen.size()));
        }
        if (!cached(chosen, offsets))
            for (std::size_t row = 0; row < rows; ++row) {
                const auto first = offsets[row], count = offsets[row + 1] - first;
                Vec gathered_k(count * kv_width), gathered_v(gathered_k.size());
                for (std::size_t i = 0; i < count; ++i) {
                    std::copy_n(
                        keys_.begin() + chosen[first + i] * kv_width,
                        kv_width,
                        gathered_k.begin() + i * kv_width
                    );
                    std::copy_n(
                        values_.begin() + chosen[first + i] * kv_width,
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
                    count,
                    s.heads,
                    s.kv_heads,
                    s.dim,
                    count - 1,
                    scale,
                    std::span(attended).subspan(row * width, width),
                    w.cancel()
                );
            }
    }
    w.add_attention(
        std::chrono::duration<double>(std::chrono::steady_clock::now() - started)
            .count()
    );
    for (std::size_t i = 0; i < attended.size(); ++i)
        attended[i] *= sigmoid(gate[i]);
    return w.project(compute, prefix_ + "attn_output.weight", attended);
}

void bind_moe(GgufWeights& w, const std::string& p, const MoeSpec& s) {
    w.matrix(p + "ffn_gate_inp.weight", s.experts, s.hidden);
    w.matrix(p + "ffn_gate_exps.weight", s.ff, s.hidden, s.experts);
    w.matrix(p + "ffn_up_exps.weight", s.ff, s.hidden, s.experts);
    w.matrix(p + "ffn_down_exps.weight", s.hidden, s.ff, s.experts);
    w.vector(p + "ffn_gate_inp_shexp.weight", {s.hidden});
    w.matrix(p + "ffn_gate_shexp.weight", s.shared_ff, s.hidden);
    w.matrix(p + "ffn_up_shexp.weight", s.shared_ff, s.hidden);
    w.matrix(p + "ffn_down_shexp.weight", s.hidden, s.shared_ff);
}

MoeLayer::MoeLayer(const GgufWeights& w, std::string prefix, const MoeSpec& spec)
    : spec_(spec), prefix_(std::move(prefix)),
      shared_gate_(w.read(prefix_ + "ffn_gate_inp_shexp.weight")) {
}

Vec MoeLayer::forward(
    DenseCompute& compute, const GgufWeights& w, std::span<const float> x
) const {
    const auto& s = spec_;
    const auto& p = prefix_;
    const auto rows = x.size() / s.hidden;
    const std::array<GgufWeights::Projection, 3> first{
        {{p + "ffn_gate_inp.weight", x},
         {p + "ffn_gate_shexp.weight", x},
         {p + "ffn_up_shexp.weight", x}}
    };
    auto routed_and_shared = w.project(compute, first);
    const auto& logits = routed_and_shared[0];
    auto& shared = routed_and_shared[1];
    {
        const auto& up = routed_and_shared[2];
        for (std::size_t j = 0; j < shared.size(); ++j)
            shared[j] = silu(shared[j]) * up[j];
    }
    // Softmax router, top-k by probability, renormalized; contributions
    // are summed in selection order as llama.cpp aggregates them.
    std::vector<std::size_t> chosen(rows * s.top_k);
    Vec weight(rows * s.top_k);
    std::vector<std::size_t> order(s.experts);
    Vec probability(s.experts);
    for (std::size_t row = 0; row < rows; ++row) {
        const auto* l = logits.data() + row * s.experts;
        const float maximum = *std::max_element(l, l + s.experts);
        double total = 0;
        for (std::size_t e = 0; e < s.experts; ++e)
            total += probability[e] = std::exp(l[e] - maximum);
        for (auto& v : probability)
            v = float(v / total);
        std::iota(order.begin(), order.end(), 0);
        std::partial_sort(
            order.begin(), order.begin() + s.top_k, order.end(), [&](auto a, auto b) {
                return probability[a] > probability[b];
            }
        );
        float sum = 0;
        for (std::size_t i = 0; i < s.top_k; ++i) {
            chosen[row * s.top_k + i] = order[i];
            sum += weight[row * s.top_k + i] = probability[order[i]];
        }
        sum = std::max(sum, 6.103515625e-5f);
        for (std::size_t i = 0; i < s.top_k; ++i)
            weight[row * s.top_k + i] /= sum;
    }

    // Every selected expert gathers its routed rows; gate/up, then down, each
    // run as one batch, with the shared expert's down projection in the second.
    struct Routed {
        std::size_t expert;
        std::vector<std::size_t> slots;
        Vec gathered;
    };

    std::vector<Routed> active;
    for (std::size_t e = 0; e < s.experts; ++e) {
        std::vector<std::size_t> slots;
        for (std::size_t slot = 0; slot < rows * s.top_k; ++slot)
            if (chosen[slot] == e)
                slots.push_back(slot);
        if (slots.empty())
            continue;
        Vec gathered(slots.size() * s.hidden);
        for (std::size_t r = 0; r < slots.size(); ++r)
            std::copy_n(
                x.begin() + slots[r] / s.top_k * s.hidden,
                s.hidden,
                gathered.begin() + r * s.hidden
            );
        active.push_back({e, std::move(slots), std::move(gathered)});
    }
    std::vector<GgufWeights::Projection> requests;
    requests.reserve(2 * active.size());
    for (const auto& a : active) {
        requests.push_back({p + "ffn_gate_exps.weight", a.gathered, a.expert});
        requests.push_back({p + "ffn_up_exps.weight", a.gathered, a.expert});
    }
    auto gate_up = w.project(compute, requests);
    requests.clear();
    for (std::size_t i = 0; i < active.size(); ++i) {
        auto& gate = gate_up[2 * i];
        const auto& up = gate_up[2 * i + 1];
        for (std::size_t j = 0; j < gate.size(); ++j)
            gate[j] = silu(gate[j]) * up[j];
        requests.push_back({p + "ffn_down_exps.weight", gate, active[i].expert});
    }
    requests.push_back({p + "ffn_down_shexp.weight", shared});
    const auto downs = w.project(compute, requests);
    Vec contribution(rows * s.top_k * s.hidden);
    for (std::size_t i = 0; i < active.size(); ++i) {
        const auto& slots = active[i].slots;
        for (std::size_t r = 0; r < slots.size(); ++r)
            for (std::size_t d = 0; d < s.hidden; ++d)
                contribution[slots[r] * s.hidden + d] =
                    downs[i][r * s.hidden + d] * weight[slots[r]];
    }
    const auto& down = downs.back();
    Vec output(rows * s.hidden);
    for (std::size_t row = 0; row < rows; ++row) {
        float gate = 0;
        for (std::size_t c = 0; c < s.hidden; ++c)
            gate += shared_gate_[c] * x[row * s.hidden + c];
        gate = sigmoid(gate);
        for (std::size_t d = 0; d < s.hidden; ++d) {
            float sum = 0;
            for (std::size_t i = 0; i < s.top_k; ++i)
                sum += contribution[(row * s.top_k + i) * s.hidden + d];
            output[row * s.hidden + d] = sum + down[row * s.hidden + d] * gate;
        }
    }
    return output;
}
}  // namespace kadan::text
