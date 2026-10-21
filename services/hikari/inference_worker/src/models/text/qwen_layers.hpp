#pragma once
// Layers shared by the qwen35moe and qwen4exp GGUF graphs: the gated delta net,
// the gated full attention with per-head Q/K RMSNorm and partial RoPE, and the
// softmax-routed experts with a sigmoid-gated shared expert.
#include "gguf_model.hpp"

namespace kadan::text {
struct DeltaNetSpec {
    std::size_t hidden, key_heads, value_heads, dim, conv;
    float eps;
    // qwen4exp gates the normalized output with sigmoid; qwen35moe with SiLU.
    bool sigmoid_gate;

    std::size_t channels() const {
        return dim * (2 * key_heads + value_heads);
    }
};

// Binds attn_qkv, attn_gate, ssm_* for one layer.
void bind_delta_net(GgufWeights&, const std::string& prefix, const DeltaNetSpec&);

class DeltaNetLayer {
public:
    DeltaNetLayer(const GgufWeights&, std::string prefix, const DeltaNetSpec&);
    // Rows of `hidden` in, rows of `hidden` out (after ssm_out).
    Vec forward(DenseCompute&, const GgufWeights&, std::span<const float> input);
    void reset();
    RecurrentState take_state(DenseCompute&);
    void restore_state(RecurrentState);
    static std::size_t state_bytes(const DeltaNetSpec&);
    // Device bytes of DenseCompute::delta_step state and constants.
    static std::size_t device_bytes(const DeltaNetSpec&);

private:
    DeltaNetSpec spec_;
    std::string prefix_;
    Vec conv_weight_, dt_, a_, norm_, conv_, state_;
    // Each request keeps its recurrence on the device or on the host, decided by
    // its first step: neither copy of the state is ever partial.
    enum class Mode {
        unknown,
        device,
        restored,
        host
    } mode_ = Mode::unknown;
};

struct AttentionSpec {
    std::size_t hidden, heads, kv_heads, dim, rotary;
    float base, eps;
};

// Binds attn_q (query|gate interleaved per head), attn_k/v, Q/K norms, output.
void bind_attention(GgufWeights&, const std::string& prefix, const AttentionSpec&);

class AttentionLayer {
public:
    AttentionLayer(
        const GgufWeights&,
        std::string prefix,
        const AttentionSpec&,
        std::size_t context
    );
    // Rows at positions position.. ; `sparse`, when given, lists for each row
    // the cached positions it attends (an empty list means dense causal).
    Vec forward(
        DenseCompute&,
        const GgufWeights&,
        std::span<const float> input,
        std::size_t position,
        const std::vector<std::vector<std::size_t>>* sparse = nullptr
    );
    static std::size_t state_bytes(const AttentionSpec&, std::size_t context);

    AttentionState take_state() {
        return {std::move(keys_), std::move(values_)};
    }

    void restore_state(AttentionState state) {
        need(
            state.keys.size() == keys_.size() && state.values.size() == values_.size(),
            "gguf_text_attention_snapshot_shape"
        );
        keys_.swap(state.keys);
        values_.swap(state.values);
    }

private:
    AttentionSpec spec_;
    std::string prefix_;
    std::size_t context_;
    Vec q_norm_, k_norm_, keys_, values_;
};

struct MoeSpec {
    std::size_t hidden, experts, top_k, ff, shared_ff;
};

// Binds the router, the expert stacks and the shared expert of one layer.
void bind_moe(GgufWeights&, const std::string& prefix, const MoeSpec&);

// llama.cpp build_moe_ffn with softmax gating and renormalized top-k weights,
// plus the shared expert scaled by sigmoid(ffn_gate_inp_shexp . x).
class MoeLayer {
public:
    MoeLayer(const GgufWeights&, std::string prefix, const MoeSpec&);
    Vec forward(DenseCompute&, const GgufWeights&, std::span<const float> input) const;

private:
    MoeSpec spec_;
    std::string prefix_;
    Vec shared_gate_;
};
}  // namespace kadan::text
