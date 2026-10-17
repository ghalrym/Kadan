#pragma once
// Layers shared by the qwen35 and qwen4exp GGUF graphs: the gated delta net
// and the gated full attention with per-head Q/K RMSNorm and partial RoPE.
#include "gguf_model.hpp"

namespace kadan::text {
struct DeltaNetSpec {
    std::size_t hidden, key_heads, value_heads, dim, conv;
    float eps;
    // qwen4exp gates the normalized output with sigmoid; qwen35 with SiLU.
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
    static std::size_t state_bytes(const DeltaNetSpec&);

private:
    DeltaNetSpec spec_;
    std::string prefix_;
    Vec conv_weight_, dt_, a_, norm_, conv_, state_;
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

private:
    AttentionSpec spec_;
    std::string prefix_;
    std::size_t context_;
    Vec q_norm_, k_norm_, keys_, values_;
};
}  // namespace kadan::text
