#pragma once
#include "kadan/quantization.hpp"
#include "kadan/sequence_state.hpp"
#include <memory>
namespace kadan::full {
struct Config {
    std::size_t hidden,heads,kv_heads,head_dim,rotary_dim,capacity;
    float epsilon;
    bool bf16_weights=false; // Round decoded weights before the dot product.
};
struct Weights {
    quantization::Matrix q_gate,key,value,out;
    std::span<const float> input_norm,query_norm,key_norm,frequencies;
};
struct Plan {
    std::size_t queries,kv,history_elements,norm_weights;
    std::size_t norm_offset,qg_offset,k_offset,v_offset,q_offset,gate_offset,prob_offset,core_offset,gated_offset,out_offset,scratch_floats;
    std::size_t host_state_workspace_bytes;
};
Plan plan(Config config);
void validate_weights(Config config,const Weights& weights);
// Batch-one text positions start at zero; no padding, sliding cache or arbitrary
// position ids. All activations are exact BF16 in float storage, persistent KV
// is BF16 bits. Explicit eager dtype boundaries, not fused-backend parity.
// Immutable weights are borrowed. Caller budgets host state/scratch separately.
class Reference {
public:
    Reference(Config config,Weights weights,std::size_t state_workspace_budget);
    ~Reference();
    Reference(const Reference&)=delete;
    Reference& operator=(const Reference&)=delete;
    void step(std::span<const float> input,std::span<float> output);
    void reset();
    bool valid() const;
    std::size_t tokens() const;
    std::span<const std::uint16_t> keys() const;
    std::span<const std::uint16_t> values() const;
    // Last completed step only. Probabilities use [head,capacity] with zero
    // padding beyond the committed prefix; all three spans are BF16 in float.
    std::span<const float> probabilities() const;
    std::span<const float> core() const;
    std::span<const float> gated() const;
private:
    struct Impl;std::unique_ptr<Impl> impl_;
};
} // namespace kadan::full
