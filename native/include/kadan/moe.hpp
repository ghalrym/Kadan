#pragma once
#include "kadan/quantization.hpp"
#include <memory>
namespace kadan::moe {
struct Config { std::size_t hidden,experts,top_k,intermediate,shared_intermediate; };
struct Expert { quantization::Matrix gate,up,down; };
struct Weights { std::span<const float> router,shared_gate;std::span<const Expert> experts;Expert shared; };
struct ExpertLayout { std::size_t gate_weights,gate_scales,up_weights,up_scales,down_weights,down_scales,bytes; };
struct Plan {
    std::size_t logits,probabilities,top_weights,gate,up,activation,down,accumulator,shared,result,shared_factor,scratch_floats,host_workspace_bytes;
    std::size_t router_offset,shared_gate_offset,experts_offset,shared_offset,scratch_offset,indices_offset,status_offset,device_bytes;
    ExpertLayout routed_layout,shared_layout;
};
Plan plan(Config config);
void validate_weights(Config config,const Weights& weights);
// Normalized BF16 input in float storage -> BF16 MoE branch output. No residual,
// attention state, checkpoint loading or decoder transaction is owned here.
// Experts and source weights remain immutable/alive throughout CPU reference use.
class Reference {
public:
    Reference(Config config,Weights weights,std::size_t workspace_budget);
    ~Reference();
    Reference(const Reference&)=delete;Reference& operator=(const Reference&)=delete;
    void forward(std::span<const float> input,std::span<float> output);
    void reset();
    bool valid() const;
    std::span<const unsigned> selected() const; // score order, lower id breaks exact ties
    std::span<const float> logits() const;
    std::span<const float> probabilities() const; // FP32 before top-k normalization
    std::span<const float> top_weights() const; // BF16 normalized selected weights
    std::span<const float> routed() const;
    std::span<const float> shared() const; // after shared sigmoid gating
    std::span<const float> result() const;
private:
    struct Impl;std::unique_ptr<Impl> impl_;
};
} // namespace kadan::moe
