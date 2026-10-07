#pragma once
#include "kadan/quantization.hpp"
#include "kadan/sequence_state.hpp"
#include <cstdint>
#include <memory>
#include <span>
namespace kadan::linear {
struct Config {
    std::size_t hidden,key_heads,value_heads,key_dim,value_dim,conv_kernel,capacity;
    float epsilon;
};
// Auxiliary weights are exact finite BF16 values represented in float storage.
// Scalar-scaled FP8 projections match this checkpoint's linear sublayer roles.
struct Weights {
    quantization::Matrix qkv,z,out;
    std::span<const float> input_norm,conv,a,b,a_log,dt_bias,output_norm;
};
struct Plan {
    std::size_t keys,values,channels,conv_elements,recurrent_elements;
    std::size_t norm_offset,qkv_offset,z_offset,a_offset,b_offset,core_offset,gate_offset,out_offset,scratch_floats;
    std::size_t aux_elements,host_state_workspace_bytes;
};
Plan plan(Config config);
void validate_weights(Config config,const Weights& weights);
float bf16_round(float value); // RNE; reject nonfinite input or overflow to infinity.
// Original CPU sequence oracle with admitted bounded state/workspace. Weights
// are borrowed and must remain alive/immutable. Input/output hold exact BF16
// values in float storage. Any attempted-step failure invalidates until reset.
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
    std::span<const std::uint16_t> convolution() const;
    std::span<const float> recurrent() const;
private:
    struct Impl;std::unique_ptr<Impl> impl_;
};
} // namespace kadan::linear
