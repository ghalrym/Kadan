#pragma once
#include "kadan/linear_attention.hpp"
#include "kadan/resources.hpp"
namespace kadan::cuda {
// One linear-attention sublayer, including pre-norm/projections/residual, owning
// immutable weights, bounded scratch and one sequence state on one SM86 device.
// Creating-thread/current-device, blocking legacy stream contract. Constructor
// uploads supplied synthetic/native weight views; no checkpoint file loading.
class LinearAttention {
public:
    LinearAttention(linear::Config config,const linear::Weights& weights,int device,std::shared_ptr<Resources> resources);
    ~LinearAttention();
    LinearAttention(const LinearAttention&)=delete;
    LinearAttention& operator=(const LinearAttention&)=delete;
    // Caller pins admitted current-device spans until return. FP32 storage of
    // exact BF16 activations; output is likewise rounded BF16. Exact alias okay.
    // Discard output after any failure. A failed attempted step requires reset;
    // uncertain CUDA failures instead poison the owner until close.
    void step_device(std::span<const float> input,std::span<float> output);
    void reset();
    bool valid() const;
    std::size_t tokens() const;
    void read_state(std::span<std::uint16_t> convolution,std::span<float> recurrent);
    void close();
private:
    struct Impl;std::unique_ptr<Impl> impl_;
};
} // namespace kadan::cuda
