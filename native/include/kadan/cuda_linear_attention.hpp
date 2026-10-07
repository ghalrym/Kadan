#pragma once
#include "kadan/linear_attention.hpp"
#include "kadan/cuda_error.hpp"
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
    // uncertain CUDA failures instead poison the owner until close. On
    // DeviceBufferQuarantine retain borrowed spans until successful close or
    // context teardown, even though step_device has thrown.
    void step_device(std::span<const float> input,std::span<float> output);
    void reset();
    bool valid() const;
    std::size_t tokens() const;
    void read_state(std::span<std::uint16_t> convolution,std::span<float> recurrent);
    // Diagnostic copies of the last committed core/gated values; no kernel launches.
    void read_intermediates(std::span<float> core,std::span<float> gated);
    void close();
private:
    struct Impl;std::unique_ptr<Impl> impl_;
};
} // namespace kadan::cuda
