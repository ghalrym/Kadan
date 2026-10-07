#pragma once
#include "kadan/full_attention.hpp"
#include "kadan/cuda_error.hpp"
#include "kadan/resources.hpp"
namespace kadan::cuda {
// Original one-token text attention owner; construction uploads supplied weights.
// Creating thread/current SM86 device only. Explicit blocking legacy stream.
class FullAttention {
public:
    FullAttention(full::Config config,const full::Weights& weights,int device,std::shared_ptr<Resources> resources);
    ~FullAttention();
    FullAttention(const FullAttention&)=delete;
    FullAttention& operator=(const FullAttention&)=delete;
    // Caller-admitted pinned BF16-in-float device buffers. Exact alias allowed.
    // Discard output on error. Numerical/input failures invalidate until reset;
    // runtime poison requires close. DeviceBufferQuarantine requires retaining
    // borrowed buffers through successful close or context teardown.
    void step_device(std::span<const float> input,std::span<float> output);
    void reset();
    bool valid() const;
    std::size_t tokens() const;
    // Caller-admitted host buffers; full allocated histories, including zero tail.
    void read_state(std::span<std::uint16_t> keys,std::span<std::uint16_t> values);
    void read_intermediates(std::span<float> probabilities,std::span<float> core,std::span<float> gated);
    void close(); // Uncertain cleanup retains admission; no automatic retry.
private:
    struct Impl;std::unique_ptr<Impl> impl_;
};
} // namespace kadan::cuda
