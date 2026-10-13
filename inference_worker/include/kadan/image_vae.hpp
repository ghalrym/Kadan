#pragma once
#include <atomic>
#include <functional>
#include <memory>
#include <span>
#include <string>

#include "kadan/dense_compute.hpp"
#include "kadan/media_session.hpp"

namespace kadan::image {
struct VaeConfig {
    std::size_t base = 144, latent = 64, channels = 4, residuals = 2;
};

// Qwen Image latent [points,64] -> planar [64,points], with separate F32
// multiply by stddev and add mean. Buffers are caller-admitted and disjoint.
void prepare_decoder_input(
    std::span<const float> latent,
    std::span<const float> mean,
    std::span<const float> stddev,
    std::span<float> planar,
    const std::atomic_bool&
);
// After the terminal scheduler update, download the certified ingress/latent
// once from a current owner into caller-admitted scratch, then prepare the same
// VaeDecoder input. Terminal sigma/order remain the caller's contract. Release
// caller views first; close launchers and finish the session before decoding.
// A failed handoff aborts the session. No decoder or device kernel runs here.
void prepare_decoder_input(
    MediaSession&,
    std::span<float> latent_scratch,
    std::span<const float> mean,
    std::span<const float> stddev,
    std::span<float> planar,
    const std::atomic_bool&
);

// Single-frame Qwen Image 2.1 decoder. Input/output are planar CHW. The caller
// supplies denormalized latents and owns admitted output memory. No video cache.
class VaeDecoder {
public:
    using Hook = std::function<void(std::size_t)>;
    explicit VaeDecoder(
        std::shared_ptr<Resources>, std::shared_ptr<DenseCompute> compute = {}
    );
    ~VaeDecoder();
    void load(
        const char* root,
        const std::string& shard,
        VaeConfig,
        const std::atomic_bool&,
        const Hook& hook = {}
    );
    void decode(
        std::span<const float>,
        std::size_t height,
        std::size_t width,
        std::span<float>,
        const std::atomic_bool&,
        const Hook& = {}
    );
    void unload();

private:
    struct Impl;
    std::shared_ptr<Resources> resources_;
    std::shared_ptr<DenseCompute> compute_;
    std::unique_ptr<Impl> model_;
    bool busy_ = false;
};
}  // namespace kadan::image
