#include "kadan/dense_compute.hpp"
#pragma once
#include <vector>

#include "kadan/image.hpp"

namespace kadan::image {
struct DenoiserConfig {
    BlockConfig block;
    std::size_t layers = 32, context = 4096, channels = 64;
};

// Complete text-to-image denoising transformer forward. Conditioning is supplied
// by the caller; scheduler, text encoder and image VAE are separate executors.
class Denoiser {
public:
    using Hook = std::function<void(const char*, std::size_t)>;
    explicit Denoiser(
        std::shared_ptr<Resources>, std::shared_ptr<DenseCompute> compute = {}
    );
    ~Denoiser();
    Denoiser(const Denoiser&) = delete;
    Denoiser& operator=(const Denoiser&) = delete;
    void load(
        const char* root,
        std::span<const std::string> shards,
        DenoiserConfig,
        const std::atomic_bool&,
        const Hook& hook = {}
    );
    void unload();
    // Explicit resident block slice, using actual shared modulation. Initialize
    // per-device launchers before admission. Native BF16 standard precision only.
    std::optional<MediaPlanError> prepare_feed_forward(
        MediaSession&,
        std::size_t block,
        std::size_t tokens,
        HiddenBoundary,
        const std::atomic_bool&
    );
    // Caller supplies certified conditioning/time-activated [2,state]: actual
    // timestep then text t=0, after the second timestep projection's SiLU.
    // Also supply hidden and (pre_attention) attended on every device, release
    // views, then project modulation directly into the residual/FF consumers.
    // Returns complete; close launchers before destroying session streams.
    void execute_feed_forward(
        MediaSession&,
        std::size_t block,
        std::size_t tokens,
        std::size_t text,
        HiddenBoundary,
        std::span<video::ImageFeedForwardLauncher* const>,
        const std::atomic_bool&
    );
    // F32 latent [height*width,channels], condition [text,context], velocity of
    // latent shape. All caller buffers are caller-admitted; timestep in [0,1].
    void execute(
        std::span<const float> latent,
        std::span<const float> condition,
        std::size_t text,
        std::size_t height,
        std::size_t width,
        float timestep,
        std::span<float> velocity,
        const std::atomic_bool&,
        const Hook& hook = {}
    );

private:
    struct Impl;
    std::shared_ptr<Resources> resources_;
    std::shared_ptr<DenseCompute> compute_;
    std::unique_ptr<Impl> model_;
    bool busy_ = false;
};
}  // namespace kadan::image
