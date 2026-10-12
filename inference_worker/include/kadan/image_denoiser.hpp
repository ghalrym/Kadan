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
    // Explicit resident block slice, using actual timestep conditioning and modulation.
    // Initialize per-device launchers before admission. Native BF16 standard precision
    // only.
    std::optional<MediaPlanError> prepare_feed_forward(
        MediaSession&,
        std::size_t block,
        std::size_t tokens,
        HiddenBoundary,
        const std::atomic_bool&
    );
    // Timestep in [0,1] constructs target/text-zero frequency rows, followed by
    // linear_1 + SiLU and linear_2 + SiLU, consumed directly by modulation.
    // Caller supplies certified hidden and (pre_attention) attended on every device.
    // nullopt consumes certified time2/modulation retained by this session across
    // drained block admissions of the same loaded model and timestep. Supply a
    // timestep to overwrite both for the next step.
    // Activated time2 remains available through complete for final normalization.
    // Release views before execution; modulation feeds the residual/FF consumers.
    // The last block consumes target time2 in norm_out.linear, then certifies
    // final/normalized [tokens-text,state] from the final image hidden rows.
    // Returns complete; close launchers before destroying session streams.
    void execute_feed_forward(
        MediaSession&,
        std::size_t block,
        std::size_t tokens,
        std::size_t text,
        std::optional<float> timestep,
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
    void execute_final_normalization(
        MediaSession&,
        std::size_t tokens,
        std::size_t text,
        std::span<video::ImageFeedForwardLauncher* const>,
        const std::atomic_bool&
    );
    struct Impl;
    std::shared_ptr<Resources> resources_;
    std::shared_ptr<DenseCompute> compute_;
    std::unique_ptr<Impl> model_;
    bool busy_ = false;
};
}  // namespace kadan::image
