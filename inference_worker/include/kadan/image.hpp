#pragma once
#include <array>
#include <atomic>
#include <functional>
#include <memory>
#include <span>
#include <string>

#include "kadan/dense_compute.hpp"
#include "kadan/media_session.hpp"

namespace kadan::video {
class ImageFeedForwardLauncher;
}

namespace kadan::image {
struct BlockConfig {
    std::size_t state = 4096, heads = 32, head_dim = 128, intermediate = 12288;
    std::array<std::size_t, 3> axes{16, 56, 56};
};

// Complete Qwen-Image-2.1 transformer block, text-to-image layout only.
// Text rows are causal; the one target-image block attends to the whole sequence.
// This component neither tokenizes prompts nor generates a finished image.
class TransformerBlock {
public:
    using Hook = std::function<void(const char*)>;
    explicit TransformerBlock(
        std::shared_ptr<Resources>, std::shared_ptr<DenseCompute> compute = {}
    );
    ~TransformerBlock();
    TransformerBlock(const TransformerBlock&) = delete;
    TransformerBlock& operator=(const TransformerBlock&) = delete;
    void load(
        const char* root,
        const std::string& shard,
        std::size_t block,
        BlockConfig,
        const std::atomic_bool&
    );
    void load(
        const char* root,
        std::span<const std::string> shards,
        std::size_t block,
        BlockConfig,
        const std::atomic_bool&
    );
    void unload();
    // Prepare only: actual bias-free BF16 gate/up/down checkpoint slices and
    // session-owned FF resources. Caller prepares complete post-attention hidden
    // and actual modulation [2,4*state] on every device, then releases its views.
    // Initialize/account device launchers before this fresh admission probe.
    std::optional<MediaPlanError>
    prepare_feed_forward(MediaSession&, std::size_t tokens, const std::atomic_bool&);
    // Explicit device normalization/modulation and [target,text] residual gates.
    // All devices' hidden/modulation inputs must be certified before submission.
    // Leaves the session in prepare; invoke execute_feed_forward next.
    void prepare_feed_forward_inputs(
        MediaSession&,
        std::size_t tokens,
        std::size_t text,
        std::span<video::ImageFeedForwardLauncher* const>,
        const std::atomic_bool&
    );
    // Explicit resident FF only. Caller owns initialized per-device launchers and
    // complete normalized/hidden/[target,text] gates prepared by the model.
    // Returns in complete phase; drain and close launchers before session destruction.
    void execute_feed_forward(
        MediaSession&,
        std::size_t tokens,
        std::size_t text,
        std::span<video::ImageFeedForwardLauncher* const>,
        const std::atomic_bool&
    );
    // Caller admits input/output [text+height*width,state] and modulation
    // [2,4*state]: target timestep first, text t=0 second. No padded text rows.
    void execute(
        std::span<const float> input,
        std::span<const float> modulation,
        std::size_t text,
        std::size_t height,
        std::size_t width,
        std::span<float> output,
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
