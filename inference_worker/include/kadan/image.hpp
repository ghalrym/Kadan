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

// Next consumed boundary of the certified hidden.
enum class HiddenBoundary {
    pre_qkv,
    pre_attention,
    post_attention
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
    // Prepare only: actual bias-free BF16 checkpoint slices and session resources.
    // pre_qkv adds replicated Q/K/V; pre_attention admits attn.to_out.0
    // with resident/normalized already attended.
    // supports the native BF16 branch of the existing standard projection only.
    // Initialize/account device launchers before this fresh admission probe.
    std::optional<MediaPlanError> prepare_feed_forward(
        MediaSession&, std::size_t tokens, HiddenBoundary, const std::atomic_bool&
    );
    // Explicit device normalization/modulation and [target,text] residual gates.
    // All hidden/modulation inputs must be certified on every device. At the
    // pre_attention boundary, resident/normalized holds full certified attended.
    // Project it with the admitted actual attn.to_out.0 weight, apply its residual,
    // then reuse the row buffer and gates for FF preparation. No QKV/attention
    // execution occurs here. Caller releases ingress/producer views first.
    // Leaves the session in prepare; invoke execute_feed_forward next.
    void prepare_feed_forward_inputs(
        MediaSession&,
        std::size_t tokens,
        std::size_t text,
        HiddenBoundary,
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
    friend class Denoiser;
    // Compose denoiser conditioning with the block's consumed admission.
    std::optional<MediaPlanError> prepare_feed_forward(
        MediaSession&,
        std::size_t,
        HiddenBoundary,
        const std::atomic_bool&,
        StageManifest
    );
    // Denoiser validates its shared weights; the block validates its own slices.
    void execute_feed_forward(
        MediaSession&,
        std::size_t,
        std::size_t,
        std::span<video::ImageFeedForwardLauncher* const>,
        const std::atomic_bool&,
        std::size_t shared_weights
    );
    void execute_qkv(
        MediaSession&,
        std::size_t tokens,
        std::size_t text,
        std::span<video::ImageFeedForwardLauncher* const>,
        const std::atomic_bool&
    );
    void execute_rotary(
        MediaSession&,
        std::size_t tokens,
        std::size_t text,
        std::size_t height,
        std::size_t width,
        std::span<video::ImageFeedForwardLauncher* const>,
        const std::atomic_bool&
    );
    void execute_masked_attention(
        MediaSession&,
        std::size_t tokens,
        std::size_t text,
        std::span<video::ImageFeedForwardLauncher* const>,
        const std::atomic_bool&
    );
    void execute_attention_output(
        MediaSession&,
        std::size_t tokens,
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
