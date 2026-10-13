#include "kadan/dense_compute.hpp"
#pragma once
#include <vector>

#include "kadan/image.hpp"
#include "kadan/image_timing.hpp"

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
    // Metadata-only check of every consumed block, including final output and
    // scheduler ownership, against current capacity. Does not reserve future fit.
    std::optional<MediaPlanError> preflight(
        MediaSession&,
        std::size_t text,
        std::size_t height,
        std::size_t width,
        const std::atomic_bool&
    );
    // Explicit resident block slice, using actual timestep conditioning and modulation.
    // Initialize per-device launchers before admission. Native BF16 standard precision
    // only.
    // After a completed block, admit the next loaded block as pre_qkv. Its full
    // hidden and timestep conditioning must be certified and retained on every
    // selected device, migrating exact bytes to new owners over admitted direct peers.
    // Ordered blocks of the same loaded model/step are the caller contract. Reuse
    // conditioning with execute_qkv(nullopt), then rotary and attention as usual.
    // resident_text opts into full-step latent/condition ownership. Supply the same
    // text count on every block admission; both inputs survive owner migrations.
    std::optional<MediaPlanError> prepare_feed_forward(
        MediaSession&,
        std::size_t block,
        std::size_t tokens,
        HiddenBoundary,
        const std::atomic_bool&,
        std::optional<std::size_t> resident_text = {}
    );
    // Block 0 pre_qkv/pre_attention admission includes actual ingress
    // weights/workspaces. Validated host latent [images,channels] and condition
    // [text,context] are uploaded once per selected device. Text RMS -> in_layer + GELU
    // -> out_layer and img_in certify the text-first resident/hidden consumed by the
    // block. With resident_text admission, empty spans reuse certified inputs after
    // the scheduler update; initial spans preserve the caller's seeded F32 bytes.
    // Release all caller views before execution. No attention is executed here.
    void execute_ingress(
        MediaSession&,
        std::span<const float> latent,
        std::span<const float> condition,
        std::size_t text,
        std::span<video::ImageFeedForwardLauncher* const>,
        const std::atomic_bool&
    );
    // pre_qkv admission: actual conditioning -> stage 0 norm/modulation ->
    // loaded Q/K/V. Leaves certified attention/{q,k,v} in prepare for the next
    // attention consumer. Release views first; no attention or FF executes here.
    void execute_qkv(
        MediaSession&,
        std::size_t block,
        std::size_t tokens,
        std::size_t text,
        std::optional<float> timestep,
        std::span<video::ImageFeedForwardLauncher* const>,
        const std::atomic_bool&
    );
    // Consume certified Q/K from execute_qkv with the loaded per-head learned
    // RMS weights, then rotate text/image positions in place. Leaves certified
    // Q/K in prepare for attention; V is retained. Release caller views first.
    void execute_rotary(
        MediaSession&,
        std::size_t block,
        std::size_t tokens,
        std::size_t text,
        std::size_t height,
        std::size_t width,
        std::span<video::ImageFeedForwardLauncher* const>,
        const std::atomic_bool&
    );
    // Consume certified rotated Q/K and original V, then feed attended rows
    // directly through the existing output/residual/FF and final velocity slice.
    // Uses the same pre_qkv admission and retained conditioning; returns complete.
    void execute_attention(
        MediaSession&,
        std::size_t block,
        std::size_t tokens,
        std::size_t text,
        std::span<video::ImageFeedForwardLauncher* const>,
        const std::atomic_bool&,
        DenoisingTimings* = nullptr
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
    // Actual proj_out consumes those rows and certifies final/velocity
    // [tokens-text,channels], retained at complete for the caller.
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
    // Complete last-block velocity updates the owned latent in place using the
    // existing F32 next-current delta and separate multiply/add. No host transfer.
    // Caller executes this once per completed step, using that step's sigma as
    // current and the following schedule sigma as next. Release caller views first.
    // Admit block 0 with resident_text, reuse ingress, then execute_qkv with next;
    // next == 0 leaves the final latent certified for the later decoder handoff.
    void execute_scheduler(
        MediaSession&,
        std::size_t tokens,
        std::size_t text,
        float current,
        float next,
        std::span<video::ImageFeedForwardLauncher* const>,
        const std::atomic_bool&
    );
    // Complete resident transformer step. Caller admits block 0 as pre_qkv with
    // resident_text first; later blocks are admitted here. Supply all initialized
    // device launchers; each block uses only its automatically selected owners.
    // Empty input spans reuse the prior scheduler's certified latent/conditioning.
    // Returns complete with final velocity ready for the separate scheduler.
    void execute(
        MediaSession&,
        std::span<const float> latent,
        std::span<const float> condition,
        std::size_t text,
        std::size_t height,
        std::size_t width,
        float timestep,
        std::span<video::ImageFeedForwardLauncher* const>,
        const std::atomic_bool&,
        const Hook& = {},
        DenoisingTimings* = nullptr
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
    std::variant<StageManifest, MediaPlanError> feed_forward_manifest(
        std::size_t,
        std::size_t,
        HiddenBoundary,
        const std::atomic_bool&,
        std::optional<std::size_t>,
        bool retain_latent
    );
    void execute_conditioning(
        MediaSession&,
        std::size_t block,
        std::size_t tokens,
        std::size_t text,
        std::optional<float> timestep,
        HiddenBoundary,
        std::span<video::ImageFeedForwardLauncher* const>,
        const std::atomic_bool&
    );
    void execute_final_output(
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
