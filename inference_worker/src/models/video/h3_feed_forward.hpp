#pragma once
#include <array>
#include <atomic>
#include <span>
#include <string>

#include "kadan/h3_resident.hpp"
#include "kadan/media_session.hpp"

namespace kadan::video::detail {
// One denoiser block's resident feed-forward contract. Index 0 is fc1, stored as
// gate then value planes; index 1 is fc2. Without Turbo, LoRA names stay empty.
struct H3FeedForward {
    std::size_t hidden = 5376, width = 14336;
    std::array<std::string, 2> weight, scale, lora_a, lora_b;
    std::array<WeightIdentity, 2> weight_id, scale_id, lora_a_id, lora_b_id;
    std::array<std::size_t, 2> group{}, rank{};
    std::array<float, 2> lora_scale{};
};

// Detach every launcher before ending the session, even if a release fails.
void close_h3_feed_forward(
    MediaSession&, std::span<H3ResidentLauncher* const>, bool abort = false
);

// INT8 convrot weights with per-row scales, channel-partitioned in units of four;
// LoRA A is replicated and LoRA B follows its projection's partition.
StageManifest
h3_feed_forward_manifest(const H3FeedForward&, std::size_t tokens, std::size_t groups);

// Prepare-phase boundaries on every selected device: normalized FF input, hidden,
// per-group residual gates [groups,hidden] and each token's gate group.
void upload_h3_feed_forward_inputs(
    MediaSession&,
    const H3FeedForward&,
    std::span<const float> normalized,
    std::span<const float> hidden,
    std::span<const float> gates,
    std::span<const std::uint32_t> token_groups
);

// fc1 -> SiLU gate -> fc2 -> gated residual entirely in session buffers, then
// downloads the updated hidden. Weights must already be uploaded.
void execute_h3_feed_forward(
    MediaSession&,
    std::span<H3ResidentLauncher* const>,
    const H3FeedForward&,
    std::size_t tokens,
    std::size_t groups,
    std::span<float> hidden,
    const std::atomic_bool& cancel
);
}  // namespace kadan::video::detail
