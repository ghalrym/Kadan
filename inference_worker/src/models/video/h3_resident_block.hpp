#pragma once
#include <array>
#include <atomic>
#include <span>
#include <string>

#include "kadan/h3_resident.hpp"
#include "kadan/media_session.hpp"

namespace kadan::video::detail {
// One INT8 convrot projection with its per-row scales and optional Turbo LoRA.
// Without Turbo, the LoRA names stay empty and rank is zero.
struct H3Projection {
    std::string weight, scale, lora_a, lora_b;
    WeightIdentity weight_id, scale_id, lora_a_id, lora_b_id;
    std::size_t group = 0, rank = 0;
    float lora_scale = 0;
};

// fc1 is stored as gate then value planes; fc2 projects back to hidden.
struct H3FeedForward {
    std::size_t hidden = 5376, width = 14336;
    std::array<H3Projection, 2> fc;
};

// qkv is stored as Q, K and V planes of heads * head_dim rows each. Norm weights
// are BF16 vectors: norm1/norm2 over hidden, q_norm/k_norm over one head.
struct H3Block {
    std::size_t heads = 56, head_dim = 128;
    H3Projection qkv, out;
    std::array<std::string, 4> norm;  // norm1, norm2, q_norm, k_norm
    std::array<WeightIdentity, 4> norm_id;
    H3FeedForward ff;
};

// Upper bound on resident projection rows per tile, matching the planner.
inline constexpr std::size_t h3_tile_rows = 256;

// FF-only schedule: the caller uploads the normalized FF input for every token.
StageManifest
h3_feed_forward_manifest(const H3FeedForward&, std::size_t tokens, std::size_t groups);
MediaShape h3_feed_forward_shape(const H3FeedForward&, std::size_t tokens);

// Full-block schedule: norm1/modulate, K/V for every token, then per query tile
// Q, attention, out_proj and residual, then norm2/modulate and the FF.
StageManifest h3_block_manifest(const H3Block&, std::size_t tokens, std::size_t groups);
MediaShape h3_block_shape(const H3Block&, std::size_t tokens);

// Prepare-phase boundaries on every selected device. Modulation is the block's
// AdaLN output [groups * 6, hidden]; token-groups holds each token's group.
void upload_h3_feed_forward_inputs(
    MediaSession&,
    const H3FeedForward&,
    std::span<const float> normalized,
    std::span<const float> hidden,
    std::span<const float> modulation,
    std::span<const std::uint32_t> token_groups
);
// Rotary is the [token, channel] cos/sin table, [tokens, 96] F64.
void upload_h3_block_inputs(
    MediaSession&,
    const H3Block&,
    std::span<const float> hidden,
    std::span<const float> modulation,
    std::span<const std::uint32_t> token_groups,
    std::span<const double> rotary
);

// Execute the admitted schedule entirely in session buffers, then download the
// updated hidden. Weights must already be uploaded.
void execute_h3_feed_forward(
    MediaSession&,
    std::span<H3ResidentLauncher* const>,
    const H3FeedForward&,
    std::size_t tokens,
    std::span<float> hidden,
    const std::atomic_bool& cancel
);
void execute_h3_block(
    MediaSession&,
    std::span<H3ResidentLauncher* const>,
    const H3Block&,
    std::size_t tokens,
    std::span<float> hidden,
    const std::atomic_bool& cancel
);
}  // namespace kadan::video::detail
