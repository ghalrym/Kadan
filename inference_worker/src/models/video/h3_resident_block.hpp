#pragma once
#include <array>
#include <atomic>
#include <span>
#include <string>

#include "kadan/h3_resident.hpp"
#include "kadan/media_session.hpp"

namespace kadan::video::detail {
// One INT8 convrot projection with its per-row scales and either a BF16 bias or
// optional Turbo LoRA. Absent parts keep empty names and a zero rank.
struct H3Projection {
    std::string weight, scale, lora_a, lora_b, bias;
    WeightIdentity weight_id, scale_id, lora_a_id, lora_b_id, bias_id;
    std::size_t group = 0, rank = 0;
    float lora_scale = 0;
};

// fc1 is stored as gate then value planes; fc2 projects back to hidden.
struct H3FeedForward {
    std::size_t hidden = 5376, width = 14336;
    std::array<H3Projection, 2> fc;
};

// qkv is stored as Q, K and V planes of heads * head_dim rows each. Norm weights
// are BF16 vectors: norm1/norm2 over hidden, q_norm/k_norm over one head. AdaLN
// projects the [timesteps, time_width] embedding to [timesteps, 18 * hidden],
// which is exactly the [groups * 6, hidden] modulation table.
struct H3Block {
    std::size_t heads = 56, head_dim = 128, time_width = 2688;
    H3Projection qkv, out, adaln;
    std::array<std::string, 4> norm;  // norm1, norm2, q_norm, k_norm
    std::array<WeightIdentity, 4> norm_id;
    H3FeedForward ff;
};

// Upper bound on resident projection rows per tile, matching the planner.
inline constexpr std::size_t h3_tile_rows = 256;

// Detach all launchers and end the session, retaining cleanup failures.
void close_h3_resident(
    MediaSession&, std::span<H3ResidentLauncher* const>, bool abort = false
);

// FF-only schedule: the caller uploads the normalized FF input for every token.
StageManifest
h3_feed_forward_manifest(const H3FeedForward&, std::size_t tokens, std::size_t groups);
MediaShape h3_feed_forward_shape(const H3FeedForward&, std::size_t tokens);

// Full-block schedule: norm1/modulate, K/V for every token, then per query tile
// Q, attention, out_proj and residual, then norm2/modulate and the FF. With
// `retained`, hidden, token groups, rotary and the time embedding are the previous
// block's session buffers; only this block's weights are uploaded.
StageManifest h3_block_manifest(
    const H3Block&, std::size_t tokens, std::size_t groups, bool retained
);
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
// The first block of a step uploads hidden, the [timesteps, time_width] embedding,
// token groups and rotary, the [token, channel] cos/sin table [tokens, 96] F64.
void upload_h3_block_inputs(
    MediaSession&,
    const H3Block&,
    std::span<const float> hidden,
    std::span<const float> time,
    std::span<const std::uint32_t> token_groups,
    std::span<const double> rotary
);

// Download the resident hidden after the last block of a step.
void download_h3_hidden(
    MediaSession&, std::size_t hidden_width, std::span<float> hidden
);

// Execute the admitted schedule entirely in session buffers. The FF-only schedule
// downloads the updated hidden; the full block leaves it resident for the next
// block. Weights must already be uploaded.
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
    const std::atomic_bool& cancel
);
}  // namespace kadan::video::detail
