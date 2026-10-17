#pragma once
#include <atomic>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>

#include "kadan/dense_compute.hpp"
#include "kadan/media_session.hpp"

namespace kadan::image {
inline void validate_vae_normalization(
    std::span<const float> input,
    std::span<const float> gamma,
    std::size_t points,
    std::span<float> output
) {
    const auto maximum = std::numeric_limits<std::size_t>::max();
    if (!points || gamma.empty() || gamma.size() > maximum / points
        || gamma.size() * points > maximum / sizeof(float)
        || input.size() != gamma.size() * points || output.size() != input.size()
        || !input.data() || !gamma.data() || !output.data())
        throw std::invalid_argument("image_vae_norm_shape");
    for (auto source : {input, gamma}) {
        const auto x = reinterpret_cast<std::uintptr_t>(source.data()),
                   y = reinterpret_cast<std::uintptr_t>(output.data());
        if (x <= y ? y - x < source.size_bytes() : x - y < output.size_bytes())
            throw std::invalid_argument("image_vae_norm_overlap");
    }
}

// Caller-admitted, disjoint CHW buffers. A supported device operation completes
// and validates all partitions before return; unsupported keeps the CPU formula.
void normalize_vae_features(
    std::span<const float>,
    std::span<const float>,
    std::size_t points,
    std::span<float>,
    bool activate,
    const std::atomic_bool&,
    DenseCompute* = nullptr
);

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
// planar decoder input. Terminal sigma/order remain the caller's contract. Release
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

}  // namespace kadan::image
