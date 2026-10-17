#pragma once
#include <algorithm>
#include <cmath>
#include <span>
#include <stdexcept>

#include "kadan/image_schedule.hpp"

namespace kadan::image {
inline std::size_t edit_steps(std::size_t steps, float strength) {
    if (!steps || steps > 100 || !std::isfinite(strength) || strength < 0
        || strength > 1)
        throw std::invalid_argument("image_edit_strength");
    return strength == 0 ? 0 : std::max<std::size_t>(1, steps * strength);
}

// Keep the checkpoint's actual sigma grid, including Turbo's saved schedule.
inline Schedule edit_schedule(const Schedule& full, float strength) {
    Schedule result;
    result.steps = edit_steps(full.steps, strength);
    const auto start = full.steps - result.steps;
    std::copy_n(full.sigma.begin() + start, result.steps + 1, result.sigma.begin());
    return result;
}

inline void composite_image_edit(
    std::span<float> output, std::span<const float> source, std::span<const float> mask
) {
    if (output.size() != source.size() || output.size() != mask.size() * 4)
        throw std::invalid_argument("image_edit_composite_shape");
    for (std::size_t i = 0; i < output.size(); ++i) {
        const auto amount = mask[i % mask.size()];
        // Preserve original bytes on publication, even at hard mask boundaries.
        output[i] =
            amount == 0 ? source[i] : amount * output[i] + (1.f - amount) * source[i];
    }
}
}  // namespace kadan::image
