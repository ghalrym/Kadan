#pragma once
#include <array>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace kadan::image {
// FLUX.2 4D rotary position coordinates: (T, H, W, L).
// Text tokens occupy (0, 0, 0, 0..L-1).
// Target latent cells occupy (0, 0..H-1, 0..W-1, 0).
// Ordered reference image k occupies (scale * (k + 1), 0..H_k-1, 0..W_k-1, 0).
struct ImageGrid {
    std::int64_t height = 0, width = 0;
};

struct Flux2PositionLayout {
    std::vector<std::array<std::int64_t, 4>> positions;
    std::int64_t text_tokens = 0;
    std::int64_t target_tokens = 0;
    std::int64_t reference_tokens = 0;
};

inline Flux2PositionLayout flux2_position_layout(
    std::int64_t text_tokens,
    std::int64_t target_height,
    std::int64_t target_width,
    std::span<const ImageGrid> references = {},
    std::int64_t time_scale = 10
) {
    if (text_tokens <= 0 || target_height <= 0 || target_width <= 0 || time_scale <= 0
        || references.size() > 4)
        throw std::invalid_argument("image_reference_layout");
    std::int64_t ref_tokens = 0;
    for (const auto& ref : references) {
        if (ref.height <= 0 || ref.width <= 0)
            throw std::invalid_argument("image_reference_layout");
        ref_tokens += ref.height * ref.width;
    }
    Flux2PositionLayout result;
    result.text_tokens = text_tokens;
    result.target_tokens = target_height * target_width;
    result.reference_tokens = ref_tokens;
    result.positions.reserve(
        std::size_t(text_tokens + result.target_tokens + ref_tokens)
    );
    for (std::int64_t i = 0; i < text_tokens; ++i)
        result.positions.push_back({0, 0, 0, i});
    for (std::int64_t y = 0; y < target_height; ++y)
        for (std::int64_t x = 0; x < target_width; ++x)
            result.positions.push_back({0, y, x, 0});
    for (std::size_t k = 0; k < references.size(); ++k) {
        const auto t = time_scale * std::int64_t(k + 1);
        for (std::int64_t y = 0; y < references[k].height; ++y)
            for (std::int64_t x = 0; x < references[k].width; ++x)
                result.positions.push_back({t, y, x, 0});
    }
    return result;
}
}  // namespace kadan::image
