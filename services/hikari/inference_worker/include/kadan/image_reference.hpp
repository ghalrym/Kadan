#pragma once
#include <algorithm>
#include <array>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace kadan::image {
struct ImageAttentionSegment {
    std::int64_t start, end;
    bool text;
};

// The vision-language image slots are replaced by clean VAE reference tokens.
// The target follows the entire unchanged text/reference prefix.
struct ImageReferenceLayout {
    std::vector<std::array<std::int64_t, 3>> positions;
    std::vector<ImageAttentionSegment> segments;
    std::int64_t prefix = 0;
};

inline ImageReferenceLayout image_reference_layout(
    std::int64_t text_tokens,
    std::int64_t image_start,
    std::int64_t image_slots,
    std::int64_t reference_height,
    std::int64_t reference_width,
    std::int64_t height,
    std::int64_t width
) {
    if (text_tokens <= 0 || height <= 0 || width <= 0 || image_slots < 0
        || (image_slots
            && (image_start < 0 || image_start + image_slots > text_tokens
                || reference_height <= 0 || reference_width <= 0
                || reference_height * reference_width != image_slots * 4)))
        throw std::invalid_argument("image_reference_layout");
    ImageReferenceLayout result;
    std::int64_t position = 0;
    const auto text = [&](std::int64_t count) {
        if (!count)
            return;
        const auto start = std::int64_t(result.positions.size());
        for (std::int64_t i = 0; i < count; ++i, ++position)
            result.positions.push_back({position, position, position});
        result.segments.push_back({start, std::int64_t(result.positions.size()), true});
    };
    const auto image = [&](std::int64_t h, std::int64_t w, bool reference) {
        const auto start = std::int64_t(result.positions.size());
        for (std::int64_t y = 0; y < h; ++y)
            for (std::int64_t x = 0; x < w; ++x)
                result.positions.push_back(
                    {position, y - (h - h / 2), x - (w - w / 2)}
                );
        position += std::max(h, w);
        if (reference)
            result.segments.push_back(
                {start, std::int64_t(result.positions.size()), false}
            );
    };
    if (image_slots) {
        text(image_start);
        image(reference_height, reference_width, true);
        text(text_tokens - image_start - image_slots);
    } else {
        text(text_tokens);
    }
    result.prefix = result.positions.size();
    image(height, width, false);
    return result;
}
}  // namespace kadan::image
