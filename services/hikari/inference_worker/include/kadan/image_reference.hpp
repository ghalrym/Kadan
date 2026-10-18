#pragma once
#include <algorithm>
#include <array>
#include <cstdint>
#include <stdexcept>
#include <utility>
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

// Padding separates the unchanged causal/reference prefix from the target.
// Rotary coordinates remain those of each unpadded request, so a longer sibling
// prompt cannot move another image in positional space.
struct ImagePaddedReferenceLayout {
    std::vector<std::array<std::int64_t, 3>> positions;
    std::vector<ImageAttentionSegment> segments;
    std::vector<std::uint8_t> valid_keys;
    std::int64_t valid_prefix;
};

struct ImageBatchReferenceLayout {
    std::vector<ImagePaddedReferenceLayout> items;
    std::int64_t prefix = 0;
    bool padded = false;
};

inline ImageBatchReferenceLayout
image_batch_reference_layout(std::vector<ImageReferenceLayout> layouts) {
    if (layouts.empty())
        throw std::invalid_argument("image_batch_empty");
    ImageBatchReferenceLayout batch;
    const auto target =
        std::int64_t(layouts.front().positions.size()) - layouts.front().prefix;
    for (const auto& item : layouts) {
        if (item.prefix <= 0 || target <= 0
            || std::int64_t(item.positions.size()) - item.prefix != target)
            throw std::invalid_argument("image_batch_layout");
        batch.prefix = std::max(batch.prefix, item.prefix);
    }
    for (auto& item : layouts) {
        const auto padding = batch.prefix - item.prefix;
        batch.padded = batch.padded || padding != 0;
        std::vector<std::uint8_t> valid_keys(item.positions.size(), 1);
        valid_keys.insert(valid_keys.begin() + item.prefix, padding, 0);
        item.positions.insert(item.positions.begin() + item.prefix, padding, {0, 0, 0});
        batch.items.push_back(
            {std::move(item.positions),
             std::move(item.segments),
             std::move(valid_keys),
             item.prefix}
        );
    }
    return batch;
}
}  // namespace kadan::image
