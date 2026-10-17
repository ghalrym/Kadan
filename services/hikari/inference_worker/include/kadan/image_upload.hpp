#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>

namespace kadan::serving {
inline constexpr std::size_t image_upload_maximum = 20 * 1024 * 1024;
inline constexpr std::size_t image_upload_encoded_maximum =
    4 * ((image_upload_maximum + 2) / 3) + 22;

struct ImageUpload {
    std::string png;
    unsigned width = 0, height = 0;
};

// Inspect PNG metadata before acquiring a model; pixel storage is admitted later.
ImageUpload prepare_image_upload(std::string_view data_url);

// The caller admits and owns both buffers. Source pixels are planar RGBA [-1, 1].
void decode_image_upload(
    const ImageUpload&, std::span<std::uint8_t> rgba, std::span<float> planar
);
// Mask values are luminance [0, 1], where white selects pixels to edit.
void decode_image_mask(
    const ImageUpload&, std::span<std::uint8_t> rgba, std::span<float> mask
);
}  // namespace kadan::serving
