#pragma once
#include <algorithm>
#include <cstdint>
#include <stdexcept>

namespace kadan::image {
// Coordinates are in the latent grid (one cell becomes 16 output pixels).
inline constexpr std::int64_t vae_tile_core = 16;
// Decoder: six 3x3 convolutions per up block, four nearest-neighbor
// upsample/convolution pairs, then the final 3x3 convolution need 13 cells.
inline constexpr std::int64_t vae_decode_halo = 13;
// Encoder: conv_in, four stride-two blocks and the final residual block
// need 125 pixels before and 140 after. Nine cells round both to stride 16.
inline constexpr std::int64_t vae_encode_halo = 9;

struct VaeTileAxis {
    std::int64_t begin, end;
    std::int64_t input_begin, input_end;

    std::int64_t crop_begin() const {
        return begin - input_begin;
    }
};

inline VaeTileAxis
vae_tile_axis(std::int64_t begin, std::int64_t extent, std::int64_t halo) {
    if (begin < 0 || begin >= extent || halo < 0)
        throw std::invalid_argument("image_vae_tile_axis");
    const auto end = std::min(begin + vae_tile_core, extent);
    return {
        begin,
        end,
        std::max<std::int64_t>(0, begin - halo),
        std::min(extent, end + halo)
    };
}
}  // namespace kadan::image
