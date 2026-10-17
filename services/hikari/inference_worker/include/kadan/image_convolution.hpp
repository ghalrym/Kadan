#pragma once
#include <algorithm>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>

namespace kadan::image {
// Stride-one CHW convolution with centered zero padding. Projection columns
// retain checkpoint order [input channel, kernel y, kernel x].
struct VaeConvolution {
    std::size_t channels, height, width, outputs, kernel;

#ifdef __CUDACC__
    __host__ __device__
#endif
        std::size_t points() const {
        return height * width;
    }
#ifdef __CUDACC__
    __host__ __device__
#endif
        std::size_t columns() const {
        return channels * kernel * kernel;
    }

    void validate() const {
        const auto maximum = std::numeric_limits<std::size_t>::max() / sizeof(float);
        if (!channels || !height || !width || !outputs || !kernel
            || height > maximum / width || kernel > maximum / kernel
            || channels > maximum / (kernel * kernel) || channels > maximum / points()
            || outputs > maximum / points())
            throw std::invalid_argument("image_vae_conv_shape");
    }
};

struct VaeConvolutionTile {
    VaeConvolution shape;
    std::size_t first, count;
    std::size_t capacity = 256;
};

struct VaePatchRegion {
    std::size_t top, left, height, width;
#ifdef __CUDACC__
    __host__ __device__
#endif
        std::size_t points() const {
        return height * width;
    }
};

inline constexpr auto vae_patch_padding = std::numeric_limits<std::size_t>::max();

// Pure layout mapping; callers supply a validated shape, spatial point and column.
#ifdef __CUDACC__
__host__ __device__
#endif
    inline std::size_t
    vae_patch_index(VaeConvolution shape, std::size_t point, std::size_t column) {
    const auto channel = column / (shape.kernel * shape.kernel);
    const auto y = point / shape.width + column / shape.kernel % shape.kernel;
    const auto x = point % shape.width + column % shape.kernel;
    const auto padding = shape.kernel / 2;
    if (y < padding || x < padding || y - padding >= shape.height
        || x - padding >= shape.width)
        return vae_patch_padding;
    return channel * shape.points() + (y - padding) * shape.width + x - padding;
}

// Callers supply nonzero columns and a sub-tile within the admitted planar stride.
#ifdef __CUDACC__
__host__ __device__
#endif
    inline std::size_t vae_planar_group_index(
        std::size_t matrix_index,
        std::size_t columns,
        std::size_t first,
        std::size_t stride
    ) {
    return (matrix_index % columns) * stride + first + matrix_index / columns;
}

// A bounded rectangle and halo, shared by every output-column partition. Tiles
// crossing a scanline include its full width; single-line tiles use only their ROI.
inline VaePatchRegion
vae_patch_region(VaeConvolution shape, std::size_t first, std::size_t count) {
    shape.validate();
    if (!count || first >= shape.points() || count > shape.points() - first)
        throw std::invalid_argument("image_vae_conv_tile");
    const auto padding = shape.kernel / 2;
    const auto first_y = first / shape.width,
               last_y = (first + count - 1) / shape.width;
    const auto first_x = first % shape.width,
               last_x = (first + count - 1) % shape.width;
    const auto top = first_y > padding ? first_y - padding : 0;
    const auto bottom = std::min(shape.height, last_y + padding + 1);
    const auto left = first_y == last_y && first_x > padding ? first_x - padding : 0;
    const auto right =
        first_y == last_y ? std::min(shape.width, last_x + padding + 1) : shape.width;
    return {top, left, bottom - top, right - left};
}

inline void validate_vae_convolution(
    VaeConvolution shape,
    std::span<const float> input,
    std::span<const float> bias,
    std::span<float> output
) {
    shape.validate();
    if (input.size() != shape.channels * shape.points()
        || output.size() != shape.outputs * shape.points()
        || (!bias.empty() && bias.size() != shape.outputs) || !input.data()
        || !output.data() || (!bias.empty() && !bias.data()))
        throw std::invalid_argument("image_vae_conv_shape");
    for (auto source : {input, bias}) {
        const auto x = reinterpret_cast<std::uintptr_t>(source.data()),
                   y = reinterpret_cast<std::uintptr_t>(output.data());
        if (!source.empty()
            && (x <= y ? y - x < source.size_bytes() : x - y < output.size_bytes()))
            throw std::invalid_argument("image_vae_conv_overlap");
    }
}
}  // namespace kadan::image
