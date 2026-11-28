#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <stdexcept>
#include <vector>

namespace kadan::image::detail {

// Pure C++ BF16 rounding (round-to-nearest-even) for CPU unit tests of
// FLUX.2 shared AdaLN modulation without requiring LibTorch or CUDA.
inline std::uint16_t float_to_bf16(float value) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    if ((bits & 0x7f800000u) == 0x7f800000u && (bits & 0x007fffffu) != 0)
        return std::uint16_t((bits >> 16) | 0x0040u);
    const std::uint32_t rounding = 0x7fffu + ((bits >> 16) & 1u);
    return std::uint16_t((bits + rounding) >> 16);
}

inline float bf16_to_float(std::uint16_t value) {
    const std::uint32_t bits = std::uint32_t(value) << 16;
    float out = 0.0f;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

struct Flux2CpuModulationSet {
    std::vector<float> shift;
    std::vector<float> scale;
    std::vector<std::uint16_t> gate;
};

struct Flux2CpuDoubleModulation {
    Flux2CpuModulationSet attention;
    Flux2CpuModulationSet mlp;

    explicit Flux2CpuDoubleModulation(std::span<const std::uint16_t> projected) {
        if (projected.empty() || projected.size() % 6 != 0)
            throw std::invalid_argument("flux2_double_modulation_shape");
        const std::size_t d = projected.size() / 6;
        auto fill = [&](Flux2CpuModulationSet& dst, std::size_t offset) {
            dst.shift.resize(d);
            dst.scale.resize(d);
            dst.gate.resize(d);
            for (std::size_t i = 0; i < d; ++i) {
                dst.shift[i] = bf16_to_float(projected[offset + i]);
                dst.scale[i] = bf16_to_float(projected[offset + d + i]) + 1.0f;
                dst.gate[i] = projected[offset + 2 * d + i];
            }
        };
        fill(attention, 0);
        fill(mlp, 3 * d);
    }
};

struct Flux2CpuSingleModulation {
    Flux2CpuModulationSet parallel;

    explicit Flux2CpuSingleModulation(std::span<const std::uint16_t> projected) {
        if (projected.empty() || projected.size() % 3 != 0)
            throw std::invalid_argument("flux2_single_modulation_shape");
        const std::size_t d = projected.size() / 3;
        parallel.shift.resize(d);
        parallel.scale.resize(d);
        parallel.gate.resize(d);
        for (std::size_t i = 0; i < d; ++i) {
            parallel.shift[i] = bf16_to_float(projected[i]);
            parallel.scale[i] = bf16_to_float(projected[d + i]) + 1.0f;
            parallel.gate[i] = projected[2 * d + i];
        }
    }
};

// Normalize and apply (1 + scale) * LayerNorm(x) + shift in FP32, then cast once to BF16.
inline std::vector<std::uint16_t> flux2_modulated_norm_cpu(
    std::span<const std::uint16_t> x,
    std::size_t rows,
    std::size_t dim,
    std::span<const float> shift,
    std::span<const float> scale,
    float eps = 1e-6f
) {
    if (x.size() != rows * dim || shift.size() != dim || scale.size() != dim || dim == 0)
        throw std::invalid_argument("flux2_modulated_norm_shape");
    std::vector<std::uint16_t> out(x.size());
    for (std::size_t r = 0; r < rows; ++r) {
        float sum = 0.0f;
        for (std::size_t i = 0; i < dim; ++i)
            sum += bf16_to_float(x[r * dim + i]);
        const float mean = sum / float(dim);
        float var = 0.0f;
        for (std::size_t i = 0; i < dim; ++i) {
            const float diff = bf16_to_float(x[r * dim + i]) - mean;
            var += diff * diff;
        }
        const float inv = 1.0f / std::sqrt(var / float(dim) + eps);
        for (std::size_t i = 0; i < dim; ++i) {
            const float normed = (bf16_to_float(x[r * dim + i]) - mean) * inv;
            out[r * dim + i] = float_to_bf16(normed * scale[i] + shift[i]);
        }
    }
    return out;
}

inline void flux2_gated_residual_cpu(
    std::span<std::uint16_t> x,
    std::span<const std::uint16_t> delta,
    std::span<const std::uint16_t> gate
) {
    if (x.size() != delta.size() || gate.empty() || x.size() % gate.size() != 0)
        throw std::invalid_argument("flux2_gated_residual_shape");
    const std::size_t d = gate.size();
    for (std::size_t i = 0; i < x.size(); ++i) {
        const float v = bf16_to_float(x[i])
                        + bf16_to_float(delta[i]) * bf16_to_float(gate[i % d]);
        x[i] = float_to_bf16(v);
    }
}

}  // namespace kadan::image::detail
