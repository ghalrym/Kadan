#pragma once

#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <vector>

namespace kadan::quantization {
// Reduce integer classifications so large weight spans can be vectorized without
// relaxed floating-point rules. Values, including signed zero, remain untouched.
inline bool all_finite(std::span<const float> values) {
    static_assert(sizeof(float) == sizeof(std::uint32_t));
    static_assert(std::numeric_limits<float>::is_iec559);
    std::uint32_t nonfinite = 0;
    for (float value : values)
        nonfinite |= (std::bit_cast<std::uint32_t>(value) & 0x7f800000u) == 0x7f800000u;
    return nonfinite == 0;
}

// Format primitives, independent of any checkpoint or inference implementation.
float e2m1(std::uint8_t nibble);  // Reject codes above 15; preserve signed zero.
float e4m3fn(std::uint8_t byte);  // Includes subnormals and both NaN encodings.
float fp32_le(
    std::span<const std::uint8_t> bytes
);  // Exactly four little-endian bytes.

// MLX affine row-major packed U32 storage, supplied as little-endian bytes.
// Three-bit codes cross byte and U32 boundaries; four-bit codes use low nibble
// first. Each group of 64 columns has one scale and additive bias.
struct MlxAffineMatrix {
    std::size_t rows;
    std::size_t columns;
    std::size_t bits;
    std::size_t group_size;
    std::span<const std::uint8_t> weights;
    std::span<const float> scales;
    std::span<const float> biases;
};

// FP32 reference decode with double intermediate affine arithmetic.
// Decode whole rows into caller-admitted storage without allocation. Output must
// be disjoint from inputs. Structural errors throw invalid_argument; numerical
// overflow throws overflow_error. Discard output after an exception.
void decode_mlx_affine_rows(
    const MlxAffineMatrix& matrix, std::size_t first, std::span<float> output
);

enum class Encoding {
    modelopt_nvfp4,
    modelopt_fp8,
    mxfp4
};

// Canonical unswizzled row-major views. Callers own the storage and must keep it
// alive and immutable throughout each operation. NVFP4: low nibble is first,
// block scale per 16 columns, exactly one global multiplier. FP8: scalar or row
// multiplier. MXFP4: low nibble first, one E8M0 scale per 32 columns, no global
// multiplier; scale 255 is invalid. No inference from byte length; encodings must
// be selected explicitly. These views do not describe swizzled GPU layouts.
struct Matrix {
    Encoding encoding;
    std::size_t rows;
    std::size_t columns;
    std::span<const std::uint8_t> weights;
    std::span<const std::uint8_t> block_scales;
    std::span<const float> multipliers;
};

// Allocations are limited to the requested output; max_output_bytes is explicit.
// Structural/value errors throw invalid_argument; budget failures length_error;
// unrepresentable numerical results overflow_error. No caller outputs are mutated.
std::vector<float> decode_rows(
    const Matrix& matrix,
    std::size_t first,
    std::size_t count,
    std::size_t max_output_bytes
);
// CPU numerical reference only: FP32-decoded weights, FP32 inputs, double
// accumulation, FP32 output. No activation quantization, bias, or dense scratch.
std::vector<float> matvec(
    const Matrix& matrix, std::span<const float> input, std::size_t max_output_bytes
);
// Same numerical reference in caller-admitted output, with no allocation.
// Output must not overlap any input storage; discard it after an exception.
void matvec(
    const Matrix& matrix, std::span<const float> input, std::span<float> output
);
}  // namespace kadan::quantization
