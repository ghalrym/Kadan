#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace kadan::quantization {
// Format primitives, independent of any checkpoint or inference implementation.
float e2m1(std::uint8_t nibble); // Reject codes above 15; preserve signed zero.
float e4m3fn(std::uint8_t byte); // Includes subnormals and both NaN encodings.
float fp32_le(std::span<const std::uint8_t> bytes); // Exactly four little-endian bytes.

enum class Encoding { modelopt_nvfp4, modelopt_fp8 };

// Canonical unswizzled row-major views. Callers own the storage and must keep it
// alive and immutable throughout each operation. NVFP4: low nibble is first,
// block scale per 16 columns, exactly one global multiplier. FP8: scalar or row
// multiplier. No inference from byte length; encodings must be selected explicitly.
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
std::vector<float> decode_rows(const Matrix& matrix, std::size_t first,
                               std::size_t count, std::size_t max_output_bytes);
// CPU numerical reference only: FP32-decoded weights, FP32 inputs, double
// accumulation, FP32 output. No activation quantization, bias, or dense scratch.
std::vector<float> matvec(const Matrix& matrix, std::span<const float> input,
                         std::size_t max_output_bytes);
} // namespace kadan::quantization
