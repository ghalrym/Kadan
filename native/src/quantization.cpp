#include "kadan/quantization.hpp"

#include <bit>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace kadan::quantization {
static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559);
namespace {
void require(bool ok, const char* message) {
    if (!ok) throw std::invalid_argument(message);
}
std::size_t product(std::size_t a, std::size_t b) {
    if (b != 0 && a > std::numeric_limits<std::size_t>::max() / b)
        throw std::invalid_argument("shape_overflow");
    return a * b;
}
void output_budget(std::size_t elements, std::size_t bytes) {
    if (elements > bytes / sizeof(float)) throw std::length_error("output_budget");
}
void validate(const Matrix& m) {
    require(m.rows != 0 && m.columns != 0, "empty_matrix");
    product(m.rows, m.columns); // Even packed dimensions must have a representable logical size.
    if (m.encoding == Encoding::modelopt_nvfp4) {
        require(m.columns % 16 == 0, "nvfp4_block_alignment");
        require(m.weights.size() == product(m.rows, m.columns / 2), "weight_shape");
        require(m.block_scales.size() == product(m.rows, m.columns / 16), "block_scale_shape");
        require(m.multipliers.size() == 1, "nvfp4_global_scale_shape");
        for (auto byte : m.block_scales) {
            const auto value = e4m3fn(byte);
            require(std::isfinite(value) && !std::signbit(value), "invalid_block_scale");
        }
    } else if (m.encoding == Encoding::modelopt_fp8) {
        require(m.weights.size() == product(m.rows, m.columns), "weight_shape");
        require(m.block_scales.empty(), "unexpected_block_scales");
        require(m.multipliers.size() == 1 || m.multipliers.size() == m.rows, "fp8_scale_shape");
        for (auto byte : m.weights) require(std::isfinite(e4m3fn(byte)), "nonfinite_weight");
    } else {
        throw std::invalid_argument("unsupported_encoding");
    }
    for (auto scale : m.multipliers)
        require(std::isfinite(scale) && scale > 0, "invalid_multiplier");
}
float finite_float(double value) {
    if (!std::isfinite(value) || std::abs(value) > std::numeric_limits<float>::max())
        throw std::overflow_error("nonfinite_result");
    return static_cast<float>(value);
}
float weight(const Matrix& m, std::size_t row, std::size_t column) {
    if (m.encoding == Encoding::modelopt_fp8) {
        const auto multiplier = m.multipliers[m.multipliers.size() == 1 ? 0 : row];
        return finite_float(static_cast<double>(e4m3fn(m.weights[row * m.columns + column])) * multiplier);
    }
    const auto packed = m.weights[row * (m.columns / 2) + column / 2];
    const auto code = static_cast<std::uint8_t>((packed >> (4 * (column % 2))) & 15);
    const auto scale = e4m3fn(m.block_scales[row * (m.columns / 16) + column / 16]);
    return finite_float(static_cast<double>(e2m1(code)) * scale * m.multipliers[0]);
}
} // namespace

float e2m1(std::uint8_t nibble) {
    require(nibble < 16, "invalid_fp4_code");
    const int exponent = (nibble >> 1) & 3;
    const int fraction = nibble & 1;
    const float magnitude = exponent == 0 ? fraction * 0.5F
        : std::ldexp(1.0F + fraction * 0.5F, exponent - 1);
    return nibble & 8 ? -magnitude : magnitude;
}
float e4m3fn(std::uint8_t byte) {
    const int exponent = (byte >> 3) & 15;
    const int fraction = byte & 7;
    if (exponent == 15 && fraction == 7) return std::numeric_limits<float>::quiet_NaN();
    const float magnitude = exponent == 0 ? std::ldexp(static_cast<float>(fraction), -9)
        : std::ldexp(1.0F + fraction * 0.125F, exponent - 7);
    return byte & 128 ? -magnitude : magnitude;
}
float fp32_le(std::span<const std::uint8_t> bytes) {
    require(bytes.size() == 4, "fp32_byte_count");
    std::uint32_t bits = 0;
    for (std::size_t i = 0; i < 4; ++i) bits |= std::uint32_t{bytes[i]} << (8 * i);
    return std::bit_cast<float>(bits);
}
std::vector<float> decode_rows(const Matrix& m, std::size_t first,
                               std::size_t count, std::size_t max_output_bytes) {
    validate(m);
    require(first <= m.rows && count <= m.rows - first, "row_range");
    const auto elements = product(count, m.columns);
    output_budget(elements, max_output_bytes);
    std::vector<float> result(elements);
    for (std::size_t row = 0; row < count; ++row)
        for (std::size_t column = 0; column < m.columns; ++column)
            result[row * m.columns + column] = weight(m, first + row, column);
    return result;
}
std::vector<float> matvec(const Matrix& m, std::span<const float> input,
                         std::size_t max_output_bytes) {
    validate(m);
    require(input.size() == m.columns, "input_shape");
    for (auto value : input) require(std::isfinite(value), "nonfinite_input");
    output_budget(m.rows, max_output_bytes);
    std::vector<float> output(m.rows);
    for (std::size_t row = 0; row < m.rows; ++row) {
        double sum = 0;
        for (std::size_t column = 0; column < m.columns; ++column)
            sum += static_cast<double>(weight(m, row, column)) * input[column];
        output[row] = finite_float(sum);
    }
    return output;
}
} // namespace kadan::quantization
