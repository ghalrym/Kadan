#include "kadan/cuda_plan.hpp"
#include <climits>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace kadan::cuda {
namespace {
std::size_t add(std::size_t a, std::size_t b) {
    if (b > std::numeric_limits<std::size_t>::max() - a) throw std::invalid_argument("cuda_plan_overflow");
    return a + b;
}
std::size_t times(std::size_t a, std::size_t b) {
    if (b && a > std::numeric_limits<std::size_t>::max() / b) throw std::invalid_argument("cuda_plan_overflow");
    return a * b;
}
std::size_t aligned(std::size_t value) { return add(value, 255) & ~std::size_t{255}; }
}
Nvfp4Plan plan_nvfp4(const quantization::Matrix& matrix, std::size_t budget) {
    if (matrix.encoding != quantization::Encoding::modelopt_nvfp4)
        throw std::invalid_argument("cuda_requires_modelopt_nvfp4");
    if (matrix.rows == 0 || matrix.rows > INT_MAX || matrix.columns == 0 || matrix.columns > INT_MAX)
        throw std::invalid_argument("cuda_shape_limit");
    // Full canonical shape/scale validation without allocating dense weights.
    quantization::decode_rows(matrix, 0, 0, 0);
    Nvfp4Plan p{matrix.rows, matrix.columns, matrix.weights.size(), matrix.block_scales.size(), 0, 0, 0, 0, 0};
    p.block_offset = aligned(p.weight_bytes);
    p.input_offset = aligned(add(p.block_offset, p.block_bytes));
    p.output_offset = aligned(add(p.input_offset, times(p.columns, sizeof(float))));
    p.status_offset = aligned(add(p.output_offset, times(p.rows, sizeof(float))));
    p.device_bytes = aligned(add(p.status_offset, sizeof(unsigned)));
    if (p.device_bytes > budget) throw std::length_error("cuda_device_budget");
    return p;
}
Fp8Plan plan_fp8(const quantization::Matrix& matrix, std::size_t budget) {
    if (matrix.encoding != quantization::Encoding::modelopt_fp8)
        throw std::invalid_argument("cuda_requires_modelopt_fp8");
    if (matrix.rows == 0 || matrix.rows > INT_MAX || matrix.columns == 0 || matrix.columns > INT_MAX)
        throw std::invalid_argument("cuda_shape_limit");
    quantization::decode_rows(matrix, 0, 0, 0); // Canonical shape/finite-value validation, no dense allocation.
    Fp8Plan p{matrix.rows, matrix.columns, matrix.weights.size(),
        times(matrix.multipliers.size(), sizeof(float)), matrix.multipliers.size(), 0, 0, 0, 0, 0};
    p.scale_offset = aligned(p.weight_bytes);
    p.input_offset = aligned(add(p.scale_offset, p.scale_bytes));
    p.output_offset = aligned(add(p.input_offset, times(p.columns, sizeof(float))));
    p.status_offset = aligned(add(p.output_offset, times(p.rows, sizeof(float))));
    p.device_bytes = aligned(add(p.status_offset, sizeof(unsigned)));
    if (p.device_bytes > budget) throw std::length_error("cuda_device_budget");
    // FP8 has <=4 significant bits; its product with an FP32 scale is exact
    // in double. Reject before rounding, including values rounding to FLT_MAX.
    // Immutable uploaded weights/scales preserve this invariant across tokens.
    for (std::size_t row = 0; row < p.rows; ++row) {
        const double scale = matrix.multipliers[p.scale_count == 1 ? 0 : row];
        for (std::size_t col = 0; col < p.columns; ++col) {
            const double weight = quantization::e4m3fn(matrix.weights[row * p.columns + col]) * scale;
            if (std::abs(weight) > std::numeric_limits<float>::max())
                throw std::overflow_error("nonfinite_result");
        }
    }
    return p;
}
} // namespace kadan::cuda
