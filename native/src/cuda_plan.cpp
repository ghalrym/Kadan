#include "kadan/cuda_plan.hpp"
#include <climits>
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
} // namespace kadan::cuda
