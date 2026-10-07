#pragma once
#include "kadan/quantization.hpp"
#include <cstddef>

namespace kadan::cuda {
// CPU-only allocation plan. All regions share one allocation on one device.
struct Nvfp4Plan {
    std::size_t rows, columns, weight_bytes, block_bytes;
    std::size_t block_offset, input_offset, output_offset, status_offset, device_bytes;
};
Nvfp4Plan plan_nvfp4(const quantization::Matrix& matrix, std::size_t device_budget_bytes);
} // namespace kadan::cuda
