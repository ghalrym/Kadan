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
// FP8 admission checks every byte/scale and exact decoded-weight range once.
// A plan is allocation metadata, not a transferable validation capability: the
// owning projection always plans/validates its own immutable constructor input.
struct Fp8Plan {
    std::size_t rows, columns, weight_bytes, scale_bytes, scale_count;
    std::size_t scale_offset, input_offset, output_offset, status_offset, device_bytes;
};
Fp8Plan plan_fp8(const quantization::Matrix& matrix, std::size_t device_budget_bytes);
} // namespace kadan::cuda
