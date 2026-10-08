#pragma once
#include <cuda_runtime_api.h>
#include <cstddef>
#include <cstdint>

// Internal trusted launch: only Fp8Projection supplies validated immutable data.
cudaError_t kadan_launch_fp8(const std::uint8_t* weights, const float* scales,
                            bool row_scales, const float* input, float* output,
                            unsigned* status, std::size_t rows, std::size_t columns);

// Checkpoint weight-only BF16 path; activation quantization is not performed.
cudaError_t kadan_launch_fp8_bf16(const std::uint8_t* weights, const float* scales,
                            bool row_scales, const float* input, float* output,
                            unsigned* status, std::size_t rows, std::size_t columns);
