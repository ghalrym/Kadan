#pragma once
#include <cuda_runtime_api.h>
#include <cstddef>
#include <cstdint>

// Internal launcher: pointers and dimensions are supplied by the owning wrapper.
// No allocations or host projection tiles. Launches on the legacy default stream.
cudaError_t kadan_launch_nvfp4(const std::uint8_t* weights, const std::uint8_t* scales,
                             float global, const float* input, float* output,
                             unsigned* status, std::size_t rows, std::size_t columns);

// Checkpoint weight-only BF16 path; activation quantization is not performed.
cudaError_t kadan_launch_nvfp4_bf16(const std::uint8_t* weights, const std::uint8_t* scales,
                             float global, const float* input, float* output,
                             unsigned* status, std::size_t rows, std::size_t columns);

// Internal fused operations reuse owner-bounded scratch; no new allocation.
struct Nvfp4Pair {const std::uint8_t* weights;const std::uint8_t* scales;float global;float* output;};
struct Nvfp4Accumulation {const unsigned* selected;const float* weights;std::size_t top_k;unsigned expert;float* accumulator;};
cudaError_t kadan_launch_nvfp4_pair(const std::uint8_t*,const std::uint8_t*,float,const float*,float*,unsigned*,std::size_t,std::size_t,bool,Nvfp4Pair);
cudaError_t kadan_launch_nvfp4_accumulate(const std::uint8_t*,const std::uint8_t*,float,const float*,float*,unsigned*,std::size_t,std::size_t,bool,Nvfp4Accumulation);
