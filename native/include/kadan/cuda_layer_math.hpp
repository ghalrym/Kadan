#pragma once
#include "kadan/layer_math.hpp"
#include <cuda_runtime_api.h>
namespace kadan::cuda {
// Borrowed device spans on the caller's current SM86 device. Caller admits and
// pins all storage, including one separate unsigned status word, until legacy
// stream completion. No allocation, host copies, synchronization or device switch.
// Host shape/alias errors throw before launch. Clear status to zero on the same
// stream before a chain; kernels atomically OR 1 for invalid input/range. On CUDA
// error or nonzero status discard outputs; caller handles cleanup/quarantine.
// Status must not overlap any span. Buffers must be valid for their stated sizes;
// address/device provenance remains the trusted native caller's responsibility.
cudaError_t rms_norm(math::Norm shape,std::span<const float> input,std::span<const float> weight,
                     std::span<float> output,unsigned* status);
cudaError_t gate(math::Gate kind,std::span<const float> input,std::span<const float> modulation,
                 std::span<float> output,unsigned* status);
cudaError_t rope(math::Rotary shape,std::size_t position,std::span<const float> frequencies,
                 std::span<const float> input,std::span<float> output,unsigned* status);
} // namespace kadan::cuda
