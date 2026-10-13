#pragma once

#include "kadan/gpt_oss_manifest.hpp"

namespace kadan::gpt_oss {
// Interleaved [gate,linear] pairs -> BF16 activations in float storage.
// Matches the original eager BF16 operation boundaries: alpha=1.702,
// gate upper clamp 7, linear clamp [-7,7], and linear +1. Finite exact BF16
// inputs are required. Output must be disjoint; discard it after an exception.
void swiglu(std::span<const float> interleaved, std::span<float> output);

// One selected expert, batch one; no norm, routing, residual or decoder state.
// Consumes the manifest's packed MXFP4 rows and BF16 biases in fixed 64-row
// tiles. Caller admits input/output plus a separate staging/scratch budget.
// CPU reference: FP32 decoded weights, double dot accumulation, BF16 dot
// output, then BF16 bias addition, matching the original separate einsum/add.
// No fused-kernel bit parity is promised. Output may be partial on failure.
// Optional reader is shared by weights/scales/biases for cache/cancellation.
void expert(
    const Manifest& manifest,
    std::size_t layer,
    std::size_t selected,
    std::span<const float> input,
    std::span<float> output,
    std::shared_ptr<checkpoint::MemoryBudget> workspace,
    const std::atomic_bool& cancel,
    const checkpoint::TensorReader& reader = {}
);
}  // namespace kadan::gpt_oss
