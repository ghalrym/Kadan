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

struct Routing {
    std::array<std::size_t, Config::experts_per_token> indices;
    std::array<float, Config::experts_per_token> weights;
};

// Finite BF16 logits for 32 or 128 experts. Select four in descending score
// order, then softmax only those logits in FP32 and round weights to BF16.
// Lower expert id breaks exact ties (torch.topk does not guarantee tie order).
// Rounded weights are not renormalized and need not sum to exactly one.
Routing route(std::span<const float> logits);

// Four BF16 expert rows [rank,width], weighted in rank order with FP32 FMA
// accumulation, then one BF16 output rounding. No per-product/per-expert BF16
// rounding. Width is 1..hidden; weights are finite BF16 in [0,1]. Output is
// disjoint from both inputs and must be discarded after an exception.
void weighted_sum(
    std::span<const float> values,
    std::span<const float> weights,
    std::span<float> output
);

// Consume supplied BF16 router logits and normalized BF16 input. Owns only
// four selected expert outputs plus the reused bounded expert workspace.
// No normalization, router projection, residual, or decoder state is implied.
// No fused-device bit parity; cancellation/failure invalidates partial output.
void routed_experts(
    const Manifest& manifest,
    std::size_t layer,
    std::span<const float> input,
    std::span<const float> logits,
    std::span<float> output,
    std::shared_ptr<checkpoint::MemoryBudget> workspace,
    const std::atomic_bool& cancel,
    const checkpoint::TensorReader& reader = {}
);
}  // namespace kadan::gpt_oss
