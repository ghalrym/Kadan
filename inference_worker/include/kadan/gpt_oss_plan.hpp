#pragma once
#include <array>
#include <cstddef>
#include <map>
#include <vector>

#include "kadan/gpt_oss_manifest.hpp"

namespace kadan::gpt_oss {
// Batch-one CUDA execution layout. Pure arithmetic: no files, devices or memory.
constexpr std::size_t qkv_rows =
                          (Config::heads + 2 * Config::kv_heads) * Config::head_dim,
                      attention_width = Config::heads * Config::head_dim,
                      kv_width = Config::kv_heads * Config::head_dim,
                      mxfp4_groups = Config::hidden / 32,
                      gate_up_rows = 2 * Config::intermediate, attention_chunk = 256,
                      alignment = 256;
// Packed E2M1 bytes plus one E8M0 scale per 32 columns.
constexpr std::size_t mxfp4_row_bytes = mxfp4_groups * 17;
// [gate_up blocks][gate_up scales][down blocks][down scales]
constexpr std::size_t expert_bytes = (gate_up_rows + Config::hidden) * mxfp4_row_bytes;

constexpr std::size_t aligned(std::size_t bytes) {
    return (bytes + alignment - 1) / alignment * alignment;
}

constexpr bool sliding(std::size_t layer) {
    return layer % 2 == 0;
}

// Keys retained by one layer: the 128-token window or the full context.
std::size_t layer_slots(std::size_t layer, std::size_t capacity);

// BF16 norms, fused QKV, output, sinks, router and per-expert biases.
std::size_t layer_dense_bytes(std::size_t experts);
// One layer's resident experts, contiguous after its dense weights.
std::size_t layer_expert_bytes(std::size_t resident);
// BF16 K and V for every retained slot.
std::size_t layer_state_bytes(std::size_t layer, std::size_t capacity);
// Final norm and LM head, on the device that runs the last layer.
std::size_t head_bytes();
// Per-device activations, attention partials and status words.
std::size_t scratch_bytes(std::size_t capacity, std::size_t experts);
// FP32 logits and the selected token, beside the head.
std::size_t head_state_bytes();
// Host-mapped BF16 embedding table; each step reads only one row.
std::size_t embedding_bytes();

// YaRN parameters from the validated rope_scaling (factor 32, beta 32/1,
// original context 4096, truncate false), matching Transformers.
struct Rope {
    std::array<float, Config::head_dim / 2> inverse_frequency;
    float scaling;
};

Rope rope();

struct DeviceBytes {
    std::size_t weights = 0, state = 0;
};

struct Placement {
    // Device that runs the last layer, the final norm and LM head.
    int head = 0;
    std::vector<int> devices;
    // Experts [0, resident) of each layer live beside it; the rest are mapped
    // host memory that kernels read across the bus.
    std::vector<std::size_t> resident;
    std::map<int, DeviceBytes> allocations;
    std::size_t host_bytes = 0;
};

// Contiguous layer ranges, ending on the largest device. Prefers complete
// residency on one device, then across devices, then the fewest host experts.
// Throws runtime_error when the dense layers and context state cannot fit.
Placement place(
    const Config&,
    std::size_t capacity,
    const std::map<int, std::size_t>& budgets,
    std::size_t headroom
);
}  // namespace kadan::gpt_oss
