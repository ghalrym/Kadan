#pragma once
#include <variant>
#include <vector>

namespace kadan::text {
struct AttentionState {
    std::vector<float> keys, values;
};

struct InklingLayerState {
    AttentionState attention;
    // Causal short-convolution history for K, V, attn_post_conv, and ffn_post_conv:
    // (l_cache - 1) * (2 * n_embd_k_gqa + 2 * n_embd) floats.
    std::vector<float> conv;
    bool sliding_window = false;
};

struct ModelState {
    std::vector<std::variant<AttentionState, InklingLayerState>> layers;
};
}  // namespace kadan::text
