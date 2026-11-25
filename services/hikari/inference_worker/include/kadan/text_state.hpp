#pragma once
#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <type_traits>
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

inline std::size_t
leading_floats(std::size_t full_size, std::size_t positions, std::size_t context) {
    if (!context || !positions || !full_size)
        return 0;
    if (positions >= context)
        return full_size;
    if (full_size % context == 0)
        return (full_size / context) * positions;
    const auto per_pos = full_size / context;
    const auto rem = full_size % context;
    const auto extra = (rem * positions + context - 1) / context;
    return std::min(full_size, per_pos * positions + extra);
}

inline std::size_t state_memory_bytes(const ModelState& state) {
    std::size_t bytes =
        state.layers.capacity() * sizeof(decltype(state.layers)::value_type);
    const auto attention_bytes = [](const AttentionState& s) {
        return (s.keys.capacity() + s.values.capacity()) * sizeof(float);
    };
    for (const auto& layer : state.layers) {
        std::visit(
            [&](const auto& s) {
                using T = std::decay_t<decltype(s)>;
                if constexpr (std::is_same_v<T, AttentionState>)
                    bytes += attention_bytes(s);
                else
                    bytes += attention_bytes(s.attention)
                             + s.conv.capacity() * sizeof(float);
            },
            layer
        );
    }
    return bytes;
}

// Copies only the leading `positions` rows of full-context KV buffers into a
// right-sized checkpoint; sliding-window ring buffers and short-convolution
// histories are copied whole.
inline ModelState
compact_state(const ModelState& full, std::size_t positions, std::size_t context) {
    if (!context || positions > context)
        throw std::runtime_error("gguf_text_checkpoint_shape");
    ModelState out;
    out.layers.reserve(full.layers.size());
    const auto slice = [&](const std::vector<float>& src) {
        const auto count = leading_floats(src.size(), positions, context);
        return std::vector<float>(src.begin(), src.begin() + count);
    };
    for (const auto& layer : full.layers) {
        std::visit(
            [&](const auto& src) {
                using T = std::decay_t<decltype(src)>;
                if constexpr (std::is_same_v<T, AttentionState>) {
                    out.layers.emplace_back(
                        AttentionState{slice(src.keys), slice(src.values)}
                    );
                } else {
                    InklingLayerState dst;
                    dst.sliding_window = src.sliding_window;
                    if (src.sliding_window) {
                        dst.attention.keys = src.attention.keys;
                        dst.attention.values = src.attention.values;
                    } else {
                        dst.attention.keys = slice(src.attention.keys);
                        dst.attention.values = slice(src.attention.values);
                    }
                    dst.conv = src.conv;
                    out.layers.emplace_back(std::move(dst));
                }
            },
            layer
        );
    }
    return out;
}

// Restores a compacted checkpoint into a full-capacity live ModelState.
inline void restore_compact_state(
    const ModelState& compact,
    ModelState& target,
    std::size_t positions,
    std::size_t context
) {
    if (!context || positions > context || compact.layers.size() != target.layers.size())
        throw std::runtime_error("gguf_text_checkpoint_layers");
    const auto restore_leading = [&](const std::vector<float>& src,
                                     std::vector<float>& dst) {
        const auto expected = leading_floats(dst.size(), positions, context);
        if (src.size() != expected || expected > dst.size())
            throw std::runtime_error("gguf_text_checkpoint_shape");
        std::copy_n(src.begin(), expected, dst.begin());
    };
    for (std::size_t i = 0; i < compact.layers.size(); ++i) {
        if (compact.layers[i].index() != target.layers[i].index())
            throw std::runtime_error("gguf_text_checkpoint_layers");
        std::visit(
            [&](const auto& src) {
                using T = std::decay_t<decltype(src)>;
                auto& dst = std::get<T>(target.layers[i]);
                if constexpr (std::is_same_v<T, AttentionState>) {
                    restore_leading(src.keys, dst.keys);
                    restore_leading(src.values, dst.values);
                } else {
                    if (src.sliding_window != dst.sliding_window
                        || src.conv.size() != dst.conv.size())
                        throw std::runtime_error("gguf_text_checkpoint_shape");
                    if (src.sliding_window) {
                        if (src.attention.keys.size() != dst.attention.keys.size()
                            || src.attention.values.size() != dst.attention.values.size())
                            throw std::runtime_error("gguf_text_checkpoint_shape");
                        std::copy(
                            src.attention.keys.begin(),
                            src.attention.keys.end(),
                            dst.attention.keys.begin()
                        );
                        std::copy(
                            src.attention.values.begin(),
                            src.attention.values.end(),
                            dst.attention.values.begin()
                        );
                    } else {
                        restore_leading(src.attention.keys, dst.attention.keys);
                        restore_leading(src.attention.values, dst.attention.values);
                    }
                    std::copy(src.conv.begin(), src.conv.end(), dst.conv.begin());
                }
            },
            compact.layers[i]
        );
    }
}
}  // namespace kadan::text
