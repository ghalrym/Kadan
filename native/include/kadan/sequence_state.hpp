#pragma once
#include "kadan/model_manifest.hpp"
#include <array>
#include <bitset>
#include <span>

namespace kadan {
struct LayerStateLayout {
    checkpoint::LayerKind kind{};
    std::size_t device = 0;
    std::size_t key_offset = 0, value_offset = 0, kv_token_bytes = 0;
    std::size_t conv_offset = 0, conv_bytes = 0, recurrent_offset = 0, recurrent_bytes = 0;
};
struct SequenceStatePlan {
    std::size_t layers = 0, token_capacity = 0, devices = 0;
    std::array<LayerStateLayout,256> layer{};
    std::array<std::size_t,64> device_bytes{};
};
// Batch one. Full attention: BF16 K/V [token,kv_head,head_dim]. Linear:
// BF16 convolution [qkv_channel,kernel_position], FP32 recurrent [value_head,key_dim,value_dim].
// Metadata calculation only; returned public values are not an admission capability.
SequenceStatePlan plan_sequence_state(const checkpoint::TextArchitecture& architecture,
    std::size_t token_capacity, std::span<const std::size_t> layer_devices,
    std::span<const std::size_t> device_budgets);

using StateStep = std::uint64_t;
// CPU sequencing reference. A producer must finish every layer before commit.
// In-place recurrent updates cannot be rolled back: abort invalidates ALL state
// until a synchronized physical reset. No partial prefix reuse is promised.
class StateCursor {
public:
    StateCursor(std::size_t layers,std::size_t capacity);
    StateStep begin();
    void check_step(StateStep step) const;
    void written(StateStep step,std::size_t layer);
    void ready_to_commit(StateStep step) const;
    void commit(StateStep step);
    void abort(StateStep step);
    void reset(); // Caller must finish zeroing physical storage before this transition.
    void invalidate() noexcept;
    void close() noexcept;
    std::size_t committed_tokens() const { return tokens_; }
    bool active() const { return active_!=0; }
    bool valid() const { return valid_ && !closed_; }
private:
    std::size_t layers_, capacity_, tokens_ = 0;
    StateStep next_ = 0, active_ = 0;
    std::bitset<256> written_;
    bool valid_ = true, closed_ = false;
};
} // namespace kadan
