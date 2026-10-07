#pragma once
#include "kadan/sequence_state.hpp"
#include "kadan/resources.hpp"
#include <memory>

namespace kadan::cuda {
struct LayerStateView {
    checkpoint::LayerKind kind;
    // Device pointers, never host-dereference. Valid only during the active step.
    // Full attention: key/value history includes a writable current-token slot.
    // Producer writes that slot before reading attention history through it.
    void* key_history = nullptr;
    void* value_history = nullptr;
    void* key_current = nullptr;
    void* value_current = nullptr;
    void* convolution = nullptr;
    void* recurrent = nullptr;
    std::size_t history_tokens = 0, kv_token_bytes = 0, conv_bytes = 0, recurrent_bytes = 0;
};
// Single-device batch-one state arena. Same creating-thread/current-SM86-device
// contract as projections. No device changes, model execution, private streams
// or multi-GPU commit protocol. Caller retains the shared resource authority.
class SequenceState {
public:
    SequenceState(const checkpoint::TextArchitecture& architecture,std::size_t token_capacity,
                  int device,std::shared_ptr<Resources> resources);
    ~SequenceState();
    SequenceState(const SequenceState&) = delete;
    SequenceState& operator=(const SequenceState&) = delete;
    const SequenceStatePlan& plan() const;
    std::size_t committed_tokens() const;
    // Host-only health query; false after invalidation, runtime poison or close.
    bool valid() const;
    StateStep begin();
    LayerStateView layer(StateStep step,std::size_t index);
    // Trusted producer acknowledgement, not proof of buffer initialization.
    // Enqueue writes on the legacy stream before marking; commit synchronizes.
    void written(StateStep step,std::size_t index);
    void commit(StateStep step); // Synchronize before publishing logical progress.
    void abort(StateStep step); // Synchronize and invalidate; reset required.
    void reset(); // Zero all allocated bytes; only then permit a fresh sequence.
    // Idle valid-state diagnostic copy to caller-admitted RAM; no allocation.
    void read_bytes(std::size_t offset,std::span<std::uint8_t> destination);
    void close(); // Uncertain cleanup retains admission; no automatic retries.
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace kadan::cuda
