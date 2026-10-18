#pragma once
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <string>

#include "kadan/checkpoint.hpp"
#include "kadan/gguf_checkpoint.hpp"
#include "kadan/model_worker.hpp"
#include "kadan/worker_compute.hpp"

namespace kadan::text {
// Native text engine for the tier GGUF checkpoints: qwen35moe and qwen4exp.
// Checkpoint data stays on disk; projections stream packed rows through the
// worker's bounded device banks and layer math is F32 on the host.

// Validates the exact metadata contract of a supported architecture and
// returns its config model type ("qwen35moe" or "qwen4exp").
std::string validate_metadata(const checkpoint::GgufMetadata&);
// Validates the config.json derived from that metadata.
void validate_config(const char* root, std::shared_ptr<checkpoint::MemoryBudget>);

// host covers metadata, parameters, state and scratch. cache is host RAM for
// weight slices that do not win VRAM residency; zero streams them from disk.
struct Plan {
    std::size_t context, host, packed, tensors, execution_weights, cache = 0;
};

Plan plan(
    const char* root, std::size_t context, const std::atomic_bool* cancel = nullptr
);

class Engine final : public serving::ResidentEngine {
public:
    Engine(
        const char* root,
        Plan,
        const ComputePlan&,
        std::shared_ptr<Resources> owner = {},
        const std::atomic_bool* cancel = nullptr
    );
    ~Engine() override;
    void set_cancel(const std::atomic_bool&);
    serving::Info info() const override;
    void begin_request() override;
    void end_request() override;
    void park() override;
    void reset() override;
    // false commits a token without computing the vocabulary projection.
    serving::Token step(unsigned token, bool logits) override;
    // Layer-major bounded prefill. Progress token counts commit only after all
    // layers of a chunk; failures poison the request state until reset.
    serving::Token prefill(
        std::span<const unsigned>,
        const std::function<void(const char*, std::size_t)>& progress
    );
    void close() override;
    std::shared_ptr<Resources> resources() const;
    Footprint execution_footprint() const;
    bool can_expand() const;

    // Work since the previous call, and current weight residency.
    struct Timing {
        double projection_seconds, attention_seconds, prepare_seconds, device_seconds,
            host_wait_seconds, recurrent_seconds;
        std::size_t projections, batches;
        std::uint64_t streamed_bytes, host_bytes, packed_bytes, device_bytes, ram_bytes;
        std::size_t device_slices, ram_slices;
    };

    Timing take_timing();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}  // namespace kadan::text
