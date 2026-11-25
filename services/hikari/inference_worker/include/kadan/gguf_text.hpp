#pragma once
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <string>

#include "kadan/checkpoint.hpp"
#include "kadan/gguf_checkpoint.hpp"
#include "kadan/inference_cache.hpp"
#include "kadan/model_worker.hpp"
#include "kadan/text_prefix.hpp"
#include "kadan/text_projection_timing.hpp"
#include "kadan/text_state.hpp"
#include "kadan/worker_compute.hpp"

namespace kadan::text {
// Native text + vision engine for the GGUF checkpoint: inkling2_loop + inkling2v.
// Checkpoint data stays on disk; projections stream packed rows through the
// worker's bounded device banks and layer math is F32 on the host.

// Validates the exact metadata contract of a supported architecture and
// returns its config model type ("inkling2_loop").
std::string validate_metadata(const checkpoint::GgufMetadata&);
// Validates the config.json derived from that metadata.
void validate_config(const char* root, std::shared_ptr<checkpoint::MemoryBudget>);

// Pre-projected image embeddings that replace prompt tokens starting at `offset`
// (absolute prompt position) without applying token_embd_norm.
struct ImageSpan {
    std::size_t offset = 0;
    std::size_t tokens = 0;
    std::vector<float> embeddings;
};

// Returns the exact number of Inkling 2 vision tokens a PNG image produces.
std::size_t image_token_count(std::span<const std::uint8_t> png_bytes);

// host covers metadata, parameters, state and scratch. cache is host RAM for
// weight slices that do not win VRAM residency; zero streams them from disk.
struct Plan {
    std::size_t context, host, packed, tensors, execution_weights, cache = 0;
};

Plan plan(
    const char* root, std::size_t context, const std::atomic_bool* cancel = nullptr
);

struct PrefixSnapshot final : serving::InferenceCache {
    Prefix prefix;
    ModelState state;
    std::vector<WeightIdentity> weights;
    std::size_t context = 0;
    Bytes bytes() const override;
};

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
    std::size_t begin_request(
        std::span<const unsigned> prompt,
        std::string_view conversation,
        bool reuse_prefix
    ) override;
    void end_request() override;
    // Prefill of the next request keeps a host copy of the model state after
    // this many prompt tokens (zero for none). A later request in the same
    // conversation whose prompt extends those tokens resumes from that copy
    // when the live state has moved past them, as when callers replace the
    // last message each turn.
    void checkpoint_at(std::size_t tokens);
    Footprint prefix_memory() const;
    std::unique_ptr<PrefixSnapshot> take_prefix();
    bool restore_prefix(std::unique_ptr<PrefixSnapshot>);
    std::size_t stored_prefix_tokens() const;
    std::string_view prefix_reason() const;
    void park() override;
    void reset() override;
    // Encodes a PNG image through the companion mmproj vision encoder into
    // [tokens * n_embd] projected embeddings ready for prefill().
    ImageSpan encode_image(std::span<const std::uint8_t> png_bytes);
    // false commits a token without computing the vocabulary projection.
    serving::Token step(unsigned token, bool logits) override;
    // Layer-major bounded prefill. Progress token counts commit only after all
    // layers of a chunk; failures poison the request state until reset.
    serving::Token prefill(
        std::span<const unsigned>,
        const std::function<void(const char*, std::size_t)>& progress
    );
    serving::Token prefill(
        std::span<const unsigned>,
        std::span<const ImageSpan> images,
        const std::function<void(const char*, std::size_t)>& progress
    );
    void close() override;
    std::shared_ptr<Resources> resources() const;

    // Work since the previous call, and current weight residency.
    struct Timing {
        double projection_seconds, attention_seconds, prepare_seconds, device_seconds,
            host_wait_seconds, recurrent_seconds;
        std::size_t projections, batches;
        std::uint64_t streamed_bytes, host_bytes, packed_bytes, device_bytes, ram_bytes;
        std::size_t device_slices, ram_slices;
        ProjectionTiming projection_detail;
    };

    Timing take_timing();
    void set_projection_timing(bool enabled);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}  // namespace kadan::text
