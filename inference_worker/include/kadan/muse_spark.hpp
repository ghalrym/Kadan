#pragma once

#include "kadan/model_worker.hpp"
#include "kadan/worker_compute.hpp"

namespace kadan::muse_spark {
// Text-only decoder. Checkpoint data stays on disk; projections use the worker's
// bounded source/weight banks. MLX affine3/4 has BF16 activation boundaries;
// the verified GGUF Q3_K_XL tensor inventory uses FP32 graph arithmetic and
// explicit conversion semantics. No multimodal or speculative draft execution.
void validate_config(const char*, std::shared_ptr<checkpoint::MemoryBudget>);

struct Plan {
    std::size_t context, host, packed, tensors, execution_weights;
};

std::size_t state_bytes(std::size_t context);
Plan plan(const char*, std::size_t context, const std::atomic_bool* cancel = nullptr);

class Engine final : public serving::ResidentEngine {
public:
    Engine(
        const char*,
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
    // false commits a prefill token without computing the vocabulary projection.
    serving::Token step(unsigned token, bool logits) override;
    void close() override;
    std::shared_ptr<Resources> resources() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}  // namespace kadan::muse_spark
