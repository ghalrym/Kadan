#pragma once

#include "kadan/checkpoint.hpp"

namespace kadan::gpt_oss {
// The two official root HF MXFP4 checkpoints share these decoder dimensions.
struct Config {
    std::size_t layers, experts;
    static constexpr std::size_t hidden = 2880, intermediate = 2880, vocab = 201088,
                                 heads = 64, kv_heads = 8, head_dim = 64,
                                 experts_per_token = 4, max_context = 131072,
                                 sliding_window = 128;
};

// Config-only validation: no index, shards, tensor payloads or devices required.
// Acquisition can call this on staged metadata before downloading weights.
Config read_config(const char* root, std::shared_ptr<checkpoint::MemoryBudget>);

enum class ExpertProjection {
    gate_up,
    down
};

// Complete root HF tensor contract. Construction reads metadata only; no
// inference capability, resource reservation or placement decision is implied.
class Manifest {
public:
    Manifest(const char* root, std::shared_ptr<checkpoint::MemoryBudget>);
    ~Manifest();
    Manifest(const Manifest&) = delete;
    Manifest& operator=(const Manifest&) = delete;
    const Config& config() const;
    std::uint64_t stored_bytes() const;
    std::size_t tensor_count() const;
    checkpoint::TensorInfo tensor(std::string_view name) const;
    void read_tensor(
        std::string_view name, std::size_t offset, std::span<std::uint8_t>
    ) const;
    // Expert payloads retain their separate admitted staging budget. Optional
    // reader routes through the owning worker's cache/cancellation policy.
    checkpoint::Projection load_expert_rows(
        std::size_t layer,
        ExpertProjection projection,
        std::size_t expert,
        std::size_t first,
        std::size_t count,
        std::size_t payload_bytes,
        std::shared_ptr<checkpoint::MemoryBudget> payload_memory,
        const checkpoint::TensorReader& reader = {}
    ) const;
    void check_unchanged() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}  // namespace kadan::gpt_oss
