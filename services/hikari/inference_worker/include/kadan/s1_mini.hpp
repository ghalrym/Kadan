#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <span>
#include <string>

#include "kadan/checkpoint.hpp"
#include "kadan/resources.hpp"

namespace kadan::s1 {
// The supported S1-mini checkpoint is the dense Qwen3-0.6B architecture.
// This is deliberately not a general Qwen configuration or chat executor.
struct Config {
    std::size_t hidden, intermediate, layers, heads, kv_heads, head_dim, vocabulary;
    std::size_t max_positions;
    float epsilon;
    double rope_theta;
};

// Metadata validation: never opens weights or executes model arithmetic.
Config
read_config(const char* root, const std::shared_ptr<checkpoint::MemoryBudget>& budget);

class Model {
public:
    // CPU execution expands BF16 weights exactly, then uses FP32 storage and
    // FP32/FP64 accumulation. It does not reproduce BF16 intermediate rounding.
    static constexpr std::size_t max_capacity = 4096;
    explicit Model(std::shared_ptr<Resources>);
    ~Model();
    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;

    void load(const std::string& root, const std::atomic_bool& cancel);
    void begin_request(std::size_t capacity);
    // Prefill consumes every prompt token; its return is the first output token.
    std::uint32_t
    prefill(std::span<const std::uint32_t> prompt, const std::atomic_bool& cancel);
    // Consume the previous output token and return the next greedy token.
    std::uint32_t step(std::uint32_t token, const std::atomic_bool& cancel);
    void reset_request();
    void close();
    bool is_eos(std::uint32_t token) const;
    const Config& config() const;

private:
    struct Impl;
    std::shared_ptr<Resources> resources_;
    std::unique_ptr<Impl> impl_;
};
}  // namespace kadan::s1
