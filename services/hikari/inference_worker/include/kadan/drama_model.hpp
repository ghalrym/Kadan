#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <span>
#include <string>

#include "kadan/dense_compute.hpp"
#include "kadan/resources.hpp"

namespace kadan::drama {
// Native experimental reference arithmetic: F32 activations, BF16 dense weights,
// original block-FP8 experts dequantized on demand. This is not the upstream
// fused FP8 kernel implementation and its latency must be measured separately.
struct ModelInfo {
    std::size_t context_limit = 4096;
    std::uint32_t vocabulary = 252288;
    std::uint32_t semantic_begin = 248099;
    std::uint32_t semantic_end = 252194;
    std::uint32_t eos = 248046;
    std::size_t codebooks = 10;
    std::size_t codebook_size = 4096;
    Bytes host_bytes = 0;
    bool fp8_experts = false;
};

// Config-only inspection: never opens or reads checkpoint weight payloads.
ModelInfo inspect(const std::filesystem::path&, std::size_t context_limit = 4096);

struct Sampling {
    float temperature = 0.8f;
    float top_p = 0.8f;
    std::size_t top_k = 30;
    std::uint64_t seed = 0;
};

class Generator {
public:
    Generator(
        std::shared_ptr<Resources>,
        std::shared_ptr<DenseCompute>,
        std::size_t context_limit = 4096
    );
    ~Generator();
    Generator(const Generator&) = delete;
    Generator& operator=(const Generator&) = delete;

    // Validates headers/shapes and admits all host state before payload reads.
    // Weights remain on disk; synchronous DenseCompute callbacks stream bounded
    // row slices, selecting only routed experts. No independently owned device.
    void load(const std::filesystem::path&, const std::atomic_bool& cancel);
    const ModelInfo& info() const;
    // Text-only encoded prompt. Semantic reference frames require a richer input
    // contract and are rejected. Each callback holds exactly ten codec codes.
    bool generate(
        std::span<const std::uint32_t> prompt,
        std::size_t max_frames,
        const std::atomic_bool& cancel,
        const std::function<void(std::span<const std::uint32_t>)>& on_frame,
        Sampling sampling = {}
    );

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}  // namespace kadan::drama
