#pragma once
#include <atomic>
#include <cstdint>
#include <memory>
#include <span>
#include <string>

#include "gguf_model.hpp"
#include "kadan/dense_compute.hpp"
#include "kadan/gguf_text.hpp"

namespace kadan::text {
// Companion Inkling 2 vision encoder (mmproj-Inkling-2-Mini-*.gguf).
// Loads metadata lazily when an image is encoded, runs ViT + 2x2 spatial merge +
// MLP projector, and releases its device weight slices immediately afterward so
// full VRAM returns to the LLM's Strata cache.
class InklingVision {
public:
    // Finds the unique mmproj-*.gguf companion file in `root`, if present.
    static std::string find_mmproj(const char* root);

    InklingVision(const std::string& mmproj_path, const std::atomic_bool* cancel);
    ~InklingVision();
    InklingVision(const InklingVision&) = delete;
    InklingVision& operator=(const InklingVision&) = delete;

    void set_cancel(const std::atomic_bool& cancel);

    // Decodes a PNG image and returns [tokens * 2304] projected embeddings.
    ImageSpan encode(std::span<const std::uint8_t> png_bytes, DenseCompute& compute);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}  // namespace kadan::text
