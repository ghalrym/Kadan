#pragma once
#include <atomic>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "kadan/dense_compute.hpp"

namespace kadan::drama {
struct CodecTensorSpec {
    std::string name;
    std::vector<std::uint64_t> shape;
    std::size_t count;
};

// Canonical fp32 decoder tensors after dim-0 weight-norm folding.
std::vector<CodecTensorSpec> codec_tensor_specs();
// Pure validation; no model allocation or execution.
std::size_t codec_window_samples(std::span<const std::uint32_t> codes);

// Fish s2-pro codec used by Drama 3. Serialized worker owner; cancellation
// alone is concurrent. Each window starts with zero causal convolution context
// and local RoPE positions, matching upstream bounded-window decoding.
class Codec {
public:
    static constexpr std::size_t sample_rate = 44100, codebooks = 10;
    static constexpr std::size_t samples_per_frame = 2048, max_frames = 128;
    explicit Codec(std::shared_ptr<Resources>, std::shared_ptr<DenseCompute> = {});
    ~Codec();
    Codec(const Codec&) = delete;
    Codec& operator=(const Codec&) = delete;
    void load(const char* directory, const std::atomic_bool& cancel);
    void unload();
    // Caller retains overlap frames and trims overlap*2048 samples. Returned
    // host PCM belongs to caller and must remain charged while queued/retained.
    std::vector<float> decode_window(
        std::span<const std::uint32_t> frame_major_codes, const std::atomic_bool& cancel
    );

private:
    struct Impl;
    std::shared_ptr<Resources> resources_;
    std::shared_ptr<DenseCompute> compute_;
    std::unique_ptr<Impl> model_;
    bool busy_ = false;
};
}  // namespace kadan::drama
