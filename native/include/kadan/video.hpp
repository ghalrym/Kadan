#pragma once
#include "kadan/resources.hpp"
#include <atomic>
#include <functional>
#include <memory>
#include <span>
#include <string>

namespace kadan::video {
// Real H3 video VAE decoder input stage only. Not a video generation backend.
// Serialized executor ownership: the queue owner must not call concurrently.
// Cancellation is the only cross-thread operation. Resources outlives the stage.
class H3DecoderInput {
public:
    static constexpr std::size_t channels = 24, hidden = 2048;
    static constexpr std::size_t max_tokens = 4096; // At most 32 MiB payload per artifact.
    static constexpr Bytes weight_bytes = (48 + 24*24 + 24 + 2048*24 + 2048)*sizeof(float);
    static constexpr Bytes metadata_bytes = 4*1024*1024;
    explicit H3DecoderInput(std::shared_ptr<Resources> resources);
    ~H3DecoderInput();
    H3DecoderInput(const H3DecoderInput&) = delete;
    H3DecoderInput& operator=(const H3DecoderInput&) = delete;
    void load(const char* root, const std::string& basename, const std::atomic_bool& cancel);
    void unload(); // Physical RAM cleanup before ledger release; no GPU allocation.
    bool loaded() const { return weights_ != nullptr; }
    // Caller owns/admitted normalized FP32 token-major [tokens,24] input.
    // Output is diagnostic FP32 [tokens,2048], not frames. Rejects overwrite.
    // on_token is an observation/cancellation hook; must not reenter this object.
    void execute(std::span<const float> normalized, const std::string& output,
                 const std::atomic_bool& cancel,
                 const std::function<void(std::size_t)>& on_token = {});
private:
    Footprint host(Bytes bytes) const;
    std::shared_ptr<Resources> resources_;
    std::unique_ptr<float[]> weights_;
    Handle resident_ = 0;
};
// Block 0 pre-attention boundary, CPU F32 equations over F16 checkpoint weights.
// Output layout [tokens,32,3,64]: Q/K/V are interleaved per head.
// No attention, rotary positions, frame decoding or generation backend.
class H3DecoderQkv {
public:
    static constexpr std::size_t hidden = 2048, width = 6144, max_tokens = 8;
    static constexpr Bytes weight_bytes = (hidden + width*hidden + width)*sizeof(float);
    static constexpr Bytes metadata_bytes = H3DecoderInput::metadata_bytes;
    static constexpr Bytes scratch_bytes = (hidden+width)*sizeof(float);
    explicit H3DecoderQkv(std::shared_ptr<Resources> resources);
    ~H3DecoderQkv();
    H3DecoderQkv(const H3DecoderQkv&) = delete;
    H3DecoderQkv& operator=(const H3DecoderQkv&) = delete;
    void load(const char* root, const std::string& basename, const std::atomic_bool& cancel);
    void unload();
    bool loaded() const { return weights_ != nullptr; }
    // Serialized owner; caller admits [tokens,2048] F32 input for the whole call.
    // Observation hook runs after each 64 projection rows and must not reenter.
    void execute(std::span<const float> input, const std::string& output,
                 const std::atomic_bool& cancel,
                 const std::function<void(std::size_t)>& on_rows = {});
private:
    Footprint host(Bytes bytes) const;
    std::shared_ptr<Resources> resources_;
    std::unique_ptr<float[]> weights_;
    Handle resident_ = 0;
};
} // namespace kadan::video
