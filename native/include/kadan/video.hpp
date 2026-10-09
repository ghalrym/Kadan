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
// H3 per-head affine-free Q/K RMSNorm and 48-channel 3D NeoX rotary stage.
// No learned weights; the resident is an admitted 8-value frequency table.
class H3QkRope {
public:
    static constexpr std::size_t heads=32, head_dim=64, width=6144, max_tokens=8;
    static constexpr Bytes resident_bytes=8*sizeof(float);
    static constexpr Bytes scratch_bytes=(3*head_dim+48)*sizeof(float);
    explicit H3QkRope(std::shared_ptr<Resources> resources);
    ~H3QkRope();
    H3QkRope(const H3QkRope&) = delete;
    H3QkRope& operator=(const H3QkRope&) = delete;
    void load(const std::atomic_bool& cancel);
    void unload();
    bool loaded() const { return frequencies_ != nullptr; }
    // Serialized owner, atomic cancellation; Resources outlives this stage.
    // Caller admits both F32 QKV [tokens,32,3,64] and normalized coordinates
    // [tokens,3] (time,height,width, each in [-1,1]) for the entire call.
    // Output preserves QKV layout and V bits. Hook observes completed heads;
    // it must not reenter the stage. No coordinate/token assembly is implied.
    void execute(std::span<const float> qkv, std::span<const float> coordinates,
                 const std::string& output, const std::atomic_bool& cancel,
                 const std::function<void(std::size_t)>& on_heads = {});
private:
    Footprint host(Bytes bytes) const;
    std::shared_ptr<Resources> resources_;
    std::unique_ptr<float[]> frequencies_;
    Handle resident_ = 0;
};
// Block 0 non-causal attention over supplied post-RoPE QKV, then to_out.
// Caller assembles the complete sequence; this bounded component is not a decoder.
class H3DecoderAttention {
public:
    static constexpr std::size_t heads=32, head_dim=64, hidden=2048, width=6144, max_tokens=8;
    static constexpr Bytes weight_bytes=(hidden*hidden+hidden)*sizeof(float);
    static constexpr Bytes metadata_bytes=H3DecoderInput::metadata_bytes;
    static constexpr Bytes scratch_bytes=(hidden*2+max_tokens)*sizeof(float);
    explicit H3DecoderAttention(std::shared_ptr<Resources> resources);
    ~H3DecoderAttention();
    H3DecoderAttention(const H3DecoderAttention&) = delete;
    H3DecoderAttention& operator=(const H3DecoderAttention&) = delete;
    void load(const char* root, const std::string& basename, const std::atomic_bool& cancel);
    void unload();
    bool loaded() const { return weights_ != nullptr; }
    // Serialized owner; caller admits F32 [tokens,32,3,64] for the whole call.
    // No mask, causal mode or dropout. Hook observes completed projection rows
    // in groups of 64 and must not reenter. Output is [tokens,2048], not frames.
    void execute(std::span<const float> qkv, const std::string& output,
                 const std::atomic_bool& cancel,
                 const std::function<void(std::size_t)>& on_rows = {});
private:
    Footprint host(Bytes bytes) const;
    std::shared_ptr<Resources> resources_;
    std::unique_ptr<float[]> weights_;
    Handle resident_ = 0;
};
} // namespace kadan::video
