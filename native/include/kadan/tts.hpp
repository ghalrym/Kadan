#pragma once
#include "kadan/resources.hpp"
#include <array>
#include <atomic>
#include <functional>
#include <memory>
#include <span>
#include <string>
namespace kadan::tts {
// Qwen3-TTS text projection only; caller supplies text embeddings.
// One serialized owner. Only cancellation may change concurrently.
class TextProjection {
public:
    static constexpr std::size_t hidden=2048, max_tokens=2048;
    static constexpr Bytes weight_bytes=(2*2048*2048+2*2048)*4, metadata_bytes=4*1024*1024;
    explicit TextProjection(std::shared_ptr<Resources> resources);
    ~TextProjection();
    TextProjection(const TextProjection&)=delete;
    TextProjection& operator=(const TextProjection&)=delete;
    void load(const char* root,const std::string& basename,const std::atomic_bool& cancel);
    void unload();
    bool loaded() const {return weights_!=nullptr;}
    // Caller admits input and returned output. Projected embedding, never audio.
    // Optional observation hook cannot reenter the executor.
    std::array<float,hidden> execute(std::span<const float> input,
        const std::atomic_bool& cancel,const std::function<void()>& observed={});
    // Row-major [tokens, hidden], 1..max_tokens; caller admits both buffers.
    // Buffers must not overlap. One residency pin and fixed one-row scratch
    // cover the sequence. On cancellation/failure, discard partially written output.
    // Called before each row while pinned; the hook cannot reenter execution.
    void execute_sequence(std::span<const float> input, std::span<float> output,
        const std::atomic_bool& cancel,
        const std::function<void(std::size_t)>& observed={});
private:
    Footprint host(Bytes bytes) const;
    std::shared_ptr<Resources> resources_;
    std::unique_ptr<float[]> weights_;
    Handle resident_=0;
};
}
