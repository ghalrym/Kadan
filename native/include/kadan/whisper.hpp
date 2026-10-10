#pragma once
#include "kadan/resources.hpp"
#include <atomic>
#include <functional>
#include <memory>
#include <span>
#include <string>
namespace kadan::stt {
struct WhisperDimensions {
    std::size_t mels,audio_context,audio_state,audio_heads,audio_layers;
    std::size_t vocabulary,text_context,text_state,text_heads,text_layers;
};
// Original CPU Whisper encoder/decoder. No Python inference or implicit downloads.
// Serialized owner; only cancellation may change concurrently. Outputs are caller
// admitted. Discard partial output on failure. No tokenizer, segmentation or beam search.
class Whisper {
public:
    using Hook=std::function<void(const char*,std::size_t)>;
    explicit Whisper(std::shared_ptr<Resources> resources);
    ~Whisper();
    Whisper(const Whisper&)=delete;
    Whisper& operator=(const Whisper&)=delete;
    void load(const char* root,const std::string& shard,WhisperDimensions dimensions,
              const std::atomic_bool& cancel);
    void unload();
    bool loaded() const;
    // [mels,2*audio_context] -> [audio_context,audio_state].
    void encode(std::span<const float> mel,std::span<float> encoded,
                const std::atomic_bool& cancel,const Hook& hook={});
    // Entire token prefix -> logits for its final position [vocabulary].
    // Recomputes prefixes; no persistent KV allocation or GPU execution.
    void decode(std::span<const std::uint32_t> tokens,std::span<const float> encoded,
                std::span<float> logits,const std::atomic_bool& cancel,const Hook& hook={});
    // Prompt + bounded greedy continuation. Output contains generated IDs only;
    // stop token is included. Returns count. Caller supplies explicit suppression.
    std::size_t greedy(std::span<const std::uint32_t> prompt,std::span<const float> encoded,
                std::span<std::uint32_t> output,std::uint32_t stop_token,
                std::span<const std::uint32_t> suppressed,
                const std::atomic_bool& cancel,const Hook& hook={},
                std::span<const std::uint32_t> first_suppressed={});
private:
    struct Impl;
    std::shared_ptr<Resources> resources_;
    std::unique_ptr<Impl> model_;
    bool busy_=false;
};
}
