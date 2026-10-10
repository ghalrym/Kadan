#pragma once
#include "kadan/resources.hpp"
#include <atomic>
#include <functional>
#include <memory>
#include <span>
#include <string>
namespace kadan::tts {
struct AudioConfig {
    std::size_t codebooks=16,entries=2048,code_dim=512,latent=1024;
    std::size_t state=512,heads=16,head_dim=64,layers=8,intermediate=1024;
    std::size_t decoder=1536,window=72;
};
// Original full Qwen3-TTS 12.5-Hz codec decoder, codes -> 24-kHz waveform.
// Not a text/code generator. Serialized owner; only cancellation is concurrent.
class AudioDecoder {
public:
    static constexpr std::size_t samples_per_frame=1920,max_frames=300;
    using Hook=std::function<void(const char*,std::size_t)>;
    explicit AudioDecoder(std::shared_ptr<Resources> resources);
    ~AudioDecoder();
    AudioDecoder(const AudioDecoder&)=delete;
    AudioDecoder& operator=(const AudioDecoder&)=delete;
    void load(const char* root,const std::string& shard,AudioConfig config,const std::atomic_bool& cancel);
    void unload();
    // Caller admits frame-major codes [frames,codebooks] and output
    // [frames*1920]. Discard output on any failure/cancellation.
    void decode(std::span<const std::uint32_t> codes,std::span<float> waveform,
                const std::atomic_bool& cancel,const Hook& hook={});
private:
    struct Impl;
    std::shared_ptr<Resources> resources_;
    std::unique_ptr<Impl> model_;
    bool busy_=false;
};
}
