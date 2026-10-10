#pragma once
#include "kadan/resources.hpp"
#include <array>
#include <atomic>
#include <functional>
#include <memory>
#include <span>
#include <string>
namespace kadan::tts {
struct StackConfig{std::size_t state,heads,kv_heads,head_dim,layers,intermediate;};
struct GeneratorConfig{
    StackConfig talker{2048,16,8,128,28,6144},predictor{1024,16,8,128,5,3072};
    std::size_t text_vocabulary=151936,codec_vocabulary=3072,codebooks=16,entries=2048;
};
struct VoicePrompt{
    std::array<std::uint32_t,3> role{151644,77091,198};
    std::uint32_t text_bos=151672,text_eos=151673,text_pad=151671;
    std::uint32_t think=2154,think_bos=2156,language=2050,think_eos=2157,speaker=3061;
    std::uint32_t codec_pad=2148,codec_bos=2149,codec_eos=2150;
};
struct GeneratedCodes{std::size_t frames;bool stopped;};
// Full custom-voice, non-streaming talker and residual-code predictor. Greedy
// generation with explicit first-codebook repetition penalty; no voice cloning.
// Serialized owner. Caller admits input token IDs and frame-major codec output.
class CodeGenerator{
public:
    using Hook=std::function<void(const char*,std::size_t)>;
    explicit CodeGenerator(std::shared_ptr<Resources> resources);
    ~CodeGenerator();
    CodeGenerator(const CodeGenerator&)=delete;
    CodeGenerator& operator=(const CodeGenerator&)=delete;
    void load(const char* root,const std::string& shard,GeneratorConfig config,const std::atomic_bool& cancel);
    void unload();
    // At most 2048 text tokens / 300 output frames. On stopped=false the caller
    // must report truncation; this is not a complete speech result.
    GeneratedCodes generate(std::span<const std::uint32_t> text,const VoicePrompt& voice,
        std::span<std::uint32_t> codes,const std::atomic_bool& cancel,const Hook& hook={},
        std::size_t minimum_frames=2,float repetition_penalty=1.05f);
private:
    struct Impl;
    std::shared_ptr<Resources> resources_;
    std::unique_ptr<Impl> model_;
    bool busy_=false;
};
}
