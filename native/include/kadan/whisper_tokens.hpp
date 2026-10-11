#pragma once
#include "kadan/resources.hpp"
#include <atomic>
#include <memory>
#include <span>
#include <string>
#include <vector>
namespace kadan::stt {
// Offline-exported Whisper byte vocabulary and English no-timestamp decode policy.
// This decodes generated IDs, not arbitrary input text into BPE tokens.
class WhisperTokens {
public:
    explicit WhisperTokens(std::shared_ptr<Resources> resources);
    ~WhisperTokens();
    WhisperTokens(const WhisperTokens&)=delete;
    WhisperTokens& operator=(const WhisperTokens&)=delete;
    void load(const char* path,std::size_t vocabulary,const std::atomic_bool& cancel);
    void unload();
    std::span<const std::uint32_t> prompt()const{return prompt_;}
    std::span<const std::uint32_t> suppressed()const{return suppressed_;}
    std::span<const std::uint32_t> first_suppressed()const{return first_;}
    std::uint32_t eos()const{return eos_;}
    // Caller admits output capacity >= max_text_bytes. Replaces malformed UTF-8.
    static constexpr std::size_t max_text_bytes=448*128*3;
    std::string decode(std::span<const std::uint32_t> ids)const;
private:
    std::shared_ptr<Resources> resources_;Handle resident_=0;
    std::vector<std::string> pieces_;
    std::vector<std::uint32_t> prompt_,suppressed_,first_;
    std::uint32_t eos_=0;
};
}
