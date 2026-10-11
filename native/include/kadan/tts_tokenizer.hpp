#pragma once
#include "kadan/resources.hpp"
#include <atomic>
#include <memory>
#include <span>
#include <string>
namespace kadan::tts {
class TextTokenizer {
public:
    // 8,000 API Unicode characters: at most 32,000 UTF-8 input bytes.
    // Allow canonical-normalization expansion; execution still requires admission.
    static constexpr std::size_t max_input_bytes=32000,max_tokens=128000;
    explicit TextTokenizer(std::shared_ptr<Resources>);
    ~TextTokenizer();
    TextTokenizer(const TextTokenizer&)=delete;
    TextTokenizer& operator=(const TextTokenizer&)=delete;
    void load(const std::string& tokenizer_json,const std::atomic_bool&);
    void unload();
    // Verbatim prompt presentation, NFC/regex/byte-level BPE, no template or
    // inserted special tokens. Caller owns/admitted output storage.
    std::size_t encode(const std::string&,std::span<std::uint32_t>,const std::atomic_bool&);
private:
    struct Impl;
    std::shared_ptr<Resources> resources_;
    std::unique_ptr<Impl> impl_;
    Handle reservation_=0;
    bool executing_=false;
};
}
