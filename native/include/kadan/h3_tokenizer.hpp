#pragma once
#include "kadan/resources.hpp"
#include <atomic>
#include <memory>
#include <span>
#include <string>
namespace kadan::video {
class H3Tokenizer {
public:
    static constexpr std::size_t max_input_bytes=8192,max_tokens=512;
    explicit H3Tokenizer(std::shared_ptr<Resources>);
    ~H3Tokenizer();
    H3Tokenizer(const H3Tokenizer&)=delete;
    H3Tokenizer& operator=(const H3Tokenizer&)=delete;
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
