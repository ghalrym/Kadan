#pragma once
#include "kadan/resources.hpp"
#include <array>
#include <atomic>
#include <functional>
#include <memory>
#include <span>
#include <string>
namespace kadan::decision {
// Terminal Laya scorer/action MLP only; caller supplies post-head activations.
// One serialized owner. Only cancellation may change concurrently.
class TerminalHeads {
public:
    static constexpr std::size_t hidden=1024, action_input=1028;
    static constexpr Bytes weight_bytes=1316611*4, metadata_bytes=4*1024*1024;
    explicit TerminalHeads(std::shared_ptr<Resources> resources);
    ~TerminalHeads();
    TerminalHeads(const TerminalHeads&)=delete;
    TerminalHeads& operator=(const TerminalHeads&)=delete;
    void load(const char* root,const std::string& basename,const std::atomic_bool& cancel);
    void unload();
    bool loaded() const {return weights_!=nullptr;}
    // Caller admits input. Returns raw scorer and two action logits, not answers.
    // Optional observation hook cannot reenter the executor.
    std::array<float,3> execute(std::span<const float> marker,std::span<const float> action,
        const std::atomic_bool& cancel,const std::function<void()>& observed={});
private:
    Footprint host(Bytes bytes) const;
    std::shared_ptr<Resources> resources_;
    std::unique_ptr<float[]> weights_;
    Handle resident_=0;
};
}
