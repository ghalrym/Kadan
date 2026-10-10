#pragma once
#include "kadan/dense_compute.hpp"
#include <atomic>
#include <functional>
#include <memory>
#include <span>
#include <string>
namespace kadan::image {
struct TextConfig{std::size_t vocabulary=151936,state=4096,layers=36,heads=32,kv_heads=8,head_dim=128,intermediate=12288;};
// Complete text-only Qwen3-VL conditioning graph. Returns every token's final
// decoder-layer output BEFORE the final RMSNorm, matching Qwen-Image-2.1.
// Tokenization/template presentation and system-prefix slicing belong to caller.
class TextEncoder {
public:
 using Hook=std::function<void(std::size_t)>;
 explicit TextEncoder(std::shared_ptr<Resources>,std::shared_ptr<DenseCompute> compute={});
 ~TextEncoder();
 TextEncoder(const TextEncoder&)=delete;TextEncoder& operator=(const TextEncoder&)=delete;
 void load(const char* root,std::span<const std::string> shards,TextConfig,const std::atomic_bool&,const Hook& hook={});
 void unload();
 // Caller-admitted IDs and F32 [tokens,state] output; 1..2048 text tokens.
 void execute(std::span<const std::uint32_t>,std::span<float>,const std::atomic_bool&,const Hook& hook={});
private:
 struct Impl;std::shared_ptr<Resources> resources_;std::shared_ptr<DenseCompute> compute_;std::unique_ptr<Impl> model_;bool busy_=false;
};
}
