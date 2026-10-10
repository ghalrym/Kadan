#pragma once
#include "kadan/dense_compute.hpp"
#include <atomic>
#include <functional>
#include <memory>
#include <span>
#include <string>
namespace kadan::image {
struct VaeConfig { std::size_t base=144, latent=64, channels=4, residuals=2; };
// Single-frame Qwen Image 2.1 decoder. Input/output are planar CHW. The caller
// supplies denormalized latents and owns admitted output memory. No video cache.
class VaeDecoder {
public:
    using Hook=std::function<void(std::size_t)>;
    explicit VaeDecoder(std::shared_ptr<Resources>,std::shared_ptr<DenseCompute> compute={});
    ~VaeDecoder();
    void load(const char* root,const std::string& shard,VaeConfig,const std::atomic_bool&);
    void decode(std::span<const float>,std::size_t height,std::size_t width,std::span<float>,const std::atomic_bool&,const Hook& = {});
    void unload();
private:
    struct Impl;
    std::shared_ptr<Resources> resources_;std::shared_ptr<DenseCompute> compute_;
    std::unique_ptr<Impl> model_;
    bool busy_=false;
};
}
