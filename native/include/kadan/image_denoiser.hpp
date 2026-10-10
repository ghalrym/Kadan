#include "kadan/dense_compute.hpp"
#pragma once
#include "kadan/image.hpp"
#include <vector>
namespace kadan::image {
struct DenoiserConfig {BlockConfig block;std::size_t layers=32,context=4096,channels=64;};
// Complete text-to-image denoising transformer forward. Conditioning is supplied
// by the caller; scheduler, text encoder and image VAE are separate executors.
class Denoiser {
public:
    using Hook=std::function<void(const char*,std::size_t)>;
    explicit Denoiser(std::shared_ptr<Resources>,std::shared_ptr<DenseCompute> compute={});
    ~Denoiser();
    Denoiser(const Denoiser&)=delete;
    Denoiser& operator=(const Denoiser&)=delete;
    void load(const char* root,std::span<const std::string> shards,DenoiserConfig,const std::atomic_bool&);
    void unload();
    // F32 latent [height*width,channels], condition [text,context], velocity of
    // latent shape. All caller buffers are caller-admitted; timestep in [0,1].
    void execute(std::span<const float> latent,std::span<const float> condition,
        std::size_t text,std::size_t height,std::size_t width,float timestep,
        std::span<float> velocity,const std::atomic_bool&,const Hook& hook={});
private:
    struct Impl;std::shared_ptr<Resources> resources_;std::shared_ptr<DenseCompute> compute_;std::unique_ptr<Impl> model_;bool busy_=false;
};
}
