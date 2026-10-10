#pragma once
#include "kadan/resources.hpp"
#include "kadan/checkpoint_cache.hpp"
#include "kadan/h3_compute.hpp"
#include <atomic>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <vector>
namespace kadan::video {
struct H3GenerationRequest {
    std::string prompt, output;
    std::size_t width=64, height=64, frames=22, updates=8;
    std::uint64_t seed=0;
};
struct H3GenerationPaths { std::string tokenizer, text, denoiser, turbo, vae, audio_vae; };
// Header-only preparation. The caller has admitted `limit` in addition to the
// execution envelope; retained raw bytes and metadata cannot exceed it.
std::shared_ptr<checkpoint::ReadCache> h3_weight_cache(std::shared_ptr<Resources>,
    const H3GenerationPaths&,Bytes limit,const std::atomic_bool&);
// Native joint video/audio path with optional CUDA execution. With audio_vae,
// output is MP4 with stereo32kHz audio, or diagnostic YUV4MPEG2 plus OUTPUT.wav.
// An empty audio_vae permits silent diagnostic fixtures. F32/F64, not BF16 parity.
class H3Generation {
public:
    using Hook=std::function<void(const char*,std::size_t)>;
    explicit H3Generation(std::shared_ptr<Resources> resources, std::shared_ptr<H3Compute> compute={},std::shared_ptr<checkpoint::ReadCache> cache={});
    void execute(const H3GenerationPaths&,const H3GenerationRequest&,
                 const std::atomic_bool&,const Hook& = {});
private:
    std::shared_ptr<Resources> resources_;
    std::shared_ptr<H3Compute> compute_;
    std::shared_ptr<checkpoint::ReadCache> cache_;
    bool busy_=false;
};
namespace h3 {
// Seven latent tokens decode to 28 raw frames. Drop three frames from each
// 20-frame subclip; blend the previous five-frame tail into the new head.
void temporal_join(std::span<float> raw, std::span<float> tail, std::size_t plane, bool previous);
std::vector<float> sigmas(std::size_t updates,float shift);
// Text rows inherit the video timestep; audio rows use their own sigma schedule.
void timesteps(std::span<float> output,std::size_t text_rows,std::size_t video_rows,float video_sigma,float audio_sigma);
void advance(std::span<float> state,std::span<const float> velocity,float current,float next);
// Maps normalized packed [T,H/2,W/2,24*2*2] to VAE [T,H,W,24].
void unpack(std::span<const float>,std::span<float>,std::size_t time,std::size_t height,std::size_t width);
}
}
