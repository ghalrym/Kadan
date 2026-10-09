#pragma once
#include "kadan/checkpoint.hpp"
#include "kadan/resources.hpp"
#include <atomic>
#include <functional>
#include <memory>
#include <span>
#include <string>
namespace kadan::video {
// Complete released FL2VA denoiser. CPU F32 activation contract with F64
// reductions; not production BF16/GPU parity. One serialized execution owner.
class H3Denoiser {
public:
    static constexpr std::size_t hidden=5376, layers=50, heads=56, head_dim=128;
    static constexpr std::size_t max_tokens=1024, text_width=5120, video_width=96, audio_width=32;
    static constexpr Bytes metadata_bytes=4*1024*1024;
    using Hook=std::function<void(const char*,std::size_t)>;
    struct Input {
        // Packed order is text, video, audio. Inputs and output spans are
        // caller-owned and caller-admitted; execution owns all intermediate RAM.
        std::span<const float> text, video, audio;
        std::span<const float> positions; // [packed_tokens,3], temporal/height/width
        // Times and modality tags are explicit for every packed row. Conditions
        // conventionally use t=1. Tags follow checkpoint AdaLN: 0/1/2.
        std::span<const float> timesteps;
        std::span<const std::uint32_t> tags;
    };
    explicit H3Denoiser(std::shared_ptr<Resources>);
    ~H3Denoiser();
    H3Denoiser(const H3Denoiser&)=delete;
    H3Denoiser& operator=(const H3Denoiser&)=delete;
    void load(const char* root,const std::string& name,const std::atomic_bool&);
    void load_turbo(const char* root,const std::string& name,const std::atomic_bool&);
    void unload();
    bool loaded() const{return bool(shard_);}
    bool turbo_loaded() const{return bool(turbo_);}
    // Computes both output heads and returns selected live video/audio rows.
    // All 2 refiner + 50 denoiser blocks execute; no skipped-block cache.
    void execute(const Input&,std::span<float> video_out,std::span<float> audio_out,
                 const std::atomic_bool&,const Hook& hook={});
private:
    std::shared_ptr<Resources> resources_;
    std::unique_ptr<checkpoint::Shard> shard_,turbo_;
    Handle metadata_=0,turbo_metadata_=0;
    bool executing_=false;
};
}
