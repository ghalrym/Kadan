#pragma once
#include "kadan/dense_compute.hpp"
#include "kadan/checkpoint.hpp"
#include <atomic>
#include <functional>
#include <memory>
#include <span>
#include <string>
namespace kadan::video {
struct H3AudioConfig { std::size_t projection=2048, initial=1024; };
// Decode normalized H3 audio latents, not text or an encoder posterior.
// Serialized owner. Only cancellation may change concurrently.
class H3AudioDecoder {
public:
 static constexpr std::size_t channels=32, stereo=2, samples_per_latent=800, sample_rate=32000, max_latents=640;
 using Hook=std::function<void(const char*,std::size_t)>;
 explicit H3AudioDecoder(std::shared_ptr<Resources>,std::shared_ptr<DenseCompute> compute={});
 ~H3AudioDecoder();
 H3AudioDecoder(const H3AudioDecoder&)=delete;
 H3AudioDecoder& operator=(const H3AudioDecoder&)=delete;
 void load(const char* root,const std::string& shard,H3AudioConfig,const std::atomic_bool&,std::shared_ptr<checkpoint::ReadCache> cache={});
 void unload();
 // Caller admits nonoverlapping input [stereo,time,32] and interleaved output
 // [time*800,stereo]. Discard output on any failure, including cancellation.
 void decode(std::span<const float>,std::size_t time,std::span<float>,const std::atomic_bool&,const Hook& hook={});
private:
 struct Impl;
 std::shared_ptr<Resources> resources_;std::shared_ptr<DenseCompute> compute_;
 std::unique_ptr<Impl> model_;bool busy_=false;
};
}
