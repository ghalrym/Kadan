#pragma once
#include "kadan/checkpoint.hpp"
#include "kadan/resources.hpp"
#include "kadan/h3_compute.hpp"
#include "kadan/h3_tokenizer.hpp"
#include <atomic>
#include <functional>
#include <memory>
#include <span>
#include <string>
namespace kadan::video {
// Complete 50-layer text-only H3 conditioning graph over caller-tokenized IDs.
// F32 activations, F64 reductions/rotation, serialized ConvRot INT8 projections.
// No vision tower,
// tokenizer, denoising or video generation registration. Serialized owner.
class H3TextEncoder {
public:
    static constexpr std::size_t hidden=5120, layers=50, vocab=151936, max_tokens=H3Tokenizer::max_tokens;
    static constexpr Bytes metadata_bytes=4*1024*1024;
    using Hook=std::function<void(const char*,std::size_t)>;
    explicit H3TextEncoder(std::shared_ptr<Resources> resources, std::shared_ptr<H3Compute> compute={});
    ~H3TextEncoder();
    H3TextEncoder(const H3TextEncoder&)=delete;
    H3TextEncoder& operator=(const H3TextEncoder&)=delete;
    void load(const char* root,const std::string& basename,const std::atomic_bool& cancel, std::shared_ptr<checkpoint::ReadCache> cache={});
    void unload();
    bool loaded() const {return bool(shard_);}
    // Caller owns and admits IDs. Output F32 [tokens,5120] after layer 50,
    // before a final norm, matching H3's selected feature boundary.
    void execute(std::span<const std::uint32_t> ids,const std::string& output,
                 const std::atomic_bool& cancel,const Hook& hook={});
private:
    Footprint host(Bytes bytes) const;
    std::shared_ptr<Resources> resources_;
    std::shared_ptr<H3Compute> compute_;
    std::unique_ptr<checkpoint::Shard> shard_;
    Handle metadata_=0;
    bool executing_=false;
};
}
