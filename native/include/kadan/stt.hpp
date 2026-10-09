#pragma once
#include "kadan/resources.hpp"
#include <atomic>
#include <functional>
#include <memory>
#include <span>

namespace kadan::stt {
// Whisper-compatible log-mel frontend only. One serialized owner; cancellation
// may change concurrently. Caller owns/admit filters, PCM and output spans.
class LogMel {
public:
    static constexpr std::size_t fft=400, frequencies=201, hop=160;
    static constexpr std::size_t min_samples=201, max_samples=16000;
    static constexpr Bytes table_bytes=(2*frequencies*fft+fft)*sizeof(double);
    static constexpr Bytes scratch_bytes=(fft+frequencies)*sizeof(double);
    explicit LogMel(std::shared_ptr<Resources> resources);
    ~LogMel();
    LogMel(const LogMel&)=delete;
    LogMel& operator=(const LogMel&)=delete;
    static std::size_t frames(std::size_t samples);
    static Bytes resident_bytes(std::size_t bins);
    void load(std::size_t bins,std::span<const float> filters,const std::atomic_bool& cancel);
    void unload();
    bool loaded() const {return tables_!=nullptr;}
    // Band-major [bins, floor(samples/160)]. Partial output must be discarded
    // on any exception/cancellation. No transcript or audio decoder is provided.
    // Observation is called once per frame while pinned; it cannot reenter.
    void execute(std::span<const float> pcm,std::span<float> output,
        const std::atomic_bool& cancel,const std::function<void(std::size_t)>& observed={});
private:
    Footprint host(Bytes bytes) const;
    std::shared_ptr<Resources> resources_;
    std::unique_ptr<double[]> tables_;
    std::unique_ptr<float[]> filters_;
    std::size_t bins_=0;
    Handle resident_=0;
};
}
