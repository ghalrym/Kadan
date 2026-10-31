#pragma once

#include <array>
#include <atomic>
#include <cmath>
#include <memory>
#include <span>

#include "kadan/resources.hpp"

namespace kadan::stt {
struct WhisperCrossAttention {
    std::span<const float> keys, values;
};

// Encoded audio and model weights remain immutable for this request. The full
// cache is admitted before decoding, and no layer becomes reusable mid-fill.
class WhisperCrossAttentionCache {
public:
    static constexpr Bytes metadata_bytes = 512;

    static Bytes bytes(std::size_t layers, std::size_t positions, std::size_t width) {
        if (!layers || layers > 32 || !positions || positions > 1500 || !width
            || width > 1280)
            throw std::invalid_argument("whisper_cross_attention_shape");
        return 2 * Bytes(layers) * positions * width * sizeof(float) + metadata_bytes;
    }

    WhisperCrossAttentionCache(
        Resources& resources,
        std::size_t layers,
        std::size_t positions,
        std::size_t width
    )
        : resources_(resources), layers_(layers), elements_(positions * width) {
        auto footprint = resources_.snapshot().capacity;
        std::fill(footprint.begin(), footprint.end(), 0);
        footprint[0] = bytes(layers, positions, width);
        handle_ = resources_.reserve(Workload::speech, std::move(footprint));
        try {
            data_ = std::make_unique<float[]>(2 * layers_ * elements_);
        } catch (...) {
            resources_.released(handle_);
            throw;
        }
    }

    ~WhisperCrossAttentionCache() {
        data_.reset();
        resources_.released(handle_);
    }

    WhisperCrossAttentionCache(const WhisperCrossAttentionCache&) = delete;
    WhisperCrossAttentionCache& operator=(const WhisperCrossAttentionCache&) = delete;

    template <class Fill>
    WhisperCrossAttention
    get(std::size_t layer, const std::atomic_bool& cancel, Fill&& fill) {
        stop(cancel);
        if (layer >= layers_)
            throw std::out_of_range("whisper_cross_attention_layer");
        auto keys = std::span(data_.get() + 2 * layer * elements_, elements_);
        auto values = std::span(keys.data() + elements_, elements_);
        if (!ready_[layer]) {
            fill(keys, values);
            for (auto input : {keys, values})
                for (std::size_t i = 0; i < input.size(); ++i) {
                    if (i % 4096 == 0)
                        stop(cancel);
                    if (!std::isfinite(input[i]))
                        throw std::runtime_error("whisper_nonfinite");
                }
            stop(cancel);
            ready_[layer] = true;
        }
        return {keys, values};
    }

private:
    static void stop(const std::atomic_bool& cancel) {
        if (cancel.load())
            throw std::runtime_error("whisper_cancelled");
    }

    Resources& resources_;
    std::size_t layers_, elements_;
    Handle handle_ = 0;
    std::unique_ptr<float[]> data_;
    std::array<bool, 32> ready_{};
};

static_assert(
    sizeof(WhisperCrossAttentionCache) <= WhisperCrossAttentionCache::metadata_bytes
);
}  // namespace kadan::stt
