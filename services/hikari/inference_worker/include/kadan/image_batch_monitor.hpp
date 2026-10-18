#pragma once

#include <chrono>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <optional>
#include <thread>

#include "kadan/resources.hpp"

namespace kadan::serving {
struct ImageHostMeasurement {
    Bytes baseline;
    Bytes peak;
};

// Process RSS is useful only while the scheduler proves the image batch is
// isolated. Cancellation polling continues even when calibration is invalid.
class ImageBatchMonitor {
public:
    using Probe = std::function<Bytes()>;
    using UpdateCancellation = std::function<void()>;
    using Isolated = std::function<bool()>;

    ImageBatchMonitor(Probe probe, UpdateCancellation update, Isolated isolated)
        : probe_(std::move(probe)), update_(std::move(update)),
          isolated_(std::move(isolated)) {
        contaminated_ = !isolated_();
        if (!contaminated_) {
            try {
                baseline_ = peak_ = probe_();
            } catch (...) {
                failed_ = true;
            }
        }
        sampler_ = std::thread([this] {
            std::unique_lock lock(mutex_);
            while (!done_) {
                update_();
                sample();
                changed_.wait_for(lock, std::chrono::milliseconds(10), [&] {
                    return done_;
                });
            }
        });
    }

    ~ImageBatchMonitor() {
        stop();
    }

    std::optional<ImageHostMeasurement> finish() {
        stop();
        update_();
        sample();
        if (contaminated_)
            return std::nullopt;
        if (failed_)
            throw std::runtime_error("image_host_memory_probe");
        return ImageHostMeasurement{baseline_, peak_};
    }

private:
    void sample() {
        // The validity predicate includes a monotonic activity epoch, so even
        // a foreign workload that starts and finishes between samples is seen.
        contaminated_ = contaminated_ || !isolated_();
        if (contaminated_)
            return;
        try {
            peak_ = std::max(peak_, probe_());
        } catch (...) {
            failed_ = true;
        }
        contaminated_ = !isolated_();
    }

    void stop() {
        {
            std::lock_guard lock(mutex_);
            done_ = true;
        }
        changed_.notify_one();
        if (sampler_.joinable())
            sampler_.join();
    }

    Probe probe_;
    UpdateCancellation update_;
    Isolated isolated_;
    Bytes baseline_ = 0, peak_ = 0;
    bool done_ = false, contaminated_ = false, failed_ = false;
    std::mutex mutex_;
    std::condition_variable changed_;
    std::thread sampler_;
};
}  // namespace kadan::serving
