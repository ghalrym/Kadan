#pragma once

#include <algorithm>
#include <cstdint>
#include <mutex>
#include <stdexcept>

namespace kadan::image {
// One private ATen pool owns these backing segments. Admission precedes cudaMalloc;
// a failed allocation rolls back, and a free is credited only after CUDA confirms it.
class ImagePoolMemory {
public:
    explicit ImagePoolMemory(std::uint64_t limit) : limit_(limit) {
    }

    bool acquire(std::uint64_t bytes) {
        std::lock_guard lock(mutex_);
        if (bytes > limit_ - current_)
            return false;
        current_ += bytes;
        peak_ = std::max(peak_, current_);
        return true;
    }

    void release(std::uint64_t bytes) {
        std::lock_guard lock(mutex_);
        if (bytes > current_)
            throw std::logic_error("image_pool_release");
        current_ -= bytes;
    }

    std::uint64_t current() const {
        std::lock_guard lock(mutex_);
        return current_;
    }

    std::uint64_t peak() const {
        std::lock_guard lock(mutex_);
        return peak_;
    }

private:
    const std::uint64_t limit_;
    mutable std::mutex mutex_;
    std::uint64_t current_ = 0, peak_ = 0;
};
}  // namespace kadan::image
