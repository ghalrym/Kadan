#pragma once

#include <algorithm>

#include "kadan/resources.hpp"

namespace kadan::image {
inline Bytes image_device_weight_bytes(Bytes payload, Bytes tensors) {
    constexpr Bytes segment = 2ULL * 1024 * 1024;
    const auto limit = std::numeric_limits<Bytes>::max();
    if (payload > limit - segment || tensors > (limit - payload - segment) / segment)
        throw std::overflow_error("image_device_weight_inventory");
    return payload + (tensors + 1) * segment;
}

// The caller has already reserved request buffers. Leave the stage's host
// staging envelope and half the remaining RAM available for changing pressure.
inline Bytes image_weight_cache_bytes(Bytes available, Bytes staging, Bytes stored) {
    constexpr Bytes metadata = 8ULL * 1024 * 1024;
    if (!stored || available <= staging)
        return 0;
    if (stored > std::numeric_limits<Bytes>::max() - metadata)
        throw std::overflow_error("image_weight_cache_inventory");
    return std::min(stored + metadata, (available - staging) / 2);
}
}  // namespace kadan::image
