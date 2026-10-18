#include "kadan/image_weight_cache.hpp"

#include <source_location>

namespace {
void check(bool value, std::source_location at = std::source_location::current()) {
    if (!value)
        throw std::runtime_error(
            "image_weight_cache_test:" + std::to_string(at.line())
        );
}
}  // namespace

int main() {
    using namespace kadan;
    using image::image_weight_cache_bytes;
    constexpr Bytes mib = 1024 * 1024;
    check(image_weight_cache_bytes(128 * mib, 32 * mib, 16 * mib) == 24 * mib);
    check(image_weight_cache_bytes(128 * mib, 32 * mib, 128 * mib) == 48 * mib);
    check(image_weight_cache_bytes(32 * mib, 32 * mib, 16 * mib) == 0);
    check(image_weight_cache_bytes(16 * mib, 32 * mib, 16 * mib) == 0);
    check(image_weight_cache_bytes(128 * mib, 32 * mib, 0) == 0);
    bool overflow = false;
    try {
        (void)image_weight_cache_bytes(128 * mib, 0, std::numeric_limits<Bytes>::max());
    } catch (const std::overflow_error&) {
        overflow = true;
    }
    check(overflow);
    check(image::image_device_weight_bytes(16 * mib, 4) == 26 * mib);
    overflow = false;
    try {
        (void)image::image_device_weight_bytes(std::numeric_limits<Bytes>::max(), 1);
    } catch (const std::overflow_error&) {
        overflow = true;
    }
    check(overflow);
    overflow = false;
    try {
        (void)image::image_device_weight_bytes(1, std::numeric_limits<Bytes>::max());
    } catch (const std::overflow_error&) {
        overflow = true;
    }
    check(overflow);
}
