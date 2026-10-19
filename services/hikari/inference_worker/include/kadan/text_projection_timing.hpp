#pragma once
#include <cstddef>
#include <cstdint>

namespace kadan::text {
// Fixed-size sampled phase summary; no tensor names, shapes or input data.
struct ProjectionTiming {
    std::uint64_t packed_calls = 0, host_calls = 0, large_row_calls = 0,
                  unsupported_calls = 0, rows = 0, batch_calls = 0,
                  batch_rejections = 0;
    double large_row_seconds = 0, unsupported_seconds = 0;

    enum class Path {
        packed,
        host,
        large_rows,
        batch_unsupported
    };

    void add(std::size_t count, Path path, double source_seconds = 0) noexcept {
        rows += count;
        switch (path) {
        case Path::host:
            ++host_calls;
            break;
        case Path::packed:
            ++packed_calls;
            break;
        case Path::large_rows:
            ++large_row_calls;
            large_row_seconds += source_seconds;
            break;
        case Path::batch_unsupported:
            ++unsupported_calls;
            unsupported_seconds += source_seconds;
            break;
        }
    }
};
}  // namespace kadan::text
