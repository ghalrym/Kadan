#pragma once
#include <algorithm>
#include <climits>
#include <cstddef>
#include <cstring>
#include <optional>
#include <span>

#include "kadan/dense_compute.hpp"
#include "kadan/device_placement.hpp"

namespace kadan {
// Reject the entire batch before any source reads or output writes. Device count
// does not restrict packed batching; each device owns its existing column slice.
inline bool gguf_batch_supported(
    std::span<const DenseCompute::GgufBatchJob> jobs, std::size_t devices
) {
    if (!devices || jobs.empty())
        return false;
    for (const auto& job : jobs) {
        if (!job.identity || !job.source || !job.in || !job.out
            || job.in > std::size_t(INT_MAX) || job.out > std::size_t(INT_MAX)
            || job.input.empty() || job.input.size() % job.in
            || job.input.size() / job.in > DenseCompute::gguf_batch_max_rows
            || job.output.size() != (job.input.size() / job.in) * job.out
            || (job.encoding
                && (!gguf::supported(*job.encoding)
                    || job.in % gguf::block_values(*job.encoding)
                    || job.in / gguf::block_values(*job.encoding)
                           > gguf_projection_staging_bytes
                                 / gguf::block_bytes(*job.encoding)))
            || (!job.encoding && job.in * sizeof(float) > gguf_projection_staging_bytes)
            || job.in % 8)
            return false;
    }
    return true;
}

// Geometry matches an individual source projection on this device. In particular,
// chunking starts at the device boundary, not at a global chunk boundary.
struct GgufBatchPartition {
    std::size_t first, count, chunk_columns;

    std::size_t chunks() const {
        return count ? (count + chunk_columns - 1) / chunk_columns : 0;
    }

    void scatter(
        std::span<const std::byte> local, std::span<float> output, std::size_t out
    ) const {
        if (!count)
            return;
        const auto row_bytes = count * sizeof(float);
        for (std::size_t row = 0; row < local.size() / row_bytes; ++row)
            std::memcpy(
                output.data() + row * out + first,
                local.data() + row * row_bytes,
                row_bytes
            );
    }
};

inline GgufBatchPartition gguf_batch_partition(
    std::size_t in,
    std::size_t out,
    std::optional<gguf::Encoding> encoding,
    std::span<const Bytes> capacities,
    std::size_t device,
    Bytes scratch_limit
) {
    const auto first = weight_boundary(out, capacities, device),
               count = weight_boundary(out, capacities, device + 1) - first;
    if (!count)
        return {first, 0, 0};
    const auto plan =
        encoding
            ? gguf_projection_plan(in, count, *encoding, false, scratch_limit)
            : projection_plan(
                  in,
                  std::min(count, std::size_t(16 * 1024 * 1024) / (in * sizeof(float))),
                  sizeof(float),
                  false,
                  1,
                  scratch_limit
              );
    return {first, count, plan.columns};
}
}  // namespace kadan
