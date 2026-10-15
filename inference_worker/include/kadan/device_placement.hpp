#pragma once
#include <algorithm>
#include <charconv>
#include <optional>
#include <span>
#include <string_view>

#include "kadan/gguf_quantization.hpp"
#include "kadan/image_convolution.hpp"
#include "kadan/resources.hpp"

namespace kadan {
// Budgets are indexed by physical device, independent of selection order.
inline Footprint device_budgets(
    std::span<const int> devices,
    std::string_view scalar,
    std::string_view indexed,
    Bytes minimum
) {
    Footprint result(3);
    std::vector<bool> seen(3);
    auto parse = [&](std::string_view value) {
        Bytes bytes = 0;
        auto p = std::from_chars(value.data(), value.data() + value.size(), bytes);
        if (value.empty() || p.ec != std::errc{} || p.ptr != value.data() + value.size()
            || bytes < minimum || bytes > 24ULL * 1024 * 1024 * 1024)
            throw std::invalid_argument("device_budget");
        return bytes;
    };
    if (indexed.empty()) {
        auto bytes = parse(scalar);
        for (int d : devices)
            result.at(d + 1) = bytes;
        return result;
    }
    while (!indexed.empty()) {
        const auto comma = indexed.find(',');
        const auto item = indexed.substr(0, comma);
        if (item.size() < 3 || item[1] != ':' || (item[0] != '0' && item[0] != '1'))
            throw std::invalid_argument("device_budget_mapping");
        const int d = item[0] - '0';
        if (seen[d + 1]
            || std::find(devices.begin(), devices.end(), d) == devices.end())
            throw std::invalid_argument("device_budget_mapping");
        seen[d + 1] = true;
        result[d + 1] = parse(item.substr(2));
        if (comma == std::string_view::npos)
            break;
        indexed.remove_prefix(comma + 1);
        if (indexed.empty())
            throw std::invalid_argument("device_budget_mapping");
    }
    for (int d : devices)
        if (!seen.at(d + 1))
            throw std::invalid_argument("device_budget_mapping");
    return result;
}

// Retry only a confirmed lack of memory; other device failures must propagate.
// The owner keeps its admission for the entire wait. Pause is bounded by caller.
template <class Stop, class Attempt, class Pause>
void wait_for_device_allocation(Stop stop, Attempt attempt, Pause pause) {
    for (;;) {
        if (stop())
            throw std::runtime_error("h3_cuda_cancelled");
        if (attempt())
            return;
        pause();
    }
}

// Include alignment slack for all CUDA workspace slices. Geometry depends on
// device admission, not token count, so retained projection keys remain stable.
inline constexpr Bytes projection_status_bytes = 256;

struct ProjectionPlan {
    std::size_t rows, columns;
    Bytes bytes;
};

inline ProjectionPlan projection_plan(
    std::size_t in,
    std::size_t out,
    std::size_t width,
    bool precise,
    std::size_t alignment,
    Bytes limit
) {
    if (!in || in > 2147483647ULL || !out || out > 2147483647ULL
        || (width != 1 && width != 2 && width != 4)
        || (alignment != 1 && alignment != 4) || out % alignment)
        throw std::invalid_argument("projection_layout");
    for (std::size_t tile = 256; tile; tile /= 2) {
        const Bytes fixed = 8 * 1024 * 1024 + 4096 + projection_status_bytes
                            + tile * in * (9 + (precise ? 8 : 0)) + tile * 4;
        const Bytes per_column =
            in * (width + (precise ? 8 : 0)) + 8 + tile * (8 + (precise ? 8 : 0));
        if (fixed >= limit)
            continue;
        const auto columns =
            std::min<Bytes>(out, (limit - fixed) / per_column) / alignment * alignment;
        if (columns)
            return {tile, columns, fixed + columns * per_column};
    }
    throw std::invalid_argument("projection_scratch_limit");
}

// Packed source staging and its separate decode status stay bounded even when
// compressed tiles are retained. The existing projection accounts for decoded
// F32 weights, optional F64 conversion, GEMM workspace and input/output slices.
inline constexpr Bytes gguf_projection_staging_bytes = 16 * 1024 * 1024;
inline constexpr Bytes gguf_projection_reserve_bytes =
    gguf_projection_staging_bytes + projection_status_bytes;

inline std::size_t gguf_projection_row_bytes(std::size_t in, gguf::Encoding encoding) {
    if (encoding != gguf::Encoding::q3_k && encoding != gguf::Encoding::q4_k
        && encoding != gguf::Encoding::q8_0)
        throw std::invalid_argument("gguf_projection_encoding");
    const auto values = gguf::block_values(encoding),
               bytes = gguf::block_bytes(encoding);
    if (!in || in > 2147483647ULL || in % values
        || in / values > gguf_projection_staging_bytes / bytes)
        throw std::invalid_argument("gguf_projection_layout");
    return in / values * bytes;
}

inline ProjectionPlan gguf_projection_plan(
    std::size_t in, std::size_t out, gguf::Encoding encoding, bool precise, Bytes limit
) {
    const auto row_bytes = gguf_projection_row_bytes(in, encoding);
    if (!out || out > 2147483647ULL || limit <= gguf_projection_reserve_bytes)
        throw std::invalid_argument("gguf_projection_layout");
    auto plan = projection_plan(
        in,
        std::min<Bytes>(out, gguf_projection_staging_bytes / row_bytes),
        sizeof(float),
        precise,
        1,
        limit - gguf_projection_reserve_bytes
    );
    plan.bytes += gguf_projection_reserve_bytes;
    return plan;
}

struct VaeConvolutionPlan {
    ProjectionPlan projection;
    std::size_t points;
    Bytes patch_bytes;
    Bytes bytes;
};

inline std::optional<VaeConvolutionPlan> image_vae_convolution_plan(
    image::VaeConvolution shape,
    std::size_t outputs,
    Bytes limit,
    std::size_t maximum_points = 4096
) {
    shape.validate();
    if (!outputs || outputs > shape.outputs)
        throw std::invalid_argument("image_vae_conv_shape");
    if (maximum_points < 256 || maximum_points > 4096
        || (maximum_points & (maximum_points - 1)))
        throw std::invalid_argument("image_vae_conv_group");
    if ((shape.kernel != 1 && shape.kernel != 3) || shape.columns() > 2147483647ULL
        || shape.outputs > 2147483647ULL || shape.points() > 2147483647ULL / 4)
        return {};
    const auto host_columns = (16 * 1024 * 1024) / (shape.columns() * 4);
    if (!host_columns)
        return {};
    auto aligned = [](Bytes n) {
        return (n + 255) / 256 * 256;
    };
    auto patch = [&](std::size_t points) {
        // Worst scanline crossing and halo, including non-divisible widths.
        const auto band = std::min(
            shape.height,
            1 + (points - 2 + shape.width) / shape.width + shape.kernel - 1
        );
        return aligned(shape.channels * band * shape.width * 4);
    };
    const auto patch_bytes = patch(256);
    if (patch_bytes >= limit)
        return {};
    try {
        // Keep the original projection geometry; added staging cannot change
        // GEMM dimensions or retained weight partitions.
        const auto projection = projection_plan(
            shape.columns(),
            std::min(outputs, host_columns),
            4,
            false,
            1,
            limit - patch_bytes
        );
        for (auto points = maximum_points; points > 256; points /= 2) {
            if (points > shape.points())
                continue;
            const auto raw = patch(points);
            const auto extra = aligned(points * shape.columns() * 4)
                               - aligned(projection.rows * shape.columns() * 4)
                               + aligned(points * projection.columns * 4)
                               - aligned(projection.rows * projection.columns * 4);
            const auto bytes = projection.bytes + raw + extra;
            if (bytes <= limit)
                return VaeConvolutionPlan{projection, points, raw, bytes};
        }
        // A group that does not fit retains the admitted 256-point GPU path,
        // including its existing streamed-input behavior under tight budgets.
        return VaeConvolutionPlan{
            projection, 256, patch_bytes, projection.bytes + patch_bytes
        };
    } catch (const std::invalid_argument& error) {
        if (std::string_view(error.what()) == "projection_scratch_limit")
            return {};
        throw;
    }
}

// Keep a complete projection input only when the existing workspace can hold it.
// Count the original tile once; division keeps the fit check overflow-safe.
inline std::size_t projection_input_rows(
    std::size_t rows, std::size_t in, const ProjectionPlan& plan, Bytes limit
) {
    if (!rows || !in || in > 2147483647ULL || !plan.rows || plan.bytes > limit)
        throw std::invalid_argument("projection_input_layout");
    if (rows <= plan.rows)
        return plan.rows;
    return rows - plan.rows <= (limit - plan.bytes) / (in * 4) ? rows : plan.rows;
}

inline std::size_t projection_columns(
    std::size_t in,
    std::size_t out,
    std::size_t width,
    bool precise,
    std::size_t alignment
) {
    return projection_plan(
               in, out, width, precise, alignment, 2ULL * 1024 * 1024 * 1024
    )
        .columns;
}

// Both uncached BF16 weights and both FP32 results fit at once. Weight staging
// remains sequential and bounded to 16MiB, independent of the retained bank.
inline ProjectionPlan
image_silu_pair_plan(std::size_t in, std::size_t out, Bytes limit) {
    if (!in || in > 2147483647ULL || !out || out > 2147483647ULL)
        throw std::invalid_argument("image_pair_layout");
    const auto staging_columns = (16 * 1024 * 1024) / (in * 2);
    for (std::size_t rows = 256; rows; rows /= 2) {
        // Eight aligned slices: F32/BF16 input, two weights, two outputs,
        // BLAS workspace and finite status. Each can waste at most 255 bytes.
        const Bytes fixed = 8 * 1024 * 1024 + 8 * 255 + 4 + rows * in * 6;
        const Bytes per_column = in * 4 + rows * 8;
        if (fixed >= limit)
            continue;
        const auto columns = std::min<Bytes>(
            std::min(out, staging_columns), (limit - fixed) / per_column
        );
        if (columns)
            return {rows, columns, fixed + columns * per_column};
    }
    throw std::invalid_argument("image_pair_scratch_limit");
}

inline std::size_t image_silu_pair_input_rows(
    std::size_t rows, std::size_t in, const ProjectionPlan& plan, Bytes limit
) {
    if (!rows || !in || in > 2147483647ULL || !plan.rows || plan.bytes > limit)
        throw std::invalid_argument("image_pair_input_layout");
    if (rows <= plan.rows)
        return plan.rows;
    return rows - plan.rows <= (limit - plan.bytes) / (in * 6) ? rows : plan.rows;
}

struct VaeNormalizationPlan {
    std::size_t points;
    Bytes bytes;
};

// Two planar tiles, gamma and finite status. Spatial tiling bounds memory and
// cancellation latency without changing the ordered channel reduction.
inline std::optional<VaeNormalizationPlan>
image_vae_normalization_plan(std::size_t channels, std::size_t points, Bytes limit) {
    if (!channels || !points)
        throw std::invalid_argument("image_vae_norm_shape");
    // Keep host/device pitches within CUDA's signed 31-bit limit.
    if (channels > 2147483647ULL || points > 2147483647ULL / 4)
        return {};
    auto aligned = [](Bytes bytes) {
        return (bytes + 255) / 256 * 256;
    };
    for (auto count = std::min<std::size_t>(points, 65536); count; count /= 2) {
        const auto bytes = 2 * aligned(channels * count * 4) + aligned(channels * 4)
                           + projection_status_bytes;
        if (bytes <= limit)
            return VaeNormalizationPlan{count, bytes};
    }
    return {};
}

struct AttentionPlan {
    std::size_t queries, keys;
    bool full_heads;
    Bytes bytes;
};

inline AttentionPlan attention_plan(
    std::size_t queries,
    std::size_t keys,
    std::size_t dim,
    std::size_t width,
    Bytes limit
) {
    if (!queries || queries > 2147483647ULL || !keys || keys > 2147483647ULL || !dim
        || dim > 2048 || (width != 4 && width != 8))
        throw std::invalid_argument("attention_layout");
    auto q = std::min<std::size_t>(128, queries), k = std::min<std::size_t>(4096, keys);
    auto bytes = [&](bool full) {
        return 8ULL * 1024 * 1024 + 2048
               + width
                     * ((full ? queries : q) * dim + 2 * (full ? keys : k) * dim
                        + q * dim + q * k + 2 * q);
    };
    if (bytes(true) <= limit)
        return {q, k, true, bytes(true)};
    for (;;) {
        if (bytes(false) <= limit)
            return {q, k, false, bytes(false)};
        if (k > 1)
            k = (k + 1) / 2;
        else if (q > 1)
            q = (q + 1) / 2;
        else
            throw std::invalid_argument("attention_scratch_limit");
    }
}

// Complete, monotone output-channel partitions; callers pass groups of four for
// integer projections. Long double avoids overflowing units*capacity.
inline std::size_t weight_boundary(
    std::size_t units, std::span<const Bytes> capacities, std::size_t index
) {
    if (index >= capacities.size())
        return units;
    long double total = 0, prefix = 0;
    for (std::size_t i = 0; i < capacities.size(); ++i) {
        total += capacities[i];
        if (i < index)
            prefix += capacities[i];
    }
    return total ? std::min(
                       units,
                       std::size_t(static_cast<long double>(units) * prefix / total)
                   )
                 : units * index / capacities.size();
}
}  // namespace kadan
