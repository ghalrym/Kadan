#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <span>
#include <stdexcept>

namespace kadan::muse_spark::batch {
inline constexpr std::size_t max_rows = 256;
inline constexpr std::size_t scratch_limit = 128u << 20;
inline constexpr std::size_t expert_count = 256;
inline constexpr std::size_t selected_experts = 8;

// The caller supplies the peak simultaneous allocation, including gathered
// expert rows and outputs, rather than the sum of allocations over a layer.
inline std::size_t
plan(std::size_t remaining, std::size_t scratch_per_row, std::size_t fixed_bytes) {
    if (!scratch_per_row)
        throw std::invalid_argument("muse_spark_batch_scratch");
    if (!remaining)
        return 0;
    if (fixed_bytes > scratch_limit || scratch_per_row > scratch_limit - fixed_bytes)
        throw std::length_error("muse_spark_batch_scratch");
    return std::min(
        {remaining, max_rows, (scratch_limit - fixed_bytes) / scratch_per_row}
    );
}

template <class T, std::size_t Extent>
std::span<T> row(std::span<T, Extent> values, std::size_t width, std::size_t index) {
    if (!width || values.size() % width || index >= values.size() / width)
        throw std::invalid_argument("muse_spark_batch_row");
    return values.subspan(index * width, width);
}

inline std::size_t position(std::size_t base, std::size_t offset) {
    if (offset > std::numeric_limits<std::size_t>::max() - base)
        throw std::overflow_error("muse_spark_batch_position");
    return base + offset;
}

struct Ring {
    std::size_t slot;
    std::size_t first;  // Absolute first visible position, not a ring index.
    std::size_t count;
};

// Commit one row before reading attention. Committing all rows of a chunk first
// would overwrite still-visible keys when a sliding ring wraps.
inline Ring ring(std::size_t absolute, std::size_t capacity) {
    if (!capacity)
        throw std::invalid_argument("muse_spark_batch_capacity");
    const auto end = position(absolute, 1);
    const auto count = std::min(end, capacity);
    return {absolute % capacity, end - count, count};
}

struct Route {
    std::array<std::size_t, selected_experts> experts;
    std::array<float, selected_experts> weights;
};

inline Route route(std::span<const float> probabilities, std::span<const float> bias) {
    if (probabilities.size() != expert_count || bias.size() != expert_count)
        throw std::invalid_argument("muse_spark_batch_router_shape");
    for (std::size_t i = 0; i < expert_count; ++i)
        if (!std::isfinite(probabilities[i]) || !std::isfinite(bias[i])
            || !std::isfinite(probabilities[i] + bias[i]))
            throw std::invalid_argument("muse_spark_router_nonfinite");
    std::array<std::size_t, expert_count> order;
    std::iota(order.begin(), order.end(), 0);
    std::partial_sort(
        order.begin(),
        order.begin() + selected_experts,
        order.end(),
        [&](auto a, auto b) {
            const auto sa = probabilities[a] + bias[a];
            const auto sb = probabilities[b] + bias[b];
            return sa == sb ? a < b : sa > sb;
        }
    );
    // Preserve the serial path's score-ranked FP32 sum before sorting experts
    // into the reference's accumulation order. Bias only selects the experts.
    float sum = 1e-15f;
    for (std::size_t i = 0; i < selected_experts; ++i)
        sum += probabilities[order[i]];
    if (!std::isfinite(sum) || sum <= 0)
        throw std::invalid_argument("muse_spark_router_nonfinite");
    std::sort(order.begin(), order.begin() + selected_experts);
    Route result{};
    for (std::size_t i = 0; i < selected_experts; ++i) {
        result.experts[i] = order[i];
        result.weights[i] = probabilities[order[i]] / sum;
    }
    return result;
}

inline std::size_t route_index(const Route& route, std::size_t expert) {
    const auto at =
        std::lower_bound(route.experts.begin(), route.experts.end(), expert);
    return at != route.experts.end() && *at == expert
               ? static_cast<std::size_t>(at - route.experts.begin())
               : selected_experts;
}
}  // namespace kadan::muse_spark::batch
