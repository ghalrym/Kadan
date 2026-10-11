#pragma once
#include <compare>
#include <cstdint>

namespace kadan {
// Identity survives reopening a verified immutable shard; never a host pointer.
struct WeightIdentity {
    std::uint64_t device = 0;
    std::uint64_t inode = 0;
    std::uint64_t file_size = 0;
    std::uint64_t modified_seconds = 0;
    std::uint64_t modified_nanoseconds = 0;
    std::uint64_t changed_seconds = 0;
    std::uint64_t changed_nanoseconds = 0;
    std::uint64_t payload_begin = 0;
    std::uint64_t payload_end = 0;

    auto operator<=>(const WeightIdentity&) const = default;
};
}  // namespace kadan
