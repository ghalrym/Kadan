#pragma once
#include <array>
#include <cstdint>
namespace kadan {
// Device/inode/size/timestamps and the tensor's absolute payload interval.
// Identity survives reopening a verified immutable shard; never a host pointer.
using WeightIdentity=std::array<std::uint64_t,9>;
}
