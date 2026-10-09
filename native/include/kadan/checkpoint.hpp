#pragma once

#include "kadan/quantization.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <functional>
#include <memory_resource>
#include <mutex>
#include <string_view>
#include <vector>

namespace kadan::checkpoint {
// Allocator accounting for a caller-admitted RAM envelope, not machine capacity.
// Includes header/parser/tensor tables and owned projection buffers allocated
// through this resource. Fixed object/fd/stack overhead is outside the counter.
class MemoryBudget final : public std::pmr::memory_resource {
public:
    explicit MemoryBudget(std::size_t bytes) : limit_(bytes) {}
    std::size_t used() const;
    std::size_t limit() const { return limit_; }
private:
    void* do_allocate(std::size_t bytes, std::size_t alignment) override;
    void do_deallocate(void* pointer, std::size_t bytes, std::size_t alignment) override;
    bool do_is_equal(const std::pmr::memory_resource& other) const noexcept override { return this == &other; }
    const std::size_t limit_;
    std::size_t used_ = 0;
    mutable std::mutex mutex_;
};

struct Limits {
    std::size_t header_bytes = 16 * 1024 * 1024;
    std::size_t tensors = 65536;
    std::size_t metadata_value_bytes = 512; // Names and schema strings stay capped at 512.
};

enum class Dtype { u8, fp8, fp32, bf16, fp16, i8 };
// Borrowed name remains valid for the Shard lifetime; dimensions/byte count copied.
struct TensorInfo {
    std::string_view name;
    Dtype dtype;
    std::array<std::uint64_t, 8> shape;
    std::size_t rank;
    std::uint64_t bytes;
};

class Projection {
public:
    Projection(const Projection&) = delete;
    Projection& operator=(const Projection&) = delete;
    Projection(Projection&&) noexcept;
    Projection& operator=(Projection&&) = delete;
    quantization::Matrix view() const &;
    quantization::Matrix view() const && = delete;
private:
    friend class Shard;
    explicit Projection(std::shared_ptr<MemoryBudget> budget);
    // Declared first, so every buffer is freed before its allocator is destroyed.
    std::shared_ptr<MemoryBudget> budget_;
    std::pmr::vector<std::uint8_t> weights_, blocks_;
    std::pmr::vector<float> multipliers_;
    quantization::Encoding encoding_{};
    std::size_t rows_ = 0, columns_ = 0;
};

// Linux POSIX reader. The root is trusted; shard_name must be one local basename.
// Keeps an O_NOFOLLOW regular-file descriptor; never maps or reads all payloads.
using TensorReader = std::function<void(std::string_view, std::size_t, std::span<std::uint8_t>)>;
class Shard {
public:
    Shard(const char* root, std::string_view shard_name,
          std::shared_ptr<MemoryBudget> budget, Limits limits = {});
    ~Shard();
    Shard(const Shard&) = delete;
    Shard& operator=(const Shard&) = delete;
    std::size_t tensor_count() const;
    TensorInfo tensor(std::string_view name) const;
    std::size_t tensor_index(std::string_view name) const;
    void check_unchanged() const;
    // Caller owns/admitted destination; no hidden payload allocation.
    void read_tensor(std::string_view name, std::size_t offset, std::span<std::uint8_t> destination) const;
    // Explicit ModelOpt contract. All companion tensors must be in this shard.
    // Reads only selected rows plus their block/row scales and scalar multiplier.
    // payload_budget caps final owned tensor bytes, separately from allocator quota.
    // Optional payload_memory separates staging/host-bank admission from metadata.
    Projection load_modelopt_rows(std::string_view prefix, std::size_t first,
                                  std::size_t count, std::size_t payload_budget,
                                  std::shared_ptr<MemoryBudget> payload_memory = {}, const TensorReader& reader = {}) const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace kadan::checkpoint
