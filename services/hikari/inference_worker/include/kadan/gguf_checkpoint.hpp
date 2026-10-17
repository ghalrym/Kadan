#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "kadan/weight_identity.hpp"

namespace kadan::checkpoint {
struct GgufTensor {
    std::string name;
    // GGML order: columns, rows, experts; omitted dimensions are one.
    std::array<std::uint64_t, 3> shape{1, 1, 1};
    std::size_t rank;
    std::uint32_t type;
    std::uint64_t bytes;
};

// Independently readable first split metadata, for preparing assets before the
// remaining checkpoint files exist. Never reads tensor payloads.
class GgufMetadata {
public:
    GgufMetadata(const std::string& path, const std::atomic_bool& cancel);
    ~GgufMetadata();
    GgufMetadata(const GgufMetadata&) = delete;
    GgufMetadata& operator=(const GgufMetadata&) = delete;
    std::size_t tensor_count() const;
    std::string string(std::string_view key) const;
    std::uint64_t integer(std::string_view key) const;
    double number(std::string_view key) const;
    bool boolean(std::string_view key) const;
    std::vector<bool> booleans(std::string_view key) const;
    std::vector<std::string> strings(std::string_view key) const;
    std::vector<std::int64_t> integers(std::string_view key) const;
    std::vector<double> numbers(std::string_view key) const;
    bool has(std::string_view key) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// GGUF v3 checkpoint: one unsplit file or a numbered split set. Construction
// reads bounded metadata only. Payload access uses immutable file descriptors
// and bounded packed scratch. Supports F32 and every gguf::Encoding; no ggml
// graph or inference runtime is created.
class GgufCheckpoint {
public:
    GgufCheckpoint(const std::string& root, const std::atomic_bool& cancel);
    ~GgufCheckpoint();
    GgufCheckpoint(const GgufCheckpoint&) = delete;
    GgufCheckpoint& operator=(const GgufCheckpoint&) = delete;
    std::size_t tensor_count() const;
    const GgufTensor& tensor(std::string_view name) const;
    const GgufTensor& tensor_at(std::size_t index) const;
    WeightIdentity identity(std::string_view name, std::size_t expert = 0) const;
    void check_unchanged() const;
    // Output contains whole rows; caller admits its allocation. Expert is zero
    // for ordinary matrices. On cancellation or error, discard partial output.
    void read_rows(
        std::string_view name,
        std::size_t expert,
        std::size_t first_row,
        std::span<float> output,
        const std::atomic_bool& cancel
    ) const;
    // Copies encoded whole rows without decoding or finite validation. Caller
    // admits output storage and validates decoded values. Concurrent calls use
    // independent caller buffers and pread; discard partial output on failure.
    void read_packed_rows(
        std::string_view name,
        std::size_t expert,
        std::size_t first_row,
        std::span<std::byte> output,
        const std::atomic_bool& cancel
    ) const;
    // Reads a whole F32 tensor of any rank; caller admits the output.
    void read_vector(
        std::string_view name, std::span<float> output, const std::atomic_bool& cancel
    ) const;
    std::string string(std::string_view key) const;
    std::uint64_t integer(std::string_view key) const;
    double number(std::string_view key) const;
    bool boolean(std::string_view key) const;
    std::vector<bool> booleans(std::string_view key) const;
    std::vector<std::int64_t> integers(std::string_view key) const;
    std::vector<double> numbers(std::string_view key) const;
    bool has(std::string_view key) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}  // namespace kadan::checkpoint
