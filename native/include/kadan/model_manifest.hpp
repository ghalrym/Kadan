#pragma once
#include "kadan/checkpoint.hpp"
#include <array>

namespace kadan::checkpoint {
enum class LayerKind { linear_attention, full_attention };
struct TextArchitecture {
    std::size_t hidden, vocab, layers, experts, experts_per_token, intermediate, shared_intermediate;
    std::size_t attention_heads, kv_heads, head_dim, key_heads, value_heads, key_dim, value_dim, conv_kernel;
    std::size_t max_context, bos_token, eos_token, rotary_dim;
    double rms_epsilon, rope_theta;
    std::array<std::size_t, 3> rope_sections{};
    std::array<LayerKind, 256> layer_types{};
};
enum class ItemKind { dense, fp8, nvfp4 };
struct ModelItem {
    explicit ModelItem(std::pmr::memory_resource* r) : name(r) {}
    std::pmr::string name; // Tensor name for dense; projection prefix otherwise.
    ItemKind kind{};
    int layer = -1, expert = -1; // -1 is global/non-routed; experts are host backed.
    std::size_t shard = 0, rows = 0, columns = 0;
    std::uint64_t payload_bytes = 0, device_bytes = 0;
};
struct ManifestLimits {
    std::size_t config_bytes = 1024 * 1024, index_bytes = 16 * 1024 * 1024;
    std::size_t shards = 64, tensors = 200000;
    Limits shard{};
};
struct PlacementOptions {
    std::span<const std::size_t> device_capacity, device_headroom, expert_slots, layer_devices;
    std::size_t io_device, host_capacity, staging_bytes;
};
struct DevicePlacement {
    std::uint64_t resident_bytes = 0, expert_cache_bytes = 0, headroom_bytes = 0, total_bytes = 0;
};
class Placement {
public:
    explicit Placement(std::shared_ptr<MemoryBudget> budget);
    Placement(const Placement&) = delete;
    Placement& operator=(const Placement&) = delete;
    Placement(Placement&&) noexcept = default;
    Placement& operator=(Placement&&) = delete;
    std::span<const DevicePlacement> devices() const { return devices_; }
    std::span<const std::size_t> item_devices() const { return item_devices_; }
    std::uint64_t host_bytes = 0, expert_host_bytes = 0, metadata_bytes = 0, staging_bytes = 0, max_transfer_bytes = 0;
private:
    friend class ModelManifest;
    std::shared_ptr<MemoryBudget> budget_;
    std::pmr::vector<DevicePlacement> devices_;
    std::pmr::vector<std::size_t> item_devices_;
};
// Full index/shard coverage, text-decoder role/shape validation and explicit
// placement metadata. No model payload reads or device probes at construction.
// Vision/MTP tensors are structurally validated but explicitly excluded from
// text execution. Not an executable model, admission reservation or scheduler.
class ModelManifest {
public:
    ModelManifest(const char* root, std::shared_ptr<MemoryBudget> budget, ManifestLimits limits = {});
    ~ModelManifest();
    ModelManifest(const ModelManifest&) = delete;
    ModelManifest& operator=(const ModelManifest&) = delete;
    const TextArchitecture& architecture() const;
    std::span<const ModelItem> items() const;
    std::size_t tensor_count() const;
    std::size_t excluded_tensor_count() const;
    std::uint64_t checkpoint_bytes() const;
    // Physical primary tensor dtype/shape; borrowed name follows manifest lifetime.
    TensorInfo primary_tensor(std::size_t item) const;
    Placement place(const PlacementOptions& options) const;
    // Uses pinned validated shard descriptors; payload values are validated by
    // the existing projection loader (and numerical admission by CUDA owner).
    // Supply a separately admitted staging/bank MemoryBudget for model payloads;
    // otherwise the metadata allocator is shared, as in the original Shard API.
    Projection load_projection_rows(std::size_t item, std::size_t first, std::size_t count,
                                    std::size_t payload_budget,
                                    std::shared_ptr<MemoryBudget> payload_memory = {}) const;
    void read_dense(std::size_t item, std::size_t offset, std::span<std::uint8_t> destination) const;
    float read_input_scale(std::size_t item) const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace kadan::checkpoint
