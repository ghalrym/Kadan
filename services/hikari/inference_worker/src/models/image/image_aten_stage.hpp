#pragma once
#include <ATen/ATen.h>
#include <ATen/Context.h>
#include <ATen/cuda/CUDAContextLight.h>
#include <algorithm>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAGuard.h>
#include <chrono>
#include <cstdio>
#include <map>
#include <optional>

#include "kadan/checkpoint.hpp"
#include "kadan/image_aten.hpp"
#include "kadan/image_aten_plan.hpp"
#include "kadan/image_weight_cache.hpp"

namespace kadan::image::detail {
using Tensor = at::Tensor;
using Clock = std::chrono::steady_clock;
constexpr Bytes mib = 1024 * 1024;
constexpr const char* text_prefix = "model.language_model.";

inline void need(bool value, const char* error) {
    if (!value)
        throw std::runtime_error(error);
}

inline void stop(const std::atomic_bool& cancel) {
    need(!cancel.load(), "image_cancelled");
}

struct Weight {
    checkpoint::Shard* shard;
    checkpoint::TensorInfo info;
};

// Shard owns O_NOFOLLOW descriptors and validates exact byte ranges. Preserve the
// existing checkpoint reader; neither Python nor a new deserializer is involved.
struct Inventory {
    std::vector<std::unique_ptr<checkpoint::Shard>> shards;
    std::map<std::string, Weight, std::less<>> weights;
    Bytes bytes = 0, largest = 0;
    AtenImageStage stage;
    at::ScalarType dtype;

    explicit Inventory(
        const std::string& root,
        std::span<const std::string> files,
        AtenImageStage component,
        const std::atomic_bool& cancel
    )
        : stage(component), dtype(
                                component == AtenImageStage::vae
                                        || component == AtenImageStage::vae_encoder
                                    ? at::kFloat
                                    : at::kBFloat16
                            ) {
        const std::string_view prefix = component == AtenImageStage::text ? text_prefix
                                        : component == AtenImageStage::vision
                                            ? "model.visual."
                                            : "";
        for (const auto& file : files) {
            stop(cancel);
            auto shard = std::make_unique<checkpoint::Shard>(
                root.c_str(), file, std::make_shared<checkpoint::MemoryBudget>(16 * mib)
            );
            for (std::size_t i = 0; i < shard->tensor_count(); ++i) {
                auto info = shard->tensor_at(i);
                if (!info.name.starts_with(prefix))
                    continue;
                if (stage == AtenImageStage::vae && !info.name.starts_with("decoder.")
                    && !info.name.starts_with("post_quant_conv."))
                    continue;
                if (stage == AtenImageStage::vae_encoder
                    && ((!info.name.starts_with("encoder.")
                         && !info.name.starts_with("quant_conv."))
                        || info.name.find(".time_conv.") != std::string_view::npos))
                    continue;
                need(
                    info.dtype == checkpoint::Dtype::bf16
                        || info.dtype == checkpoint::Dtype::fp32,
                    "image_aten_weight_dtype"
                );
                std::string name(info.name.substr(prefix.size()));
                // Text conditioning stops before the final language-model norm.
                if (stage == AtenImageStage::text && name != "embed_tokens.weight"
                    && !name.starts_with("layers."))
                    continue;
                need(
                    weights.emplace(name, Weight{shard.get(), info}).second,
                    "image_aten_duplicate_tensor"
                );
                Bytes elements = 1;
                for (std::size_t d = 0; d < info.rank; ++d)
                    elements *= info.shape[d];
                bytes += elements * (dtype == at::kFloat ? 4 : 2);
                largest = std::max(
                    {largest, info.bytes, elements * (dtype == at::kFloat ? 4 : 2)}
                );
            }
            shards.push_back(std::move(shard));
        }
        need(!weights.empty(), "image_aten_empty_inventory");
    }

    void
    shape(const std::string& name, std::initializer_list<std::uint64_t> shape) const {
        const auto& info = weights.at(name).info;
        need(
            info.rank == shape.size()
                && std::equal(shape.begin(), shape.end(), info.shape.begin()),
            "image_aten_weight_shape"
        );
    }

    void validate_text() const {
        shape("embed_tokens.weight", {151936, 4096});
        for (int i = 0; i < 36; ++i) {
            auto p = "layers." + std::to_string(i) + ".";
            for (auto n : {"input_layernorm.weight", "post_attention_layernorm.weight"})
                shape(p + n, {4096});
            for (auto n : {"q_norm.weight", "k_norm.weight"})
                shape(p + "self_attn." + n, {128});
            for (auto n : {"q_proj.weight", "o_proj.weight"})
                shape(p + "self_attn." + n, {4096, 4096});
            for (auto n : {"k_proj.weight", "v_proj.weight"})
                shape(p + "self_attn." + n, {1024, 4096});
            for (auto n : {"gate_proj.weight", "up_proj.weight"})
                shape(p + "mlp." + n, {12288, 4096});
            shape(p + "mlp.down_proj.weight", {4096, 12288});
        }
    }

    void validate_denoiser() const {
        shape("img_in.weight", {4096, 64});
        shape("txt_in.text_norm.weight", {4096});
        for (auto n :
             {"txt_in.in_layer.weight",
              "txt_in.out_layer.weight",
              "time_text_embed.timestep_embedder.linear_2.weight",
              "norm_out.linear.weight"})
            shape(n, {4096, 4096});
        shape("time_text_embed.timestep_embedder.linear_1.weight", {4096, 256});
        shape("modulation.1.weight", {16384, 4096});
        shape("proj_out.weight", {64, 4096});
        for (int i = 0; i < 32; ++i) {
            auto p = "transformer_blocks." + std::to_string(i) + ".";
            for (auto n :
                 {"to_q.weight", "to_k.weight", "to_v.weight", "to_out.0.weight"})
                shape(p + "attn." + n, {4096, 4096});
            for (auto n : {"norm_q.weight", "norm_k.weight"})
                shape(p + "attn." + n, {128});
            for (auto n : {"proj.weight", "gate_layer.weight"})
                shape(p + "img_mlp." + n, {12288, 4096});
            shape(p + "img_mlp.out.weight", {4096, 12288});
        }
    }

    void validate_vae() const {
        auto conv = [&](const std::string& p,
                        std::uint64_t in,
                        std::uint64_t out,
                        std::uint64_t kernel) {
            shape(p + ".weight", {out, in, kernel, kernel});
            shape(p + ".bias", {out});
        };
        auto residual = [&](const std::string& p, std::uint64_t in, std::uint64_t out) {
            shape(p + ".norm1.gamma", {in, 1, 1, 1});
            shape(p + ".norm2.gamma", {out, 1, 1, 1});
            conv(p + ".conv1", in, out, 3);
            conv(p + ".conv2", out, out, 3);
            if (in != out)
                conv(p + ".conv_shortcut", in, out, 1);
        };
        if (stage == AtenImageStage::vae_encoder) {
            conv("encoder.conv_in", 4, 96, 3);
            const std::array<std::uint64_t, 6> dims{1, 1, 2, 4, 8, 8};
            for (int i = 0; i < 5; ++i) {
                const auto p = "encoder.down_blocks." + std::to_string(i);
                for (int j = 0; j < 2; ++j)
                    residual(
                        p + ".resnets." + std::to_string(j),
                        96 * dims[j ? i + 1 : i],
                        96 * dims[i + 1]
                    );
                if (i < 4)
                    conv(
                        p + ".downsampler.resample.1",
                        96 * dims[i + 1],
                        96 * dims[i + 1],
                        3
                    );
            }
            residual("encoder.mid_block.resnets.0", 768, 768);
            residual("encoder.mid_block.resnets.1", 768, 768);
            shape("encoder.mid_block.attentions.0.norm.gamma", {768, 1, 1});
            conv("encoder.mid_block.attentions.0.to_qkv", 768, 2304, 1);
            conv("encoder.mid_block.attentions.0.proj", 768, 768, 1);
            shape("encoder.norm_out.gamma", {768, 1, 1, 1});
            conv("encoder.conv_out", 768, 128, 3);
            conv("quant_conv", 128, 128, 1);
            return;
        }
        conv("post_quant_conv", 64, 64, 1);
        conv("decoder.conv_in", 64, 1152, 3);
        residual("decoder.mid_block.resnets.0", 1152, 1152);
        residual("decoder.mid_block.resnets.1", 1152, 1152);
        shape("decoder.mid_block.attentions.0.norm.gamma", {1152, 1, 1});
        conv("decoder.mid_block.attentions.0.to_qkv", 1152, 3456, 1);
        conv("decoder.mid_block.attentions.0.proj", 1152, 1152, 1);
        const std::array<std::uint64_t, 6> dims{8, 8, 8, 4, 2, 1};
        for (int i = 0; i < 5; ++i) {
            auto p = "decoder.up_blocks." + std::to_string(i);
            for (int j = 0; j < 3; ++j)
                residual(
                    p + ".resnets." + std::to_string(j),
                    144 * dims[j ? i + 1 : i],
                    144 * dims[i + 1]
                );
            if (i < 4)
                conv(
                    p + ".upsampler.resample.1", 144 * dims[i + 1], 144 * dims[i + 1], 3
                );
        }
        shape("decoder.norm_out.gamma", {144, 1, 1, 1});
        conv("decoder.conv_out", 144, 4, 3);
    }
};

inline Snapshot stage_snapshot(Resources& resources, const std::vector<int>& devices) {
    at::globalContext().lazyInitDevice(c10::DeviceType::CUDA);
    auto snapshot = resources.snapshot();
    for (int device : devices) {
        need(
            device >= 0 && (std::size_t(device) + 1) < snapshot.capacity.size(),
            "image_aten_device"
        );
        c10::cuda::CUDAGuard guard(device);
        std::size_t free = 0, total = 0;
        C10_CUDA_CHECK(cudaMemGetInfo(&free, &total));
        auto ledger_free = snapshot.capacity[device + 1] - snapshot.used[device + 1];
        snapshot.capacity[device + 1] =
            snapshot.used[device + 1] + std::min<Bytes>(free, ledger_free);
    }
    return snapshot;
}

// Component weights remain owned by the model while idle. Workspace reservations
// remain request-local, and every eviction confirms device cleanup before release.
class DeviceWeightCache {
public:
    using Bank = std::map<std::string, Tensor, std::less<>>;

    struct Entry {
        Inventory* inventory;
        AtenImageStage stage;
        int device;
        Handle handle;
        Bytes bytes;
        Bytes allocated = 0;
        bool ready = false;
        bool pinned = false;
        std::uint64_t used = 0;
        Bank bank;
    };

private:
    using Key = std::pair<Inventory*, int>;
    Resources& resources_;
    std::map<Key, std::unique_ptr<Entry>> entries_;
    std::map<Key, double> load_seconds_;
    std::uint64_t clock_ = 0;

    void report(const char* event, const Entry& entry) const {
        std::fprintf(
            stderr,
            "KADAN_ATEN_IMAGE_CACHE {\"location\":\"device\",\"event\":\"%s\","
            "\"device\":%d,\"stage\":%d,\"reserved_bytes\":%llu,"
            "\"allocated_bytes\":%llu}\n",
            event,
            entry.device,
            int(entry.stage),
            static_cast<unsigned long long>(entry.bytes),
            static_cast<unsigned long long>(entry.allocated)
        );
        std::fflush(stderr);
    }

public:
    explicit DeviceWeightCache(Resources& resources) : resources_(resources) {
    }

    ~DeviceWeightCache() noexcept {
        try {
            clear();
        } catch (...) { /* Preserve unconfirmed reservations. */
        }
    }

    static Bytes weight_bytes(const Inventory& inventory) {
        // CUDA's large allocation segments round to 2 MiB. Reserve an upper
        // envelope per tensor, including a small-pool segment, rather than
        // treating tensor payload bytes as total retained allocator memory.
        return image_device_weight_bytes(inventory.bytes, inventory.weights.size());
    }

    Entry* find(const Inventory& inventory, int device) const {
        const auto it = entries_.find({const_cast<Inventory*>(&inventory), device});
        return it == entries_.end() ? nullptr : it->second.get();
    }

    bool contains(const Inventory& inventory, int device) const {
        const auto* entry = find(inventory, device);
        return entry && entry->ready;
    }

    std::optional<double> load_cost(const Inventory& inventory, int device) const {
        const auto it =
            load_seconds_.find({const_cast<Inventory*>(&inventory), device});
        return it == load_seconds_.end() ? std::nullopt : std::optional(it->second);
    }

    Bytes device_bytes(int device) const {
        Bytes result = 0;
        for (const auto& [key, entry] : entries_)
            if (entry->device == device)
                result += entry->bytes;
        return result;
    }

    Bytes allocated_bytes(int device) const {
        Bytes result = 0;
        for (const auto& [key, entry] : entries_)
            if (entry->device == device)
                result += entry->allocated;
        return result;
    }

    Snapshot snapshot_for(Snapshot snapshot, const Inventory& inventory) const {
        for (const auto& [key, entry] : entries_)
            if (entry->inventory == &inventory && entry->ready)
                snapshot.used.at(entry->device + 1) -= entry->bytes;
        return snapshot;
    }

    Snapshot reclaimable_snapshot(Snapshot snapshot) const {
        for (const auto& [key, entry] : entries_)
            if (!entry->pinned)
                snapshot.used.at(entry->device + 1) -= entry->bytes;
        return snapshot;
    }

    double eviction_cost(
        const Snapshot& snapshot, const Inventory& inventory, const AtenStagePlan& plan
    ) const {
        const auto* existing =
            plan.stream_weights ? nullptr : find(inventory, plan.device);
        const auto required = plan.device_bytes - (existing ? existing->bytes : 0);
        auto free =
            snapshot.capacity.at(plan.device + 1) - snapshot.used.at(plan.device + 1);
        std::vector<const Entry*> candidates;
        for (const auto& [key, entry] : entries_)
            if (entry->device == plan.device && !entry->pinned
                && entry.get() != existing)
                candidates.push_back(entry.get());
        std::sort(
            candidates.begin(), candidates.end(), [](const auto* a, const auto* b) {
                return a->used < b->used;
            }
        );
        double seconds = 0;
        for (const auto* entry : candidates) {
            if (free >= required)
                break;
            free += entry->allocated;
            seconds += load_cost(*entry->inventory, entry->device).value_or(0);
        }
        return seconds;
    }

    AtenStagePlan plan(
        const Snapshot& snapshot,
        const std::vector<int>& devices,
        const Inventory& inventory,
        Bytes workspace,
        Bytes tiled = 0
    ) const {
        const auto owned = snapshot_for(snapshot, inventory);
        for (int device : devices) {
            if (!contains(inventory, device))
                continue;
            try {
                const auto candidate = aten_stage_plan(
                    owned,
                    {device},
                    weight_bytes(inventory),
                    inventory.largest,
                    workspace,
                    tiled
                );
                if (!candidate.stream_weights)
                    return candidate;
            } catch (const ResourceExhausted&) {
            }
        }
        std::optional<AtenStagePlan> streamed;
        try {
            auto candidate = aten_stage_plan(
                owned,
                devices,
                weight_bytes(inventory),
                inventory.largest,
                workspace,
                tiled
            );
            if (!candidate.stream_weights)
                return candidate;
            streamed = candidate;
        } catch (const ResourceExhausted&) {
        }
        // This is a projection only. Stage::make_room physically evicts and
        // re-probes before reserving any additional memory from this plan.
        try {
            return aten_stage_plan(
                reclaimable_snapshot(snapshot),
                devices,
                weight_bytes(inventory),
                inventory.largest,
                workspace,
                tiled
            );
        } catch (const ResourceExhausted&) {
            if (streamed)
                return *streamed;
            throw;
        }
    }

    void verify(int device) const {
        const auto stats = c10::cuda::CUDACachingAllocator::getDeviceStats(device);
        need(
            Bytes(stats.allocated_bytes[0].current) == allocated_bytes(device)
                && Bytes(stats.reserved_bytes[0].current) <= device_bytes(device),
            "image_aten_cleanup_unconfirmed"
        );
    }

    void evict(Entry& entry) {
        need(!entry.pinned, "image_aten_cache_pinned");
        c10::cuda::CUDAGuard guard(entry.device);
        c10::cuda::device_synchronize();
        if (entry.ready)
            resources_.begin_eviction(entry.handle);
        entry.bank.clear();
        entry.allocated = 0;
        at::cuda::clearCublasWorkspaces();
        c10::cuda::CUDACachingAllocator::emptyCache();
        const auto device = entry.device;
        const auto handle = entry.handle;
        const Key key{entry.inventory, device};
        const auto stats = c10::cuda::CUDACachingAllocator::getDeviceStats(device);
        need(
            Bytes(stats.allocated_bytes[0].current) == allocated_bytes(device)
                && Bytes(stats.reserved_bytes[0].current)
                       <= device_bytes(device) - entry.bytes,
            "image_aten_cleanup_unconfirmed"
        );
        report("eviction", entry);
        resources_.released(handle);
        entries_.erase(key);
        verify(device);
    }

    void clear() {
        while (!entries_.empty())
            evict(*entries_.begin()->second);
    }

    void make_room(const Inventory& inventory, const AtenStagePlan& plan) {
        const auto existing =
            plan.stream_weights ? nullptr : find(inventory, plan.device);
        const auto required = plan.device_bytes - (existing ? existing->bytes : 0);
        for (;;) {
            const auto snapshot = stage_snapshot(resources_, {plan.device});
            const auto free =
                snapshot.capacity[plan.device + 1] - snapshot.used[plan.device + 1];
            if (free >= required)
                return;
            Entry* oldest = nullptr;
            for (const auto& [key, entry] : entries_)
                if (entry->device == plan.device && !entry->pinned
                    && entry.get() != existing
                    && (!oldest || entry->used < oldest->used))
                    oldest = entry.get();
            if (!oldest)
                throw ResourceExhausted("image_aten_component_headroom");
            evict(*oldest);
        }
    }

    Entry& create(Inventory& inventory, int device) {
        need(!find(inventory, device), "image_aten_cache_duplicate");
        auto footprint = resources_.snapshot().capacity;
        std::fill(footprint.begin(), footprint.end(), 0);
        footprint[device + 1] = weight_bytes(inventory);
        auto entry = std::make_unique<Entry>();
        entry->inventory = &inventory;
        entry->stage = inventory.stage;
        entry->device = device;
        entry->bytes = footprint[device + 1];
        const auto [it, inserted] =
            entries_.emplace(Key{&inventory, device}, std::move(entry));
        need(inserted, "image_aten_cache_duplicate");
        try {
            it->second->handle = resources_.reserve(Workload::image, footprint);
        } catch (...) {
            entries_.erase(it);
            throw;
        }
        return *it->second;
    }

    void loaded(Entry& entry, Bytes allocated, double seconds) {
        entry.allocated = allocated;
        load_seconds_[{entry.inventory, entry.device}] = seconds;
        resources_.loaded(entry.handle);
        entry.ready = true;
        report("miss", entry);
    }

    void pin(Entry& entry, bool hit) {
        resources_.pin(entry.handle);
        entry.pinned = true;
        entry.used = ++clock_;
        if (hit)
            report("hit", entry);
    }

    void unpin(Entry& entry) {
        resources_.unpin(entry.handle);
        entry.pinned = false;
        report("retained", entry);
    }
};

// Admission precedes allocation. ATen's allocator is capped at this ledger
// envelope; clear its cached blocks before acknowledging release to the worker.
class Stage {
    Resources& resources_;
    DeviceWeightCache& cache_;
    DeviceWeightCache::Entry* cached_ = nullptr;
    Handle handle_ = 0;
    Inventory& inventory_;
    const std::atomic_bool& cancel_;
    AtenStagePlan plan_;
    BatchMeasurement* measurement_;

    Tensor read_weight(const std::string& name) const {
        stop(cancel_);
        const auto& weight = inventory_.weights.at(name);
        std::vector<std::int64_t> shape(
            weight.info.shape.begin(), weight.info.shape.begin() + weight.info.rank
        );
        const auto stored =
            weight.info.dtype == checkpoint::Dtype::bf16 ? at::kBFloat16 : at::kFloat;
        auto cpu = at::empty(shape, at::TensorOptions().dtype(stored).device(at::kCPU));
        need(cpu.nbytes() == weight.info.bytes, "image_aten_weight_bytes");
        auto* destination = static_cast<std::uint8_t*>(cpu.data_ptr());
        for (std::size_t offset = 0; offset < cpu.nbytes(); offset += 8 * mib) {
            stop(cancel_);
            auto bytes = std::min<std::size_t>(8 * mib, cpu.nbytes() - offset);
            weight.shard->read_tensor(
                weight.info.name, offset, {destination + offset, bytes}
            );
        }
        weight.shard->check_unchanged();
        return cpu.to(at::Device(at::kCUDA, device), inventory_.dtype, false, true);
    }

public:
    int device;

    Stage(
        Resources& r,
        DeviceWeightCache& cache,
        const std::vector<int>& devices,
        Inventory& inventory,
        Bytes workspace,
        const std::atomic_bool& cancel,
        Bytes tiled_workspace = 0,
        BatchMeasurement* measurement = nullptr
    )
        : Stage(
              r,
              cache,
              inventory,
              cache.plan(
                  stage_snapshot(r, devices),
                  devices,
                  inventory,
                  workspace,
                  tiled_workspace
              ),
              cancel,
              measurement
          ) {
    }

    Stage(
        Resources& r,
        DeviceWeightCache& cache,
        Inventory& inventory,
        AtenStagePlan plan,
        const std::atomic_bool& cancel,
        BatchMeasurement* measurement,
        const Stage* resident_source = nullptr
    )
        : resources_(r), cache_(cache), inventory_(inventory), cancel_(cancel),
          plan_(plan), measurement_(measurement), device(plan_.device) {
        need(
            !resident_source
                || (&resident_source->inventory_ == &inventory
                    && !resident_source->plan_.stream_weights && !plan_.stream_weights),
            "image_aten_replica_inventory"
        );
        cache_.make_room(inventory, plan_);
        const bool hit = !plan_.stream_weights && cache_.contains(inventory, device);
        auto footprint = r.snapshot().capacity;
        std::fill(footprint.begin(), footprint.end(), 0);
        footprint[0] = plan_.host_bytes;
        footprint[device + 1] = plan_.device_bytes;
        if (!plan_.stream_weights)
            footprint[device + 1] -= DeviceWeightCache::weight_bytes(inventory);
        handle_ = r.reserve(Workload::image, footprint);
        try {
            at::globalContext().lazyInitDevice(c10::DeviceType::CUDA);
            c10::cuda::CUDAGuard guard(device);
            if (!plan_.stream_weights) {
                cached_ = cache_.find(inventory, device);
                if (!cached_)
                    cached_ = &cache_.create(inventory, device);
            }
            std::size_t free = 0, total = 0;
            C10_CUDA_CHECK(cudaMemGetInfo(&free, &total));
            const auto additional =
                footprint[device + 1] + (cached_ && !hit ? cached_->bytes : 0);
            need(free >= additional, "image_aten_physical_headroom");
            c10::cuda::CUDACachingAllocator::resetPeakStats(device);
            c10::cuda::CUDACachingAllocator::setMemoryFraction(
                double(r.snapshot().used[device + 1]) / double(total), device
            );
            if (cached_ && !hit) {
                const auto started = Clock::now();
                const auto before =
                    c10::cuda::CUDACachingAllocator::getDeviceStats(device)
                        .allocated_bytes[0]
                        .current;
                for (const auto& [name, weight] : inventory.weights) {
                    stop(cancel_);
                    cached_->bank.emplace(
                        name,
                        resident_source ? resident_source->cached_->bank.at(name).to(
                                              at::Device(at::kCUDA, device),
                                              inventory_.dtype,
                                              false,
                                              true
                                          )
                                        : read_weight(name)
                    );
                }
                sync();
                const auto allocated =
                    c10::cuda::CUDACachingAllocator::getDeviceStats(device)
                        .allocated_bytes[0]
                        .current
                    - before;
                cache_.loaded(
                    *cached_,
                    allocated,
                    std::chrono::duration<double>(Clock::now() - started).count()
                );
            }
            if (cached_)
                cache_.pin(*cached_, hit);
            sync();
        } catch (...) {
            auto error = std::current_exception();
            close();
            std::rethrow_exception(error);
        }
    }

    ~Stage() noexcept {
        try {
            close();
        } catch (...) { /* Keep admission when cleanup is unconfirmed. */
        }
    }

    void sync() const {
        c10::cuda::CUDAGuard guard(device);
        c10::cuda::device_synchronize();
    }

    void close() {
        if (!handle_)
            return;
        c10::cuda::CUDAGuard guard(device);
        sync();
        if (measurement_) {
            const auto stats = c10::cuda::CUDACachingAllocator::getDeviceStats(device);
            measurement_->peak_allocated[device + 1] = std::max<Bytes>(
                measurement_->peak_allocated[device + 1], stats.allocated_bytes[0].peak
            );
            measurement_->peak_reserved[device + 1] = std::max<Bytes>(
                measurement_->peak_reserved[device + 1], stats.reserved_bytes[0].peak
            );
            const auto host = resources_.snapshot().used[0];
            measurement_->peak_allocated[0] =
                std::max(measurement_->peak_allocated[0], host);
            measurement_->peak_reserved[0] = measurement_->peak_allocated[0];
        }
        if (cached_) {
            if (cached_->pinned)
                cache_.unpin(*cached_);
            if (!cached_->ready)
                cache_.evict(*cached_);
            cached_ = nullptr;
        }
        at::cuda::clearCublasWorkspaces();
        c10::cuda::CUDACachingAllocator::emptyCache();
        // Caller temporaries must be gone; only explicitly retained banks may
        // survive this stage. Their reservations remain in the shared ledger.
        cache_.verify(device);
        resources_.released(handle_);
        handle_ = 0;
    }

    bool tiled_vae() const {
        return plan_.tile_vae;
    }

    Tensor weight(const std::string& name) const {
        return plan_.stream_weights ? read_weight(name) : cached_->bank.at(name);
    }

    Tensor linear(const std::string& p, const Tensor& x) const {
        // Text stores BF16 weights but uses the reference FP32 forward.
        return at::linear(x, weight(p + ".weight").to(x.scalar_type()));
    }

    Tensor rms(const std::string& p, const Tensor& x, bool centered = false) const {
        auto gamma = weight(p + ".weight").to(at::kFloat);
        if (centered)
            gamma = gamma + 1;
        return at::rms_norm(x.to(at::kFloat), {x.size(-1)}, gamma, 1e-6)
            .to(x.scalar_type());
    }
};

inline Tensor cpu_float(const Tensor& x) {
    auto cpu = x.to(at::kCPU, at::kFloat).contiguous();
    need(at::isfinite(cpu).all().item<bool>(), "image_aten_nonfinite");
    return cpu;
}

}  // namespace kadan::image::detail
