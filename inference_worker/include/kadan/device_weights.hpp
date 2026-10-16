#pragma once
#include <algorithm>
#include <atomic>
#include <map>
#include <span>

#ifdef KADAN_TRACE_WEIGHT_CACHE
#include <array>
#include <cinttypes>
#include <cstdio>
#include <string_view>
#endif

#include "kadan/checkpoint.hpp"
#include "kadan/cleanup_error.hpp"
#include "kadan/resources.hpp"

namespace kadan {
class WeightDeviceOperations {
public:
    virtual ~WeightDeviceOperations() = default;
    // Free physical bytes available for weights after the scratch allowance.
    virtual Bytes available_weight_bytes() = 0;
    // nullptr means confirmed allocation OOM, without an outstanding allocation.
    virtual void* allocate_weight(Bytes) = 0;
    virtual void copy_weight(void*, std::size_t, std::span<const std::byte>) = 0;
    // Both pointers belong to this selected device; offset applies to both.
    virtual void copy_weight_device(
        void* destination, std::size_t offset, const void* source, std::size_t bytes
    ) = 0;
    virtual void free_weight(void*) = 0;
    virtual void synchronize_weights() = 0;
};

// One selected device, used only on its owned thread. Exact F32/I8 bytes;
// scratch remains separate. The fixed envelope uses one ledger slot even for
// thousands of projections. Failed physical cleanup keeps that envelope charged.
class DeviceWeights {
public:
    struct Key {
        WeightIdentity source{};
        std::size_t first = 0, count = 0, width = 0;
        auto operator<=>(const Key&) const = default;
    };

    static constexpr Bytes metadata_bytes = 4 * 1024 * 1024;

    DeviceWeights(
        Resources& resources,
        int device,
        Workload workload,
        Bytes limit,
        WeightDeviceOperations& operations
    )
        : resources_(resources), device_(device), workload_(workload), limit_(limit),
          maximum_(limit), operations_(operations), metadata_(metadata_bytes),
          entries_(&metadata_) {
#ifdef KADAN_TRACE_WEIGHT_CACHE
        trace_event("open");
#endif
    }

    DeviceWeights(const DeviceWeights&) = delete;
    DeviceWeights& operator=(const DeviceWeights&) = delete;

    Bytes reserved() const {
        return reservation_ ? limit_ : 0;
    }

    Bytes allocated() const {
        return allocated_;
    }

    bool quarantined() const {
        return quarantined_;
    }

    void bound(Bytes bytes) {
        healthy();
        if (reservation_)
            throw std::runtime_error("weight_plan_already_loaded");
        limit_ = std::min(maximum_, bytes);
#ifdef KADAN_TRACE_WEIGHT_CACHE
        trace_event("bound");
#endif
    }

    Bytes capacity() const {
        return limit_;
    }

    Bytes maximum_capacity() const {
        return maximum_;
    }

    void* find(const Key& key, [[maybe_unused]] bool retain_misses = true) const {
        healthy();
        auto* pointer = lookup(key);
#ifdef KADAN_TRACE_WEIGHT_CACHE
        ++trace_.accesses;
        if (pointer) {
            ++trace_.hits;
            trace_.hit_bytes += trace_bytes(key);
        } else {
            ++trace_.misses;
            trace_.miss_bytes += trace_bytes(key);
        }
        trace_event("access", key, pointer != nullptr, retain_misses);
#endif
        return pointer;
    }

    // nullptr explicitly selects caller-owned streaming scratch. A cache miss
    // cannot consume its scratch reserve or silently change representation.
    void* retain(
        const Key& key,
        std::span<const std::byte> source,
        const std::atomic_bool& cancel
    ) {
        return retain_bytes(key, source.data(), source.size(), false, cancel);
    }

    // The caller validates immutable packed bytes on this device before admission.
    // Copy into bank ownership while the source scratch remains alive.
    void* retain_device(
        const Key& key, const void* source, Bytes bytes, const std::atomic_bool& cancel
    ) {
        if (bytes && !source)
            throw std::invalid_argument("weight_device_source");
        return retain_bytes(key, source, bytes, true, cancel);
    }

    // Keep the admitted bank envelope/partition for the next stage. Only physical
    // tiles matching these exact tensor identities are retired.
    Bytes release_sources(std::span<const WeightIdentity> sources) {
        healthy();
        const auto before = allocated_;
#ifdef KADAN_TRACE_WEIGHT_CACHE
        for (const auto& source : sources)
            trace_event("retire_source", {source, 0, 0, 0});
#endif
        try {
            for (auto it = entries_.begin(); it != entries_.end();) {
                auto entry = it++;
                if (std::find(sources.begin(), sources.end(), entry->first.source)
                    != sources.end())
                    drop(entry, "retire");
            }
        } catch (...) {
            quarantined_ = true;
#ifdef KADAN_TRACE_WEIGHT_CACHE
            trace_summary();
#endif
            throw;
        }
        return before - allocated_;
    }

    void release() {
        healthy();
        try {
            if (!entries_.empty())
                operations_.synchronize_weights();
            while (!entries_.empty())
                drop(entries_.begin(), "release");
            if (reservation_) {
                resources_.released(reservation_);
                reservation_ = 0;
            }
#ifdef KADAN_TRACE_WEIGHT_CACHE
            trace_event("clear");
            trace_summary();
#endif
        } catch (...) {
            quarantined_ = true;
#ifdef KADAN_TRACE_WEIGHT_CACHE
            trace_summary();
#endif
            throw;
        }
    }

private:
    void* retain_bytes(
        const Key& key,
        const void* source,
        Bytes bytes,
        bool device_source,
        const std::atomic_bool& cancel
    ) {
        healthy();
        stop(cancel);
        if (!key.width
            || key.count > std::numeric_limits<std::size_t>::max() / key.width
            || key.count * key.width != bytes)
            throw std::invalid_argument("weight_representation_size");
        if (auto pointer = lookup(key))
            return pointer;
        if (!bytes) {
            trace_refusal(key, "empty");
            return nullptr;
        }
        if (bytes > limit_ - allocated_) {
            trace_refusal(key, "envelope");
            return nullptr;
        }
        if (bytes > operations_.available_weight_bytes()) {
            trace_refusal(key, "physical");
            return nullptr;
        }
        if (!reservation_) {
            auto footprint = resources_.snapshot().capacity;
            if (device_ < 0 || std::size_t(device_) + 1 >= footprint.size())
                throw std::invalid_argument("weight_device");
            std::fill(footprint.begin(), footprint.end(), 0);
            footprint[0] = metadata_bytes;
            footprint[device_ + 1] = limit_;
            reservation_ = resources_.reserve(workload_, std::move(footprint));
        }
        auto entry = entries_.end();
        try {
            entry = entries_.try_emplace(key).first;
        } catch (const std::bad_alloc&) {
            trace_refusal(key, "metadata");
            return nullptr;
        }
        try {
            entry->second.pointer = operations_.allocate_weight(bytes);
            if (!entry->second.pointer) {
                entries_.erase(entry);
                trace_refusal(key, "allocation");
                return nullptr;
            }
            entry->second.bytes = bytes;
            allocated_ += bytes;
            for (std::size_t at = 0; at < bytes;) {
                stop(cancel);
                const auto count = std::min<std::size_t>(1024 * 1024, bytes - at);
                if (device_source)
                    operations_.copy_weight_device(
                        entry->second.pointer, at, source, count
                    );
                else
                    operations_.copy_weight(
                        entry->second.pointer,
                        at,
                        {static_cast<const std::byte*>(source) + at, count}
                    );
                at += count;
            }
            operations_.synchronize_weights();
            stop(cancel);
            entry->second.ready = true;
#ifdef KADAN_TRACE_WEIGHT_CACHE
            trace_.admitted_bytes += bytes;
            trace_event("admit", key);
#endif
            return entry->second.pointer;
        } catch (...) {
            const auto error = std::current_exception();
            rethrow_after_cleanup(error, [&] {
                try {
                    drop(entry, "rollback");
                } catch (...) {
                    quarantined_ = true;
#ifdef KADAN_TRACE_WEIGHT_CACHE
                    trace_summary();
#endif
                    throw;
                }
            });
        }
    }

    struct Entry {
        void* pointer = nullptr;
        Bytes bytes = 0;
        bool ready = false;
    };

    using Entries = std::pmr::map<Key, Entry>;

    void* lookup(const Key& key) const {
        const auto it = entries_.find(key);
        return it == entries_.end() || !it->second.ready ? nullptr : it->second.pointer;
    }

    void trace_refusal(
        [[maybe_unused]] const Key& key, [[maybe_unused]] const char* reason
    ) const {
#ifdef KADAN_TRACE_WEIGHT_CACHE
        ++trace_.refused;
        trace_event("refuse", key, false, true, reason);
#endif
    }

    void healthy() const {
        if (quarantined_)
            throw std::runtime_error("weight_cleanup_unconfirmed");
    }

    static void stop(const std::atomic_bool& cancel) {
        if (cancel.load())
            throw std::runtime_error("weight_cancelled");
    }

    void drop(Entries::iterator entry, [[maybe_unused]] const char* reason) {
        if (entry->second.pointer) {
            operations_.synchronize_weights();
            operations_.free_weight(entry->second.pointer);
            allocated_ -= entry->second.bytes;
        }
#ifdef KADAN_TRACE_WEIGHT_CACHE
        if (std::string_view(reason) != "rollback")
            trace_.retired_bytes += entry->second.bytes;
        trace_event("drop", entry->first, false, true, reason);
#endif
        entries_.erase(entry);
    }

#ifdef KADAN_TRACE_WEIGHT_CACHE
    // Fixed counters only; there is no event buffer or identity dictionary.
    // These control fields fit the bank owner's existing metadata allowance.
    struct TraceCounters {
        std::uint64_t sequence = 0, accesses = 0, hits = 0, misses = 0, hit_bytes = 0,
                      miss_bytes = 0, admitted_bytes = 0, refused = 0,
                      retired_bytes = 0, emitted = 0, dropped = 0;
        bool output_failed = false;
    };

    // Across all banks on one device: 131072 ordinary records, 1024 summaries,
    // and at most one truncation notice for each allowance (132098 lines total).
    static constexpr std::uint64_t trace_record_limit = 131072,
                                   trace_summary_limit = 1024;
    inline static std::atomic_uint64_t next_trace_bank_ = 0;
    inline static std::array<std::atomic_uint64_t, Resources::max_devices>
        trace_device_records_{};
    inline static std::array<std::atomic_uint64_t, Resources::max_devices>
        trace_device_summaries_{};
    const std::uint64_t trace_bank_ = ++next_trace_bank_;
    mutable TraceCounters trace_;

    static Bytes trace_bytes(const Key& key) {
        return key.width && key.count <= std::numeric_limits<Bytes>::max() / key.width
                   ? key.count * key.width
                   : 0;
    }

    void trace_event(const char* event) const noexcept {
        trace_event(event, {WeightIdentity{}, 0, 0, 0});
    }

    void trace_event(
        const char* event,
        const Key& key,
        bool hit = false,
        bool eligible = true,
        const char* reason = ""
    ) const noexcept {
        const auto sequence = ++trace_.sequence;
        if (device_ < 0 || std::size_t(device_) >= trace_device_records_.size())
            return;
        const auto emitted = trace_device_records_[device_].fetch_add(1);
        if (emitted >= trace_record_limit) {
            ++trace_.dropped;
            if (emitted == trace_record_limit
                && std::fprintf(
                       stderr,
                       "{\"weight_cache_trace\":1,\"bank\":%" PRIu64
                       ",\"device\":%d,\"seq\":%" PRIu64
                       ",\"event\":\"truncated\",\"capacity\":%" PRIu64
                       ",\"allocated\":%" PRIu64 "}\n",
                       trace_bank_,
                       device_,
                       sequence,
                       limit_,
                       allocated_
                   ) < 0)
                trace_.output_failed = true;
            return;
        }
        ++trace_.emitted;
        const auto& source = key.source;
        if (std::fprintf(
                stderr,
                "{\"weight_cache_trace\":1,\"bank\":%" PRIu64
                ",\"device\":%d,\"seq\":%" PRIu64 ",\"event\":\"%s\",\"key\":[%" PRIu64
                ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64
                ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64
                "],\"bytes\":%" PRIu64
                ",\"hit\":%s,\"eligible\":%s,\"capacity\":%" PRIu64
                ",\"allocated\":%" PRIu64 ",\"reason\":\"%s\"}\n",
                trace_bank_,
                device_,
                sequence,
                event,
                source.device,
                source.inode,
                source.file_size,
                source.modified_seconds,
                source.modified_nanoseconds,
                source.changed_seconds,
                source.changed_nanoseconds,
                source.payload_begin,
                source.payload_end,
                std::uint64_t(key.first),
                std::uint64_t(key.count),
                std::uint64_t(key.width),
                trace_bytes(key),
                hit ? "true" : "false",
                eligible ? "true" : "false",
                limit_,
                allocated_,
                reason
            )
            < 0)
            trace_.output_failed = true;
    }

    void trace_summary() const noexcept {
        const auto sequence = ++trace_.sequence;
        if (device_ < 0 || std::size_t(device_) >= trace_device_summaries_.size())
            return;
        const auto emitted = trace_device_summaries_[device_].fetch_add(1);
        if (emitted >= trace_summary_limit) {
            ++trace_.dropped;
            if (emitted == trace_summary_limit
                && std::fprintf(
                       stderr,
                       "{\"weight_cache_trace\":1,\"bank\":%" PRIu64
                       ",\"device\":%d,\"seq\":%" PRIu64
                       ",\"event\":\"truncated\",\"reason\":\"summary_limit\","
                       "\"capacity\":%" PRIu64 ",\"allocated\":%" PRIu64 "}\n",
                       trace_bank_,
                       device_,
                       sequence,
                       limit_,
                       allocated_
                   ) < 0)
                trace_.output_failed = true;
            return;
        }
        if (std::fprintf(
                stderr,
                "{\"weight_cache_trace\":1,\"bank\":%" PRIu64
                ",\"device\":%d,\"seq\":%" PRIu64
                ",\"event\":\"summary\",\"capacity\":%" PRIu64 ",\"allocated\":%" PRIu64
                ",\"accesses\":%" PRIu64 ",\"hits\":%" PRIu64 ",\"misses\":%" PRIu64
                ",\"hit_bytes\":%" PRIu64 ",\"miss_bytes\":%" PRIu64
                ",\"admitted_bytes\":%" PRIu64 ",\"refused\":%" PRIu64
                ",\"retired_bytes\":%" PRIu64 ",\"emitted\":%" PRIu64
                ",\"dropped\":%" PRIu64 ",\"quarantined\":%s,\"output_failed\":%s}\n",
                trace_bank_,
                device_,
                sequence,
                limit_,
                allocated_,
                trace_.accesses,
                trace_.hits,
                trace_.misses,
                trace_.hit_bytes,
                trace_.miss_bytes,
                trace_.admitted_bytes,
                trace_.refused,
                trace_.retired_bytes,
                trace_.emitted,
                trace_.dropped,
                quarantined_ ? "true" : "false",
                trace_.output_failed ? "true" : "false"
            )
            < 0)
            trace_.output_failed = true;
    }
#endif

    Resources& resources_;
    int device_;
    Workload workload_;
    Bytes limit_, maximum_;
    WeightDeviceOperations& operations_;
    checkpoint::MemoryBudget metadata_;
    Entries entries_;
    Handle reservation_ = 0;
    Bytes allocated_ = 0;
    bool quarantined_ = false;
};
}  // namespace kadan
