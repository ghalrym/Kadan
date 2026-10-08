#pragma once

#include "kadan/checkpoint.hpp"
#include "kadan/resources.hpp"
#include <atomic>
#include <functional>
#include <optional>

namespace kadan::serving {
// Immutable checkpoint tensor backing, independent of KV/state and scratch.
// Single executor thread; callbacks may inspect snapshots, but not mutate this
// store. Shard metadata has its own caller-admitted MemoryBudget.
class WeightBacking {
public:
    struct Stats { Bytes ram, cold, staging; std::size_t entries; Handle transfer; };
    using Sink = std::function<void(std::size_t, std::span<const std::uint8_t>)>;
    WeightBacking(std::shared_ptr<Resources>, Bytes ram_limit, Bytes cold_limit,
                  std::size_t chunk_limit, std::size_t entry_limit = 1024);
    ~WeightBacking();
    WeightBacking(const WeightBacking&) = delete;
    WeightBacking& operator=(const WeightBacking&) = delete;
    // Versioned model/tensor key; a cold reference never writes/deletes the file.
    void add(std::string key, Workload, std::shared_ptr<const checkpoint::Shard>, std::string tensor);
    bool retain(const std::string&, const std::atomic_bool* cancelled = nullptr);
    void evict(const std::string&);
    void forget(const std::string&);
    // Reserve source + destination + staging peak BEFORE destination allocation.
    // destination includes only immutable weight storage; request state/scratch
    // must have separate reservations in the same Resources instance.
    Handle begin_transfer(const std::string&, Footprint destination);
    // One attempt per ticket. Sink consumes synchronously; spans must not escape.
    // Even on error/cancellation, ticket stays charged until physical cleanup.
    void copy(Handle, const Sink&, const std::atomic_bool* cancelled = nullptr);
    // After successful synchronous copy, discard staging and hand the resident
    // destination reservation to the executor. Executor now owns pin/evict/free.
    Handle commit(Handle);
    void cleaned(Handle, bool physical_cleanup_succeeded);
    Stats stats() const;
private:
    struct Entry {
        Workload workload;
        std::shared_ptr<const checkpoint::Shard> source;
        std::string tensor;
        Bytes bytes;
        std::unique_ptr<std::uint8_t[]> ram;
        Handle reservation = 0;
        std::uint64_t age = 0;
    };
    struct Transfer {
        std::string key;
        Handle reservation;
        std::unique_ptr<std::uint8_t[]> staging;
        std::size_t bytes;
        bool attempted = false, complete = false;
        Footprint destination;
    };
    void idle() const;
    void drop(Entry&);
    void touch(Entry&);
    Footprint host(Bytes) const;
    std::shared_ptr<Resources> resources_;
    Bytes ram_limit_, cold_limit_, ram_ = 0, cold_ = 0;
    std::size_t chunk_limit_, entry_limit_;
    std::map<std::string, Entry> entries_;
    std::optional<Transfer> transfer_;
    bool busy_ = false;
    std::uint64_t clock_ = 0;
};
} // namespace kadan::serving
