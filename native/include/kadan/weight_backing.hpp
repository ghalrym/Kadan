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
    static constexpr std::size_t default_entry_limit=1024, default_control_bytes=2*1024*1024;
    static constexpr std::size_t model_entry_limit=131072, model_control_bytes=256*1024*1024;
    struct Stats { Bytes ram=0, cold=0, staging=0; std::size_t entries=0; Handle transfer=0;
        Bytes capacity=0, hits=0, misses=0, hit_bytes=0, source_bytes=0, evictions=0; };
    using Sink = std::function<void(std::size_t, std::span<const std::uint8_t>)>;
    WeightBacking(std::shared_ptr<Resources>, Bytes ram_limit, Bytes cold_limit,
                  std::size_t chunk_limit, std::size_t entry_limit = 1024, bool aggregate = false);
    ~WeightBacking();
    WeightBacking(const WeightBacking&) = delete;
    WeightBacking& operator=(const WeightBacking&) = delete;
    // Versioned model/tensor key; a cold reference never writes/deletes the file.
    void add(std::string key, Workload, std::shared_ptr<const checkpoint::Shard>, std::string tensor);
    // Caller owns/admitted destination (e.g. loader staging). Cache misses fall
    // back to bounded checkpoint reads when cache metadata or RAM is full.
    void read_through(const std::string&, Workload, std::shared_ptr<const checkpoint::Shard>,
                      const std::string& tensor, std::size_t offset, std::span<std::uint8_t>,
                      const std::atomic_bool* cancelled = nullptr);
    bool retain(const std::string&, const std::atomic_bool* cancelled = nullptr);
    bool room_for_reservations(std::size_t count);
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
        bool attempted = false, complete = false, cleanup_started = false;
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
    // Model mode reserves one fixed cache envelope, not one ledger slot per tensor.
    Handle aggregate_ = 0;
    Bytes hits_=0, misses_=0, hit_bytes_=0, source_bytes_=0, evictions_=0;
};
} // namespace kadan::serving
