#pragma once

#include <cstdint>
#include <limits>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace kadan {
using Bytes = std::uint64_t;
using Handle = std::uint64_t;
// Index zero is host RAM; subsequent entries are independent device budgets.
using Footprint = std::vector<Bytes>;
enum class Workload { llm, image, video, speech, tts, decision };
enum class State { loading, resident, evicting };
struct Resident {
    Workload workload;
    Footprint bytes;
    State state = State::loading;
    std::uint64_t pins = 0;
};
struct Snapshot { Footprint capacity; Footprint used; std::size_t residents; };

// Accounting only. One worker owns this ledger; callers acknowledge physical
// cleanup before releasing bytes. No callbacks run under its mutex.
class Resources {
public:
    // Bound control-plane storage even when every reservation requests zero bytes.
    static constexpr std::size_t max_residents = 1024;
    static constexpr std::size_t max_devices = 64;
    explicit Resources(Footprint capacity) : capacity_(std::move(capacity)), used_(capacity_.size()) {
        if (capacity_.empty()) throw std::invalid_argument("host_budget_required");
        if (capacity_.size() > max_devices + 1) throw std::invalid_argument("too_many_devices");
    }
    Handle reserve(Workload workload, Footprint bytes) {
        std::lock_guard lock(mutex_);
        if (bytes.size() != capacity_.size()) throw std::invalid_argument("budget_shape");
        if (residents_.size() >= max_residents) throw std::runtime_error("resident_limit");
        for (std::size_t i = 0; i < bytes.size(); ++i)
            if (bytes[i] > capacity_[i] - used_[i]) throw std::runtime_error("exhausted");
        if (next_ == std::numeric_limits<Handle>::max()) throw std::runtime_error("handles_exhausted");
        const auto handle = next_;
        // Allocate the record before changing accounting (strong exception safety).
        residents_.emplace(handle, Resident{workload, std::move(bytes)});
        for (std::size_t i = 0; i < used_.size(); ++i) used_[i] += residents_.at(handle).bytes[i];
        ++next_;
        return handle;
    }
    void loaded(Handle handle) {
        std::lock_guard lock(mutex_);
        auto& r = get(handle);
        require(r.state == State::loading, "not_loading");
        r.state = State::resident;
    }
    void pin(Handle handle) {
        std::lock_guard lock(mutex_);
        auto& r = get(handle);
        require(r.state == State::resident, "not_resident");
        require(r.pins != std::numeric_limits<std::uint64_t>::max(), "pins_exhausted");
        ++r.pins;
    }
    void unpin(Handle handle) {
        std::lock_guard lock(mutex_);
        auto& r = get(handle);
        require(r.pins != 0, "not_pinned");
        --r.pins;
    }
    void begin_eviction(Handle handle) {
        std::lock_guard lock(mutex_);
        auto& r = get(handle);
        require(r.state == State::resident && r.pins == 0, "busy");
        r.state = State::evicting;
    }
    void eviction_failed(Handle handle) {
        std::lock_guard lock(mutex_);
        auto& r = get(handle);
        require(r.state == State::evicting, "not_evicting");
        r.state = State::resident;
    }
    // For a failed/aborted load or successful eviction, only after allocations
    // and asynchronous device work have been cleaned up by the executor.
    void released(Handle handle) {
        std::lock_guard lock(mutex_);
        const auto& r = get(handle);
        require(r.state != State::resident && r.pins == 0, "busy");
        for (std::size_t i = 0; i < used_.size(); ++i) used_[i] -= r.bytes[i];
        residents_.erase(handle);
    }
    Snapshot snapshot() const {
        std::lock_guard lock(mutex_);
        return {capacity_, used_, residents_.size()};
    }
private:
    static void require(bool ok, const char* error) { if (!ok) throw std::runtime_error(error); }
    Resident& get(Handle handle) {
        auto it = residents_.find(handle);
        if (it == residents_.end()) throw std::runtime_error("stale_handle");
        return it->second;
    }
    Footprint capacity_, used_;
    std::map<Handle, Resident> residents_;
    Handle next_ = 1;
    mutable std::mutex mutex_;
};
} // namespace kadan
