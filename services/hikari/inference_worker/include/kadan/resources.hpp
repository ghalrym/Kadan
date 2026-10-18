#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
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
enum class Workload {
    llm,
    image,
    video,
    speech,
    formatting,
    tts,
    decision
};
enum class State {
    loading,
    resident,
    evicting
};

struct Resident {
    Workload workload;
    Footprint bytes;
    State state = State::loading;
    std::uint64_t pins = 0;
};

struct Snapshot {
    Footprint capacity;
    Footprint used;
    std::size_t residents;
};

struct ResourceExhausted : std::runtime_error {
    using std::runtime_error::runtime_error;
};

// Accounting only. One worker owns this ledger; callers acknowledge physical
// cleanup before releasing bytes. No callbacks run under its mutex.
class Resources {
public:
    static constexpr std::size_t max_residents = 1024;
    static constexpr std::size_t max_devices = 64;

    explicit Resources(Footprint capacity)
        : ledger_(std::make_shared<Ledger>(std::move(capacity))) {
        if (ledger_->capacity.empty())
            throw std::invalid_argument("host_budget_required");
        if (ledger_->capacity.size() > max_devices + 1)
            throw std::invalid_argument("too_many_devices");
    }

    Resources(const Resources&) = delete;
    Resources& operator=(const Resources&) = delete;

    ~Resources() {
        if (!scope_)
            return;
        std::lock_guard lock(ledger_->mutex);
        auto& local = ledger_->scopes.at(scope_);
        // An abandoned scope returns only unmaterialized escrow. Actual bytes
        // remain charged until physical cleanup is acknowledged by its owner.
        for (std::size_t i = 0; i < local.used.size(); ++i) {
            ledger_->used[i] -= local.allowance[i] - local.used[i];
            local.allowance[i] = local.used[i];
        }
        if (!local.residents)
            ledger_->scopes.erase(scope_);
    }

    // Acquire the full load/execution envelope atomically, including lazy KV,
    // activation and workspace allocations that have not materialized yet.
    std::shared_ptr<Resources> scope(Footprint allowance) {
        return make_scope(std::move(allowance));
    }

    void admit(Footprint allowance) {
        std::lock_guard lock(ledger_->mutex);
        require(scope_ != 0, "scope_required");
        auto& local = ledger_->scopes.at(scope_);
        require(allowance.size() == ledger_->capacity.size(), "budget_shape");
        for (std::size_t i = 0; i < allowance.size(); ++i) {
            require(allowance[i] >= local.used[i], "allowance_below_usage");
            const auto previous = local.allowance[i];
            if (allowance[i] > ledger_->capacity[i] - (ledger_->used[i] - previous))
                throw ResourceExhausted("exhausted");
        }
        for (std::size_t i = 0; i < allowance.size(); ++i) {
            const auto previous = local.allowance[i];
            ledger_->used[i] = ledger_->used[i] - previous + allowance[i];
        }
        local.allowance.swap(allowance);
    }

    // Preserve resident weights while returning the unused execution envelope.
    // Reuse must reacquire an envelope with admit() before allocating again.
    void idle() {
        std::lock_guard lock(ledger_->mutex);
        require(scope_ != 0, "scope_required");
        auto& local = ledger_->scopes.at(scope_);
        for (std::size_t i = 0; i < local.used.size(); ++i) {
            ledger_->used[i] -= local.allowance[i] - local.used[i];
            local.allowance[i] = local.used[i];
        }
    }

    bool empty() const {
        std::lock_guard lock(ledger_->mutex);
        return empty_locked();
    }

    void reset_capacity(Footprint capacity) {
        std::lock_guard lock(ledger_->mutex);
        require(scope_ == 0, "root_required");
        if (capacity.size() != ledger_->capacity.size() || !empty_locked())
            throw std::runtime_error("capacity_requires_empty_ledger");
        ledger_->capacity_limit = capacity;
        ledger_->capacity.swap(capacity);
    }

    void refresh_available(const Footprint& available) {
        std::lock_guard lock(ledger_->mutex);
        if (available.size() != ledger_->capacity.size())
            throw std::invalid_argument("budget_shape");
        // Keep a stable admission epoch while any real or escrowed allocation
        // exists. A probe includes neither all lazy allocations nor proof of
        // which reservations materialized. The scheduler separately clips new
        // envelopes to physical free memory; never add reservations to a probe.
        if (!empty_locked())
            return;
        for (std::size_t i = 0; i < available.size(); ++i)
            ledger_->capacity[i] = std::min(ledger_->capacity_limit[i], available[i]);
    }

    Handle reserve(Workload workload, Footprint bytes) {
        std::lock_guard lock(ledger_->mutex);
        if (bytes.size() != ledger_->capacity.size())
            throw std::invalid_argument("budget_shape");
        if (ledger_->residents.size() >= max_residents)
            throw std::runtime_error("resident_limit");
        const auto view = snapshot_locked();
        for (std::size_t i = 0; i < bytes.size(); ++i)
            if (bytes[i] > view.capacity[i] - view.used[i])
                throw ResourceExhausted("exhausted");
        if (ledger_->next == std::numeric_limits<Handle>::max())
            throw std::runtime_error("handles_exhausted");
        const auto handle = ledger_->next;
        auto& local = ledger_->scopes.at(scope_);
        ledger_->residents.emplace(
            handle, Entry{scope_, Resident{workload, std::move(bytes)}}
        );
        const auto& reserved = ledger_->residents.at(handle).resident.bytes;
        for (std::size_t i = 0; i < reserved.size(); ++i) {
            local.used[i] += reserved[i];
            if (!scope_)
                ledger_->used[i] += reserved[i];
        }
        ++local.residents;
        ++ledger_->next;
        return handle;
    }

    void resize_loading(Handle handle, Footprint bytes) {
        std::lock_guard lock(ledger_->mutex);
        auto& r = get(handle);
        require(r.state == State::loading && r.pins == 0, "not_loading");
        require(bytes.size() == ledger_->capacity.size(), "budget_shape");
        const auto view = snapshot_locked();
        for (std::size_t i = 0; i < bytes.size(); ++i)
            if (bytes[i] > r.bytes[i]
                && bytes[i] - r.bytes[i] > view.capacity[i] - view.used[i])
                throw ResourceExhausted("exhausted");
        auto& local = ledger_->scopes.at(scope_);
        for (std::size_t i = 0; i < bytes.size(); ++i) {
            local.used[i] = local.used[i] - r.bytes[i] + bytes[i];
            if (!scope_)
                ledger_->used[i] = ledger_->used[i] - r.bytes[i] + bytes[i];
        }
        r.bytes.swap(bytes);
    }

    void loaded(Handle handle) {
        std::lock_guard lock(ledger_->mutex);
        auto& r = get(handle);
        require(r.state == State::loading, "not_loading");
        r.state = State::resident;
    }

    void pin(Handle handle) {
        std::lock_guard lock(ledger_->mutex);
        auto& r = get(handle);
        require(r.state == State::resident, "not_resident");
        require(r.pins != std::numeric_limits<std::uint64_t>::max(), "pins_exhausted");
        ++r.pins;
    }

    void unpin(Handle handle) {
        std::lock_guard lock(ledger_->mutex);
        auto& r = get(handle);
        require(r.pins != 0, "not_pinned");
        --r.pins;
    }

    void begin_eviction(Handle handle) {
        std::lock_guard lock(ledger_->mutex);
        auto& r = get(handle);
        require(r.state == State::resident && r.pins == 0, "busy");
        r.state = State::evicting;
    }

    void eviction_failed(Handle handle) {
        std::lock_guard lock(ledger_->mutex);
        auto& r = get(handle);
        require(r.state == State::evicting, "not_evicting");
        r.state = State::resident;
    }

    void released(Handle handle) {
        std::lock_guard lock(ledger_->mutex);
        const auto& r = get(handle);
        require(r.state != State::resident && r.pins == 0, "busy");
        auto& local = ledger_->scopes.at(scope_);
        for (std::size_t i = 0; i < r.bytes.size(); ++i) {
            local.used[i] -= r.bytes[i];
            if (!scope_)
                ledger_->used[i] -= r.bytes[i];
        }
        --local.residents;
        ledger_->residents.erase(handle);
    }

    Resident reservation(Handle handle) const {
        std::lock_guard lock(ledger_->mutex);
        return get(handle);
    }

    Snapshot snapshot() const {
        std::lock_guard lock(ledger_->mutex);
        return snapshot_locked();
    }

private:
    struct Entry {
        std::uint64_t scope;
        Resident resident;
    };

    struct Scope {
        Footprint used, allowance;
        std::size_t residents = 0;
    };

    struct Ledger {
        explicit Ledger(Footprint budget)
            : capacity_limit(budget), capacity(std::move(budget)),
              used(capacity.size()) {
            scopes.emplace(0, Scope{Footprint(capacity.size()), {}, 0});
        }

        Footprint capacity_limit, capacity, used;
        std::map<Handle, Entry> residents;
        std::map<std::uint64_t, Scope> scopes;
        Handle next = 1;
        std::uint64_t next_scope = 1;
        mutable std::mutex mutex;
    };

    Resources(std::shared_ptr<Ledger> ledger, std::uint64_t scope)
        : ledger_(std::move(ledger)), scope_(scope) {
    }

    std::shared_ptr<Resources> make_scope(Footprint allowance) {
        std::lock_guard lock(ledger_->mutex);
        if (allowance.size() != ledger_->capacity.size())
            throw std::invalid_argument("budget_shape");
        if (ledger_->scopes.size() > max_residents)
            throw std::runtime_error("scope_limit");
        if (ledger_->next_scope == std::numeric_limits<std::uint64_t>::max())
            throw std::runtime_error("scopes_exhausted");
        for (std::size_t i = 0; i < allowance.size(); ++i)
            if (allowance[i] > ledger_->capacity[i] - ledger_->used[i])
                throw ResourceExhausted("exhausted");
        // Allocate before changing root accounting. Construct with scope zero so
        // allocation failures cannot invoke a destructor for an absent record.
        auto result = std::shared_ptr<Resources>(new Resources(ledger_, 0));
        const auto id = ledger_->next_scope;
        ledger_->scopes.emplace(
            id, Scope{Footprint(ledger_->capacity.size()), std::move(allowance), 0}
        );
        for (std::size_t i = 0; i < ledger_->capacity.size(); ++i)
            ledger_->used[i] += ledger_->scopes.at(id).allowance[i];
        result->scope_ = id;
        ++ledger_->next_scope;
        return result;
    }

    bool empty_locked() const {
        return ledger_->residents.empty()
               && std::all_of(
                   ledger_->used.begin(), ledger_->used.end(), [](auto bytes) {
                       return bytes == 0;
                   }
               );
    }

    Snapshot snapshot_locked() const {
        if (!scope_)
            return {ledger_->capacity, ledger_->used, ledger_->residents.size()};
        const auto& local = ledger_->scopes.at(scope_);
        return {local.allowance, local.used, local.residents};
    }

    static void require(bool ok, const char* error) {
        if (!ok)
            throw std::runtime_error(error);
    }

    Resident& get(Handle handle) const {
        auto it = ledger_->residents.find(handle);
        if (it == ledger_->residents.end() || it->second.scope != scope_)
            throw std::runtime_error("stale_handle");
        return it->second.resident;
    }

    std::shared_ptr<Ledger> ledger_;
    std::uint64_t scope_ = 0;
};
}  // namespace kadan
