#pragma once

#include "kadan/resources.hpp"
#include <deque>
#include <optional>
#include <memory>

namespace kadan::serving {
// Single-owner event-loop component. The transport serializes calls in receipt
// order. Executors remain modality-specific and acknowledge physical work.
class GenerationQueue {
public:
    struct Model {
        std::string key; // Include checkpoint revision and execution configuration.
        Workload workload;
        Footprint bytes; // Includes all model and per-request peak allocations.
        bool operator==(const Model&) const = default;
    };
    enum class Kind { idle, load, execute, cleanup };
    struct Action { Kind kind; Handle request; Handle reservation; };

    explicit GenerationQueue(Footprint capacity, std::size_t limit = 32)
        : GenerationQueue(std::make_shared<Resources>(std::move(capacity)), limit) {}
    explicit GenerationQueue(std::shared_ptr<Resources> resources, std::size_t limit = 32)
        : resources_(std::move(resources)), limit_(limit) {
        if (!resources_) throw std::invalid_argument("resources_required");
        if (!limit || limit > 1024) throw std::invalid_argument("queue_limit");
    }
    Handle submit(Model model) {
        require(!stopping_, "stopping");
        require(!model.key.empty() && model.key.size() <= 256, "model_key");
        require(model.workload == Workload::llm || model.workload == Workload::image || model.workload == Workload::decision || model.workload == Workload::video, "workload");
        const auto capacity = resources_->snapshot().capacity;
        require(model.bytes.size() == capacity.size(), "budget_shape");
        for (std::size_t i = 0; i < capacity.size(); ++i)
            require(model.bytes[i] <= capacity[i], "exhausted");
        require(queue_.size() < limit_, "queue_full");
        require(next_ != std::numeric_limits<Handle>::max(), "requests_exhausted");
        queue_.push_back({next_, std::move(model)});
        return next_++;
    }
    // Repeated polls repeat the outstanding action; adapters must not launch it
    // twice. No later request can bypass an unacknowledged action.
    Action poll() {
        if (phase_ != Kind::idle) return action();
        if (resident_ && (stopping_ || (!queue_.empty() && !(resident_->model == queue_.front().model)))) {
            resources_->begin_eviction(resident_->handle);
            phase_ = Kind::cleanup;
            return action();
        }
        if (queue_.empty()) return {Kind::idle, 0, 0};
        if (!resident_) {
            // Construct before reserving to preserve accounting on allocation failure.
            Resident next{queue_.front().model, 0};
            next.handle = resources_->reserve(next.model.workload, next.model.bytes);
            resident_.emplace(std::move(next));
            phase_ = Kind::load;
        } else {
            resources_->pin(resident_->handle);
            phase_ = Kind::execute;
        }
        active_ = queue_.front().id;
        return action();
    }
    void loaded(Handle id, bool success) {
        check(id, Kind::load);
        if (!success || cancelled_) {
            // Even failed loads may own allocations. Keep their reservation.
            phase_ = Kind::cleanup;
            return;
        }
        resources_->loaded(resident_->handle);
        resources_->pin(resident_->handle);
        phase_ = Kind::execute;
    }
    // Call only after request-local work is synchronized and cleaned up.
    void completed(Handle id, bool reusable) {
        check(id, Kind::execute);
        resources_->unpin(resident_->handle);
        if (!reusable || cancelled_ || stopping_) {
            resources_->begin_eviction(resident_->handle);
            phase_ = Kind::cleanup;
        } else {
            finish();
            phase_ = Kind::idle;
        }
    }
    // A false acknowledgement retains accounting and blocks FIFO advancement.
    // Retrying cleanup must be safe in the executor, including partial frees.
    void cleaned(Handle reservation, bool success) {
        require(phase_ == Kind::cleanup && resident_ && resident_->handle == reservation, "stale_cleanup");
        if (!success) return;
        resources_->released(resident_->handle);
        resident_.reset();
        if (active_) finish();
        phase_ = Kind::idle;
    }
    bool cancel(Handle id) {
        if (active_ == id && active_) { cancelled_ = true; return true; }
        for (auto it = queue_.begin(); it != queue_.end(); ++it) {
            if (it->id == id) { queue_.erase(it); return true; }
        }
        return false;
    }
    bool cancellation_requested() const { return cancelled_; }
    void stop() {
        stopping_ = true;
        while (queue_.size() > (active_ ? 1u : 0u)) queue_.pop_back();
        if (active_) cancelled_ = true;
    }
    void evict_idle() {
        require(!active_ && queue_.empty() && phase_ == Kind::idle, "busy");
        if (resident_) { resources_->begin_eviction(resident_->handle); phase_ = Kind::cleanup; }
    }
    Snapshot snapshot() const { return resources_->snapshot(); }
    std::size_t pending() const { return queue_.size(); }
private:
    struct Request { Handle id; Model model; };
    struct Resident { Model model; Handle handle; };
    static void require(bool condition, const char* error) {
        if (!condition) throw std::runtime_error(error);
    }
    void check(Handle id, Kind expected) const {
        require(phase_ == expected && active_ && active_ == id, "stale_request");
    }
    Action action() const { return {phase_, active_, resident_->handle}; }
    void finish() { queue_.pop_front(); active_ = 0; cancelled_ = false; }
    std::shared_ptr<Resources> resources_;
    std::size_t limit_;
    std::deque<Request> queue_;
    std::optional<Resident> resident_;
    Handle next_ = 1, active_ = 0;
    Kind phase_ = Kind::idle;
    bool cancelled_ = false, stopping_ = false;
};
} // namespace kadan::serving
