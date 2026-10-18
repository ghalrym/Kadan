#pragma once
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <deque>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <thread>

#include "kadan/inference_request.hpp"

namespace kadan::serving {
enum class JobState {
    queued,
    preparing,
    loading,
    running,
    waiting_for_resources,
    succeeded,
    failed,
    cancelled
};

// A retry is legal only before inference begins and after physical cleanup.
struct AdmissionWait : ResourceExhausted {
    using ResourceExhausted::ResourceExhausted;
};

struct CleanupFailure : std::runtime_error {
    using std::runtime_error::runtime_error;
};

struct PublicationFailure : std::runtime_error {
    using std::runtime_error::runtime_error;
};

struct InferenceBatchItem {
    const InferenceRequest* request;
    const std::atomic_bool* cancel;
};

struct InferenceBatchResult {
    Json result;
    Json failure;
};

// Prompts and seeds vary; a denoising batch shares shape, sampling and its
// reference image. Source-latent edits keep their existing singleton path.
inline bool compatible_images(const InferenceRequest& a, const InferenceRequest& b) {
    const auto* left = std::get_if<ImageRequest>(&a.input);
    const auto* right = std::get_if<ImageRequest>(&b.input);
    if (!left || !right || a.workload != Workload::image
        || b.workload != Workload::image || a.operation != Operation::generate
        || b.operation != Operation::generate || a.model != b.model
        || a.configuration != b.configuration || left->width != right->width
        || left->height != right->height || left->steps != right->steps
        || left->reference != right->reference || left->strength != right->strength
        || left->source_width != right->source_width
        || left->source_height != right->source_height || left->edit_mask
        || right->edit_mask || bool(left->source) != bool(right->source))
        return false;
    if (left->source)
        return left->reference && left->source->png == right->source->png;
    return true;
}

// Unknown backends remain exclusive until their handles, streams and mutable
// request state are explicitly isolated from other resident engines.
enum class BackendExecution {
    exclusive,
    isolated
};

class InferenceEngine {
public:
    virtual ~InferenceEngine() = default;

    virtual BackendExecution execution_policy() const {
        return BackendExecution::exclusive;
    }

    virtual void
    load(const InferenceRequest&, const Footprint&, const std::atomic_bool&) = 0;
    virtual void
    prepare(const InferenceRequest&, const Footprint&, const std::atomic_bool&) = 0;
    virtual Json execute(const InferenceRequest&, const std::atomic_bool&) = 0;
    virtual void close() = 0;

    // A complete bound, including allocations deferred until execution. Backends
    // without a tighter plan keep their entire admission; sampled usage is not
    // enough to release promised KV, cache or workspace capacity.
    virtual Footprint execution_footprint(const Footprint& allowance) const {
        return allowance;
    }

    // Whether additional planning memory can increase cache or workspace capacity.
    virtual bool can_expand() const {
        return true;
    }

    virtual std::size_t batch_capacity(const InferenceRequest&, const Footprint&) {
        return 1;
    }

    virtual void prepare_batch(
        std::span<const InferenceBatchItem> batch, const Footprint& available
    ) {
        if (batch.size() != 1)
            throw std::logic_error("unsupported_batch");
        prepare(*batch[0].request, available, *batch[0].cancel);
    }

    virtual std::vector<InferenceBatchResult>
    execute_batch(std::span<const InferenceBatchItem> batch) {
        if (batch.size() != 1)
            throw std::logic_error("unsupported_batch");
        return {{execute(*batch[0].request, *batch[0].cancel), nullptr}};
    }
};

// Process-wide observations (such as RSS) are attributable only while one
// engine is active. A changed epoch also detects a peer that started and ended
// entirely between two observations.
class ExecutionActivity {
public:
    struct Snapshot {
        std::size_t active;
        std::uint64_t epoch;
    };

    class Scope {
    public:
        explicit Scope(std::shared_ptr<ExecutionActivity> activity)
            : activity_(std::move(activity)) {
            std::lock_guard lock(activity_->mutex_);
            ++activity_->active_;
            ++activity_->epoch_;
        }

        ~Scope() {
            std::lock_guard lock(activity_->mutex_);
            --activity_->active_;
            ++activity_->epoch_;
        }

        Scope(const Scope&) = delete;
        Scope& operator=(const Scope&) = delete;

    private:
        std::shared_ptr<ExecutionActivity> activity_;
    };

    Snapshot snapshot() const {
        std::lock_guard lock(mutex_);
        return {active_, epoch_};
    }

private:
    mutable std::mutex mutex_;
    std::size_t active_ = 0;
    std::uint64_t epoch_ = 0;
};

// A context belongs to one resident engine. The scheduler changes its identity
// only while idle; callbacks may read it throughout asynchronous execution.
struct ExecutionContext {
    std::string request_id;
    std::shared_ptr<ExecutionActivity> activity;
};

// Transport only submits/cancels. One scheduler thread owns admission, residency
// and completion; execution threads own distinct engines and isolated contexts.
class InferenceOwner {
public:
    // Factories construct control state only. Device allocation and backend
    // global-state mutation belong in load(), after compatibility admission.
    using Factory = std::function<std::unique_ptr<InferenceEngine>(
        Workload, std::shared_ptr<Resources>, std::shared_ptr<ExecutionContext>
    )>;
    using Prepare = std::function<void(InferenceRequest&, const std::atomic_bool&)>;
    using Probe = std::function<Footprint()>;
    using Clock = std::function<std::chrono::steady_clock::time_point()>;
    using Publish = std::function<void(const std::string&, JobState, const Json&)>;

    InferenceOwner(
        Footprint ceiling,
        Factory factory,
        Probe probe,
        Publish publish,
        std::size_t limit = 32,
        Prepare prepare = {},
        Clock clock = [] { return std::chrono::steady_clock::now(); },
        std::vector<std::size_t> execution_capacity = {}
    )
        : ceiling_(std::move(ceiling)),
          resources_(std::make_shared<Resources>(ceiling_)),
          factory_(std::move(factory)), probe_(std::move(probe)),
          publish_(std::move(publish)), prepare_(std::move(prepare)),
          clock_(std::move(clock)), execution_capacity_(std::move(execution_capacity)),
          limit_(limit) {
        if (!limit_ || limit_ > 1024)
            throw std::invalid_argument("queue_limit");
        if (execution_capacity_.empty())
            execution_capacity_.assign(ceiling_.size() - 1, 1);
        if (execution_capacity_.size() != ceiling_.size() - 1
            || std::any_of(
                execution_capacity_.begin(), execution_capacity_.end(), [&](auto n) {
                    return !n || n > queues_.size();
                }
            ))
            throw std::invalid_argument("execution_capacity");
    }

    ~InferenceOwner() {
        stop();
        // A poisoned transport must still keep engine/job storage alive until all
        // execution threads have acknowledged cancellation.
        for (auto& resident : residents_)
            if (resident->execution.valid())
                resident->execution.wait();
    }

    void submit(InferenceRequest request) {
        std::lock_guard guard(mutex_);
        if (stopping_ || poisoned_)
            throw std::runtime_error("worker_unavailable");
        if (request.id.empty() || request.id.size() > 128
            || request.model.size() > 4096)
            throw std::invalid_argument("request_bounds");
        if (pending_locked() >= limit_)
            throw std::runtime_error("queue_full");
        for (const auto& queue : queues_)
            for (const auto& job : queue)
                if (job->request.id == request.id)
                    throw std::runtime_error("duplicate_request");
        auto& queue = queues_.at(static_cast<std::size_t>(request.workload));
        queue.push_back(std::make_shared<Job>(std::move(request), clock_()));
        try {
            publish_(queue.back()->request.id, JobState::queued, "");
        } catch (...) {
            queue.pop_back();
            throw PublicationFailure("acceptance_publication_failed");
        }
    }

    bool cancel(const std::string& id) {
        std::lock_guard guard(mutex_);
        for (const auto& queue : queues_)
            for (const auto& job : queue)
                if (job->request.id == id) {
                    job->cancel.store(true);
                    return true;
                }
        return false;
    }

    void stop() {
        std::lock_guard guard(mutex_);
        stopping_ = true;
        for (const auto& queue : queues_)
            for (const auto& job : queue)
                job->cancel.store(true);
    }

    // Nonblocking execution: each call completes one ready batch or attempts one
    // queue head. Round robin bounds bypass by other workloads; FIFO within each
    // workload retains image cohort ordering and cancellation semantics.
    bool advance() {
        if (poisoned_.load())
            throw CleanupFailure("cleanup_unconfirmed");
        for (auto& resident : residents_)
            if (resident->execution.valid()
                && resident->execution.wait_for(std::chrono::seconds(0))
                       == std::future_status::ready) {
                complete(*resident);
                attempted_.fill(false);
                return true;
            }
        std::shared_ptr<Job> job;
        if (waiting_ && bypasses_ >= queues_.size()) {
            // Let a feasible wide request obtain all of its resources after a
            // bounded number of bypasses by smaller requests.
            if (executing())
                return false;
            job = waiting_;
        } else {
            std::lock_guard guard(mutex_);
            for (int pass = 0; pass < 2 && !job; ++pass) {
                for (std::size_t offset = 0; offset < queues_.size(); ++offset) {
                    const auto index = (next_queue_ + offset) % queues_.size();
                    if (!attempted_[index] && !queues_[index].empty()
                        && !queues_[index].front()->executing) {
                        job = queues_[index].front();
                        attempted_[index] = true;
                        break;
                    }
                }
                if (!job)
                    attempted_.fill(false);
            }
        }
        if (!job)
            return false;
        prepare_job(job);
        if (job->cancel.load()) {
            finish(job, JobState::cancelled, "");
            return true;
        }
        if (!job->error.is_null()) {
            finish(job, JobState::failed, job->error);
            return true;
        }
        const auto& request = job->request;
        if (request.operation == Operation::download
            || request.operation == Operation::configure) {
            finish(job, JobState::succeeded, Json::object());
            return true;
        }
        if (request.operation == Operation::unload) {
            for (const auto& resident : residents_)
                if (resident->workload == request.workload
                    && resident->execution.valid())
                    return false;
            for (auto& resident : residents_)
                if (resident->workload == request.workload)
                    cleanup(*resident);
            erase_closed();
            finish(job, JobState::succeeded, Json::object());
            return true;
        }
        ResidentEngine* resident = nullptr;
        std::vector<std::shared_ptr<Job>> jobs{job};
        try {
            for (auto& candidate : residents_)
                if (candidate->engine && candidate->model == request.model
                    && candidate->workload == request.workload
                    && candidate->configuration == request.configuration) {
                    resident = candidate.get();
                    break;
                }
            if (resident
                && (resident->execution.valid()
                    || !execution_available(
                        resident->engine->execution_policy(), resident->devices
                    ))) {
                wait_for(job);
                return false;
            }
            if (resident && request.operation == Operation::load) {
                cleanup(*resident);
                resident = nullptr;
                erase_closed();
            }
            if (resident && should_replan(*resident, *job)) {
                // Device/cache plans belong to the engine. Rebuild only an idle
                // constrained plan; changing its ledger cannot enlarge banks.
                cleanup(*resident);
                resident = nullptr;
                erase_closed();
            }
            if (resident) {
                admit(*resident);
                resident->activity =
                    std::make_unique<ExecutionActivity::Scope>(activity_);
            }
            if (!resident) {
                resident = load(job);
                if (!resident) {
                    wait_for(job);
                    return false;
                }
            }
            resident->context->request_id = request.id;
            resident->last_used = ++use_order_;
            auto available = availability(*resident);
            const auto capacity = resident->engine->batch_capacity(request, available);
            bool blocked = false;
            std::size_t images = image_count(request);
            if (capacity > images && compatible_images(request, request)) {
                std::vector<std::shared_ptr<Job>> candidates;
                {
                    std::lock_guard guard(mutex_);
                    const auto& queue =
                        queues_.at(static_cast<std::size_t>(request.workload));
                    candidates.assign(std::next(queue.begin()), queue.end());
                }
                for (const auto& candidate : candidates) {
                    if (candidate->cancel.load()
                        || candidate->request.operation != Operation::generate) {
                        blocked = true;
                        break;
                    }
                    prepare_job(candidate);
                    if (!candidate->error.is_null() || candidate->cancel.load()
                        || !compatible_images(request, candidate->request)
                        || image_count(candidate->request) > capacity - images) {
                        blocked = true;
                        break;
                    }
                    jobs.push_back(candidate);
                    images += image_count(candidate->request);
                    if (images == capacity)
                        break;
                }
                const auto target = std::min<std::size_t>(
                    capacity,
                    std::get<ImageRequest>(request.input)
                        .batch_collection_target.value_or(capacity)
                );
                if (!blocked && images < target
                    && clock_() - job->submitted < std::chrono::seconds(10)) {
                    resident->resources->idle();
                    resident->activity.reset();
                    return false;
                }
            }
            std::vector<InferenceBatchItem> batch;
            batch.reserve(jobs.size());
            for (const auto& current : jobs)
                batch.push_back({&current->request, &current->cancel});
            for (;;) {
                try {
                    resident->engine->prepare_batch(batch, available);
                    break;
                } catch (const ResourceExhausted&) {
                    if (jobs.size() == 1)
                        throw;
                    jobs.pop_back();
                    batch.pop_back();
                }
            }
            if (std::all_of(jobs.begin(), jobs.end(), [](const auto& current) {
                    return current->cancel.load();
                })) {
                cleanup(*resident);
                finish_batch(jobs, "");
                return true;
            }
            const auto bound =
                resident->engine->execution_footprint(resident->allowance);
            const auto charged = resident->resources->snapshot().used;
            if (bound.size() != charged.size())
                throw std::runtime_error("execution_footprint_shape");
            for (std::size_t i = 0; i < bound.size(); ++i)
                if (bound[i] < charged[i] || bound[i] > resident->allowance[i])
                    throw std::runtime_error("execution_footprint_bounds");
            resident->resources->admit(bound);
            resident->allowance = bound;
            std::erase_if(resident->devices, [&](auto device) {
                return bound[device] == 0;
            });
            resident->jobs = jobs;
            for (const auto& current : jobs) {
                current->executing = true;
                if (!current->cancel.load())
                    publish_(current->request.id, JobState::running, "");
            }
            auto* engine = resident->engine.get();
            if (waiting_ == job) {
                waiting_.reset();
                bypasses_ = 0;
            } else if (waiting_)
                ++bypasses_;
            next_queue_ =
                (static_cast<std::size_t>(request.workload) + 1) % queues_.size();
            attempted_.fill(false);
            resident->execution =
                std::async(std::launch::async, [engine, batch = std::move(batch)] {
                    return engine->execute_batch(batch);
                });
            return true;
        } catch (const CleanupFailure&) {
            throw;
        } catch (const PublicationFailure&) {
            throw;
        } catch (const ResourceExhausted&) {
            if (resident)
                cleanup(*resident);
            const bool expanded = !job->full_budget;
            job->full_budget = true;
            // A fair share that is too small is not evidence that an idle model
            // must go. Try the complete remaining envelope before reclaiming it.
            const bool reclaimed = !expanded && evict_idle();
            if (!reclaimed && !executing() && waiting_ == job
                && bypasses_ >= queues_.size()) {
                waiting_.reset();
                bypasses_ = 0;
            } else
                wait_for(job);
            publish_(request.id, JobState::waiting_for_resources, "");
            return false;
        } catch (const RequestError& error) {
            if (resident)
                cleanup(*resident);
            finish_batch(
                jobs, {{"error", error.what()}, {"status_code", error.status}}
            );
            return true;
        } catch (const std::exception& error) {
            if (resident)
                cleanup(*resident);
            finish_batch(jobs, error.what());
            return true;
        }
    }

    void shutdown() {
        stop();
        while (pending()) {
            if (!advance())
                std::this_thread::yield();
        }
        for (auto& resident : residents_)
            cleanup(*resident);
        erase_closed();
    }

    std::size_t pending() const {
        std::lock_guard guard(mutex_);
        return pending_locked();
    }

    bool executing() const {
        return std::any_of(
            residents_.begin(), residents_.end(), [](const auto& resident) {
                return resident->execution.valid();
            }
        );
    }

    Snapshot snapshot() const {
        return resources_->snapshot();
    }

    std::vector<std::string> resident_models() const {
        std::vector<std::string> models;
        for (const auto& resident : residents_)
            if (resident->engine && resident->loaded)
                models.push_back(resident->model);
        return models;
    }

private:
    struct Job {
        Job(InferenceRequest q, std::chrono::steady_clock::time_point time)
            : request(std::move(q)), submitted(time) {
        }

        InferenceRequest request;
        Json error;
        std::chrono::steady_clock::time_point submitted;
        std::atomic_bool cancel{false};
        bool executing = false;
        bool full_budget = false;
    };

    struct ResidentEngine {
        std::string model;
        Workload workload;
        EngineConfiguration configuration;
        std::shared_ptr<Resources> resources;
        Footprint allowance, planned_allowance;
        std::shared_ptr<ExecutionContext> context =
            std::make_shared<ExecutionContext>();
        std::unique_ptr<InferenceEngine> engine;
        std::vector<std::size_t> devices;
        std::vector<std::shared_ptr<Job>> jobs;
        std::future<std::vector<InferenceBatchResult>> execution;
        std::unique_ptr<ExecutionActivity::Scope> activity;
        std::uint64_t last_used = 0;
        bool loaded = false, constrained = false;
    };

    Footprint probe() {
        auto free = probe_();
        if (free.size() != ceiling_.size())
            throw std::runtime_error("probe_shape");
        for (std::size_t i = 0; i < free.size(); ++i)
            free[i] = std::min(free[i], ceiling_[i]);
        // A stable epoch bounds all promises, including lazy workspaces. Physical
        // free memory clips new promises but never revokes an active allowance.
        if (resources_->empty())
            resources_->reset_capacity(free);
        return free;
    }

    Footprint availability(ResidentEngine& resident) {
        auto free = probe();
        const auto state = resident.resources->snapshot();
        for (std::size_t i = 0; i < free.size(); ++i)
            free[i] = std::min(free[i], state.capacity[i] - state.used[i]);
        return free;
    }

    void admit(ResidentEngine& resident) {
        for (;;) {
            try {
                const auto free = probe();
                const auto state = resident.resources->snapshot();
                for (std::size_t i = 0; i < free.size(); ++i)
                    if (resident.allowance[i] > state.used[i]
                        && resident.allowance[i] - state.used[i] > free[i])
                        throw AdmissionWait("physical_headroom");
                resident.resources->admit(resident.allowance);
                return;
            } catch (const ResourceExhausted&) {
                // Keep the requested model when retiring an idle peer suffices.
                if (!evict_idle(&resident))
                    throw;
            }
        }
    }

    std::size_t device_executions(std::size_t device) const {
        return std::count_if(
            residents_.begin(), residents_.end(), [&](const auto& resident) {
                return resident->execution.valid()
                       && std::find(
                              resident->devices.begin(), resident->devices.end(), device
                          ) != resident->devices.end();
            }
        );
    }

    bool execution_available(
        BackendExecution policy, const std::vector<std::size_t>& devices
    ) const {
        for (const auto& resident : residents_)
            if (resident->execution.valid()
                && (policy == BackendExecution::exclusive
                    || resident->engine->execution_policy()
                           == BackendExecution::exclusive))
                return false;
        for (auto device : devices)
            if (device_executions(device) >= execution_capacity_.at(device - 1))
                return false;
        return true;
    }

    struct Placement {
        Footprint allowance;
        std::vector<std::size_t> devices;
        bool available = true, constrained = false, memory_blocked = false;
    };

    static bool uses_gpu(Workload workload) {
        return workload != Workload::decision && workload != Workload::formatting;
    }

    Placement placement(const Job& job, const Footprint& reclaimable = {}) {
        auto free = probe();
        const auto global = resources_->snapshot();
        for (std::size_t i = 0; i < free.size(); ++i) {
            const auto own = reclaimable.empty() ? 0 : reclaimable[i];
            free[i] = std::min(free[i], global.capacity[i]);
            free[i] += std::min(own, global.capacity[i] - free[i]);
            free[i] = std::min(free[i], global.capacity[i] - global.used[i] + own);
        }
        std::size_t demand = 0, gpu_demand = 0;
        {
            std::lock_guard guard(mutex_);
            for (const auto& queue : queues_) {
                if (queue.empty() || queue.front()->executing)
                    continue;
                const auto& next = queue.front();
                if (next->cancel.load()
                    || (next->request.operation != Operation::generate
                        && next->request.operation != Operation::completion
                        && next->request.operation != Operation::load))
                    continue;
                ++demand;
                gpu_demand += uses_gpu(next->request.workload);
            }
        }
        demand = job.full_budget ? 1 : std::max<std::size_t>(1, demand);
        gpu_demand = job.full_budget ? 1 : std::max<std::size_t>(1, gpu_demand);
        Placement result{Footprint(free.size()), {}};
        if (!execution_available(BackendExecution::isolated, {})) {
            result.available = false;
            return result;
        }
        for (std::size_t i = 1; uses_gpu(job.request.workload) && i < free.size(); ++i)
            if (free[i] && device_executions(i) < execution_capacity_[i - 1])
                result.devices.push_back(i);
        if (uses_gpu(job.request.workload) && free.size() > 1
            && result.devices.empty()) {
            result.available = false;
            for (std::size_t i = 1; i < free.size(); ++i)
                result.memory_blocked |=
                    device_executions(i) < execution_capacity_[i - 1];
            return result;
        }
        std::stable_sort(
            result.devices.begin(), result.devices.end(), [&](auto a, auto b) {
                return free[a] > free[b];
            }
        );
        const auto device_count = result.devices.size();
        const auto shares =
            device_count ? (gpu_demand + device_count - 1) / device_count : 1;
        if (device_count)
            result.devices.resize(std::max<std::size_t>(1, device_count / gpu_demand));
        // Divide only among queued executable workloads. Execution slots limit
        // compute overlap; they are not phantom consumers of memory.
        result.allowance[0] = free[0] / demand;
        for (auto device : result.devices)
            result.allowance[device] = free[device] / shares;
        for (std::size_t i = 0; i < free.size(); ++i)
            if ((i == 0 || uses_gpu(job.request.workload))
                && result.allowance[i] < global.capacity[i])
                result.constrained = true;
        return result;
    }

    bool should_replan(const ResidentEngine& resident, const Job& job) {
        if (!resident.constrained || !resident.engine->can_expand())
            return false;
        const auto desired = placement(job, resident.resources->snapshot().used);
        for (std::size_t i = 0; i < desired.allowance.size(); ++i)
            if (desired.allowance[i] > resident.planned_allowance[i])
                return true;
        return false;
    }

    ResidentEngine* load(const std::shared_ptr<Job>& job) {
        const auto& request = job->request;
        erase_closed();
        if (residents_.size() >= limit_)
            evict_idle();
        auto plan = placement(*job);
        if (!plan.available) {
            if (plan.memory_blocked)
                evict_idle();
            return nullptr;
        }
        auto entry = std::make_unique<ResidentEngine>();
        entry->model = request.model;
        entry->workload = request.workload;
        entry->configuration = request.configuration;
        entry->devices = std::move(plan.devices);
        const auto& allowance = plan.allowance;
        entry->resources = resources_->scope(allowance);
        entry->allowance = entry->planned_allowance = allowance;
        entry->constrained = plan.constrained;
        entry->context->request_id = request.id;
        entry->context->activity = activity_;
        entry->last_used = ++use_order_;
        auto* resident = entry.get();
        residents_.push_back(std::move(entry));
        try {
            resident->activity = std::make_unique<ExecutionActivity::Scope>(activity_);
            resident->engine =
                factory_(request.workload, resident->resources, resident->context);
            if (!resident->engine)
                throw std::runtime_error("unsupported_workload");
            if (!execution_available(
                    resident->engine->execution_policy(), resident->devices
                )) {
                cleanup(*resident);
                erase_closed();
                return nullptr;
            }
            publish_(request.id, JobState::loading, "");
            resident->engine->load(request, allowance, job->cancel);
            resident->loaded = true;
            return resident;
        } catch (...) {
            cleanup(*resident);
            erase_closed();
            throw;
        }
    }

    void complete(ResidentEngine& resident) {
        const auto jobs = resident.jobs;
        const auto& request = jobs.front()->request;
        try {
            auto results = resident.execution.get();
            if (results.size() != jobs.size())
                throw std::runtime_error("batch_result_count");
            if (std::any_of(
                    jobs.begin(),
                    jobs.end(),
                    [](const auto& job) { return job->cancel.load(); }
                )
                || std::any_of(results.begin(), results.end(), [](const auto& result) {
                       return !result.failure.is_null();
                   }))
                cleanup(resident);
            else if (compatible_images(request, request)) {
                try {
                    const auto free = availability(resident);
                    const auto next = std::clamp<std::size_t>(
                        resident.engine->batch_capacity(request, free),
                        1,
                        std::min<std::size_t>(limit_, 32)
                    );
                    for (auto& result : results)
                        if (result.result.is_object())
                            result.result["next_batch_capacity"] = next;
                } catch (const std::exception&) {
                    // Advice cannot discard completed images.
                }
                if (std::any_of(jobs.begin(), jobs.end(), [](const auto& job) {
                        return job->cancel.load();
                    })) {
                    for (auto& result : results)
                        if (result.result.is_object())
                            result.result.erase("next_batch_capacity");
                    cleanup(resident);
                }
            }
            resident.resources->idle();
            for (std::size_t i = 0; i < jobs.size(); ++i) {
                if (jobs[i]->cancel.load())
                    finish(jobs[i], JobState::cancelled, "");
                else if (!results[i].failure.is_null())
                    finish(jobs[i], JobState::failed, results[i].failure);
                else
                    finish(jobs[i], JobState::succeeded, results[i].result);
            }
        } catch (const CleanupFailure&) {
            throw;
        } catch (const PublicationFailure&) {
            throw;
        } catch (const RequestError& error) {
            cleanup(resident);
            finish_batch(
                jobs, {{"error", error.what()}, {"status_code", error.status}}
            );
        } catch (const ResourceExhausted&) {
            cleanup(resident);
            finish_batch(jobs, "admission_after_execution");
        } catch (const std::exception& error) {
            cleanup(resident);
            finish_batch(jobs, error.what());
        }
        resident.jobs.clear();
        resident.activity.reset();
    }

    void wait_for(const std::shared_ptr<Job>& job) {
        if (!waiting_) {
            waiting_ = job;
            bypasses_ = 0;
        }
    }

    bool evict_idle(const ResidentEngine* keep = nullptr) {
        ResidentEngine* oldest = nullptr;
        for (auto& resident : residents_)
            if (resident.get() != keep && resident->engine
                && !resident->execution.valid()
                && (!oldest || resident->last_used < oldest->last_used))
                oldest = resident.get();
        if (oldest)
            cleanup(*oldest);
        erase_closed();
        return oldest != nullptr;
    }

    void erase_closed() {
        std::erase_if(residents_, [](const auto& resident) {
            return !resident->engine && !resident->execution.valid();
        });
    }

    void cleanup(ResidentEngine& resident) {
        if (resident.execution.valid())
            throw CleanupFailure("engine_in_use");
        try {
            if (!resident.activity)
                resident.activity =
                    std::make_unique<ExecutionActivity::Scope>(activity_);
            if (resident.engine)
                resident.engine->close();
            resident.engine.reset();
            resident.loaded = false;
            const auto state = resident.resources->snapshot();
            if (state.residents
                || std::any_of(state.used.begin(), state.used.end(), [](auto n) {
                       return n != 0;
                   }))
                throw std::runtime_error("cleanup_unconfirmed");
            resident.resources->idle();
            resident.activity.reset();
        } catch (...) {
            poisoned_.store(true);
            stop();
            throw CleanupFailure("cleanup_unconfirmed");
        }
    }

    void prepare_job(const std::shared_ptr<Job>& job) {
        if (job->request.prepared || job->cancel.load())
            return;
        publish_(job->request.id, JobState::preparing, "");
        ExecutionActivity::Scope activity(activity_);
        try {
            if (!prepare_)
                throw std::runtime_error("preparation_unavailable");
            prepare_(job->request, job->cancel);
        } catch (const RequestError& error) {
            job->error = {{"error", error.what()}, {"status_code", error.status}};
        } catch (const std::exception& error) {
            job->error = error.what();
        }
        job->request.prepared = true;
    }

    void
    finish_batch(const std::vector<std::shared_ptr<Job>>& jobs, const Json& error) {
        for (const auto& job : jobs)
            finish(
                job, job->cancel.load() ? JobState::cancelled : JobState::failed, error
            );
    }

    void finish(const std::shared_ptr<Job>& job, JobState state, const Json& result) {
        {
            std::lock_guard guard(mutex_);
            auto& queue = queues_.at(static_cast<std::size_t>(job->request.workload));
            if (queue.empty() || queue.front() != job)
                throw std::runtime_error("fifo_ownership");
            queue.pop_front();
            if (waiting_ == job) {
                waiting_.reset();
                bypasses_ = 0;
            }
            if (!job->executing) {
                next_queue_ = (static_cast<std::size_t>(job->request.workload) + 1)
                              % queues_.size();
                attempted_.fill(false);
            }
        }
        try {
            if (state == JobState::failed && !result.is_object())
                publish_(
                    job->request.id, state, {{"error", result}, {"status_code", 502}}
                );
            else
                publish_(job->request.id, state, result);
        } catch (...) {
            poisoned_.store(true);
            stop();
            throw PublicationFailure("terminal_publication_failed");
        }
    }

    std::size_t pending_locked() const {
        std::size_t count = 0;
        for (const auto& queue : queues_)
            count += queue.size();
        return count;
    }

    static std::size_t image_count(const InferenceRequest& request) {
        const auto* input = std::get_if<ImageRequest>(&request.input);
        return input ? input->count : 1;
    }

    Footprint ceiling_;
    std::shared_ptr<Resources> resources_;
    Factory factory_;
    Probe probe_;
    Publish publish_;
    Prepare prepare_;
    Clock clock_;
    std::vector<std::size_t> execution_capacity_;
    std::shared_ptr<ExecutionActivity> activity_ =
        std::make_shared<ExecutionActivity>();
    std::size_t limit_, next_queue_ = 0;
    mutable std::mutex mutex_;
    std::array<std::deque<std::shared_ptr<Job>>, 7> queues_;
    std::array<bool, 7> attempted_{};
    std::vector<std::unique_ptr<ResidentEngine>> residents_;
    std::uint64_t use_order_ = 0;
    std::shared_ptr<Job> waiting_;
    std::size_t bypasses_ = 0;
    bool stopping_ = false;
    std::atomic_bool poisoned_{false};
};
}  // namespace kadan::serving
