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

class InferenceEngine {
public:
    virtual ~InferenceEngine() = default;
    virtual void
    load(const InferenceRequest&, const Footprint&, const std::atomic_bool&) = 0;
    virtual void
    prepare(const InferenceRequest&, const Footprint&, const std::atomic_bool&) = 0;
    virtual Json execute(const InferenceRequest&, const std::atomic_bool&) = 0;
    virtual void close() = 0;

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

// A context belongs to one resident engine. The scheduler changes its identity
// only while idle; callbacks may read it throughout asynchronous execution.
struct ExecutionContext {
    std::string request_id;
};

// Transport only submits/cancels. One scheduler thread owns admission, residency
// and completion; execution threads own distinct engines and disjoint GPUs.
class InferenceOwner {
public:
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
        Clock clock = [] { return std::chrono::steady_clock::now(); }
    )
        : ceiling_(std::move(ceiling)),
          resources_(std::make_shared<Resources>(ceiling_)),
          factory_(std::move(factory)), probe_(std::move(probe)),
          publish_(std::move(publish)), prepare_(std::move(prepare)),
          clock_(std::move(clock)), limit_(limit) {
        if (!limit_ || limit_ > 1024)
            throw std::invalid_argument("queue_limit");
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
        // Image calibration includes process-wide host RSS and ATen settings.
        // Preserve that measurement window across preparation and execution.
        const bool image_active =
            std::any_of(residents_.begin(), residents_.end(), [](const auto& resident) {
                return resident->workload == Workload::image
                       && resident->execution.valid();
            });
        if (!job->cancel.load()
            && (image_active
                || (job->request.workload == Workload::image && executing()))) {
            wait_for(job);
            return false;
        }
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
                && (resident->execution.valid() || devices_busy(resident->devices))) {
                wait_for(job);
                return false;
            }
            if (resident && request.operation == Operation::load) {
                cleanup(*resident);
                resident = nullptr;
                erase_closed();
            }
            if (resident)
                admit(*resident);
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
            if (!job->full_device && request.workload != Workload::image)
                job->full_device = true;
            else if (!job->wide && ceiling_.size() > 2)
                job->wide = true;
            else
                job->full_host = true;
            const bool reclaimed = evict_idle();
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
        bool full_device = false, wide = false, full_host = false;
    };

    struct ResidentEngine {
        std::string model;
        Workload workload;
        EngineConfiguration configuration;
        std::shared_ptr<Resources> resources;
        Footprint allowance;
        std::shared_ptr<ExecutionContext> context =
            std::make_shared<ExecutionContext>();
        std::unique_ptr<InferenceEngine> engine;
        std::vector<std::size_t> devices;
        std::vector<std::shared_ptr<Job>> jobs;
        std::future<std::vector<InferenceBatchResult>> execution;
        std::uint64_t last_used = 0;
        bool loaded = false;
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

    bool devices_busy(const std::vector<std::size_t>& devices) const {
        for (const auto& resident : residents_)
            if (resident->execution.valid())
                for (auto device : devices)
                    if (std::find(
                            resident->devices.begin(), resident->devices.end(), device
                        )
                        != resident->devices.end())
                        return true;
        return false;
    }

    ResidentEngine* load(const std::shared_ptr<Job>& job) {
        const auto& request = job->request;
        erase_closed();
        if (residents_.size() >= limit_)
            evict_idle();
        auto free = probe();
        const auto global = resources_->snapshot();
        for (std::size_t i = 0; i < free.size(); ++i)
            free[i] = std::min(free[i], global.capacity[i] - global.used[i]);
        if ((job->wide || job->full_host) && executing())
            return nullptr;
        const bool uses_gpu = request.workload != Workload::decision
                              && request.workload != Workload::formatting;
        std::vector<std::size_t> devices;
        for (std::size_t i = 1; uses_gpu && i < free.size(); ++i)
            if (free[i] && !devices_busy({i}))
                devices.push_back(i);
        if (uses_gpu && free.size() > 1 && devices.empty()) {
            evict_idle();
            return nullptr;
        }
        std::stable_sort(devices.begin(), devices.end(), [&](auto a, auto b) {
            return free[a] > free[b];
        });
        auto entry = std::make_unique<ResidentEngine>();
        entry->model = request.model;
        entry->workload = request.workload;
        entry->configuration = request.configuration;
        entry->devices = devices;
        // Preserve the image engine's existing multi-device batch planner.
        // Other workloads try one GPU before widening on load/prepare pressure.
        if (!job->wide && request.workload != Workload::image
            && entry->devices.size() > 1)
            entry->devices.resize(1);
        Footprint allowance(ceiling_.size());
        const auto slots = std::max<std::size_t>(2, ceiling_.size() - 1);
        const auto share = global.capacity[0] / slots
                           * std::max<std::size_t>(1, entry->devices.size());
        allowance[0] = job->full_host ? free[0] : std::min(free[0], share);
        for (auto device : entry->devices)
            allowance[device] =
                job->full_device || request.workload == Workload::image
                    ? free[device]
                    : std::min(free[device], global.capacity[device] / 2);
        entry->resources = resources_->scope(allowance);
        entry->allowance = allowance;
        entry->context->request_id = request.id;
        entry->last_used = ++use_order_;
        auto* resident = entry.get();
        residents_.push_back(std::move(entry));
        try {
            resident->engine =
                factory_(request.workload, resident->resources, resident->context);
            if (!resident->engine)
                throw std::runtime_error("unsupported_workload");
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
