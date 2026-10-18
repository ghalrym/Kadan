#pragma once
#include <algorithm>
#include <atomic>
#include <chrono>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <utility>

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

// Exactly one execution thread calls advance(). Transport threads may only
// submit, cancel or stop. Engines receive the same ledger; no modality process
// is involved.
class InferenceOwner {
public:
    using Factory = std::function<
        std::unique_ptr<InferenceEngine>(Workload, std::shared_ptr<Resources>)>;
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
        bool loaded = true
    )
        : ceiling_(std::move(ceiling)),
          resources_(std::make_shared<Resources>(ceiling_)),
          factory_(std::move(factory)), probe_(std::move(probe)),
          publish_(std::move(publish)), prepare_(std::move(prepare)),
          clock_(std::move(clock)), limit_(limit), loaded_(loaded) {
        if (!limit_ || limit_ > 1024)
            throw std::invalid_argument("queue_limit");
    }

    // The API grants a budget; the owner only pulls jobs while loaded. Engines
    // still load lazily from the queue head, bounded by this ceiling. Shrinking
    // a budget is refused while a job runs, because the API frees that memory
    // in its own ledger as soon as the call returns.
    bool load(Footprint budget) {
        std::lock_guard guard(mutex_);
        if (budget.size() != ceiling_.size())
            throw std::invalid_argument("budget_shape");
        bool shrinks = false;
        for (std::size_t i = 0; i < budget.size(); ++i)
            shrinks = shrinks || budget[i] < ceiling_[i];
        if (shrinks && executing_)
            return false;
        // A resident engine is rebuilt inside a changed budget before its next
        // job; growth waits for the running job instead of being refused.
        if (budget != ceiling_ && !model_.empty())
            release_requested_ = true;
        ceiling_ = std::move(budget);
        loaded_ = true;
        return true;
    }

    // Stops pulling jobs and releases the engine on the execution thread. The
    // control listener reports "offloaded" once memory is physically free.
    bool offload() {
        std::lock_guard guard(mutex_);
        if (executing_)
            return false;
        loaded_ = false;
        release_requested_ = true;
        std::fill(ceiling_.begin(), ceiling_.end(), 0);
        return true;
    }

    bool busy() const {
        std::lock_guard guard(mutex_);
        return executing_;
    }

    struct Status {
        bool loaded, busy;
        std::size_t queued;
        std::string model;
        Footprint budget;
        Snapshot memory;
    };

    Status status() const {
        std::lock_guard guard(mutex_);
        return {
            loaded_, executing_, queue_.size(), model_, ceiling_, resources_->snapshot()
        };
    }

    void set_control_listener(std::function<void(const char*)> listener) {
        std::lock_guard guard(mutex_);
        control_ = std::move(listener);
    }

    void submit(InferenceRequest request) {
        std::lock_guard guard(mutex_);
        if (stopping_ || poisoned_)
            throw std::runtime_error("worker_unavailable");
        if (request.id.empty() || request.id.size() > 128
            || request.model.size() > 4096)
            throw std::invalid_argument("request_bounds");
        if (queue_.size() >= limit_)
            throw std::runtime_error("queue_full");
        for (const auto& job : queue_)
            if (job->request.id == request.id)
                throw std::runtime_error("duplicate_request");
        queue_.push_back(std::make_shared<Job>(std::move(request), clock_()));
        // Keep acceptance and admission under the same lock: advance() must not
        // publish preparation or completion before the queued acknowledgement.
        try {
            publish_(queue_.back()->request.id, JobState::queued, "");
        } catch (...) {
            queue_.pop_back();
            throw PublicationFailure("acceptance_publication_failed");
        }
    }

    bool cancel(const std::string& id) {
        std::lock_guard guard(mutex_);
        for (const auto& job : queue_)
            if (job->request.id == id) {
                job->cancel.store(true);
                return true;
            }
        return false;
    }

    void stop() {
        std::lock_guard guard(mutex_);
        stopping_ = true;
        for (const auto& job : queue_)
            job->cancel.store(true);
    }

    // Returns false for no work or a recoverable physical-admission wait. A
    // waiting head remains at the front; the transport cannot overtake it with
    // another job.
    bool advance() {
        std::shared_ptr<Job> job;
        bool release = false;
        std::function<void(const char*)> control;
        {
            std::lock_guard guard(mutex_);
            if (poisoned_)
                throw std::runtime_error("cleanup_unconfirmed");
            release = std::exchange(release_requested_, false);
            control = control_;
            // An offloaded owner still retires a cancelled head so callers are
            // not left waiting for a load that may never come.
            if (!release
                && (queue_.empty() || (!loaded_ && !queue_.front()->cancel.load())))
                return false;
            if (!release) {
                job = queue_.front();
                executing_ = true;
            }
        }
        if (release) {
            cleanup();
            {
                std::lock_guard guard(mutex_);
                model_.clear();
            }
            if (control)
                control(loaded() ? "resized" : "offloaded");
            return true;
        }

        struct Executing {
            InferenceOwner& owner;

            ~Executing() {
                std::lock_guard guard(owner.mutex_);
                owner.executing_ = false;
                owner.model_ = owner.resident_model_;
            }
        } executing{*this};

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
        bool execution_started = false;
        std::vector<std::shared_ptr<Job>> jobs{job};
        try {
            if (request.operation == Operation::download
                || request.operation == Operation::configure) {
                finish(job, JobState::succeeded, Json::object());
                return true;
            }
            if (request.operation == Operation::unload) {
                cleanup();
                finish(job, JobState::succeeded, Json::object());
                return true;
            }
            if (!engine_ || resident_model_ != request.model
                || resident_workload_ != request.workload
                || resident_configuration_ != request.configuration
                || request.operation == Operation::load) {
                cleanup();  // Physical release precedes the fresh probe: never add
                            // logical reclaim credit.
                auto free = bounded_probe();
                resources_->reset_capacity(free);
                engine_ = factory_(request.workload, resources_);
                if (!engine_)
                    throw std::runtime_error("unsupported_workload");
                publish_(request.id, JobState::loading, "");
                engine_->load(request, free, job->cancel);
                resident_model_ = request.model;
                resident_workload_ = request.workload;
                resident_configuration_ = request.configuration;
                {
                    std::lock_guard guard(mutex_);
                    model_ = resident_model_;
                }
            }
            if (job->cancel.load()) {
                cleanup();
                finish(job, JobState::cancelled, "");
                return true;
            }
            // Request-local scratch and context admission is required even when
            // the same model stays resident. No inference runs in prepare().
            auto available = bounded_probe();
            resources_->refresh_available(available);
            const auto capacity = engine_->batch_capacity(request, available);
            bool blocked = false;
            std::size_t images = image_count(request);
            if (capacity > images && compatible_images(request, request)) {
                // Snapshot candidates without holding the transport lock during
                // preparation. Jobs remain in the FIFO until terminal delivery.
                std::vector<std::shared_ptr<Job>> candidates;
                {
                    std::lock_guard guard(mutex_);
                    candidates.assign(std::next(queue_.begin()), queue_.end());
                }
                for (const auto& candidate : candidates) {
                    if (candidate->cancel.load()
                        || candidate->request.workload != Workload::image
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
                // A producer may have fewer images to supply than worker capacity.
                // Scan the complete compatible FIFO prefix first: this hint only
                // shortens waiting, never limits an already queued batch.
                const auto target = std::min<std::size_t>(
                    capacity,
                    std::get<ImageRequest>(request.input)
                        .batch_collection_target.value_or(capacity)
                );
                if (!blocked && images < target
                    && clock_() - job->submitted < std::chrono::seconds(10))
                    return false;
            }
            std::vector<InferenceBatchItem> batch;
            batch.reserve(jobs.size());
            for (const auto& current : jobs)
                batch.push_back({&current->request, &current->cancel});
            for (;;) {
                try {
                    engine_->prepare_batch(batch, available);
                    break;
                } catch (const ResourceExhausted&) {
                    if (jobs.size() == 1)
                        throw;
                    // A later prompt or the combined host buffers may not fit.
                    // Retain that suffix in the FIFO and admit the oldest prefix.
                    jobs.pop_back();
                    batch.pop_back();
                }
            }
            if (std::all_of(jobs.begin(), jobs.end(), [](const auto& current) {
                    return current->cancel.load();
                })) {
                cleanup();
                finish_batch(jobs, "");
                return true;
            }
            for (const auto& current : jobs)
                if (!current->cancel.load())
                    publish_(current->request.id, JobState::running, "");
            execution_started = true;
            auto results = engine_->execute_batch(batch);
            if (results.size() != jobs.size())
                throw std::runtime_error("batch_result_count");
            if (std::any_of(
                    jobs.begin(),
                    jobs.end(),
                    [](const auto& current) { return current->cancel.load(); }
                )
                || std::any_of(results.begin(), results.end(), [](const auto& result) {
                       return !result.failure.is_null();
                   }))
                cleanup();
            else if (compatible_images(request, request)) {
                // execute_batch has recorded successful measurements and released
                // request scratch. This fresh projection is a producer hint, not
                // a reservation: the next FIFO batch always repeats admission.
                try {
                    auto free = bounded_probe();
                    resources_->refresh_available(free);
                    const auto next = std::clamp<std::size_t>(
                        engine_->batch_capacity(request, free),
                        1,
                        std::min<std::size_t>(limit_, 32)
                    );
                    for (auto& result : results)
                        if (result.result.is_object())
                            result.result["next_batch_capacity"] = next;
                } catch (const std::exception&) {
                    // A failed advisory probe cannot discard completed images.
                }
                // Transport cancellation may arrive during that extra probe or
                // projection. Discard advice and release the resident model before
                // publishing cancellation, just as on the normal execution path.
                if (std::any_of(jobs.begin(), jobs.end(), [](const auto& current) {
                        return current->cancel.load();
                    })) {
                    for (auto& result : results)
                        if (result.result.is_object())
                            result.result.erase("next_batch_capacity");
                    cleanup();  // Cleanup failures must escape the advisory catch.
                }
            }
            for (std::size_t index = 0; index < jobs.size(); ++index) {
                const auto& current = jobs[index];
                if (current->cancel.load())
                    finish(current, JobState::cancelled, "");
                else if (!results[index].failure.is_null())
                    finish(current, JobState::failed, results[index].failure);
                else
                    finish(current, JobState::succeeded, results[index].result);
            }
            return true;
        } catch (const CleanupFailure&) {
            throw;
        } catch (const PublicationFailure&) {
            throw;
        } catch (const ResourceExhausted& error) {
            cleanup();
            if (execution_started) {
                finish_batch(jobs, "admission_after_execution");
                return true;
            }
            if (job->cancel.load()) {
                finish(job, JobState::cancelled, "");
                return true;
            }
            // The reason tells the API which memory the engine could not get.
            publish_(
                request.id, JobState::waiting_for_resources, {{"error", error.what()}}
            );
            return false;
        } catch (const RequestError& error) {
            const Json failure{{"error", error.what()}, {"status_code", error.status}};
            cleanup();
            finish_batch(jobs, failure);
            return true;
        } catch (const std::exception& error) {
            const std::string diagnostic = error.what();
            cleanup();  // Failed cleanup poisons the owner and retains the FIFO head.
            finish_batch(jobs, diagnostic);
            return true;
        }
    }

    void shutdown() {
        stop();
        {
            std::lock_guard guard(mutex_);
            for (const auto& job : queue_)
                job->request.prepared = true;
        }
        while (pending())
            advance();
        cleanup();
    }

    std::size_t pending() const {
        std::lock_guard guard(mutex_);
        return queue_.size();
    }

    Snapshot snapshot() const {
        return resources_->snapshot();
    }

    // Physical free memory bounded by the granted budget, for engines that
    // size work from their own probes.
    Footprint bounded_probe() {
        auto free = probe_();
        std::lock_guard guard(mutex_);
        if (free.size() != ceiling_.size())
            throw std::runtime_error("probe_shape");
        for (std::size_t i = 0; i < free.size(); ++i)
            free[i] = std::min(free[i], ceiling_[i]);
        return free;
    }

private:
    bool loaded() const {
        std::lock_guard guard(mutex_);
        return loaded_;
    }

    struct Job {
        Job(InferenceRequest q, std::chrono::steady_clock::time_point time)
            : request(std::move(q)), submitted(time) {
        }

        InferenceRequest request;
        Json error;
        std::chrono::steady_clock::time_point submitted;
        std::atomic_bool cancel{false};
    };

    static std::size_t image_count(const InferenceRequest& request) {
        const auto* input = std::get_if<ImageRequest>(&request.input);
        return input ? input->count : 1;
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
        for (const auto& current : jobs)
            finish(
                current,
                current->cancel.load() ? JobState::cancelled : JobState::failed,
                error
            );
    }

    void cleanup() {
        try {
            if (engine_)
                engine_->close();
            engine_.reset();
            resident_model_.clear();
            auto state = resources_->snapshot();
            if (state.residents
                || std::any_of(state.used.begin(), state.used.end(), [](auto n) {
                       return n != 0;
                   }))
                throw std::runtime_error("cleanup_unconfirmed");
        } catch (...) {
            std::lock_guard guard(mutex_);
            poisoned_ = true;
            throw CleanupFailure("cleanup_unconfirmed");
        }
    }

    void finish(const std::shared_ptr<Job>& job, JobState state, const Json& result) {
        {
            std::lock_guard guard(mutex_);
            if (queue_.empty() || queue_.front() != job)
                throw std::runtime_error("fifo_ownership");
            queue_.pop_front();  // Terminal execution is committed before transport
                                 // delivery.
        }
        try {
            if (state == JobState::failed && !result.is_object())
                publish_(
                    job->request.id, state, {{"error", result}, {"status_code", 502}}
                );
            else
                publish_(job->request.id, state, result);
        } catch (...) {
            std::lock_guard guard(mutex_);
            poisoned_ = true;
            throw PublicationFailure("terminal_publication_failed");
        }
    }

    Footprint ceiling_;
    std::shared_ptr<Resources> resources_;
    Factory factory_;
    Probe probe_;
    Publish publish_;
    Prepare prepare_;
    Clock clock_;
    std::size_t limit_;
    mutable std::mutex mutex_;
    std::deque<std::shared_ptr<Job>> queue_;
    std::unique_ptr<InferenceEngine> engine_;
    std::string resident_model_;
    EngineConfiguration resident_configuration_;
    Workload resident_workload_ = Workload::llm;
    bool stopping_ = false, poisoned_ = false;
    bool loaded_, executing_ = false, release_requested_ = false;
    std::string model_;
    std::function<void(const char*)> control_;
};
}  // namespace kadan::serving
