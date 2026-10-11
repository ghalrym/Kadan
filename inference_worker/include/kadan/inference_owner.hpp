#pragma once
#include <algorithm>
#include <atomic>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <string>

#include "kadan/inference_request.hpp"

namespace kadan::serving {
enum class JobState {
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

class InferenceEngine {
public:
    virtual ~InferenceEngine() = default;
    virtual void
    load(const InferenceRequest&, const Footprint&, const std::atomic_bool&) = 0;
    virtual void
    prepare(const InferenceRequest&, const Footprint&, const std::atomic_bool&) = 0;
    virtual Json execute(const InferenceRequest&, const std::atomic_bool&) = 0;
    virtual void close() = 0;
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
    using Publish = std::function<void(const std::string&, JobState, const Json&)>;

    InferenceOwner(
        Footprint ceiling,
        Factory factory,
        Probe probe,
        Publish publish,
        std::size_t limit = 32,
        Prepare prepare = {}
    )
        : ceiling_(std::move(ceiling)),
          resources_(std::make_shared<Resources>(ceiling_)),
          factory_(std::move(factory)), probe_(std::move(probe)),
          publish_(std::move(publish)), prepare_(std::move(prepare)), limit_(limit) {
        if (!limit_ || limit_ > 1024)
            throw std::invalid_argument("queue_limit");
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
        queue_.push_back(std::make_shared<Job>(std::move(request)));
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
        {
            std::lock_guard guard(mutex_);
            if (poisoned_)
                throw std::runtime_error("cleanup_unconfirmed");
            if (queue_.empty())
                return false;
            job = queue_.front();
        }
        if (!job->request.prepared && !job->cancel.load()) {
            publish_(job->request.id, JobState::preparing, "");
            try {
                if (!prepare_)
                    throw std::runtime_error("preparation_unavailable");
                prepare_(job->request, job->cancel);
            } catch (const std::exception& error) {
                job->error = error.what();
            }
            job->request.prepared = true;
        }
        if (job->cancel.load()) {
            finish(job, JobState::cancelled, "");
            return true;
        }
        if (!job->error.empty()) {
            finish(job, JobState::failed, job->error);
            return true;
        }
        const auto& request = job->request;
        bool execution_started = false;
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
                auto free = probe_();
                if (free.size() != ceiling_.size())
                    throw std::runtime_error("probe_shape");
                for (std::size_t i = 0; i < free.size(); ++i)
                    free[i] = std::min(free[i], ceiling_[i]);
                resources_->reset_capacity(free);
                engine_ = factory_(request.workload, resources_);
                if (!engine_)
                    throw std::runtime_error("unsupported_workload");
                publish_(request.id, JobState::loading, "");
                engine_->load(request, free, job->cancel);
                resident_model_ = request.model;
                resident_workload_ = request.workload;
                resident_configuration_ = request.configuration;
            }
            if (job->cancel.load()) {
                cleanup();
                finish(job, JobState::cancelled, "");
                return true;
            }
            // Request-local scratch and context admission is required even when
            // the same model stays resident. No inference runs in prepare().
            auto available = probe_();
            if (available.size() != ceiling_.size())
                throw std::runtime_error("probe_shape");
            for (std::size_t index = 0; index < available.size(); ++index)
                available[index] = std::min(available[index], ceiling_[index]);
            resources_->refresh_available(available);
            engine_->prepare(request, available, job->cancel);
            if (job->cancel.load()) {
                cleanup();
                finish(job, JobState::cancelled, "");
                return true;
            }
            publish_(request.id, JobState::running, "");
            execution_started = true;
            auto result = engine_->execute(request, job->cancel);
            if (job->cancel.load()) {
                cleanup();
                finish(job, JobState::cancelled, "");
            } else
                finish(job, JobState::succeeded, result);
            return true;
        } catch (const CleanupFailure&) {
            throw;
        } catch (const PublicationFailure&) {
            throw;
        } catch (const ResourceExhausted&) {
            cleanup();
            if (execution_started) {
                finish(job, JobState::failed, "admission_after_execution");
                return true;
            }
            if (job->cancel.load()) {
                finish(job, JobState::cancelled, "");
                return true;
            }
            publish_(request.id, JobState::waiting_for_resources, "");
            return false;
        } catch (const std::exception& error) {
            const std::string diagnostic = error.what();
            cleanup();  // Failed cleanup poisons the owner and retains the FIFO head.
            finish(
                job,
                job->cancel.load() ? JobState::cancelled : JobState::failed,
                diagnostic
            );
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

private:
    struct Job {
        explicit Job(InferenceRequest q) : request(std::move(q)) {
        }

        InferenceRequest request;
        std::string error;
        std::atomic_bool cancel{false};
    };

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
    std::size_t limit_;
    mutable std::mutex mutex_;
    std::deque<std::shared_ptr<Job>> queue_;
    std::unique_ptr<InferenceEngine> engine_;
    std::string resident_model_;
    EngineConfiguration resident_configuration_;
    Workload resident_workload_ = Workload::llm;
    bool stopping_ = false, poisoned_ = false;
};
}  // namespace kadan::serving
