#pragma once

#include <atomic>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include "kadan/inference_owner.hpp"

// One native process per machine. It hosts a worker per inference type, each
// with its own owner, queue and API-granted budget, and routes API commands to
// them. Transport, engines and preparation are injected.
namespace kadan::serving {
// Per-worker callbacks into the transport. Each worker's owner thread is the
// only writer of `active`, so engine progress needs no lock.
struct WorkerChannel {
    std::string name;
    std::function<void(const Json&)> send;
    // Free memory bounded by this worker's budget.
    std::function<Footprint()> available;
    std::string active;

    void progress(const char* phase, std::size_t value) const {
        send(
            {{"id", active}, {"state", "progress"}, {"phase", phase}, {"value", value}}
        );
    }

    void content(const std::string& text) const {
        send({{"id", active}, {"state", "content"}, {"content", text}});
    }
};

struct WorkerSpec {
    std::string name;
    std::optional<Workload> workload;  // Empty for the model store worker.
};

// Every inference type, plus "store" for downloads and settings.
std::vector<WorkerSpec> worker_specs(const std::vector<std::string>& names);
const char* worker_name(Workload workload);

class MachineHost {
public:
    using Send = std::function<void(const Json&)>;
    using Factory = std::function<std::unique_ptr<InferenceEngine>(
        Workload, std::shared_ptr<Resources>, WorkerChannel&
    )>;
    using Prepare =
        std::function<void(InferenceRequest&, const std::atomic_bool&, WorkerChannel&)>;
    // {"model", "minimum": [host, device], "preferred": [host, device]}
    using Requirement =
        std::function<Json(Workload, const std::optional<std::string>& model)>;

    struct Hooks {
        Send send;
        Factory factory;
        Prepare prepare;
        InferenceOwner::Probe probe;
        Requirement requirement;
        std::function<Json()> machine, models;
        std::function<void(const std::string&)> cleanup;
    };

    MachineHost(std::vector<WorkerSpec> specs, std::size_t devices, Hooks hooks);
    ~MachineHost();

    // Handles one API command; failures are reported to the API, never thrown.
    void handle(const std::string& line);
    // Runs each worker on its own thread until stop().
    void start();
    // Cancels every job, retires them, releases engines and joins threads.
    void stop();
    // For tests and shutdown: advances each worker once on the caller's thread.
    bool advance_all();

    bool stopping() const {
        return stopping_.load();
    }

    // Set when a worker thread failed in a way that poisons its owner.
    std::optional<std::string> failure() const;
    Json status() const;

private:
    struct Worker {
        WorkerSpec spec;
        WorkerChannel channel;
        std::unique_ptr<InferenceOwner> owner;
        std::thread thread;
    };

    Worker& worker(const std::string& name);
    Worker& route(const InferenceRequest& request);
    Json worker_status(const Worker& worker) const;
    void
    publish(Worker& worker, const std::string& id, JobState state, const Json& result);
    void run(Worker& worker);

    std::vector<std::unique_ptr<Worker>> workers_;
    std::size_t devices_;
    Hooks hooks_;
    std::mutex store_;  // Settings and checkpoint preparation share one model store.
    mutable std::mutex failure_mutex_;
    std::optional<std::string> failure_;
    std::atomic_bool stopping_ = false;
};
}  // namespace kadan::serving
