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

// One worker program's command handling: one inference type, its owner and
// queue, and the API-granted budget, plus a store lane for downloads and
// settings so they never wait for a budget. Transport, engine and preparation
// are injected by the program.
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
    std::optional<Workload> workload;  // Empty for the store lane.
};

const char* worker_name(Workload workload);

class WorkerHost {
public:
    using Send = std::function<void(const Json&)>;
    using Factory = std::function<std::unique_ptr<InferenceEngine>(
        Workload, std::shared_ptr<Resources>, WorkerChannel&
    )>;
    using Prepare =
        std::function<void(InferenceRequest&, const std::atomic_bool&, WorkerChannel&)>;
    // {"model", "minimum": [host, device], "preferred": [host, device],
    //  "device_floor": per extra GPU}. `context` is a queued load's explicit
    // context limit (null for the supported maximum); absent uses the saved one.
    using Requirement = std::function<Json(
        Workload,
        const std::optional<std::string>& model,
        const std::optional<Json>& context
    )>;

    struct Hooks {
        Send send;
        Factory factory;
        Prepare prepare;
        InferenceOwner::Probe probe;
        Requirement requirement;
        std::function<Json()> machine, models;
        std::function<void(const std::string&)> cleanup;
    };

    WorkerHost(Workload workload, std::size_t devices, Hooks hooks);
    ~WorkerHost();

    // Handles one API command; failures are reported to the API, never thrown.
    void handle(const std::string& line);
    // Runs the inference and store lanes on their own threads until stop().
    void start();
    // Cancels every job, retires them, releases engines and joins threads.
    void stop();
    // For tests: advances each lane once on the caller's thread.
    bool advance_all();

    bool stopping() const {
        return stopping_.load();
    }

    // Set when a lane failed in a way that poisons its owner.
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
