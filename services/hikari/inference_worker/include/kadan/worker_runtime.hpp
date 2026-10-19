#pragma once

#include <functional>
#include <memory>

#include "kadan/inference_engines.hpp"
#include "kadan/preparation.hpp"
#include "kadan/worker_host.hpp"

// The shared runtime of every worker program. A program supplies its one
// engine; the runtime connects to the API, answers the worker contract (load,
// offload, is_busy, status, requirement), and owns its model store lane.
namespace kadan::serving {
struct WorkerEnvironment {
    std::shared_ptr<Resources> resources;
    WorkerChannel& channel;
    ModelStore& store;
};

struct WorkerProgram {
    Workload workload;
    // CPU-only programs run where no NVIDIA driver is present.
    bool uses_gpu;
    std::function<std::unique_ptr<InferenceEngine>(WorkerEnvironment&)> engine;
    // Adds program-specific fields to the "prepared" progress event.
    std::function<void(const InferenceRequest&, Json&)> prepared = {};
};

// Reads HIKARI_API_URL, HIKARI_WORKER_TOKEN and KADAN_MODEL_DIR. Exits when the
// API connection drops; a restart policy starts a fresh process.
int run_worker(const WorkerProgram& program);
}  // namespace kadan::serving
