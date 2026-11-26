#pragma once

#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <string>

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

struct WorkerConfig {
    std::string workload;
    std::string api_url;
    std::string token;
    std::filesystem::path model_dir;
    std::string machine_name;
    std::string instance;
    std::optional<int> gpu_index;
    std::optional<int> cpu_threads;
    bool watch_stdin = false;
};

// Parses CLI flags (--workload, --api-url, --token, --model-dir, --machine-name,
// --instance, --gpu, --cpu-threads, --config <json>, --watch-stdin) with optional
// JSON config file and environment variable fallbacks.
WorkerConfig parse_worker_config(int argc, char** argv);

struct WorkerProgram {
    Workload workload;
    // CPU-only programs run where no NVIDIA driver is present.
    bool uses_gpu;
    std::function<std::unique_ptr<InferenceEngine>(WorkerEnvironment&)> engine;
    // Adds program-specific fields to the "prepared" progress event.
    std::function<void(const InferenceRequest&, Json&)> prepared = {};
};

// Runs a worker using explicit runtime configuration (with environment fallbacks
// for any fields left empty). Exits when the API connection drops or shutdown is
// requested.
int run_worker(const WorkerProgram& program, const WorkerConfig& config);

// Reads HIKARI_API_URL, HIKARI_WORKER_TOKEN and KADAN_MODEL_DIR. Exits when the
// API connection drops; a restart policy starts a fresh process.
int run_worker(const WorkerProgram& program);
}  // namespace kadan::serving
