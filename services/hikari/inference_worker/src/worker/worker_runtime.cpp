#include "kadan/worker_runtime.hpp"

#include <array>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <thread>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

#include "kadan/platform.hpp"
#include "kadan/websocket.hpp"

namespace kadan::serving {
namespace {
std::atomic_bool stopping = false;

void stop_signal(int) {
    stopping.store(true);
}

#ifdef _WIN32
BOOL WINAPI win32_ctrl_handler(DWORD) {
    stopping.store(true);
    return TRUE;
}
#endif

class PhysicalMemory {
public:
    // CPU-only programs run without an NVIDIA driver and report no GPUs.
    explicit PhysicalMemory(bool uses_gpu, std::optional<int> selected_gpu = std::nullopt)
        : selected_gpu_(selected_gpu) {
        library_ = platform::DynamicLibrary(platform::nvml_library_name());
        if (!library_) {
            require(!uses_gpu, "nvml_unavailable");
            return;
        }
        initialize_ = symbol<int (*)()>("nvmlInit_v2");
        shutdown_ = symbol<int (*)()>("nvmlShutdown");
        count_ = symbol<int (*)(unsigned*)>("nvmlDeviceGetCount_v2");
        device_ = symbol<int (*)(unsigned, void**)>("nvmlDeviceGetHandleByIndex_v2");
        memory_ = symbol<int (*)(void*, Memory*)>("nvmlDeviceGetMemoryInfo");
        uuid_ = symbol<int (*)(void*, char*, unsigned)>("nvmlDeviceGetUUID");
        unsigned total = 0;
        require(
            initialize_() == 0 && count_(&total) == 0 && total <= Resources::max_devices,
            "nvml_initialize"
        );
        if (selected_gpu_) {
            require(
                *selected_gpu_ >= 0 && static_cast<unsigned>(*selected_gpu_) < total,
                "invalid_gpu_index"
            );
            device_indices_.push_back(static_cast<unsigned>(*selected_gpu_));
        } else {
            for (unsigned i = 0; i < total; ++i)
                device_indices_.push_back(i);
        }
    }

    ~PhysicalMemory() {
        if (shutdown_)
            shutdown_();
    }

    Footprint probe(bool container_limit = true) const {
        Footprint available(device_indices_.size() + 1);
        const auto mem = platform::probe_memory(container_limit);
        available[0] = mem.ram_free;
        for (std::size_t i = 0; i < device_indices_.size(); ++i) {
            void* device = nullptr;
            Memory memory{};
            require(
                device_(device_indices_[i], &device) == 0 && memory_(device, &memory) == 0,
                "device_memory_probe"
            );
            constexpr Bytes margin = 512ULL * 1024 * 1024;
            available[i + 1] = memory.free > margin ? memory.free - margin : 0;
        }
        return available;
    }

    std::size_t devices() const {
        return device_indices_.size();
    }

    // Computer shape and current free memory for the API's ledger. Workers on
    // one computer share its RAM and GPUs: the host boot/machine ID identifies
    // the computer across processes, and each GPU's UUID identifies the GPU.
    Json machine(const std::string& name, const std::filesystem::path& models) const {
        const std::string boot = platform::machine_boot_id();
        require(!boot.empty(), "computer_identity");
        const auto label = name.empty() ? "computer-" + boot.substr(0, 8) : name;
        const auto mem = platform::probe_memory(true);
        const auto free = probe();
        Json devices = Json::array();
        for (std::size_t i = 0; i < device_indices_.size(); ++i) {
            void* device = nullptr;
            Memory memory{};
            require(
                device_(device_indices_[i], &device) == 0 && memory_(device, &memory) == 0,
                "device_memory_probe"
            );
            std::array<char, 96> uuid{};
            require(
                uuid_(device, uuid.data(), unsigned(uuid.size())) == 0, "device_uuid"
            );
            devices.push_back(
                {{"index", static_cast<unsigned>(i)},
                 {"uuid", std::string(uuid.data())},
                 {"vram_total", memory.total},
                 {"vram_used", memory.used},
                 {"vram_free", free[i + 1]}}
            );
        }
        const auto disk = platform::probe_disk(models);
        return {
            {"id", boot},
            {"name", label},
            {"ram_total", mem.ram_total},
            {"ram_free", free[0]},
            {"ram_limit", mem.ram_limit ? Json(*mem.ram_limit) : Json(nullptr)},
            {"physical_ram_total", mem.physical_ram_total},
            {"physical_ram_free", mem.physical_ram_free},
            {"disk_total", disk.total},
            {"disk_free", disk.available},
            {"devices", devices}
        };
    }

private:
    struct Memory {
        unsigned long long total, free, used;
    };

    template <class Function> Function symbol(const char* name) {
        auto function = library_.symbol<Function>(name);
        require(function, "nvml_symbol");
        return function;
    }

    std::optional<int> selected_gpu_;
    platform::DynamicLibrary library_;
    int (*initialize_)() = nullptr;
    int (*shutdown_)() = nullptr;
    int (*count_)(unsigned*) = nullptr;
    int (*device_)(unsigned, void**) = nullptr;
    int (*memory_)(void*, Memory*) = nullptr;
    int (*uuid_)(void*, char*, unsigned) = nullptr;
    std::vector<unsigned> device_indices_;
};

std::string environment(const char* name, std::string fallback = {}) {
    const char* value = std::getenv(name);
    return value && *value ? value : fallback;
}

std::unique_ptr<websocket::Client>
connect(const websocket::Url& url, const std::string& token) {
    for (auto delay = std::chrono::seconds(1); !stopping.load();
         delay = std::min(delay * 2, std::chrono::seconds(30))) {
        try {
            return std::make_unique<websocket::Client>(url, token, 64 * 1024 * 1024);
        } catch (const std::exception& error) {
            // A rejected token never succeeds on retry.
            if (std::string(error.what()) == "worker_token_rejected")
                throw;
            std::cerr << "worker: connect failed: " << error.what() << '\n';
            for (auto waited = std::chrono::milliseconds(0);
                 waited < delay && !stopping.load();
                 waited += std::chrono::milliseconds(100))
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }
    return nullptr;
}

int serve(const WorkerProgram& program, const WorkerConfig& config) {
    const auto name = std::string(worker_name(program.workload));
    const auto token = !config.token.empty()
                           ? config.token
                           : environment("HIKARI_WORKER_TOKEN", "insecure-api-key");
    const auto api_url =
        !config.api_url.empty() ? config.api_url : environment("HIKARI_API_URL");
    const auto url = websocket::Url::parse(api_url);
    std::filesystem::path models = config.model_dir;
    if (models.empty()) {
        const auto env_dir = environment("KADAN_MODEL_DIR");
        models = env_dir.empty() ? platform::default_model_dir()
                                 : std::filesystem::path(env_dir);
    }
    const auto machine_name = !config.machine_name.empty()
                                  ? config.machine_name
                                  : environment("HIKARI_WORKER_MACHINE");
    const auto instance = !config.instance.empty()
                              ? config.instance
                              : environment("HIKARI_WORKER_INSTANCE", platform::host_name());

    std::thread stdin_watcher;
    if (config.watch_stdin) {
        stdin_watcher = std::thread([] {
            char buf[64];
            while (std::cin.read(buf, sizeof(buf)) || std::cin.gcount() > 0) {
            }
            stopping.store(true);
        });
        stdin_watcher.detach();
    }

    // Each worker process owns its own job workspace and locks it.
    const auto workspace = models / "inference" / "jobs" / (name + "-" + instance);
    std::filesystem::create_directories(workspace.parent_path());
    platform::FileLock process_lock;
    require(
        process_lock.acquire(workspace.string() + ".lock", true),
        "worker_already_running"
    );
    JobWorkspace files(workspace);
    PhysicalMemory memory(program.uses_gpu, config.gpu_index);
    auto client = connect(url, token);
    if (!client)
        return 0;
    auto send = [&](const Json& message) {
        const auto text = message.dump();
        require(text.size() <= 128 * 1024 * 1024, "worker_output_bound");
        client->send_text(text);
    };
    WorkerChannel* downloading = nullptr;
    ModelStore store(
        models,
        worker_data_directory(),
        {},
        [&](const std::string& model, std::uint64_t downloaded, std::uint64_t total) {
            if (!downloading)
                return;
            send(
                {{"id", downloading->active},
                 {"state", "progress"},
                 {"phase", "download"},
                 {"model", model},
                 {"value", downloaded},
                 {"total", total}}
            );
        }
    );
    WorkerHost::Hooks hooks;
    hooks.send = send;
    hooks.probe = [&] {
        return memory.probe();
    };
    hooks.factory =
        [&](Workload, std::shared_ptr<Resources> resources, WorkerChannel& channel) {
            WorkerEnvironment env{std::move(resources), channel, store};
            return program.engine(env);
        };
    hooks.prepare = [&](InferenceRequest& request,
                        const std::atomic_bool& cancel,
                        WorkerChannel& channel) {
        if (request.operation == Operation::download)
            downloading = &channel;
        prepare_task(request, store, files, cancel);
        Json context = nullptr;
        if (request.context) {
            const auto& limits = *request.context;
            context = {
                {"configured",
                 limits.configured ? Json(*limits.configured) : Json(nullptr)},
                {"effective", limits.effective},
                {"supported", limits.supported}
            };
        }
        Json prepared = {
            {"id", request.id},
            {"state", "progress"},
            {"phase", "prepared"},
            {"model", request.model},
            {"context", context}
        };
        if (program.prepared)
            program.prepared(request, prepared);
        send(prepared);
    };
    hooks.requirement = [&](Workload kind,
                            const std::optional<std::string>& model,
                            const std::optional<Json>& context) {
        return worker_requirement(kind, model, store, context);
    };
    hooks.machine = [&] {
        return memory.machine(machine_name, models);
    };
    hooks.models = [&] {
        return store.status();
    };
    hooks.cleanup = [&](const std::string& id) {
        files.cleanup(id);
    };
    WorkerHost host(program.workload, memory.devices(), hooks);
    const auto pid = platform::process_id();
    send(
        {{"state", "hello"},
         {"protocol", 5},
         {"pid", pid},
         {"worker", name},
         {"instance", instance},
         {"machine", hooks.machine()}}
    );
    send({{"state", "models"}, {"models", store.status()}});
    send({{"state", "ready"}, {"protocol", 5}, {"pid", pid}});
    host.start();
    int status = 0;
    try {
        // Ping every 20 seconds; any frame back, pongs included, shows the API
        // is alive. A minute without one means it is gone.
        auto pinged = std::chrono::steady_clock::now();
        while (!stopping.load() && !host.stopping()) {
            if (auto message = client->receive(std::chrono::milliseconds(500)))
                host.handle(*message);
            const auto now = std::chrono::steady_clock::now();
            require(
                now - client->last_heard() < std::chrono::seconds(60), "api_silent"
            );
            if (now - pinged >= std::chrono::seconds(20)) {
                client->ping();
                pinged = now;
            }
        }
    } catch (const std::exception& error) {
        std::cerr << "worker: connection lost: " << error.what() << '\n';
        status = 3;
    }
    host.stop();
    if (const auto failure = host.failure()) {
        std::cerr << "worker: " << *failure << '\n';
        status = 1;
    }
    client->close();
    process_lock.release();
    return status;
}
}  // namespace

WorkerConfig parse_worker_config(int argc, char** argv) {
    WorkerConfig cfg;
    std::string config_path;
    auto take_value = [&](int& i, std::string_view arg, std::string_view prefix) -> std::optional<std::string> {
        if (arg == prefix) {
            if (i + 1 >= argc)
                throw std::invalid_argument(std::string("missing value for ") + std::string(prefix));
            return std::string(argv[++i]);
        }
        const std::string eq = std::string(prefix) + "=";
        if (arg.starts_with(eq))
            return std::string(arg.substr(eq.size()));
        return std::nullopt;
    };

    // First pass: check for --config so CLI flags override JSON file values.
    for (int i = 1; i < argc; ++i) {
        if (auto val = take_value(i, argv[i], "--config"))
            config_path = *val;
    }
    if (!config_path.empty()) {
        const auto doc = read_document(config_path);
        if (doc.contains("workload") && doc["workload"].is_string())
            cfg.workload = doc["workload"].get<std::string>();
        if (doc.contains("api_url") && doc["api_url"].is_string())
            cfg.api_url = doc["api_url"].get<std::string>();
        if (doc.contains("worker_token") && doc["worker_token"].is_string())
            cfg.token = doc["worker_token"].get<std::string>();
        if (doc.contains("token") && doc["token"].is_string())
            cfg.token = doc["token"].get<std::string>();
        if (doc.contains("model_dir") && doc["model_dir"].is_string())
            cfg.model_dir = doc["model_dir"].get<std::string>();
        if (doc.contains("machine_name") && doc["machine_name"].is_string())
            cfg.machine_name = doc["machine_name"].get<std::string>();
        if (doc.contains("instance") && doc["instance"].is_string())
            cfg.instance = doc["instance"].get<std::string>();
        if (doc.contains("gpu_index") && doc["gpu_index"].is_number_integer())
            cfg.gpu_index = doc["gpu_index"].get<int>();
        if (doc.contains("cpu_threads") && doc["cpu_threads"].is_number_integer())
            cfg.cpu_threads = doc["cpu_threads"].get<int>();
    }

    // Second pass: CLI flags override config file.
    for (int i = 1; i < argc; ++i) {
        std::string_view arg = argv[i];
        if (auto v = take_value(i, arg, "--config")) {
            (void)v;
        } else if (auto v = take_value(i, arg, "--workload")) {
            cfg.workload = *v;
        } else if (auto v = take_value(i, arg, "--api-url")) {
            cfg.api_url = *v;
        } else if (auto v = take_value(i, arg, "--token")) {
            cfg.token = *v;
        } else if (auto v = take_value(i, arg, "--model-dir")) {
            cfg.model_dir = *v;
        } else if (auto v = take_value(i, arg, "--machine-name")) {
            cfg.machine_name = *v;
        } else if (auto v = take_value(i, arg, "--instance")) {
            cfg.instance = *v;
        } else if (auto v = take_value(i, arg, "--gpu")) {
            cfg.gpu_index = std::stoi(*v);
        } else if (auto v = take_value(i, arg, "--cpu-threads")) {
            cfg.cpu_threads = std::stoi(*v);
        } else if (arg == "--watch-stdin") {
            cfg.watch_stdin = true;
        }
    }
    return cfg;
}

int run_worker(const WorkerProgram& program, const WorkerConfig& config) {
    stopping.store(false);
    std::signal(SIGTERM, stop_signal);
    std::signal(SIGINT, stop_signal);
#ifndef _WIN32
    std::signal(SIGPIPE, SIG_IGN);
#else
    SetConsoleCtrlHandler(win32_ctrl_handler, TRUE);
#endif
    try {
        return serve(program, config);
    } catch (const std::exception& error) {
        std::cerr << "worker " << worker_name(program.workload) << ": " << error.what()
                  << '\n';
        return 1;
    }
}

int run_worker(const WorkerProgram& program) {
    return run_worker(program, WorkerConfig{});
}
}  // namespace kadan::serving
