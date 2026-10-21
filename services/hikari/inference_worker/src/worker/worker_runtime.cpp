#include "kadan/worker_runtime.hpp"

#include <array>
#include <chrono>
#include <csignal>
#include <dlfcn.h>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <sys/file.h>
#include <sys/statvfs.h>
#include <thread>
#include <unistd.h>

#include "kadan/websocket.hpp"

namespace kadan::serving {
namespace {
std::atomic_bool stopping = false;

void stop_signal(int) {
    stopping.store(true);
}

class PhysicalMemory {
public:
    // CPU-only programs run without an NVIDIA driver and report no GPUs.
    explicit PhysicalMemory(bool uses_gpu) {
        library_ = dlopen("libnvidia-ml.so.1", RTLD_NOW);
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
        require(
            initialize_() == 0 && count_(&devices_) == 0
                && devices_ <= Resources::max_devices,
            "nvml_initialize"
        );
    }

    ~PhysicalMemory() {
        if (shutdown_)
            shutdown_();
        if (library_)
            dlclose(library_);
    }

    Footprint probe() const {
        Footprint available(devices_ + 1);
        std::ifstream meminfo("/proc/meminfo");
        std::string line;
        while (std::getline(meminfo, line)) {
            if (line.starts_with("MemAvailable:")) {
                available[0] = std::stoull(line.substr(13)) * 1024;
                break;
            }
        }
        require(available[0] > 0, "host_memory_probe");
        std::ifstream limit_file("/sys/fs/cgroup/memory.max"),
            current_file("/sys/fs/cgroup/memory.current");
        std::string limit;
        Bytes current = 0;
        if (limit_file >> limit && current_file >> current && limit != "max") {
            const auto maximum = std::stoull(limit);
            available[0] = std::min<Bytes>(
                available[0], maximum > current ? maximum - current : 0
            );
        }
        constexpr Bytes host_margin = 1024ULL * 1024 * 1024;
        available[0] = available[0] > host_margin ? available[0] - host_margin : 0;
        for (unsigned index = 0; index < devices_; ++index) {
            void* device = nullptr;
            Memory memory{};
            require(
                device_(index, &device) == 0 && memory_(device, &memory) == 0,
                "device_memory_probe"
            );
            constexpr Bytes margin = 512ULL * 1024 * 1024;
            available[index + 1] = memory.free > margin ? memory.free - margin : 0;
        }
        return available;
    }

    std::size_t devices() const {
        return devices_;
    }

    // Computer shape and current free memory for the API's ledger. Workers on
    // one computer share its RAM and GPUs: the kernel boot ID identifies the
    // computer across containers, and each GPU's UUID identifies the GPU.
    Json machine(const std::string& name, const std::filesystem::path& models) const {
        std::string boot;
        std::ifstream("/proc/sys/kernel/random/boot_id") >> boot;
        require(!boot.empty(), "computer_identity");
        // A container's hostname names the container, not the computer.
        const auto label = name.empty() ? "computer-" + boot.substr(0, 8) : name;
        Bytes ram_total = 0;
        std::ifstream meminfo("/proc/meminfo");
        std::string line;
        while (std::getline(meminfo, line))
            if (line.starts_with("MemTotal:")) {
                ram_total = std::stoull(line.substr(9)) * 1024;
                break;
            }
        std::ifstream limit_file("/sys/fs/cgroup/memory.max");
        std::string limit;
        if (limit_file >> limit && limit != "max")
            ram_total = std::min<Bytes>(ram_total, std::stoull(limit));
        const auto free = probe();
        Json devices = Json::array();
        for (unsigned index = 0; index < devices_; ++index) {
            void* device = nullptr;
            Memory memory{};
            require(
                device_(index, &device) == 0 && memory_(device, &memory) == 0,
                "device_memory_probe"
            );
            std::array<char, 96> uuid{};
            require(
                uuid_(device, uuid.data(), unsigned(uuid.size())) == 0, "device_uuid"
            );
            devices.push_back(
                {{"index", index},
                 {"uuid", std::string(uuid.data())},
                 {"vram_total", memory.total},
                 {"vram_used", memory.used},
                 {"vram_free", free[index + 1]}}
            );
        }
        struct statvfs disk{};
        require(statvfs(models.c_str(), &disk) == 0, "disk_probe");
        return {
            {"id", boot},
            {"name", label},
            {"ram_total", ram_total},
            {"ram_free", free[0]},
            {"disk_total", Bytes(disk.f_blocks) * disk.f_frsize},
            {"disk_free", Bytes(disk.f_bavail) * disk.f_frsize},
            {"devices", devices}
        };
    }

private:
    struct Memory {
        unsigned long long total, free, used;
    };

    template <class Function> Function symbol(const char* name) {
        auto function = reinterpret_cast<Function>(dlsym(library_, name));
        require(function, "nvml_symbol");
        return function;
    }

    void* library_ = nullptr;
    int (*initialize_)() = nullptr;
    int (*shutdown_)() = nullptr;
    int (*count_)(unsigned*) = nullptr;
    int (*device_)(unsigned, void**) = nullptr;
    int (*memory_)(void*, Memory*) = nullptr;
    int (*uuid_)(void*, char*, unsigned) = nullptr;
    unsigned devices_ = 0;
};

std::string environment(const char* name, std::string fallback = {}) {
    const char* value = std::getenv(name);
    return value && *value ? value : fallback;
}

std::string host_name() {
    std::array<char, 256> host{};
    gethostname(host.data(), host.size() - 1);
    return host.data();
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

int serve(const WorkerProgram& program) {
    const auto name = std::string(worker_name(program.workload));
    const auto token = environment("HIKARI_WORKER_TOKEN", "insecure-api-key");
    const auto url = websocket::Url::parse(environment("HIKARI_API_URL"));
    const auto models = std::filesystem::path(environment(
        "KADAN_MODEL_DIR", environment("HOME", "/tmp") + "/.local/share/kadan/models"
    ));
    // Replicas share the model volume, so each process owns its own
    // workspace, named after its container, and locks it.
    const auto instance = environment("HIKARI_WORKER_INSTANCE", host_name());
    const auto workspace = models / "inference" / "jobs" / (name + "-" + instance);
    std::filesystem::create_directories(workspace.parent_path());
    const auto lock_path = workspace.string() + ".lock";
    int lock = open(lock_path.c_str(), O_CREAT | O_RDWR | O_CLOEXEC | O_NOFOLLOW, 0600);
    require(lock >= 0 && flock(lock, LOCK_EX | LOCK_NB) == 0, "worker_already_running");
    JobWorkspace files(workspace);
    PhysicalMemory memory(program.uses_gpu);
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
            WorkerEnvironment environment{std::move(resources), channel, store};
            return program.engine(environment);
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
        return memory.machine(environment("HIKARI_WORKER_MACHINE"), models);
    };
    hooks.models = [&] {
        return store.status();
    };
    hooks.cleanup = [&](const std::string& id) {
        files.cleanup(id);
    };
    WorkerHost host(program.workload, memory.devices(), hooks);
    send(
        {{"state", "hello"},
         {"protocol", 4},
         {"pid", getpid()},
         {"worker", name},
         {"instance", instance},
         {"machine", hooks.machine()}}
    );
    send({{"state", "models"}, {"models", store.status()}});
    send({{"state", "ready"}, {"protocol", 4}, {"pid", getpid()}});
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
    close(lock);
    return status;
}
}  // namespace

int run_worker(const WorkerProgram& program) {
    std::signal(SIGTERM, stop_signal);
    std::signal(SIGINT, stop_signal);
    std::signal(SIGPIPE, SIG_IGN);
    try {
        return serve(program);
    } catch (const std::exception& error) {
        std::cerr << "worker " << worker_name(program.workload) << ": " << error.what()
                  << '\n';
        return 1;
    }
}
}  // namespace kadan::serving
