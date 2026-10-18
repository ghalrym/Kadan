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

#include "kadan/image_edit.hpp"
#include "kadan/inference_engines.hpp"
#include "kadan/machine_host.hpp"
#include "kadan/preparation.hpp"
#include "kadan/websocket.hpp"

namespace kadan::serving {
namespace {
std::atomic_bool stopping = false;

void stop_signal(int) {
    stopping.store(true);
}

class PhysicalMemory {
public:
    PhysicalMemory() {
        library_ = dlopen("libnvidia-ml.so.1", RTLD_NOW);
        require(library_, "nvml_unavailable");
        initialize_ = symbol<int (*)()>("nvmlInit_v2");
        shutdown_ = symbol<int (*)()>("nvmlShutdown");
        count_ = symbol<int (*)(unsigned*)>("nvmlDeviceGetCount_v2");
        device_ = symbol<int (*)(unsigned, void**)>("nvmlDeviceGetHandleByIndex_v2");
        memory_ = symbol<int (*)(void*, Memory*)>("nvmlDeviceGetMemoryInfo");
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

    // Machine shape and current free memory for the API's ledger.
    Json machine(const std::string& name, const std::filesystem::path& models) const {
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
            devices.push_back(
                {{"index", index},
                 {"vram_total", memory.total},
                 {"vram_free", free[index + 1]}}
            );
        }
        struct statvfs disk{};
        require(statvfs(models.c_str(), &disk) == 0, "disk_probe");
        return {
            {"name", name},
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
    unsigned devices_ = 0;
};

}  // namespace

std::string environment(const char* name, std::string fallback = {}) {
    const char* value = std::getenv(name);
    return value && *value ? value : fallback;
}

std::vector<std::string> split(const std::string& text) {
    std::vector<std::string> parts;
    std::size_t start = 0;
    while (start <= text.size()) {
        const auto end = std::min(text.find(',', start), text.size());
        if (end > start)
            parts.push_back(text.substr(start, end - start));
        start = end + 1;
    }
    return parts;
}

std::string machine_name() {
    std::array<char, 256> host{};
    gethostname(host.data(), host.size() - 1);
    return environment("HIKARI_WORKER_MACHINE", host.data());
}

// Exits on any connection loss; a supervisor (the API for its local machine,
// a restart policy elsewhere) starts a fresh process that reconnects.
int serve(const char* lock_path, const websocket::Url& url) {
    const auto token = environment("HIKARI_WORKER_TOKEN");
    require(!token.empty(), "HIKARI_WORKER_TOKEN is required");
    int lock = open(lock_path, O_CREAT | O_RDWR | O_CLOEXEC | O_NOFOLLOW, 0600);
    require(
        lock >= 0 && flock(lock, LOCK_EX | LOCK_NB) == 0,
        "inference_worker_already_running"
    );
    const auto data = std::filesystem::path(lock_path).parent_path();
    JobWorkspace files(data / "jobs");
    PhysicalMemory memory;
    std::unique_ptr<websocket::Client> client;
    for (auto delay = std::chrono::seconds(1); !client && !stopping.load();
         delay = std::min(delay * 2, std::chrono::seconds(30))) {
        try {
            client = std::make_unique<websocket::Client>(url, token, 64 * 1024 * 1024);
        } catch (const std::exception& error) {
            // A rejected token never succeeds on retry.
            if (std::string(error.what()) == "worker_token_rejected")
                throw;
            std::cerr << "inference_worker: connect failed: " << error.what() << '\n';
            for (auto waited = std::chrono::milliseconds(0);
                 waited < delay && !stopping.load();
                 waited += std::chrono::milliseconds(100))
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }
    if (!client)
        return 0;
    auto send = [&](const Json& message) {
        const auto text = message.dump();
        require(text.size() <= 128 * 1024 * 1024, "worker_output_bound");
        client->send_text(text);
    };
    WorkerChannel* downloading = nullptr;
    ModelStore store(
        data.parent_path(),
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
    MachineHost::Hooks hooks;
    hooks.send = send;
    hooks.probe = [&] {
        return memory.probe();
    };
    hooks.factory = [&](Workload kind,
                        std::shared_ptr<Resources> resources,
                        WorkerChannel& channel) {
        Progress progress = [&channel](const char* phase, std::size_t value) {
            channel.progress(phase, value);
        };
        switch (kind) {
        case Workload::llm:
            return chat_engine(
                resources,
                progress,
                [&channel](const std::string& text) { channel.content(text); },
                store
            );
        case Workload::image:
            return image_engine(
                resources,
                progress,
                [&](const std::string& id, const char* phase, std::size_t value) {
                    send(
                        {{"id", id},
                         {"state", "progress"},
                         {"phase", phase},
                         {"value", value}}
                    );
                },
                channel.available
            );
        case Workload::video:
            return video_engine(resources, progress);
        case Workload::speech:
            return whisper_engine(resources, progress);
        case Workload::formatting:
            return transcription_formatting_engine(resources, progress);
        case Workload::tts:
            return speech_engine(resources, progress);
        case Workload::decision:
            return decision_engine(resources, progress);
        }
        throw std::invalid_argument("unsupported_workload");
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
        if (const auto* image = std::get_if<ImageRequest>(&request.input))
            prepared["total"] =
                image->source && !image->reference
                    ? kadan::image::edit_steps(image->steps, image->strength)
                    : image->steps;
        send(prepared);
    };
    hooks.requirement = [&](Workload kind,
                            const std::optional<std::string>& model,
                            const std::optional<Json>& context) {
        return worker_requirement(kind, model, store, context);
    };
    hooks.machine = [&] {
        return memory.machine(machine_name(), data.parent_path());
    };
    hooks.models = [&] {
        return store.status();
    };
    hooks.cleanup = [&](const std::string& id) {
        files.cleanup(id);
    };
    const auto specs = worker_specs(split(environment("HIKARI_WORKER_WORKLOADS")));
    MachineHost host(specs, memory.devices(), hooks);
    Json names = Json::array();
    for (const auto& spec : specs)
        names.push_back(spec.name);
    send(
        {{"state", "hello"},
         {"protocol", 3},
         {"pid", getpid()},
         {"machine", hooks.machine()},
         {"workers", names}}
    );
    send({{"state", "models"}, {"models", store.status()}});
    send({{"state", "ready"}, {"protocol", 3}, {"pid", getpid()}});
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
        std::cerr << "inference_worker: connection lost: " << error.what() << '\n';
        status = 3;
    }
    host.stop();
    if (const auto failure = host.failure()) {
        std::cerr << "inference_worker: " << *failure << '\n';
        status = 1;
    }
    client->close();
    close(lock);
    return status;
}
}  // namespace kadan::serving

int main(int argc, char** argv) {
    std::signal(SIGTERM, kadan::serving::stop_signal);
    std::signal(SIGINT, kadan::serving::stop_signal);
    std::signal(SIGPIPE, SIG_IGN);
    try {
        kadan::serving::require(
            argc == 3, "usage: kadan-inference-worker LOCK_PATH API_WEBSOCKET_URL"
        );
        return kadan::serving::serve(
            argv[1], kadan::serving::websocket::Url::parse(argv[2])
        );
    } catch (const std::exception& error) {
        std::cerr << "inference_worker: " << error.what() << '\n';
        return 1;
    }
}
