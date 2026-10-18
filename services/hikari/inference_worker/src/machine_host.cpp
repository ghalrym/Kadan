#include "kadan/machine_host.hpp"

#include <algorithm>
#include <chrono>
#include <stdexcept>

namespace kadan::serving {
namespace {
const char* state_name(JobState state) {
    switch (state) {
    case JobState::queued:
        return "queued";
    case JobState::preparing:
        return "preparing";
    case JobState::loading:
        return "loading";
    case JobState::running:
        return "running";
    case JobState::waiting_for_resources:
        return "waiting_for_resources";
    case JobState::succeeded:
        return "succeeded";
    case JobState::failed:
        return "failed";
    case JobState::cancelled:
        return "cancelled";
    }
    throw std::invalid_argument("job_state");
}

constexpr Workload workloads[]{
    Workload::llm,
    Workload::decision,
    Workload::tts,
    Workload::speech,
    Workload::formatting,
    Workload::image,
    Workload::video,
};

bool valid_id(const std::string& id) {
    return id.size() == 32
           && id.find_first_not_of("0123456789abcdef") == std::string::npos;
}
}  // namespace

const char* worker_name(Workload workload) {
    switch (workload) {
    case Workload::llm:
        return "llm";
    case Workload::decision:
        return "decision";
    case Workload::tts:
        return "tts";
    case Workload::speech:
        return "speech";
    case Workload::formatting:
        return "formatting";
    case Workload::image:
        return "image";
    case Workload::video:
        return "video";
    }
    throw std::invalid_argument("workload");
}

std::vector<WorkerSpec> worker_specs(const std::vector<std::string>& names) {
    std::vector<WorkerSpec> specs{{"store", std::nullopt}};
    for (const auto workload : workloads)
        if (names.empty()
            || std::find(names.begin(), names.end(), worker_name(workload))
                   != names.end())
            specs.push_back({worker_name(workload), workload});
    for (const auto& name : names) {
        const bool known =
            std::any_of(specs.begin(), specs.end(), [&](const auto& spec) {
                return spec.name == name;
            });
        if (!known)
            throw std::invalid_argument("unknown_worker: " + name);
    }
    return specs;
}

MachineHost::MachineHost(
    std::vector<WorkerSpec> specs, std::size_t devices, Hooks hooks
)
    : devices_(devices), hooks_(std::move(hooks)) {
    for (auto& spec : specs) {
        auto worker = std::make_unique<Worker>();
        worker->spec = std::move(spec);
        worker->channel.name = worker->spec.name;
        worker->channel.send = hooks_.send;
        auto& current = *worker;
        // Store work never loads an engine, so it is always "loaded" with no
        // budget; inference workers wait for the API to grant one.
        const bool store = !current.spec.workload;
        current.owner = std::make_unique<InferenceOwner>(
            Footprint(devices_ + 1, 0),
            [this, &current](Workload kind, std::shared_ptr<Resources> resources) {
                return hooks_.factory(kind, std::move(resources), current.channel);
            },
            hooks_.probe,
            [this,
             &current](const std::string& id, JobState state, const Json& result) {
                publish(current, id, state, result);
            },
            32,
            [this,
             &current](InferenceRequest& request, const std::atomic_bool& cancel) {
                current.channel.active = request.id;
                // Downloads take their own file lock and may run for minutes.
                if (request.operation == Operation::download) {
                    hooks_.prepare(request, cancel, current.channel);
                    return;
                }
                std::lock_guard guard(store_);
                hooks_.prepare(request, cancel, current.channel);
            },
            [] { return std::chrono::steady_clock::now(); },
            store
        );
        current.channel.available = [&current] {
            return current.owner->bounded_probe();
        };
        current.owner->set_control_listener([this, &current](const char* event) {
            auto message = worker_status(current);
            message["state"] = "worker";
            message["event"] = event;
            hooks_.send(message);
        });
        workers_.push_back(std::move(worker));
    }
}

MachineHost::~MachineHost() {
    try {
        stop();
    } catch (...) {
    }
}

MachineHost::Worker& MachineHost::worker(const std::string& name) {
    for (auto& worker : workers_)
        if (worker->spec.name == name)
            return *worker;
    throw RequestError("Worker is not hosted on this machine.", 404);
}

MachineHost::Worker& MachineHost::route(const InferenceRequest& request) {
    if (request.operation == Operation::download
        || request.operation == Operation::configure)
        return worker("store");
    return worker(worker_name(request.workload));
}

void MachineHost::publish(
    Worker& worker, const std::string& id, JobState state, const Json& result
) {
    if (state == JobState::queued) {
        hooks_.send({{"id", id}, {"state", "queued"}, {"worker", worker.spec.name}});
        return;
    }
    worker.channel.active = id;
    Json message{
        {"id", id}, {"state", state_name(state)}, {"worker", worker.spec.name}
    };
    if (state == JobState::succeeded) {
        message["result"] = result;
    } else if (state == JobState::failed) {
        message["error"] = result.at("error");
        message["status_code"] = result.at("status_code");
    } else if (state == JobState::waiting_for_resources && result.is_object()) {
        message["error"] = result.value("error", "");
    }
    if (state == JobState::succeeded || state == JobState::failed
        || state == JobState::cancelled) {
        hooks_.cleanup(id);
        hooks_.send({{"state", "models"}, {"models", hooks_.models()}});
    }
    const auto snapshot = worker.owner->snapshot();
    message["memory"] = {
        {"capacity", snapshot.capacity},
        {"used", snapshot.used},
        {"residents", snapshot.residents}
    };
    hooks_.send(message);
}

Json MachineHost::worker_status(const Worker& worker) const {
    const auto status = worker.owner->status();
    return {
        {"worker", worker.spec.name},
        {"loaded", status.loaded},
        {"busy", status.busy},
        {"releasing", status.releasing},
        {"queued", status.queued},
        {"model", status.model.empty() ? Json(nullptr) : Json(status.model)},
        {"budget", status.budget},
        {"used", status.memory.used},
        {"capacity", status.memory.capacity}
    };
}

Json MachineHost::status() const {
    Json workers = Json::array();
    for (const auto& worker : workers_)
        workers.push_back(worker_status(*worker));
    return {{"machine", hooks_.machine()}, {"workers", workers}};
}

void MachineHost::handle(const std::string& line) {
    std::string id;
    try {
        const auto command_message = Json::parse(line);
        id = command_message.at("id");
        if (!valid_id(id))
            throw std::invalid_argument("request_id");
        const auto command = command_message.at("command").get<std::string>();
        const auto reply = [&](Json message) {
            message["id"] = id;
            hooks_.send(message);
        };
        if (command == "submit") {
            auto request = parse_request(command_message);
            route(request).owner->submit(std::move(request));
        } else if (command == "cancel") {
            for (auto& worker : workers_)
                if (worker->owner->cancel(id))
                    break;
        } else if (command == "stop") {
            stopping_.store(true);
        } else if (command == "status") {
            auto message = status();
            message["state"] = "status";
            reply(message);
        } else {
            auto& target = worker(command_message.at("worker").get<std::string>());
            if (!target.spec.workload)
                throw RequestError("The store worker has no engine.", 422);
            if (command == "load") {
                const auto budget = command_message.at("budget").get<Footprint>();
                if (budget.size() != devices_ + 1)
                    throw RequestError("Budget does not match this machine.", 422);
                const bool ok = target.owner->load(budget);
                reply(
                    {{"state", "control"},
                     {"worker", target.spec.name},
                     {"ok", ok},
                     {"error", ok ? Json(nullptr) : Json("busy")}}
                );
            } else if (command == "offload") {
                const bool ok = target.owner->offload();
                reply(
                    {{"state", "control"},
                     {"worker", target.spec.name},
                     {"ok", ok},
                     {"error", ok ? Json(nullptr) : Json("busy")}}
                );
            } else if (command == "is_busy") {
                reply(
                    {{"state", "control"},
                     {"worker", target.spec.name},
                     {"ok", true},
                     {"busy", target.owner->busy()}}
                );
            } else if (command == "requirement") {
                std::optional<std::string> model;
                if (command_message.contains("model")
                    && !command_message["model"].is_null())
                    model = command_message["model"].get<std::string>();
                Json message;
                {
                    std::lock_guard guard(store_);
                    message = hooks_.requirement(*target.spec.workload, model);
                }
                message["state"] = "requirement";
                message["worker"] = target.spec.name;
                reply(message);
            } else
                throw std::invalid_argument("worker_command");
        }
    } catch (const RequestError& error) {
        hooks_.send(
            {{"id", id},
             {"state", "rejected"},
             {"error", error.what()},
             {"status_code", error.status}}
        );
    } catch (const std::exception& error) {
        hooks_.send({{"id", id}, {"state", "rejected"}, {"error", error.what()}});
    }
}

void MachineHost::run(Worker& worker) {
    try {
        while (!stopping_.load()) {
            if (!worker.owner->advance())
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    } catch (const std::exception& error) {
        std::lock_guard guard(failure_mutex_);
        if (!failure_)
            failure_ = worker.spec.name + ": " + error.what();
        stopping_.store(true);
    }
}

void MachineHost::start() {
    for (auto& worker : workers_)
        worker->thread = std::thread([this, &current = *worker] { run(current); });
}

bool MachineHost::advance_all() {
    bool progressed = false;
    for (auto& worker : workers_)
        progressed = worker->owner->advance() || progressed;
    return progressed;
}

std::optional<std::string> MachineHost::failure() const {
    std::lock_guard guard(failure_mutex_);
    return failure_;
}

void MachineHost::stop() {
    stopping_.store(true);
    for (auto& worker : workers_)
        worker->owner->stop();
    for (auto& worker : workers_)
        if (worker->thread.joinable())
            worker->thread.join();
    std::exception_ptr first;
    for (auto& worker : workers_) {
        try {
            worker->owner->shutdown();
        } catch (...) {
            if (!first)
                first = std::current_exception();
        }
    }
    if (first)
        std::rethrow_exception(first);
}
}  // namespace kadan::serving
