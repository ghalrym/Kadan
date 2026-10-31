#include "kadan/worker_host.hpp"

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

bool valid_id(const std::string& id) {
    return id.size() == 32
           && id.find_first_not_of("0123456789abcdef") == std::string::npos;
}
}  // namespace

const char* worker_name(Workload workload) {
    switch (workload) {
    case Workload::llm:
        return "llm";
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

WorkerHost::WorkerHost(Workload workload, std::size_t devices, Hooks hooks)
    : devices_(devices), hooks_(std::move(hooks)) {
    std::vector<WorkerSpec> specs{
        {"store", std::nullopt}, {worker_name(workload), workload}
    };
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

WorkerHost::~WorkerHost() {
    try {
        stop();
    } catch (...) {
    }
}

WorkerHost::Worker& WorkerHost::worker(const std::string& name) {
    for (auto& worker : workers_)
        if (worker->spec.name == name)
            return *worker;
    throw RequestError("This worker does not run " + name + " work.", 404);
}

WorkerHost::Worker& WorkerHost::route(const InferenceRequest& request) {
    if (request.operation == Operation::download
        || request.operation == Operation::configure)
        return worker("store");
    return worker(worker_name(request.workload));
}

void WorkerHost::publish(
    Worker& worker, const std::string& id, JobState state, const Json& result
) {
    Json message{
        {"id", id}, {"state", state_name(state)}, {"worker", worker.spec.name}
    };
    {
        std::lock_guard guard(traces_mutex_);
        if (traces_.contains(id))
            message["trace_unix_ns"] = trace_unix_ns();
    }
    if (state == JobState::queued) {
        hooks_.send(message);
        return;
    }
    worker.channel.active = id;
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
        {
            std::lock_guard guard(traces_mutex_);
            auto found = traces_.find(id);
            if (found != traces_.end()) {
                std::lock_guard buffer_guard(found->second->mutex);
                message["trace_spans"] = std::move(found->second->spans);
                traces_.erase(found);
            }
        }
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

Json WorkerHost::worker_status(const Worker& worker) const {
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

Json WorkerHost::status() const {
    Json workers = Json::array();
    for (const auto& worker : workers_)
        workers.push_back(worker_status(*worker));
    return {{"machine", hooks_.machine()}, {"workers", workers}};
}

void WorkerHost::handle(const std::string& line) {
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
            const auto parent = command_message.value("traceparent", std::string{});
            const bool sampled =
                parent.size() == 55 && parent.substr(0, 3) == "00-"
                && std::string("13579bdf").find(parent.back()) != std::string::npos
                && parent.find_first_not_of("0123456789abcdef-") == std::string::npos;
            std::shared_ptr<TraceBuffer> buffer;
            if (sampled) {
                buffer = std::make_shared<TraceBuffer>();
                {
                    std::lock_guard guard(traces_mutex_);
                    // A duplicate submit must not replace the original context.
                    if (!traces_.try_emplace(id, buffer).second)
                        throw RequestError("Request is already active.", 409);
                }
                request.trace = [buffer, parent](
                                    const char* phase,
                                    std::int64_t start,
                                    std::int64_t end,
                                    bool failed,
                                    const TextPhaseMetrics* metrics,
                                    const CudaPhaseMetrics* cuda
                                ) {
                    std::lock_guard guard(buffer->mutex);
                    if (buffer->spans.size() < 16) {
                        Json span{
                            {"traceparent", parent},
                            {"phase", phase},
                            {"start", start},
                            {"end", end},
                            {"failed", failed}
                        };
                        if (metrics)
                            span["metrics"] = {
                                {"tokens", metrics->tokens},
                                {"projection_seconds", metrics->projection_seconds},
                                {"attention_seconds", metrics->attention_seconds},
                                {"prepare_seconds", metrics->prepare_seconds},
                                {"device_seconds", metrics->device_seconds},
                                {"host_wait_seconds", metrics->host_wait_seconds},
                                {"recurrent_seconds", metrics->recurrent_seconds},
                                {"projections", metrics->projections},
                                {"projection_batches", metrics->projection_batches},
                                {"streamed_bytes", metrics->streamed_bytes},
                                {"host_expert_bytes", metrics->host_expert_bytes},
                                {"packed_bytes", metrics->packed_bytes},
                                {"device_bytes", metrics->device_bytes},
                                {"device_slices", metrics->device_slices},
                                {"ram_bytes", metrics->ram_bytes},
                                {"ram_slices", metrics->ram_slices},
                                {"packed_calls", metrics->packed_calls},
                                {"host_calls", metrics->host_calls},
                                {"large_row_calls", metrics->large_row_calls},
                                {"unsupported_calls", metrics->unsupported_calls},
                                {"rows", metrics->rows},
                                {"batch_calls", metrics->batch_calls},
                                {"batch_rejections", metrics->batch_rejections},
                                {"large_row_seconds", metrics->large_row_seconds},
                                {"unsupported_seconds", metrics->unsupported_seconds}
                            };
                        if (cuda) {
                            Json kernels = Json::array(), copies = Json::array();
                            for (const auto& k : cuda->kernels)
                                kernels.push_back(
                                    Json::array(
                                        {k.calls,
                                         k.host_seconds,
                                         k.measured,
                                         k.gpu_seconds}
                                    )
                                );
                            for (const auto& c : cuda->copies)
                                copies.push_back(
                                    Json::array({c.calls, c.host_seconds})
                                );
                            span["cuda"] = {
                                {"kernels", std::move(kernels)},
                                {"copies", std::move(copies)},
                                {"sync_calls", cuda->sync.calls},
                                {"sync_seconds", cuda->sync.host_seconds},
                                {"dropped", cuda->dropped},
                                {"unavailable", cuda->unavailable},
                                {"shared_batch_size", cuda->shared_batch_size}
                            };
                        }
                        buffer->spans.push_back(std::move(span));
                    }
                };
            }
            try {
                route(request).owner->submit(std::move(request));
            } catch (...) {
                std::lock_guard guard(traces_mutex_);
                auto found = traces_.find(id);
                if (found != traces_.end() && found->second == buffer)
                    traces_.erase(found);
                throw;
            }
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
            // Budget commands always address this program's inference lane.
            auto& target = *workers_.back();
            if (command_message.contains("worker")
                && command_message["worker"] != target.spec.name)
                throw RequestError("This worker runs " + target.spec.name + ".", 404);
            if (command == "load") {
                const auto budget = command_message.at("budget").get<Footprint>();
                if (budget.size() != devices_ + 1)
                    throw RequestError("Budget does not match this machine.", 422);
                const auto idle = command_message.value("kv_cache_idle_seconds", 3600);
                const bool ok = target.owner->load(budget, std::chrono::seconds(idle));
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
                std::optional<Json> context;
                if (command_message.contains("context"))
                    context = command_message["context"];
                Json message;
                {
                    std::lock_guard guard(store_);
                    message = hooks_.requirement(*target.spec.workload, model, context);
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

void WorkerHost::run(Worker& worker) {
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

void WorkerHost::start() {
    for (auto& worker : workers_)
        worker->thread = std::thread([this, &current = *worker] { run(current); });
}

bool WorkerHost::advance_all() {
    bool progressed = false;
    for (auto& worker : workers_)
        progressed = worker->owner->advance() || progressed;
    return progressed;
}

std::optional<std::string> WorkerHost::failure() const {
    std::lock_guard guard(failure_mutex_);
    return failure_;
}

void WorkerHost::stop() {
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
