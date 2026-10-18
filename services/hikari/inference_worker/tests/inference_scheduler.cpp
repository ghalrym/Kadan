#include <condition_variable>
#include <iostream>
#include <map>
#include <set>

#include "kadan/inference_owner.hpp"

using namespace kadan;
using namespace kadan::serving;
using namespace std::chrono_literals;

namespace {
void check(bool condition, const char* message) {
    if (!condition)
        throw std::runtime_error(message);
}

struct Gate {
    std::mutex mutex;
    std::condition_variable changed;
    bool entered = false, release = false, cancellation_seen = false;

    void wait_for(bool Gate::* condition) {
        std::unique_lock lock(mutex);
        check(
            changed.wait_for(lock, 5s, [&] { return this->*condition; }),
            "fake_execution_timeout"
        );
    }

    void open() {
        std::lock_guard lock(mutex);
        release = true;
        changed.notify_all();
    }
};

struct Model {
    Bytes host = 10, device = 20, scratch = 5, device_scratch = 0;
    bool lazy_scratch = false;
    std::shared_ptr<Resources> resources;
    bool fail_execution = false;
    Gate gate;
    std::size_t loads = 0, executions = 0, closes = 0;
    std::vector<std::size_t> devices;
    std::vector<std::string> context_ids;
};

struct Event {
    std::string id;
    JobState state;
};

struct State {
    std::mutex mutex;
    std::map<std::string, Model> models;
    std::vector<Event> events;
    std::vector<std::string> order;
    std::size_t active = 0, peak_active = 0;

    bool has(const std::string& id, JobState wanted) {
        std::lock_guard lock(mutex);
        return std::any_of(events.begin(), events.end(), [&](const auto& event) {
            return event.id == id && event.state == wanted;
        });
    }
};

class FakeEngine : public InferenceEngine {
public:
    FakeEngine(
        State& state,
        std::shared_ptr<Resources> resources,
        std::shared_ptr<ExecutionContext> context
    )
        : state_(state), resources_(std::move(resources)),
          context_(std::move(context)) {
    }

    void load(
        const InferenceRequest& request, const Footprint& free, const std::atomic_bool&
    ) override {
        name_ = request.model;
        model_ = &state_.models.at(name_);
        {
            std::lock_guard lock(state_.mutex);
            model_->resources = resources_;
        }
        Footprint bytes(free.size());
        bytes[0] = model_->host;
        Bytes remaining = free.size() == 1 ? 0 : model_->device;
        std::vector<std::size_t> devices;
        for (std::size_t i = 1; i < free.size() && remaining; ++i) {
            bytes[i] = std::min(free[i], remaining);
            remaining -= bytes[i];
            if (bytes[i])
                devices.push_back(i);
        }
        if (remaining)
            throw ResourceExhausted("fake_weights_do_not_fit");
        weights_ = resources_->reserve(request.workload, bytes);
        resources_->loaded(weights_);
        std::lock_guard lock(state_.mutex);
        ++model_->loads;
        model_->devices = std::move(devices);
        state_.order.push_back("load:" + name_);
    }

    void prepare(
        const InferenceRequest& request, const Footprint& free, const std::atomic_bool&
    ) override {
        Footprint bytes(free.size());
        bytes[0] = model_->scratch;
        Bytes remaining = model_->device_scratch;
        for (std::size_t i = 1; i < free.size() && remaining; ++i) {
            bytes[i] = std::min(free[i], remaining);
            remaining -= bytes[i];
        }
        if (remaining)
            throw ResourceExhausted("fake_workspace_does_not_fit");
        if (model_->lazy_scratch)
            lazy_scratch_ = bytes;
        else
            scratch_ = resources_->reserve(request.workload, bytes);
        resources_->pin(weights_);
    }

    Json
    execute(const InferenceRequest& request, const std::atomic_bool& cancel) override {
        {
            std::lock_guard lock(state_.mutex);
            ++model_->executions;
            ++state_.active;
            state_.peak_active = std::max(state_.peak_active, state_.active);
            model_->context_ids.push_back(context_->request_id);
            state_.order.push_back("execute:" + request.id);
        }
        {
            std::unique_lock lock(model_->gate.mutex);
            model_->gate.entered = true;
            model_->gate.changed.notify_all();
            while (!model_->gate.release) {
                if (cancel.load()) {
                    model_->gate.cancellation_seen = true;
                    model_->gate.changed.notify_all();
                }
                model_->gate.changed.wait_for(lock, 1ms);
            }
        }
        std::exception_ptr allocation_failure;
        if (model_->lazy_scratch) {
            try {
                scratch_ = resources_->reserve(request.workload, lazy_scratch_);
            } catch (...) {
                allocation_failure = std::current_exception();
            }
        }
        resources_->unpin(weights_);
        if (scratch_)
            resources_->released(scratch_);
        scratch_ = 0;
        {
            std::lock_guard lock(state_.mutex);
            --state_.active;
            check(
                context_->request_id == request.id,
                "execution_context_changed_in_flight"
            );
            state_.order.push_back("exit:" + request.id);
        }
        if (allocation_failure)
            std::rethrow_exception(allocation_failure);
        if (model_->fail_execution)
            throw ResourceExhausted("fake_execution_failure");
        return request.id;
    }

    void close() override {
        const bool had_weights = weights_ != 0;
        if (weights_) {
            resources_->begin_eviction(weights_);
            resources_->released(weights_);
            weights_ = 0;
        }
        if (model_) {
            std::lock_guard lock(state_.mutex);
            if (had_weights)
                ++model_->closes;
            state_.order.push_back("close:" + name_);
        }
    }

private:
    State& state_;
    std::shared_ptr<Resources> resources_;
    std::shared_ptr<ExecutionContext> context_;
    Model* model_ = nullptr;
    std::string name_;
    Handle weights_ = 0, scratch_ = 0;
    Footprint lazy_scratch_;
};

struct Harness {
    State state;
    Footprint capacity;
    bool physical_probe = false;
    InferenceOwner owner;

    explicit Harness(Footprint budget)
        : capacity(std::move(budget)),
          owner(
              capacity,
              [&](auto, auto resources, auto context) {
                  return std::make_unique<FakeEngine>(state, resources, context);
              },
              [&] {
                  auto free = capacity;
                  if (physical_probe) {
                      std::lock_guard lock(state.mutex);
                      for (const auto& [name, model] : state.models) {
                          if (!model.resources)
                              continue;
                          const auto actual = model.resources->snapshot().used;
                          for (std::size_t i = 0; i < free.size(); ++i)
                              free[i] -= std::min(free[i], actual[i]);
                      }
                  }
                  return free;
              },
              [&](const auto& id, auto event, const auto&) {
                  std::lock_guard lock(state.mutex);
                  state.events.push_back({id, event});
                  if (event == JobState::cancelled)
                      state.order.push_back("cancelled:" + id);
              }
          ) {
    }

    ~Harness() {
        for (auto& [name, model] : state.models)
            model.gate.open();
        try {
            owner.shutdown();
        } catch (...) {
        }
    }

    void submit(std::string id, Workload workload, std::string model = {}) {
        InferenceRequest request;
        request.id = std::move(id);
        request.model = model.empty() ? request.id : std::move(model);
        request.workload = workload;
        request.prepared = true;
        owner.submit(std::move(request));
    }

    template <class Predicate> void advance_until(Predicate done) {
        const auto deadline = std::chrono::steady_clock::now() + 5s;
        while (!done()) {
            check(std::chrono::steady_clock::now() < deadline, "scheduler_timeout");
            owner.advance();
            std::this_thread::yield();
        }
    }

    void drain() {
        advance_until([&] { return owner.pending() == 0; });
    }
};

void disjoint_devices_execute_concurrently() {
    for (std::size_t devices : {2, 3}) {
        Harness h(Footprint(devices + 1, 100));
        h.state.models.try_emplace("chat");
        h.state.models.try_emplace("transcription");
        h.state.models.try_emplace("speech");
        h.submit("chat", Workload::llm);
        h.submit("transcription", Workload::speech);
        if (devices == 3)
            h.submit("speech", Workload::tts);
        check(h.owner.advance(), "chat_not_dispatched");
        h.state.models.at("chat").gate.wait_for(&Gate::entered);
        check(h.owner.advance(), "transcription_not_dispatched_concurrently");
        h.state.models.at("transcription").gate.wait_for(&Gate::entered);
        if (devices == 3) {
            check(h.owner.advance(), "third_gpu_not_dispatched");
            h.state.models.at("speech").gate.wait_for(&Gate::entered);
        }
        {
            std::lock_guard lock(h.state.mutex);
            check(h.state.active == devices, "executions_did_not_overlap");
            std::set<std::size_t> assigned;
            for (const auto& [name, model] : h.state.models)
                if (model.executions) {
                    check(
                        model.devices.size() == 1, "small_model_used_multiple_devices"
                    );
                    check(
                        assigned.insert(model.devices.front()).second, "device_overlap"
                    );
                    check(
                        model.context_ids == std::vector<std::string>{name},
                        "wrong_execution_context"
                    );
                }
        }
        for (auto& [name, model] : h.state.models)
            model.gate.open();
        h.drain();
        check(h.owner.resident_models().size() == devices, "models_not_retained");
    }
}

void one_device_falls_back_and_reuses_residents() {
    Harness h({100, 100});
    auto& chat = h.state.models["chat"];
    auto& speech = h.state.models["speech"];
    h.submit("chat", Workload::llm);
    h.submit("speech", Workload::speech);
    check(h.owner.advance(), "chat_not_dispatched");
    chat.gate.wait_for(&Gate::entered);
    check(!h.owner.advance(), "busy_device_admitted_second_execution");
    check(
        h.owner.resident_models() == std::vector<std::string>{"chat"},
        "active_model_evicted"
    );
    chat.gate.open();
    h.advance_until([&] { return h.state.has("chat", JobState::succeeded); });
    h.advance_until([&] { return h.state.has("speech", JobState::running); });
    speech.gate.wait_for(&Gate::entered);
    speech.gate.open();
    h.drain();
    check(h.owner.resident_models().size() == 2, "sequential_models_not_resident");
    h.submit("chat-again", Workload::llm, "chat");
    h.drain();
    h.submit("speech-again", Workload::speech, "speech");
    h.drain();
    check(chat.loads == 1 && speech.loads == 1, "resident_model_reloaded");
    check(chat.closes == 0 && speech.closes == 0, "alternating_residents_evicted");
    check(speech.executions == 2, "second_resident_not_reused");
    check(
        chat.executions == 2 && h.state.peak_active == 1, "sequential_execution_broken"
    );
    check(
        chat.context_ids == std::vector<std::string>({"chat", "chat-again"}),
        "context_not_updated_between_jobs"
    );
}

void model_expands_to_multiple_devices_when_needed() {
    Harness h({100, 100, 100, 100});
    auto& chat = h.state.models["large-chat"];
    auto& speech = h.state.models["speech"];
    chat.device = 150;
    h.submit("large-chat", Workload::llm);
    h.advance_until([&] { return h.state.has("large-chat", JobState::running); });
    chat.gate.wait_for(&Gate::entered);
    h.submit("speech", Workload::speech);
    check(
        chat.devices == std::vector<std::size_t>({1, 2}),
        "model_did_not_expand_across_gpus"
    );
    // The scheduler reserves all selected lanes for a multi-device engine,
    // including a device its planner did not need after admission.
    check(!h.owner.advance(), "multi_device_execution_lanes_overlapped");
    chat.gate.open();
    h.advance_until([&] { return h.state.has("large-chat", JobState::succeeded); });
    h.advance_until([&] { return h.state.has("speech", JobState::running); });
    speech.gate.wait_for(&Gate::entered);
    speech.gate.open();
    h.drain();
    check(chat.loads == 1, "failed_single_device_attempt_became_resident");
}

void round_robin_preserves_each_workloads_fifo() {
    Harness h({100, 100});
    auto& chat = h.state.models["chat"];
    auto& speech = h.state.models["speech"];
    h.submit("chat-first", Workload::llm, "chat");
    h.submit("chat-second", Workload::llm, "chat");
    h.submit("speech", Workload::speech);
    check(h.owner.advance(), "first_chat_not_dispatched");
    chat.gate.wait_for(&Gate::entered);
    check(!h.owner.advance(), "speech_overlapped_busy_device");
    chat.gate.open();
    h.advance_until([&] { return h.state.has("chat-first", JobState::succeeded); });
    check(h.owner.advance(), "speech_not_dispatched_after_chat");
    check(
        h.state.has("speech", JobState::running), "chat_backlog_bypassed_waiting_speech"
    );
    speech.gate.wait_for(&Gate::entered);
    check(chat.executions == 1, "chat_queue_bypassed_other_workload");
    speech.gate.open();
    h.drain();
    check(
        chat.context_ids == std::vector<std::string>({"chat-first", "chat-second"}),
        "workload_fifo_reordered"
    );
}

void pressure_never_evicts_active_models() {
    Harness h({30, 100, 100});
    auto& chat = h.state.models["chat"];
    auto& speech = h.state.models["speech"];
    chat.host = 10;
    speech.host = 20;
    h.submit("chat", Workload::llm);
    h.submit("speech", Workload::speech);
    h.advance_until([&] { return h.state.has("chat", JobState::running); });
    chat.gate.wait_for(&Gate::entered);
    check(!h.owner.advance(), "global_host_budget_overcommitted");
    check(
        h.owner.snapshot().used == Footprint({15, 50, 0}),
        "failed_admission_leaked_resources"
    );
    check(
        h.state.has("speech", JobState::waiting_for_resources), "pressure_not_reported"
    );
    check(
        chat.closes == 0 && speech.executions == 0,
        "active_model_evicted_or_pressure_ignored"
    );
    chat.gate.open();
    h.advance_until([&] { return h.state.has("chat", JobState::succeeded); });
    h.advance_until([&] { return h.state.has("speech", JobState::running); });
    speech.gate.wait_for(&Gate::entered);
    check(chat.closes == 1, "idle_model_not_reclaimed_for_waiting_work");
    speech.gate.open();
    h.drain();
}

void idle_weights_are_evicted_when_all_device_memory_is_reserved() {
    Harness h({100, 100});
    auto& chat = h.state.models["chat"];
    auto& speech = h.state.models["speech"];
    chat.device = 100;
    chat.gate.open();
    h.submit("chat", Workload::llm);
    h.drain();
    check(
        h.owner.snapshot().used == Footprint({10, 100}),
        "full_device_model_not_retained"
    );
    h.submit("speech", Workload::speech);
    h.advance_until([&] { return h.state.has("speech", JobState::running); });
    speech.gate.wait_for(&Gate::entered);
    check(chat.closes == 1, "full_idle_device_not_reclaimed");
    speech.gate.open();
    h.drain();
}

void queued_and_running_cancellation_wait_for_cleanup() {
    Harness h({100, 100});
    auto& chat = h.state.models["chat"];
    auto& speech = h.state.models["speech"];
    h.submit("chat", Workload::llm);
    h.submit("speech", Workload::speech);
    check(h.owner.advance(), "chat_not_dispatched");
    chat.gate.wait_for(&Gate::entered);
    check(h.owner.cancel("speech"), "queued_cancel_missing");
    check(h.owner.advance(), "queued_cancel_not_completed");
    check(
        h.state.has("speech", JobState::cancelled) && speech.loads == 0,
        "queued_cancel_loaded_model"
    );
    check(h.owner.cancel("chat"), "running_cancel_missing");
    chat.gate.wait_for(&Gate::cancellation_seen);
    check(!h.owner.advance(), "running_cancel_completed_before_execution_exit");
    check(!h.state.has("chat", JobState::cancelled), "premature_cancel_terminal");
    check(
        h.owner.snapshot().used == Footprint({50, 50}),
        "running_cancel_released_in_use_resources"
    );
    chat.gate.open();
    h.drain();
    check(h.state.has("chat", JobState::cancelled), "running_cancel_terminal_missing");
    check(h.owner.snapshot().residents == 0, "cancelled_resources_retained");
    const auto closed =
        std::find(h.state.order.begin(), h.state.order.end(), "close:chat");
    const auto terminal =
        std::find(h.state.order.begin(), h.state.order.end(), "cancelled:chat");
    check(closed < terminal, "cancelled_before_cleanup");
}

void execution_failure_is_cleaned_up_without_replay() {
    Harness h({100, 100});
    auto& chat = h.state.models["chat"];
    chat.fail_execution = true;
    chat.gate.open();
    h.submit("chat", Workload::llm);
    h.drain();
    check(h.state.has("chat", JobState::failed), "execution_failure_not_terminal");
    check(chat.executions == 1 && chat.loads == 1, "execution_failure_replayed");
    check(
        chat.closes == 1 && h.owner.snapshot().residents == 0, "failed_execution_leaked"
    );
    check(!h.owner.advance(), "failed_request_still_queued");
}

void blocked_queue_does_not_starve_other_workloads() {
    Harness h({100, 100});
    auto& chat = h.state.models["chat"];
    auto& speech = h.state.models["speech"];
    chat.host = 101;
    h.submit("chat", Workload::llm);
    h.submit("speech", Workload::speech);
    check(!h.owner.advance(), "oversized_request_admitted");
    check(h.owner.advance(), "blocked_chat_starved_speech");
    speech.gate.wait_for(&Gate::entered);
    check(
        speech.executions == 1 && chat.executions == 0, "fair_queue_selection_failed"
    );
    check(h.owner.cancel("chat"), "blocked_request_not_cancellable");
    speech.gate.open();
    h.drain();
}

void resident_larger_than_half_capacity_keeps_its_workspace() {
    Harness h({100, 100});
    h.physical_probe = true;
    auto& chat = h.state.models["large-chat"];
    chat.host = chat.device = 60;
    chat.scratch = chat.device_scratch = 30;
    chat.gate.open();
    h.submit("large-chat", Workload::llm);
    h.drain();
    check(
        h.state.has("large-chat", JobState::succeeded),
        "resident_memory_was_double_charged"
    );
    check(chat.loads == 1 && chat.executions == 1, "large_resident_retried");
    check(h.owner.snapshot().used == Footprint({60, 60}), "large_weights_not_retained");
}

void lazy_workspace_is_protected_from_second_admission() {
    Harness h({100, 100, 100});
    h.physical_probe = true;
    auto& chat = h.state.models["chat"];
    auto& speech = h.state.models["speech"];
    chat.scratch = 40;
    chat.lazy_scratch = true;
    speech.host = 45;
    h.submit("chat", Workload::llm);
    h.submit("speech", Workload::speech);
    check(h.owner.advance(), "lazy_chat_not_dispatched");
    chat.gate.wait_for(&Gate::entered);
    check(h.owner.advance(), "second_execution_not_admitted");
    speech.gate.wait_for(&Gate::entered);
    check(
        h.owner.snapshot().used == Footprint({100, 50, 50}),
        "lazy_workspace_not_escrowed"
    );
    check(
        chat.resources->snapshot().used[0] == 10,
        "workspace_materialized_before_overlap"
    );
    check(
        chat.resources->snapshot().capacity[0] == 50, "second_admission_stole_workspace"
    );
    chat.gate.open();
    h.advance_until([&] {
        return h.state.has("chat", JobState::succeeded)
               || h.state.has("chat", JobState::failed);
    });
    check(h.state.has("chat", JobState::succeeded), "lazy_workspace_allocation_failed");
    check(h.owner.executing(), "other_execution_was_not_held_during_lazy_allocation");
    speech.gate.open();
    h.drain();
}

void prepare_can_expand_a_model_that_loaded_on_one_device() {
    Harness h({100, 100, 100});
    auto& chat = h.state.models["chat"];
    chat.device_scratch = 150;
    chat.gate.open();
    h.submit("chat", Workload::llm);
    h.drain();
    check(h.state.has("chat", JobState::succeeded), "prepare_workspace_did_not_expand");
    check(chat.loads >= 2 && chat.executions == 1, "prepare_retry_replayed_execution");
    check(chat.closes == chat.loads - 1, "failed_narrow_prepare_not_cleaned_up");
}

void a_waiting_wide_model_has_bounded_bypass() {
    Harness h({100, 100, 100});
    auto& held = h.state.models["held"];
    auto& wide = h.state.models["wide"];
    auto& small = h.state.models["small"];
    wide.device = 150;
    small.gate.open();
    h.submit("held", Workload::speech);
    check(h.owner.advance(), "held_execution_not_dispatched");
    held.gate.wait_for(&Gate::entered);
    h.submit("wide", Workload::llm);
    check(!h.owner.advance(), "wide_model_fit_one_device");
    for (std::size_t i = 0; i < 12; ++i)
        h.submit("small-" + std::to_string(i), Workload::tts, "small");
    h.advance_until([&] { return h.state.has("small-6", JobState::succeeded); });
    // Seven other workload dispatches exhaust the bounded bypass allowance.
    // Keep one execution held so a scheduler lacking the drain barrier would
    // dispatch small-7 and continue starving the two-device request.
    for (int i = 0; i < 16; ++i)
        h.owner.advance();
    check(
        !h.state.has("small-7", JobState::running),
        "wide_request_bypassed_without_bound"
    );
    held.gate.open();
    h.advance_until([&] { return h.state.has("wide", JobState::running); });
    wide.gate.wait_for(&Gate::entered);
    check(
        !h.state.has("small-7", JobState::running),
        "small_request_preempted_aged_wide_request"
    );
    wide.gate.open();
    h.drain();
    check(wide.executions == 1, "wide_request_replayed");
}

void shutdown_waits_for_running_cleanup() {
    Harness h({100, 100});
    auto& chat = h.state.models["chat"];
    h.submit("chat", Workload::llm);
    check(h.owner.advance(), "chat_not_dispatched");
    chat.gate.wait_for(&Gate::entered);
    auto shutdown = std::async(std::launch::async, [&] { h.owner.shutdown(); });

    struct ReleaseOnFailure {
        Gate& gate;

        ~ReleaseOnFailure() {
            gate.open();
        }
    } release{chat.gate};

    // Observation of cancellation proves shutdown has started while the fake
    // still holds its resource pin and cannot yet acknowledge physical cleanup.
    chat.gate.wait_for(&Gate::cancellation_seen);
    check(shutdown.wait_for(0s) != std::future_status::ready, "shutdown_did_not_wait");
    check(
        h.owner.snapshot().used == Footprint({50, 50}),
        "shutdown_released_active_resources"
    );
    chat.gate.open();
    check(shutdown.wait_for(5s) == std::future_status::ready, "shutdown_timeout");
    shutdown.get();
    check(
        h.owner.pending() == 0 && h.owner.snapshot().residents == 0,
        "shutdown_cleanup_incomplete"
    );
    check(h.state.has("chat", JobState::cancelled), "shutdown_terminal_missing");
}
}  // namespace

int main() {
    const std::pair<const char*, void (*)()> tests[]{
        {"disjoint_devices", disjoint_devices_execute_concurrently},
        {"sequential_residency", one_device_falls_back_and_reuses_residents},
        {"multi_device_model", model_expands_to_multiple_devices_when_needed},
        {"round_robin_fifo", round_robin_preserves_each_workloads_fifo},
        {"global_pressure", pressure_never_evicts_active_models},
        {"full_idle_gpu", idle_weights_are_evicted_when_all_device_memory_is_reserved},
        {"cancellation_cleanup", queued_and_running_cancellation_wait_for_cleanup},
        {"failure_no_replay", execution_failure_is_cleaned_up_without_replay},
        {"blocked_queue_fairness", blocked_queue_does_not_starve_other_workloads},
        {"large_resident_workspace",
         resident_larger_than_half_capacity_keeps_its_workspace},
        {"lazy_workspace", lazy_workspace_is_protected_from_second_admission},
        {"prepare_widens", prepare_can_expand_a_model_that_loaded_on_one_device},
        {"bounded_wide_bypass", a_waiting_wide_model_has_bounded_bypass},
        {"shutdown_cleanup", shutdown_waits_for_running_cleanup},
    };
    bool failed = false;
    for (const auto& [name, test] : tests) {
        try {
            test();
        } catch (const std::exception& error) {
            std::cerr << name << ": " << error.what() << '\n';
            failed = true;
        }
    }
    return failed ? 1 : 0;
}
