#include "kadan/inference_owner.hpp"

#include <cassert>
#include <iostream>
#include <thread>
#include <tuple>
using namespace kadan;
using namespace kadan::serving;

struct FakeState {
    std::vector<std::string> events;
    std::vector<Footprint> loads;
    bool wait_once = false, fail_close = false, invalid_input = false;
    unsigned preparations = 0, close_calls = 0;
    std::shared_ptr<Resources> ledger;
};

struct FakeEngine : InferenceEngine {
    FakeState& state;
    std::shared_ptr<Resources> ledger;
    Handle handle = 0;
    std::string name;

    FakeEngine(FakeState& s, std::shared_ptr<Resources> r)
        : state(s), ledger(std::move(r)) {
        if (state.ledger)
            assert(state.ledger == ledger);
        else
            state.ledger = ledger;
    }

    void load(
        const InferenceRequest& request, const Footprint& free, const std::atomic_bool&
    ) override {
        name = request.model;
        state.events.push_back("load:" + name);
        state.loads.push_back(free);
        handle = ledger->reserve(request.workload, {10, 0, 0});
        if (state.wait_once) {
            state.wait_once = false;
            throw AdmissionWait("physical_pressure");
        }
    }

    void prepare(
        const InferenceRequest&, const Footprint&, const std::atomic_bool&
    ) override {
        ++state.preparations;
        if (state.invalid_input)
            throw RequestError("Image mask must be opaque", 422);
    }

    Json execute(const InferenceRequest&, const std::atomic_bool&) override {
        state.events.push_back("execute:" + name);
        return name;
    }

    void close() override {
        ++state.close_calls;
        if (state.fail_close)
            throw std::runtime_error("cleanup_failed");
        if (handle) {
            ledger->released(handle);
            handle = 0;
            state.events.push_back("close:" + name);
        }
    }
};

InferenceRequest request(std::string id, Workload workload = Workload::image) {
    InferenceRequest result;
    result.id = id;
    result.model = id;
    result.workload = workload;
    result.prepared = true;
    return result;
}

void fifo_shares_ledger() {
    FakeState state;
    std::vector<std::tuple<std::string, JobState, Json>> replies;
    Footprint free{100, 80, 70};
    InferenceOwner owner(
        {100, 100, 100},
        [&](auto, std::shared_ptr<Resources> r) {
            return std::make_unique<FakeEngine>(state, r);
        },
        [&] { return free; },
        [&](auto id, auto kind, auto value) { replies.emplace_back(id, kind, value); }
    );
    const Workload workloads[]{
        Workload::llm,
        Workload::decision,
        Workload::image,
        Workload::video,
        Workload::speech,
        Workload::tts
    };
    for (std::size_t i = 0; i < 6; ++i)
        owner.submit(request(std::to_string(i), workloads[i]));
    for (std::size_t i = 0; i < 6; ++i) {
        assert(owner.advance());
        assert(std::get<0>(replies.back()) == std::to_string(i));
    }
    assert(owner.pending() == 0);
    owner.shutdown();
    assert(owner.snapshot().residents == 0);
    // Reclaim overestimation cannot carry across handoff: the probe occurs after
    // the actual previous engine cleanup, and only its measured free bytes enter
    // load.
    assert(state.loads.size() == 6);
    for (const auto& p : state.loads)
        assert(p == free);
}

void admission_retries_head() {
    std::vector<std::tuple<std::string, JobState, Json>> replies;
    FakeState retry;
    retry.wait_once = true;
    std::size_t probes = 0;
    replies.clear();
    InferenceOwner pending(
        {100, 100, 100},
        [&](auto, std::shared_ptr<Resources> r) {
            return std::make_unique<FakeEngine>(retry, r);
        },
        [&] { return ++probes == 1 ? Footprint{100, 80, 70} : Footprint{100, 40, 30}; },
        [&](auto id, auto kind, auto value) { replies.emplace_back(id, kind, value); }
    );
    pending.submit(request("head"));
    pending.submit(request("tail"));
    assert(!pending.advance());
    assert(pending.pending() == 2);
    assert(retry.events == std::vector<std::string>({"load:head", "close:head"}));
    assert(pending.advance());
    assert(retry.loads.back() == Footprint({100, 40, 30}));
    assert(
        std::get<0>(replies.back()) == "head"
        && std::get<1>(replies.back()) == JobState::succeeded
    );
    assert(pending.cancel("tail"));
    assert(pending.advance());
    assert(std::get<1>(replies.back()) == JobState::cancelled);
    pending.shutdown();
}

void cleanup_failure_stops_owner() {
    FakeState broken;
    InferenceOwner quarantine(
        {100, 100, 100},
        [&](auto, std::shared_ptr<Resources> r) {
            return std::make_unique<FakeEngine>(broken, r);
        },
        [] { return Footprint{100, 80, 70}; },
        [](auto, auto, auto) {}
    );
    quarantine.submit(request("first"));
    assert(quarantine.advance());
    broken.fail_close = true;
    quarantine.submit(request("second"));
    bool failed = false;
    try {
        quarantine.advance();
    } catch (const std::runtime_error&) {
        failed = true;
    }
    assert(failed);
    assert(quarantine.pending() == 1 && quarantine.snapshot().residents == 1);
    failed = false;
    try {
        quarantine.advance();
    } catch (const std::runtime_error&) {
        failed = true;
    }
    assert(failed);
    assert(broken.loads.size() == 1 && broken.close_calls == 1);
}

void reused_model_prepares_each_request() {
    FakeState same;
    InferenceOwner reusable(
        {100, 100, 100},
        [&](auto, std::shared_ptr<Resources> r) {
            return std::make_unique<FakeEngine>(same, r);
        },
        [] { return Footprint{100, 80, 70}; },
        [](auto, auto, auto) {}
    );
    auto first = request("one");
    first.model = "same";
    auto second = request("two");
    second.model = "same";
    reusable.submit(first);
    reusable.submit(second);
    assert(reusable.advance());
    assert(reusable.advance());
    assert(same.loads.size() == 1 && same.preparations == 2);
    reusable.shutdown();
}

void terminal_delivery_never_replays() {
    bool failed = false;
    FakeState delivery;
    InferenceOwner transport(
        {100, 100, 100},
        [&](auto, std::shared_ptr<Resources> r) {
            return std::make_unique<FakeEngine>(delivery, r);
        },
        [] { return Footprint{100, 80, 70}; },
        [](auto, auto state, auto) {
            if (state == JobState::succeeded)
                throw std::runtime_error("reader_gone");
        }
    );
    transport.submit(request("once"));
    failed = false;
    try {
        transport.advance();
    } catch (const PublicationFailure&) {
        failed = true;
    }
    assert(failed && transport.pending() == 0);
    failed = false;
    try {
        transport.advance();
    } catch (const std::runtime_error&) {
        failed = true;
    }
    assert(failed);
    assert(
        std::count(delivery.events.begin(), delivery.events.end(), "execute:once") == 1
    );
}

void changed_configuration_reloads_and_gpu_pressure_reconciles() {
    FakeState state;
    Footprint free{100, 80, 70};
    InferenceOwner owner(
        {100, 100, 100},
        [&](auto, auto resources) {
            return std::make_unique<FakeEngine>(state, resources);
        },
        [&] { return free; },
        [](auto, auto, auto) {}
    );
    auto first = request("first", Workload::decision);
    first.model = "same";
    first.configuration.context_limit = 8192;
    owner.submit(first);
    assert(owner.advance());
    auto second = first;
    second.id = "second";
    free[1] = 79;
    owner.submit(second);
    assert(owner.advance());
    assert(state.loads.size() == 1);
    assert(owner.snapshot().capacity[1] == 79);
    auto third = first;
    third.id = "third";
    third.configuration.context_limit = 65536;
    owner.submit(third);
    assert(owner.advance());
    assert(state.loads.size() == 2);
    owner.shutdown();
}

void refreshed_reservations_never_create_physical_capacity() {
    Resources resources({100});
    auto resident = resources.reserve(Workload::llm, {40});
    // Only ten physical bytes back a conservative forty-byte reservation.
    resources.refresh_available({90});
    assert(resources.snapshot().capacity[0] == 100);
    resources.released(resident);
    bool rejected = false;
    try {
        resources.reserve(Workload::llm, {130});
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    assert(rejected);
}

void resident_gpu_pressure_replans_before_execution() {
    struct PlannedEngine : FakeEngine {
        Bytes planned = 0;
        using FakeEngine::FakeEngine;

        void load(
            const InferenceRequest& request,
            const Footprint& available,
            const std::atomic_bool& cancel
        ) override {
            FakeEngine::load(request, available, cancel);
            planned = available[1];
        }

        void prepare(
            const InferenceRequest&, const Footprint& available, const std::atomic_bool&
        ) override {
            if (available[1] < planned)
                throw AdmissionWait("request_state_pressure");
        }
    };

    FakeState state;
    Footprint free{100, 80, 70};
    std::vector<JobState> events;
    InferenceOwner owner(
        {100, 100, 100},
        [&](auto, auto ledger) {
            return std::make_unique<PlannedEngine>(state, ledger);
        },
        [&] { return free; },
        [&](auto, auto event, auto) { events.push_back(event); }
    );
    auto head = request("first");
    head.model = "same";
    owner.submit(head);
    assert(owner.advance());
    free[1] = 40;
    head.id = "second";
    owner.submit(head);
    assert(!owner.advance());
    assert(events.back() == JobState::waiting_for_resources);
    assert(owner.pending() == 1);
    assert(state.close_calls == 1);
    assert(owner.advance());
    assert(events.back() == JobState::succeeded);
    assert(state.loads.size() == 2 && state.loads.back()[1] == 40);
    owner.shutdown();
}

struct ScratchEngine : FakeEngine {
    bool during_execution;

    ScratchEngine(
        FakeState& state,
        std::shared_ptr<Resources> ledger,
        bool during_execution = false
    )
        : FakeEngine(state, std::move(ledger)), during_execution(during_execution) {
    }

    void reserve_scratch() {
        const auto scratch = ledger->reserve(Workload::llm, {20, 0, 0});
        ledger->released(scratch);
    }

    void prepare(
        const InferenceRequest& request, const Footprint&, const std::atomic_bool&
    ) override {
        if (request.operation != Operation::load && !during_execution)
            reserve_scratch();
    }

    Json
    execute(const InferenceRequest& request, const std::atomic_bool& cancel) override {
        auto result = FakeEngine::execute(request, cancel);
        if (during_execution)
            reserve_scratch();
        return result;
    }
};

void resident_load_recovers_scratch_capacity() {
    FakeState state;
    unsigned probes = 0;
    std::vector<JobState> events;
    InferenceOwner owner(
        {100, 100, 100},
        [&](auto, auto ledger) {
            return std::make_unique<ScratchEngine>(state, ledger);
        },
        [&] { return Footprint{++probes == 2 ? 5u : 100u, 80, 70}; },
        [&](auto, auto event, auto) { events.push_back(event); }
    );
    auto load = request("load", Workload::llm);
    load.model = "same";
    load.operation = Operation::load;
    owner.submit(load);
    assert(owner.advance());
    assert(events.back() == JobState::succeeded);
    assert(owner.snapshot().capacity[0] == 15);
    auto completion = load;
    completion.id = "completion";
    completion.operation = Operation::completion;
    owner.submit(completion);
    assert(owner.advance());
    assert(events.back() == JobState::succeeded);
    assert(state.loads.size() == 1);
    owner.shutdown();
}

void reservation_exhaustion_retries_only_before_execution() {
    for (bool during_execution : {false, true}) {
        FakeState state;
        std::vector<JobState> events;
        Footprint free{15, 80, 70};
        InferenceOwner owner(
            {100, 100, 100},
            [&](auto, auto ledger) {
                return std::make_unique<ScratchEngine>(state, ledger, during_execution);
            },
            [&] { return free; },
            [&](auto, auto event, auto) { events.push_back(event); }
        );
        owner.submit(request("head"));
        owner.submit(request("tail"));
        assert(owner.advance() == during_execution);
        assert(owner.pending() == (during_execution ? 1u : 2u));
        assert(
            events.back()
            == (during_execution ? JobState::failed : JobState::waiting_for_resources)
        );
        assert(state.close_calls == 1 && owner.snapshot().residents == 0);
        free[0] = 100;
        assert(owner.advance());
        assert(events.back() == JobState::succeeded);
        assert(
            std::count(state.events.begin(), state.events.end(), "execute:head") == 1
        );
        if (!during_execution) {
            assert(owner.cancel("tail"));
            assert(owner.advance());
        }
        owner.shutdown();
    }
}

void worker_prepares_only_the_fifo_head() {
    FakeState state;
    std::vector<std::string> prepared;
    std::vector<JobState> events;
    InferenceOwner owner(
        {100, 100, 100},
        [&](auto, auto ledger) { return std::make_unique<FakeEngine>(state, ledger); },
        [] { return Footprint{100, 100, 100}; },
        [&](auto, auto event, auto) { events.push_back(event); },
        32,
        [&](InferenceRequest& job, const std::atomic_bool&) {
            prepared.push_back(job.id);
            if (job.id == "bad")
                throw std::runtime_error("preparation_failed");
            job.model = "resolved";
            job.configuration.checkpoint = "/prepared";
        }
    );
    for (const auto* id : {"head", "cancelled", "bad", "tail"}) {
        auto job = request(id);
        job.prepared = false;
        owner.submit(job);
    }
    assert(prepared.empty());
    assert(owner.cancel("cancelled"));
    assert(owner.advance());
    assert(prepared == std::vector<std::string>{"head"});
    assert(events.back() == JobState::succeeded);
    assert(owner.advance());
    assert(events.back() == JobState::cancelled);
    assert(owner.advance());
    assert(events.back() == JobState::failed);
    assert(owner.advance());
    assert(events.back() == JobState::succeeded);
    assert((prepared == std::vector<std::string>{"head", "bad", "tail"}));
    assert(state.preparations == 2);
    owner.shutdown();
}

void cancellation_during_worker_preparation_never_executes() {
    FakeState state;
    std::vector<JobState> events;
    InferenceOwner* current = nullptr;
    InferenceOwner owner(
        {100, 100, 100},
        [&](auto, auto ledger) { return std::make_unique<FakeEngine>(state, ledger); },
        [] { return Footprint{100, 100, 100}; },
        [&](auto, auto event, auto) { events.push_back(event); },
        32,
        [&](InferenceRequest& job, const std::atomic_bool& cancel) {
            assert(current->cancel(job.id));
            assert(cancel.load());
        }
    );
    current = &owner;
    auto job = request("head");
    job.prepared = false;
    owner.submit(job);
    assert(owner.advance());
    assert(events.back() == JobState::cancelled);
    assert(state.events.empty());
    assert(owner.pending() == 0);
    owner.shutdown();
}

void download_uses_fifo_without_loading_engine() {
    FakeState state;
    std::vector<std::string> prepared;
    std::vector<JobState> events;
    InferenceOwner owner(
        {100, 100, 100},
        [&](auto, auto ledger) { return std::make_unique<FakeEngine>(state, ledger); },
        [] { return Footprint{100, 100, 100}; },
        [&](auto, auto event, auto) { events.push_back(event); },
        32,
        [&](InferenceRequest& job, const std::atomic_bool&) {
            prepared.push_back(job.id);
        }
    );
    auto download = request("download");
    download.operation = Operation::download;
    download.input = DownloadRequest{"small", false};
    download.prepared = false;
    owner.submit(download);
    owner.submit(request("inference"));
    assert(owner.advance());
    assert(prepared == std::vector<std::string>{"download"});
    assert(state.loads.empty() && events.back() == JobState::succeeded);
    assert(owner.advance());
    assert(state.loads.size() == 1);
    owner.shutdown();
}

void acceptance_precedes_fast_completion() {
    for (const bool fail_preparation : {false, true}) {
        std::vector<JobState> events;
        std::atomic_bool finished{false};
        InferenceOwner owner(
            {100},
            {},
            [] { return Footprint{100}; },
            [&](auto, JobState event, auto) {
                events.push_back(event);
                if (event == JobState::succeeded || event == JobState::failed)
                    finished.store(true);
            },
            32,
            [&](auto&, auto&) {
                if (fail_preparation)
                    throw std::runtime_error("preparation failed");
            }
        );
        std::thread executor([&] {
            while (!finished.load()) {
                owner.advance();
                std::this_thread::yield();
            }
        });
        auto job = request("configuration");
        job.operation = Operation::configure;
        job.prepared = false;
        owner.submit(job);
        executor.join();
        assert(
            events
            == std::vector<JobState>(
                {JobState::queued,
                 JobState::preparing,
                 fail_preparation ? JobState::failed : JobState::succeeded}
            )
        );
        owner.shutdown();
    }
}

void failed_acceptance_never_runs() {
    bool prepared = false;
    InferenceOwner owner(
        {100},
        {},
        [] { return Footprint{100}; },
        [](auto, JobState event, auto) {
            assert(event == JobState::queued);
            throw std::runtime_error("transport failed");
        },
        32,
        [&](auto&, auto&) { prepared = true; }
    );
    auto job = request("configuration");
    job.operation = Operation::configure;
    job.prepared = false;
    bool rejected = false;
    try {
        owner.submit(job);
    } catch (const PublicationFailure&) {
        rejected = true;
    }
    assert(rejected && owner.pending() == 0);
    assert(!owner.advance() && !prepared);
    owner.shutdown();
}

void invalid_engine_input_preserves_status_and_releases_memory() {
    FakeState state;
    state.invalid_input = true;
    Json failure;
    InferenceOwner owner(
        {100, 100, 100},
        [&](auto, auto resources) {
            return std::make_unique<FakeEngine>(state, resources);
        },
        [] { return Footprint{100, 80, 70}; },
        [&](auto, JobState kind, const Json& value) {
            if (kind == JobState::failed)
                failure = value;
        }
    );
    owner.submit(request("invalid-upload"));
    assert(owner.advance());
    assert(
        failure.at("status_code") == 422
        && failure.at("error") == "Image mask must be opaque"
    );
    assert(owner.pending() == 0 && owner.snapshot().residents == 0);
    assert(
        std::find(state.events.begin(), state.events.end(), "execute:invalid-upload")
        == state.events.end()
    );
    owner.shutdown();
}

int main() {
    invalid_engine_input_preserves_status_and_releases_memory();
    acceptance_precedes_fast_completion();
    failed_acceptance_never_runs();
    cancellation_during_worker_preparation_never_executes();
    worker_prepares_only_the_fifo_head();
    resident_load_recovers_scratch_capacity();
    reservation_exhaustion_retries_only_before_execution();
    resident_gpu_pressure_replans_before_execution();
    refreshed_reservations_never_create_physical_capacity();
    changed_configuration_reloads_and_gpu_pressure_reconciles();
    download_uses_fifo_without_loading_engine();
    fifo_shares_ledger();
    admission_retries_head();
    cleanup_failure_stops_owner();
    reused_model_prepares_each_request();
    terminal_delivery_never_replays();
    std::cout << "Inference owner fake-engine units passed\n";
}
