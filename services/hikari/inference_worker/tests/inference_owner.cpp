#include "kadan/inference_owner.hpp"

#include <cassert>
#include <iostream>
#include <thread>
#include <tuple>

#include "kadan/image_batch_memory.hpp"
#include "kadan/image_batch_output.hpp"
#include "kadan/image_batch_performance.hpp"
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
        state.ledger = ledger;
    }

    void load(
        const InferenceRequest& request, const Footprint& free, const std::atomic_bool&
    ) override {
        name = request.model;
        state.events.push_back("load:" + name);
        state.loads.push_back(free);
        Footprint reservation(free.size());
        reservation[0] = 10;
        handle = ledger->reserve(request.workload, reservation);
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

bool complete_one(InferenceOwner& owner) {
    const auto before = owner.pending();
    bool progressed = owner.advance();
    while (owner.pending() == before && owner.executing()) {
        owner.advance();
        std::this_thread::yield();
    }
    return owner.pending() < before || progressed;
}

void workload_queues_share_ledger() {
    FakeState state;
    std::vector<std::tuple<std::string, JobState, Json>> replies;
    Footprint free{100, 80, 70};
    InferenceOwner owner(
        {100, 100, 100},
        [&](auto, std::shared_ptr<Resources> r, auto) {
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
    for (std::size_t i = 0; i < 6; ++i) {
        owner.submit(request(std::to_string(i), workloads[i]));
        assert(complete_one(owner));
        assert(std::get<0>(replies.back()) == std::to_string(i));
    }
    assert(owner.snapshot().residents == 6);
    assert(owner.pending() == 0);
    owner.shutdown();
    assert(owner.snapshot().residents == 0);
    // Every new resident sees only remaining global capacity, scoped to its
    // assigned device set; existing models survive unrelated queue work.
    assert(state.loads.size() == 6);
    for (std::size_t i = 0; i < state.loads.size(); ++i) {
        assert(state.loads[i][0] <= 100 - i * 10);
        assert(state.loads[i][1] <= free[1]);
        assert(state.loads[i][2] <= free[2]);
    }
}

void admission_retries_head() {
    std::vector<std::tuple<std::string, JobState, Json>> replies;
    FakeState retry;
    retry.wait_once = true;
    std::size_t probes = 0;
    replies.clear();
    InferenceOwner pending(
        {100, 100, 100},
        [&](auto, std::shared_ptr<Resources> r, auto) {
            return std::make_unique<FakeEngine>(retry, r);
        },
        [&] { return ++probes == 1 ? Footprint{100, 80, 70} : Footprint{100, 40, 30}; },
        [&](auto id, auto kind, auto value) { replies.emplace_back(id, kind, value); }
    );
    pending.submit(request("head"));
    pending.submit(request("tail"));
    assert(!complete_one(pending));
    assert(pending.pending() == 2);
    assert(retry.events == std::vector<std::string>({"load:head", "close:head"}));
    assert(complete_one(pending));
    assert(retry.loads.back() == Footprint({100, 40, 30}));
    assert(
        std::get<0>(replies.back()) == "head"
        && std::get<1>(replies.back()) == JobState::succeeded
    );
    assert(pending.cancel("tail"));
    assert(complete_one(pending));
    assert(std::get<1>(replies.back()) == JobState::cancelled);
    pending.shutdown();
}

void cleanup_failure_stops_owner() {
    FakeState broken;
    InferenceOwner quarantine(
        {100, 100, 100},
        [&](auto, std::shared_ptr<Resources> r, auto) {
            return std::make_unique<FakeEngine>(broken, r);
        },
        [] { return Footprint{100, 80, 70}; },
        [](auto, auto, auto) {}
    );
    quarantine.submit(request("first"));
    assert(complete_one(quarantine));
    broken.fail_close = true;
    auto unload = request("second");
    unload.operation = Operation::unload;
    quarantine.submit(unload);
    bool failed = false;
    try {
        complete_one(quarantine);
    } catch (const std::runtime_error&) {
        failed = true;
    }
    assert(failed);
    assert(quarantine.pending() == 1 && quarantine.snapshot().residents == 1);
    failed = false;
    try {
        complete_one(quarantine);
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
        [&](auto, std::shared_ptr<Resources> r, auto) {
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
    assert(complete_one(reusable));
    assert(complete_one(reusable));
    assert(same.loads.size() == 1 && same.preparations == 2);
    reusable.shutdown();
}

void terminal_delivery_never_replays() {
    bool failed = false;
    FakeState delivery;
    InferenceOwner transport(
        {100, 100, 100},
        [&](auto, std::shared_ptr<Resources> r, auto) {
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
        complete_one(transport);
    } catch (const PublicationFailure&) {
        failed = true;
    }
    assert(failed && transport.pending() == 0);
    failed = false;
    try {
        complete_one(transport);
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
        [&](auto, auto resources, auto) {
            return std::make_unique<FakeEngine>(state, resources);
        },
        [&] { return free; },
        [](auto, auto, auto) {}
    );
    auto first = request("first", Workload::llm);
    first.model = "same";
    first.configuration.context_limit = 8192;
    owner.submit(first);
    assert(complete_one(owner));
    auto second = first;
    second.id = "second";
    free[1] = 39;
    owner.submit(second);
    assert(!complete_one(owner));
    assert(complete_one(owner));
    assert(state.loads.size() == 2);
    assert(owner.snapshot().capacity[1] == 39);
    auto third = first;
    third.id = "third";
    third.configuration.context_limit = 65536;
    owner.submit(third);
    assert(complete_one(owner));
    assert(state.loads.size() == 3);
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
        std::size_t device = 0;
        using FakeEngine::FakeEngine;

        void load(
            const InferenceRequest& request,
            const Footprint& available,
            const std::atomic_bool& cancel
        ) override {
            FakeEngine::load(request, available, cancel);
            device = std::size_t(
                std::distance(
                    available.begin(),
                    std::find_if(
                        std::next(available.begin()), available.end(), [](auto bytes) {
                            return bytes != 0;
                        }
                    )
                )
            );
            assert(device < available.size());
            planned = available[device];
        }

        void prepare(
            const InferenceRequest&, const Footprint& available, const std::atomic_bool&
        ) override {
            if (available[device] < planned)
                throw AdmissionWait("request_state_pressure");
        }
    };

    FakeState state;
    Footprint free{100, 80, 70};
    std::vector<JobState> events;
    InferenceOwner owner(
        {100, 100, 100},
        [&](auto, auto ledger, auto) {
            return std::make_unique<PlannedEngine>(state, ledger);
        },
        [&] { return free; },
        [&](auto, auto event, auto) { events.push_back(event); }
    );
    auto head = request("first");
    head.model = "same";
    owner.submit(head);
    assert(complete_one(owner));
    free[1] = 30;
    head.id = "second";
    owner.submit(head);
    assert(!complete_one(owner));
    assert(events.back() == JobState::waiting_for_resources);
    assert(owner.pending() == 1);
    assert(state.close_calls == 1);
    assert(complete_one(owner));
    assert(events.back() == JobState::succeeded);
    assert(state.loads.size() == 2);
    assert(state.loads.back()[1] == 30 && state.loads.back()[2] == 70);
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
        [&](auto, auto ledger, auto) {
            return std::make_unique<ScratchEngine>(state, ledger);
        },
        [&] { return Footprint{++probes == 2 ? 5u : 100u, 80, 70}; },
        [&](auto, auto event, auto) { events.push_back(event); }
    );
    auto load = request("load", Workload::llm);
    load.model = "same";
    load.operation = Operation::load;
    owner.submit(load);
    assert(complete_one(owner));
    assert(events.back() == JobState::succeeded);
    assert(owner.snapshot().capacity[0] == 100);
    auto completion = load;
    completion.id = "completion";
    completion.operation = Operation::completion;
    owner.submit(completion);
    assert(complete_one(owner));
    assert(events.back() == JobState::succeeded);
    assert(state.loads.size() == 1);
    owner.shutdown();
}

void reservation_exhaustion_retries_only_before_execution() {
    for (bool during_execution : {false, true}) {
        FakeState state;
        std::vector<JobState> events;
        // Physical host capacity fits weights but not the complete workspace.
        Footprint free{29, 80, 70};
        InferenceOwner owner(
            {100, 100, 100},
            [&](auto, auto ledger, auto) {
                return std::make_unique<ScratchEngine>(state, ledger, during_execution);
            },
            [&] { return free; },
            [&](auto, auto event, auto) { events.push_back(event); }
        );
        owner.submit(request("head"));
        owner.submit(request("tail"));
        assert(complete_one(owner) == during_execution);
        assert(owner.pending() == (during_execution ? 1u : 2u));
        assert(
            events.back()
            == (during_execution ? JobState::failed : JobState::waiting_for_resources)
        );
        assert(state.close_calls == 1 && owner.snapshot().residents == 0);
        free[0] = 100;
        assert(complete_one(owner));
        assert(events.back() == JobState::succeeded);
        assert(
            std::count(state.events.begin(), state.events.end(), "execute:head") == 1
        );
        if (!during_execution) {
            assert(owner.cancel("tail"));
            assert(complete_one(owner));
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
        [&](auto, auto ledger, auto) {
            return std::make_unique<FakeEngine>(state, ledger);
        },
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
    assert(complete_one(owner));
    assert(prepared == std::vector<std::string>{"head"});
    assert(events.back() == JobState::succeeded);
    assert(complete_one(owner));
    assert(events.back() == JobState::cancelled);
    assert(complete_one(owner));
    assert(events.back() == JobState::failed);
    assert(complete_one(owner));
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
        [&](auto, auto ledger, auto) {
            return std::make_unique<FakeEngine>(state, ledger);
        },
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
    assert(complete_one(owner));
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
        [&](auto, auto ledger, auto) {
            return std::make_unique<FakeEngine>(state, ledger);
        },
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
    assert(complete_one(owner));
    assert(prepared == std::vector<std::string>{"download"});
    assert(state.loads.empty() && events.back() == JobState::succeeded);
    assert(complete_one(owner));
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
                complete_one(owner);
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
    assert(!complete_one(owner) && !prepared);
    owner.shutdown();
}

void invalid_engine_input_preserves_status_and_releases_memory() {
    FakeState state;
    state.invalid_input = true;
    Json failure;
    InferenceOwner owner(
        {100, 100, 100},
        [&](auto, auto resources, auto) {
            return std::make_unique<FakeEngine>(state, resources);
        },
        [] { return Footprint{100, 80, 70}; },
        [&](auto, JobState kind, const Json& value) {
            if (kind == JobState::failed)
                failure = value;
        }
    );
    owner.submit(request("invalid-upload"));
    assert(complete_one(owner));
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

struct BatchState : FakeState {
    std::size_t capacity = 2;
    std::vector<std::vector<std::string>> batches;
    std::function<void()> during_execution;
    bool fail_batch = false, image_results = false;
    std::string blocked_preparation;
};

struct BatchEngine : FakeEngine {
    BatchState& batch_state;
    std::vector<std::size_t> devices;

    BatchEngine(BatchState& state, std::shared_ptr<Resources> ledger)
        : FakeEngine(state, std::move(ledger)), batch_state(state) {
    }

    void load(
        const InferenceRequest& request,
        const Footprint& free,
        const std::atomic_bool& cancel
    ) override {
        FakeEngine::load(request, free, cancel);
        for (std::size_t i = 1; i < free.size(); ++i)
            if (free[i])
                devices.push_back(i);
    }

    std::size_t
    batch_capacity(const InferenceRequest&, const Footprint& available) override {
        // A masked-out GPU is not a pressured execution lane.
        return std::any_of(
                   devices.begin(),
                   devices.end(),
                   [&](auto device) { return available[device] < 40; }
               )
                   ? 1
                   : batch_state.capacity;
    }

    void prepare_batch(
        std::span<const InferenceBatchItem> items, const Footprint&
    ) override {
        for (const auto& item : items)
            if (item.request->id == batch_state.blocked_preparation)
                throw AdmissionWait("longer_prompt_does_not_fit");
    }

    std::vector<InferenceBatchResult>
    execute_batch(std::span<const InferenceBatchItem> items) override {
        std::vector<std::string> ids;
        std::vector<InferenceBatchResult> results;
        for (const auto& item : items) {
            ids.push_back(item.request->id);
            const auto seed = std::get<ImageRequest>(item.request->input).seed.value();
            results.push_back(
                {batch_state.image_results ? Json{{"image", {{"seed", seed}}}}
                                           : Json(seed),
                 nullptr}
            );
        }
        batch_state.batches.push_back(ids);
        if (batch_state.during_execution)
            batch_state.during_execution();
        if (batch_state.fail_batch)
            throw std::runtime_error("shared_denoising_failed");
        return results;
    }
};

InferenceRequest image_request(std::string id, std::uint64_t seed = 1) {
    auto result = request(id);
    result.model = "image-model";
    ImageRequest image;
    image.prompt = id;
    image.width = image.height = 256;
    image.steps = 8;
    image.seed = seed;
    image.count = 1;
    result.input = image;
    return result;
}

void throughput_target_controls_collection_and_advice() {
    struct ThroughputEngine : BatchEngine {
        image::ImageBatchPerformance& performance;
        std::vector<image::ImageBatchPlan>& plans;

        ThroughputEngine(
            BatchState& state,
            std::shared_ptr<Resources> ledger,
            image::ImageBatchPerformance& policy,
            std::vector<image::ImageBatchPlan>& candidates
        )
            : BatchEngine(state, ledger), performance(policy), plans(candidates) {
        }

        std::size_t batch_capacity(const InferenceRequest&, const Footprint&) override {
            return performance.capacity(plans);
        }
    };

    BatchState state;
    state.image_results = true;
    image::ImageBatchPerformance performance;
    std::vector<image::ImageBatchPlan> plans(8, {{10, 10, 10}, {0, std::nullopt}});
    performance.observe(1, 3, 0, 2);
    state.during_execution = [&] {
        performance.observe(2, 5, 0, 4);
    };
    std::vector<std::size_t> hints;
    const auto now = std::chrono::steady_clock::time_point{};
    InferenceOwner owner(
        {100, 100, 100},
        [&](auto, auto ledger, auto) {
            return std::make_unique<ThroughputEngine>(
                state, ledger, performance, plans
            );
        },
        [] { return Footprint{100, 80, 70}; },
        [&](auto, auto event, auto value) {
            if (event == JobState::succeeded)
                hints.push_back(value.at("next_batch_capacity"));
        },
        32,
        {},
        [&] { return now; }
    );
    owner.submit(image_request("a"));
    owner.submit(image_request("b"));
    assert(complete_one(owner));
    assert((hints == std::vector<std::size_t>{3, 3}));
    owner.submit(image_request("c"));
    owner.submit(image_request("d"));
    assert(!complete_one(owner));
    // A bank-displacing third image no longer pays. Collection must immediately
    // execute the selected pair, without waiting for the ten-second deadline.
    plans[2].cost.loading_seconds = 30;
    assert(complete_one(owner));
    assert(
        (state.batches == std::vector<std::vector<std::string>>{{"a", "b"}, {"c", "d"}})
    );
    assert((hints == std::vector<std::size_t>{3, 3, 2, 2}));
    assert(owner.pending() == 0);
    owner.shutdown();
}

void image_capacity_hint_uses_successful_measurement_and_fresh_pressure() {
    for (const auto devices : {1, 2, 3}) {
        for (const auto mode : {0, 1, 2, 3}) {
            BatchState state;
            state.capacity = 1;
            state.image_results = true;
            bool executed = false;
            Json delivered;
            Footprint free(devices + 1, 100);
            for (int device = 0; device < devices; ++device)
                free[device + 1] = 80 - device * 10;
            state.during_execution = [&] {
                executed = true;
                state.capacity = 5;
                if (mode == 1)
                    free[1] = 30;
                if (mode == 3)
                    state.capacity = 100;
            };
            InferenceOwner owner(
                Footprint(devices + 1, 100),
                [&](auto, auto ledger, auto) {
                    return std::make_unique<BatchEngine>(state, ledger);
                },
                [&] {
                    if (executed && mode == 2)
                        throw std::runtime_error("physical_probe_unavailable");
                    return free;
                },
                [&](auto, auto event, auto value) {
                    if (event == JobState::succeeded)
                        delivered = value;
                }
            );
            owner.submit(image_request("capacity", 42));
            assert(complete_one(owner));
            assert(delivered.at("image").at("seed") == 42);
            if (mode == 2)
                assert(!delivered.contains("next_batch_capacity"));
            else
                assert(
                    delivered.at("next_batch_capacity")
                    == (mode == 1   ? 1
                        : mode == 3 ? 32
                                    : 5)
                );
            assert(owner.pending() == 0);
            owner.shutdown();
        }
    }
}

void cancellation_during_capacity_probe_cleans_up_before_terminals() {
    for (const bool fail_probe : {false, true}) {
        for (const bool fail_cleanup : {false, true}) {
            BatchState state;
            state.image_results = true;
            bool executed = false, cancelled_in_probe = false;
            std::vector<std::pair<std::string, JobState>> terminal;
            state.during_execution = [&] {
                executed = true;
            };
            InferenceOwner owner(
                {100, 100, 100},
                [&](auto, auto ledger, auto) {
                    return std::make_unique<BatchEngine>(state, ledger);
                },
                [&] {
                    if (executed) {
                        assert(owner.cancel("first"));
                        cancelled_in_probe = true;
                        state.fail_close = fail_cleanup;
                        if (fail_probe)
                            throw std::runtime_error("physical_probe_unavailable");
                    }
                    return Footprint{100, 80, 70};
                },
                [&](auto id, auto event, auto result) {
                    if (event == JobState::succeeded || event == JobState::cancelled) {
                        const auto snapshot = owner.snapshot();
                        assert(snapshot.residents == 0);
                        assert(snapshot.used == Footprint({0, 0, 0}));
                        terminal.emplace_back(id, event);
                    }
                    if (event == JobState::succeeded) {
                        assert(result.at("image").at("seed") == 91);
                        assert(!result.contains("next_batch_capacity"));
                    }
                }
            );
            owner.submit(image_request("first", 42));
            owner.submit(image_request("second", 91));
            bool cleanup_failed = false;
            try {
                assert(complete_one(owner));
            } catch (const CleanupFailure&) {
                cleanup_failed = true;
            }
            assert(cancelled_in_probe);
            assert(state.close_calls == 1);
            assert(cleanup_failed == fail_cleanup);
            if (fail_cleanup) {
                assert(terminal.empty());
                assert(owner.pending() == 2 && owner.snapshot().residents == 1);
            } else {
                assert(
                    (terminal
                     == std::vector<std::pair<std::string, JobState>>{
                         {"first", JobState::cancelled}, {"second", JobState::succeeded}
                     })
                );
                assert(owner.pending() == 0);
                owner.shutdown();
            }
        }
    }
}

void later_candidate_pressure_does_not_block_the_admissible_head() {
    BatchState state;
    state.blocked_preparation = "blocked";
    std::vector<std::pair<std::string, JobState>> events;
    InferenceOwner owner(
        {100, 100, 100},
        [&](auto, auto ledger, auto) {
            return std::make_unique<BatchEngine>(state, ledger);
        },
        [] { return Footprint{100, 80, 70}; },
        [&](auto id, auto event, auto) { events.emplace_back(id, event); }
    );
    owner.submit(image_request("head"));
    owner.submit(image_request("blocked"));
    owner.submit(image_request("tail"));
    assert(complete_one(owner));
    assert((state.batches == std::vector<std::vector<std::string>>{{"head"}}));
    assert(
        (events.back() == std::pair<std::string, JobState>{"head", JobState::succeeded})
    );
    assert(owner.pending() == 2);
    assert(!complete_one(owner));
    assert((
        events.back()
        == std::pair<std::string, JobState>{"blocked", JobState::waiting_for_resources}
    ));
    assert(owner.pending() == 2 && owner.snapshot().residents == 0);
    assert(state.batches.size() == 1);
    state.blocked_preparation.clear();
    assert(complete_one(owner));
    assert((state.batches.back() == std::vector<std::string>{"blocked", "tail"}));
    assert(owner.pending() == 0);
    owner.shutdown();
}

void full_image_batch_launches_without_waiting_and_preserves_seeds() {
    BatchState state;
    std::vector<std::pair<std::string, Json>> results;
    InferenceOwner owner(
        {100, 100, 100},
        [&](auto, auto ledger, auto) {
            return std::make_unique<BatchEngine>(state, ledger);
        },
        [] { return Footprint{100, 80, 70}; },
        [&](auto id, auto event, auto value) {
            if (event == JobState::succeeded)
                results.emplace_back(id, value);
        }
    );
    owner.submit(image_request("pose-a", 42));
    owner.submit(image_request("pose-b", 91));
    assert(complete_one(owner));
    assert(
        (state.batches == std::vector<std::vector<std::string>>{{"pose-a", "pose-b"}})
    );
    assert(
        (results
         == std::vector<std::pair<std::string, Json>>{{"pose-a", 42}, {"pose-b", 91}})
    );
    assert(owner.pending() == 0);
    owner.shutdown();
}

void image_collection_deadline_uses_oldest_submission() {
    BatchState state;
    auto now = std::chrono::steady_clock::time_point{};
    InferenceOwner owner(
        {100, 100, 100},
        [&](auto, auto ledger, auto) {
            return std::make_unique<BatchEngine>(state, ledger);
        },
        [] { return Footprint{100, 80, 70}; },
        [](auto, auto, auto) {},
        32,
        {},
        [&] { return now; }
    );
    owner.submit(image_request("oldest"));
    assert(!complete_one(owner));
    now += std::chrono::seconds(9);
    assert(!complete_one(owner));
    now += std::chrono::seconds(1);
    assert(complete_one(owner));
    assert((state.batches == std::vector<std::vector<std::string>>{{"oldest"}}));
    owner.shutdown();
}

void cohort_target_only_shortens_collection_waiting() {
    struct Case {
        unsigned capacity, target, first_count, queued, expected;
    };

    for (const auto test :
         {Case{5, 2, 1, 2, 2},
          Case{5, 1, 1, 4, 4},
          Case{2, 32, 1, 3, 2},
          Case{5, 3, 2, 2, 2},
          Case{1, 32, 1, 2, 1}}) {
        BatchState state;
        state.capacity = test.capacity;
        state.image_results = true;
        std::vector<unsigned> advice;
        const auto now = std::chrono::steady_clock::time_point{};
        InferenceOwner owner(
            {100, 100, 100},
            [&](auto, auto ledger, auto) {
                return std::make_unique<BatchEngine>(state, ledger);
            },
            [] { return Footprint{100, 80, 70}; },
            [&](auto, auto event, auto value) {
                if (event == JobState::succeeded)
                    advice.push_back(value.at("next_batch_capacity"));
            },
            32,
            {},
            [&] { return now; }
        );
        for (unsigned i = 0; i < test.queued; ++i) {
            auto request = image_request(std::to_string(i));
            auto& image = std::get<ImageRequest>(request.input);
            image.batch_collection_target = i == 0 ? test.target : i % 2 ? 1 : 32;
            if (i == 0)
                image.count = test.first_count;
            owner.submit(request);
        }
        // The clock never advances: an available cohort launches immediately,
        // but its hint neither caps the prefix nor replaces capacity advice.
        assert(complete_one(owner));
        assert(state.batches.front().size() == test.expected);
        assert(owner.pending() == test.queued - test.expected);
        for (unsigned i = 0; i < test.expected; ++i) {
            assert(state.batches.front()[i] == std::to_string(i));
            assert(advice[i] == test.capacity);
        }
        owner.shutdown();
    }
}

void incomplete_cohort_keeps_deadline_and_fifo_barriers() {
    for (const bool incompatible : {false, true}) {
        BatchState state;
        state.capacity = 5;
        auto now = std::chrono::steady_clock::time_point{};
        InferenceOwner owner(
            {100, 100, 100},
            [&](auto, auto ledger, auto) {
                return std::make_unique<BatchEngine>(state, ledger);
            },
            [] { return Footprint{100, 80, 70}; },
            [](auto, auto, auto) {},
            32,
            {},
            [&] { return now; }
        );
        auto head = image_request("head");
        std::get<ImageRequest>(head.input).batch_collection_target = 3;
        owner.submit(head);
        auto second = image_request("second");
        if (incompatible)
            std::get<ImageRequest>(second.input).width = 512;
        owner.submit(second);
        if (incompatible) {
            owner.submit(image_request("third"));
            assert(complete_one(owner));
            assert((state.batches.front() == std::vector<std::string>{"head"}));
            assert(owner.pending() == 2);
        } else {
            assert(!complete_one(owner));
            now += std::chrono::seconds(9);
            assert(!complete_one(owner));
            now += std::chrono::seconds(1);
            assert(complete_one(owner));
            assert(
                (state.batches.front() == std::vector<std::string>{"head", "second"})
            );
            assert(owner.pending() == 0);
        }
        owner.shutdown();
    }
}

void incompatible_image_head_is_not_overtaken() {
    BatchState state;
    InferenceOwner owner(
        {100, 100, 100},
        [&](auto, auto ledger, auto) {
            return std::make_unique<BatchEngine>(state, ledger);
        },
        [] { return Footprint{100, 80, 70}; },
        [](auto, auto, auto) {}
    );
    owner.submit(image_request("first"));
    auto incompatible = image_request("different-shape");
    std::get<ImageRequest>(incompatible.input).width = 512;
    owner.submit(incompatible);
    owner.submit(image_request("third"));
    assert(complete_one(owner));
    assert(complete_one(owner));
    assert(
        (state.batches
         == std::vector<std::vector<std::string>>{{"first"}, {"different-shape"}})
    );
    assert(owner.pending() == 1);
    owner.shutdown();
}

void cancellation_does_not_cancel_a_batch_sibling() {
    BatchState state;
    state.image_results = true;
    std::vector<std::pair<std::string, JobState>> terminal;
    InferenceOwner owner(
        {100, 100, 100},
        [&](auto, auto ledger, auto) {
            return std::make_unique<BatchEngine>(state, ledger);
        },
        [] { return Footprint{100, 80, 70}; },
        [&](auto id, auto event, auto result) {
            if (event == JobState::succeeded)
                assert(!result.contains("next_batch_capacity"));
            if (event == JobState::succeeded || event == JobState::cancelled)
                terminal.emplace_back(id, event);
        }
    );
    state.during_execution = [&] {
        assert(owner.cancel("first"));
    };
    owner.submit(image_request("first"));
    owner.submit(image_request("second"));
    assert(complete_one(owner));
    assert(
        (terminal
         == std::vector<std::pair<std::string, JobState>>{
             {"first", JobState::cancelled}, {"second", JobState::succeeded}
         })
    );
    assert(state.close_calls == 1);
    owner.shutdown();
}

void failed_batch_preserves_the_waiting_queue() {
    BatchState state;
    state.fail_batch = true;
    std::vector<std::string> failed;
    InferenceOwner owner(
        {100, 100, 100},
        [&](auto, auto ledger, auto) {
            return std::make_unique<BatchEngine>(state, ledger);
        },
        [] { return Footprint{100, 80, 70}; },
        [&](auto id, auto event, auto) {
            if (event == JobState::failed)
                failed.push_back(id);
        }
    );
    owner.submit(image_request("first"));
    owner.submit(image_request("second"));
    owner.submit(image_request("later"));
    assert(complete_one(owner));
    assert((failed == std::vector<std::string>{"first", "second"}));
    assert(owner.pending() == 1 && owner.snapshot().residents == 0);
    assert(owner.cancel("later"));
    assert(complete_one(owner));
    owner.shutdown();
}

void physical_pressure_reduces_batch_before_launch() {
    BatchState state;
    Footprint available{100, 80, 70};
    InferenceOwner owner(
        {100, 100, 100},
        [&](auto, auto ledger, auto) {
            return std::make_unique<BatchEngine>(state, ledger);
        },
        [&] { return available; },
        [](auto, auto, auto) {}
    );
    owner.submit(image_request("first"));
    assert(!complete_one(owner));
    owner.submit(image_request("second"));
    // Pressure every feasible device so re-placement cannot avoid it.
    available[1] = available[2] = 20;
    assert(!complete_one(owner));
    assert(complete_one(owner));
    assert((state.batches == std::vector<std::vector<std::string>>{{"first"}}));
    owner.shutdown();
}

void shared_reference_compatibility_ignores_prompt_and_seed() {
    auto first = image_request("pose-a", 42), second = image_request("pose-b", 43);
    auto& a = std::get<ImageRequest>(first.input);
    auto& b = std::get<ImageRequest>(second.input);
    a.reference = b.reference = true;
    a.source = b.source = ImageUpload{"same-reference", 256, 256};
    a.source_width = b.source_width = a.source_height = b.source_height = 256;
    assert(compatible_images(first, second));
    b.source->png = "other-reference";
    assert(!compatible_images(first, second));
    b.source = a.source;
    b.steps = 10;
    assert(!compatible_images(first, second));
    b.steps = a.steps;
    b.reference = a.reference = false;
    assert(!compatible_images(first, second));
}

void image_batch_counts_images_and_does_not_split_fifo_requests() {
    BatchState state;
    state.capacity = 4;
    InferenceOwner owner(
        {100, 100, 100},
        [&](auto, auto ledger, auto) {
            return std::make_unique<BatchEngine>(state, ledger);
        },
        [] { return Footprint{100, 80, 70}; },
        [](auto, auto, auto) {}
    );
    auto first = image_request("two-images"), second = image_request("another-two");
    std::get<ImageRequest>(first.input).count = 2;
    std::get<ImageRequest>(second.input).count = 2;
    owner.submit(first);
    owner.submit(second);
    assert(complete_one(owner));
    assert(
        (state.batches
         == std::vector<std::vector<std::string>>{{"two-images", "another-two"}})
    );
    owner.submit(image_request("one-image"));
    auto four = image_request("four-images");
    std::get<ImageRequest>(four.input).count = 4;
    owner.submit(four);
    owner.submit(image_request("later-small"));
    assert(complete_one(owner));
    assert((state.batches.back() == std::vector<std::string>{"one-image"}));
    assert(complete_one(owner));
    assert((state.batches.back() == std::vector<std::string>{"four-images"}));
    owner.shutdown();
}

void cancelled_collection_head_unblocks_next_request() {
    BatchState state;
    std::vector<std::pair<std::string, JobState>> events;
    InferenceOwner owner(
        {100, 100, 100},
        [&](auto, auto ledger, auto) {
            return std::make_unique<BatchEngine>(state, ledger);
        },
        [] { return Footprint{100, 80, 70}; },
        [&](auto id, auto event, auto) { events.emplace_back(id, event); }
    );
    owner.submit(image_request("cancelled"));
    assert(!complete_one(owner));
    assert(owner.cancel("cancelled"));
    owner.submit(image_request("first"));
    owner.submit(image_request("second"));
    assert(complete_one(owner));
    assert(
        (events.back()
         == std::pair<std::string, JobState>{"cancelled", JobState::cancelled})
    );
    assert(complete_one(owner));
    assert(
        (state.batches == std::vector<std::vector<std::string>>{{"first", "second"}})
    );
    owner.shutdown();
}

void later_subbatch_failure_preserves_completed_request_outputs() {
    ImageBatchOutput completed{1, {"first/image-0.png"}, nullptr};
    ImageBatchOutput incomplete{2, {"second/image-0.png"}, nullptr};
    ImageBatchOutput waiting{1, {}, nullptr};
    ImageBatchOutput rejected{
        1, {}, {{"error", "invalid_prompt"}, {"status_code", 422}}
    };
    for (auto* output : {&completed, &incomplete, &waiting, &rejected})
        output->fail_if_incomplete("later_subbatch_failed");
    assert(completed.failure.is_null());
    assert(completed.paths.size() == completed.count);
    assert(incomplete.failure == "later_subbatch_failed");
    assert(waiting.failure == "later_subbatch_failed");
    assert(rejected.failure.at("status_code") == 422);
}

void failed_candidate_does_not_change_head_error_status() {
    BatchState state;
    state.fail_batch = true;
    std::vector<std::pair<std::string, Json>> failures;
    InferenceOwner owner(
        {100, 100, 100},
        [&](auto, auto ledger, auto) {
            return std::make_unique<BatchEngine>(state, ledger);
        },
        [] { return Footprint{100, 80, 70}; },
        [&](auto id, auto event, const auto& value) {
            if (event == JobState::failed)
                failures.emplace_back(id, value);
        },
        32,
        [](auto&, auto&) { throw RequestError("invalid_candidate", 422); }
    );
    owner.submit(image_request("head"));
    auto candidate = image_request("candidate");
    candidate.prepared = false;
    owner.submit(candidate);
    assert(complete_one(owner));
    assert(complete_one(owner));
    assert(failures.size() == 2);
    assert(failures[0].first == "head" && failures[0].second.at("status_code") == 502);
    assert(
        failures[1].first == "candidate" && failures[1].second.at("status_code") == 422
    );
    owner.shutdown();
}

void image_memory_growth_is_measured_and_devices_are_independent() {
    constexpr Bytes gib = 1024ULL * 1024 * 1024;
    ImageBatchMemory memory;
    const Footprint free{64 * gib, 24 * gib, 24 * gib};
    const std::vector<Footprint> plans(16, {gib, gib, gib});
    assert(memory.capacity(free, plans) == 1);
    memory.observe(1, {2 * gib, 19 * gib, 4 * gib}, {gib, 2 * gib, 0});
    assert(memory.capacity(free, plans) == 2);
    memory.observe(2, {3 * gib, 21 * gib, 4 * gib}, {gib, 2 * gib, 0});
    assert(memory.capacity(free, plans) == 3);
    memory.observe(3, {4 * gib, 23 * gib, 4 * gib}, {gib, 2 * gib, 0});
    assert(memory.capacity(free, plans) == 3);
    assert(memory.capacity({64 * gib, 22 * gib, 100 * gib}, plans) == 2);
    assert(memory.capacity({3 * gib, 100 * gib, 100 * gib}, plans) == 1);
    assert(memory.capacity(free, std::span(plans).first(1)) == 1);

    ImageBatchMemory replicas;
    replicas.observe(1, {2 * gib, 15 * gib, 0}, {gib, gib, 0});
    replicas.observe(2, {3 * gib, 15 * gib, 15 * gib}, {gib, gib, gib});
    // The first second-GPU sample includes fixed weights, not 15 GiB per image.
    assert(replicas.capacity(free, plans) == 3);
    replicas.observe(3, {4 * gib, 16 * gib, 15 * gib}, {gib, gib, gib});
    assert(replicas.capacity(free, plans) == 4);
    assert(replicas.capacity({64 * gib, 24 * gib, 15 * gib}, plans) == 1);
    const std::vector<Footprint> single_device_plans(16, {gib, 18 * gib, 0});
    assert(replicas.capacity({64 * gib, 24 * gib, gib}, single_device_plans) == 4);
    // A current placement's planned floor still bounds growth on its active GPU.
    assert(replicas.capacity({64 * gib, 18 * gib, gib}, single_device_plans) == 1);
}

int main() {
    cohort_target_only_shortens_collection_waiting();
    incomplete_cohort_keeps_deadline_and_fifo_barriers();
    throughput_target_controls_collection_and_advice();
    image_capacity_hint_uses_successful_measurement_and_fresh_pressure();
    cancellation_during_capacity_probe_cleans_up_before_terminals();
    later_candidate_pressure_does_not_block_the_admissible_head();
    later_subbatch_failure_preserves_completed_request_outputs();
    failed_candidate_does_not_change_head_error_status();
    cancelled_collection_head_unblocks_next_request();
    image_batch_counts_images_and_does_not_split_fifo_requests();
    image_memory_growth_is_measured_and_devices_are_independent();
    shared_reference_compatibility_ignores_prompt_and_seed();
    physical_pressure_reduces_batch_before_launch();
    failed_batch_preserves_the_waiting_queue();
    cancellation_does_not_cancel_a_batch_sibling();
    incompatible_image_head_is_not_overtaken();
    image_collection_deadline_uses_oldest_submission();
    full_image_batch_launches_without_waiting_and_preserves_seeds();
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
    workload_queues_share_ledger();
    admission_retries_head();
    cleanup_failure_stops_owner();
    reused_model_prepares_each_request();
    terminal_delivery_never_replays();
    std::cout << "Inference owner fake-engine units passed\n";
}
