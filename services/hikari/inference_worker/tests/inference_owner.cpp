#include "kadan/inference_owner.hpp"

#include <cassert>
#include <iostream>
#include <thread>
#include <tuple>

#include "kadan/image_batch_memory.hpp"
#include "kadan/image_batch_output.hpp"
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

struct BatchState : FakeState {
    std::size_t capacity = 2;
    std::vector<std::vector<std::string>> batches;
    std::function<void()> during_execution;
    bool fail_batch = false;
    std::string blocked_preparation;
};

struct BatchEngine : FakeEngine {
    BatchState& batch_state;

    BatchEngine(BatchState& state, std::shared_ptr<Resources> ledger)
        : FakeEngine(state, std::move(ledger)), batch_state(state) {
    }

    std::size_t
    batch_capacity(const InferenceRequest&, const Footprint& available) override {
        // Tests can inject physical pressure independently for either device.
        return available[1] < 40 || available[2] < 40 ? 1 : batch_state.capacity;
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
            results.push_back(
                {std::get<ImageRequest>(item.request->input).seed.value(), nullptr}
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

void later_candidate_pressure_does_not_block_the_admissible_head() {
    BatchState state;
    state.blocked_preparation = "blocked";
    std::vector<std::pair<std::string, JobState>> events;
    InferenceOwner owner(
        {100, 100, 100},
        [&](auto, auto ledger) { return std::make_unique<BatchEngine>(state, ledger); },
        [] { return Footprint{100, 80, 70}; },
        [&](auto id, auto event, auto) { events.emplace_back(id, event); }
    );
    owner.submit(image_request("head"));
    owner.submit(image_request("blocked"));
    owner.submit(image_request("tail"));
    assert(owner.advance());
    assert((state.batches == std::vector<std::vector<std::string>>{{"head"}}));
    assert(
        (events.back() == std::pair<std::string, JobState>{"head", JobState::succeeded})
    );
    assert(owner.pending() == 2);
    assert(!owner.advance());
    assert((
        events.back()
        == std::pair<std::string, JobState>{"blocked", JobState::waiting_for_resources}
    ));
    assert(owner.pending() == 2 && owner.snapshot().residents == 0);
    assert(state.batches.size() == 1);
    state.blocked_preparation.clear();
    assert(owner.advance());
    assert((state.batches.back() == std::vector<std::string>{"blocked", "tail"}));
    assert(owner.pending() == 0);
    owner.shutdown();
}

void full_image_batch_launches_without_waiting_and_preserves_seeds() {
    BatchState state;
    std::vector<std::pair<std::string, Json>> results;
    InferenceOwner owner(
        {100, 100, 100},
        [&](auto, auto ledger) { return std::make_unique<BatchEngine>(state, ledger); },
        [] { return Footprint{100, 80, 70}; },
        [&](auto id, auto event, auto value) {
            if (event == JobState::succeeded)
                results.emplace_back(id, value);
        }
    );
    owner.submit(image_request("pose-a", 42));
    owner.submit(image_request("pose-b", 91));
    assert(owner.advance());
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
        [&](auto, auto ledger) { return std::make_unique<BatchEngine>(state, ledger); },
        [] { return Footprint{100, 80, 70}; },
        [](auto, auto, auto) {},
        32,
        {},
        [&] { return now; }
    );
    owner.submit(image_request("oldest"));
    assert(!owner.advance());
    now += std::chrono::seconds(9);
    assert(!owner.advance());
    now += std::chrono::seconds(1);
    assert(owner.advance());
    assert((state.batches == std::vector<std::vector<std::string>>{{"oldest"}}));
    owner.shutdown();
}

void incompatible_image_head_is_not_overtaken() {
    BatchState state;
    InferenceOwner owner(
        {100, 100, 100},
        [&](auto, auto ledger) { return std::make_unique<BatchEngine>(state, ledger); },
        [] { return Footprint{100, 80, 70}; },
        [](auto, auto, auto) {}
    );
    owner.submit(image_request("first"));
    auto incompatible = image_request("different-shape");
    std::get<ImageRequest>(incompatible.input).width = 512;
    owner.submit(incompatible);
    owner.submit(image_request("third"));
    assert(owner.advance());
    assert(owner.advance());
    assert(
        (state.batches
         == std::vector<std::vector<std::string>>{{"first"}, {"different-shape"}})
    );
    assert(owner.pending() == 1);
    owner.shutdown();
}

void cancellation_does_not_cancel_a_batch_sibling() {
    BatchState state;
    std::vector<std::pair<std::string, JobState>> terminal;
    InferenceOwner owner(
        {100, 100, 100},
        [&](auto, auto ledger) { return std::make_unique<BatchEngine>(state, ledger); },
        [] { return Footprint{100, 80, 70}; },
        [&](auto id, auto event, auto) {
            if (event == JobState::succeeded || event == JobState::cancelled)
                terminal.emplace_back(id, event);
        }
    );
    state.during_execution = [&] {
        assert(owner.cancel("first"));
    };
    owner.submit(image_request("first"));
    owner.submit(image_request("second"));
    assert(owner.advance());
    assert(
        (terminal
         == std::vector<std::pair<std::string, JobState>>{
             {"first", JobState::cancelled}, {"second", JobState::succeeded}
         })
    );
    owner.shutdown();
}

void failed_batch_preserves_the_waiting_queue() {
    BatchState state;
    state.fail_batch = true;
    std::vector<std::string> failed;
    InferenceOwner owner(
        {100, 100, 100},
        [&](auto, auto ledger) { return std::make_unique<BatchEngine>(state, ledger); },
        [] { return Footprint{100, 80, 70}; },
        [&](auto id, auto event, auto) {
            if (event == JobState::failed)
                failed.push_back(id);
        }
    );
    owner.submit(image_request("first"));
    owner.submit(image_request("second"));
    owner.submit(image_request("later"));
    assert(owner.advance());
    assert((failed == std::vector<std::string>{"first", "second"}));
    assert(owner.pending() == 1 && owner.snapshot().residents == 0);
    assert(owner.cancel("later"));
    assert(owner.advance());
    owner.shutdown();
}

void physical_pressure_reduces_batch_before_launch() {
    BatchState state;
    Footprint available{100, 80, 70};
    InferenceOwner owner(
        {100, 100, 100},
        [&](auto, auto ledger) { return std::make_unique<BatchEngine>(state, ledger); },
        [&] { return available; },
        [](auto, auto, auto) {}
    );
    owner.submit(image_request("first"));
    assert(!owner.advance());
    owner.submit(image_request("second"));
    available[2] = 20;
    assert(owner.advance());
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
        [&](auto, auto ledger) { return std::make_unique<BatchEngine>(state, ledger); },
        [] { return Footprint{100, 80, 70}; },
        [](auto, auto, auto) {}
    );
    auto first = image_request("two-images"), second = image_request("another-two");
    std::get<ImageRequest>(first.input).count = 2;
    std::get<ImageRequest>(second.input).count = 2;
    owner.submit(first);
    owner.submit(second);
    assert(owner.advance());
    assert(
        (state.batches
         == std::vector<std::vector<std::string>>{{"two-images", "another-two"}})
    );
    owner.submit(image_request("one-image"));
    auto four = image_request("four-images");
    std::get<ImageRequest>(four.input).count = 4;
    owner.submit(four);
    owner.submit(image_request("later-small"));
    assert(owner.advance());
    assert((state.batches.back() == std::vector<std::string>{"one-image"}));
    assert(owner.advance());
    assert((state.batches.back() == std::vector<std::string>{"four-images"}));
    owner.shutdown();
}

void cancelled_collection_head_unblocks_next_request() {
    BatchState state;
    std::vector<std::pair<std::string, JobState>> events;
    InferenceOwner owner(
        {100, 100, 100},
        [&](auto, auto ledger) { return std::make_unique<BatchEngine>(state, ledger); },
        [] { return Footprint{100, 80, 70}; },
        [&](auto id, auto event, auto) { events.emplace_back(id, event); }
    );
    owner.submit(image_request("cancelled"));
    assert(!owner.advance());
    assert(owner.cancel("cancelled"));
    owner.submit(image_request("first"));
    owner.submit(image_request("second"));
    assert(owner.advance());
    assert(
        (events.back()
         == std::pair<std::string, JobState>{"cancelled", JobState::cancelled})
    );
    assert(owner.advance());
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
        [&](auto, auto ledger) { return std::make_unique<BatchEngine>(state, ledger); },
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
    assert(owner.advance());
    assert(owner.advance());
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
    assert(memory.capacity(free, 16) == 1);
    memory.observe(1, {2 * gib, 19 * gib, 4 * gib}, {gib, 2 * gib, 0});
    assert(memory.capacity(free, 16) == 2);
    memory.observe(2, {3 * gib, 21 * gib, 4 * gib}, {gib, 2 * gib, 0});
    assert(memory.capacity(free, 16) == 3);
    memory.observe(3, {4 * gib, 23 * gib, 4 * gib}, {gib, 2 * gib, 0});
    assert(memory.capacity(free, 16) == 3);
    assert(memory.capacity({64 * gib, 22 * gib, 100 * gib}, 16) == 2);
    assert(memory.capacity({3 * gib, 100 * gib, 100 * gib}, 16) == 1);
    assert(memory.capacity(free, 1) == 1);
}

int main() {
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
    fifo_shares_ledger();
    admission_retries_head();
    cleanup_failure_stops_owner();
    reused_model_prepares_each_request();
    terminal_delivery_never_replays();
    std::cout << "Inference owner fake-engine units passed\n";
}
