#include "kadan/device_workspace.hpp"

#include <iostream>
#include <set>
#include <string>

#include "kadan/device_placement.hpp"
#include "kadan/projection_math.hpp"
using namespace kadan;

namespace {
void check(bool ok) {
    if (!ok)
        throw std::runtime_error("workspace_test_failed");
}

template <class F> void fails(F function) {
    try {
        function();
    } catch (const std::exception&) {
        return;
    }
    throw std::runtime_error("expected_failure");
}

struct Counters {
    Bytes allocated = 0, peak = 0;
    unsigned allocations = 0, frees = 0, creates = 0, destroys = 0, resets = 0;
    std::vector<std::byte> host_storage;
    unsigned host_allocations = 0, host_frees = 0;
    Bytes expected_host_bytes_on_free = 0;
    bool null_host = false;
    bool handle = false;
    unsigned status = 0, status_resets = 0, status_reads = 0;
    std::string fail, cleanup_fail;
    std::vector<std::string> events;
    std::set<std::thread::id> threads;
};

struct FakeOperations final : DeviceOperations {
    Resources& resources;
    std::shared_ptr<Counters> counts;
    int device;

    FakeOperations(Resources& r, std::shared_ptr<Counters> c, int d)
        : resources(r), counts(std::move(c)), device(d) {
    }

    void event(const char* name) {
        counts->events.emplace_back(name);
        counts->threads.insert(std::this_thread::get_id());
        if (counts->fail == name || counts->cleanup_fail == name)
            throw std::runtime_error(name);
    }

    void select(int d) override {
        check(d == device);
        event("select");
    }

    void allocate(Bytes bytes) override {
        check(!counts->allocated && resources.snapshot().used[device + 1] >= bytes);
        event("allocate");
        counts->allocated = bytes;
        counts->status = 0;
        counts->peak = std::max(counts->peak, bytes);
        ++counts->allocations;
    }

    void free() override {
        check(counts->allocated && !counts->handle);
        event("free");
        counts->allocated = 0;
        ++counts->frees;
    }

    void* allocate_host(Bytes bytes) override {
        check(counts->host_storage.empty() && resources.snapshot().used[0] >= bytes);
        event("allocate_host");
        if (counts->null_host)
            return nullptr;
        counts->host_storage.resize(bytes);
        ++counts->host_allocations;
        return counts->host_storage.data();
    }

    void free_host(void* pointer) override {
        check(
            pointer == counts->host_storage.data() && !counts->host_storage.empty()
            && resources.snapshot().used[0] >= counts->host_storage.size()
            && (counts->events.back() == "synchronize"
                || counts->events.back() == "destroy")
        );
        if (counts->expected_host_bytes_on_free)
            check(resources.snapshot().used[0] == counts->expected_host_bytes_on_free);
        event("free_host");
        counts->host_storage.clear();
        ++counts->host_frees;
    }

    // Fake status operations only: no tensor arithmetic or inference.
    void reset_status() {
        check(counts->allocated >= projection_status_bytes);
        event("status_reset");
        counts->status = 0;
        ++counts->status_resets;
    }

    void observe_status(std::uint32_t bits) {
        check(counts->allocated >= projection_status_bytes);
        if (!projection_output_finite(bits))
            counts->status = 1;
    }

    unsigned read_status() {
        event("status_read");
        ++counts->status_reads;
        return counts->status;
    }

    void create_handle() override {
        check(!counts->handle);
        event("create");
        counts->handle = true;
        ++counts->creates;
    }

    void configure_handle() override {
        check(counts->handle);
        event("configure");
    }

    void destroy_handle() override {
        check(counts->handle);
        event("destroy");
        counts->handle = false;
        ++counts->destroys;
    }

    void synchronize() override {
        event("synchronize");
    }

    void reset() override {
        check(!counts->allocated && !counts->handle && counts->host_storage.empty());
        event("reset");
        ++counts->resets;
    }
};

std::unique_ptr<DeviceOperations>
fake(Resources& r, std::shared_ptr<Counters> c, int d = 0) {
    return std::make_unique<FakeOperations>(r, std::move(c), d);
}

void operation_and_cleanup_errors() {
    Resources r({1024, 512});
    auto c = std::make_shared<Counters>();
    DeviceWorkspace w(r, 0, Workload::image, fake(r, c));
    c->fail = "configure";
    c->cleanup_fail = "destroy";
    bool combined = false;
    try {
        w.ensure(128);
    } catch (const OperationCleanupError& e) {
        combined = true;
        check(
            exception_message(e.operation) == "configure"
            && exception_message(e.cleanup) == "destroy"
        );
        check(
            std::string(e.what()).find("configure") != std::string::npos
            && std::string(e.what()).find("destroy") != std::string::npos
        );
    }
    check(combined && w.quarantined() && r.snapshot().used[1] == 128);
    Resources other({1024, 512});
    auto d = std::make_shared<Counters>();
    DeviceThread thread(other, 0, Workload::image, fake(other, d));
    std::atomic_bool failed = false;
    auto result = thread.submit_work(
        [&](DeviceWorkspace& workspace) {
            workspace.ensure(64);
            d->fail = "synchronize";
            throw std::invalid_argument("request_error");
        },
        &failed
    );
    combined = false;
    try {
        result.get();
    } catch (const OperationCleanupError& e) {
        combined = true;
        check(
            exception_message(e.operation) == "request_error"
            && exception_message(e.cleanup) == "synchronize"
        );
    }
    check(combined && failed.load() && other.snapshot().used[1] == 64);
    bool original = false;
    try {
        rethrow_after_cleanup(
            std::make_exception_ptr(std::invalid_argument("original")), [] {}
        );
    } catch (const std::invalid_argument& e) {
        original = std::string(e.what()) == "original";
    }
    check(original);
}

void reuse_growth_and_release() {
    Resources r({1024, 512});
    auto c = std::make_shared<Counters>();
    DeviceWorkspace w(r, 0, Workload::image, fake(r, c));
    w.ensure(128);
    w.synchronize();
    w.ensure(64);
    w.ensure(128);
    check(
        c->allocations == 1 && c->creates == 1 && w.retained_bytes() == 128
        && r.snapshot().used[1] == 128
    );
    w.ensure(512);
    check(
        c->allocations == 2 && c->frees == 1 && c->peak == 512
        && r.snapshot().residents == 1
    );
    w.release();
    check(r.snapshot().residents == 0 && c->allocated == 0 && !c->handle);
    const auto events = c->events.size();
    w.release();
    check(c->events.size() == events + 1);  // Synchronization is harmless/idempotent.
    w.reset();
    check(c->resets == 1);
    const auto after = c->events.size();
    w.reset();
    check(c->events.size() == after);
    w.ensure(32);
    w.reset();
    check(c->resets == 2 && r.snapshot().used[1] == 0);
}

void accounting_and_construction_failures() {
    for (const auto* fault : {"allocate", "create", "configure"}) {
        Resources r({1024, 512});
        auto c = std::make_shared<Counters>();
        c->fail = fault;
        DeviceWorkspace w(r, 0, Workload::tts, fake(r, c));
        fails([&] { w.ensure(128); });
        check(
            !w.quarantined() && r.snapshot().residents == 0 && !c->allocated
            && !c->handle
        );
        c->fail.clear();
        w.ensure(128);
        w.reset();
        check(r.snapshot().residents == 0);
    }
    Resources r({1024, 512});
    auto c = std::make_shared<Counters>();
    DeviceWorkspace w(r, 0, Workload::tts, fake(r, c));
    auto other = r.reserve(Workload::llm, {0, 256});
    w.ensure(128);
    fails([&] { w.ensure(300); });
    check(!w.quarantined() && r.snapshot().used[1] == 256 && c->peak == 128);
    r.released(other);
    w.ensure(512);
    w.release();
    check(r.snapshot().residents == 0);
    fails([&] { w.ensure(513); });
    check(r.snapshot().residents == 0);
}

void host_staging_reuse_and_growth() {
    constexpr auto capacity = DeviceWorkspace::host_staging_capacity;
    Resources r({capacity, 512});
    auto c = std::make_shared<Counters>();
    DeviceWorkspace w(r, 0, Workload::llm, fake(r, c));
    w.ensure(128);
    check(c->host_allocations == 0 && r.snapshot().used[0] == 0);
    auto staging = w.host_staging(110);
    check(staging.size() == 110 && staging.data() == c->host_storage.data());
    staging.front() = std::byte{42};
    check(r.snapshot().used == Footprint({capacity, 128}));
    check(w.host_staging(capacity).data() == staging.data());
    w.ensure(512);
    check(
        w.host_staging(34).data() == staging.data()
        && w.host_staging(34).front() == std::byte{42} && c->host_allocations == 1
        && c->host_frees == 0 && r.snapshot().used == Footprint({capacity, 512})
    );
    w.release();
    check(
        c->host_storage.empty() && c->host_frees == 1 && c->frees == 2
        && r.snapshot().residents == 0
    );
    w.release();
    check(c->host_frees == 1);
}

void packed_reader_workspace_lifetime() {
    constexpr auto staging_bytes = DeviceWorkspace::host_staging_capacity;
    const auto team_bytes = PackedReadTeam::host_bytes();
    Resources resources({team_bytes + staging_bytes + 4096, 1024});
    auto counts = std::make_shared<Counters>();
    DeviceWorkspace workspace(resources, 0, Workload::llm, fake(resources, counts));
    check(resources.snapshot().used == Footprint({0, 0}));
    workspace.ensure(128);
    check(resources.snapshot().used == Footprint({0, 128}));
    auto* team = &workspace.packed_read_team();
    check(resources.snapshot().used == Footprint({team_bytes, 128}));
    auto staging = workspace.host_staging(32);
    staging.front() = std::byte{42};
    check(resources.snapshot().used == Footprint({team_bytes + staging_bytes, 128}));
    workspace.ensure(512);
    check(&workspace.packed_read_team() == team);
    check(workspace.host_staging(32).data() == staging.data());
    check(workspace.host_staging(32).front() == std::byte{42});
    check(resources.snapshot().used == Footprint({team_bytes + staging_bytes, 512}));
    unsigned reads = 0;
    team->read(
        [&](std::size_t first, std::span<std::byte> destination) {
            ++reads;
            check(
                first == 9 && destination.data() == staging.data()
                && destination.size() == 32
            );
            check(resources.snapshot().used[0] == team_bytes + staging_bytes);
            std::fill(destination.begin(), destination.end(), std::byte{7});
        },
        9,
        8,
        staging
    );
    check(reads == 1 && staging.back() == std::byte{7});
    // free_host runs before its own ledger reservation is retired. The reader
    // team's separate charge must already be gone when this fake acknowledges it.
    counts->expected_host_bytes_on_free = staging_bytes;
    workspace.release();
    check(counts->host_frees == 1 && resources.snapshot().used == Footprint({0, 0}));
    check(resources.snapshot().residents == 0);

    auto& recreated = workspace.packed_read_team();
    check(resources.snapshot().used == Footprint({team_bytes, 0}));
    auto next = workspace.host_staging(8);
    recreated.read(
        [&](std::size_t first, std::span<std::byte> destination) {
            ++reads;
            check(first == 3 && destination.size() == 8);
            check(resources.snapshot().used[0] == team_bytes + staging_bytes);
            destination.front() = std::byte{11};
        },
        3,
        8,
        next
    );
    check(reads == 2 && next.front() == std::byte{11});
    workspace.release();
    check(counts->host_allocations == 2 && counts->host_frees == 2);
    check(
        resources.snapshot().used == Footprint({0, 0})
        && resources.snapshot().residents == 0
    );
}

void host_staging_admission_and_allocation_failures() {
    constexpr auto capacity = DeviceWorkspace::host_staging_capacity;
    Resources r({capacity, 512});
    auto c = std::make_shared<Counters>();
    DeviceWorkspace w(r, 0, Workload::llm, fake(r, c));
    fails([&] { w.host_staging(0); });
    fails([&] { w.host_staging(capacity + 1); });
    check(c->events.empty() && r.snapshot().residents == 0);
    auto other = r.reserve(Workload::image, {1, 0});
    fails([&] { w.host_staging(1); });
    check(c->host_allocations == 0 && r.snapshot().used == Footprint({1, 0}));
    r.released(other);
    for (bool null_result : {false, true}) {
        c->fail = null_result ? "" : "allocate_host";
        c->null_host = null_result;
        fails([&] { w.host_staging(144); });
        check(
            !w.quarantined() && c->host_storage.empty() && r.snapshot().residents == 0
        );
    }
    c->fail.clear();
    c->null_host = false;
    check(w.host_staging(144).size() == 144);
    w.release();
    check(c->host_allocations == 1 && c->host_frees == 1);
}

void host_staging_uncertain_cleanup_stays_charged() {
    constexpr auto capacity = DeviceWorkspace::host_staging_capacity;
    for (const auto* fault : {"synchronize", "free_host"}) {
        Resources r({capacity, 512});
        auto c = std::make_shared<Counters>();
        DeviceWorkspace w(r, 0, Workload::llm, fake(r, c));
        auto pointer = w.host_staging(110).data();
        w.ensure(128);
        c->fail = fault;
        fails([&] { w.release(); });
        check(
            w.quarantined() && c->host_storage.data() == pointer
            && c->host_storage.size() == capacity && c->host_frees == 0
            && c->allocated == 128 && r.snapshot().used == Footprint({capacity, 128})
            && r.snapshot().residents == 2
        );
        const auto events = c->events.size();
        c->fail.clear();
        fails([&] { w.host_staging(34); });
        fails([&] { w.reset(); });
        check(c->events.size() == events && r.snapshot().used[0] == capacity);
    }
}

void host_staging_owned_threads_and_cancel() {
    constexpr auto capacity = DeviceWorkspace::host_staging_capacity;
    Resources r({2 * capacity, 512, 512});
    auto a = std::make_shared<Counters>(), b = std::make_shared<Counters>();
    {
        DeviceThread first(r, 0, Workload::llm, fake(r, a, 0)),
            second(r, 1, Workload::llm, fake(r, b, 1));
        auto x = first.submit_work([](DeviceWorkspace& w) { w.host_staging(110); });
        auto y = second.submit_work([](DeviceWorkspace& w) { w.host_staging(144); });
        x.get();
        y.get();
        check(
            a->host_storage.data() != b->host_storage.data()
            && r.snapshot().used == Footprint({2 * capacity, 0, 0})
        );
        auto pointer = b->host_storage.data();
        std::atomic_bool failed = false;
        auto cancelled = first.submit_work(
            [](DeviceWorkspace& w) {
                w.host_staging(34);
                throw std::runtime_error("cancelled");
            },
            &failed
        );
        fails([&] { cancelled.get(); });
        check(
            failed.load() && a->host_storage.empty() && a->host_frees == 1
            && b->host_storage.data() == pointer && b->host_frees == 0
            && r.snapshot().used == Footprint({capacity, 0, 0})
        );
    }
    check(
        r.snapshot().residents == 0 && b->host_frees == 1 && a->threads.size() == 1
        && b->threads.size() == 1 && *a->threads.begin() != *b->threads.begin()
    );
}

void uncertain_cleanup_stays_charged() {
    for (const auto* fault : {"synchronize", "destroy", "free"}) {
        Resources r({1024, 512});
        auto c = std::make_shared<Counters>();
        DeviceWorkspace w(r, 0, Workload::image, fake(r, c));
        w.ensure(128);
        c->fail = fault;
        fails([&] { w.release(); });
        check(w.quarantined() && r.snapshot().used[1] == 128);
        const auto events = c->events.size();
        c->fail.clear();
        fails([&] { w.ensure(64); });
        fails([&] { w.reset(); });
        check(c->events.size() == events && r.snapshot().residents == 1);
    }
    // Growth must also quarantine a failed destruction rather than retry it.
    Resources r({1024, 512});
    auto c = std::make_shared<Counters>();
    DeviceWorkspace w(r, 0, Workload::image, fake(r, c));
    w.ensure(128);
    c->fail = "destroy";
    fails([&] { w.ensure(256); });
    check(w.quarantined() && r.snapshot().used[1] == 128 && c->allocations == 1);
}

void reset_failure_keeps_context_accounting() {
    Resources r({1024, 512});
    auto context = r.reserve(Workload::tts, {0, 64});
    auto c = std::make_shared<Counters>();
    DeviceWorkspace w(r, 0, Workload::tts, fake(r, c));
    w.ensure(128);
    c->fail = "reset";
    fails([&] { w.reset(); });
    check(w.quarantined() && r.snapshot().used[1] == 64 && r.snapshot().residents == 1);
    // The owner has received no reset acknowledgement and must not release context.
    check(r.reservation(context).bytes[1] == 64);
}

void persistent_threads_and_cancel() {
    Resources r({1024, 512, 512});
    auto a = std::make_shared<Counters>(), b = std::make_shared<Counters>();
    {
        DeviceThread first(r, 0, Workload::image, fake(r, a, 0)),
            second(r, 1, Workload::image, fake(r, b, 1));
        for (int i = 0; i < 3; ++i) {
            auto x = first.submit_work([](DeviceWorkspace& w) { w.ensure(128); });
            auto y = second.submit_work([](DeviceWorkspace& w) { w.ensure(256); });
            x.get();
            y.get();
        }
        check(
            a->threads.size() == 1 && b->threads.size() == 1
            && *a->threads.begin() != *b->threads.begin()
        );
        check(
            a->allocations == 1 && b->allocations == 1
            && r.snapshot().used == Footprint({0, 128, 256})
        );
        std::atomic_bool failed = false;
        auto cancelled = first.submit_work(
            [](DeviceWorkspace& w) {
                w.ensure(64);
                throw std::runtime_error("cancelled");
            },
            &failed
        );
        fails([&] { cancelled.get(); });
        check(
            failed.load() && !a->allocated && r.snapshot().used[1] == 0
            && r.snapshot().used[2] == 256
        );
        second.submit([](DeviceWorkspace& w) { w.release(); }).get();
        check(r.snapshot().residents == 0);
        first.submit_work([](DeviceWorkspace& w) { w.ensure(64); }).get();
        first.submit([](DeviceWorkspace& w) { w.reset(); }).get();
        second.submit([](DeviceWorkspace& w) { w.reset(); }).get();
        check(
            a->threads.size() == 1 && b->threads.size() == 1 && a->resets == 1
            && b->resets == 1
        );
        // Destruction drains a pending job before releasing its resources.
        first.submit_work([](DeviceWorkspace& w) { w.ensure(64); });
    }
    check(r.snapshot().residents == 0 && a->threads.size() == 1);
}

void projection_status_lifecycle() {
    Resources r({1024, 1024, 1024});
    auto a = std::make_shared<Counters>(), b = std::make_shared<Counters>();
    {
        DeviceThread first(r, 0, Workload::image, fake(r, a, 0)),
            second(r, 1, Workload::image, fake(r, b, 1));
        // Complete admission precedes reset; smaller later layouts cannot discard
        // a flag set by an early tile. A fresh projection resets reused scratch.
        for (bool invalid : {true, false}) {
            auto task = [invalid](DeviceWorkspace& w) {
                w.ensure(768);
                auto& ops = static_cast<FakeOperations&>(w.operations());
                ops.reset_status();
                ops.observe_status(invalid ? 0x7f800000U : 0U);
                w.ensure(512);
                ops.observe_status(0x80000001U);
                w.ensure(256);
                ops.observe_status(0x80000000U);
                check(ops.read_status() == unsigned(invalid));
            };
            auto x = first.submit_work(task), y = second.submit_work(task);
            x.get();
            y.get();
            check(r.snapshot().used == Footprint({0, 768, 768}));
        }
        check(a->allocations == 1 && b->allocations == 1);
        check(a->status_resets == 2 && b->status_resets == 2);
        check(a->status_reads == 2 && b->status_reads == 2);
        std::atomic_bool failed = false, nonfinite = false;
        std::promise<void> status_read;
        auto ready = status_read.get_future();
        auto slow = first.submit_work(
            [&](DeviceWorkspace& w) {
                ready.wait();
                check(!failed.load());
                auto& ops = static_cast<FakeOperations&>(w.operations());
                ops.reset_status();
                ops.observe_status(1U);
                if (ops.read_status())
                    nonfinite = true;
            },
            &failed
        );
        auto invalid = second.submit_work(
            [&](DeviceWorkspace& w) {
                auto& ops = static_cast<FakeOperations&>(w.operations());
                ops.reset_status();
                ops.observe_status(0xff800000U);
                if (ops.read_status())
                    nonfinite = true;
                status_read.set_value();
            },
            &failed
        );
        // Report validation only after both futures join: an invalid early device
        // must not cancel the slower device and mask the nonfinite error.
        slow.get();
        invalid.get();
        try {
            if (nonfinite.load())
                throw std::runtime_error("h3_cuda_nonfinite_output");
            check(false);
        } catch (const std::runtime_error& e) {
            check(std::string(e.what()) == "h3_cuda_nonfinite_output");
        }
        check(!failed.load() && r.snapshot().used == Footprint({0, 768, 768}));
        first.submit([](DeviceWorkspace& w) { w.release(); }).get();
        second.submit([](DeviceWorkspace& w) { w.release(); }).get();
        check(r.snapshot().residents == 0);
        // Status-copy errors take the same cleanup path; uncertain cleanup stays
        // charged, including the aligned scalar's allocation.
        a->fail = "status_read";
        a->cleanup_fail = "synchronize";
        auto copy_failure = first.submit_work([](DeviceWorkspace& w) {
            w.ensure(768);
            auto& ops = static_cast<FakeOperations&>(w.operations());
            ops.reset_status();
            ops.read_status();
        });
        try {
            copy_failure.get();
            check(false);
        } catch (const OperationCleanupError& e) {
            check(exception_message(e.operation) == "status_read");
            check(exception_message(e.cleanup) == "synchronize");
        }
        check(r.snapshot().used == Footprint({0, 768, 0}));
    }
    check(r.snapshot().used == Footprint({0, 768, 0}));
}

void selection_failure_never_uses_default_device() {
    Resources r({1024, 512});
    auto c = std::make_shared<Counters>();
    c->fail = "select";
    DeviceWorkspace w(r, 0, Workload::tts, fake(r, c));
    fails([&] { w.ensure(128); });
    check(
        w.quarantined() && c->events == std::vector<std::string>{"select"}
        && r.snapshot().residents == 0
    );
}

void destruction_does_not_acknowledge_failed_cleanup() {
    Resources r({1024, 512});
    auto c = std::make_shared<Counters>();
    {
        DeviceThread thread(r, 0, Workload::image, fake(r, c));
        thread.submit_work([](DeviceWorkspace& w) { w.ensure(128); }).get();
        c->fail = "synchronize";
    }
    check(r.snapshot().used[1] == 128 && c->allocated == 128 && c->frees == 0);
}

void telemetry_context_follows_device_tasks() {
    Resources resources({1024, 512});
    auto counts = std::make_shared<Counters>();
    DeviceThread device(resources, 0, Workload::image, fake(resources, counts));
    {
        serving::CudaPhaseScope phase(true);
        const auto& context = phase.context();
        device
            .submit_work([context](DeviceWorkspace&) {
                check(serving::cuda_phase == context);
                serving::cuda_phase->metrics.observed = true;
                serving::cuda_phase->metrics.sync.calls = 1;
            })
            .get();
        check(phase.snapshot().sync.calls == 1);
    }
    device.submit([](DeviceWorkspace&) { check(!serving::cuda_phase); }).get();
}

void bounded_mailbox() {
    Resources r({1024, 512});
    auto c = std::make_shared<Counters>();
    DeviceThread t(r, 0, Workload::tts, fake(r, c));
    std::promise<void> entered, finish;
    auto finished = finish.get_future();
    auto active = t.submit([&](DeviceWorkspace&) {
        entered.set_value();
        finished.wait();
    });
    entered.get_future().get();
    auto pending = t.submit([](DeviceWorkspace&) {});
    bool rejected = false;
    try {
        t.submit([](DeviceWorkspace&) {});
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    finish.set_value();
    active.get();
    pending.get();
    check(rejected);
}

void joining_failed_devices() {
    // A fake resource task fails while another still owns caller storage.
    // Joining must release both lifetimes before propagating the first error.
    const image::VaeConvolutionTile group{{1, 2, 4097, 3, 1}, 0, 4096, 4096};
    const auto plan =
        *image_vae_convolution_plan(group.shape, group.shape.outputs, 64 * 1024 * 1024);
    check(plan.points == group.capacity && plan.projection.rows == 256);
    for (bool submission_failed : {false, true}) {
        Resources resources({1024, plan.bytes, plan.bytes});
        auto a = std::make_shared<Counters>(), b = std::make_shared<Counters>();
        DeviceThread first(resources, 0, Workload::image, fake(resources, a, 0)),
            second(resources, 1, Workload::image, fake(resources, b, 1));
        std::atomic_bool failed = false;
        std::promise<void> entered, finish;
        auto completed = finish.get_future();
        int callback_owned = 17;
        std::vector<std::future<void>> tasks;
        tasks.push_back(first.submit_work(
            [&](DeviceWorkspace& workspace) {
                workspace.ensure(plan.bytes);
                // Cancellation/error after an inner tile keeps the full group
                // charged until owned cleanup, without another allocation.
                workspace.ensure(256);
                throw std::runtime_error("pair_cancelled_or_gemm_error");
            },
            &failed
        ));
        tasks.push_back(second.submit_work([&](DeviceWorkspace& workspace) {
            workspace.ensure(plan.bytes);
            auto& ops = static_cast<FakeOperations&>(workspace.operations());
            ops.reset_status();
            entered.set_value();
            completed.wait();
            check(callback_owned == 17 && group.count == 4096);
            for (std::size_t at = 0; at < group.count; at += plan.projection.rows) {
                workspace.ensure(256);
                ops.observe_status(at == 0 ? 0x7f800000U : 0x80000000U);
            }
            check(ops.read_status() == 1);
        }));
        entered.get_future().get();
        auto error =
            submission_failed
                ? std::make_exception_ptr(std::runtime_error("submission_failed"))
                : std::exception_ptr{};
        finish.set_value();
        join_device_tasks(tasks, error);
        check(!tasks[0].valid() && !tasks[1].valid());
        check(error && failed.load() && !a->allocated && b->allocated == plan.bytes);
        check(
            a->allocations == 1 && b->allocations == 1 && b->status_resets == 1
            && b->status_reads == 1
        );
        callback_owned = 0;
        try {
            std::rethrow_exception(error);
        } catch (const std::runtime_error& e) {
            check(
                std::string(e.what())
                == (submission_failed ? "submission_failed"
                                      : "pair_cancelled_or_gemm_error")
            );
        }
        second.submit([](DeviceWorkspace& workspace) { workspace.release(); }).get();
        check(resources.snapshot().residents == 0);
    }
}
}  // namespace

int main() {
    operation_and_cleanup_errors();
    reuse_growth_and_release();
    accounting_and_construction_failures();
    host_staging_reuse_and_growth();
    packed_reader_workspace_lifetime();
    host_staging_admission_and_allocation_failures();
    host_staging_uncertain_cleanup_stays_charged();
    host_staging_owned_threads_and_cancel();
    uncertain_cleanup_stays_charged();
    reset_failure_keeps_context_accounting();
    persistent_threads_and_cancel();
    projection_status_lifecycle();
    selection_failure_never_uses_default_device();
    destruction_does_not_acknowledge_failed_cleanup();
    telemetry_context_follows_device_tasks();
    bounded_mailbox();
    joining_failed_devices();
    std::cout << "Mock device workspace lifecycle passed (no CUDA or inference)\n";
}
