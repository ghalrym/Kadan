#include "kadan/media_session.hpp"

#include <array>
#include <iostream>
#include <map>
#include <source_location>

#include "kadan/cleanup_error.hpp"
#include "kadan/worker_compute.hpp"
using namespace kadan;

namespace {
void check(bool ok, std::source_location loc = std::source_location::current()) {
    if (!ok)
        throw std::runtime_error("media_session_test:" + std::to_string(loc.line()));
}

template <class F> std::string fails(F fn) {
    try {
        fn();
    } catch (const std::exception& e) {
        return e.what();
    }
    throw std::runtime_error("media_session_expected_failure");
}

struct Trace {
    std::map<int, Bytes> free{{0, 1000000}, {1, 1000000}};
    std::map<void*, std::pair<int, Bytes>> allocations;
    std::vector<int> drains;
    std::vector<std::pair<int, MediaEvent>> waits;
    std::size_t allocations_started = 0, releases = 0;
    bool peer01 = true, peer10 = true;
    int fail_allocate = -1, fail_drain = -1, fail_free = -1;
    bool fail_record = false;
    std::uint32_t status = 0;
};

struct Fake final : MediaDeviceOperations {
    std::shared_ptr<Trace> trace;
    std::shared_ptr<Resources> resources;
    std::map<std::uint64_t, int> events;
    std::uint64_t next = 1;

    explicit Fake(std::shared_ptr<Trace> t, std::shared_ptr<Resources> r)
        : trace(std::move(t)), resources(std::move(r)) {
    }

    std::size_t allocation_alignment() const override {
        return 256;
    }

    void* stream(int) override {
        return nullptr;
    }

    Bytes free_memory(int d) override {
        return trace->free.at(d);
    }

    bool direct_peer(int a, int b) override {
        return a == 0 && b == 1 ? trace->peer01 : trace->peer10;
    }

    void* allocate(int d, Bytes bytes, std::size_t) override {
        ++trace->allocations_started;
        check(resources->snapshot().used[d + 1] >= bytes);
        if (d == trace->fail_allocate)
            throw std::runtime_error("allocate");
        void* p = ::operator new(bytes);
        trace->allocations.emplace(p, std::pair{d, bytes});
        trace->free[d] -= bytes;
        return p;
    }

    void free(int d, void* p) override {
        if (d == trace->fail_free)
            throw std::runtime_error("free");
        check(trace->allocations.at(p).first == d);
        trace->free[d] += trace->allocations.at(p).second;
        trace->allocations.erase(p);
        ::operator delete(p);
        ++trace->releases;
    }

    void wait(int d, MediaEvent event) override {
        check(events.at(event.id) == event.device);
        trace->waits.emplace_back(d, event);
    }

    MediaEvent record(int d) override {
        if (trace->fail_record)
            throw std::runtime_error("record");
        auto id = next++;
        events.emplace(id, d);
        return {d, id};
    }

    void synchronize(int d) override {
        trace->drains.push_back(d);
        if (d == trace->fail_drain)
            throw std::runtime_error("drain");
    }

    void release_events(int d) override {
        for (auto it = events.begin(); it != events.end();) {
            if (it->second == d)
                it = events.erase(it);
            else
                ++it;
        }
    }

    std::uint32_t read_status(int, const void*) override {
        return trace->status;
    }
};

struct Fixture {
    std::shared_ptr<Resources> resources =
        std::make_shared<Resources>(Footprint{1000000, 1000000, 1000000});
    std::shared_ptr<Trace> trace = std::make_shared<Trace>();
    std::shared_ptr<MediaSessionActivity> activity =
        std::make_shared<MediaSessionActivity>();
    std::atomic_bool cancel = false;
    MediaSession session{
        resources,
        activity,
        std::make_unique<Fake>(trace, resources),
        Workload::image,
        {{0, 0, 0}, {1, 0, 0}},
        cancel
    };
};

StageManifest decoder() {
    StageManifest m;
    m.stage = MediaStage::image_decode;
    m.workspaces_complete = true;
    m.status_bytes = 4;
    m.tensors = {
        {"boundary", 64},
        {"kv-only", 64, MediaPhase::kv, MediaPhase::kv},
        {"ff-only", 64, MediaPhase::ff, MediaPhase::ff}
    };
    return m;
}

MediaShape shape() {
    return {4, 4, 4, 2, 2, 8, 0, 2};
}

DeviceTensorView tensor(
    MediaSession& s,
    int device,
    std::string_view name,
    std::size_t rows = 4,
    std::size_t columns = 4
) {
    return s.tensor(device, name, DeviceDType::f32, {rows, columns}, {columns, 1});
}

DeviceTensorView flag(MediaSession& s, int device) {
    return s.tensor(device, "resident/status", DeviceDType::u32, {1}, {1});
}

void produce(
    MediaSession& s, int device, DeviceTensorView& output, DeviceTensorView& status
) {
    std::array<DeviceTensorView*, 2> outputs{&output, &status};
    s.launch(device, {}, outputs, [](void*, auto, auto) {
    });  // Lifecycle fake, no numerical computation.
    s.complete(outputs, status);
}

void initial_admission() {
    Fixture f;
    auto m = decoder();
    check(!f.session.admit(m, shape()));
    const auto peak = f.session.plan().devices.front();
    check(f.resources->snapshot().used[peak.device + 1] == peak.new_ledger);
    check(f.session.evidence().peers.size() == 2 && f.activity->owners == 1);
    check(
        fails([&] { f.session.admit(m, shape()); }) == "media_session_stage_boundary"
    );
    WorkerCompute worker({{1000}, {}}, Workload::image);
    worker.media_activity = f.activity;
    check(fails([&] { worker.idle(); }) == "compute_media_session_busy");
    MediaSession moved = std::move(f.session);
    moved.finish();
    worker.idle();
    check(f.resources->snapshot().residents == 0 && f.activity->owners == 0);
}

void lifecycle_failures() {
    for (int failure : {0, 1, 2, 3}) {
        Fixture f;
        if (failure == 0) {
            f.trace->fail_allocate = 0;
            check(fails([&] { f.session.admit(decoder(), shape()); }) == "allocate");
            check(f.resources->snapshot().residents == 0);
            continue;
        }
        check(!f.session.admit(decoder(), shape()));
        f.trace->drains.clear();
        if (failure == 1)
            f.cancel = true;
        if (failure == 2)
            f.trace->fail_drain = 0;
        if (failure == 3)
            f.trace->fail_free = f.session.plan().devices.front().device;
        check(
            fails([&] { f.session.finish(); })
            == (failure == 1   ? "media_session_cancelled"
                : failure == 2 ? "drain"
                               : "free")
        );
        check(f.trace->drains == std::vector<int>({0, 1}));
        if (failure == 1)
            check(f.resources->snapshot().residents == 0);
        else {
            check(f.resources->snapshot().residents == 1 && f.activity->owners == 1);
            check(
                fails([&] { f.session.finish(); })
                == "media_session_cleanup_unconfirmed"
            );
            for (const auto& [pointer, allocation] : f.trace->allocations) {
                (void)allocation;
                ::operator delete(pointer);
            }
            f.trace->allocations.clear();
        }
    }
}

void ownership_views() {
    Fixture f;
    check(!f.session.admit(decoder(), shape()));
    const int d = f.session.plan().devices.front().device;
    auto view = tensor(f.session, d, "boundary");
    auto tile = f.session.rows(view, 1, 2);
    check(tile.layout().offset == 16 && tile.layout().bytes == 32);
    check(
        fails([&] { f.session.admit(decoder(), shape()); }) == "media_session_live_view"
    );
    check(fails([&] { f.session.finish(); }) == "media_session_live_view");
    view = {};
    check(f.resources->snapshot().residents == 1);
    tile = {};
    check(f.resources->snapshot().residents == 0 && f.activity->owners == 0);
}

void stage_ownership() {
    Fixture f;
    auto m = decoder();
    m.weights = {
        {"w",
         {1, 2, 3, 0, 0, 0, 0, 0, 32},
         TensorStorage::bf16_exact,
         4,
         4,
         WeightPartition::replicated}
    };
    check(!f.session.admit(m, shape()));
    auto view = f.session.weight(f.session.plan().weights.front());
    const auto pointer = view.layout().data;
    const auto generation = view.layout().generation;
    view = {};
    f.session.advance(MediaPhase::complete);
    auto before = f.trace->allocations_started;
    check(!f.session.admit(m, shape()));
    view = f.session.weight(f.session.plan().weights.front());
    check(
        view.layout().data == pointer && view.layout().generation == generation
        && f.trace->allocations_started == before
    );
    view = {};
    f.session.advance(MediaPhase::complete);
    m.weights[0].identity.inode++;
    check(!f.session.admit(m, shape()));
    check(f.trace->allocations_started == before + 1);
    f.session.advance(MediaPhase::kv);
    auto temporary =
        tensor(f.session, f.session.plan().devices.front().device, "kv-only");
    check(
        fails([&] { f.session.advance(MediaPhase::ff); }) == "media_session_live_alias"
    );
    temporary = {};
    f.session.advance(MediaPhase::ff);
    f.session.finish();
}

void event_regions() {
    Fixture f;
    check(!f.session.admit(decoder(), shape()));
    int d = f.session.plan().devices.front().device;
    f.session.advance(MediaPhase::ff);
    auto view = tensor(f.session, d, "ff-only"), status = flag(f.session, d);
    auto first = f.session.rows(view, 0, 2), second = f.session.rows(view, 2, 2);
    produce(f.session, d, first, status);
    std::array<DeviceTensorView*, 1> read{&view}, write{&status};
    check(
        fails([&] { f.session.launch(d, read, write, [](void*, auto, auto) {}); })
        == "media_session_unready_input"
    );
    produce(f.session, d, second, status);
    f.session.launch(d, read, write, [](void*, auto, auto) {});
    f.session.complete(write, status);
    first = {};
    second = {};
    view = {};
    status = {};
    f.session.finish();
}

void admission_and_cleanup() {
    Fixture f;
    auto m = decoder();
    check(!f.session.admit(m, shape()));
    check(f.activity->owners == 1 && f.trace->allocations_started > 0);
    const auto peak = f.session.plan().devices.front();
    check(f.resources->snapshot().used[peak.device + 1] == peak.new_ledger);
    check(f.session.evidence().peers.size() == 2);
    auto v = tensor(f.session, peak.device, "boundary");
    check(
        v.layout().dtype == DeviceDType::f32 && v.layout().generation
        && v.layout().strides[0] == 4
    );
    const auto original = v.layout().data;
    check(fails([&] { f.session.admit(m, shape()); }) == "media_session_live_view");
    check(v.layout().data == original);
    check(fails([&] { f.session.finish(); }) == "media_session_live_view");
    check(f.activity->owners == 1 && !f.trace->allocations.empty());
    v = {};
    check(f.resources->snapshot().residents == 0 && f.activity->owners == 0);
}

void weight_prefix_identity() {
    for (bool reversed : {false, true}) {
        Fixture f;
        auto m = decoder();
        m.weights = {
            {"w",
             {1, 2, 16, 0, 0, 0, 0, 0, 16},
             TensorStorage::bf16_exact,
             2,
             4,
             WeightPartition::replicated},
            {"w/sub",
             {1, 3, 48, 0, 0, 0, 0, 0, 48},
             TensorStorage::f32,
             2,
             6,
             WeightPartition::replicated}
        };
        if (reversed)
            std::swap(m.weights[0], m.weights[1]);
        check(!f.session.admit(m, shape()));
        for (const auto& slice : f.session.plan().weights) {
            auto view = f.session.weight(slice);
            const bool nested = slice.name == "w/sub";
            check(
                view.layout().dtype == (nested ? DeviceDType::f32 : DeviceDType::bf16)
            );
            check(
                view.layout().dimensions
                == std::vector<std::size_t>{2, nested ? 6u : 4u}
            );
        }
        f.session.finish();
    }
}

void reacquired_output_certificate() {
    Fixture f;
    check(!f.session.admit(decoder(), shape()));
    const int d = f.session.plan().devices.front().device;
    f.session.advance(MediaPhase::ff);
    {
        auto full = tensor(f.session, d, "ff-only");
        auto half = f.session.rows(full, 0, 2);
        auto status =
            f.session.tensor(d, "resident/status", DeviceDType::u32, {1}, {1});
        std::array<DeviceTensorView*, 2> outputs{&half, &status};
        f.session.launch(d, {}, outputs, [](void*, auto, auto) {});
        f.session.complete(outputs, status);
    }
    auto full = tensor(f.session, d, "ff-only");
    auto status = f.session.tensor(d, "resident/status", DeviceDType::u32, {1}, {1});
    std::array<DeviceTensorView*, 2> outputs{&full, &status};
    check(
        fails([&] { f.session.complete(outputs, status); }) == "media_session_status"
    );
    std::array<DeviceTensorView*, 1> read{&full}, write{&status};
    check(
        fails([&] { f.session.launch(d, read, write, [](void*, auto, auto) {}); })
        == "media_session_unready_input"
    );
    f.session.launch(d, {}, outputs, [](void*, auto, auto) {});
    f.session.complete(outputs, status);
    f.session.launch(d, read, write, [](void*, auto, auto) {});
    f.session.complete(write, status);
}

void typed_admission_failures() {
    Fixture f;
    auto m = decoder();
    m.workspaces_complete = false;
    auto error = f.session.admit(m, shape());
    check(error && error->reason == PlanFailure::unsupported_layout);
    m.workspaces_complete = true;
    m.allocation_alignment = 512;
    error = f.session.admit(m, shape());
    check(error && error->reason == PlanFailure::unsupported_layout);
    m.allocation_alignment = 256;
    m.stage = MediaStage::denoise;
    f.trace->free[0] = f.trace->free[1] = 1;
    error = f.session.admit(m, shape());
    check(error && error->reason == PlanFailure::capacity);
    check(f.trace->allocations_started == 0 && f.resources->snapshot().residents == 0);
    m.stage = MediaStage::denoise;
    f.trace->free[0] = f.trace->free[1] = 1;
    f.trace->peer10 = false;
    error = f.session.admit(m, shape());
    check(error && error->reason == PlanFailure::topology);
    check(f.trace->allocations_started == 0);
}

void uncertain_cleanup_and_first_error() {
    Fixture f;
    check(!f.session.admit(decoder(), shape()));
    int d = f.session.plan().devices.front().device;
    auto v = tensor(f.session, d, "boundary");
    auto status = flag(f.session, d);
    f.trace->fail_drain = 0;
    f.trace->drains.clear();
    std::array<DeviceTensorView*, 2> outputs{&v, &status};
    try {
        f.session.launch(d, {}, outputs, [](void*, auto, auto) {
            throw std::runtime_error("launch");
        });
        check(false);
    } catch (const OperationCleanupError& e) {
        check(
            exception_message(e.operation) == "launch"
            && exception_message(e.cleanup) == "drain"
        );
    }
    check(f.trace->drains == std::vector<int>({0, 1}));
    v = {};
    status = {};
    check(
        f.resources->snapshot().residents == 1 && f.activity->owners == 1
        && !f.trace->allocations.empty()
    );
    // Fake process teardown: real uncertain device ownership remains charged.
    for (const auto& [p, allocation] : f.trace->allocations) {
        (void)allocation;
        ::operator delete(p);
    }
    f.trace->allocations.clear();
}

void status_and_move_lifecycle() {
    Fixture f;
    check(!f.session.admit(decoder(), shape()));
    int d = f.session.plan().devices.front().device;
    auto v = tensor(f.session, d, "boundary");
    auto status = flag(f.session, d);
    std::array<DeviceTensorView*, 2> outputs{&v, &status};
    f.session.launch(d, {}, outputs, [](void*, auto, auto) {});
    check(
        fails([&] { f.session.advance(MediaPhase::kv); })
        == "media_session_unchecked_status"
    );
    f.trace->status = 1;
    check(
        fails([&] { f.session.complete(outputs, status); })
        == "media_session_nonfinite_output"
    );
    MediaSession moved = std::move(f.session);
    v = {};
    status = {};
    check(f.resources->snapshot().residents == 0);
}

void repeated_admission_invalidates_kv() {
    Fixture f;
    StageManifest m;
    m.workspaces_complete = true;
    m.status_bytes = 4;
    check(!f.session.admit(m, shape()));
    f.session.advance(MediaPhase::kv);
    const auto d = f.session.plan().devices.front().device;
    {
        auto key = tensor(f.session, d, "resident/key"),
             value = tensor(f.session, d, "resident/value");
        auto status = flag(f.session, d);
        produce(f.session, d, key, status);
        produce(f.session, d, value, status);
    }
    f.session.advance(MediaPhase::complete);
    check(!f.session.admit(m, shape()));
    f.session.advance(MediaPhase::kv);
    check(
        fails([&] { f.session.advance(MediaPhase::attention); })
        == "media_session_incomplete_kv"
    );
    check(f.resources->snapshot().residents == 0);
}

void kv_barrier_and_worker_idle() {
    Fixture f;
    StageManifest m;
    m.status_bytes = 4;
    m.workspaces_complete = true;
    check(!f.session.admit(m, shape()));
    int d = f.session.plan().devices.front().device;
    f.session.advance(MediaPhase::kv);
    auto key = tensor(f.session, d, "resident/key");
    auto value = tensor(f.session, d, "resident/value");
    auto status = flag(f.session, d);
    auto half = f.session.rows(key, 0, 2);
    produce(f.session, d, half, status);
    check(
        fails([&] { f.session.advance(MediaPhase::attention); }).find("incomplete_kv")
        != std::string::npos
    );
    // A rejected barrier is terminal; its allocations remain pinned by views.
    half = {};
    key = {};
    value = {};
    status = {};
    check(f.resources->snapshot().residents == 0);
    WorkerCompute worker({{1000}, {}}, Workload::image);
    ++worker.media_activity->owners;
    check(fails([&] { worker.idle(); }) == "compute_media_session_busy");
    --worker.media_activity->owners;
    worker.idle();
}
}  // namespace

int main() {
    initial_admission();
    lifecycle_failures();
    ownership_views();
    stage_ownership();
    weight_prefix_identity();
    reacquired_output_certificate();
    admission_and_cleanup();
    typed_admission_failures();
    uncertain_cleanup_and_first_error();
    status_and_move_lifecycle();
    repeated_admission_invalidates_kv();
    kv_barrier_and_worker_idle();
    event_regions();
    std::cout
        << "Media session CPU lifecycle units passed; no CUDA or forward execution\n";
}
