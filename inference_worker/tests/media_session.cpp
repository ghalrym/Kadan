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
    std::size_t allocations_started = 0, releases = 0;
    bool peer01 = true, peer10 = true;
    int fail_allocate = -1, fail_drain = -1, fail_free = -1;
};

struct Fake final : MediaDeviceOperations {
    std::shared_ptr<Trace> trace;
    std::shared_ptr<Resources> resources;

    explicit Fake(std::shared_ptr<Trace> t, std::shared_ptr<Resources> r)
        : trace(std::move(t)), resources(std::move(r)) {
    }

    std::size_t allocation_alignment() const override {
        return 256;
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

    void synchronize(int d) override {
        trace->drains.push_back(d);
        if (d == trace->fail_drain)
            throw std::runtime_error("drain");
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
}  // namespace

int main() {
    initial_admission();
    lifecycle_failures();
    ownership_views();
    stage_ownership();
    weight_prefix_identity();
    typed_admission_failures();
    std::cout
        << "Media session CPU lifecycle units passed; no CUDA or forward execution\n";
}
