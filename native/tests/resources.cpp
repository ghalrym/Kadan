#include "kadan/resources.hpp"
#include <atomic>
#include <barrier>
#include <string_view>
#include <iostream>
#include <thread>

void check(bool ok) { if (!ok) throw std::runtime_error("check_failed"); }
template<class F> void fails(F action, std::string_view expected = {}) {
    bool failed = false;
    try { action(); } catch (const std::exception& error) {
        check(expected.empty() || expected == error.what());
        failed = true;
    }
    check(failed);
}
// Coordinated starts exercise either legal mutex ordering without sleeps or
// requiring a particular thread to win. Pins remain held until both finish.
void pin_eviction_race() {
    using namespace kadan;
    Resources r({1});
    auto h = r.reserve(Workload::llm, {1});
    r.loaded(h);
    std::barrier start(3);
    bool pinned = false, evicting = false;
    std::string pin_error, eviction_error;
    std::thread pin([&] {
        start.arrive_and_wait();
        try { r.pin(h); pinned = true; }
        catch (const std::exception& error) { pin_error = error.what(); }
    });
    std::thread evict([&] {
        start.arrive_and_wait();
        try { r.begin_eviction(h); evicting = true; }
        catch (const std::exception& error) { eviction_error = error.what(); }
    });
    start.arrive_and_wait();
    pin.join(); evict.join();
    check(pinned != evicting);
    check(r.snapshot().used == Footprint({1}));
    if (pinned) {
        check(eviction_error == "busy" && pin_error.empty());
        fails([&] { r.released(h); }, "busy");
        r.unpin(h); r.begin_eviction(h);
    } else {
        check(pin_error == "not_resident" && eviction_error.empty());
    }
    r.released(h);
    check(r.snapshot().residents == 0);
}

void cleanup_admission_race() {
    using namespace kadan;
    Resources r({10, 10});
    const auto old = r.reserve(Workload::llm, {10, 10});
    r.loaded(old); r.begin_eviction(old);
    fails([&] { r.reserve(Workload::image, {10, 10}); }, "exhausted");
    std::barrier start(3);
    Handle replacement = 0;
    std::string admission_error, cleanup_error;
    std::thread cleanup([&] {
        start.arrive_and_wait();
        try { r.released(old); }
        catch (const std::exception& error) { cleanup_error = error.what(); }
    });
    std::thread admission([&] {
        start.arrive_and_wait();
        try { replacement = r.reserve(Workload::image, {10, 10}); }
        catch (const std::exception& error) { admission_error = error.what(); }
    });
    start.arrive_and_wait();
    cleanup.join(); admission.join();
    check(cleanup_error.empty());
    if (replacement == 0) {
        check(admission_error == "exhausted");
        check(r.snapshot().used == Footprint({0, 0}));
        replacement = r.reserve(Workload::image, {10, 10});
    } else {
        check(admission_error.empty());
    }
    check(replacement != old && r.snapshot().residents == 1);
    check(r.snapshot().used == Footprint({10, 10}));
    fails([&] { r.loaded(old); }, "stale_handle");
    fails([&] { r.pin(old); }, "stale_handle");
    fails([&] { r.begin_eviction(old); }, "stale_handle");
    fails([&] { r.released(old); }, "stale_handle");
    check(r.snapshot().used == Footprint({10, 10}));
    r.loaded(replacement); r.pin(replacement); r.unpin(replacement);
    r.begin_eviction(replacement); r.released(replacement);
    check(r.snapshot().used == Footprint({0, 0}));
}

void control_plane_limit() {
    using namespace kadan;
    Resources r({0});
    std::vector<Handle> handles;
    for (std::size_t i = 0; i < Resources::max_residents; ++i)
        handles.push_back(r.reserve(Workload::llm, {0}));
    fails([&] { r.reserve(Workload::llm, {0}); }, "resident_limit");
    check(r.snapshot().residents == Resources::max_residents);
    check(r.snapshot().used == Footprint({0}));
    // Loading, resident, and evicting records all count until cleanup is acknowledged.
    r.loaded(handles[0]); r.begin_eviction(handles[0]);
    fails([&] { r.reserve(Workload::llm, {0}); }, "resident_limit");
    r.released(handles[0]);
    const auto replacement = r.reserve(Workload::llm, {0});
    check(replacement > handles.back());
    fails([&] { r.released(handles[0]); }, "stale_handle");
    fails([&] { r.reserve(Workload::llm, {0}); }, "resident_limit");
    r.released(replacement);
    for (std::size_t i = 1; i < handles.size(); ++i) r.released(handles[i]);
    check(r.snapshot().residents == 0);
    fails([] { Resources too_wide(Footprint(Resources::max_devices + 2, 0)); }, "too_many_devices");
}

int main() {
    using namespace kadan;
    try {
        control_plane_limit();
        for (int i = 0; i < 32; ++i) {
            pin_eviction_race();
            cleanup_admission_race();
        }
        Resources r({100, 24, 24});
        fails([&] { r.reserve(Workload::llm, {0, 25, 0}); }); // Cannot pool VRAM.
        fails([&] { r.reserve(Workload::llm, {1}); });
        auto h = r.reserve(Workload::llm, {70, 20, 10});
        fails([&] { r.pin(h); }); // Loading is not executable.
        fails([&] { r.reserve(Workload::video, {31, 0, 0}); });
        check(r.snapshot().used == Footprint({70, 20, 10}));
        r.loaded(h); r.pin(h); r.pin(h);
        fails([&] { r.begin_eviction(h); });
        fails([&] { r.released(h); });
        r.unpin(h); fails([&] { r.begin_eviction(h); }); r.unpin(h);
        fails([&] { r.unpin(h); });
        r.begin_eviction(h);
        fails([&] { r.pin(h); });
        fails([&] { r.reserve(Workload::image, {40, 0, 0}); }); // Cleanup not acknowledged.
        r.eviction_failed(h); r.pin(h); r.unpin(h); r.begin_eviction(h); r.released(h);
        auto next = r.reserve(Workload::decision, {100, 24, 24});
        check(next != h);
        fails([&] { r.released(h); });
        check(r.snapshot().used == Footprint({100, 24, 24}));
        r.released(next); // Aborted load after physical cleanup.
        check(r.snapshot().used == Footprint({0, 0, 0}));
        Resources large({std::numeric_limits<Bytes>::max()});
        auto full = large.reserve(Workload::llm, {std::numeric_limits<Bytes>::max()});
        fails([&] { large.reserve(Workload::llm, {1}); });
        large.released(full);
        Resources concurrent({1});
        std::atomic<int> admitted{0};
        auto attempt = [&] {
            try { concurrent.reserve(Workload::llm, {1}); ++admitted; }
            catch (const std::runtime_error&) {}
        };
        std::thread a(attempt), b(attempt); a.join(); b.join();
        check(admitted == 1 && concurrent.snapshot().used == Footprint({1}));
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
