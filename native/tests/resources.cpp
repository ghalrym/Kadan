#include "kadan/resources.hpp"
#include <atomic>
#include <iostream>
#include <thread>

void check(bool ok) { if (!ok) throw std::runtime_error("check_failed"); }
template<class F> void fails(F action) {
    bool failed = false;
    try { action(); } catch (const std::exception&) { failed = true; }
    check(failed);
}
int main() {
    using namespace kadan;
    try {
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
