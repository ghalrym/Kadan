#include "kadan/resources.hpp"

#include <atomic>
#include <barrier>
#include <iostream>
#include <string_view>
#include <thread>

void check(bool ok) {
    if (!ok)
        throw std::runtime_error("check_failed");
}

template <class F> void fails(F action, std::string_view expected = {}) {
    bool failed = false;
    try {
        action();
    } catch (const std::exception& error) {
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
        try {
            r.pin(h);
            pinned = true;
        } catch (const std::exception& error) {
            pin_error = error.what();
        }
    });
    std::thread evict([&] {
        start.arrive_and_wait();
        try {
            r.begin_eviction(h);
            evicting = true;
        } catch (const std::exception& error) {
            eviction_error = error.what();
        }
    });
    start.arrive_and_wait();
    pin.join();
    evict.join();
    check(pinned != evicting);
    check(r.snapshot().used == Footprint({1}));
    if (pinned) {
        check(eviction_error == "busy" && pin_error.empty());
        fails([&] { r.released(h); }, "busy");
        r.unpin(h);
        r.begin_eviction(h);
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
    r.loaded(old);
    r.begin_eviction(old);
    fails([&] { r.reserve(Workload::image, {10, 10}); }, "exhausted");
    std::barrier start(3);
    Handle replacement = 0;
    std::string admission_error, cleanup_error;
    std::thread cleanup([&] {
        start.arrive_and_wait();
        try {
            r.released(old);
        } catch (const std::exception& error) {
            cleanup_error = error.what();
        }
    });
    std::thread admission([&] {
        start.arrive_and_wait();
        try {
            replacement = r.reserve(Workload::image, {10, 10});
        } catch (const std::exception& error) {
            admission_error = error.what();
        }
    });
    start.arrive_and_wait();
    cleanup.join();
    admission.join();
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
    r.loaded(replacement);
    r.pin(replacement);
    r.unpin(replacement);
    r.begin_eviction(replacement);
    r.released(replacement);
    check(r.snapshot().used == Footprint({0, 0}));
}

void control_plane_limit() {
    using namespace kadan;
    Resources r({0});
    std::vector<Handle> handles;
    handles.reserve(Resources::max_residents);
    for (std::size_t i = 0; i < Resources::max_residents; ++i)
        handles.push_back(r.reserve(Workload::llm, {0}));
    fails([&] { r.reserve(Workload::llm, {0}); }, "resident_limit");
    check(r.snapshot().residents == Resources::max_residents);
    check(r.snapshot().used == Footprint({0}));
    // Loading, resident, and evicting records all count until cleanup is acknowledged.
    r.loaded(handles[0]);
    r.begin_eviction(handles[0]);
    fails([&] { r.reserve(Workload::llm, {0}); }, "resident_limit");
    r.released(handles[0]);
    const auto replacement = r.reserve(Workload::llm, {0});
    check(replacement > handles.back());
    fails([&] { r.released(handles[0]); }, "stale_handle");
    fails([&] { r.reserve(Workload::llm, {0}); }, "resident_limit");
    r.released(replacement);
    for (std::size_t i = 1; i < handles.size(); ++i)
        r.released(handles[i]);
    check(r.snapshot().residents == 0);
    fails(
        [] { Resources too_wide(Footprint(Resources::max_devices + 2, 0)); },
        "too_many_devices"
    );
}

void capacity_recovers_within_last_clean_measurement() {
    using namespace kadan;
    Resources resources({100, 80});
    const auto resident = resources.reserve(Workload::llm, {40, 20});
    resources.refresh_available({5, 10});
    check(resources.snapshot().capacity == Footprint({100, 80}));
    resources.released(resident);
    resources.refresh_available({5, 10});
    check(resources.snapshot().capacity == Footprint({5, 10}));
    resources.refresh_available({100, 100});
    check(resources.snapshot().capacity == Footprint({100, 80}));
    resources.reset_capacity({70, 50});
    resources.refresh_available({10, 5});
    resources.refresh_available({100, 100});
    check(resources.snapshot().capacity == Footprint({70, 50}));
}

void scopes_share_admission_and_isolate_handles() {
    using namespace kadan;
    Resources root({100, 60, 40});
    auto chat = root.scope({60, 60, 10});
    auto image = root.scope({40, 0, 30});
    const auto chat_handle = chat->reserve(Workload::llm, {30, 20, 0});
    const auto image_handle = image->reserve(Workload::image, {40, 0, 30});
    check(root.snapshot().used == Footprint({100, 60, 40}));
    check(root.snapshot().residents == 2);
    check(chat->snapshot().used == Footprint({30, 20, 0}));
    check(chat->snapshot().capacity == Footprint({60, 60, 10}));
    check(chat->snapshot().residents == 1);
    fails([&] { chat->reserve(Workload::llm, {31, 0, 0}); }, "exhausted");
    fails([&] { image->reservation(chat_handle); }, "stale_handle");
    fails([&] { image->resize_loading(chat_handle, {0, 0, 0}); }, "stale_handle");
    fails([&] { image->loaded(chat_handle); }, "stale_handle");
    chat->loaded(chat_handle);
    chat->pin(chat_handle);
    fails([&] { image->pin(chat_handle); }, "stale_handle");
    fails([&] { image->unpin(chat_handle); }, "stale_handle");
    fails([&] { image->begin_eviction(chat_handle); }, "stale_handle");
    fails([&] { root.begin_eviction(chat_handle); }, "stale_handle");
    fails([&] { chat->begin_eviction(chat_handle); }, "busy");
    chat->unpin(chat_handle);
    chat->begin_eviction(chat_handle);
    fails([&] { image->eviction_failed(chat_handle); }, "stale_handle");
    fails([&] { image->released(chat_handle); }, "stale_handle");
    chat->released(chat_handle);
    check(chat->snapshot().residents == 0);
    check(chat->snapshot().capacity == Footprint({60, 60, 10}));
    chat->idle();
    check(root.snapshot().used == Footprint({40, 0, 30}));
    image->released(image_handle);
    image->idle();
    check(root.snapshot().residents == 0);
}

void scope_resize_is_atomic() {
    using namespace kadan;
    Resources root({100, 80});
    auto first = root.scope({30, 20});
    auto second = root.scope({70, 60});
    const auto a = first->reserve(Workload::llm, {20, 10});
    const auto b = second->reserve(Workload::image, {70, 60});
    fails([&] { first->resize_loading(a, {31, 20}); }, "exhausted");
    check(first->reservation(a).bytes == Footprint({20, 10}));
    check(root.snapshot().used == Footprint({100, 80}));
    first->resize_loading(a, {30, 20});
    check(root.snapshot().used == Footprint({100, 80}));
    second->resize_loading(b, {50, 40});
    second->idle();
    first->admit({50, 40});
    check(first->snapshot().capacity == Footprint({50, 40}));
    first->resize_loading(a, {50, 40});
    check(root.snapshot().used == Footprint({100, 80}));
    first->released(a);
    second->released(b);
    first->idle();
    second->idle();
    check(root.empty());
}

void scope_device_limits_are_enforced() {
    using namespace kadan;
    Resources root({100, 80, 60});
    auto first = root.scope({40, 80, 0});
    fails([&] { root.scope({61, 0, 60}); }, "exhausted");
    auto second = root.scope({60, 0, 60});
    fails([&] { root.scope({100}); }, "budget_shape");
    check(first->snapshot().capacity == Footprint({40, 80, 0}));
    check(root.snapshot().used == Footprint({100, 80, 60}));
    check(!root.empty() && root.snapshot().residents == 0);
    fails(
        [&] { root.reset_capacity({100, 80, 60}); }, "capacity_requires_empty_ledger"
    );
    fails([&] { first->reserve(Workload::llm, {41, 0, 0}); }, "exhausted");
    fails([&] { first->reserve(Workload::llm, {0, 0, 1}); }, "exhausted");
    const auto a = first->reserve(Workload::llm, {30, 80, 0});
    fails([&] { first->reserve(Workload::llm, {11, 0, 0}); }, "exhausted");
    fails([&] { first->resize_loading(a, {30, 80, 1}); }, "exhausted");
    fails([&] { first->resize_loading(a, {41, 80, 0}); }, "exhausted");
    const auto b = second->reserve(Workload::image, {60, 0, 60});
    check(root.snapshot().used == Footprint({100, 80, 60}));
    check(first->snapshot().used == Footprint({30, 80, 0}));
    check(first->snapshot().capacity == Footprint({40, 80, 0}));
    first->released(a);
    second->released(b);
    check(!root.empty());
    first->idle();
    second->idle();
    check(root.empty());
}

void scopes_retain_unacknowledged_cleanup() {
    using namespace kadan;
    Resources root({100});
    auto first = root.scope({100});
    auto second = root.scope({0});
    const auto h = first->reserve(Workload::llm, {100});
    first->loaded(h);
    first->begin_eviction(h);
    fails([&] { second->admit({1}); }, "exhausted");
    first->eviction_failed(h);
    check(root.snapshot().used == Footprint({100}));
    first.reset();
    check(root.snapshot().used == Footprint({100}));
    check(root.snapshot().residents == 1);
    fails([&] { second->admit({1}); }, "exhausted");
    fails([&] { root.reset_capacity({100}); }, "capacity_requires_empty_ledger");
    fails([&] { second->reset_capacity({100}); }, "root_required");
}

void physical_refresh_preserves_the_admission_epoch() {
    using namespace kadan;
    Resources root({100, 80});
    auto scope = root.scope({80, 60});
    const auto weights = scope->reserve(Workload::llm, {40, 30});
    root.refresh_available({5, 5});
    check(root.snapshot().capacity == Footprint({100, 80}));
    check(scope->snapshot().capacity == Footprint({80, 60}));
    // Execution can materialize the remainder of its already-admitted envelope
    // even though a fresh physical probe is smaller than resident allocations.
    const auto workspace = scope->reserve(Workload::llm, {40, 30});
    check(root.snapshot().used == Footprint({80, 60}));
    fails([&] { root.scope({21, 0}); }, "exhausted");
    fails([&] { root.refresh_available({1}); }, "budget_shape");
    scope->released(workspace);
    scope->released(weights);
    scope->idle();
    root.refresh_available({50, 20});
    check(root.snapshot().capacity == Footprint({50, 20}));
    root.refresh_available({100, 100});
    check(root.snapshot().capacity == Footprint({100, 80}));
}

void idle_envelopes_must_be_reacquired_atomically() {
    using namespace kadan;
    Resources root({100});
    auto first = root.scope({80});
    const auto weights = first->reserve(Workload::llm, {30});
    first->idle();
    check(root.snapshot().used == Footprint({30}));
    check(first->snapshot().capacity == Footprint({30}));
    fails([&] { first->reserve(Workload::llm, {1}); }, "exhausted");
    auto second = root.scope({60});
    fails([&] { first->admit({80}); }, "exhausted");
    check(first->snapshot().capacity == Footprint({30}));
    check(root.snapshot().used == Footprint({90}));
    second->idle();
    first->admit({80});
    check(root.snapshot().used == Footprint({80}));
    fails([&] { first->admit({29}); }, "allowance_below_usage");
    check(root.snapshot().used == Footprint({80}));
    first->resize_loading(weights, {80});
    check(root.snapshot().used == Footprint({80}));
    first->resize_loading(weights, {10});
    check(root.snapshot().used == Footprint({80}));
    first->idle();
    check(root.snapshot().used == Footprint({10}));
    first->released(weights);
    first->idle();
    check(root.empty());
}

void abandoning_a_scope_returns_only_unused_escrow() {
    using namespace kadan;
    Resources root({100});
    {
        auto scope = root.scope({80});
        check(root.snapshot().used == Footprint({80}));
    }
    check(root.empty());
    {
        auto scope = root.scope({80});
        const auto weights = scope->reserve(Workload::llm, {30});
        scope->loaded(weights);
    }
    check(root.snapshot().used == Footprint({30}));
    check(root.snapshot().residents == 1);
    check(!root.empty());
    auto other = root.scope({70});
    fails([&] { root.reserve(Workload::image, {1}); }, "exhausted");
}

void concurrent_envelopes_cannot_spend_the_same_capacity() {
    using namespace kadan;
    Resources root({100, 100});
    std::barrier start(3);
    std::shared_ptr<Resources> first, second;
    std::string first_error, second_error;
    auto admit = [&](auto& result, auto& error) {
        start.arrive_and_wait();
        try {
            result = root.scope({60, 60});
        } catch (const std::exception& exception) {
            error = exception.what();
        }
    };
    std::thread one([&] { admit(first, first_error); });
    std::thread two([&] { admit(second, second_error); });
    start.arrive_and_wait();
    one.join();
    two.join();
    check(bool(first) != bool(second));
    check((first ? second_error : first_error) == "exhausted");
    check(root.snapshot().used == Footprint({60, 60}));
    check(root.snapshot().residents == 0);
    first.reset();
    second.reset();
    check(root.empty());
}

void scopes_share_control_plane_limit() {
    using namespace kadan;
    Resources root({0});
    auto first = root.scope({0});
    auto second = root.scope({0});
    std::vector<Handle> handles;
    for (std::size_t i = 0; i < Resources::max_residents; ++i)
        handles.push_back(first->reserve(Workload::llm, {0}));
    check(second->snapshot().residents == 0);
    fails([&] { second->reserve(Workload::image, {0}); }, "resident_limit");
    first->released(handles.back());
    handles.pop_back();
    const auto replacement = second->reserve(Workload::image, {0});
    check(root.snapshot().residents == Resources::max_residents);
    for (const auto handle : handles)
        first->released(handle);
    second->released(replacement);
    check(root.snapshot().residents == 0);
}

void scopes_keep_the_shared_ledger_alive() {
    using namespace kadan;
    auto root = std::make_unique<Resources>(Footprint{10});
    auto first = root->scope({10});
    auto second = root->scope({0});
    root.reset();
    const auto h = first->reserve(Workload::llm, {10});
    fails([&] { second->admit({1}); }, "exhausted");
    first->released(h);
    first->idle();
    second->admit({10});
    check(second->snapshot().capacity == Footprint({10}));
}

void scoped_admission_race() {
    using namespace kadan;
    Resources root({10, 10});
    auto first = root.scope({0, 0});
    auto second = root.scope({0, 0});
    std::barrier start(3);
    Handle a = 0, b = 0;
    std::string first_error, second_error;
    auto reserve = [&](Resources& scope, Handle& result, std::string& error) {
        start.arrive_and_wait();
        try {
            scope.admit({6, 6});
            result = scope.reserve(Workload::llm, {6, 6});
        } catch (const std::exception& exception) {
            error = exception.what();
        }
    };
    std::thread one([&] { reserve(*first, a, first_error); });
    std::thread two([&] { reserve(*second, b, second_error); });
    start.arrive_and_wait();
    one.join();
    two.join();
    check((a != 0) != (b != 0));
    check(root.snapshot().used == Footprint({6, 6}));
    check(root.snapshot().residents == 1);
    if (a) {
        check(first_error.empty() && second_error == "exhausted");
        first->released(a);
    } else {
        check(second_error.empty() && first_error == "exhausted");
        second->released(b);
    }
    first->idle();
    second->idle();
    check(root.empty());
}

int main() {
    using namespace kadan;
    try {
        scopes_share_control_plane_limit();
        scopes_keep_the_shared_ledger_alive();
        scopes_share_admission_and_isolate_handles();
        scope_resize_is_atomic();
        scope_device_limits_are_enforced();
        scopes_retain_unacknowledged_cleanup();
        physical_refresh_preserves_the_admission_epoch();
        idle_envelopes_must_be_reacquired_atomically();
        abandoning_a_scope_returns_only_unused_escrow();
        capacity_recovers_within_last_clean_measurement();
        control_plane_limit();
        for (int i = 0; i < 32; ++i) {
            concurrent_envelopes_cannot_spend_the_same_capacity();
            scoped_admission_race();
            pin_eviction_race();
            cleanup_admission_race();
        }
        Resources r({100, 24, 24});
        fails([&] { r.reserve(Workload::llm, {0, 25, 0}); });  // Cannot pool VRAM.
        fails([&] { r.reserve(Workload::llm, {1}); });
        auto h = r.reserve(Workload::llm, {70, 20, 10});
        fails([&] { r.pin(h); });  // Loading is not executable.
        fails([&] { r.reserve(Workload::video, {31, 0, 0}); });
        check(r.snapshot().used == Footprint({70, 20, 10}));
        r.loaded(h);
        r.pin(h);
        r.pin(h);
        fails([&] { r.begin_eviction(h); });
        fails([&] { r.released(h); });
        r.unpin(h);
        fails([&] { r.begin_eviction(h); });
        r.unpin(h);
        fails([&] { r.unpin(h); });
        r.begin_eviction(h);
        fails([&] { r.pin(h); });
        fails([&] {
            r.reserve(Workload::image, {40, 0, 0});
        });  // Cleanup not acknowledged.
        r.eviction_failed(h);
        r.pin(h);
        r.unpin(h);
        r.begin_eviction(h);
        r.released(h);
        auto next = r.reserve(Workload::decision, {100, 24, 24});
        check(next != h);
        fails([&] { r.released(h); });
        check(r.snapshot().used == Footprint({100, 24, 24}));
        r.released(next);  // Aborted load after physical cleanup.
        check(r.snapshot().used == Footprint({0, 0, 0}));
        Resources large({std::numeric_limits<Bytes>::max()});
        auto full = large.reserve(Workload::llm, {std::numeric_limits<Bytes>::max()});
        fails([&] { large.reserve(Workload::llm, {1}); });
        large.released(full);
        Resources concurrent({1});
        std::atomic<int> admitted{0};
        auto attempt = [&] {
            try {
                concurrent.reserve(Workload::llm, {1});
                ++admitted;
            } catch (const std::runtime_error&) {
            }
        };
        std::thread a(attempt), b(attempt);
        a.join();
        b.join();
        check(admitted == 1 && concurrent.snapshot().used == Footprint({1}));
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
