#include "kadan/packed_read_team.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <set>
#include <thread>
#include <vector>

namespace {
using namespace kadan;
using namespace std::chrono_literals;

void check(bool condition, const char* message) {
    if (!condition)
        throw std::runtime_error(message);
}

template <class Function> void fails(Function function) {
    try {
        function();
    } catch (const std::exception&) {
        return;
    }
    throw std::runtime_error("packed_read_expected_failure");
}

struct Region {
    std::size_t first, offset, bytes;
};

void rows_are_disjoint_parallel_and_reused() {
    const auto admission = PackedReadTeam::host_bytes();
    Resources resources({admission});
    PackedReadTeam team(resources, Workload::llm);
    constexpr std::size_t row_bytes = 8192, rows = 257, first = 37;
    std::vector<std::byte> output(rows * row_bytes);
    std::set<std::thread::id> previous_threads;
    for (unsigned repeat = 0; repeat < 3; ++repeat) {
        output.resize((repeat ? rows : rows - 1) * row_bytes);
        std::fill(output.begin(), output.end(), std::byte{0});
        std::mutex mutex;
        std::condition_variable condition;
        std::vector<Region> regions;
        std::set<std::thread::id> threads;
        unsigned entered = 0;
        bool timed_out = false;
        auto owned = std::make_shared<int>(7);
        std::weak_ptr<int> lifetime = owned;
        PackedReadTeam::Source reader = [&, owned = std::move(owned)](
                                            std::size_t begin,
                                            std::span<std::byte> destination
                                        ) {
            check(*owned == 7, "packed_read_parallel_callback_alive");
            check(
                resources.snapshot().used[0] == admission, "packed_read_stack_charge"
            );
            const auto offset = std::size_t(destination.data() - output.data());
            {
                std::unique_lock lock(mutex);
                check(
                    begin >= first && destination.size() % row_bytes == 0,
                    "packed_read_whole_rows"
                );
                check(
                    offset == (begin - first) * row_bytes && offset <= output.size()
                        && destination.size() <= output.size() - offset,
                    "packed_read_first_row_offset"
                );
                for (const auto& region : regions)
                    check(
                        offset + destination.size() <= region.offset
                            || region.offset + region.bytes <= offset,
                        "packed_read_nonoverlap"
                    );
                regions.push_back({begin, offset, destination.size()});
                threads.insert(std::this_thread::get_id());
                ++entered;
                condition.notify_all();
                // The timeout only prevents a broken implementation hanging the
                // unit. Success requires all four callbacks genuinely overlap.
                if (!condition.wait_for(lock, 5s, [&] {
                        return entered == 4 || timed_out;
                    })) {
                    timed_out = true;
                    condition.notify_all();
                }
                check(!timed_out, "packed_read_four_concurrent_callbacks");
            }
            for (std::size_t byte = 0; byte < destination.size(); ++byte)
                destination[byte] = std::byte((offset + byte + repeat + first) % 251);
        };
        team.read(reader, first, row_bytes, output);
        reader = {};
        check(
            lifetime.expired(), "packed_read_parallel_callback_released_after_return"
        );
        check(regions.size() == 4 && threads.size() == 4, "packed_read_four_regions");
        check(
            threads.contains(std::this_thread::get_id()),
            "packed_read_caller_participates"
        );
        if (repeat)
            check(threads == previous_threads, "packed_read_persistent_workers_reused");
        previous_threads = threads;
        std::sort(regions.begin(), regions.end(), [](const Region& a, const Region& b) {
            return a.offset < b.offset;
        });
        std::size_t covered = 0;
        for (const auto& region : regions) {
            check(
                region.offset == covered && region.bytes > 0,
                "packed_read_exact_once_coverage"
            );
            covered += region.bytes;
        }
        check(covered == output.size(), "packed_read_all_bytes_covered");
        for (std::size_t byte = 0; byte < output.size(); ++byte)
            check(
                output[byte] == std::byte((byte + repeat + first) % 251),
                "packed_read_output_matches_serial_oracle"
            );
    }
    team.close();
    check(
        resources.snapshot().used == Footprint({0}),
        "packed_read_explicit_close_releases_charge"
    );
    team.close();
    fails([&] {
        team.read([](std::size_t, std::span<std::byte>) {}, 0, row_bytes, output);
    });
}

void small_reads_and_callback_lifetime() {
    const auto admission = PackedReadTeam::host_bytes();
    Resources resources({admission});
    {
        PackedReadTeam team(resources, Workload::llm);
        std::array<std::byte, 32> small{};
        unsigned calls = 0;
        const auto small_caller = std::this_thread::get_id();
        std::weak_ptr<int> lifetime;
        {
            auto owned = std::make_shared<int>(9);
            lifetime = owned;
            PackedReadTeam::Source reader =
                [&, owned](std::size_t first, std::span<std::byte> destination) {
                    ++calls;
                    check(
                        first == 11 && destination.data() == small.data()
                            && destination.size() == small.size(),
                        "packed_read_small_whole_request"
                    );
                    check(
                        std::this_thread::get_id() == small_caller,
                        "packed_read_small_callback"
                    );
                    check(
                        *owned == 9 && resources.snapshot().used[0] == admission,
                        "packed_read_small_lifetime_and_charge"
                    );
                };
            team.read(reader, 11, 8, small);
        }
        check(calls == 1, "packed_read_below_threshold_serial");
        check(lifetime.expired(), "packed_read_no_callback_retained_after_return");
        std::vector<std::byte> below_threshold(
            PackedReadTeam::parallel_threshold - 8192
        );
        team.read(
            [&](std::size_t first, std::span<std::byte> destination) {
                ++calls;
                check(
                    first == 5 && destination.size() == below_threshold.size()
                        && std::this_thread::get_id() == small_caller,
                    "packed_read_just_below_threshold_serial"
                );
            },
            5,
            8192,
            below_threshold
        );
        std::vector<std::byte> few_rows(PackedReadTeam::parallel_threshold);
        const auto caller = std::this_thread::get_id();
        team.read(
            [&](std::size_t first, std::span<std::byte> destination) {
                ++calls;
                check(
                    first == 3 && destination.size() == few_rows.size()
                        && std::this_thread::get_id() == caller,
                    "packed_read_fewer_than_four_rows_serial"
                );
            },
            3,
            few_rows.size(),
            few_rows
        );
        check(calls == 3, "packed_read_serial_calls_exact");
        const auto no_read = [&](std::size_t, std::span<std::byte>) {
            ++calls;
        };
        fails([&] { team.read(no_read, 0, 0, small); });
        fails([&] { team.read(no_read, 0, 8, {}); });
        fails([&] { team.read(no_read, 0, 7, small); });
        fails([&] {
            team.read(no_read, std::numeric_limits<std::size_t>::max(), 8, small);
        });
        check(calls == 3, "packed_read_invalid_layout_before_callback");
    }
    check(
        resources.snapshot().used == Footprint({0}),
        "packed_read_destructor_releases_charge"
    );
    Resources too_small({admission - 1});
    fails([&] { PackedReadTeam rejected(too_small, Workload::llm); });
    check(
        too_small.snapshot().used == Footprint({0}),
        "packed_read_failed_admission_no_charge"
    );
}

void failed_read_drains_every_callback(const char* message) {
    const auto admission = PackedReadTeam::host_bytes();
    Resources resources({admission});
    PackedReadTeam team(resources, Workload::llm);
    constexpr std::size_t first = 17, row_bytes = 8192;
    auto buffer = std::make_unique<std::vector<std::byte>>(257 * row_bytes);
    std::mutex mutex;
    std::condition_variable condition;
    unsigned entered = 0, completed = 0;
    bool release = false, returned = false, caught = false;
    std::jthread caller([&] {
        try {
            team.read(
                [&](std::size_t begin, std::span<std::byte> destination) {
                    {
                        std::unique_lock lock(mutex);
                        ++entered;
                        condition.notify_all();
                        if (begin != first)
                            condition.wait(lock, [&] { return release; });
                    }
                    check(
                        resources.snapshot().used[0] == admission,
                        "packed_read_failure_charge"
                    );
                    if (begin != first)
                        std::fill(
                            destination.begin(), destination.end(), std::byte{0x5a}
                        );
                    {
                        std::lock_guard lock(mutex);
                        ++completed;
                        condition.notify_all();
                    }
                    if (begin == first)
                        throw std::runtime_error(message);
                },
                first,
                row_bytes,
                *buffer
            );
        } catch (const std::runtime_error& error) {
            caught = std::string(error.what()) == message;
        }
        std::lock_guard lock(mutex);
        returned = true;
        condition.notify_all();
    });
    bool all_entered = false, returned_early = false;
    {
        std::unique_lock lock(mutex);
        all_entered = condition.wait_for(lock, 5s, [&] {
            return entered == 4 || returned;
        }) && entered == 4;
        returned_early = returned;
        release = true;
        condition.notify_all();
    }
    caller.join();
    check(
        all_entered && !returned_early,
        "packed_read_error_waits_for_outstanding_regions"
    );
    check(
        caught && completed == 4 && returned,
        "packed_read_error_propagated_after_all_drained"
    );
    // Caller storage can be destroyed immediately after either return path.
    buffer.reset();
    std::array<std::byte, 8> next{};
    unsigned calls = 0;
    team.read([&](std::size_t, std::span<std::byte>) { ++calls; }, 0, 8, next);
    check(calls == 1, "packed_read_reusable_after_reader_failure");
    team.close();
    check(
        resources.snapshot().used == Footprint({0}),
        "packed_read_failed_request_cleanup"
    );
}
}  // namespace

// Whole items run once each across several lanes; the team remains reusable
// after an item fails and reads still work afterwards.
void items_run_once_across_lanes() {
    const auto admission = PackedReadTeam::host_bytes();
    Resources resources({admission});
    PackedReadTeam team(resources, Workload::llm);
    for (int round = 0; round < 20; ++round) {
        std::vector<std::atomic<int>> seen(57);
        std::mutex mutex;
        std::set<std::thread::id> lanes;
        team.each(seen.size(), [&](std::size_t i) {
            ++seen[i];
            std::this_thread::sleep_for(100us);
            std::lock_guard lock(mutex);
            lanes.insert(std::this_thread::get_id());
        });
        for (const auto& count : seen)
            check(count == 1, "packed_items_once");
        check(lanes.size() > 1, "packed_items_parallel");
    }
    fails([&] {
        team.each(8, [](std::size_t i) {
            if (i == 3)
                throw std::runtime_error("item_failed");
        });
    });
    std::vector<std::byte> destination(16);
    team.read(
        [](std::size_t, std::span<std::byte> to) {
            std::fill(to.begin(), to.end(), std::byte{7});
        },
        0,
        4,
        destination
    );
    check(destination[15] == std::byte{7}, "packed_items_then_read");
    team.close();
}

int main() {
    try {
        rows_are_disjoint_parallel_and_reused();
        items_run_once_across_lanes();
        small_reads_and_callback_lifetime();
        failed_read_drains_every_callback("reader_failed");
        failed_read_drains_every_callback("cancelled");
        std::cout << "packed read team tests passed\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
