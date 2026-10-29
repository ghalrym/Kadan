#include "../src/models/video/h3_wall_timing.hpp"

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using namespace kadan;
using namespace kadan::video::detail;

namespace {
std::uint64_t ticks = 0;
std::vector<H3WallRecord> records;

std::uint64_t clock_value() {
    return ticks;
}

void collect(const H3WallRecord& record) {
    records.push_back(record);
}

void check(bool value) {
    if (!value)
        throw std::runtime_error("h3_wall_timing_test");
}

void disabled_does_not_observe_or_change_callable_result() {
    std::atomic_bool cancel = false;
    records.clear();
    int calls = 0, value = 0;
    {
        H3WallTiming wall(false, cancel, collect, []() -> std::uint64_t {
            throw std::runtime_error("disabled clock must not run");
        });
        value = wall.measure(H3WallStage::block, 0, H3WallPhase::describe, 1, [&] {
            ++calls;
            return 17;
        });
    }
    check(calls == 1 && value == 17 && records.empty());
}

void phase_totals_and_plan_metadata_are_bounded() {
    std::atomic_bool cancel = false;
    records.clear();
    ticks = 0;
    MediaResidencyPlan plan;
    plan.rows = 16;
    plan.queries = 8;
    plan.keys = 32;
    plan.devices.resize(2);
    plan.devices[0].device = 1;
    plan.devices[0].arena_bytes = 100;
    plan.devices[1].device = 3;
    plan.devices[1].arena_bytes = 200;
    plan.weights.resize(2);
    plan.weights[0].stored_bytes = 300;
    plan.weights[1].stored_bytes = 400;
    {
        H3WallTiming wall(true, cancel, collect, clock_value);
        for (std::size_t i = 0; i < H3WallTiming::max_records + 5; ++i)
            wall.measure(
                H3WallStage::block,
                i,
                H3WallPhase::prepare,
                42,
                [] { ticks += 7; },
                &plan
            );
    }
    check(records.size() == H3WallTiming::max_records + 1);
    const auto& row = records.front();
    check(!row.summary && row.tokens == 42 && row.wall_ns == 7);
    check(row.plan_observed && row.rows == 16 && row.queries == 8 && row.keys == 32);
    check(row.weight_bytes == 700);
    check(row.device_count == 2 && row.devices[1].id == 3);
    check(row.devices[0].arena_bytes == 100 && row.devices[1].arena_bytes == 200);
    const auto& summary = records.back();
    check(summary.summary && summary.dropped == 5);
    check(summary.calls[1] == H3WallTiming::max_records + 5);
    check(summary.phase_wall_ns[1] == 7 * (H3WallTiming::max_records + 5));
}

void cancellation_and_sink_failure_preserve_primary_exception() {
    std::atomic_bool cancel = false;
    records.clear();
    ticks = 0;
    auto sink = [](const H3WallRecord& record) {
        collect(record);
        throw std::runtime_error("diagnostic sink");
    };
    try {
        H3WallTiming wall(true, cancel, sink, clock_value);
        wall.measure(H3WallStage::block, 4, H3WallPhase::execute, 12, [&] {
            ticks = 25;
            cancel.store(true);
            throw std::runtime_error("primary cancellation cleanup");
        });
        check(false);
    } catch (const std::runtime_error& error) {
        check(std::string(error.what()) == "primary cancellation cleanup");
    }
    check(records.size() == 2 && records[0].wall_ns == 25);
    check(records[0].failed && records[0].cancelled);
    check(records.back().summary && records.back().failed && records.back().cancelled);
    check(records.back().dropped == 1);
}

void broken_clock_does_not_skip_work_or_fabricate_duration() {
    std::atomic_bool cancel = false;
    records.clear();
    bool called = false;
    {
        H3WallTiming wall(true, cancel, collect, []() -> std::uint64_t {
            throw std::runtime_error("diagnostic clock");
        });
        wall.measure(H3WallStage::final, 0, H3WallPhase::execute, 8, [&] {
            called = true;
        });
    }
    check(called && records.size() == 1 && records[0].summary);
    check(records[0].dropped == 1 && records[0].phase_wall_ns[2] == 0);
    check(records[0].calls[2] == 1);
}

void cancellation_flag_is_not_an_execution_failure() {
    std::atomic_bool cancel = false;
    records.clear();
    ticks = 0;
    {
        H3WallTiming wall(true, cancel, collect, clock_value);
        wall.measure(H3WallStage::block, 0, H3WallPhase::describe, 4, [] {
            ticks += 3;
        });
        const int result =
            wall.measure(H3WallStage::block, 0, H3WallPhase::execute, 4, [&] {
                ticks += 9;
                cancel.store(true);
                return 42;
            });
        check(result == 42);
    }
    check(records.size() == 3 && records.back().summary);
    check(!records.back().failed && records.back().cancelled);
    check(records.back().phase_wall_ns[0] == 3 && records.back().phase_wall_ns[2] == 9);
    check(!records[0].plan_observed && !records[0].failed && !records[0].cancelled);
}

void enabled_observation_preserves_reference_and_move_only_results() {
    std::atomic_bool cancel = false;
    records.clear();
    ticks = 0;
    H3WallTiming wall(true, cancel, collect, clock_value);
    int value = 7;
    int& reference =
        wall.measure(H3WallStage::block, 0, H3WallPhase::describe, 1, [&]() -> int& {
            return value;
        });
    reference = 9;
    check(value == 9);
    auto moved = wall.measure(H3WallStage::block, 0, H3WallPhase::prepare, 1, [] {
        return std::make_unique<int>(17);
    });
    check(*moved == 17);
}

void nested_prepare_is_silent_when_disabled_and_aggregates_slices_when_enabled() {
    std::atomic_bool cancel = false;
    records.clear();
    {
        H3WallTiming wall(false, cancel, collect, []() -> std::uint64_t {
            throw std::runtime_error("disabled nested clock");
        });
        check(
            wall.measure_prepare(
                H3WallStage::block,
                0,
                8,
                [](H3PrepareTiming* timing) {
                    check(timing == nullptr);
                    return 42;
                }
            )
            == 42
        );
    }
    check(records.empty());
    ticks = 0;
    {
        H3WallTiming wall(true, cancel, collect, clock_value);
        for (std::size_t block = 0; block < 2; ++block)
            wall.measure_prepare(
                H3WallStage::block, block, 8, [](H3PrepareTiming* timing) {
                    if (!timing)
                        throw std::runtime_error("enabled collector missing");
                    timing->phase(H3PreparePhase::admission);
                    ticks += 2;
                    timing->pause();
                    for (auto dtype :
                         {H3PrepareDtype::i8,
                          H3PrepareDtype::bf16,
                          H3PrepareDtype::fp32}) {
                        timing->count(dtype, H3PrepareCount::started_slices, 1);
                        timing->count(dtype, H3PrepareCount::planned_bytes, 128);
                        timing->phase(H3PreparePhase::staging_allocation);
                        ticks += 3;
                        timing->phase(H3PreparePhase::checkpoint_read);
                        ticks += 5;
                        timing->count(dtype, H3PrepareCount::read_bytes, 128);
                        timing->phase(H3PreparePhase::validation);
                        if (dtype != H3PrepareDtype::i8) {
                            ticks += 11;
                            timing->count(dtype, H3PrepareCount::validated_bytes, 128);
                        }
                        timing->phase(H3PreparePhase::upload);
                        ticks += 7;
                        timing->count(dtype, H3PrepareCount::uploaded_slices, 1);
                        timing->count(dtype, H3PrepareCount::uploaded_bytes, 128);
                        timing->pause();
                    }
                }
            );
    }
    // Two prepare payloads and one summary, regardless of slice count.
    check(records.size() == 3 && records.back().summary);
    const auto& first = records[0];
    check(first.prepare_observed && first.phase == H3WallPhase::prepare);
    check(
        first.wall_ns == 69
        && first.prepare.wall_ns == std::array<std::uint64_t, 5>{2, 9, 15, 22, 21}
    );
    check(first.prepare.dtype[0][std::size_t(H3PrepareCount::validated_bytes)] == 0);
    check(first.prepare.dtype[1][std::size_t(H3PrepareCount::validated_bytes)] == 128);
    check(first.prepare.dtype[2][std::size_t(H3PrepareCount::uploaded_bytes)] == 128);
    const auto& summary = records.back();
    check(summary.phase_wall_ns[1] == 138 && summary.prepare.wall_ns[3] == 44);
    check(summary.prepare.dtype[0][std::size_t(H3PrepareCount::uploaded_slices)] == 2);
}

void nested_partial_cancellation_and_sink_failure_preserve_primary_exception() {
    std::atomic_bool cancel = false;
    records.clear();
    ticks = 0;
    try {
        H3WallTiming wall(
            true,
            cancel,
            [](const H3WallRecord& record) {
                collect(record);
                throw std::runtime_error("nested diagnostic sink");
            },
            clock_value
        );
        wall.measure_prepare(H3WallStage::block, 27, 8, [&](H3PrepareTiming* timing) {
            if (!timing)
                throw std::runtime_error("enabled collector missing");
            try {
                timing->phase(H3PreparePhase::admission);
                ticks += 2;
                timing->count(H3PrepareDtype::bf16, H3PrepareCount::started_slices, 1);
                timing->count(H3PrepareDtype::bf16, H3PrepareCount::planned_bytes, 100);
                timing->phase(H3PreparePhase::staging_allocation);
                ticks += 3;
                timing->phase(H3PreparePhase::checkpoint_read);
                timing->count(H3PrepareDtype::bf16, H3PrepareCount::read_bytes, 40);
                ticks += 7;
                cancel.store(true);
                throw std::runtime_error("primary prepare cancellation");
            } catch (...) {
                timing->pause();
                ticks += 100;  // Simulated abort work stays outside the nested buckets.
                throw;
            }
        });
        check(false);
    } catch (const std::runtime_error& error) {
        check(std::string(error.what()) == "primary prepare cancellation");
    }
    check(records.size() == 2 && records[0].failed && records[0].cancelled);
    check(records[0].wall_ns == 112 && records[0].prepare.wall_ns[2] == 7);
    const auto& counts = records[0].prepare.dtype[std::size_t(H3PrepareDtype::bf16)];
    check(counts[std::size_t(H3PrepareCount::read_bytes)] == 40);
    check(counts[std::size_t(H3PrepareCount::uploaded_slices)] == 0);
    check(counts[std::size_t(H3PrepareCount::uploaded_bytes)] == 0);
    check(records.back().prepare_observed && records.back().dropped == 1);
}

void nested_clock_faults_and_counter_overflow_do_not_change_work() {
    std::atomic_bool cancel = false;
    records.clear();
    ticks = 0;
    static unsigned clock_calls = 0;
    clock_calls = 0;
    {
        H3WallTiming wall(true, cancel, collect, []() -> std::uint64_t {
            if (++clock_calls == 2)
                throw std::runtime_error("nested diagnostic clock");
            return ticks;
        });
        wall.measure_prepare(H3WallStage::final, 0, 8, [](H3PrepareTiming* timing) {
            if (!timing)
                throw std::runtime_error("enabled collector missing");
            timing->phase(H3PreparePhase::admission);
            ticks += 5;
            timing->phase(H3PreparePhase::checkpoint_read);
            timing->count(
                H3PrepareDtype::other,
                H3PrepareCount::planned_bytes,
                std::numeric_limits<std::uint64_t>::max()
            );
            timing->count(H3PrepareDtype::other, H3PrepareCount::planned_bytes, 1);
            ticks += 4;
        });
    }
    check(records.size() == 2 && records[0].wall_ns == 9);
    check(records[0].prepare.dropped == 1 && records[0].prepare.wall_ns[0] == 0);
    check(records[0].prepare.wall_ns[2] == 4 && !records.back().failed);
    check(
        records.back().prepare.dtype[3][2] == std::numeric_limits<std::uint64_t>::max()
    );
}
}  // namespace

int main() {
    disabled_does_not_observe_or_change_callable_result();
    phase_totals_and_plan_metadata_are_bounded();
    cancellation_and_sink_failure_preserve_primary_exception();
    broken_clock_does_not_skip_work_or_fabricate_duration();
    cancellation_flag_is_not_an_execution_failure();
    enabled_observation_preserves_reference_and_move_only_results();
    nested_prepare_is_silent_when_disabled_and_aggregates_slices_when_enabled();
    nested_partial_cancellation_and_sink_failure_preserve_primary_exception();
    nested_clock_faults_and_counter_overflow_do_not_change_work();
}
