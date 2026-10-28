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
}  // namespace

int main() {
    disabled_does_not_observe_or_change_callable_result();
    phase_totals_and_plan_metadata_are_bounded();
    cancellation_and_sink_failure_preserve_primary_exception();
    broken_clock_does_not_skip_work_or_fabricate_duration();
    cancellation_flag_is_not_an_execution_failure();
    enabled_observation_preserves_reference_and_move_only_results();
}
