// Clock/aggregation and JSON data only; no model, forward pass or device.
#include "kadan/image_timing.hpp"

#include <iostream>
#include <limits>
#include <source_location>
#include <vector>

using namespace kadan;
using namespace kadan::image;

namespace {
std::uint64_t tick = 0;
std::size_t clock_reads = 0;

std::uint64_t clock_value() noexcept {
    ++clock_reads;
    return tick;
}

void check(bool ok, std::source_location at = std::source_location::current()) {
    if (!ok)
        throw std::runtime_error("image_timing_test:" + std::to_string(at.line()));
}

void aggregation_and_reset() {
    std::vector<DenoisingStepTiming> records;
    const DenoisingTimings::Sink sink = [&](const auto& value) {
        records.push_back(value);
    };
    DenoisingTimings timing(sink, clock_value);
    MediaTransferCounters start, finish;
    start.weight_upload = 100;
    finish.weight_upload = 500;
    finish.control_download = 12;
    tick = 0;
    timing.reset(1, 16390, 6, 1.f, .02f, start);
    timing.observe_rows(512);
    timing.observe_rows(256);
    timing.observe_device(0);
    timing.observe_device(0);
    timing.observe_device(1);
    tick = 10;
    {
        DenoisingStageTimer timer(&timing, DenoisingStage::qkv);
        tick = 25;
    }
    tick = 30;
    {
        DenoisingStageTimer timer(&timing, DenoisingStage::qkv);
        tick = 50;
    }
    timing.completed_block();
    tick = 70;
    timing.publish(DenoisingOutcome::completed, finish);
    timing.publish(DenoisingOutcome::failed, finish);
    check(records.size() == 1);
    const auto& first = records[0];
    const auto& qkv = first.stages[std::size_t(DenoisingStage::qkv)];
    check(first.step == 1 && first.tokens == 16390 && first.text == 6);
    check(first.rows_min == 256 && first.rows_max == 512);
    check(first.device_count == 2 && first.devices[0] == 0 && first.devices[1] == 1);
    check(
        first.completed_blocks == 1 && first.wall_ns == 70 && first.unaccounted_ns == 35
    );
    check(qkv.calls == 2 && qkv.total_ns == 35 && qkv.min_ns == 15 && qkv.max_ns == 20);
    check(qkv.first_ns == 15 && qkv.last_ns == 20);
    check(
        first.transfers.weight_upload == 400 && first.transfers.control_download == 12
    );
    check(!first.transfer_counters_reset);
    nlohmann::json json = first;
    check(json["stages"]["qkv"]["total_ns"] == 35);
    check(json["attention_precision_requested"] == "pedantic_fp32");
    check(
        json["outcome"] == "completed" && json["transfer_bytes"]["weight_upload"] == 400
    );

    tick = 100;
    timing.reset(
        2, 16390, 6, .02f, 0.f, finish, AttentionPrecision::experimental_pv_tf32
    );
    tick = 110;
    timing.publish(DenoisingOutcome::cancelled, finish);
    check(records.size() == 2);
    const auto& second = records[1];
    check(second.step == 2 && second.started_ns == 100 && second.wall_ns == 10);
    json = second;
    check(json["attention_precision_requested"] == "experimental_pv_tf32");
    check(second.outcome == DenoisingOutcome::cancelled && second.next_sigma == 0.f);
    check(
        second.device_count == 0 && second.rows_min == 0 && second.completed_blocks == 0
    );
    check(
        second.transfers.weight_upload == 0 && second.transfers.control_download == 0
    );
    for (const auto& stage : second.stages)
        check(stage.calls == 0 && stage.total_ns == 0 && stage.first_ns == 0);

    timing.reset(1, 100, 4, 1.f, .02f, finish);
    timing.publish(DenoisingOutcome::not_selected, finish);
    check(records.size() == 3);
    json = records.back();
    check(json["attention_precision_requested"] == "pedantic_fp32");
}

void cancelled_scope_and_nonfatal_sink() {
    std::vector<DenoisingStepTiming> records;
    const DenoisingTimings::Sink sink = [&](const auto& value) {
        records.push_back(value);
        throw std::runtime_error("diagnostic_sink_failed");
    };
    DenoisingTimings timing(sink, clock_value);
    tick = 200;
    timing.reset(1, 100, 4, 1.f, .02f, {});
    try {
        tick = 210;
        DenoisingStageTimer timer(&timing, DenoisingStage::attention);
        tick = 230;
        throw std::runtime_error("cancelled");
    } catch (const std::runtime_error& error) {
        check(std::string_view(error.what()) == "cancelled");
        const auto original = std::current_exception();
        tick = 250;
        timing.publish(DenoisingOutcome::cancelled, {});
        check(std::current_exception() == original);
    }
    timing.publish(DenoisingOutcome::failed, {});
    check(records.size() == 1 && records[0].outcome == DenoisingOutcome::cancelled);
    check(records[0].stages[std::size_t(DenoisingStage::attention)].total_ns == 20);
    check(records[0].wall_ns == 50 && records[0].completed_blocks == 0);
}

void disabled_reads_no_clock() {
    std::size_t published = 0;
    const DenoisingTimings::Sink sink = [&](const auto&) {
        ++published;
    };
    DenoisingTimings inactive(sink, clock_value);
    clock_reads = 0;
    for (int i = 0; i < 100; ++i) {
        DenoisingStageTimer disabled(nullptr, DenoisingStage::qkv);
        DenoisingStageTimer not_started(&inactive, DenoisingStage::feed_forward);
    }
    inactive.publish(DenoisingOutcome::failed, {});
    check(clock_reads == 0 && published == 0);
}

void bounded_devices_counters_and_record() {
    DenoisingStepTiming result;
    const DenoisingTimings::Sink sink = [&](const auto& value) {
        result = value;
    };
    DenoisingTimings timing(sink, clock_value);
    MediaTransferCounters start;
    start.weight_upload = 100;
    tick = 0;
    timing.reset(1, 100, 4, 1.f, .02f, start);
    for (int device = 0; device < 20; ++device)
        timing.observe_device(device);
    tick = 10;
    timing.publish(DenoisingOutcome::failed, {});
    check(result.device_count == 16 && result.devices_truncated);
    check(result.transfer_counters_reset && result.transfers.weight_upload == 0);
    check(result.outcome == DenoisingOutcome::failed);

    const auto maximum = std::numeric_limits<std::uint64_t>::max();
    result.attention_precision_requested = AttentionPrecision::experimental_pv_tf32;
    result.started_ns = result.wall_ns = result.unaccounted_ns = maximum;
    result.step = result.tokens = result.text = result.completed_blocks = maximum;
    result.rows_min = result.rows_max = maximum;
    for (auto& stage : result.stages)
        stage = {maximum, maximum, maximum, maximum, maximum, maximum};
    result.transfers = {
        maximum, maximum, maximum, maximum, maximum, maximum, maximum, maximum
    };
    for (auto& device : result.devices)
        device = std::numeric_limits<int>::max();
    nlohmann::json json = result;
    json["id"] = std::string(32, 'f');
    json["image"] = 4;
    check(("KADAN_IMAGE_TIMING " + json.dump() + '\n').size() <= 4096);
}
}  // namespace

int main() {
    aggregation_and_reset();
    cancelled_scope_and_nonfatal_sink();
    disabled_reads_no_clock();
    bounded_devices_counters_and_record();
    std::cout << "image timing scalar units passed (no inference)\n";
}
