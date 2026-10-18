// Pure collection policy: no inference, CUDA calls or model fixtures.
#include "kadan/image_batch_performance.hpp"

#include <iostream>
#include <source_location>

#include "kadan/image_aten_performance.hpp"

namespace {
using namespace kadan::image;

void check(bool value, std::source_location at = std::source_location::current()) {
    if (!value)
        throw std::runtime_error(
            "image_batch_performance_test:" + std::to_string(at.line())
        );
}

std::vector<ImageBatchPlan> plans(std::size_t count) {
    return std::vector<ImageBatchPlan>(count, {{100, 100}, {0, std::nullopt}});
}

void bounded_single_device_learning() {
    ImageBatchPerformance policy;
    const auto candidates = plans(32);
    check(policy.capacity(candidates) == 1);
    policy.observe(1, 3, 0, 2);
    check(policy.capacity(candidates) == 2);
    // Advice and collection can ask repeatedly without consuming exploration.
    check(policy.capacity(candidates) == 2);
    policy.observe(2, 5, 0, 4);
    check(policy.capacity(candidates) == 3);
    policy.observe(3, 7, 0, 6);
    check(policy.capacity(candidates) == 4);
    // A larger measured batch loses throughput. Its successor is not probed
    // just because memory permits it; return to the faster measured size.
    policy.observe(4, 12, 0, 10);
    check(policy.capacity(candidates) == 3);
    check(policy.capacity(std::span(candidates).first(2)) == 2);
    check(policy.capacity(std::span(candidates).first(1)) == 1);
    check(policy.capacity({}) == 1);
}

void expensive_displacement_and_unknown_device() {
    ImageBatchPerformance policy;
    auto candidates = plans(4);
    // An initial cold upload must not make a warm reference request appear
    // cheap enough to justify another large upload or text-bank eviction.
    policy.observe(1, 103, 100, 2);
    candidates[1].cost.loading_seconds = 12;
    check(policy.capacity(candidates) == 1);
    // No destination-device denoising estimate: known reload still gates the
    // candidate, including a parallel plan selected as a calibration probe.
    candidates[1].cost.denoise_seconds.reset();
    check(policy.capacity(candidates) == 1);
    candidates[1].cost.loading_seconds = 0;
    check(policy.capacity(candidates) == 2);
    policy.observe(2, 5, 0, 4);
    candidates[2].cost.loading_seconds = 30;
    check(policy.capacity(candidates) == 2);
    // A costly successful calibration is normalized to compute time. Today's
    // cost of displacing banks remains explicit instead of becoming baseline.
    policy.observe(3, 37, 30, 6);
    candidates[3].cost.loading_seconds = 30;
    check(policy.capacity(candidates) == 2);
}

void beneficial_residency_and_heterogeneous_execution() {
    ImageBatchPerformance policy;
    auto candidates = plans(4);
    policy.observe(1, 3, 0, 2);
    policy.observe(2, 6, 0, 4);
    // A measured faster parallel placement improves throughput even with an
    // upload. The supplied duration is the slowest lane, not summed GPUs.
    candidates[2].cost = {1, 2};
    check(policy.capacity(candidates) == 3);
    policy.observe(3, 6, 1, 2);
    // Three heterogeneous devices remain eligible when their actual critical
    // lane and current residency make four images faster per image.
    candidates[3].cost = {0, 2.5};
    check(policy.capacity(candidates) == 4);
    candidates[3].cost = {0, 12};
    check(policy.capacity(candidates) == 3);
    // New residency can make a formerly expensive size worthwhile again.
    candidates[3].cost = {0, 2};
    check(policy.capacity(candidates) == 4);
}

void invalid_observations_and_costs() {
    ImageBatchPerformance policy;
    auto candidates = plans(32);
    const auto nan = std::numeric_limits<double>::quiet_NaN();
    policy.observe(0, 3, 0, 2);
    policy.observe(33, 3, 0, 2);
    policy.observe(1, nan, 0, 2);
    policy.observe(1, 3, -1, 2);
    policy.observe(1, 3, 2, 2);
    check(policy.capacity(candidates) == 1);
    policy.observe(1, 3, 0, 2);
    candidates[1].cost.loading_seconds = nan;
    check(policy.capacity(candidates) == 1);
    candidates[1].cost = {0, -1};
    check(policy.capacity(candidates) == 1);
    candidates[1].cost = {0, std::nullopt};
    check(policy.capacity(candidates) == 2);
    // A distinct geometry/model owns a fresh history, never the base-image rate.
    ImageBatchPerformance other;
    check(other.capacity(candidates) == 1);
}

void streaming_is_a_separate_execution_regime() {
    ImageBatchPerformance policy;
    auto candidates = plans(4);
    policy.observe(1, 3, 0, 2);
    candidates[1].cost.streamed_weights = true;
    check(policy.capacity(candidates) == 1);
    // Pressure can require operator streaming even for one image. Learn that
    // execution rate independently; repeated streamed reads are not subtracted
    // as though their weights survived for the next batch.
    for (auto& candidate : candidates)
        candidate.cost.streamed_weights = true;
    check(policy.capacity(candidates) == 1);
    policy.observe(1, 21, 0, 20, true);
    check(policy.capacity(candidates) == 2);
    policy.observe(2, 31, 0, 30, true);
    check(policy.capacity(candidates) == 3);
    // Recovering residency restores its own bounded learning history.
    for (auto& candidate : candidates)
        candidate.cost.streamed_weights = false;
    check(policy.capacity(candidates) == 2);
}

void native_estimates_preserve_count_specific_regressions() {
    ImageBatchPerformance collection;
    AtenDenoiserPerformance execution;
    const AtenDenoiserWorkload workload{624, 128, 256, 8, true, true};
    for (std::size_t images = 1; images <= 4; ++images) {
        const auto denoise = images == 4 ? 12.0 : 2.0 * images;
        collection.observe(images, denoise + images, 0, denoise);
        const std::array timing{AtenDenoiserMeasurement{0, images, false, 0, denoise}};
        execution.observe(workload, timing);
    }
    auto candidates = plans(5);
    for (std::size_t images = 1; images <= candidates.size(); ++images) {
        const std::array lane{AtenDenoiserLane{0, images, {0, false, false, 100, 100}}};
        candidates[images - 1].cost.denoise_seconds =
            execution.denoise_seconds(workload, lane);
        check(candidates[images - 1].cost.denoise_seconds.has_value());
    }
    // Native estimates must preserve the measured 3s/image sizes 1..3 after
    // observing a slower 4s/image size four, rather than extrapolate it to all.
    check(collection.capacity(candidates) == 3);

    // A genuinely better placement may still replace the old single-device
    // size-four timing. Keep using candidate estimates, not stale whole-batch
    // latency regardless of which devices now execute it.
    const std::array parallel_timing{
        AtenDenoiserMeasurement{0, 2, false, 0, 4},
        AtenDenoiserMeasurement{1, 2, true, 0, 3}
    };
    execution.observe(workload, parallel_timing);
    const std::array pair{
        AtenDenoiserLane{0, 2, {0, false, false, 100, 100}},
        AtenDenoiserLane{2, 2, {1, false, false, 100, 100}}
    };
    candidates[3].cost.denoise_seconds = execution.denoise_seconds(workload, pair);
    check(candidates[3].cost.denoise_seconds == 4.0);
    check(collection.capacity(candidates) == 4);
    // A new observation replaces only the matching device/role/count sample.
    const std::array slower_pair{
        AtenDenoiserMeasurement{0, 2, false, 0, 4},
        AtenDenoiserMeasurement{1, 2, true, 0, 16}
    };
    execution.observe(workload, slower_pair);
    candidates[3].cost.denoise_seconds = execution.denoise_seconds(workload, pair);
    check(collection.capacity(candidates) == 3);
}
}  // namespace

int main() {
    try {
        bounded_single_device_learning();
        expensive_displacement_and_unknown_device();
        beneficial_residency_and_heterogeneous_execution();
        invalid_observations_and_costs();
        streaming_is_a_separate_execution_regime();
        native_estimates_preserve_count_specific_regressions();
        std::cout << "image batch performance policy units passed\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
