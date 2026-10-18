// Timing/admission policy only; no model forwards or device calls.
#include "kadan/image_aten_performance.hpp"

#include <array>
#include <iostream>
#include <limits>
#include <source_location>

namespace {
using namespace kadan::image;

void check(bool value, std::source_location at = std::source_location::current()) {
    if (!value)
        throw std::runtime_error(
            "image_aten_performance_test:" + std::to_string(at.line())
        );
}

std::vector<AtenDenoiserLane> single(std::size_t images, bool streaming = false) {
    return {{0, images, {0, streaming, false, 100, 200}}};
}

std::vector<AtenDenoiserLane> parallel(std::size_t images) {
    return {
        {0, (images + 1) / 2, {0, false, false, 100, 200}},
        {(images + 1) / 2, images / 2, {1, false, false, 100, 200}}
    };
}

constexpr AtenDenoiserWorkload workload{624, 128, 256, 8, false};

void deferred_batch_calibration() {
    AtenDenoiserPerformance policy;
    const std::vector baseline{AtenDenoiserMeasurement{0, 2, false, 5, 4}};
    const std::vector costly{
        AtenDenoiserMeasurement{0, 1, false, 5, 2},
        AtenDenoiserMeasurement{1, 1, true, 3, 2}
    };
    policy.defer_observation(workload, baseline);
    check(!policy.denoise_seconds(workload, single(2)));
    check(policy.choose_candidates(workload, {single(2), parallel(2)}).size() == 1);
    // An overlapped or cancelled first run cannot consume its singleton probe.
    policy.record_batch_measurement(false);
    policy.record_batch_measurement(true);
    check(!policy.denoise_seconds(workload, single(2)));
    policy.defer_observation(workload, baseline);
    policy.record_batch_measurement(true);
    check(policy.denoise_seconds(workload, single(2)) == 4);
    check(policy.choose_candidates(workload, {single(2), parallel(2)}).size() == 2);
    // Rejecting a replica probe preserves the prior estimate and pending probe.
    policy.defer_observation(workload, costly);
    policy.record_batch_measurement(false);
    check(policy.choose_candidates(workload, {single(2), parallel(2)}).size() == 2);
    policy.defer_observation(workload, costly);
    policy.record_batch_measurement(true);
    check(policy.choose_candidates(workload, {single(2), parallel(2)}).size() == 1);
    const std::vector contaminated{AtenDenoiserMeasurement{0, 2, false, 5, 400}};
    policy.defer_observation(workload, contaminated);
    policy.record_batch_measurement(false);
    check(policy.denoise_seconds(workload, single(2)) == 4);
    // A failed batch is discarded even if a later caller accepts with no sample.
    policy.record_batch_measurement(true);
    check(policy.denoise_seconds(workload, single(2)) == 4);
}

void loading_and_amortization() {
    AtenDenoiserPerformance policy;
    check(policy.choose_candidates(workload, {single(2), parallel(2)}).size() == 1);
    const std::array baseline{AtenDenoiserMeasurement{0, 2, false, 5, 4}};
    policy.observe(workload, baseline);
    // Repeated admission never consumes the probe before execution succeeds.
    check(policy.choose_candidates(workload, {single(2), parallel(2)}).size() == 2);
    check(policy.choose_candidates(workload, {single(2), parallel(2)}).size() == 2);
    const std::array costly{
        AtenDenoiserMeasurement{0, 1, false, 5, 2},
        AtenDenoiserMeasurement{1, 1, true, 3, 2}
    };
    policy.observe(workload, costly);
    // Faster resident computation alone does not justify the replication cost.
    check(policy.choose_candidates(workload, {single(2), parallel(2)}).size() == 1);
    // A larger batch can amortize measured fixed loading costs.
    check(policy.choose_candidates(workload, {single(4), parallel(4)}).size() == 2);
    const std::array slow_second{
        AtenDenoiserMeasurement{0, 1, false, 5, 2},
        AtenDenoiserMeasurement{1, 1, true, 3, 10}
    };
    policy.observe(workload, slow_second);
    check(policy.choose_candidates(workload, {single(4), parallel(4)}).size() == 1);
}

void separate_loading_regimes_and_shapes() {
    AtenDenoiserPerformance policy;
    const std::array baseline{AtenDenoiserMeasurement{0, 2, false, 5, 4}};
    const std::array costly{
        AtenDenoiserMeasurement{0, 1, false, 5, 2},
        AtenDenoiserMeasurement{1, 1, true, 3, 2}
    };
    policy.observe(workload, baseline);
    policy.observe(workload, costly);
    auto warm = workload;
    warm.weights_cached = true;
    check(policy.choose_candidates(warm, {single(2), parallel(2)}).size() == 1);
    const std::array warm_baseline{AtenDenoiserMeasurement{0, 2, false, 1, 4}};
    const std::array warm_parallel{
        AtenDenoiserMeasurement{0, 1, false, 1, 2},
        AtenDenoiserMeasurement{1, 1, true, 1, 2}
    };
    policy.observe(warm, warm_baseline);
    check(policy.choose_candidates(warm, {single(2), parallel(2)}).size() == 2);
    policy.observe(warm, warm_parallel);
    check(policy.choose_candidates(warm, {single(2), parallel(2)}).size() == 2);
    check(policy.choose_candidates(workload, {single(2), parallel(2)}).size() == 1);
    auto different = warm;
    ++different.steps;
    check(policy.choose_candidates(different, {single(2), parallel(2)}).size() == 1);
    different = warm;
    ++different.tokens;
    check(policy.choose_candidates(different, {single(2), parallel(2)}).size() == 1);
    different = warm;
    ++different.points;
    check(policy.choose_candidates(different, {single(2), parallel(2)}).size() == 1);
    different = warm;
    ++different.reference_points;
    check(policy.choose_candidates(different, {single(2), parallel(2)}).size() == 1);
}

void retained_banks_and_reload_cost() {
    AtenDenoiserPerformance policy;
    auto retained = workload;
    retained.weights_cached = true;
    retained.gpu_weights_cached = true;
    const std::array baseline{AtenDenoiserMeasurement{0, 2, false, 5, 4}};
    const std::array split{
        AtenDenoiserMeasurement{0, 1, false, 5, 2},
        AtenDenoiserMeasurement{1, 1, true, 3, 2}
    };
    policy.observe(retained, baseline);
    policy.observe(retained, split);
    // Current bank residency overrides older upload/copy timings.
    const AtenDenoiserCost single_hit{{0}, 0};
    const AtenDenoiserCost both_hit{{0, 0}, 0};
    check(
        policy
            .choose_candidates(
                retained, {single(2), parallel(2)}, {single_hit, both_hit}
            )
            .size()
        == 2
    );
    // Evicting the text bank costs more on the next batch than denoising saves.
    const AtenDenoiserCost evicts_text{{0, 0}, 3};
    check(
        policy
            .choose_candidates(
                retained, {single(2), parallel(2)}, {single_hit, evicts_text}
            )
            .size()
        == 1
    );
    check(
        policy
            .choose_candidates(
                retained, {single(4), parallel(4)}, {single_hit, evicts_text}
            )
            .size()
        == 2
    );
    // A known miss must override an earlier retained-bank hit as well.
    const AtenDenoiserCost second_miss{{0, 5}, 0};
    check(
        policy
            .choose_candidates(
                retained, {single(2), parallel(2)}, {single_hit, second_miss}
            )
            .size()
        == 1
    );
    // Compute timing can come from a replica when its current load cost is known.
    auto second_device = single(2);
    second_device.front().stage.device = 1;
    check(
        policy
            .choose_candidates(
                retained, {second_device, parallel(2)}, {single_hit, both_hit}
            )
            .size()
        == 2
    );
    auto host_only = retained;
    host_only.gpu_weights_cached = false;
    check(
        policy
            .choose_candidates(
                host_only, {single(2), parallel(2)}, {single_hit, both_hit}
            )
            .size()
        == 1
    );
    const AtenDenoiserCost missing_override{{0}, 0};
    const AtenDenoiserCost invalid_penalty{{0, 0}, -1};
    const AtenDenoiserCost invalid_load{
        {0, std::numeric_limits<double>::infinity()}, 0
    };
    for (const auto& invalid : {missing_override, invalid_penalty, invalid_load})
        check(
            policy
                .choose_candidates(
                    retained, {single(2), parallel(2)}, {single_hit, invalid}
                )
                .size()
            == 1
        );
}

void calibration_cost_gate() {
    AtenDenoiserPerformance policy;
    const std::array baseline{AtenDenoiserMeasurement{0, 2, false, 0, 4}};
    const AtenDenoiserCost hit{{0}, 0};
    const AtenDenoiserCost upload{{0, 4}, 0};
    const AtenDenoiserCost reload{{0, 0}, 5};
    const AtenDenoiserCost combined{{0, 2}, 2};
    const AtenDenoiserCost unknown_upload{{0, std::nullopt}, 4};
    // New prompt lengths retain independent timing histories, but must not
    // repeat probes whose known loading costs already exhaust their baseline.
    for (const auto tokens : {128, 129}) {
        auto shape = workload;
        shape.tokens = tokens;
        policy.observe(shape, baseline);
        for (const auto& expensive : {upload, reload, combined, unknown_upload})
            check(
                policy
                    .choose_candidates(
                        shape, {single(2), parallel(2)}, {hit, expensive}
                    )
                    .size()
                == 1
            );
    }
    // The gate does not permanently disable a device: larger batches can
    // amortize the same cost, and a smaller current cost restores eligibility.
    check(
        policy.choose_candidates(workload, {single(4), parallel(4)}, {hit, upload})
            .size()
        == 2
    );
    const AtenDenoiserCost cheaper{{0, 1}, 1};
    check(
        policy.choose_candidates(workload, {single(2), parallel(2)}, {hit, cheaper})
            .size()
        == 2
    );
    const AtenDenoiserCost unknown{{0, std::nullopt}, 0};
    check(
        policy.choose_candidates(workload, {single(2), parallel(2)}, {hit, unknown})
            .size()
        == 2
    );
    // Planning alone never consumes calibration; a successful observation does.
    check(
        policy.choose_candidates(workload, {single(2), parallel(2)}, {hit, unknown})
            .size()
        == 2
    );
    // When no single-device plan fits, its cost cannot veto the required pair.
    check(
        policy.choose_candidates(workload, {{}, parallel(2)}, {{}, reload}).size() == 2
    );
    check(policy.choose_candidates(workload, {single(2)}, {hit}).size() == 1);
}

void narrower_calibration_candidates() {
    AtenDenoiserPerformance policy;
    const std::array baseline{AtenDenoiserMeasurement{0, 6, false, 0, 6}};
    policy.observe(workload, baseline);
    const std::vector<AtenDenoiserLane> three{
        {0, 2, {0, false, false, 100, 200}},
        {2, 2, {1, false, false, 100, 200}},
        {4, 2, {5, false, false, 100, 200}}
    };
    const std::vector<AtenDenoiserLane> four{
        {0, 2, {0, false, false, 100, 200}},
        {2, 2, {1, false, false, 100, 200}},
        {4, 1, {5, false, false, 100, 200}},
        {5, 1, {7, false, false, 100, 200}}
    };
    const AtenDenoiserCost hit{{0}, 0};
    const AtenDenoiserCost pair{{0, 1}, 0};
    const AtenDenoiserCost costly_three{{0, 1, 3}, 2};
    const AtenDenoiserCost costly_four{{0, 1, 3, 4}, 2};
    check(
        policy
            .choose_candidates(
                workload,
                {single(6), parallel(6), three, four},
                {hit, pair, costly_three, costly_four}
            )
            .size()
        == 2
    );
    // Candidate order must not hide a worthwhile narrower subset.
    check(
        policy
            .choose_candidates(
                workload,
                {four, three, parallel(6), single(6)},
                {costly_four, costly_three, pair, hit}
            )
            .size()
        == 2
    );
    const AtenDenoiserCost cheap_three{{0, 1, 1}, 0};
    check(
        policy
            .choose_candidates(
                workload,
                {single(6), parallel(6), three, four},
                {hit, pair, cheap_three, costly_four}
            )
            .size()
        == 3
    );
}

void memory_and_incomplete_observations() {
    AtenDenoiserPerformance policy;
    check(policy.choose_candidates(workload, {{}, parallel(2)}).size() == 2);
    check(
        policy.choose_candidates(workload, {single(2, true), parallel(2)}).size() == 2
    );
    check(policy.choose_candidates(workload, {single(1), single(1)}).size() == 1);
    const std::array invalid{AtenDenoiserMeasurement{
        0, 2, false, 1, std::numeric_limits<double>::quiet_NaN()
    }};
    policy.observe(workload, invalid);
    check(policy.choose_candidates(workload, {single(2), parallel(2)}).size() == 1);
    const std::array baseline{AtenDenoiserMeasurement{0, 2, false, 5, 4}};
    policy.observe(workload, baseline);
    const std::array incomplete{
        AtenDenoiserMeasurement{0, 1, false, 1, 2},
        AtenDenoiserMeasurement{1, 1, true, 1, 0}
    };
    policy.observe(workload, incomplete);
    check(policy.choose_candidates(workload, {single(2), parallel(2)}).size() == 2);
}

void heterogeneous_device_candidates() {
    AtenDenoiserPerformance policy;
    const std::vector<AtenDenoiserLane> three{
        {0, 2, {0, false, false, 100, 200}},
        {2, 2, {1, false, false, 100, 200}},
        {4, 2, {2, false, false, 100, 200}}
    };
    const std::vector candidates{single(6), parallel(6), three};
    check(policy.choose_candidates(workload, {single(6)}).size() == 1);
    check(policy.choose_candidates(workload, candidates).size() == 1);
    const std::array baseline{AtenDenoiserMeasurement{0, 6, false, 0.2, 6}};
    policy.observe(workload, baseline);
    check(policy.choose_candidates(workload, candidates).size() == 3);
    const std::array slow_third{
        AtenDenoiserMeasurement{0, 2, false, 0.2, 2},
        AtenDenoiserMeasurement{1, 2, true, 0.2, 2},
        AtenDenoiserMeasurement{2, 2, true, 0.2, 10}
    };
    policy.observe(workload, slow_third);
    check(policy.choose_candidates(workload, candidates).size() == 2);
    const std::array narrower{
        AtenDenoiserMeasurement{0, 3, false, 0.2, 3},
        AtenDenoiserMeasurement{1, 3, true, 0.2, 3}
    };
    policy.observe(workload, narrower);
    // Running the chosen pair does not erase the slow third device's timing.
    check(policy.choose_candidates(workload, candidates).size() == 2);
    auto fast_third = slow_third;
    fast_third[2].denoise_seconds = 2;
    policy.observe(workload, fast_third);
    check(policy.choose_candidates(workload, candidates).size() == 3);
    check(policy.choose_candidates(workload, {single(6)}).size() == 1);
    // A fourth device can be discovered, then excluded when its cost is high.
    const std::vector<AtenDenoiserLane> four{
        {0, 2, {0, false, false, 100, 200}},
        {2, 2, {1, false, false, 100, 200}},
        {4, 1, {2, false, false, 100, 200}},
        {5, 1, {3, false, false, 100, 200}}
    };
    auto expanded = candidates;
    expanded.push_back(four);
    check(policy.choose_candidates(workload, expanded).size() == 4);
    const std::array slow_fourth{
        AtenDenoiserMeasurement{0, 2, false, 0.2, 2},
        AtenDenoiserMeasurement{1, 2, true, 0.2, 2},
        AtenDenoiserMeasurement{2, 1, true, 0.2, 1},
        AtenDenoiserMeasurement{3, 1, true, 0.2, 20}
    };
    policy.observe(workload, slow_fourth);
    check(policy.choose_candidates(workload, expanded).size() == 3);
    // When memory excludes a single-device plan, discovery starts from two.
    AtenDenoiserPerformance constrained;
    check(constrained.choose_candidates(workload, {parallel(6), three}).size() == 2);
    constrained.observe(workload, narrower);
    check(constrained.choose_candidates(workload, {parallel(6), three}).size() == 3);
}

void bounded_history() {
    AtenDenoiserPerformance policy;
    const std::array baseline{AtenDenoiserMeasurement{0, 2, false, 5, 4}};
    for (std::size_t i = 0; i <= 64; ++i) {
        auto shape = workload;
        shape.steps += i;
        policy.observe(shape, baseline);
    }
    check(policy.choose_candidates(workload, {single(2), parallel(2)}).size() == 1);
    auto latest = workload;
    latest.steps += 64;
    check(policy.choose_candidates(latest, {single(2), parallel(2)}).size() == 2);
}
}  // namespace

int main() {
    try {
        deferred_batch_calibration();
        loading_and_amortization();
        separate_loading_regimes_and_shapes();
        retained_banks_and_reload_cost();
        calibration_cost_gate();
        narrower_calibration_candidates();
        memory_and_incomplete_observations();
        heterogeneous_device_candidates();
        bounded_history();
        std::cout << "image ATen performance policy units passed\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
