#include "kadan/image_batch_monitor.hpp"

#include <atomic>
#include <future>
#include <iostream>

using namespace kadan;
using namespace kadan::serving;
using namespace std::chrono_literals;

namespace {
void check(bool condition, const char* error) {
    if (!condition)
        throw std::runtime_error(error);
}

void isolated_batch_records_its_peak() {
    std::atomic<Bytes> rss{100};
    std::atomic<unsigned> cancellations{0};
    ImageBatchMonitor monitor(
        [&] { return rss.load(); }, [&] { ++cancellations; }, [] { return true; }
    );
    rss.store(250);
    const auto measured = monitor.finish();
    check(measured.has_value(), "isolated_measurement_discarded");
    check(measured->baseline == 100 && measured->peak == 250, "wrong_batch_peak");
    check(cancellations.load() != 0, "cancellation_not_polled");
}

void preexisting_overlap_is_not_sampled() {
    std::atomic<unsigned> probes{0};
    ImageBatchMonitor monitor(
        [&] {
            ++probes;
            return Bytes{1000};
        },
        [] {},
        [] { return false; }
    );
    check(!monitor.finish(), "preexisting_overlap_calibrated");
    check(probes.load() == 0, "contaminated_rss_sampled");
}

void completed_overlap_between_samples_invalidates_measurement() {
    std::atomic<std::uint64_t> activity_epoch{7};
    const auto initial = activity_epoch.load();
    ImageBatchMonitor monitor(
        [] { return Bytes{100}; },
        [] {},
        [&] { return activity_epoch.load() == initial; }
    );
    // A foreign operation starts and ends before the next RSS sample. Active
    // count alone would miss it; the scheduler's epoch preserves that evidence.
    activity_epoch.fetch_add(2);
    check(!monitor.finish(), "completed_overlap_calibrated");
}

void overlap_during_an_rss_read_invalidates_measurement() {
    std::atomic<std::uint64_t> epoch{0};
    std::atomic_bool overlap{false};
    ImageBatchMonitor monitor(
        [&] {
            if (overlap.load())
                ++epoch;
            return Bytes{100};
        },
        [] {},
        [&] { return epoch.load() == 0; }
    );
    overlap.store(true);
    check(!monitor.finish(), "overlap_inside_probe_calibrated");
}

void contaminated_probe_failure_does_not_fail_a_batch() {
    std::atomic_bool isolated{true};
    ImageBatchMonitor monitor(
        []() -> Bytes { throw std::runtime_error("unavailable_rss"); },
        [] {},
        [&] { return isolated.load(); }
    );
    isolated.store(false);
    check(!monitor.finish(), "contaminated_failed_probe_calibrated");
}

void isolated_probe_failure_is_reported() {
    ImageBatchMonitor monitor(
        []() -> Bytes { throw std::runtime_error("unavailable_rss"); },
        [] {},
        [] { return true; }
    );
    try {
        monitor.finish();
    } catch (const std::runtime_error& error) {
        check(
            std::string(error.what()) == "image_host_memory_probe", "wrong_probe_error"
        );
        return;
    }
    throw std::runtime_error("probe_failure_ignored");
}

void cancellation_continues_after_calibration_is_invalidated() {
    std::atomic_bool cancelled{false}, observed{false};
    std::promise<void> cancellation;
    auto notified = cancellation.get_future();
    ImageBatchMonitor monitor(
        [] { return Bytes{100}; },
        [&] {
            if (cancelled.load() && !observed.exchange(true))
                cancellation.set_value();
        },
        [] { return false; }
    );
    cancelled.store(true);
    check(
        notified.wait_for(2s) == std::future_status::ready,
        "cancel_monitor_stopped_with_calibration"
    );
    check(!monitor.finish(), "cancelled_overlap_calibrated");
}

void destruction_joins_sampler_before_releasing_callback_storage() {
    std::promise<void> entered, release;
    std::atomic_bool announced{false};
    auto started = entered.get_future();
    auto permission = release.get_future().share();
    auto monitor = std::make_unique<ImageBatchMonitor>(
        [] { return Bytes{100}; },
        [&] {
            if (!announced.exchange(true))
                entered.set_value();
            permission.wait();
        },
        [] { return false; }
    );
    if (started.wait_for(2s) != std::future_status::ready) {
        release.set_value();
        throw std::runtime_error("sampler_not_started");
    }
    auto cleanup = std::async(std::launch::async, [&] { monitor.reset(); });
    const bool blocked = cleanup.wait_for(0s) != std::future_status::ready;
    release.set_value();
    check(cleanup.wait_for(2s) == std::future_status::ready, "sampler_cleanup_timeout");
    cleanup.get();
    check(blocked, "sampler_outlived_callback_storage");
}
}  // namespace

int main() {
    try {
        isolated_batch_records_its_peak();
        preexisting_overlap_is_not_sampled();
        completed_overlap_between_samples_invalidates_measurement();
        overlap_during_an_rss_read_invalidates_measurement();
        contaminated_probe_failure_does_not_fail_a_batch();
        isolated_probe_failure_is_reported();
        cancellation_continues_after_calibration_is_invalidated();
        destruction_joins_sampler_before_releasing_callback_storage();
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
