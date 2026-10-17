#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <functional>
#include <string_view>

#include <nlohmann/json.hpp>

#include "kadan/image_attention.hpp"
#include "kadan/media_session.hpp"

namespace kadan::image {
enum class DenoisingStage {
    preflight,
    preparation,
    ingress,
    qkv,
    rotary,
    attention,
    attention_output,
    feed_forward,
    final_output,
    scheduler,
    count
};

inline constexpr std::array<std::string_view, 10> denoising_stage_names{
    "preflight",
    "preparation",
    "ingress",
    "qkv",
    "rotary",
    "attention",
    "attention_output",
    "feed_forward",
    "final_output",
    "scheduler"
};

struct DenoisingStageTime {
    std::uint64_t calls = 0, total_ns = 0, min_ns = 0, max_ns = 0, first_ns = 0,
                  last_ns = 0;
};

enum class DenoisingOutcome {
    completed,
    cancelled,
    failed,
    not_selected
};

struct DenoisingStepTiming {
    AttentionPrecision attention_precision_requested =
        AttentionPrecision::pedantic_fp32;
    std::size_t step = 0, tokens = 0, text = 0, completed_blocks = 0, rows_min = 0,
                rows_max = 0, device_count = 0;
    float sigma = 0, next_sigma = 0;
    std::array<int, 16> devices{};
    bool devices_truncated = false, transfer_counters_reset = false;
    DenoisingOutcome outcome = DenoisingOutcome::completed;
    std::uint64_t started_ns = 0, wall_ns = 0, unaccounted_ns = 0;
    std::array<DenoisingStageTime, std::size_t(DenoisingStage::count)> stages{};
    MediaTransferCounters transfers;
};

inline void to_json(nlohmann::json& summary, const DenoisingStepTiming& value) {
    nlohmann::json stages = nlohmann::json::object();
    for (std::size_t stage = 0; stage < value.stages.size(); ++stage) {
        const auto& time = value.stages[stage];
        stages[denoising_stage_names[stage]] = {
            {"calls", time.calls},
            {"total_ns", time.total_ns},
            {"min_ns", time.min_ns},
            {"max_ns", time.max_ns},
            {"first_ns", time.first_ns},
            {"last_ns", time.last_ns}
        };
    }
    nlohmann::json devices = nlohmann::json::array();
    for (std::size_t device = 0; device < value.device_count; ++device)
        devices.push_back(value.devices[device]);
    const std::array outcomes{"completed", "cancelled", "failed", "not_selected"};
    const auto& bytes = value.transfers;
    summary = {
        {"type", "image_denoising_step"},
        {"version", 1},
        {"backend", "resident_bf16"},
        {"attention_precision_requested",
         attention_precision_name(value.attention_precision_requested)},
        {"outcome", outcomes.at(std::size_t(value.outcome))},
        {"step", value.step},
        {"tokens", value.tokens},
        {"text_tokens", value.text},
        {"sigma", value.sigma},
        {"next_sigma", value.next_sigma},
        {"completed_blocks", value.completed_blocks},
        {"rows_min", value.rows_min},
        {"rows_max", value.rows_max},
        {"devices", devices},
        {"devices_truncated", value.devices_truncated},
        {"started_steady_ns", value.started_ns},
        {"wall_ns", value.wall_ns},
        {"unaccounted_ns", value.unaccounted_ns},
        {"stages", stages},
        {"transfer_counters_reset", value.transfer_counters_reset},
        {"transfer_bytes",
         {{"activation_upload", bytes.activation_upload},
          {"activation_download", bytes.activation_download},
          {"local_activation", bytes.local_activation},
          {"input_upload", bytes.input_upload},
          {"weight_upload", bytes.weight_upload},
          {"output_download", bytes.output_download},
          {"peer_activation", bytes.peer_activation},
          {"control_download", bytes.control_download}}}
    };
}

// Fixed storage per attempted resident step. Diagnostics cannot change execution
// or exception cleanup; failed/cancelled stages include their unwind time.
class DenoisingTimings {
public:
    using Sink = std::function<void(const DenoisingStepTiming&)>;
    using Clock = std::uint64_t (*)() noexcept;

    explicit DenoisingTimings(const Sink& sink, Clock clock = now)
        : sink_(sink), clock_(clock) {
    }

    // The generator's sink outlives this step collector; never borrow a temporary.
    DenoisingTimings(Sink&&, Clock = now) = delete;
    DenoisingTimings(const DenoisingTimings&) = delete;
    DenoisingTimings& operator=(const DenoisingTimings&) = delete;

    void reset(
        std::size_t step,
        std::size_t tokens,
        std::size_t text,
        float sigma,
        float next,
        const MediaTransferCounters& transfers,
        AttentionPrecision precision = AttentionPrecision::pedantic_fp32
    ) noexcept {
        data_ = {};
        data_.attention_precision_requested = precision;
        data_.step = step;
        data_.tokens = tokens;
        data_.text = text;
        data_.sigma = sigma;
        data_.next_sigma = next;
        baseline_ = transfers;
        data_.started_ns = clock_();
        active_ = true;
    }

    void observe_rows(std::size_t rows) noexcept {
        if (!data_.rows_min || rows < data_.rows_min)
            data_.rows_min = rows;
        data_.rows_max = std::max(data_.rows_max, rows);
    }

    void observe_device(int device) noexcept {
        for (std::size_t i = 0; i < data_.device_count; ++i)
            if (data_.devices[i] == device)
                return;
        if (data_.device_count == data_.devices.size())
            data_.devices_truncated = true;
        else
            data_.devices[data_.device_count++] = device;
    }

    void completed_block() noexcept {
        ++data_.completed_blocks;
    }

    void
    publish(DenoisingOutcome outcome, const MediaTransferCounters& counters) noexcept {
        if (!active_)
            return;
        active_ = false;
        data_.outcome = outcome;
        data_.wall_ns = clock_() - data_.started_ns;
        std::uint64_t accounted = 0;
        for (const auto& stage : data_.stages)
            accounted += stage.total_ns;
        data_.unaccounted_ns =
            data_.wall_ns > accounted ? data_.wall_ns - accounted : 0;
        auto delta = [&](Bytes current, Bytes previous) {
            if (current < previous) {
                data_.transfer_counters_reset = true;
                return Bytes{0};
            }
            return current - previous;
        };
        data_.transfers = {
            delta(counters.activation_upload, baseline_.activation_upload),
            delta(counters.activation_download, baseline_.activation_download),
            delta(counters.local_activation, baseline_.local_activation),
            delta(counters.input_upload, baseline_.input_upload),
            delta(counters.weight_upload, baseline_.weight_upload),
            delta(counters.output_download, baseline_.output_download),
            delta(counters.peer_activation, baseline_.peer_activation),
            delta(counters.control_download, baseline_.control_download)
        };
        try {
            sink_(data_);
        } catch (...) {
            // A diagnostic sink must not replace inference errors or cancel cleanup.
        }
    }

private:
    friend class DenoisingStageTimer;

    static std::uint64_t now() noexcept {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(
                   std::chrono::steady_clock::now().time_since_epoch()
        )
            .count();
    }

    void add(DenoisingStage stage, std::uint64_t elapsed) noexcept {
        auto& value = data_.stages[std::size_t(stage)];
        if (!value.calls)
            value.min_ns = value.first_ns = elapsed;
        ++value.calls;
        value.total_ns += elapsed;
        value.min_ns = std::min(value.min_ns, elapsed);
        value.max_ns = std::max(value.max_ns, elapsed);
        value.last_ns = elapsed;
    }

    const Sink& sink_;
    Clock clock_;
    DenoisingStepTiming data_;
    MediaTransferCounters baseline_;
    bool active_ = false;
};

// A null/inactive owner reads no clock and records nothing. No timer is placed
// inside a GEMM, tile or kernel; normal operator synchronization stays unchanged.
class DenoisingStageTimer {
public:
    DenoisingStageTimer(DenoisingTimings* owner, DenoisingStage stage) noexcept
        : owner_(owner && owner->active_ ? owner : nullptr), stage_(stage),
          started_(owner_ ? owner_->clock_() : 0) {
    }

    ~DenoisingStageTimer() {
        if (owner_)
            owner_->add(stage_, owner_->clock_() - started_);
    }

    DenoisingStageTimer(const DenoisingStageTimer&) = delete;
    DenoisingStageTimer& operator=(const DenoisingStageTimer&) = delete;

private:
    DenoisingTimings* owner_;
    DenoisingStage stage_;
    std::uint64_t started_;
};
}  // namespace kadan::image
