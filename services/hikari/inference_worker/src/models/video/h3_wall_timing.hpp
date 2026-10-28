#pragma once
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <exception>
#include <limits>
#include <utility>

#include "kadan/media_residency.hpp"

namespace kadan::video::detail {
enum class H3WallStage {
    refiner,
    block,
    feed_forward,
    final
};
enum class H3WallPhase {
    describe,
    prepare,
    execute
};

struct H3WallDevice {
    int id = -1;
    Bytes arena_bytes = 0;
};

struct H3WallRecord {
    bool summary = false, failed = false, cancelled = false, plan_observed = false;
    H3WallStage stage = H3WallStage::block;
    H3WallPhase phase = H3WallPhase::describe;
    std::size_t index = 0, tokens = 0, rows = 0, queries = 0, keys = 0;
    Bytes weight_bytes = 0;
    std::array<H3WallDevice, 2> devices{};
    std::size_t device_count = 0, records = 0, dropped = 0;
    std::uint64_t wall_ns = 0;
    std::array<std::uint64_t, 3> phase_wall_ns{}, calls{};
};

// Wall time only. No device events, synchronization, payloads or dynamic storage.
class H3WallTiming {
public:
    using Sink = void (*)(const H3WallRecord&);
    using Clock = std::uint64_t (*)();
    static constexpr std::size_t max_records = 192;

    H3WallTiming(
        bool enabled, const std::atomic_bool& cancel, Sink sink, Clock clock = now
    ) noexcept
        : enabled_(enabled), cancel_(cancel), sink_(sink), clock_(clock),
          exceptions_(std::uncaught_exceptions()) {
    }

    ~H3WallTiming() noexcept {
        if (!enabled_)
            return;
        summary_.summary = true;
        summary_.failed = std::uncaught_exceptions() > exceptions_;
        summary_.cancelled = cancel_.load();
        emit(summary_);
    }

    template <class F>
    decltype(auto) measure(
        H3WallStage stage,
        std::size_t index,
        H3WallPhase phase,
        std::size_t tokens,
        F&& function,
        const MediaResidencyPlan* plan = nullptr
    ) {
        if (!enabled_)
            return std::forward<F>(function)();
        Scope scope(*this, stage, index, phase, tokens, plan);
        return std::forward<F>(function)();
    }

    H3WallTiming(const H3WallTiming&) = delete;
    H3WallTiming& operator=(const H3WallTiming&) = delete;

private:
    static std::uint64_t now() noexcept {
        return std::uint64_t(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now().time_since_epoch()
            )
                .count()
        );
    }

    static std::uint64_t add(std::uint64_t a, std::uint64_t b) noexcept {
        return a > std::numeric_limits<std::uint64_t>::max() - b
                   ? std::numeric_limits<std::uint64_t>::max()
                   : a + b;
    }

    bool read_clock(std::uint64_t& value) noexcept {
        try {
            value = clock_();
            return true;
        } catch (...) {
            return false;
        }
    }

    void emit(const H3WallRecord& record) noexcept {
        try {
            if (sink_)
                sink_(record);
        } catch (...) {
            ++summary_.dropped;
        }
    }

    class Scope {
    public:
        Scope(
            H3WallTiming& owner,
            H3WallStage stage,
            std::size_t index,
            H3WallPhase phase,
            std::size_t tokens,
            const MediaResidencyPlan* plan
        ) noexcept
            : owner_(owner), exceptions_(std::uncaught_exceptions()) {
            record_.stage = stage;
            record_.index = index;
            record_.phase = phase;
            record_.tokens = tokens;
            if (plan) {
                record_.plan_observed = true;
                record_.rows = plan->rows;
                record_.queries = plan->queries;
                record_.keys = plan->keys;
                for (const auto& weight : plan->weights)
                    record_.weight_bytes =
                        add(record_.weight_bytes, weight.stored_bytes);
                record_.device_count =
                    std::min(plan->devices.size(), record_.devices.size());
                for (std::size_t i = 0; i < record_.device_count; ++i)
                    record_.devices[i] = {
                        plan->devices[i].device, plan->devices[i].arena_bytes
                    };
            }
            valid_ = owner_.read_clock(start_);
        }

        ~Scope() noexcept {
            auto& summary = owner_.summary_;
            const auto phase = std::size_t(record_.phase);
            ++summary.calls[phase];
            std::uint64_t end = 0;
            if (!valid_ || !owner_.read_clock(end) || end < start_) {
                ++summary.dropped;
                return;
            }
            record_.wall_ns = end - start_;
            record_.failed = std::uncaught_exceptions() > exceptions_;
            record_.cancelled = owner_.cancel_.load();
            summary.phase_wall_ns[phase] =
                add(summary.phase_wall_ns[phase], record_.wall_ns);
            if (summary.records == max_records) {
                ++summary.dropped;
                return;
            }
            ++summary.records;
            owner_.emit(record_);
        }

    private:
        H3WallTiming& owner_;
        H3WallRecord record_;
        int exceptions_;
        std::uint64_t start_ = 0;
        bool valid_ = false;
    };

    bool enabled_;
    const std::atomic_bool& cancel_;
    Sink sink_;
    Clock clock_;
    int exceptions_;
    H3WallRecord summary_;
};
}  // namespace kadan::video::detail
