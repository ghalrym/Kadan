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

enum class H3PreparePhase {
    admission,
    staging_allocation,
    checkpoint_read,
    validation,
    upload
};
enum class H3PrepareDtype {
    i8,
    bf16,
    fp32,
    other
};
enum class H3PrepareCount {
    started_slices,
    uploaded_slices,
    planned_bytes,
    read_bytes,
    validated_bytes,
    uploaded_bytes
};

inline std::uint64_t h3_wall_add(std::uint64_t a, std::uint64_t b) noexcept {
    return a > std::numeric_limits<std::uint64_t>::max() - b
               ? std::numeric_limits<std::uint64_t>::max()
               : a + b;
}

struct H3PrepareRecord {
    std::array<std::uint64_t, 5> wall_ns{}, calls{};
    std::array<std::array<std::uint64_t, 6>, 4> dtype{};
    std::uint64_t dropped = 0;

    void merge(const H3PrepareRecord& other) noexcept {
        for (std::size_t i = 0; i < wall_ns.size(); ++i) {
            wall_ns[i] = h3_wall_add(wall_ns[i], other.wall_ns[i]);
            calls[i] = h3_wall_add(calls[i], other.calls[i]);
        }
        for (std::size_t i = 0; i < dtype.size(); ++i)
            for (std::size_t j = 0; j < dtype[i].size(); ++j)
                dtype[i][j] = h3_wall_add(dtype[i][j], other.dtype[i][j]);
        dropped = h3_wall_add(dropped, other.dropped);
    }
};

// Collect inside a single outer prepare observation; never emits slice records.
class H3PrepareTiming {
public:
    using Clock = std::uint64_t (*)();

    H3PrepareTiming(H3PrepareRecord& record, Clock clock) noexcept
        : record_(record), clock_(clock) {
    }

    ~H3PrepareTiming() noexcept {
        pause();
    }

    void phase(H3PreparePhase phase) noexcept {
        pause();
        phase_ = std::size_t(phase);
        record_.calls[phase_] = h3_wall_add(record_.calls[phase_], 1);
        active_ = true;
        valid_ = read_clock(start_);
    }

    void pause() noexcept {
        if (!active_)
            return;
        active_ = false;
        std::uint64_t end = 0;
        if (!valid_ || !read_clock(end) || end < start_) {
            record_.dropped = h3_wall_add(record_.dropped, 1);
            return;
        }
        record_.wall_ns[phase_] = h3_wall_add(record_.wall_ns[phase_], end - start_);
    }

    void
    count(H3PrepareDtype dtype, H3PrepareCount counter, std::uint64_t value) noexcept {
        auto& target = record_.dtype[std::size_t(dtype)][std::size_t(counter)];
        target = h3_wall_add(target, value);
    }

    H3PrepareTiming(const H3PrepareTiming&) = delete;
    H3PrepareTiming& operator=(const H3PrepareTiming&) = delete;

private:
    bool read_clock(std::uint64_t& value) noexcept {
        try {
            value = clock_();
            return true;
        } catch (...) {
            return false;
        }
    }

    H3PrepareRecord& record_;
    Clock clock_;
    std::size_t phase_ = 0;
    std::uint64_t start_ = 0;
    bool active_ = false, valid_ = false;
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
    bool prepare_observed = false;
    H3PrepareRecord prepare;
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
        Scope scope(*this, stage, index, phase, tokens, plan, nullptr);
        return std::forward<F>(function)();
    }

    template <class F>
    decltype(auto) measure_prepare(
        H3WallStage stage, std::size_t index, std::size_t tokens, F&& function
    ) {
        if (!enabled_)
            return std::forward<F>(function)(nullptr);
        H3PrepareRecord nested;
        Scope scope(
            *this, stage, index, H3WallPhase::prepare, tokens, nullptr, &nested
        );
        H3PrepareTiming timing(nested, clock_);
        return std::forward<F>(function)(&timing);
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
        return h3_wall_add(a, b);
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
            const MediaResidencyPlan* plan,
            const H3PrepareRecord* nested
        ) noexcept
            : owner_(owner), nested_(nested), exceptions_(std::uncaught_exceptions()) {
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
            if (nested_) {
                record_.prepare_observed = summary.prepare_observed = true;
                record_.prepare = *nested_;
                summary.prepare.merge(*nested_);
            }
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
        const H3PrepareRecord* nested_;
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
