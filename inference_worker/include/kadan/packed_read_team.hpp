#pragma once
#include <algorithm>
#include <array>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <limits>
#include <mutex>
#include <pthread.h>
#include <span>
#include <system_error>
#include <thread>
#include <utility>

#include "kadan/cleanup_error.hpp"
#include "kadan/resources.hpp"

namespace kadan {
// One device owner reads the current packed tile. Seven persistent host-only
// workers and the caller consume disjoint whole rows; no future tile is queued.
class PackedReadTeam {
public:
    using Source = std::function<void(std::size_t, std::span<std::byte>)>;
    static constexpr std::size_t lanes = 8;
    static constexpr std::size_t parallel_threshold = 2 * 1024 * 1024;
    static constexpr Bytes metadata_bytes = 64 * 1024;

    static Bytes host_bytes() {
        pthread_attr_t attributes;
        checked(pthread_getattr_default_np(&attributes));
        std::size_t stack = 0, guard = 0;
        const auto stack_error = pthread_attr_getstacksize(&attributes, &stack);
        const auto guard_error = pthread_attr_getguardsize(&attributes, &guard);
        const auto destroy_error = pthread_attr_destroy(&attributes);
        checked(stack_error);
        checked(guard_error);
        checked(destroy_error);
        constexpr auto workers = lanes - 1;
        if (stack > (std::numeric_limits<Bytes>::max() - metadata_bytes) / workers
            || guard > (std::numeric_limits<Bytes>::max() - metadata_bytes) / workers
                           - stack)
            throw std::overflow_error("packed_read_stack_budget");
        return workers * (Bytes(stack) + guard) + metadata_bytes;
    }

    PackedReadTeam(Resources& resources, Workload workload) : resources_(resources) {
        auto footprint = resources_.snapshot().capacity;
        std::fill(footprint.begin(), footprint.end(), 0);
        footprint[0] = host_bytes();
        reservation_ = resources_.reserve(workload, std::move(footprint));
        try {
            for (std::size_t lane = 0; lane < threads_.size(); ++lane)
                threads_[lane] = std::thread([this, lane] { run(lane); });
            resources_.loaded(reservation_);
            resident_ = true;
            resources_.pin(reservation_);
            pinned_ = true;
        } catch (...) {
            const auto error = std::current_exception();
            rethrow_after_cleanup(error, [&] { close(); });
        }
    }

    PackedReadTeam(const PackedReadTeam&) = delete;
    PackedReadTeam& operator=(const PackedReadTeam&) = delete;

    ~PackedReadTeam() noexcept {
        try {
            close();
        } catch (...) {
            // Keep accounting when thread cleanup cannot be acknowledged.
        }
    }

    void read(
        const Source& source,
        std::size_t first_row,
        std::size_t row_bytes,
        std::span<std::byte> destination
    ) {
        if (!source || !row_bytes || destination.empty()
            || destination.size() % row_bytes)
            throw std::invalid_argument("packed_read_rows");
        const auto rows = destination.size() / row_bytes;
        if (rows > std::numeric_limits<std::size_t>::max() - first_row)
            throw std::overflow_error("packed_read_row_range");
        const bool parallel = destination.size() >= parallel_threshold && rows >= lanes;
        {
            std::lock_guard lock(mutex_);
            if (closing_ || busy_)
                throw std::runtime_error("packed_read_team_unavailable");
            busy_ = true;
            errors_.fill(nullptr);
            if (parallel) {
                source_ = &source;
                first_row_ = first_row;
                row_bytes_ = row_bytes;
                destination_ = destination;
                remaining_ = threads_.size();
                ++round_;
            }
        }
        if (parallel)
            ready_.notify_all();
        std::exception_ptr caller_error;
        try {
            if (parallel)
                read_region(source, first_row, row_bytes, destination, lanes - 1);
            else
                source(first_row, destination);
        } catch (...) {
            caller_error = std::current_exception();
        }
        std::unique_lock lock(mutex_);
        errors_[lanes - 1] = caller_error;
        if (parallel)
            done_.wait(lock, [&] { return remaining_ == 0; });
        source_ = nullptr;
        destination_ = {};
        busy_ = false;
        std::exception_ptr error;
        for (const auto& failure : errors_)
            if (failure && !error)
                error = failure;
        lock.unlock();
        done_.notify_all();
        if (error)
            std::rethrow_exception(error);
    }

    // Called by the same owner after read() returns. Joining precedes release of
    // both the staging buffer and the stack/control reservation.
    void close() {
        {
            std::unique_lock lock(mutex_);
            closing_ = true;
            done_.wait(lock, [&] { return !busy_; });
            stopping_ = true;
        }
        ready_.notify_all();
        for (auto& thread : threads_)
            if (thread.joinable())
                thread.join();
        if (reservation_) {
            if (pinned_) {
                resources_.unpin(reservation_);
                pinned_ = false;
            }
            if (resident_) {
                resources_.begin_eviction(reservation_);
                resident_ = false;
            }
            resources_.released(reservation_);
            reservation_ = 0;
        }
    }

private:
    static void checked(int result) {
        if (result)
            throw std::system_error(
                result, std::generic_category(), "packed_read_team"
            );
    }

    static std::size_t boundary(std::size_t rows, std::size_t lane) {
        return rows / lanes * lane + rows % lanes * lane / lanes;
    }

    static void read_region(
        const Source& source,
        std::size_t first_row,
        std::size_t row_bytes,
        std::span<std::byte> destination,
        std::size_t lane
    ) {
        const auto rows = destination.size() / row_bytes;
        const auto first = boundary(rows, lane), end = boundary(rows, lane + 1);
        source(
            first_row + first,
            destination.subspan(first * row_bytes, (end - first) * row_bytes)
        );
    }

    void run(std::size_t lane) noexcept {
        std::uint64_t seen = 0;
        for (;;) {
            std::unique_lock lock(mutex_);
            ready_.wait(lock, [&] { return stopping_ || round_ != seen; });
            if (stopping_)
                return;
            seen = round_;
            const auto* source = source_;
            const auto first = first_row_, row_bytes = row_bytes_;
            const auto destination = destination_;
            lock.unlock();
            std::exception_ptr error;
            try {
                read_region(*source, first, row_bytes, destination, lane);
            } catch (...) {
                error = std::current_exception();
            }
            lock.lock();
            errors_[lane] = error;
            --remaining_;
            lock.unlock();
            done_.notify_all();
        }
    }

    Resources& resources_;
    Handle reservation_ = 0;
    bool resident_ = false, pinned_ = false;
    std::mutex mutex_;
    std::condition_variable ready_, done_;
    std::array<std::thread, lanes - 1> threads_;
    std::array<std::exception_ptr, lanes> errors_{};
    const Source* source_ = nullptr;
    std::span<std::byte> destination_;
    std::size_t first_row_ = 0, row_bytes_ = 0, remaining_ = 0;
    std::uint64_t round_ = 0;
    bool busy_ = false, closing_ = false, stopping_ = false;
};
}  // namespace kadan
