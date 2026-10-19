#pragma once

#include <chrono>
#include <cstdint>
#include <exception>
#include <functional>

namespace kadan::serving {
using TraceSink = std::function<void(const char*, std::int64_t, std::int64_t, bool)>;

inline std::int64_t trace_unix_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::system_clock::now().time_since_epoch()
    )
        .count();
}

class InferenceTrace {
public:
    InferenceTrace(const TraceSink& sink, const char* phase)
        : sink_(sink), phase_(phase), start_(sink ? trace_unix_ns() : 0),
          exceptions_(std::uncaught_exceptions()) {
    }

    ~InferenceTrace() noexcept {
        if (sink_) {
            try {
                sink_(
                    phase_,
                    start_,
                    trace_unix_ns(),
                    std::uncaught_exceptions() > exceptions_
                );
            } catch (...) {
                // Observability must never alter completion or cancellation.
            }
        }
    }

    InferenceTrace(const InferenceTrace&) = delete;
    InferenceTrace& operator=(const InferenceTrace&) = delete;

private:
    const TraceSink& sink_;
    const char* phase_;
    std::int64_t start_;
    int exceptions_;
};
}  // namespace kadan::serving
