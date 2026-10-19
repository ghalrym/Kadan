#pragma once

#include <chrono>
#include <cstdint>
#include <exception>
#include <functional>
#include <optional>

#include "kadan/cuda_phase_metrics.hpp"

namespace kadan::serving {
// Fixed numeric details attached only to existing sampled text phase spans.
struct TextPhaseMetrics {
    std::uint64_t tokens = 0;
    double projection_seconds = 0;
    double attention_seconds = 0;
    double prepare_seconds = 0;
    double device_seconds = 0;
    double host_wait_seconds = 0;
    double recurrent_seconds = 0;
    std::uint64_t projections = 0;
    std::uint64_t projection_batches = 0;
    std::uint64_t streamed_bytes = 0;
    std::uint64_t host_expert_bytes = 0;
    std::uint64_t packed_bytes = 0;
    std::uint64_t device_bytes = 0;
    std::uint64_t device_slices = 0;
    std::uint64_t ram_bytes = 0;
    std::uint64_t ram_slices = 0;
    std::uint64_t packed_calls = 0;
    std::uint64_t host_calls = 0;
    std::uint64_t large_row_calls = 0;
    std::uint64_t unsupported_calls = 0;
    std::uint64_t rows = 0;
    std::uint64_t batch_calls = 0;
    std::uint64_t batch_rejections = 0;
    double large_row_seconds = 0;
    double unsupported_seconds = 0;
};

using TraceSink = std::function<void(
    const char*,
    std::int64_t,
    std::int64_t,
    bool,
    const TextPhaseMetrics*,
    const CudaPhaseMetrics*
)>;

inline std::int64_t trace_unix_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::system_clock::now().time_since_epoch()
    )
        .count();
}

class InferenceTrace {
public:
    InferenceTrace(
        const TraceSink& sink, const char* phase, std::shared_ptr<CudaPhase> shared = {}
    )
        : cuda_(shared ? shared : make_context(bool(sink))), sink_(sink), phase_(phase),
          start_(sink ? trace_unix_ns() : 0), exceptions_(std::uncaught_exceptions()) {
    }

    ~InferenceTrace() noexcept {
        if (sink_) {
            try {
                const auto cuda = cuda_.snapshot();
                sink_(
                    phase_,
                    start_,
                    trace_unix_ns(),
                    std::uncaught_exceptions() > exceptions_,
                    metrics_ ? &*metrics_ : nullptr,
                    cuda.observed ? &cuda : nullptr
                );
            } catch (...) {
                // Observability must never alter completion or cancellation.
            }
        }
    }

    void text_metrics(TextPhaseMetrics metrics) noexcept {
        if (sink_)
            metrics_ = metrics;
    }

    InferenceTrace(const InferenceTrace&) = delete;
    InferenceTrace& operator=(const InferenceTrace&) = delete;

private:
    static std::shared_ptr<CudaPhase> make_context(bool enabled) noexcept {
        if (enabled) {
            try {
                return std::make_shared<CudaPhase>();
            } catch (...) {
            }
        }
        return {};
    }

    CudaPhaseScope cuda_;
    const TraceSink& sink_;
    const char* phase_;
    std::int64_t start_;
    int exceptions_;
    std::optional<TextPhaseMetrics> metrics_;
};
}  // namespace kadan::serving
