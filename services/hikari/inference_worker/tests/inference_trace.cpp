#include "kadan/inference_trace.hpp"

#include <cassert>
#include <stdexcept>
#include <string>
#include <vector>
using namespace kadan::serving;

int main() {
    std::vector<bool> failed;
    TraceSink sink = [&](const char* phase,
                         std::int64_t start,
                         std::int64_t end,
                         bool error,
                         const TextPhaseMetrics* metrics,
                         const CudaPhaseMetrics*) {
        assert(std::string(phase) == "load");
        assert(start > 0 && end >= start);
        assert(metrics == nullptr);
        failed.push_back(error);
    };
    {
        InferenceTrace tracing(sink, "load");
    }
    try {
        InferenceTrace tracing(sink, "load");
        throw std::runtime_error("private contents");
    } catch (const std::runtime_error&) {
    }
    assert(failed == std::vector<bool>({false, true}));
    TraceSink broken = [](const char*,
                          std::int64_t,
                          std::int64_t,
                          bool,
                          const TextPhaseMetrics*,
                          const CudaPhaseMetrics*) {
        throw std::runtime_error("export unavailable");
    };
    {
        InferenceTrace tracing(broken, "load");
    }
    TextPhaseMetrics observed;
    TraceSink sampled = [&](const char* phase,
                            std::int64_t,
                            std::int64_t,
                            bool,
                            const TextPhaseMetrics* metrics,
                            const CudaPhaseMetrics*) {
        assert(std::string(phase) == "prefill" && metrics);
        observed = *metrics;
    };
    {
        InferenceTrace tracing(sampled, "prefill");
        TextPhaseMetrics metrics;
        metrics.tokens = 9;
        metrics.large_row_calls = 2;
        metrics.large_row_seconds = .5;
        tracing.text_metrics(metrics);
    }
    assert(
        observed.tokens == 9 && observed.large_row_calls == 2
        && observed.large_row_seconds == .5
    );
    TraceSink disabled;
    {
        InferenceTrace tracing(disabled, "load");
        tracing.text_metrics(observed);
    }
    // Nested leaf phases own their work; aggregate parents do not duplicate it.
    std::vector<CudaPhaseMetrics> cuda_observed;
    TraceSink cuda_sink = [&](const char*,
                              std::int64_t,
                              std::int64_t,
                              bool,
                              const TextPhaseMetrics*,
                              const CudaPhaseMetrics* cuda) {
        if (cuda)
            cuda_observed.push_back(*cuda);
    };
    {
        InferenceTrace parent(cuda_sink, "execute");
        const auto parent_context = cuda_phase;
        {
            InferenceTrace leaf(cuda_sink, "prefill");
            assert(cuda_phase != parent_context);
            cuda_phase->metrics.observed = true;
            cuda_phase->metrics.kernels[0].calls = 3;
        }
        assert(cuda_phase == parent_context);
    }
    assert(cuda_observed.size() == 1 && cuda_observed[0].kernels[0].calls == 3);
    assert(!cuda_phase);
    // One shared GPU batch, reported explicitly as shared on each sampled span.
    {
        CudaPhaseScope batch(true, 2);
        InferenceTrace first(cuda_sink, "execute", batch.context());
        InferenceTrace second(cuda_sink, "execute", batch.context());
        cuda_phase->metrics.observed = true;
        cuda_phase->metrics.kernels[0].calls = 5;
    }
    assert(cuda_observed.size() == 3);
    for (std::size_t i = 1; i < 3; ++i) {
        assert(cuda_observed[i].shared_batch_size == 2);
        assert(cuda_observed[i].kernels[0].calls == 5);
    }
    assert(!cuda_phase);
}
