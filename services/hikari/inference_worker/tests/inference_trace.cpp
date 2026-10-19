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
                         const TextPhaseMetrics* metrics) {
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
    TraceSink broken =
        [](const char*, std::int64_t, std::int64_t, bool, const TextPhaseMetrics*) {
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
                            const TextPhaseMetrics* metrics) {
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
}
