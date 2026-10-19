#include "kadan/inference_trace.hpp"

#include <cassert>
#include <stdexcept>
#include <string>
#include <vector>
using namespace kadan::serving;

int main() {
    std::vector<bool> failed;
    TraceSink sink =
        [&](const char* phase, std::int64_t start, std::int64_t end, bool error) {
            assert(std::string(phase) == "load");
            assert(start > 0 && end >= start);
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
    TraceSink broken = [](const char*, std::int64_t, std::int64_t, bool) {
        throw std::runtime_error("export unavailable");
    };
    {
        InferenceTrace tracing(broken, "load");
    }
    TraceSink disabled;
    {
        InferenceTrace tracing(disabled, "load");
    }
}
