// The speech synthesis worker: Qwen CustomVoice.
#include "kadan/worker_runtime.hpp"

int main() {
    using namespace kadan::serving;
    return run_worker({kadan::Workload::tts, true, [](WorkerEnvironment& environment) {
                           auto& channel = environment.channel;
                           return speech_engine(
                               environment.resources,
                               [&channel](const char* phase, std::size_t value) {
                                   channel.progress(phase, value);
                               }
                           );
                       }});
}
