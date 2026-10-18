// The transcription worker: native Whisper for English audio.
#include "kadan/worker_runtime.hpp"

int main() {
    using namespace kadan::serving;
    return run_worker(
        {kadan::Workload::speech, true, [](WorkerEnvironment& environment) {
             auto& channel = environment.channel;
             return whisper_engine(
                 environment.resources,
                 [&channel](const char* phase, std::size_t value) {
                     channel.progress(phase, value);
                 }
             );
         }}
    );
}
