// The transcript formatting worker: S1-mini on the CPU.
#include "kadan/worker_runtime.hpp"

int main() {
    using namespace kadan::serving;
    return run_worker(
        {kadan::Workload::formatting, false, [](WorkerEnvironment& environment) {
             auto& channel = environment.channel;
             return transcription_formatting_engine(
                 environment.resources,
                 [&channel](const char* phase, std::size_t value) {
                     channel.progress(phase, value);
                 }
             );
         }}
    );
}
