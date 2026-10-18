// The decisions worker: Laya on the CPU for structured answers.
#include "kadan/worker_runtime.hpp"

int main() {
    using namespace kadan::serving;
    return run_worker(
        {kadan::Workload::decision, false, [](WorkerEnvironment& environment) {
             auto& channel = environment.channel;
             return decision_engine(
                 environment.resources,
                 [&channel](const char* phase, std::size_t value) {
                     channel.progress(phase, value);
                 }
             );
         }}
    );
}
