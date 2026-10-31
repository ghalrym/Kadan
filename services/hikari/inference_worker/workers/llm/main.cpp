// The language worker: explicit native GGUF mixture-of-experts checkpoints for chat.
#include "kadan/worker_runtime.hpp"

int main() {
    using namespace kadan::serving;
    return run_worker({kadan::Workload::llm, true, [](WorkerEnvironment& environment) {
                           auto& channel = environment.channel;
                           return chat_engine(
                               environment.resources,
                               [&channel](const char* phase, std::size_t value) {
                                   channel.progress(phase, value);
                               },
                               [&channel](const std::string& text) {
                                   channel.content(text);
                               }
                           );
                       }});
}
