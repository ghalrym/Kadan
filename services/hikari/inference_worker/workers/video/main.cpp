// The video worker: H3 video with audio, returned inline as MP4 bytes.
#include "kadan/worker_runtime.hpp"

int main() {
    using namespace kadan::serving;
    return run_worker(
        {kadan::Workload::video, true, [](WorkerEnvironment& environment) {
             auto& channel = environment.channel;
             return video_engine(
                 environment.resources,
                 [&channel](const char* phase, std::size_t value) {
                     channel.progress(phase, value);
                 }
             );
         }}
    );
}
