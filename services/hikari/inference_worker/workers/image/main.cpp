// The image worker: Qwen Image generation, batching compatible requests.
#include "kadan/image_edit.hpp"
#include "kadan/worker_runtime.hpp"

int main() {
    using namespace kadan::serving;
    return run_worker(
        {kadan::Workload::image,
         true,
         [](WorkerEnvironment& environment) {
             auto& channel = environment.channel;
             return image_engine(
                 environment.resources,
                 [&channel](const char* phase, std::size_t value) {
                     channel.progress(phase, value);
                 },
                 // Batched requests report their own progress.
                 [&channel](
                     const std::string& id, const char* phase, std::size_t value
                 ) {
                     channel.send(
                         {{"id", id},
                          {"state", "progress"},
                          {"phase", phase},
                          {"value", value}}
                     );
                 },
                 channel.available
             );
         },
         // Image jobs report their step count; store jobs have none.
         [](const InferenceRequest& request, Json& prepared) {
             const auto* image = std::get_if<ImageRequest>(&request.input);
             if (image)
                 prepared["total"] =
                     image->source && !image->reference
                         ? kadan::image::edit_steps(image->steps, image->strength)
                         : image->steps;
         }}
    );
}
