#include <random>

#include "kadan/preparation.hpp"

namespace kadan::serving {
void prepare_input(
    ImageRequest& input,
    InferenceRequest& request,
    ModelStore& store,
    const std::atomic_bool& cancel
) {
    if (input.model != "qwen-image-2.1" && input.model != "qwen-image-2.1-turbo")
        throw RequestError("Unsupported image model", 422);
    request.model = input.model;
    // The pinned Turbo grid determines updates, even when steps was supplied.
    if (input.model == "qwen-image-2.1-turbo")
        input.steps = 8;
    request.configuration.checkpoint = store.acquire(request.model, cancel);
    if (input.aspect == "1:1") {
        input.width = 2048;
        input.height = 2048;
    } else if (input.aspect == "4:3") {
        input.width = 2400;
        input.height = 1792;
    } else if (input.aspect == "3:4") {
        input.width = 1792;
        input.height = 2400;
    } else if (input.aspect == "16:9") {
        input.width = 2752;
        input.height = 1536;
    } else
        throw RequestError("Unsupported image aspect", 422);
    if (!input.seed)
        input.seed = std::random_device{}();
}

Json present_image(
    const InferenceRequest& request, const std::vector<fs::path>& paths
) {
    const auto& input = std::get<ImageRequest>(request.input);
    preparation_check(paths.size() == input.count, "Invalid image output count");
    Json images = Json::array(), seeds = Json::array();
    for (const auto& path : paths)
        images.push_back(base64_encode(artifact(
            path,
            request.workspace,
            std::size_t(input.width) * input.height * 4 + 1024 * 1024
        )));
    for (unsigned i = 0; i < input.count; ++i)
        seeds.push_back(*input.seed + i);
    const std::string label = input.aspect == "1:1"   ? "square"
                              : input.aspect == "4:3" ? "landscape"
                              : input.aspect == "3:4" ? "portrait"
                                                      : "wide";
    return {
        {"image",
         {{"id", request.id},
          {"mode", "Generate"},
          {"prompt", input.prompt},
          {"aspect", label},
          {"seeds", seeds},
          {"meta",
           std::to_string(input.width) + "×" + std::to_string(input.height) + " · "
               + std::to_string(input.steps) + " steps · native CUDA"},
          {"images_base64", images},
          {"mime_type", "image/png"}}}
    };
}
}  // namespace kadan::serving
