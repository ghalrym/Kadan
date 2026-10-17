#include <array>
#include <cmath>
#include <limits>
#include <random>

#include "kadan/image_edit.hpp"
#include "kadan/preparation.hpp"

namespace kadan::serving {
void prepare_input(
    ImageRequest& input,
    InferenceRequest& request,
    ModelStore& store,
    const std::atomic_bool& cancel
) {
    preparation_check(
        input.count == 1 || input.count == 2 || input.count == 4,
        "Image count must be 1, 2, or 4",
        422
    );
    preparation_check(
        input.steps >= 2 && input.steps <= 100, "Invalid image steps", 422
    );
    preparation_check(
        std::isfinite(input.strength) && input.strength >= 0 && input.strength <= 1,
        "Image strength must be between 0 and 1",
        422
    );
    preparation_check(
        !input.mask || input.image, "Image mask requires a source image", 422
    );
    preparation_check(
        !input.reference || (input.image && !input.mask),
        "Reference editing requires an image without a mask",
        422
    );
    if (input.image) {
        input.source = prepare_image_upload(*input.image);
        input.image.reset();
        input.source_width = input.source->width;
        input.source_height = input.source->height;
        preparation_check(
            !input.reference || input.source_width * input.source_height >= 65536,
            "Reference image must contain at least 65,536 pixels",
            422
        );
        if (!input.reference || !input.width) {
            input.width = input.source_width;
            input.height = input.source_height;
        }
        preparation_check(
            input.width >= 32 && input.height >= 32 && input.width <= 3072
                && input.height <= 3072 && input.width % 32 == 0
                && input.height % 32 == 0
                && std::size_t(input.width) * input.height <= 4608 * 1024,
            "Invalid image output dimensions",
            422
        );
        if (input.mask) {
            input.edit_mask = prepare_image_upload(*input.mask);
            input.mask.reset();
            preparation_check(
                input.edit_mask->width == input.width
                    && input.edit_mask->height == input.height,
                "Image mask dimensions must match the source image",
                422
            );
        }
        const double ratio = double(input.width) / input.height;
        double distance = std::numeric_limits<double>::infinity();
        for (const auto& [name, candidate] : std::array{
                 std::pair{"1:1", 1.0},
                 std::pair{"4:3", 4.0 / 3.0},
                 std::pair{"3:4", 3.0 / 4.0},
                 std::pair{"16:9", 16.0 / 9.0}
             }) {
            const double difference = std::abs(std::log(ratio / candidate));
            if (difference < distance) {
                input.aspect = name;
                distance = difference;
            }
        }
    }
    // Every tier generates images with Turbo, so no tier choice is needed.
    request.model = input.model.value_or("qwen-image-2.1-turbo");
    if (request.model != "qwen-image-2.1-turbo")
        throw RequestError("Unsupported image model", 422);
    // The pinned Turbo grid determines updates, even when steps was supplied.
    input.steps = 8;
    if (!input.source) {
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
    }
    request.configuration.checkpoint = store.acquire(request.model, cancel);
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
          {"mode", input.source ? "Edit" : "Generate"},
          {"prompt", input.prompt},
          {"aspect", label},
          {"seeds", seeds},
          {"meta",
           std::to_string(input.width) + "×" + std::to_string(input.height) + " · "
               + std::to_string(
                   input.source && !input.reference
                       ? image::edit_steps(input.steps, input.strength)
                       : input.steps
               )
               + " steps"},
          {"images_base64", images},
          {"mime_type", "image/png"}}}
    };
}
}  // namespace kadan::serving
