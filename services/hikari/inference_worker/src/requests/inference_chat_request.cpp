#include <string_view>

#include "kadan/inkling_variant.hpp"
#include "kadan/preparation.hpp"

namespace kadan::serving {
namespace {
constexpr std::size_t chat_image_max_bytes = 20ULL * 1024 * 1024;
constexpr std::size_t chat_image_max_encoded = 28ULL * 1024 * 1024;
constexpr std::string_view png_data_url_prefix = "data:image/png;base64,";
constexpr std::uint8_t png_magic[8] = {0x89, 'P', 'N', 'G', '\r', '\n', 0x1a, '\n'};

std::vector<std::uint8_t> decode_chat_png(std::string_view data_url) {
    preparation_check(
        data_url.size() <= chat_image_max_encoded
            && data_url.starts_with(png_data_url_prefix),
        "Chat images must be PNG data URLs up to 20 MiB",
        422
    );
    const auto decoded = base64_decode(
        data_url.substr(png_data_url_prefix.size()),
        chat_image_max_encoded - png_data_url_prefix.size()
    );
    preparation_check(
        decoded.size() >= sizeof(png_magic) && decoded.size() <= chat_image_max_bytes
            && std::equal(
                std::begin(png_magic),
                std::end(png_magic),
                reinterpret_cast<const std::uint8_t*>(decoded.data())
            ),
        "Invalid PNG image data",
        422
    );
    return std::vector<std::uint8_t>(
        reinterpret_cast<const std::uint8_t*>(decoded.data()),
        reinterpret_cast<const std::uint8_t*>(decoded.data()) + decoded.size()
    );
}

void prepare_llm_configuration(
    const std::optional<std::string>& requested_model,
    const std::optional<std::string>& requested_variant,
    bool context_supplied,
    std::optional<std::size_t> requested_context,
    InferenceRequest& request,
    ModelStore& store,
    const std::atomic_bool& cancel
) {
    std::string model_id = requested_model.value_or(store.selected_llm_id());
    if (requested_variant) {
        const auto variant = parse_inkling_variant(*requested_variant);
        preparation_check(
            variant.has_value(), "Weight variant must be 'base' or 'abliterated'.", 422
        );
        if (requested_model) {
            const auto model_var = inkling_model_variant(*requested_model);
            preparation_check(
                model_var.has_value() && *model_var == *variant,
                "Requested model does not match requested weight_variant.",
                422
            );
        } else {
            model_id = resolve_inkling_variant_model(model_id, *variant);
        }
    }
    request.model = std::move(model_id);
    preparation_check(
        store.entry(request.model).inference_available,
        "This checkpoint has no supported inference engine.",
        422
    );
    request.configuration.checkpoint = store.require_local(request.model, cancel);
    const auto config = read_document(request.configuration.checkpoint / "config.json");
    const auto text = config.value("text_config", config);
    const auto supported = text.at("max_position_embeddings").get<std::size_t>();
    auto configured = requested_context;
    if (request.operation != Operation::load || !context_supplied) {
        const auto saved = store.context(request.model);
        configured =
            saved.is_null() ? std::nullopt : std::optional(saved.get<std::size_t>());
    }
    const auto effective =
        configured.value_or(std::min(store.default_llm_context(), supported));
    preparation_check(
        supported > 0 && effective > 0 && effective <= supported,
        "Configured context exceeds the checkpoint limit.",
        422
    );
    request.context = ContextLimits{configured, effective, supported};
    request.configuration.context_limit = effective;
}
}  // namespace

void prepare_input(
    ChatRequest& input,
    InferenceRequest& request,
    ModelStore& store,
    const std::atomic_bool& cancel
) {
    prepare_llm_configuration(
        input.model,
        input.weight_variant,
        input.context_supplied,
        input.context_limit,
        request,
        store,
        cancel
    );
    preparation_check(
        request.operation == Operation::load || !input.messages.empty(),
        "Chat requires messages",
        422
    );
    for (auto& msg : input.messages) {
        msg.decoded_images.clear();
        msg.decoded_images.reserve(msg.images.size());
        for (const auto& url : msg.images)
            msg.decoded_images.push_back(decode_chat_png(url));
    }
}

void prepare_input(
    DecisionRequest& input,
    InferenceRequest& request,
    ModelStore& store,
    const std::atomic_bool& cancel
) {
    // Share the active Inkling 2 Mini checkpoint and context limit so the
    // resident ChatEngine serves decisions without reloading or evicting chat.
    prepare_llm_configuration(
        input.model, input.weight_variant, false, std::nullopt, request, store, cancel
    );
}

Json present_chat(const InferenceRequest& request, Json result) {
    result["model"] = request.model;
    return result;
}
}  // namespace kadan::serving
