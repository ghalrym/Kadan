#include "kadan/preparation.hpp"

namespace kadan::serving {
void prepare_input(
    ChatRequest& input,
    InferenceRequest& request,
    ModelStore& store,
    const std::atomic_bool& cancel
) {
    preparation_check(
        input.model.has_value(),
        "Native language requests require an explicit checkpoint.",
        422
    );
    request.model = *input.model;
    preparation_check(
        store.entry(request.model).inference_available,
        "This checkpoint has no supported inference engine.",
        422
    );
    request.configuration.checkpoint = store.require_local(request.model, cancel);
    const auto config = read_document(request.configuration.checkpoint / "config.json");
    const auto text = config.value("text_config", config);
    const auto supported = text.at("max_position_embeddings").get<std::size_t>();
    auto configured = input.context_limit;
    if (request.operation != Operation::load || !input.context_supplied) {
        const auto saved = store.context(request.model);
        configured =
            saved.is_null() ? std::nullopt : std::optional(saved.get<std::size_t>());
    }
    const auto effective = configured.value_or(supported);
    preparation_check(
        supported > 0 && effective > 0 && effective <= supported,
        "Configured context exceeds the checkpoint limit.",
        422
    );
    request.context = ContextLimits{configured, effective, supported};
    request.configuration.context_limit = effective;
    preparation_check(
        request.operation == Operation::load || !input.messages.empty(),
        "Chat requires messages",
        422
    );
}

Json present_chat(const InferenceRequest& request, Json result) {
    result["model"] = request.model;
    return result;
}
}  // namespace kadan::serving
