#include "kadan/preparation.hpp"

namespace kadan::serving {
void prepare_input(
    ChatRequest& input,
    InferenceRequest& request,
    ModelStore& store,
    const std::atomic_bool& cancel
) {
    request.model = input.model ? *input.model : store.selected();
    preparation_check(
        store.entry(request.model).inference_available,
        "This checkpoint has no supported inference engine.",
        422
    );
    request.configuration.checkpoint = store.acquire(request.model, cancel);
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
    const auto& input = std::get<ChatRequest>(request.input);
    result["model"] = request.model;
    result["cache"] = nullptr;
    if (input.conversation_id)
        result["cache"] = {
            {"hit", false},
            {"reused_tokens", 0},
            {"stored_tokens", 0},
            {"host_bytes", 0},
            {"device_bytes", Json::object()},
            {"reason", "native_prefix_reuse_unavailable"},
            {"retention_reason", "disabled"},
            {"limit_bytes", 0}
        };
    return result;
}
}  // namespace kadan::serving
