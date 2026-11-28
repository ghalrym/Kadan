#include "kadan/preparation.hpp"
#include "kadan/qwen3_asr.hpp"

namespace kadan::serving {
void prepare_input(
    TranscriptionRequest& input,
    InferenceRequest& request,
    ModelStore& store,
    const std::atomic_bool& cancel
) {
    preparation_check(
        !input.model || *input.model == "qwen3-asr-1.7b",
        "Hikari uses Qwen3-ASR 1.7B.",
        422
    );
    if (input.language && !input.language->empty()) {
        const auto trimmed = *input.language;
        if (trimmed != "auto" && trimmed != "Auto") {
            preparation_check(
                stt::normalize_language_hint(trimmed).has_value(),
                "Unsupported transcription language.",
                422
            );
        }
    }
    request.model = "qwen3-asr-1.7b";
    const auto checkpoint = store.require_local(request.model, cancel);
    const auto tokenizer =
        store.root() / "speech-tokenizer" / (request.model + "-" + store.entry(request.model).revision);
    if (!fs::exists(tokenizer))
        export_speech_tokenizer(checkpoint, tokenizer, cancel);
    preparation_check(
        fs::is_regular_file(tokenizer / "tokenizer.json"),
        "Transcription tokenizer assets are incomplete"
    );
    input.tokenizer = tokenizer;
    request.configuration.checkpoint = checkpoint;
    input.pcm = pcm_audio(input.audio);
    std::string().swap(input.audio);
}

Json present_transcription(
    const InferenceRequest& request,
    const std::string& text,
    const std::optional<std::string>& language
) {
    return {
        {"text", text},
        {"language", language ? Json(*language) : Json(nullptr)},
        {"model", request.model}
    };
}
}  // namespace kadan::serving
