#include "kadan/preparation.hpp"

namespace kadan::serving {

void prepare_input(
    VisionRequest& input,
    InferenceRequest& request,
    ModelStore& store,
    const std::atomic_bool& cancel
) {
    preparation_check(
        input.model == "lfm3-vl-2b-bf16",
        "This worker serves only the pinned LFM3-VL BF16 model.",
        422
    );
    request.model = input.model;
    preparation_check(
        store.entry(request.model).inference_available,
        "This checkpoint has no supported inference engine.",
        422
    );
    request.configuration.checkpoint = store.require_local(request.model, cancel);
    request.configuration.context_limit = 128000;
}

void prepare_input(
    AudioCallRequest& input,
    InferenceRequest& request,
    ModelStore& store,
    const std::atomic_bool& cancel
) {
    preparation_check(
        input.model == "lfm3-audio-2b-bf16",
        "The audio worker only serves explicitly selected audio-call turns",
        422
    );
    if (input.mode == "transcribe") {
        preparation_check(
            input.audio.has_value() && input.messages.empty(),
            "Transcription needs one audio utterance and no history",
            422
        );
    } else if (input.mode == "tts") {
        preparation_check(
            input.script.has_value() && !input.script->empty()
                && input.script->find_first_not_of(" \t\r\n") != std::string::npos
                && input.script->size() <= 240,
            "TTS needs one bounded known script",
            422
        );
    } else if (input.mode == "turn") {
        preparation_check(
            input.audio.has_value() && !input.messages.empty()
                && input.messages.front().role == "system",
            "Audio call needs the session's spoken instructions",
            422
        );
    } else {
        preparation_check(
            !input.audio.has_value() && !input.messages.empty()
                && input.messages.front().role == "system",
            "Audio call needs text history",
            422
        );
    }
    request.model = input.model;
    preparation_check(
        store.entry(request.model).inference_available,
        "This checkpoint has no supported inference engine.",
        422
    );
    request.configuration.checkpoint = store.require_local(request.model, cancel);
    request.configuration.context_limit = 32768;
}

}  // namespace kadan::serving
