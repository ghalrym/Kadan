#include <iomanip>
#include <sstream>

#include "kadan/preparation.hpp"

namespace kadan::serving {
void prepare_input(
    SpeechRequest& input,
    InferenceRequest& request,
    ModelStore& store,
    const std::atomic_bool& cancel
) {
    request.model = input.model;
    request.configuration.checkpoint = store.require_local(request.model, cancel);
    input.tokenizer =
        configured_path("KADAN_TTS_TOKENIZER", store.root() / "tts/tokenizer.json");
    if (!fs::is_regular_file(input.tokenizer))
        export_speech_tokenizer(
            request.configuration.checkpoint, input.tokenizer.parent_path(), cancel
        );
}

Json present_speech(const InferenceRequest& request, const fs::path& output) {
    const auto& input = std::get<SpeechRequest>(request.input);
    const auto audio = artifact(output, request.workspace, 4 * 1024 * 1024);
    const auto info = wave(audio);
    std::ostringstream duration;
    duration << std::fixed << std::setprecision(1)
             << double(info.bytes) / info.alignment / info.rate << 's';
    return {
        {"voice", input.speaker},
        {"meta", "WAV"},
        {"script", input.script},
        {"time", duration.str()},
        {"audio_base64", base64_encode(audio)},
        {"mime_type", "audio/wav"}
    };
}
}  // namespace kadan::serving
