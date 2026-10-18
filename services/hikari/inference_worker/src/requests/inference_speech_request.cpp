#include <algorithm>
#include <iomanip>
#include <sstream>

#include "kadan/drama_prompt.hpp"
#include "kadan/preparation.hpp"

namespace kadan::serving {
void prepare_input(
    SpeechRequest& input,
    InferenceRequest& request,
    ModelStore& store,
    const std::atomic_bool& cancel
) {
    request.model = input.model;
    preparation_check(
        !input.expected_tier || store.tier() == input.expected_tier,
        "The active hardware tier changed; restart the conversation.",
        409
    );
    const bool drama = input.model == "drama-3" || input.model == "drama-3-int4";
    preparation_check(
        !input.expected_tier || drama
            || store.hardware_tier(*input.expected_tier).models.at("speech")
                   == std::optional<std::string>(request.model),
        "Conversation speech model does not match the active tier.",
        409
    );
    preparation_check(
        input.model != "drama-3-int4",
        "Drama 3 INT4 native packed-weight decoding is not supported.",
        422
    );
    const auto& entry = store.entry(request.model);
    preparation_check(
        !(input.require_installed || drama) || store.complete(entry),
        "Install the speech checkpoint in Settings before starting a conversation.",
        409
    );
    if (drama) {
        preparation_check(
            input.stream, "Drama 3 requires streamed speech output.", 422
        );
        preparation_check(
            (input.speaker.empty() || input.speaker == "default")
                && input.instruction.empty() && input.language == "Auto",
            "Drama 3 currently supports the default voice without language or "
            "instruction overrides.",
            422
        );
        try {
            (void)drama::prompt_segments(input.script);
        } catch (const std::invalid_argument& error) {
            throw RequestError(error.what(), 422);
        }
        // Never acquire here: conversation requests must not start downloads.
        request.configuration.checkpoint =
            store.root() / (entry.id + "-" + entry.revision);
        input.tokenizer = request.configuration.checkpoint / "tokenizer.json";
        const auto codec = std::find_if(
            entry.manifest.begin(), entry.manifest.end(), [](const auto& file) {
                return file.name == "codec.pth";
            }
        );
        preparation_check(
            codec != entry.manifest.end(), "Drama codec manifest missing"
        );
        export_drama_codec(
            request.configuration.checkpoint / "codec.pth",
            codec->digest,
            request.configuration.checkpoint / "codec",
            cancel
        );
        return;
    }
    request.configuration.checkpoint = store.acquire(request.model, cancel);
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
