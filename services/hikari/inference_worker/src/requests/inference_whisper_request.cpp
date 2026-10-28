#include <fstream>

#include "kadan/preparation.hpp"

namespace kadan::serving {
void prepare_input(
    TranscriptionRequest& input,
    InferenceRequest& request,
    ModelStore& store,
    const std::atomic_bool& cancel
) {
    if (!input.formatting.has_value())
        input.formatting = store.whisper_formatting();
    request.model =
        input.model ? store.whisper_model(*input.model) : store.whisper_selected();
    preparation_check(
        !input.language || *input.language == "en" || *input.language == "english"
            || *input.language == "English",
        "The Whisper checkpoint supports English.",
        422
    );
    const auto root = configured_path(
        "KADAN_WHISPER_MODEL_ROOT", store.root() / "whisper" / request.model
    );
    input.assets = configured_path("KADAN_WHISPER_ASSET_ROOT", root / "assets");
    if (!fs::exists(root)) {
        const auto& entry = store.entry("whisper-" + request.model);
        const auto source = store.acquire("whisper-" + request.model, cancel);
        export_whisper(source / (request.model + ".pt"), entry.revision, root, cancel);
    }
    if (!fs::exists(input.assets)) {
        std::ifstream dimensions(root / "dimensions.txt");
        std::size_t value = 0, vocabulary = 0;
        for (int i = 0; i < 6; ++i) {
            dimensions >> value;
            if (i == 5)
                vocabulary = value;
        }
        preparation_check(bool(dimensions), "Whisper dimensions are incomplete");
        export_whisper_assets(
            vocabulary, store.data() / "assets", input.assets, cancel
        );
    }
    for (const auto& file :
         {root / "model.safetensors",
          root / "dimensions.txt",
          input.assets / "english.tokens"})
        preparation_check(fs::is_regular_file(file), "Whisper assets are incomplete");
    request.configuration.checkpoint = root;
    input.pcm = pcm_audio(input.audio);
    std::string().swap(input.audio);
}

Json present_transcription(const InferenceRequest& request, const std::string& text) {
    const auto& input = std::get<TranscriptionRequest>(request.input);
    return {
        {"text", text},
        {"raw_text", text},
        {"language", "en"},
        {"model", request.model},
        {"formatting_status", input.formatting.value() ? "requested" : "disabled"},
        {"formatting_model", nullptr}
    };
}
}  // namespace kadan::serving
