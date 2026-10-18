#include <cstdlib>

#include "kadan/preparation.hpp"

namespace kadan::serving {
void prepare_input(
    VideoRequest& input,
    InferenceRequest& request,
    ModelStore& store,
    const std::atomic_bool& cancel
) {
    if (!input.duration_supplied || !input.resolution_supplied) {
        const auto tier = store.tier();
        preparation_check(
            tier.has_value(),
            "Choose a hardware tier in Settings before submitting a video.",
            422
        );
        const auto& defaults = store.hardware_tier(*tier).video_defaults;
        if (!input.duration_supplied)
            input.duration = defaults.duration;
        if (!input.resolution_supplied)
            input.short_edge = defaults.resolution == "480p" ? 480 : 768;
    }
    request.model = input.model;
    const auto checkpoint = store.acquire(request.model, cancel);
    request.configuration.checkpoint = checkpoint;
    input.paths = {
        (checkpoint / "FL2VA/tokenizer/tokenizer.json").string(),
        (checkpoint / "FL2VA/text_encoder/model.safetensors").string(),
        (checkpoint / "FL2VA/transformer/model.safetensors").string(),
        (checkpoint / "loras/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors")
            .string(),
        (checkpoint / "FL2VA/video_vae/model.safetensors").string(),
        (checkpoint / "FL2VA/audio_vae/model.safetensors").string()
    };
}

Json present_video(const InferenceRequest& request, const fs::path& source) {
    preparation_check(
        source.parent_path() == request.workspace && !fs::is_symlink(source)
            && fs::is_regular_file(source) && fs::file_size(source) > 0,
        "Invalid video output"
    );
    return {
        {"status", "Done"},
        {"progress", 100},
        {"progress_text", "Complete"},
        // The API may run on another machine, so the file travels in the result.
        {"video", base64_encode(read_bytes(source, 90 * 1024 * 1024))}
    };
}
}  // namespace kadan::serving
