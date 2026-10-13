#include <cstdlib>

#include "kadan/preparation.hpp"

namespace kadan::serving {
void prepare_input(
    VideoRequest& input,
    InferenceRequest& request,
    ModelStore& store,
    const std::atomic_bool& cancel
) {
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
    const auto root = configured_path(
        "KADAN_VIDEO_DIR",
        fs::path(std::getenv("HOME") ? std::getenv("HOME") : "/tmp")
            / ".local/share/kadan/videos"
    );
    fs::create_directories(root);
    const auto staging = root / ("." + request.id + ".partial.mp4");
    try {
        fs::copy_file(source, staging, fs::copy_options::overwrite_existing);
        fs::rename(staging, root / (request.id + ".mp4"));
    } catch (...) {
        fs::remove(staging);
        throw;
    }
    return {
        {"status", "Done"},
        {"progress", 100},
        {"progress_text", "Complete"},
        {"output_url", "/v1/videos/" + request.id + "/content"}
    };
}
}  // namespace kadan::serving
