#include "kadan/preparation.hpp"

#include <cstdlib>

namespace kadan::serving {
fs::path configured_path(const char* name, const fs::path& fallback) {
    const auto* value = std::getenv(name);
    if (!value)
        return fallback;
    std::string configured(value);
    if (configured == "~" || configured.starts_with("~/")) {
        const auto* home = std::getenv("HOME");
        preparation_check(home != nullptr, "Home directory is unavailable", 503);
        return configured == "~" ? fs::path(home)
                                 : fs::path(home) / configured.substr(2);
    }
    return configured;
}

std::string
artifact(const fs::path& path, const fs::path& workspace, std::size_t maximum) {
    preparation_check(
        path.parent_path() == workspace && !fs::is_symlink(path)
            && fs::is_regular_file(path) && fs::file_size(path) > 0,
        "Inference worker returned an invalid artifact"
    );
    return read_bytes(path, maximum);
}

JobWorkspace::JobWorkspace(fs::path root) : root_(std::move(root)) {
    fs::create_directories(root_);
    // The worker's exclusive process lock is already held.
    for (const auto& entry : fs::directory_iterator(root_))
        fs::remove_all(entry.path());
}

fs::path JobWorkspace::create(const std::string& id) {
    auto workspace = root_ / id;
    preparation_check(fs::create_directory(workspace), "Worker workspace exists");
    return workspace;
}

void JobWorkspace::cleanup(const std::string& id) {
    fs::remove_all(root_ / id);
}

void prepare_task(
    InferenceRequest& request,
    ModelStore& store,
    JobWorkspace& files,
    const std::atomic_bool& cancel
) {
    preparation_cancel(cancel);
    request.workspace = files.create(request.id);
    std::visit(
        [&](auto& input) { prepare_input(input, request, store, cancel); },
        request.input
    );
}

void prepare_input(
    DownloadRequest& input,
    InferenceRequest& request,
    ModelStore& store,
    const std::atomic_bool& cancel
) {
    const auto& entry = store.entry(input.model);
    preparation_check(
        !entry.license_notice || input.license_acknowledged,
        "Read the model license notice and explicitly continue before downloading",
        422
    );
    request.model = input.model;
    request.configuration.checkpoint = store.download(input.model, cancel);
}

void prepare_input(
    WhisperSelectionRequest& input,
    InferenceRequest& request,
    ModelStore& store,
    const std::atomic_bool&
) {
    request.model = input.model;
    store.select_whisper(input.model);
}

void prepare_input(
    WhisperFormattingRequest& input,
    InferenceRequest&,
    ModelStore& store,
    const std::atomic_bool&
) {
    store.set_whisper_formatting(input.enabled);
}
}  // namespace kadan::serving
