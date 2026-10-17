#pragma once

#include <atomic>
#include <filesystem>
#include <functional>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "kadan/inference_owner.hpp"

namespace kadan::serving {
using Json = nlohmann::json;
namespace fs = std::filesystem;

void preparation_check(bool condition, const std::string& message, int status = 502);
void preparation_cancel(const std::atomic_bool& cancel);
Json read_document(const fs::path& path, std::size_t maximum = 32 * 1024 * 1024);
void write_document(const fs::path& path, const Json& value);
std::string read_bytes(const fs::path& path, std::size_t maximum);
std::string base64_encode(std::string_view bytes);
std::string
base64_decode(std::string_view encoded, std::size_t maximum = 3 * 1024 * 1024);
std::string file_digest(const fs::path& path, const std::atomic_bool& cancel);
fs::path worker_data_directory();
// Validate the GGUF architecture and derive config.json and tokenizer.json
// from the first file's metadata before any remaining split is downloaded.
void prepare_gguf_assets(
    const fs::path&, const std::string& first_file, const std::atomic_bool& cancel
);

struct ManifestFile {
    std::string name;
    std::uint64_t size;
    std::string digest, algorithm, url;

    static ManifestFile parse(const Json& value);
    void validate() const;
    bool operator==(const ManifestFile&) const = default;
};

void to_json(Json& value, const ManifestFile& file);
std::vector<ManifestFile> parse_manifest(const Json& value);

struct CatalogEntry {
    std::string id, repo_id, revision, kind, layout;
    std::string license;
    std::uint64_t estimated_bytes = 0;
    std::optional<std::string> display_name, license_url, license_notice;
    bool inference_available = false;
    std::string asset_url, asset_name;
    std::vector<std::string> required_files, component_paths, weight_paths;
    std::vector<ManifestFile> manifest;

    static CatalogEntry parse(const Json& value);
};

struct VideoDefaults {
    unsigned duration;
    std::string resolution;
};

// A hardware tier fixes one checkpoint per modality (null when unassigned).
struct HardwareTier {
    std::string id, name;
    unsigned vram_gb, ram_gb;
    VideoDefaults video_defaults;
    std::map<std::string, std::optional<std::string>> models;

    static HardwareTier parse(const Json& value);
};

class ModelStore {
public:
    using Fetch = std::function<std::int64_t(
        const std::string&,
        const std::function<void(std::string_view)>&,
        const std::atomic_bool&,
        bool,
        std::uint64_t
    )>;
    using DownloadProgress =
        std::function<void(const std::string&, std::uint64_t, std::uint64_t)>;
    ModelStore(
        fs::path root, fs::path data, Fetch fetch = {}, DownloadProgress progress = {}
    );
    const CatalogEntry& entry(const std::string& model) const;
    fs::path acquire(const std::string& model, const std::atomic_bool& cancel);
    // The language model of the selected hardware tier.
    std::string selected() const;
    std::optional<std::string> tier() const;
    const HardwareTier& hardware_tier(const std::string& id) const;
    void select_tier(const std::string& id);
    std::string whisper_selected() const;
    std::string whisper_model(const std::string& name) const;
    Json context(const std::string& model) const;
    void set_context(const std::string& model, const Json& value);
    void select_whisper(const std::string& model);
    bool whisper_formatting() const;
    void set_whisper_formatting(bool enabled);
    Json status() const;
    Json architecture_context(const std::string& model) const;

    const fs::path& root() const {
        return root_;
    }

    const fs::path& data() const {
        return data_;
    }

    bool complete(const CatalogEntry& entry) const;
    bool allowed(const std::string& name, const CatalogEntry& entry) const;
    void validate_files(
        const CatalogEntry& entry, const std::vector<ManifestFile>& files
    ) const;

private:
    std::vector<ManifestFile>
    manifest(const CatalogEntry& entry, const std::atomic_bool& cancel) const;
    fs::path root_, data_;
    std::vector<CatalogEntry> catalog_;
    std::vector<HardwareTier> tiers_;
    std::vector<std::string> assets_;
    Fetch fetch_;
    DownloadProgress progress_;
};

// Converts checkpoint containers and tokenizer metadata only; never runs a model.
void export_whisper(
    const fs::path& source,
    const std::string& expected,
    const fs::path& destination,
    const std::atomic_bool& cancel
);
void export_whisper_assets(
    std::size_t vocabulary,
    const fs::path& data,
    const fs::path& destination,
    const std::atomic_bool& cancel
);
void export_speech_tokenizer(
    const fs::path& checkpoint,
    const fs::path& destination,
    const std::atomic_bool& cancel
);

struct Wave {
    std::size_t offset = 0, bytes = 0;
    unsigned channels = 0, rate = 0, bits = 0, alignment = 0;
};

Wave wave(std::string_view bytes);
std::vector<float> pcm_audio(std::string_view audio);
fs::path configured_path(const char* name, const fs::path& fallback);
std::string
artifact(const fs::path& path, const fs::path& workspace, std::size_t maximum);

class JobWorkspace {
public:
    explicit JobWorkspace(fs::path root);
    fs::path create(const std::string& id);
    void cleanup(const std::string& id);

private:
    fs::path root_;
};

void prepare_task(
    InferenceRequest&, ModelStore&, JobWorkspace&, const std::atomic_bool&
);
void prepare_input(
    ChatRequest&, InferenceRequest&, ModelStore&, const std::atomic_bool&
);
void prepare_input(
    ImageRequest&, InferenceRequest&, ModelStore&, const std::atomic_bool&
);
void prepare_input(
    DecisionRequest&, InferenceRequest&, ModelStore&, const std::atomic_bool&
);
void prepare_input(
    SpeechRequest&, InferenceRequest&, ModelStore&, const std::atomic_bool&
);
void prepare_input(
    TranscriptionRequest&, InferenceRequest&, ModelStore&, const std::atomic_bool&
);
void prepare_input(
    TranscriptionFormattingRequest&,
    InferenceRequest&,
    ModelStore&,
    const std::atomic_bool&
);
void prepare_input(
    VideoRequest&, InferenceRequest&, ModelStore&, const std::atomic_bool&
);
void prepare_input(
    DownloadRequest&, InferenceRequest&, ModelStore&, const std::atomic_bool&
);
void prepare_input(
    TierSelectionRequest&, InferenceRequest&, ModelStore&, const std::atomic_bool&
);
void prepare_input(
    ContextRequest&, InferenceRequest&, ModelStore&, const std::atomic_bool&
);
void prepare_input(
    WhisperSelectionRequest&, InferenceRequest&, ModelStore&, const std::atomic_bool&
);
void prepare_input(
    WhisperFormattingRequest&, InferenceRequest&, ModelStore&, const std::atomic_bool&
);
Json present_chat(const InferenceRequest&, Json result);
Json present_image(const InferenceRequest&, const std::vector<fs::path>& images);
Json present_decisions(const std::vector<DecisionAnswer>& answers);
Json present_speech(const InferenceRequest&, const fs::path& output);
Json present_transcription(const InferenceRequest&, const std::string& text);
Json present_video(const InferenceRequest&, const fs::path& output);
}  // namespace kadan::serving
