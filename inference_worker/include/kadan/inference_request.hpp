#pragma once
#include <filesystem>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

#include <nlohmann/json.hpp>

#include "kadan/h3_generation.hpp"
#include "kadan/image_upload.hpp"
#include "kadan/resources.hpp"

namespace kadan::serving {
using Json = nlohmann::json;

struct RequestError : std::runtime_error {
    int status;

    explicit RequestError(const std::string& message, int code = 502)
        : std::runtime_error(message), status(code) {
    }
};
enum class Operation {
    generate,
    completion,
    load,
    unload,
    download,
    configure
};

struct ChatMessage {
    std::string role, text;
};

struct ChatRequest {
    std::vector<ChatMessage> messages;
    std::optional<std::string> model, conversation_id;
    bool context_supplied = false;
    std::optional<std::size_t> context_limit;
    std::string architecture;
    std::size_t max_output_tokens = 256;
};

struct ImageRequest {
    std::string prompt, aspect = "1:1";
    std::optional<std::string> model;
    std::optional<std::uint64_t> seed;
    unsigned count = 4, steps = 50, width = 0, height = 0;
    std::optional<std::string> image, mask;
    std::optional<ImageUpload> source, edit_mask;
    float strength = .65f;
};

struct ChoiceOption {
    std::string key, description;
};

struct ChoiceQuestion {
    std::vector<ChoiceOption> options;
};

struct ScoreQuestion {
    std::vector<std::string> levels;
};

struct NoulQuestion {
    std::string true_when, false_when;
};

struct DecisionQuestion {
    std::string key, instructions;
    std::variant<ChoiceQuestion, ScoreQuestion, NoulQuestion> criteria;

    std::string type() const {
        return std::holds_alternative<ChoiceQuestion>(criteria)  ? "Choice"
               : std::holds_alternative<ScoreQuestion>(criteria) ? "Score"
                                                                 : "Noul";
    }
};

struct DecisionRequest {
    std::string state;
    std::vector<DecisionQuestion> questions;
};

struct ChoiceResult {
    std::string value;
    std::map<std::string, double> probabilities;
};

struct ScoreResult {
    double value;
    std::map<std::string, double> probabilities;
};

struct NoulResult {
    double value;
};

struct DecisionAnswer {
    std::string key;
    double confidence;
    std::variant<ChoiceResult, ScoreResult, NoulResult> result;
};

struct SpeechRequest {
    std::string script, model = "qwen-tts-1.7b-custom", language = "Auto";
    std::string speaker, instruction;
    std::filesystem::path tokenizer;
};

struct TranscriptionRequest {
    std::string audio;
    std::optional<std::string> model, language;
    bool formatting = true;
    std::filesystem::path assets;
    std::vector<float> pcm;
};

struct VideoRequest {
    std::string prompt, model = "h3-fl2va-int8-turbo", aspect = "16:9";
    unsigned short_edge = 768, duration = 8;
    std::uint64_t seed = 42;
    video::H3GenerationPaths paths;
};

struct DownloadRequest {
    std::string model;
    bool license_acknowledged = false;
};

struct ModelSelectionRequest {
    std::string model;
};

struct ContextRequest {
    std::string model;
    std::optional<std::size_t> limit;
};

struct WhisperSelectionRequest {
    std::string model;
};

struct ImageSelectionRequest {
    std::string model;
};

using TaskInput = std::variant<
    ChatRequest,
    ImageRequest,
    DecisionRequest,
    SpeechRequest,
    TranscriptionRequest,
    VideoRequest,
    DownloadRequest,
    ModelSelectionRequest,
    ContextRequest,
    WhisperSelectionRequest,
    ImageSelectionRequest>;

struct EngineConfiguration {
    std::filesystem::path checkpoint;
    std::size_t context_limit = 0;
    bool operator==(const EngineConfiguration&) const = default;
};

struct ContextLimits {
    std::optional<std::size_t> configured;
    std::size_t effective = 0, supported = 0;
};

struct InferenceRequest {
    std::string id, model;
    Operation operation = Operation::generate;
    TaskInput input;
    Workload workload = Workload::llm;
    bool prepared = false;
    EngineConfiguration configuration;
    std::filesystem::path workspace;
    std::optional<ContextLimits> context;
};

InferenceRequest parse_request(const Json& envelope);
}  // namespace kadan::serving
