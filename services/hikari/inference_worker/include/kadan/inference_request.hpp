#pragma once
#include <filesystem>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

#include <nlohmann/json.hpp>

#include "kadan/image_upload.hpp"
#include "kadan/inference_trace.hpp"
#include "kadan/lfm3.hpp"
#include "kadan/resources.hpp"
#include "kadan/text_decision.hpp"

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

struct ChatTool {
    std::string name;
    std::string description;
    Json parameters = Json::object();
    std::string type = "function";
};

struct ChatToolCall {
    std::string id;
    std::string name;
    Json arguments = Json::object();
};

struct ChatMessage {
    std::string role, text;
    std::vector<std::string> images = {};
    std::vector<std::vector<std::uint8_t>> decoded_images = {};
    std::vector<ChatToolCall> tool_calls = {};
    std::optional<std::string> tool_call_id = std::nullopt;
    std::optional<std::string> name = std::nullopt;
    std::optional<std::string> reasoning_content = std::nullopt;
    // Byte offset in text where each ordered content image occurred.
    std::vector<std::size_t> image_offsets = {};
};

struct ChatRequest {
    std::vector<ChatMessage> messages;
    std::vector<ChatTool> tools;
    std::vector<lfm3::Lfm3ChatMessage> lfm3_messages;
    Json lfm3_tools;
    std::optional<std::string> model, conversation_id;
    bool reuse_prefix = true;
    bool stream = false;
    bool context_supplied = false;
    std::optional<std::size_t> context_limit;
    std::optional<std::size_t> max_output_tokens;
    std::optional<std::size_t> max_tokens;
    std::optional<std::size_t> reasoning_tokens;
    std::optional<std::string> reasoning_effort;
};

struct VisionRequest {
    std::string model = "lfm3-vl-2b-bf16";
    std::string question;
    std::string media_type;
    std::string image_base64;
    std::optional<std::size_t> max_new_tokens;
};

struct AudioCallRequest {
    std::string model = "lfm3-audio-2b-bf16";
    std::string mode;
    std::optional<std::string> audio;
    std::optional<std::string> script;
    std::vector<lfm3::Lfm3ChatMessage> messages;
    bool sentence_tts = false;
    bool codec_streaming_probe = false;
};

struct DecisionRequest {
    std::optional<std::string> model;
    std::string state;
    std::vector<text::DecisionQuestion> questions;
};

struct ImageRequest {
    std::string prompt, aspect = "1:1";
    std::optional<std::string> model;
    std::optional<std::uint64_t> seed;
    unsigned count = 4, steps = 4, width = 0, height = 0;
    std::optional<std::string> image, mask;
    std::vector<std::string> reference_images;
    std::optional<ImageUpload> source, edit_mask;
    std::vector<ImageUpload> reference_sources;
    float strength = .65f;
    bool reference = false;
    std::size_t source_width = 0, source_height = 0;
    std::optional<unsigned> batch_collection_target;
};

struct SpeechRequest {
    std::string script, model = "qwen-tts-1.7b-custom", language = "Auto";
    std::string speaker, instruction;
    std::filesystem::path tokenizer;
};

struct TranscriptionRequest {
    std::string audio;
    std::optional<std::string> model, language;
    std::optional<bool> formatting;
    std::filesystem::path assets;
    std::vector<float> pcm;
};

struct TranscriptionFormattingRequest {
    std::string text;
};

struct DownloadRequest {
    std::string model;
    bool license_acknowledged = false;
};

struct ContextRequest {
    std::string model;
    std::optional<std::size_t> limit;
};

struct WhisperFormattingRequest {
    bool enabled;
};

struct LlmSelectionRequest {
    std::string model;
};

using TaskInput = std::variant<
    ChatRequest,
    VisionRequest,
    AudioCallRequest,
    DecisionRequest,
    ImageRequest,
    SpeechRequest,
    TranscriptionRequest,
    TranscriptionFormattingRequest,
    DownloadRequest,
    ContextRequest,
    WhisperFormattingRequest,
    LlmSelectionRequest>;

struct EngineConfiguration {
    std::filesystem::path checkpoint;
    std::size_t context_limit = 0;
    bool operator==(const EngineConfiguration&) const = default;
};

struct ContextLimits {
    std::optional<std::size_t> configured;
    std::size_t effective = 0, supported = 0;
};

enum class Priority : std::uint8_t {
    interactive = 0,
    background = 1
};

struct InferenceRequest {
    std::string id, model;
    TraceSink trace;
    Operation operation = Operation::generate;
    TaskInput input;
    Workload workload = Workload::llm;
    Priority priority = Priority::interactive;
    bool prepared = false;
    EngineConfiguration configuration;
    std::filesystem::path workspace;
    std::optional<ContextLimits> context;
};

InferenceRequest parse_request(const Json& envelope);
}  // namespace kadan::serving
