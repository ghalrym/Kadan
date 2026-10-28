#include "kadan/inference_request.hpp"

#include "kadan/preparation.hpp"

namespace kadan::serving {
namespace {
template <class T> std::optional<T> optional(const Json& body, const char* key) {
    if (!body.contains(key) || body.at(key).is_null())
        return std::nullopt;
    return body.at(key).get<T>();
}
}  // namespace

InferenceRequest parse_request(const Json& envelope) {
    InferenceRequest request;
    request.id = envelope.at("id");
    preparation_check(
        request.id.size() == 32
            && request.id.find_first_not_of("0123456789abcdef") == std::string::npos,
        "Invalid task id",
        422
    );
    preparation_check(envelope.at("command") == "submit", "Invalid task command", 422);
    const auto operation = envelope.at("operation").get<std::string>();
    if (operation == "generate")
        request.operation = Operation::generate;
    else if (operation == "completion")
        request.operation = Operation::completion;
    else if (operation == "load")
        request.operation = Operation::load;
    else if (operation == "unload")
        request.operation = Operation::unload;
    else if (operation == "download")
        request.operation = Operation::download;
    else if (operation == "configure")
        request.operation = Operation::configure;
    else
        throw RequestError("Invalid task operation", 422);
    const auto& body = envelope.at("input");
    const auto feature = body.at("feature").get<std::string>();
    if (feature == "llm") {
        ChatRequest input;
        input.model = optional<std::string>(body, "model_id");
        if (!input.model)
            input.model = optional<std::string>(body, "model");
        input.conversation_id = optional<std::string>(body, "conversation_id");
        input.reuse_prefix = body.value("reuse_prefix", true);
        input.context_supplied = body.contains("context_limit");
        input.context_limit = optional<std::size_t>(body, "context_limit");
        for (const auto& message : body.value("messages", Json::array()))
            input.messages.push_back({message.at("role"), message.at("text")});
        request.input = std::move(input);
        request.workload = Workload::llm;
    } else if (feature == "image") {
        ImageRequest input;
        input.prompt = body.at("prompt");
        input.model = optional<std::string>(body, "model_id");
        input.aspect = body.value("aspect", "1:1");
        input.count = body.value("count", 4u);
        if (body.contains("batch_collection_target")
            && !body.at("batch_collection_target").is_null()) {
            const auto& target = body.at("batch_collection_target");
            preparation_check(
                target.is_number_integer() && target >= 1 && target <= 32,
                "Image batch collection target must be an integer between 1 and 32",
                422
            );
            input.batch_collection_target = target.get<unsigned>();
        }
        input.steps = body.value("steps", 50u);
        input.seed = optional<std::uint64_t>(body, "seed");
        input.image = optional<std::string>(body, "image");
        input.mask = optional<std::string>(body, "mask");
        input.strength = body.value("strength", .65f);
        const auto mode = body.value("mode", "source");
        preparation_check(
            mode == "source" || mode == "reference", "Invalid image edit mode", 422
        );
        input.reference = mode == "reference";
        const auto width = optional<unsigned>(body, "width");
        const auto height = optional<unsigned>(body, "height");
        preparation_check(
            width.has_value() == height.has_value(),
            "Image output dimensions must be supplied together",
            422
        );
        preparation_check(
            !width || input.reference,
            "Output dimensions require reference editing",
            422
        );
        preparation_check(
            !input.reference
                || (input.image && !input.mask && !body.contains("strength")),
            "Reference editing requires an image and does not accept a mask or "
            "strength",
            422
        );
        preparation_check(
            !width
                || (*width >= 32 && *height >= 32 && *width <= 3072 && *height <= 3072
                    && *width % 32 == 0 && *height % 32 == 0
                    && std::size_t(*width) * *height <= 4608 * 1024),
            "Invalid image output dimensions",
            422
        );
        input.width = width.value_or(0);
        input.height = height.value_or(0);
        request.input = std::move(input);
        request.workload = Workload::image;
    } else if (feature == "decisions") {
        DecisionRequest input;
        input.state = body.at("state");
        for (const auto& question : body.at("questions")) {
            DecisionQuestion q{
                question.at("key"), question.at("instructions"), ChoiceQuestion{}
            };
            const auto kind = question.at("type").get<std::string>();
            if (kind == "Choice") {
                ChoiceQuestion choice;
                for (const auto& option : question.at("options"))
                    choice.options.push_back(
                        {option.at("key"), option.at("description")}
                    );
                q.criteria = std::move(choice);
            } else if (kind == "Score")
                q.criteria = ScoreQuestion{
                    question.at("levels").get<std::vector<std::string>>()
                };
            else if (kind == "Noul")
                q.criteria = NoulQuestion{
                    question.value("true_when", ""), question.value("false_when", "")
                };
            else
                throw RequestError("Invalid decision question", 422);
            input.questions.push_back(std::move(q));
        }
        request.input = std::move(input);
        request.workload = Workload::decision;
    } else if (feature == "tts") {
        SpeechRequest input;
        input.script = body.at("script");
        input.model = body.value("model_id", input.model);
        input.language = body.value("language", input.language);
        input.speaker = body.at("voice").at("speaker");
        input.instruction = body.at("voice").value("instruction", "");
        request.input = std::move(input);
        request.workload = Workload::tts;
    } else if (feature == "stt") {
        TranscriptionRequest input;
        input.audio = body.at("audio");
        input.model = optional<std::string>(body, "model");
        input.language = optional<std::string>(body, "language");
        input.formatting = optional<bool>(body, "formatting");
        request.input = std::move(input);
        request.workload = Workload::speech;
    } else if (feature == "stt_formatting") {
        request.input = TranscriptionFormattingRequest{body.at("text")};
        request.workload = Workload::formatting;
    } else if (feature == "video") {
        VideoRequest input;
        input.prompt = body.at("prompt");
        input.model = body.value("model", input.model);
        input.aspect = body.value("aspect", input.aspect);
        const auto resolution = optional<std::string>(body, "resolution");
        const auto duration = optional<unsigned>(body, "duration");
        input.resolution_supplied = resolution.has_value();
        input.duration_supplied = duration.has_value();
        if (resolution) {
            preparation_check(
                *resolution == "480p" || *resolution == "768p",
                "Video resolution must be 480p or 768p",
                422
            );
            input.short_edge = *resolution == "480p" ? 480 : 768;
        }
        if (duration) {
            preparation_check(
                *duration >= 1 && *duration <= 15,
                "Video duration must be 1–15 seconds",
                422
            );
            input.duration = *duration;
        }
        input.seed = body.value("seed", 42ULL);
        request.input = std::move(input);
        request.workload = Workload::video;

    } else if (feature == "model_context") {
        request.input = ContextRequest{
            body.at("model_id"), optional<std::size_t>(body, "context_limit")
        };
    } else if (feature == "whisper_selection") {
        request.input = WhisperSelectionRequest{body.at("model")};
    } else if (feature == "whisper_formatting") {
        request.input = WhisperFormattingRequest{body.at("enabled")};
    } else if (feature == "models") {
        request.input = DownloadRequest{
            body.at("model_id"), body.value("license_acknowledged", false)
        };
    } else
        throw RequestError("Invalid task feature", 422);
    const bool settings = feature == "model_context"
                          || feature == "whisper_selection"
                          || feature == "whisper_formatting";
    preparation_check(
        settings == (request.operation == Operation::configure),
        "Invalid settings operation",
        422
    );
    preparation_check(
        (feature == "models") == (request.operation == Operation::download),
        "Invalid download operation",
        422
    );
    preparation_check(
        request.operation == Operation::generate
            || request.operation == Operation::download
            || request.operation == Operation::configure || feature == "llm",
        "Invalid modality operation",
        422
    );
    return request;
}
}  // namespace kadan::serving
