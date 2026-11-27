#include <cmath>
#include <limits>

#include "kadan/inference_request.hpp"

#include "kadan/inkling_variant.hpp"
#include "kadan/preparation.hpp"

namespace kadan::serving {
namespace {
template <class T> std::optional<T> optional(const Json& body, const char* key) {
    if (!body.contains(key) || body.at(key).is_null())
        return std::nullopt;
    return body.at(key).get<T>();
}

std::optional<float> optional_bounded_float(
    const Json& body,
    const char* key,
    float min_value,
    float max_value,
    bool min_inclusive,
    const char* message
) {
    if (!body.contains(key) || body.at(key).is_null())
        return std::nullopt;
    const auto& value = body.at(key);
    preparation_check(value.is_number() && !value.is_boolean(), message, 422);
    const double raw = value.get<double>();
    preparation_check(
        std::isfinite(raw) && raw <= static_cast<double>(max_value)
            && (min_inclusive ? raw >= static_cast<double>(min_value)
                              : raw > static_cast<double>(min_value)),
        message,
        422
    );
    return static_cast<float>(raw);
}

std::optional<std::uint64_t> optional_uint64(
    const Json& body, const char* key, std::uint64_t max_value, const char* message
) {
    if (!body.contains(key) || body.at(key).is_null())
        return std::nullopt;
    const auto& value = body.at(key);
    preparation_check(value.is_number_integer() && !value.is_boolean(), message, 422);
    if (value.is_number_unsigned()) {
        const auto raw = value.get<std::uint64_t>();
        preparation_check(raw <= max_value, message, 422);
        return raw;
    }
    const auto raw = value.get<std::int64_t>();
    preparation_check(
        raw >= 0 && static_cast<std::uint64_t>(raw) <= max_value, message, 422
    );
    return static_cast<std::uint64_t>(raw);
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
    if (const auto priority = optional<std::string>(body, "priority")) {
        if (*priority == "interactive")
            request.priority = Priority::interactive;
        else if (*priority == "background")
            request.priority = Priority::background;
        else
            throw RequestError("Invalid task priority", 422);
    }
    const auto feature = body.at("feature").get<std::string>();
    if (feature == "llm") {
        ChatRequest input;
        input.model = optional<std::string>(body, "model_id");
        if (!input.model)
            input.model = optional<std::string>(body, "model");
        input.weight_variant = optional<std::string>(body, "weight_variant");
        if (input.weight_variant) {
            preparation_check(
                parse_inkling_variant(*input.weight_variant).has_value(),
                "Weight variant must be 'base' or 'abliterated'.",
                422
            );
        }
        input.conversation_id = optional<std::string>(body, "conversation_id");
        input.reuse_prefix = body.value("reuse_prefix", true);
        input.stream = body.value("stream", false);
        input.context_supplied = body.contains("context_limit");
        input.context_limit = optional<std::size_t>(body, "context_limit");
        input.max_output_tokens = optional<std::size_t>(body, "max_tokens");
        if (!input.max_output_tokens)
            input.max_output_tokens = optional<std::size_t>(body, "max_output_tokens");
        input.max_tokens = input.max_output_tokens;
        input.reasoning_tokens = optional<std::size_t>(body, "reasoning_tokens");
        input.reasoning_effort = optional<std::string>(body, "reasoning_effort");
        input.temperature = optional_bounded_float(
            body,
            "temperature",
            0.0f,
            2.0f,
            true,
            "Temperature must be between 0 and 2."
        );
        input.top_p = optional_bounded_float(
            body, "top_p", 0.0f, 1.0f, false, "top_p must be in (0, 1]."
        );
        if (const auto top_k = optional_uint64(
                body,
                "top_k",
                static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max()),
                "top_k must be a non-negative integer."
            )) {
            input.top_k = static_cast<std::size_t>(*top_k);
        }
        input.min_p = optional_bounded_float(
            body, "min_p", 0.0f, 1.0f, true, "min_p must be between 0 and 1."
        );
        input.repetition_penalty = optional_bounded_float(
            body,
            "repetition_penalty",
            1.0f,
            2.0f,
            true,
            "repetition_penalty must be between 1 and 2."
        );
        input.presence_penalty = optional_bounded_float(
            body,
            "presence_penalty",
            -2.0f,
            2.0f,
            true,
            "presence_penalty must be between -2 and 2."
        );
        input.seed = optional_uint64(
            body,
            "seed",
            static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()),
            "seed must be a non-negative integer."
        );

        if (body.contains("tools") && !body.at("tools").is_null()) {
            input.lfm3_tools = body.at("tools");
            preparation_check(
                body.at("tools").is_array() && body.at("tools").size() <= 256,
                "Invalid chat tools list",
                422
            );
            for (const auto& item : body.at("tools")) {
                preparation_check(item.is_object(), "Invalid chat tool", 422);
                const auto& fn = item.contains("function") ? item.at("function") : item;
                preparation_check(
                    fn.is_object() && fn.contains("name") && fn.at("name").is_string(),
                    "Invalid chat tool function",
                    422
                );
                ChatTool tool;
                tool.type = item.value("type", std::string("function"));
                tool.name = fn.at("name").get<std::string>();
                tool.description = fn.value("description", std::string());
                tool.parameters =
                    fn.contains("parameters") && fn.at("parameters").is_object()
                        ? fn.at("parameters")
                        : Json::object();
                input.tools.push_back(std::move(tool));
            }
        }

        for (const auto& message : body.value("messages", Json::array())) {
            ChatMessage msg;
            msg.role = message.at("role").get<std::string>();
            msg.tool_call_id = optional<std::string>(message, "tool_call_id");
            msg.name = optional<std::string>(message, "name");
            msg.reasoning_content = optional<std::string>(message, "reasoning_content");

            const Json* content_field = nullptr;
            if (message.contains("content") && !message.at("content").is_null())
                content_field = &message.at("content");
            else if (message.contains("text") && !message.at("text").is_null())
                content_field = &message.at("text");

            if (content_field) {
                if (content_field->is_string()) {
                    msg.text = content_field->get<std::string>();
                } else if (content_field->is_array()) {
                    for (const auto& part : *content_field) {
                        if (part.is_string()) {
                            msg.text += part.get<std::string>();
                            continue;
                        }
                        preparation_check(
                            part.is_object(), "Invalid chat content part", 422
                        );
                        const auto ptype = part.value("type", std::string("text"));
                        if (ptype == "text" || ptype == "input_text") {
                            const auto piece = part.value("text", std::string());
                            msg.text += piece;
                        } else if (
                            ptype == "image_url" || ptype == "image"
                            || ptype == "input_image"
                        ) {
                            preparation_check(
                                msg.role == "user" || msg.role == "tool",
                                "Images are only supported on user and tool messages",
                                422
                            );
                            std::string url;
                            if (part.contains("image_url")) {
                                const auto& iu = part.at("image_url");
                                if (iu.is_string())
                                    url = iu.get<std::string>();
                                else if (iu.is_object() && iu.contains("url"))
                                    url = iu.at("url").get<std::string>();
                            } else if (
                                part.contains("url") && part.at("url").is_string()
                            ) {
                                url = part.at("url").get<std::string>();
                            }
                            preparation_check(
                                !url.empty(), "Image content part requires a URL", 422
                            );
                            msg.image_offsets.push_back(msg.text.size());
                            msg.images.push_back(std::move(url));
                        } else {
                            throw RequestError(
                                "Unsupported chat content part type", 422
                            );
                        }
                    }
                } else {
                    throw RequestError("Invalid chat message content", 422);
                }
            }

            if (message.contains("images") && message.at("images").is_array()) {
                preparation_check(
                    msg.role == "user" || msg.role == "tool",
                    "Images are only supported on user and tool messages",
                    422
                );
                for (const auto& img : message.at("images")) {
                    preparation_check(img.is_string(), "Invalid image entry", 422);
                    msg.image_offsets.push_back(msg.text.size());
                    msg.images.push_back(img.get<std::string>());
                }
            }

            if (message.contains("tool_calls") && !message.at("tool_calls").is_null()) {
                preparation_check(
                    msg.role == "assistant" && message.at("tool_calls").is_array(),
                    "Only assistant messages may contain tool calls",
                    422
                );
                for (const auto& tc : message.at("tool_calls")) {
                    preparation_check(
                        tc.is_object() && tc.contains("function")
                            && tc.at("function").is_object(),
                        "Invalid assistant tool call",
                        422
                    );
                    const auto& fn = tc.at("function");
                    ChatToolCall call;
                    call.id = tc.value("id", std::string());
                    call.name = fn.at("name").get<std::string>();
                    if (fn.contains("arguments") && !fn.at("arguments").is_null()) {
                        const auto& raw_args = fn.at("arguments");
                        if (raw_args.is_object()) {
                            call.arguments = raw_args;
                        } else if (raw_args.is_string()) {
                            try {
                                call.arguments =
                                    Json::parse(raw_args.get<std::string>());
                            } catch (...) {
                                throw RequestError(
                                    "Tool call arguments must be valid JSON object", 422
                                );
                            }
                            preparation_check(
                                call.arguments.is_object(),
                                "Tool call arguments must be a JSON object",
                                422
                            );
                        } else {
                            throw RequestError(
                                "Tool call arguments must be a JSON object", 422
                            );
                        }
                    }
                    msg.tool_calls.push_back(std::move(call));
                }
            }

            input.messages.push_back(std::move(msg));
        }
        request.input = std::move(input);
        request.workload = Workload::llm;
    } else if (feature == "audio_call") {
        AudioCallRequest input;
        if (const auto m = optional<std::string>(body, "model_id"))
            input.model = *m;
        else if (const auto m2 = optional<std::string>(body, "model"))
            input.model = *m2;
        input.mode = body.value("mode", std::string{});
        preparation_check(
            input.mode == "transcribe" || input.mode == "reply"
                || input.mode == "answer" || input.mode == "tts"
                || input.mode == "turn",
            "Unknown audio call operation",
            422
        );
        input.audio = optional<std::string>(body, "audio");
        input.script = optional<std::string>(body, "script");
        input.sentence_tts = body.value("sentence_tts", false);
        input.codec_streaming_probe = body.value("codec_streaming_probe", false);
        preparation_check(
            !input.sentence_tts || (input.mode == "turn" && !input.codec_streaming_probe),
            "Sentence TTS requires a complete turn without the private codec probe",
            422
        );
        preparation_check(
            !input.codec_streaming_probe || input.mode == "turn",
            "The private codec probe requires a complete audio turn",
            422
        );
        for (const auto& message : body.value("messages", Json::array())) {
            const std::string role = message.at("role").get<std::string>();
            const std::string text = message.at("text").get<std::string>();
            preparation_check(
                (role == "system" || role == "user" || role == "assistant") && !text.empty(),
                "Audio call history must contain text messages without tools",
                422
            );
            input.messages.push_back({role, text, Json::array(), ""});
        }
        request.input = std::move(input);
        request.workload = Workload::lfm3_audio;
    } else if (feature == "decisions") {
        DecisionRequest input;
        if (const auto m = optional<std::string>(body, "model_id"))
            input.model = *m;
        else if (const auto m2 = optional<std::string>(body, "model"))
            input.model = *m2;
        input.weight_variant = optional<std::string>(body, "weight_variant");
        if (input.weight_variant) {
            preparation_check(
                parse_inkling_variant(*input.weight_variant).has_value(),
                "Weight variant must be 'base' or 'abliterated'.",
                422
            );
        }
        preparation_check(
            body.contains("state") && body.at("state").is_string(),
            "Decisions require state to be a valid string.",
            422
        );
        input.state = body.at("state").get<std::string>();
        preparation_check(
            input.state.find_first_not_of(" \t\r\n") != std::string::npos
                && input.state.size() <= 8000,
            "Decision state must fit 8000 characters.",
            422
        );
        preparation_check(
            body.contains("questions") && body.at("questions").is_array()
                && body.at("questions").size() >= 1 && body.at("questions").size() <= 8,
            "Decisions require one to eight questions.",
            422
        );
        Json serialized_questions = Json::array();
        std::vector<std::string> seen_keys;
        for (const auto& q : body.at("questions")) {
            preparation_check(q.is_object(), "Decision questions must be mappings.", 422);
            text::DecisionQuestion parsed_q;
            preparation_check(
                q.contains("key") && q.at("key").is_string()
                    && q.at("key").get<std::string>().find_first_not_of(" \t\r\n")
                           != std::string::npos,
                "Decisions require question key to be a valid string.",
                422
            );
            parsed_q.key = q.at("key").get<std::string>();
            preparation_check(
                std::find(seen_keys.begin(), seen_keys.end(), parsed_q.key)
                    == seen_keys.end(),
                "Decision question keys must be unique.",
                422
            );
            seen_keys.push_back(parsed_q.key);
            preparation_check(
                q.contains("instructions") && q.at("instructions").is_string()
                    && q.at("instructions").get<std::string>().find_first_not_of(" \t\r\n")
                           != std::string::npos,
                "Decisions require instructions to be a valid string.",
                422
            );
            parsed_q.instructions = q.at("instructions").get<std::string>();
            const std::string kind = q.value("type", std::string{});
            Json wire = {
                {"key", parsed_q.key},
                {"instructions", parsed_q.instructions},
                {"type", kind}
            };
            if (kind == "Choice") {
                parsed_q.kind = text::DecisionQuestion::Kind::choice;
                preparation_check(
                    q.contains("options") && q.at("options").is_array()
                        && !q.at("options").empty(),
                    "Choice questions require options.",
                    422
                );
                Json wire_opts = Json::array();
                std::vector<std::string> opt_keys;
                for (const auto& opt : q.at("options")) {
                    preparation_check(opt.is_object(), "Choice options must be mappings.", 422);
                    preparation_check(
                        opt.contains("key") && opt.at("key").is_string()
                            && opt.at("key").get<std::string>().find_first_not_of(" \t\r\n")
                                   != std::string::npos,
                        "Choice options require key to be a valid string.",
                        422
                    );
                    const std::string ok = opt.at("key").get<std::string>();
                    preparation_check(
                        std::find(opt_keys.begin(), opt_keys.end(), ok) == opt_keys.end(),
                        "Choice option keys must be unique.",
                        422
                    );
                    opt_keys.push_back(ok);
                    preparation_check(
                        opt.contains("description") && opt.at("description").is_string(),
                        "Choice options require description to be a valid string.",
                        422
                    );
                    const std::string desc = opt.at("description").get<std::string>();
                    parsed_q.options.emplace_back(ok, desc);
                    wire_opts.push_back({{"key", ok}, {"description", desc}});
                }
                wire["options"] = std::move(wire_opts);
            } else if (kind == "Score") {
                parsed_q.kind = text::DecisionQuestion::Kind::score;
                preparation_check(
                    q.contains("levels") && q.at("levels").is_array()
                        && q.at("levels").size() >= 2 && q.at("levels").size() <= 10,
                    "Score rubrics require 2 to 10 levels.",
                    422
                );
                for (const auto& lv : q.at("levels")) {
                    preparation_check(
                        lv.is_string()
                            && lv.get<std::string>().find_first_not_of(" \t\r\n")
                                   != std::string::npos,
                        "Score levels must be valid strings.",
                        422
                    );
                    parsed_q.levels.push_back(lv.get<std::string>());
                }
                wire["levels"] = parsed_q.levels;
            } else if (kind == "Noul") {
                parsed_q.kind = text::DecisionQuestion::Kind::noul;
                if (q.contains("trueWhen") && q.at("trueWhen").is_string())
                    parsed_q.true_when = q.at("trueWhen").get<std::string>();
                else if (q.contains("true_when") && q.at("true_when").is_string())
                    parsed_q.true_when = q.at("true_when").get<std::string>();
                if (q.contains("falseWhen") && q.at("falseWhen").is_string())
                    parsed_q.false_when = q.at("falseWhen").get<std::string>();
                else if (q.contains("false_when") && q.at("false_when").is_string())
                    parsed_q.false_when = q.at("false_when").get<std::string>();
                wire["true_when"] = parsed_q.true_when;
                wire["false_when"] = parsed_q.false_when;
            } else {
                throw RequestError("Decisions support Choice, Score and Noul questions only.", 422);
            }
            serialized_questions.push_back(std::move(wire));
            input.questions.push_back(std::move(parsed_q));
        }
        const std::string payload =
            Json{{"state", input.state}, {"questions", serialized_questions}}.dump();
        preparation_check(
            payload.size() <= 16000,
            "The decision request must fit 16000 characters.",
            422
        );
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
        input.steps = body.value("steps", 4u);
        input.seed = optional<std::uint64_t>(body, "seed");
        input.image = optional<std::string>(body, "image");
        input.mask = optional<std::string>(body, "mask");
        input.strength = body.value("strength", .65f);
        const auto mode = body.value("mode", "source");
        preparation_check(
            mode == "source" || mode == "reference", "Invalid image edit mode", 422
        );
        input.reference = mode == "reference";
        if (body.contains("reference_images") && !body.at("reference_images").is_null()) {
            preparation_check(
                input.reference,
                "Additional reference images require reference editing",
                422
            );
            const auto& refs = body.at("reference_images");
            preparation_check(
                refs.is_array() && !refs.empty() && refs.size() <= 3,
                "Reference images must contain between 1 and 3 entries",
                422
            );
            for (const auto& entry : refs) {
                preparation_check(
                    entry.is_string() && !entry.get<std::string>().empty(),
                    "Each reference image must be a non-empty string",
                    422
                );
                input.reference_images.push_back(entry.get<std::string>());
            }
        }
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
    } else if (feature == "model_context") {
        request.input = ContextRequest{
            body.at("model_id"), optional<std::size_t>(body, "context_limit")
        };
    } else if (feature == "whisper_formatting") {
        request.input = WhisperFormattingRequest{body.at("enabled")};
    } else if (feature == "llm_selection") {
        request.input = LlmSelectionRequest{body.at("model_id")};
    } else if (feature == "models") {
        request.input = DownloadRequest{
            body.at("model_id"), body.value("license_acknowledged", false)
        };
    } else
        throw RequestError("Invalid task feature", 422);
    const bool settings = feature == "model_context" || feature == "whisper_formatting"
                          || feature == "llm_selection";
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
