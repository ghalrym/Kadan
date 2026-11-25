#pragma once
#include <algorithm>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "kadan/inference_request.hpp"

namespace kadan::serving {
inline std::string trim_chat_text(std::string text) {
    const auto first = text.find_first_not_of(" \t\r\n");
    if (first == std::string::npos)
        return {};
    return text.substr(first, text.find_last_not_of(" \t\r\n") - first + 1);
}

inline std::string
inkling_effort_label(const std::optional<std::string>& reasoning_effort) {
    if (!reasoning_effort.has_value())
        return "0.9";
    const auto key = trim_chat_text(*reasoning_effort);
    if (key == "none")
        return "0";
    if (key == "minimal")
        return "0.1";
    if (key == "low")
        return "0.2";
    if (key == "medium")
        return "0.7";
    if (key == "high")
        return "0.9";
    if (key == "xhigh" || key == "max")
        return "0.99";
    throw std::invalid_argument("chat_reasoning_effort");
}

inline std::string inkling_role_token(std::string_view role) {
    if (role == "user")
        return "<|message_user|>";
    if (role == "assistant")
        return "<|message_model|>";
    if (role == "system")
        return "<|message_system|>";
    if (role == "tool")
        return "<|message_tool|>";
    throw std::invalid_argument("chat_role");
}

inline std::string inkling_tool_declare(std::span<const ChatTool> tools) {
    if (tools.empty())
        return {};
    Json specs = Json::array();
    for (const auto& tool : tools) {
        specs.push_back(Json{
            {"description", tool.description},
            {"name", tool.name},
            {"parameters", tool.parameters.is_object() ? tool.parameters : Json::object()},
            {"type", tool.type.empty() ? std::string("function") : tool.type}
        });
    }
    return "<|message_system|>tool_declare<|content_xml|>" + specs.dump()
           + "<|end_message|>";
}

// Renders messages in Inkling 2's exact checkpoint chat template.
// A prompt's leading messages render to its exact leading prefix (including
// tool_declare and the thinking-effort system message).
inline std::string chat_messages(
    std::span<const ChatMessage> messages,
    std::size_t byte_limit,
    std::span<const ChatTool> tools = {},
    const std::optional<std::string>& reasoning_effort = std::nullopt,
    bool emit_trailing_effort = false
) {
    std::string result = inkling_tool_declare(tools);
    const std::string effort_msg =
        "<|message_system|><|content_text|>Thinking effort level: "
        + inkling_effort_label(reasoning_effort) + "<|end_message|>";
    bool effort_emitted = false;

    for (std::size_t index = 0; index < messages.size(); ++index) {
        const auto& message = messages[index];
        std::string rtok = inkling_role_token(message.role);

        if (!effort_emitted && message.role != "system") {
            result += effort_msg;
            effort_emitted = true;
        }

        if (message.role == "tool") {
            if (message.name && !message.name->empty()) {
                rtok += *message.name;
            } else if (message.tool_call_id && !message.tool_call_id->empty()) {
                for (std::size_t prev = 0; prev < messages.size(); ++prev) {
                    if (messages[prev].role != "assistant")
                        continue;
                    for (const auto& tc : messages[prev].tool_calls) {
                        if (tc.id == *message.tool_call_id && !tc.name.empty()) {
                            rtok += tc.name;
                        }
                    }
                }
            }
        } else if (message.role == "assistant" && message.name && !message.name->empty()) {
            rtok += *message.name;
        }

        if (message.role == "assistant" && message.reasoning_content
            && !message.reasoning_content->empty()) {
            result += rtok + "<|content_thinking|>" + *message.reasoning_content
                      + "<|end_message|>";
        }

        const bool has_structured_parts =
            !message.images.empty()
            || (message.role == "assistant"
                && (!message.tool_calls.empty()
                    || (message.reasoning_content
                        && !message.reasoning_content->empty())));
        if (!message.text.empty() || !has_structured_parts) {
            result += rtok + "<|content_text|>" + message.text + "<|end_message|>";
        }

        if (!message.images.empty()) {
            if (message.role != "user" && message.role != "tool")
                throw std::invalid_argument("chat_image_role");
            for (std::size_t img = 0; img < message.images.size(); ++img) {
                result += rtok + "<|content_image|><|unused_200054|><|end_message|>";
            }
        }

        if (message.role == "assistant" && !message.tool_calls.empty()) {
            for (const auto& tc : message.tool_calls) {
                if (tc.name.empty())
                    throw std::invalid_argument("chat_tool_call_name");
                if (message.name && !message.name->empty() && *message.name != tc.name)
                    throw std::invalid_argument("chat_assistant_name_mismatch");
                const Json args =
                    tc.arguments.is_object() ? tc.arguments : Json::object();
                result += "<|message_model|>" + tc.name
                          + "<|content_invoke_tool_json|>{\"name\":" + Json(tc.name).dump()
                          + ",\"args\":" + args.dump() + "}<|end_message|>";
            }
        }

        if (message.role == "assistant") {
            result += "<|content_model_end_sampling|>";
        }

        if (result.size() > byte_limit)
            throw std::invalid_argument("chat_prompt_bytes");
    }

    if (emit_trailing_effort && !effort_emitted) {
        result += effort_msg;
        if (result.size() > byte_limit)
            throw std::invalid_argument("chat_prompt_bytes");
    }
    return result;
}

// Full Inkling 2 prompt for generation.
inline std::string chat_prompt(
    const std::vector<ChatMessage>& messages,
    std::size_t byte_limit,
    std::span<const ChatTool> tools = {},
    const std::optional<std::string>& reasoning_effort = std::nullopt
) {
    if (messages.empty() || messages.size() > byte_limit)
        throw std::invalid_argument("chat_messages");
    if (std::none_of(messages.begin(), messages.end(), [](const auto& message) {
            return message.role == "user" || message.role == "tool";
        }))
        throw std::invalid_argument("chat_user_required");
    return chat_messages(messages, byte_limit, tools, reasoning_effort, true);
}

// Formats an Inkling 2 JSON tool invocation {"name": ..., "args": {...}} into
// Kadan's canonical <tool_call><function=...><parameter=...> XML block.
inline std::string format_inkling_tool_call_xml(std::string_view raw_json) {
    try {
        const auto parsed = Json::parse(raw_json);
        if (!parsed.is_object() || !parsed.contains("name")
            || !parsed.at("name").is_string())
            return std::string(raw_json);
        const auto name = parsed.at("name").get<std::string>();
        const Json args = parsed.contains("args") && parsed.at("args").is_object()
                              ? parsed.at("args")
                          : parsed.contains("arguments")
                                  && parsed.at("arguments").is_object()
                              ? parsed.at("arguments")
                              : Json::object();
        std::string xml = "<tool_call>\n<function=" + name + ">\n";
        for (auto it = args.begin(); it != args.end(); ++it) {
            xml += "<parameter=" + it.key() + ">\n";
            if (it.value().is_string())
                xml += it.value().get<std::string>();
            else
                xml += it.value().dump();
            xml += "\n</parameter>\n";
        }
        xml += "</function>\n</tool_call>";
        return xml;
    } catch (...) {
        return std::string(raw_json);
    }
}

// Streaming state machine for Inkling 2 output tokens.
// Filters internal <|content_thinking|> blocks, streams <|content_text|> chunks
// directly, and converts <|content_invoke_tool_json|> blocks into <tool_call> XML.
class InklingStreamParser {
public:
    enum class Special {
        message_model,
        content_thinking,
        content_text,
        content_invoke_tool_json,
        end_message
    };

    bool in_thinking() const {
        return phase_ == Phase::thinking;
    }

    bool in_text() const {
        return phase_ == Phase::text;
    }

    std::string on_special(Special token) {
        switch (token) {
        case Special::message_model: {
            std::string flushed = flush_tool();
            phase_ = Phase::header;
            return flushed;
        }
        case Special::content_thinking:
            phase_ = Phase::thinking;
            return {};
        case Special::content_text:
            phase_ = Phase::text;
            return {};
        case Special::content_invoke_tool_json:
            tool_buffer_.clear();
            phase_ = Phase::tool_json;
            return {};
        case Special::end_message: {
            std::string flushed = flush_tool();
            phase_ = Phase::idle;
            return flushed;
        }
        }
        return {};
    }

    std::string on_text(std::string_view piece) {
        if (piece.empty())
            return {};
        switch (phase_) {
        case Phase::text:
        case Phase::idle:
            has_visible_output_ = true;
            return std::string(piece);
        case Phase::tool_json:
            tool_buffer_ += piece;
            return {};
        case Phase::header:
        case Phase::thinking:
            return {};
        }
        return {};
    }

    std::string finish() {
        return flush_tool();
    }

private:
    enum class Phase {
        idle,
        header,
        thinking,
        text,
        tool_json
    };

    std::string flush_tool() {
        if (phase_ != Phase::tool_json || tool_buffer_.empty()) {
            tool_buffer_.clear();
            return {};
        }
        std::string xml = format_inkling_tool_call_xml(tool_buffer_);
        tool_buffer_.clear();
        if (has_visible_output_)
            xml.insert(0, "\n");
        has_visible_output_ = true;
        return xml;
    }

    Phase phase_ = Phase::idle;
    std::string tool_buffer_;
    bool has_visible_output_ = false;
};
}  // namespace kadan::serving
