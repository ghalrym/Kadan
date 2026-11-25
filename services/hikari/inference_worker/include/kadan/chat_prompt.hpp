#pragma once
#include <algorithm>
#include <optional>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
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

struct RenderedChat {
    std::string text;
    std::vector<std::pair<std::size_t, std::size_t>> literal_ranges;

    void literal(std::string_view value) {
        if (value.empty())
            return;
        const auto start = text.size();
        text += value;
        literal_ranges.emplace_back(start, text.size());
    }
};

inline void
inkling_tool_declare(RenderedChat& result, std::span<const ChatTool> tools) {
    if (tools.empty())
        return;
    Json specs = Json::array();
    for (const auto& tool : tools) {
        specs.push_back(
            Json{
                {"description", tool.description},
                {"name", tool.name},
                {"parameters",
                 tool.parameters.is_object() ? tool.parameters : Json::object()},
                {"type", tool.type.empty() ? std::string("function") : tool.type}
            }
        );
    }
    result.text += "<|message_system|>tool_declare<|content_xml|>";
    result.literal(specs.dump());
    result.text += "<|end_message|>";
}

// Renders messages in Inkling 2's exact checkpoint chat template.
// A prompt's leading messages render to its exact leading prefix (including
// tool_declare and the thinking-effort system message).
inline RenderedChat chat_messages(
    std::span<const ChatMessage> messages,
    std::size_t byte_limit,
    std::span<const ChatTool> tools = {},
    const std::optional<std::string>& reasoning_effort = std::nullopt,
    bool emit_trailing_effort = false
) {
    RenderedChat result;
    inkling_tool_declare(result, tools);
    const std::string effort_msg =
        "<|message_system|><|content_text|>Thinking effort level: "
        + inkling_effort_label(reasoning_effort) + "<|end_message|>";
    bool effort_emitted = false;

    for (std::size_t index = 0; index < messages.size(); ++index) {
        const auto& message = messages[index];
        std::string rtok = inkling_role_token(message.role);

        if (!effort_emitted && message.role != "system") {
            result.text += effort_msg;
            effort_emitted = true;
        }

        std::string name;
        if (message.name) {
            name = *message.name;
        } else if (message.role == "tool" && message.tool_call_id) {
            for (const auto& previous : messages.first(index))
                for (const auto& call : previous.tool_calls)
                    if (call.id == *message.tool_call_id)
                        name = call.name;
        }

        auto open = [&](std::string_view channel) {
            result.text += rtok;
            result.literal(name);
            result.text += channel;
        };
        if (message.role == "assistant" && message.reasoning_content
            && !message.reasoning_content->empty()) {
            open("<|content_thinking|>");
            result.literal(*message.reasoning_content);
            result.text += "<|end_message|>";
        }
        auto emit_text = [&](std::string_view content) {
            open("<|content_text|>");
            result.literal(content);
            result.text += "<|end_message|>";
        };
        const auto text = std::string_view(message.text);
        std::size_t start = 0;
        for (std::size_t i = 0; i < message.images.size(); ++i) {
            if (message.role != "user" && message.role != "tool")
                throw std::invalid_argument("chat_image_role");
            const auto at = message.image_offsets.empty() ? text.size()
                                                          : message.image_offsets.at(i);
            if (at > text.size() || at < start)
                throw std::invalid_argument("chat_image_offset");
            if (at > start)
                emit_text(text.substr(start, at - start));
            open("<|content_image|>");
            result.text += "<|unused_200054|><|end_message|>";
            start = at;
        }
        if (start < text.size()
            || (message.images.empty() && message.tool_calls.empty()
                && (!message.reasoning_content || message.reasoning_content->empty())))
            emit_text(text.substr(start));

        if (message.role == "assistant" && !message.tool_calls.empty()) {
            for (const auto& tc : message.tool_calls) {
                if (tc.name.empty())
                    throw std::invalid_argument("chat_tool_call_name");
                if (message.name && !message.name->empty() && *message.name != tc.name)
                    throw std::invalid_argument("chat_assistant_name_mismatch");
                const Json args =
                    tc.arguments.is_object() ? tc.arguments : Json::object();
                result.text += "<|message_model|>";
                result.literal(tc.name);
                result.text += "<|content_invoke_tool_json|>";
                result.literal(
                    "{\"name\":" + Json(tc.name).dump() + ",\"args\":" + args.dump()
                    + "}"
                );
                result.text += "<|end_message|>";
            }
        }

        if (message.role == "assistant") {
            result.text += "<|content_model_end_sampling|>";
        }

        if (result.text.size() > byte_limit)
            throw std::invalid_argument("chat_prompt_bytes");
    }

    if (emit_trailing_effort && !effort_emitted) {
        result.text += effort_msg;
        if (result.text.size() > byte_limit)
            throw std::invalid_argument("chat_prompt_bytes");
    }
    return result;
}

// Full Inkling 2 prompt for generation.
inline RenderedChat chat_prompt(
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

// Only the checkpoint's tool channel can produce a call. Text and thinking are
// independent channels, even if their bytes spell a tool invocation.
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

    void on_special(Special token) {
        if (phase_ == Phase::tool_json) {
            if (token != Special::end_message)
                throw std::invalid_argument("chat_incomplete_tool_call");
            close_tool();
        }
        switch (token) {
        case Special::message_model:
            phase_ = Phase::header;
            header_.clear();
            break;
        case Special::content_thinking:
            phase_ = Phase::thinking;
            break;
        case Special::content_text:
            phase_ = Phase::text;
            break;
        case Special::content_invoke_tool_json:
            phase_ = Phase::tool_json;
            break;
        case Special::end_message:
            phase_ = Phase::idle;
            header_.clear();
            break;
        }
    }

    std::string on_text(std::string_view piece) {
        switch (phase_) {
        case Phase::text:
        case Phase::idle:
            return std::string(piece);
        case Phase::tool_json:
            tool_buffer_ += piece;
            break;
        case Phase::header:
            header_ += piece;
            break;
        case Phase::thinking:
            break;
        }
        return {};
    }

    Json finish(const std::string& request_id, bool complete) {
        if (!complete)
            return Json::array();
        if (phase_ == Phase::tool_json)
            close_tool();
        for (std::size_t i = 0; i < calls_.size(); ++i)
            calls_[i]["id"] = "call_" + request_id + "_" + std::to_string(i);
        return std::move(calls_);
    }

private:
    void close_tool() {
        std::vector<std::set<std::string>> fields;
        auto unique = [&](int, Json::parse_event_t event, Json& value) {
            if (event == Json::parse_event_t::object_start)
                fields.emplace_back();
            else if (event == Json::parse_event_t::object_end)
                fields.pop_back();
            else if (
                event == Json::parse_event_t::key
                && !fields.back().insert(value.get<std::string>()).second
            )
                throw std::invalid_argument("chat_duplicate_tool_field");
            return true;
        };
        auto call = Json::parse(tool_buffer_, unique, false);
        if (!call.is_object() || !call.contains("name") || !call["name"].is_string()
            || (call.contains("args") == call.contains("arguments"))
            || call.size() != 2)
            throw std::invalid_argument("chat_invalid_tool_call");
        const auto name = call["name"].get<std::string>();
        const auto& args = call.at(call.contains("args") ? "args" : "arguments");
        if (name.empty() || name.size() > 120 || (!header_.empty() && header_ != name)
            || name.find_first_not_of(
                   "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_"
               ) != std::string::npos
            || !args.is_object())
            throw std::invalid_argument("chat_invalid_tool_call");
        calls_.push_back(
            {{"type", "function"},
             {"function", {{"name", name}, {"arguments", args.dump()}}}}
        );
        tool_buffer_.clear();
    }

    enum class Phase {
        idle,
        header,
        thinking,
        text,
        tool_json
    };
    Phase phase_ = Phase::idle;
    std::string header_, tool_buffer_;
    Json calls_ = Json::array();
};
}  // namespace kadan::serving
