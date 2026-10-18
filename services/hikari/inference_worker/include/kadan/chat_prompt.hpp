#pragma once
#include <stdexcept>
#include <string>

#include "kadan/inference_request.hpp"

namespace kadan::serving {
inline std::string trim_chat_text(std::string text) {
    const auto first = text.find_first_not_of(" \t\r\n");
    if (first == std::string::npos)
        return {};
    return text.substr(first, text.find_last_not_of(" \t\r\n") - first + 1);
}

// Text-only form of the Qwen checkpoint template used by the API. Tools and
// multimodal messages are not part of this endpoint's supported request model.
inline std::string
chat_prompt(const std::vector<ChatMessage>& messages, std::size_t byte_limit) {
    if (messages.empty() || messages.size() > byte_limit)
        throw std::invalid_argument("chat_messages");
    std::string result;
    bool user_seen = false;
    for (std::size_t index = 0; index < messages.size(); ++index) {
        const auto& message = messages[index];
        const auto role = message.role;
        if (role != "system" && role != "user" && role != "assistant")
            throw std::invalid_argument("chat_role");
        if (role == "system" && index != 0)
            throw std::invalid_argument("chat_system_position");
        user_seen = user_seen || role == "user";
        auto text = trim_chat_text(message.text);
        if (role == "assistant") {
            std::string reasoning;
            const auto end = text.rfind("</think>");
            if (end != std::string::npos) {
                reasoning = text.substr(0, text.find("</think>"));
                const auto start = reasoning.rfind("<think>");
                if (start != std::string::npos)
                    reasoning.erase(0, start + 7);
                text.erase(0, end + 8);
            }
            text = "<think>\n" + trim_chat_text(std::move(reasoning)) + "\n</think>\n\n"
                   + trim_chat_text(std::move(text));
        }
        result += "<|im_start|>" + role + "\n" + text + "<|im_end|>\n";
        if (result.size() > byte_limit)
            throw std::invalid_argument("chat_prompt_bytes");
    }
    if (!user_seen)
        throw std::invalid_argument("chat_user_required");
    result += "<|im_start|>assistant\n<think>\n\n</think>\n\n";
    return result;
}
}  // namespace kadan::serving
