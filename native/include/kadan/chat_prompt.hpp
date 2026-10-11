#pragma once
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>

namespace kadan::serving {
inline std::string trim_chat_text(std::string text) {
  const auto first = text.find_first_not_of(" \t\r\n");
  if (first == std::string::npos)
    return {};
  return text.substr(first, text.find_last_not_of(" \t\r\n") - first + 1);
}

// Text-only forms of the two checkpoint templates used by the API. Tools and
// multimodal messages are not part of this endpoint's supported request model.
inline std::string chat_prompt(const nlohmann::json &messages, bool glm,
                               std::size_t byte_limit) {
  if (!messages.is_array() || messages.empty() || messages.size() > byte_limit)
    throw std::invalid_argument("chat_messages");
  std::string result = glm ? "[gMASK]<sop><|system|>Reasoning Effort: Max" : "";
  bool user_seen = false;
  for (std::size_t index = 0; index < messages.size(); ++index) {
    const auto &message = messages[index];
    const auto role = message.at("role").get<std::string>();
    auto text = message.at("text").get<std::string>();
    if (role != "system" && role != "user" && role != "assistant")
      throw std::invalid_argument("chat_role");
    if (!glm && role == "system" && index != 0)
      throw std::invalid_argument("chat_system_position");
    user_seen = user_seen || role == "user";
    if (!glm)
      text = trim_chat_text(std::move(text));
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
      if (glm)
        text = "<think>" + reasoning + "</think>" +
               trim_chat_text(std::move(text));
      else
        text = "<think>\n" + trim_chat_text(std::move(reasoning)) +
               "\n</think>\n\n" + trim_chat_text(std::move(text));
    }
    result += glm ? "<|" + role + "|>" + text
                  : "<|im_start|>" + role + "\n" + text + "<|im_end|>\n";
    if (result.size() > byte_limit)
      throw std::invalid_argument("chat_prompt_bytes");
  }
  if (!user_seen)
    throw std::invalid_argument("chat_user_required");
  result += glm ? "<|assistant|><think>"
                : "<|im_start|>assistant\n<think>\n\n</think>\n\n";
  return result;
}
} // namespace kadan::serving
