#include "kadan/lfm3.hpp"

#include <algorithm>
#include <cctype>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace kadan::lfm3 {
namespace {
std::string trim_copy(std::string_view s) {
    while (!s.empty() && std::isspace(static_cast<unsigned char>(s.front())))
        s.remove_prefix(1);
    while (!s.empty() && std::isspace(static_cast<unsigned char>(s.back())))
        s.remove_suffix(1);
    return std::string(s);
}

std::string render_value(const Json& v) {
    if (v.is_string())
        return v.get<std::string>();
    if (v.is_boolean())
        return v.get<bool>() ? "true" : "false";
    if (v.is_null())
        return "null";
    return v.dump();
}

std::string tool_to_json(const Json& tool) {
    const Json& fn = tool.contains("function") ? tool.at("function") : tool;
    std::string out = "{";
    bool first = true;
    for (auto it = fn.begin(); it != fn.end(); ++it) {
        if (it.key() == "defer_loading")
            continue;
        if (!first)
            out += ", ";
        first = false;
        out += Json(it.key()).dump() + ": " + it.value().dump();
    }
    out += "}";
    return out;
}

std::string render_tool_calls(const Json& calls) {
    std::string out;
    for (std::size_t i = 0; i < calls.size(); ++i) {
        if (i > 0)
            out += "\n\n";
        const auto& tc = calls[i];
        const Json& fn = tc.contains("function") ? tc.at("function") : tc;
        out += "<tool_call>\n<function=" + fn.at("name").get<std::string>() + ">\n";
        if (fn.contains("arguments") && fn.at("arguments").is_object()) {
            for (auto it = fn.at("arguments").begin(); it != fn.at("arguments").end();
                 ++it) {
                out += "<parameter=" + it.key() + ">\n" + render_value(it.value())
                       + "\n</parameter>\n";
            }
        }
        out += "</function>\n</tool_call>";
    }
    return out;
}

std::string render_tool_response(const std::string& content) {
    if (content.find("<tool_response>") != std::string::npos)
        return content;
    return "<tool_response>\n" + content + "\n</tool_response>";
}

std::string strip_think_blocks(std::string_view input) {
    std::string out;
    std::size_t pos = 0;
    while (pos < input.size()) {
        const auto open = input.find("<think>", pos);
        if (open == std::string_view::npos) {
            out.append(input.substr(pos));
            break;
        }
        out.append(input.substr(pos, open - pos));
        const auto close = input.find("</think>", open + 7);
        if (close == std::string_view::npos) {
            out.append(input.substr(open));
            break;
        }
        pos = close + 8;
    }
    return out;
}

void erase_all(std::string& text, std::string_view marker) {
    std::size_t pos = 0;
    while ((pos = text.find(marker, pos)) != std::string::npos)
        text.erase(pos, marker.size());
}

std::vector<std::string> split_long_chunk(std::string_view text, std::size_t limit) {
    std::vector<std::string> out;
    std::string remaining = trim_copy(text);
    while (remaining.size() > limit) {
        std::size_t cut = remaining.rfind(' ', limit);
        if (cut == std::string::npos || cut == 0)
            cut = limit;
        std::string head = trim_copy(std::string_view(remaining).substr(0, cut));
        if (!head.empty())
            out.push_back(std::move(head));
        remaining = trim_copy(std::string_view(remaining).substr(cut));
    }
    if (!remaining.empty())
        out.push_back(std::move(remaining));
    return out;
}
}  // namespace

void ReasoningBudget::apply(
    std::size_t generated_so_far,
    std::optional<std::uint32_t> last_token,
    std::span<float> logits
) {
    if (!active || closed || max_reasoning_tokens == 0)
        return;
    if (generated_so_far > 0 && last_token && *last_token == end_think_token) {
        closed = true;
        return;
    }
    if (generated_so_far >= max_reasoning_tokens && end_think_token < logits.size()) {
        std::fill(
            logits.begin(), logits.end(), -std::numeric_limits<float>::infinity()
        );
        logits[end_think_token] = 0.0f;
    }
}

std::string render_lfm3_chat_prompt(
    const std::vector<Lfm3ChatMessage>& messages,
    const Json& tools,
    bool enable_thinking,
    bool preserve_thinking
) {
    std::string out = "<|startoftext|>";
    std::size_t start_idx = 0;
    std::string system_prompt;
    if (!messages.empty() && messages.front().role == "system") {
        system_prompt = messages.front().content;
        start_idx = 1;
    }
    if (tools.is_array() && !tools.empty()) {
        std::vector<std::string> lines;
        for (const auto& tool : tools) {
            const Json& fn = tool.contains("function") ? tool.at("function") : tool;
            const bool deferred =
                fn.value("defer_loading", false) || tool.value("defer_loading", false);
            if (!deferred)
                lines.push_back(tool_to_json(fn));
        }
        if (!lines.empty()) {
            if (!system_prompt.empty())
                system_prompt += "\n\n";
            system_prompt += "<tools>\n";
            for (std::size_t i = 0; i < lines.size(); ++i) {
                if (i > 0)
                    system_prompt += '\n';
                system_prompt += lines[i];
            }
            system_prompt += "\n</tools>";
        }
    }
    if (!system_prompt.empty())
        out += "<|im_start|>system\n" + system_prompt + "<|im_end|>\n";

    std::ptrdiff_t last_user = -1;
    for (std::size_t i = start_idx; i < messages.size(); ++i) {
        if (messages[i].role == "user")
            last_user = static_cast<std::ptrdiff_t>(i);
    }

    std::string merged_tool_content;
    for (std::size_t i = start_idx; i < messages.size(); ++i) {
        const auto& msg = messages[i];
        const bool merge_next =
            msg.role == "tool" && !msg.tool_call_id.empty() && i + 1 < messages.size()
            && messages[i + 1].role == "tool"
            && messages[i + 1].tool_call_id == msg.tool_call_id;
        if (merge_next) {
            if (!msg.content.empty()) {
                if (!merged_tool_content.empty())
                    merged_tool_content += '\n';
                merged_tool_content += msg.content;
            }
            continue;
        }
        out += "<|im_start|>" + msg.role + "\n";
        if (msg.role == "assistant") {
            const bool keep_thinking =
                preserve_thinking || static_cast<std::ptrdiff_t>(i) > last_user;
            std::string content = msg.content;
            if (keep_thinking) {
                if (content.find("</think>") == std::string::npos)
                    out += "<think></think>";
            } else {
                const auto close = content.rfind("</think>");
                if (close != std::string::npos)
                    content = trim_copy(std::string_view(content).substr(close + 8));
                content = "<think></think>" + content;
            }
            out += content;
            if (msg.tool_calls.is_array() && !msg.tool_calls.empty())
                out += render_tool_calls(msg.tool_calls);
            out += "<|im_end|>\n";
        } else if (msg.role == "tool") {
            std::string body = merged_tool_content;
            if (!msg.content.empty()) {
                if (!body.empty())
                    body += '\n';
                body += msg.content;
            }
            merged_tool_content.clear();
            out += render_tool_response(body) + "<|im_end|>\n";
        } else {
            out += msg.content + "<|im_end|>\n";
        }
    }

    out += "<|im_start|>assistant\n";
    out += enable_thinking ? "<think>" : "<think></think>";
    return out;
}

std::string extract_lfm3_reply_text(std::string_view raw_generated) {
    const auto close = raw_generated.rfind("</think>");
    if (close == std::string_view::npos)
        throw std::runtime_error("LFM3 ended inside an unfinished reasoning block.");
    std::string text = trim_copy(raw_generated.substr(close + 8));
    if (text.find("<think>") != std::string::npos)
        throw std::runtime_error("LFM3 ended inside an unfinished reasoning block.");
    bool stripped = true;
    while (stripped) {
        stripped = false;
        for (std::string_view ending : {"<|im_end|>", "<|endoftext|>"}) {
            if (text.size() >= ending.size()
                && text.compare(text.size() - ending.size(), ending.size(), ending)
                       == 0) {
                text.resize(text.size() - ending.size());
                text = trim_copy(text);
                stripped = true;
            }
        }
    }
    if (text.find("<|im_start|>") != std::string::npos)
        throw std::runtime_error("LFM3 returned an unexpected extra conversation turn.");
    return text;
}

std::string extract_lfm3_vision_description(std::string_view raw_generated) {
    std::string text = trim_copy(strip_think_blocks(raw_generated));
    if (const auto close = text.rfind("</think>"); close != std::string::npos)
        text = trim_copy(std::string_view(text).substr(close + 8));
    bool stripped = true;
    while (stripped) {
        stripped = false;
        for (std::string_view ending : {"<|im_end|>", "<|endoftext|>"}) {
            if (text.size() >= ending.size()
                && text.compare(text.size() - ending.size(), ending.size(), ending)
                       == 0) {
                text.resize(text.size() - ending.size());
                text = trim_copy(text);
                stripped = true;
            }
        }
    }
    if (text.find("<think>") != std::string::npos || text.empty())
        throw std::runtime_error("The vision model did not complete a screenshot description.");
    if (text.find("<|im_start|>") != std::string::npos)
        throw std::runtime_error("The vision model generated another conversation turn.");
    return text;
}

std::string validate_audio_public_text(std::string_view raw, bool transcribe) {
    std::string text = strip_think_blocks(raw);
    if (text.find("<think>") != std::string::npos
        || text.find("</think>") != std::string::npos) {
        throw std::runtime_error("Audio model returned unfinished reasoning");
    }
    if (text != raw && !transcribe)
        throw std::runtime_error("Audio model returned reasoning in a spoken reply");
    for (std::string_view marker :
         {"<|audio_out|>",
          "<|audio_start|>",
          "<|text_end|>",
          "<|mixed_start|>",
          "<|mixed_end|>",
          "<|im_end|>",
          "<|endoftext|>"}) {
        erase_all(text, marker);
    }
    for (std::string_view bad :
         {"<|",
          "<tool_call",
          "</tool_call",
          "<function",
          "<parameter",
          "<tool_response",
          "</tool_response"}) {
        if (text.find(bad) != std::string::npos)
            throw std::runtime_error("Audio model returned tool or unknown control output");
    }
    return trim_copy(text);
}

std::vector<std::string> split_speech_scripts(std::string_view text) {
    // Matches experimental_worker/speech_segments.py: sentence splits capped at 240 chars.
    constexpr std::size_t kMaxChars = 240;
    std::string cleaned;
    bool in_space = false;
    for (char ch : text) {
        if (std::isspace(static_cast<unsigned char>(ch))) {
            if (!cleaned.empty() && !in_space) {
                cleaned.push_back(' ');
                in_space = true;
            }
        } else {
            cleaned.push_back(ch);
            in_space = false;
        }
    }
    cleaned = trim_copy(cleaned);
    if (cleaned.empty())
        return {};

    std::vector<std::string> sentences;
    std::size_t start = 0;
    for (std::size_t i = 0; i < cleaned.size(); ++i) {
        const char c = cleaned[i];
        const bool end_punct = (c == '.' || c == '!' || c == '?');
        if (end_punct && (i + 1 == cleaned.size() || cleaned[i + 1] == ' ')) {
            std::string seg =
                trim_copy(std::string_view(cleaned).substr(start, i - start + 1));
            if (!seg.empty())
                sentences.push_back(std::move(seg));
            start = i + 1;
            while (start < cleaned.size() && cleaned[start] == ' ')
                ++start;
        }
    }
    if (start < cleaned.size()) {
        std::string tail = trim_copy(std::string_view(cleaned).substr(start));
        if (!tail.empty())
            sentences.push_back(std::move(tail));
    }

    std::vector<std::string> result;
    for (const auto& sentence : sentences) {
        auto parts = split_long_chunk(sentence, kMaxChars);
        result.insert(
            result.end(),
            std::make_move_iterator(parts.begin()),
            std::make_move_iterator(parts.end())
        );
    }
    return result;
}

std::vector<Lfm3ChatMessage> append_call_history(
    const std::vector<Lfm3ChatMessage>& history,
    std::string_view user_text,
    std::size_t max_messages
) {
    std::vector<Lfm3ChatMessage> out;
    if (!history.empty() && history.front().role == "system")
        out.push_back(history.front());
    std::vector<Lfm3ChatMessage> non_system;
    for (std::size_t i = (!history.empty() && history.front().role == "system") ? 1 : 0;
         i < history.size();
         ++i) {
        non_system.push_back(history[i]);
    }
    non_system.push_back({"user", std::string(user_text), Json::array(), {}});
    const std::size_t keep = max_messages > out.size() ? max_messages - out.size() : 0;
    const std::size_t skip = non_system.size() > keep ? non_system.size() - keep : 0;
    for (std::size_t i = skip; i < non_system.size(); ++i)
        out.push_back(std::move(non_system[i]));
    return out;
}

}  // namespace kadan::lfm3
