#pragma once
#include <stdexcept>
#include <string>
#include <string_view>

#include "kadan/chat_prompt.hpp"

namespace kadan::serving::harmony {
// GPT-OSS Harmony special tokens from the checkpoint's tokenizer.json.
enum Token : unsigned {
    end_of_text = 199999,
    return_ = 200002,
    constrain = 200003,
    channel = 200005,
    start = 200006,
    end = 200007,
    message = 200008,
    call = 200012
};

constexpr bool is_stop(unsigned token) {
    return token == return_ || token == call || token == end_of_text;
}

// The checkpoint template's defaults, without tools. Low effort keeps the
// fixed 256-token answer budget usable without a separate reasoning control.
constexpr std::string_view reasoning_effort = "low";
// Generated analysis tokens allowed beyond the visible answer budget.
constexpr std::size_t reasoning_tokens = 2048;

// Text-only rendering matching OpenAI's openai-harmony encoder. System
// messages become developer instructions. Earlier assistant turns keep only
// their final answer: Harmony drops prior chain of thought during inference.
inline std::string prompt(
    const std::vector<ChatMessage>& messages, std::string_view date, std::size_t limit
) {
    if (messages.empty() || messages.size() > limit)
        throw std::invalid_argument("chat_messages");
    std::string result =
        "<|start|>system<|message|>You are ChatGPT, a large language model trained by "
        "OpenAI.\nKnowledge cutoff: 2024-06\nCurrent date: ";
    result += date;
    result += "\n\nReasoning: ";
    result += reasoning_effort;
    result +=
        "\n\n# Valid channels: analysis, commentary, final. Channel must be included "
        "for every message.<|end|>";
    bool user_seen = false;
    for (const auto& message : messages) {
        const auto& role = message.role;
        if (role == "system")
            result += "<|start|>developer<|message|># Instructions\n\n" + message.text
                      + "<|end|>";
        else if (role == "user") {
            user_seen = true;
            result += "<|start|>user<|message|>" + message.text + "<|end|>";
        } else if (role == "assistant") {
            auto text = message.text;
            const auto think = text.rfind("</think>");
            if (think != std::string::npos)
                text.erase(0, think + 8);
            result += "<|start|>assistant<|channel|>final<|message|>"
                      + trim_chat_text(std::move(text)) + "<|end|>";
        } else
            throw std::invalid_argument("chat_role");
        if (result.size() > limit)
            throw std::invalid_argument("chat_prompt_bytes");
    }
    if (!user_seen)
        throw std::invalid_argument("chat_user_required");
    return result + "<|start|>assistant";
}

// Converts generated Harmony messages into the API's text form: analysis
// becomes <think>...</think>, final and preamble commentary stay visible, and
// headers, tool-call bodies and control tokens are never emitted.
class Stream {
public:
    // decoded is the token's plain text; special tokens decode to nothing.
    std::string push(unsigned token, std::string_view decoded) {
        visible_ = false;
        switch (state_) {
        case State::header:
            return header(token, decoded);
        case State::content:
            return content(token, decoded);
        case State::idle:
            if (token == start)
                begin();
            return {};
        }
        return {};
    }

    // Closes reasoning cut off by a length limit.
    std::string finish() {
        state_ = State::idle;
        return close();
    }

    // The last pushed token was answer text counted against the output budget.
    bool visible() const {
        return visible_;
    }

private:
    enum class State {
        header,
        content,
        idle
    };

    // Prompts end with <|start|>assistant, so generation opens in a header.
    State state_ = State::header;
    std::string header_, channel_;
    bool thinking_ = false, tool_ = false, visible_ = false;

    void begin() {
        state_ = State::header;
        header_.clear();
    }

    std::string close() {
        if (!thinking_)
            return {};
        thinking_ = false;
        return "</think>";
    }

    std::string header(unsigned token, std::string_view decoded) {
        if (token == start) {
            begin();
            return {};
        }
        if (token == end || token == return_ || token == call || token == end_of_text) {
            state_ = State::idle;
            return {};
        }
        if (token == message)
            return open();
        // Bounded: a header is a role, recipient, channel and content type.
        if (header_.size() < 512)
            header_ += token == channel     ? "<|channel|>"
                       : token == constrain ? "<|constrain|>"
                                            : std::string(decoded);
        return {};
    }

    std::string open() {
        state_ = State::content;
        const auto at = header_.find("<|channel|>");
        channel_.clear();
        if (at != std::string::npos) {
            channel_ = header_.substr(at + 11);
            channel_ = channel_.substr(0, channel_.find_first_of(" <"));
        }
        tool_ = header_.find("to=") != std::string::npos;
        if (channel_ == "analysis" && !tool_) {
            if (thinking_)
                return {};
            thinking_ = true;
            return "<think>";
        }
        return close();
    }

    std::string content(unsigned token, std::string_view decoded) {
        if (token == end || token == return_ || token == call || token == end_of_text) {
            state_ = State::idle;
            return close();
        }
        if (token == start) {
            begin();
            return close();
        }
        if (token == channel || token == message || token == constrain || tool_)
            return {};
        if (channel_ == "analysis")
            return std::string(decoded);
        visible_ = true;
        return std::string(decoded);
    }
};
}  // namespace kadan::serving::harmony
