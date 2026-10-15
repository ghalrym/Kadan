#pragma once
#include <array>
#include <stdexcept>
#include <string>
#include <string_view>

#include "kadan/inference_request.hpp"

namespace kadan::serving::muse_spark {
// Public mlx-community/Muse-Spark-Open-816B-A42B-3bit tokenizer.json,
// revision b7521af78bddb86c0de111e48fe93ff549711de9. Its generation_config
// terminates on end_of_text or eot; eom continues the assistant's turn.
enum Token : unsigned {
    begin_of_text = 200000,
    end_of_text = 200001,
    eom = 200007,
    eot = 200008,
    pad = 200018,
    start = 200022,
    message = 200023
};

constexpr bool is_stop(unsigned token) {
    return token == end_of_text || token == eot;
}

// Worker generation allowance beyond the visible answer limit, matching its
// existing reasoning policy. Template strength 512 is not a token count.
constexpr std::size_t reasoning_tokens = 2048;

// Text-only generation prompt from the public Hub API's chat_template_jinja:
// meta-models/Muse-Spark-Open-816B-A42B, revision
// 0fb02478b02565b2ce7fc57e6051c91172774ac9. Preserve its xhigh default (512).
// ChatMessage has no tools, media, recipients, timestamps or separate reasoning.
// This renderer does not establish tokenizer, checkpoint or inference support.
inline std::string prompt(
    const std::vector<ChatMessage>& messages, std::string_view date, std::size_t limit
) {
    if (messages.empty() || messages.size() > limit)
        throw std::invalid_argument("chat_messages");
    const auto append = [limit](std::string& out, std::string_view text) {
        if (text.size() > limit - out.size())
            throw std::invalid_argument("chat_prompt_bytes");
        out += text;
    };
    const auto escape = [&](std::string_view text, bool instructions = false) {
        std::string out;
        for (std::size_t i = 0; i < text.size();) {
            auto rest = text.substr(i);
            bool replaced = false;
            for (const auto value : {"<|", "<image>", "<video>", "<audio>"}) {
                const std::string_view token(value);
                if (rest.starts_with(token)) {
                    append(out, "< ");
                    append(out, token.substr(1));
                    i += token.size();
                    replaced = true;
                    break;
                }
            }
            if (replaced)
                continue;
            if (instructions) {
                constexpr std::array<std::string_view, 4> from{
                    "Reasoning effort",
                    "Reasoning Effort",
                    "reasoning effort",
                    "REASONING EFFORT"
                };
                constexpr std::array<std::string_view, 4> to{
                    "Reasoning strength",
                    "Reasoning Strength",
                    "reasoning strength",
                    "REASONING STRENGTH"
                };
                for (std::size_t j = 0; j < from.size(); ++j)
                    if (rest.starts_with(from[j])) {
                        append(out, to[j]);
                        i += from[j].size();
                        replaced = true;
                        break;
                    }
            }
            if (!replaced) {
                append(out, text.substr(i, 1));
                ++i;
            }
        }
        return out;
    };
    bool user_seen = false;
    for (const auto& message : messages) {
        if (message.role != "system" && message.role != "user"
            && message.role != "assistant")
            throw std::invalid_argument("chat_role");
        user_seen = user_seen || message.role == "user";
        if (message.text.size() > limit)
            throw std::invalid_argument("chat_prompt_bytes");
    }
    if (!user_seen)
        throw std::invalid_argument("chat_user_required");
    std::string result;
    append(
        result,
        "<|begin_of_text|><|start|>system<|message|>You are a helpful AI "
        "assistant.\n\nKnowledge cutoff: 2026-01-04."
    );
    if (!date.empty()) {
        append(result, "\n\nCurrent date: ");
        append(result, date);
        append(result, ".");
    }
    const bool initial_system = messages.front().role == "system";
    const auto instructions =
        initial_system ? escape(messages.front().text, true) : std::string{};
    if (!instructions.empty()) {
        append(result, "\n\n");
        append(result, instructions);
    }
    auto lowered = instructions;
    for (auto& ch : lowered)
        if (ch >= 'A' && ch <= 'Z')
            ch += 'a' - 'A';
    if (lowered.find("reasoning strength:") == std::string::npos)
        append(result, "\n\nReasoning strength: 512.");
    append(
        result,
        "\n\nUse the appropriate recipient for each message:\n\n"
        "- \"self\": private reasoning and tool planning.\n"
        "- \"commentary\": user-visible intermediate updates while the assistant "
        "will continue working, including messages sent to the user before or "
        "between tool calls.\n"
        "- \"user\": messages that end the assistant turn, such as a completed "
        "response or clarification question that waits for the user's reply; "
        "do not use for pre-tool updates or partial responses.\n\n"
        "# Valid recipients: \"self\", \"commentary\", \"user\".<|eot|>"
    );
    for (std::size_t i = initial_system ? 1 : 0; i < messages.size(); ++i) {
        const auto& message = messages[i];
        append(result, "<|start|>");
        append(result, message.role);
        if (message.role == "assistant")
            append(result, " to=user");
        append(result, "<|message|>");
        append(result, escape(message.text));
        append(result, "<|eot|>");
    }
    append(result, "<|start|>assistant");
    return result;
}

// Consume the recipient messages specified by tokenizer_config.response_template.
// Text for self becomes a think block; only user/commentary count as answer
// tokens. Unknown recipients (including tools) and malformed headers stay hidden.
class Stream {
public:
    // decoded is one token's text; structural tokens are identified by id.
    std::string push(unsigned token, std::string_view decoded) {
        visible_ = false;
        if (state_ == State::done)
            return {};
        if (is_stop(token)) {
            state_ = State::done;
            return close();
        }
        if (token == start) {
            state_ = State::header;
            header_.clear();
            return close();
        }
        if (token == eom) {
            state_ = State::between;
            return close();
        }
        if (token == begin_of_text || token == pad)
            return {};
        if (state_ == State::header) {
            if (token == message)
                return open();
            // Do not accept a truncated valid recipient followed by overflow.
            if (decoded.size() > 512 - header_.size()) {
                header_.clear();
                state_ = State::between;
            } else
                header_ += decoded;
            return {};
        }
        if (state_ != State::content || token == message
            || recipient_ == Recipient::hidden)
            return {};
        visible_ = recipient_ == Recipient::visible && !decoded.empty();
        return std::string(decoded);
    }

    // Close reasoning even when generation stops at its token/context limit.
    std::string finish() {
        visible_ = false;
        state_ = State::done;
        return close();
    }

    bool visible() const {
        return visible_;
    }

private:
    enum class State {
        header,
        content,
        between,
        done
    };
    enum class Recipient {
        hidden,
        reasoning,
        visible
    };

    // The generation prompt has already emitted <|start|>assistant.
    State state_ = State::header;
    Recipient recipient_ = Recipient::hidden;
    std::string header_ = "assistant";
    bool thinking_ = false, visible_ = false;

    std::string close() {
        if (!thinking_)
            return {};
        thinking_ = false;
        return "</think>";
    }

    std::string open() {
        state_ = State::content;
        recipient_ = Recipient::hidden;
        if (header_ == "assistant to=self") {
            recipient_ = Recipient::reasoning;
            thinking_ = true;
            return "<think>";
        }
        if (header_ == "assistant to=user" || header_ == "assistant to=commentary")
            recipient_ = Recipient::visible;
        return {};
    }
};
}  // namespace kadan::serving::muse_spark
