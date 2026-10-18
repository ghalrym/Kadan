#pragma once
#include <array>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace kadan::drama {
inline std::string_view trim_prompt_space(std::string_view text) {
    constexpr std::array<std::string_view, 25> spaces = {
        "\t",           "\n",           "\v",
        "\f",           "\r",           " ",
        "\xc2\x85",     "\xc2\xa0",     "\xe1\x9a\x80",
        "\xe2\x80\x80", "\xe2\x80\x81", "\xe2\x80\x82",
        "\xe2\x80\x83", "\xe2\x80\x84", "\xe2\x80\x85",
        "\xe2\x80\x86", "\xe2\x80\x87", "\xe2\x80\x88",
        "\xe2\x80\x89", "\xe2\x80\x8a", "\xe2\x80\xa8",
        "\xe2\x80\xa9", "\xe2\x80\xaf", "\xe2\x81\x9f",
        "\xe3\x80\x80"
    };
    while (!text.empty()) {
        bool found = false;
        for (const auto space : spaces)
            if (text.starts_with(space)) {
                text.remove_prefix(space.size());
                found = true;
                break;
            }
        if (!found)
            break;
    }
    return text;
}

// Tokenize each segment separately: joining strings changes BPE at the seams.
// Matches upstream reference-bracket-v1 without voice-reference frames.
inline std::vector<std::string> prompt_segments(const std::string& text) {
    if (text.size() > 32000 || trim_prompt_space(text).empty())
        throw std::invalid_argument("drama_text_bounds");
    for (const auto token :
         {"<|im_start|>",
          "<|im_end|>",
          "<|voice|>",
          "<|text|>",
          "<|interleave|>",
          "<|semantic:"})
        if (text.find(token) != std::string::npos)
            throw std::invalid_argument("drama_reserved_prompt_token");
    for (auto at = text.find("<|speaker"); at != std::string::npos;
         at = text.find("<|speaker", at + 1))
        if (text.compare(at, 13, "<|speaker:0|>") != 0)
            throw std::invalid_argument("drama_only_default_speaker_supported");
    const auto spoken = trim_prompt_space(text).starts_with("<|speaker:0|>")
                            ? text
                            : "<|speaker:0|> " + text;
    return {
        "<|im_start|>system\n",
        "convert the provided text to speech",
        "<|im_end|>\n",
        "<|im_start|>user\n",
        spoken,
        "<|im_end|>\n",
        "<|im_start|>assistant\n<|voice|>"
    };
}
}  // namespace kadan::drama
