#pragma once
#include <stdexcept>
#include <string>

namespace kadan::s1 {
inline constexpr std::size_t max_transcript_tokens = 1000, max_output_tokens = 1024;
inline constexpr std::size_t context_tokens = 4096, max_transcript_bytes = 32000;

// Pinned S1-mini input contract: prose/general, semi-formal, thinking disabled.
// https://huggingface.co/superwhisper/s1-mini/tree/88f6b15896c73bbb13a3b596e0afe8ea0d5150b4
inline std::string prompt(const std::string& transcript) {
    if (transcript.empty() || transcript.size() > max_transcript_bytes)
        throw std::invalid_argument("Formatter transcript exceeds its input bound");
    return "<|im_start|>system\n"
           "You are a text normalizer for speech-to-text transcripts. The input begins "
           "with a control line specifying the styling, structure, and context "
           "settings; "
           "clean the transcript to match those settings and output only the cleaned "
           "text."
           "<|im_end|>\n<|im_start|>user\n"
           "[Styling: semi-formal] [Structure: prose] [Context: general]\n"
           + transcript + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
}
}  // namespace kadan::s1
