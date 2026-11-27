#pragma once
#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "kadan/text_tokenizer.hpp"

namespace kadan::image {
inline constexpr std::size_t image_prompt_length = 512;
inline constexpr std::uint32_t image_prompt_pad_token = 151643;

struct ImagePrompt {
    std::array<std::uint32_t, image_prompt_length> tokens{};
    std::size_t active_count = 0;
};

inline std::string format_image_prompt(const std::string& prompt) {
    return "<|im_start|>user\n" + prompt
           + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
}

inline ImagePrompt pack_image_prompt(std::span<const std::uint32_t> encoded) {
    if (encoded.empty())
        throw std::runtime_error("image_prompt_empty");
    ImagePrompt result;
    result.tokens.fill(image_prompt_pad_token);
    result.active_count = std::min(encoded.size(), image_prompt_length);
    std::copy_n(encoded.begin(), result.active_count, result.tokens.begin());
    return result;
}

inline ImagePrompt prepare_image_prompt(
    text::Tokenizer& tokenizer,
    const std::string& prompt,
    const std::atomic_bool& cancel
) {
    const auto formatted = format_image_prompt(prompt);
    std::vector<std::uint32_t> encoded(4096);
    const auto count = tokenizer.encode(formatted, encoded, cancel);
    return pack_image_prompt({encoded.data(), count});
}

// Point-major MT19937_64 / Box–Muller is the image API's existing seed contract.
inline void seed_image_latent(
    std::span<float> latent, std::uint64_t seed, const std::atomic_bool& cancel
) {
    std::mt19937_64 rng(seed);
    auto uniform = [&]() {
        return (double(rng() >> 11) + .5) / 9007199254740992.;
    };
    for (std::size_t i = 0; i < latent.size(); i += 2) {
        if (i % 4096 == 0 && cancel.load())
            throw std::runtime_error("image_cancelled");
        double radius = std::sqrt(-2 * std::log(uniform())),
               angle = 6.2831853071795864769 * uniform();
        latent[i] = float(radius * std::cos(angle));
        if (i + 1 < latent.size())
            latent[i + 1] = float(radius * std::sin(angle));
    }
}
}  // namespace kadan::image
