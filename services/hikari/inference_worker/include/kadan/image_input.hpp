#pragma once
#include <array>
#include <atomic>
#include <cmath>
#include <random>
#include <span>
#include <string>

#include "kadan/text_tokenizer.hpp"

namespace kadan::image {
struct ImagePrompt {
    std::array<std::uint32_t, 2048> tokens{};
    std::size_t count = 0, prefix = 0;
};

inline ImagePrompt prepare_image_prompt(
    text::Tokenizer& tokenizer,
    const std::string& prompt,
    const std::atomic_bool& cancel
) {
    constexpr const char* system =
        "<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n";
    auto raw = std::string(system) + "<|im_start|>user\n" + prompt
               + "<|im_end|>\n<|im_start|>assistant\n";
    ImagePrompt result;
    result.count = tokenizer.encode(raw, result.tokens, cancel);
    std::array<std::uint32_t, 64> prefix{};
    result.prefix = tokenizer.encode(system, prefix, cancel);
    if (result.prefix != 14 || result.count <= result.prefix)
        throw std::runtime_error("image_prompt_prefix");
    return result;
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
