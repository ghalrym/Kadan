#pragma once
#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <random>
#include <span>
#include <string>
#include <vector>

#include "kadan/text_tokenizer.hpp"

namespace kadan::image {
struct ImagePrompt {
    std::array<std::uint32_t, 8192> tokens{};
    std::size_t count = 0, prefix = 0;
    std::size_t image_start = 0, image_count = 0, grid_h = 0, grid_w = 0;
};

// Qwen3-VL replaces the single image placeholder with one token per 2x2
// patch group. The vision encoder retains all four patches before its merger.
inline void
expand_image_prompt(ImagePrompt& prompt, std::size_t width, std::size_t height) {
    if (!width || !height || width > 3072 || height > 3072 || width % 32 || height % 32
        || prompt.count > prompt.tokens.size())
        throw std::runtime_error("image_reference_dimensions");
    constexpr std::uint32_t image_token = 151655;
    const auto end = prompt.tokens.begin() + prompt.count;
    const auto first = std::find(prompt.tokens.begin(), end, image_token);
    if (first == end || std::find(first + 1, end, image_token) != end)
        throw std::runtime_error("image_reference_placeholder");
    prompt.grid_h = height / 32;
    prompt.grid_w = width / 32;
    prompt.image_count = prompt.grid_h * prompt.grid_w;
    prompt.image_start = std::size_t(first - prompt.tokens.begin());
    if (prompt.image_count > prompt.tokens.size() - prompt.count + 1)
        throw std::runtime_error("image_reference_prompt_capacity");
    std::move_backward(first + 1, end, end + prompt.image_count - 1);
    std::fill_n(first, prompt.image_count, image_token);
    prompt.count += prompt.image_count - 1;
}

// Text positions advance by one. An image uses a single temporal position and
// its merged spatial grid, after which text resumes beyond the largest axis.
inline std::array<std::vector<std::int64_t>, 3>
image_text_positions(const ImagePrompt& prompt) {
    std::array<std::vector<std::int64_t>, 3> result;
    for (auto& axis : result)
        axis.resize(prompt.count);
    std::int64_t position = 0;
    for (std::size_t i = 0; i < prompt.count;) {
        if (prompt.image_count && i == prompt.image_start) {
            for (std::size_t y = 0; y < prompt.grid_h; ++y)
                for (std::size_t x = 0; x < prompt.grid_w; ++x) {
                    result[0][i] = position;
                    result[1][i] = position + y;
                    result[2][i++] = position + x;
                }
            position += std::max(prompt.grid_h, prompt.grid_w);
        } else {
            for (auto& axis : result)
                axis[i] = position;
            ++i;
            ++position;
        }
    }
    return result;
}

inline ImagePrompt prepare_image_prompt(
    text::Tokenizer& tokenizer,
    const std::string& prompt,
    const std::atomic_bool& cancel,
    std::size_t reference_width = 0,
    std::size_t reference_height = 0
) {
    constexpr const char* system =
        "<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n";
    const auto reference =
        reference_width ? "<image1><|vision_start|><|image_pad|><|vision_end|>" : "";
    auto raw = std::string(system) + "<|im_start|>user\n" + reference + prompt
               + "<|im_end|>\n<|im_start|>assistant\n";
    ImagePrompt result;
    result.count = tokenizer.encode(raw, result.tokens, cancel);
    std::array<std::uint32_t, 64> prefix{};
    result.prefix = tokenizer.encode(system, prefix, cancel);
    if (result.prefix != 14 || result.count <= result.prefix)
        throw std::runtime_error("image_prompt_prefix");
    if (reference_width)
        expand_image_prompt(result, reference_width, reference_height);
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
