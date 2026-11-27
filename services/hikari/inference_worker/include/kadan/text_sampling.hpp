#pragma once
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace kadan::text {

struct SamplingConfig {
    float temperature = 0.0f;
    float top_p = 1.0f;
    std::size_t top_k = 0;
    float min_p = 0.0f;
    float repetition_penalty = 1.0f;
    float presence_penalty = 0.0f;
    std::size_t penalty_window = 256;
    std::optional<std::uint64_t> seed;

    bool has_penalties() const noexcept {
        return repetition_penalty != 1.0f || presence_penalty != 0.0f;
    }

    bool is_greedy() const noexcept {
        return temperature <= 0.0f && !has_penalties();
    }
};

inline std::uint64_t splitmix64(std::uint64_t x) noexcept {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

inline double next_unit_double(std::uint64_t& state) noexcept {
    state = splitmix64(state);
    // 53-bit uniform double in [0, 1).
    return static_cast<double>(state >> 11) * (1.0 / 9007199254740992.0);
}

inline unsigned sample_logits(
    std::span<float> logits,
    const SamplingConfig& config,
    std::span<const unsigned> recent_tokens,
    std::uint64_t& rng_state
) {
    if (logits.empty())
        throw std::runtime_error("text_sampling_empty_logits");

    if (config.has_penalties() && !recent_tokens.empty()) {
        const std::size_t start =
            (config.penalty_window > 0 && recent_tokens.size() > config.penalty_window)
                ? recent_tokens.size() - config.penalty_window
                : 0;
        std::vector<unsigned> seen(
            recent_tokens.begin() + static_cast<std::ptrdiff_t>(start),
            recent_tokens.end()
        );
        std::sort(seen.begin(), seen.end());
        seen.erase(std::unique(seen.begin(), seen.end()), seen.end());
        for (const unsigned id : seen) {
            if (id < logits.size() && std::isfinite(logits[id])) {
                if (config.repetition_penalty != 1.0f) {
                    logits[id] = logits[id] < 0.0f
                                     ? logits[id] * config.repetition_penalty
                                     : logits[id] / config.repetition_penalty;
                }
                if (config.presence_penalty != 0.0f) {
                    logits[id] -= config.presence_penalty;
                }
            }
        }
    }

    if (config.temperature <= 0.0f) {
        std::optional<unsigned> best;
        float best_val = -std::numeric_limits<float>::infinity();
        for (std::size_t i = 0; i < logits.size(); ++i) {
            if (std::isfinite(logits[i]) && (!best || logits[i] > best_val)) {
                best_val = logits[i];
                best = static_cast<unsigned>(i);
            }
        }
        if (!best)
            throw std::runtime_error("text_sampling_nonfinite_logits");
        return *best;
    }

    const float inv_temp = 1.0f / config.temperature;
    float max_scaled = -std::numeric_limits<float>::infinity();
    bool has_finite = false;
    for (float v : logits) {
        if (std::isfinite(v)) {
            const float s = v * inv_temp;
            if (!has_finite || s > max_scaled) {
                max_scaled = s;
                has_finite = true;
            }
        }
    }
    if (!has_finite)
        throw std::runtime_error("text_sampling_nonfinite_logits");

    // Pre-filter in log-space before sorting: min_p is mathematically exact
    // (exp(s - max_s) >= min_p <=> s >= max_s + log(min_p)), and on large
    // vocabularies without top_k we skip negligible tail entries (< exp(-25)).
    float cutoff = -std::numeric_limits<float>::infinity();
    if (config.min_p > 0.0f) {
        cutoff = max_scaled + std::log(config.min_p);
    } else if (config.top_k == 0 && logits.size() > 4096) {
        cutoff = max_scaled - 25.0f;
    }

    std::vector<std::pair<float, unsigned>> candidates;
    candidates.reserve(std::min<std::size_t>(logits.size(), 1024));
    for (std::size_t i = 0; i < logits.size(); ++i) {
        if (!std::isfinite(logits[i]))
            continue;
        const float s = logits[i] * inv_temp;
        if (s >= cutoff) {
            candidates.emplace_back(s, static_cast<unsigned>(i));
        }
    }
    if (candidates.empty())
        throw std::runtime_error("text_sampling_no_candidates");

    auto by_score = [](const auto& a, const auto& b) {
        if (a.first != b.first)
            return a.first > b.first;
        return a.second < b.second;
    };
    const std::size_t k = (config.top_k > 0 && config.top_k < candidates.size())
                              ? config.top_k
                              : candidates.size();
    if (k < candidates.size()) {
        std::partial_sort(
            candidates.begin(),
            candidates.begin() + static_cast<std::ptrdiff_t>(k),
            candidates.end(),
            by_score
        );
        candidates.resize(k);
    } else {
        std::sort(candidates.begin(), candidates.end(), by_score);
    }

    const float top_scaled = candidates.front().first;
    std::vector<double> probs(candidates.size());
    double total = 0.0;
    for (std::size_t i = 0; i < candidates.size(); ++i) {
        probs[i] = std::exp(static_cast<double>(candidates[i].first - top_scaled));
        total += probs[i];
    }
    if (total <= 0.0 || !std::isfinite(total))
        return candidates.front().second;
    for (double& p : probs)
        p /= total;

    std::size_t kept = probs.size();
    if (config.top_p > 0.0f && config.top_p < 1.0f) {
        double cumulative = 0.0;
        for (std::size_t i = 0; i < probs.size(); ++i) {
            cumulative += probs[i];
            if (cumulative >= static_cast<double>(config.top_p)) {
                kept = i + 1;
                break;
            }
        }
    }

    double kept_sum = 0.0;
    for (std::size_t i = 0; i < kept; ++i)
        kept_sum += probs[i];

    const double u = next_unit_double(rng_state) * kept_sum;
    double acc = 0.0;
    for (std::size_t i = 0; i < kept; ++i) {
        acc += probs[i];
        if (u < acc)
            return candidates[i].second;
    }
    return candidates[kept - 1].second;
}

}  // namespace kadan::text
