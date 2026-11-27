// Pure unit tests for kadan::text::sample_logits (no model load or inference).
#include "kadan/text_sampling.hpp"

#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <set>
#include <vector>

using kadan::text::sample_logits;
using kadan::text::SamplingConfig;
using kadan::text::splitmix64;

int main() {
    // 1. Default config is greedy and returns exact argmax.
    {
        SamplingConfig cfg;
        assert(cfg.is_greedy());
        std::vector<float> logits = {-2.0f, 3.5f, 1.0f, 3.5f};
        std::uint64_t rng = splitmix64(42);
        const unsigned chosen = sample_logits(logits, cfg, {}, rng);
        assert(chosen == 1u);
    }

    // 2. Presence penalty penalizes seen tokens once (regardless of repeat count)
    //    and respects penalty_window.
    {
        SamplingConfig cfg;
        cfg.temperature = 0.0f;
        cfg.presence_penalty = 1.5f;
        cfg.penalty_window = 2;
        assert(!cfg.is_greedy());

        // Token 0 is outside the 2-token window [1, 1], so only token 1 is penalized.
        std::vector<float> logits = {2.0f, 3.0f, 2.2f};
        const std::array<unsigned, 3> history = {0u, 1u, 1u};
        std::uint64_t rng = splitmix64(1);
        const unsigned chosen = sample_logits(logits, cfg, history, rng);
        // Token 1 drops from 3.0 -> 1.5; token 2 (2.2) wins over token 0 (2.0).
        assert(chosen == 2u);
    }

    // 3. Repetition penalty divides positive logits and multiplies negative logits.
    {
        SamplingConfig cfg;
        cfg.temperature = 0.0f;
        cfg.repetition_penalty = 2.0f;
        std::vector<float> logits = {-1.0f, -1.5f};
        const std::array<unsigned, 1> history = {0u};
        std::uint64_t rng = splitmix64(2);
        const unsigned chosen = sample_logits(logits, cfg, history, rng);
        // Token 0 becomes -2.0; token 1 (-1.5) wins.
        assert(chosen == 1u);
    }

    // 4. Deterministic seed produces identical samples; non-zero temperature explores.
    {
        SamplingConfig cfg;
        cfg.temperature = 1.0f;
        cfg.top_p = 1.0f;
        std::uint64_t rng_a = splitmix64(12345);
        std::uint64_t rng_b = splitmix64(12345);
        std::set<unsigned> seen;
        for (int i = 0; i < 64; ++i) {
            std::vector<float> la = {1.0f, 1.0f, 1.0f, 1.0f};
            std::vector<float> lb = la;
            const unsigned a = sample_logits(la, cfg, {}, rng_a);
            const unsigned b = sample_logits(lb, cfg, {}, rng_b);
            assert(a == b);
            seen.insert(a);
        }
        assert(seen.size() == 4u);
    }

    // 5. min_p filters low-probability tail tokens relative to p_max.
    {
        SamplingConfig cfg;
        cfg.temperature = 1.0f;
        cfg.min_p = 0.2f;
        std::uint64_t rng = splitmix64(77);
        // Token 0 has score 10.0, token 1 has score 9.0 (ratio e^-1 ~= 0.368 >= 0.2),
        // token 2 has score 7.0 (ratio e^-3 ~= 0.05 < 0.2 -> filtered out).
        for (int i = 0; i < 100; ++i) {
            std::vector<float> logits = {10.0f, 9.0f, 7.0f, 0.0f};
            const unsigned chosen = sample_logits(logits, cfg, {}, rng);
            assert(chosen == 0u || chosen == 1u);
        }
    }

    // 6. top_k restricts candidate pool strictly to top k tokens.
    {
        SamplingConfig cfg;
        cfg.temperature = 1.0f;
        cfg.top_k = 2;
        std::uint64_t rng = splitmix64(99);
        for (int i = 0; i < 100; ++i) {
            std::vector<float> logits = {1.0f, 5.0f, 4.9f, 4.8f, 4.7f};
            const unsigned chosen = sample_logits(logits, cfg, {}, rng);
            assert(chosen == 1u || chosen == 2u);
        }
    }

    // 7. top_p restricts cumulative mass while always keeping at least 1 token.
    {
        SamplingConfig cfg;
        cfg.temperature = 1.0f;
        cfg.top_p = 0.5f;
        std::uint64_t rng = splitmix64(101);
        for (int i = 0; i < 64; ++i) {
            // Token 3 alone has >95% mass -> top_p=0.5 keeps only token 3.
            std::vector<float> logits = {0.0f, 0.0f, 0.0f, 8.0f};
            const unsigned chosen = sample_logits(logits, cfg, {}, rng);
            assert(chosen == 3u);
        }
    }

    // 8. Large-vocab fast path (200k tokens) stays accurate and skips -inf/NaN.
    {
        SamplingConfig cfg;
        cfg.temperature = 0.7f;
        cfg.min_p = 0.05f;
        cfg.top_p = 0.95f;
        std::vector<float> logits(200000, -10.0f);
        logits[0] = std::numeric_limits<float>::quiet_NaN();
        logits[1] = -std::numeric_limits<float>::infinity();
        logits[123456] = 12.0f;
        logits[65432] = 11.8f;
        std::uint64_t rng = splitmix64(555);
        std::set<unsigned> seen;
        for (int i = 0; i < 32; ++i) {
            auto copy = logits;
            const unsigned chosen = sample_logits(copy, cfg, {}, rng);
            assert(chosen == 123456u || chosen == 65432u);
            seen.insert(chosen);
        }
        assert(seen.size() == 2u);
    }

    std::cout << "Text sampling units passed\n";
    return 0;
}
