#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "kadan/text_tokenizer.hpp"

namespace kadan::text {

using Json = nlohmann::json;

struct DecisionQuestion {
    enum class Kind {
        choice,
        score,
        noul
    };
    std::string key;
    Kind kind = Kind::choice;
    std::string instructions;
    // Choice: ordered (option_key, description).
    std::vector<std::pair<std::string, std::string>> options;
    // Score: 2..10 rubric level descriptions.
    std::vector<std::string> levels;
    // Noul: optional guidance.
    std::string true_when;
    std::string false_when;
};

struct DecisionPrompt {
    std::string text;
    std::vector<std::pair<std::size_t, std::size_t>> literal_ranges;
};

struct PreparedDecisionQuestion {
    std::vector<std::uint32_t> tokens;
    std::vector<std::vector<std::uint32_t>> readout_groups;
};

// Assigns short single-letter (A..Z) or multi-letter option codes for Choice questions.
std::vector<std::string> decision_option_codes(const std::vector<std::string>& labels);

// Resolves each Choice option to a unique single-token alias in `tokenizer`.
std::vector<std::pair<std::string, std::uint32_t>> resolve_choice_aliases(
    Tokenizer& tokenizer,
    const std::vector<std::string>& labels,
    const std::atomic_bool& cancel
);

// Renders a single decision question using Inkling 2 Mini's native chat template
// ending directly at <|message_model|><|content_text|> with zero thinking tokens.
// User-supplied text is tracked in `literal_ranges` so control tokens cannot be injected.
DecisionPrompt render_inkling_decision_prompt(
    std::string_view state,
    const DecisionQuestion& question,
    std::span<const std::string> choice_codes = {}
);

// Builds candidate token-ID groups (e.g. yes/no variants, option codes with optional
// leading space, or score digits 0..K-1) for single-forward-pass logit readout.
std::vector<std::vector<std::uint32_t>> decision_readout_groups(
    Tokenizer& tokenizer,
    const DecisionQuestion& question,
    std::span<const std::pair<std::string, std::uint32_t>> resolved_choices,
    const std::atomic_bool& cancel
);

// Numerically stable temperature-scaled softmax over group log-probabilities.
std::vector<double> softmax_scores(std::span<const double> scores, double temperature = 1.0);

// Evaluates a single question's next-token vocabulary logits via log-softmax over
// candidate token groups and returns the typed DecisionAnswer JSON object.
Json score_decision_logits(
    const DecisionQuestion& question,
    std::span<const float> logits,
    const std::vector<std::vector<std::uint32_t>>& groups,
    double temperature = 1.0
);

// Returns the longest common token prefix shared by all sequences (strictly less
// than the shortest sequence so every sequence retains at least one suffix token
// for next-token logit evaluation). Returns 0 when fewer than two sequences exist.
std::size_t shared_token_prefix_length(
    std::span<const PreparedDecisionQuestion> questions
);

// Renders, tokenizes, and resolves readout groups for 1..8 decision questions.
// Rejects any sequence exceeding `max_context_tokens` or combined total exceeding
// `max_total_tokens` without truncation.
std::vector<PreparedDecisionQuestion> prepare_inkling_decisions(
    std::string_view state,
    std::span<const DecisionQuestion> questions,
    Tokenizer& tokenizer,
    std::size_t max_context_tokens,
    std::size_t max_total_tokens,
    const std::atomic_bool& cancel
);

}  // namespace kadan::text
