#include "kadan/text_decision.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <limits>
#include <set>
#include <stdexcept>

namespace kadan::text {
namespace {

void check(bool ok, const char* message) {
    if (!ok)
        throw std::runtime_error(message);
}

void check_cancel(const std::atomic_bool& cancel) {
    if (cancel.load(std::memory_order_relaxed))
        throw std::runtime_error("request_cancelled");
}

std::string replace_underscores(std::string_view text) {
    std::string out(text);
    for (char& c : out)
        if (c == '_')
            c = ' ';
    return out;
}

std::vector<std::uint32_t> encode_short(
    Tokenizer& tokenizer, const std::string& text, const std::atomic_bool& cancel
) {
    if (text.empty())
        return {};
    std::vector<std::uint32_t> buf(std::max<std::size_t>(64, text.size() * 2));
    const std::size_t n =
        tokenizer.encode(text, buf, cancel, std::max<std::size_t>(32000, text.size() + 64));
    buf.resize(n);
    return buf;
}

std::optional<std::uint32_t> single_token_id(
    Tokenizer& tokenizer, const std::string& raw, const std::atomic_bool& cancel
) {
    try {
        auto ids = encode_short(tokenizer, raw, cancel);
        if (ids.size() == 1)
            return ids[0];
    } catch (...) {
    }
    return std::nullopt;
}

void append_unique(std::vector<std::uint32_t>& target, std::uint32_t id) {
    if (std::find(target.begin(), target.end(), id) == target.end())
        target.push_back(id);
}

}  // namespace

std::vector<std::string> decision_option_codes(const std::vector<std::string>& labels) {
    bool all_single_alpha = !labels.empty();
    std::vector<std::string> trimmed;
    trimmed.reserve(labels.size());
    for (const auto& lab : labels) {
        std::string t = lab;
        while (!t.empty() && std::isspace(static_cast<unsigned char>(t.front())))
            t.erase(t.begin());
        while (!t.empty() && std::isspace(static_cast<unsigned char>(t.back())))
            t.pop_back();
        if (t.size() != 1 || !std::isalpha(static_cast<unsigned char>(t[0])))
            all_single_alpha = false;
        trimmed.push_back(std::move(t));
    }
    if (all_single_alpha) {
        // Ensure uniqueness before reusing raw single-letter labels.
        std::set<std::string> seen(trimmed.begin(), trimmed.end());
        if (seen.size() == trimmed.size())
            return trimmed;
    }
    std::vector<std::string> out;
    out.reserve(labels.size());
    if (labels.size() <= 26) {
        for (std::size_t i = 0; i < labels.size(); ++i)
            out.emplace_back(1, static_cast<char>('A' + i));
        return out;
    }
    for (char c = 'A'; c <= 'Z' && out.size() < labels.size(); ++c)
        out.emplace_back(1, c);
    for (char a = 'A'; a <= 'Z' && out.size() < labels.size(); ++a)
        for (char b = 'A'; b <= 'Z' && out.size() < labels.size(); ++b)
            out.push_back(std::string{a, b});
    return out;
}

std::vector<std::pair<std::string, std::uint32_t>> resolve_choice_aliases(
    Tokenizer& tokenizer,
    const std::vector<std::string>& labels,
    const std::atomic_bool& cancel
) {
    auto codes = decision_option_codes(labels);
    std::vector<std::string> fallback;
    for (char c = 'A'; c <= 'Z'; ++c)
        fallback.emplace_back(1, c);
    for (int i = 0; i < 100; ++i) {
        char buf[8];
        std::snprintf(buf, sizeof(buf), "%02d", i);
        fallback.emplace_back(buf);
    }
    for (char c = 'a'; c <= 'z'; ++c)
        fallback.emplace_back(1, c);
    for (int i = 0; i < 200; ++i) {
        char buf[16];
        std::snprintf(buf, sizeof(buf), "#%d", i);
        fallback.emplace_back(buf);
    }
    for (char a = 'A'; a <= 'Z'; ++a)
        for (char b = 'A'; b <= 'Z'; ++b)
            fallback.push_back(std::string{a, b});

    std::set<std::uint32_t> used;
    std::vector<std::pair<std::string, std::uint32_t>> out;
    out.reserve(labels.size());
    auto take = [&](const std::string& raw) -> bool {
        auto id = single_token_id(tokenizer, raw, cancel);
        if (!id || used.contains(*id))
            return false;
        used.insert(*id);
        out.emplace_back(raw, *id);
        return true;
    };
    for (const auto& code : codes) {
        check_cancel(cancel);
        if (take(code))
            continue;
        bool ok = false;
        for (const auto& cand : fallback) {
            if (take(cand)) {
                ok = true;
                break;
            }
        }
        check(ok, "decision_no_single_token_alias");
    }
    return out;
}

DecisionPrompt render_inkling_decision_prompt(
    std::string_view state,
    const DecisionQuestion& q,
    std::span<const std::string> choice_codes
) {
    std::string body;
    if (q.kind == DecisionQuestion::Kind::choice) {
        std::vector<std::string> fallback_codes;
        if (choice_codes.empty()) {
            std::vector<std::string> labels;
            labels.reserve(q.options.size());
            for (const auto& [k, _] : q.options)
                labels.push_back(k);
            fallback_codes = decision_option_codes(labels);
            choice_codes = fallback_codes;
        }
        check(choice_codes.size() == q.options.size(), "decision_choice_code_count");
        body = q.instructions + "\n\nOptions:\n";
        for (std::size_t i = 0; i < q.options.size(); ++i) {
            if (i > 0)
                body += '\n';
            const std::string desc = q.options[i].second.empty()
                                         ? replace_underscores(q.options[i].first)
                                         : q.options[i].second;
            body += choice_codes[i] + " " + desc;
        }
        body += "\n\nReply with the option code only.";
    } else if (q.kind == DecisionQuestion::Kind::score) {
        check(q.levels.size() >= 2 && q.levels.size() <= 10, "decision_score_level_bounds");
        body = q.instructions + "\n\n";
        for (std::size_t i = 0; i < q.levels.size(); ++i) {
            if (i > 0)
                body += '\n';
            body += std::to_string(i) + " " + q.levels[i];
        }
        body += "\n\nReply with a single digit 0-" + std::to_string(q.levels.size() - 1)
                + " only.";
    } else {
        body = q.instructions;
        if (!q.true_when.empty() || !q.false_when.empty()) {
            body += "\nYes: " + q.true_when + "\nNo: " + q.false_when;
        }
        body += "\n\nReply with yes or no only.";
    }

    // Inkling 2 Mini native chat template with thinking effort 0, ending directly at
    // <|message_model|><|content_text|> so the very next token distribution is the
    // direct answer token (no thinking block, zero autoregressive decoding).
    DecisionPrompt prompt;
    prompt.text += "<|message_system|><|content_text|>Thinking effort level: 0<|end_message|>";
    prompt.text += "<|message_user|><|content_text|>";
    const std::size_t literal_start = prompt.text.size();
    prompt.text += state;
    prompt.text += "\n\nQUESTION:\n";
    prompt.text += body;
    if (prompt.text.size() > literal_start)
        prompt.literal_ranges.emplace_back(literal_start, prompt.text.size());
    prompt.text += "<|end_message|><|message_model|><|content_text|>";
    return prompt;
}

std::vector<std::vector<std::uint32_t>> decision_readout_groups(
    Tokenizer& tokenizer,
    const DecisionQuestion& q,
    std::span<const std::pair<std::string, std::uint32_t>> resolved_choices,
    const std::atomic_bool& cancel
) {
    std::vector<std::vector<std::uint32_t>> groups;
    if (q.kind == DecisionQuestion::Kind::noul) {
        std::vector<std::uint32_t> yes_ids, no_ids;
        for (const char* form : {"yes", "Yes", "YES", " yes", " Yes", " YES"}) {
            if (auto id = single_token_id(tokenizer, form, cancel))
                append_unique(yes_ids, *id);
        }
        for (const char* form : {"no", "No", "NO", " no", " No", " NO"}) {
            if (auto id = single_token_id(tokenizer, form, cancel))
                append_unique(no_ids, *id);
        }
        check(!yes_ids.empty() && !no_ids.empty(), "decision_missing_yes_no_tokens");
        groups.push_back(std::move(yes_ids));
        groups.push_back(std::move(no_ids));
        return groups;
    }
    if (q.kind == DecisionQuestion::Kind::score) {
        check(q.levels.size() >= 2 && q.levels.size() <= 10, "decision_score_level_bounds");
        std::set<std::uint32_t> used;
        std::vector<std::uint32_t> primary;
        primary.reserve(q.levels.size());
        for (std::size_t i = 0; i < q.levels.size(); ++i) {
            auto id = single_token_id(tokenizer, std::to_string(i), cancel);
            check(id.has_value(), "decision_missing_digit_token");
            used.insert(*id);
            primary.push_back(*id);
        }
        groups.reserve(q.levels.size());
        for (std::size_t i = 0; i < q.levels.size(); ++i) {
            std::vector<std::uint32_t> g{primary[i]};
            if (auto spaced = single_token_id(tokenizer, " " + std::to_string(i), cancel);
                spaced && !used.contains(*spaced)) {
                used.insert(*spaced);
                g.push_back(*spaced);
            }
            groups.push_back(std::move(g));
        }
        return groups;
    }

    std::vector<std::pair<std::string, std::uint32_t>> owned_aliases;
    if (resolved_choices.empty()) {
        std::vector<std::string> labels;
        labels.reserve(q.options.size());
        for (const auto& [k, _] : q.options)
            labels.push_back(k);
        owned_aliases = resolve_choice_aliases(tokenizer, labels, cancel);
        resolved_choices = owned_aliases;
    }
    check(resolved_choices.size() == q.options.size(), "decision_choice_alias_count");
    std::set<std::uint32_t> used;
    for (const auto& [_, tid] : resolved_choices)
        used.insert(tid);
    groups.reserve(resolved_choices.size());
    for (const auto& [code, tid] : resolved_choices) {
        std::vector<std::uint32_t> g{tid};
        if (auto spaced = single_token_id(tokenizer, " " + code, cancel);
            spaced && !used.contains(*spaced)) {
            used.insert(*spaced);
            g.push_back(*spaced);
        }
        groups.push_back(std::move(g));
    }
    return groups;
}

std::vector<double> softmax_scores(std::span<const double> scores, double temperature) {
    check(!scores.empty() && temperature > 0.0, "decision_softmax_params");
    double mx = -std::numeric_limits<double>::infinity();
    std::vector<double> scaled(scores.size());
    for (std::size_t i = 0; i < scores.size(); ++i) {
        check(std::isfinite(scores[i]), "decision_nonfinite_score");
        scaled[i] = scores[i] / temperature;
        mx = std::max(mx, scaled[i]);
    }
    double sum = 0.0;
    for (double& s : scaled) {
        s = std::exp(s - mx);
        sum += s;
    }
    check(sum > 0.0 && std::isfinite(sum), "decision_softmax_sum");
    for (double& s : scaled)
        s /= sum;
    return scaled;
}

Json score_decision_logits(
    const DecisionQuestion& q,
    std::span<const float> logits,
    const std::vector<std::vector<std::uint32_t>>& groups,
    double temperature
) {
    const std::size_t V = logits.size();
    check(V > 0 && !groups.empty(), "decision_empty_logits");

    double max_z = -std::numeric_limits<double>::infinity();
    for (std::size_t v = 0; v < V; ++v) {
        const double z = static_cast<double>(logits[v]);
        if (std::isfinite(z))
            max_z = std::max(max_z, z);
    }
    check(std::isfinite(max_z), "decision_nonfinite_logits");
    double lse = 0.0;
    for (std::size_t v = 0; v < V; ++v) {
        const double z = static_cast<double>(logits[v]);
        if (std::isfinite(z))
            lse += std::exp(z - max_z);
    }
    const double log_denom = max_z + std::log(lse);

    std::vector<double> scores(groups.size());
    for (std::size_t g = 0; g < groups.size(); ++g) {
        check(!groups[g].empty(), "decision_empty_readout_group");
        double best = -std::numeric_limits<double>::infinity();
        for (std::uint32_t tid : groups[g]) {
            check(tid < V, "decision_readout_token_range");
            const double z = static_cast<double>(logits[tid]);
            check(std::isfinite(z), "decision_nonfinite_readout_logit");
            best = std::max(best, z - log_denom);
        }
        scores[g] = best;
    }
    const auto probs = softmax_scores(scores, temperature);

    if (q.kind == DecisionQuestion::Kind::choice) {
        check(probs.size() == q.options.size(), "decision_choice_prob_count");
        Json prob_map = Json::object();
        std::size_t best_idx = 0;
        for (std::size_t i = 0; i < q.options.size(); ++i) {
            prob_map[q.options[i].first] = probs[i];
            if (probs[i] > probs[best_idx])
                best_idx = i;
        }
        return Json{
            {"key", q.key},
            {"type", "Choice"},
            {"value", q.options[best_idx].first},
            {"probabilities", std::move(prob_map)},
            {"confidence", probs[best_idx]}
        };
    }
    if (q.kind == DecisionQuestion::Kind::score) {
        check(probs.size() == q.levels.size(), "decision_score_prob_count");
        Json prob_map = Json::object();
        double mean = 0.0;
        for (std::size_t i = 0; i < q.levels.size(); ++i) {
            prob_map[std::to_string(i)] = probs[i];
            mean += static_cast<double>(i) * probs[i];
        }
        return Json{
            {"key", q.key},
            {"type", "Score"},
            {"value", mean},
            {"probabilities", std::move(prob_map)}
        };
    }
    check(probs.size() == 2, "decision_noul_prob_count");
    return Json{
        {"key", q.key},
        {"type", "Noul"},
        {"value", probs[0]},
        {"probabilities", {{"true", probs[0]}, {"false", probs[1]}}}
    };
}

std::size_t shared_token_prefix_length(
    std::span<const PreparedDecisionQuestion> questions
) {
    if (questions.size() < 2)
        return 0;
    std::size_t limit = questions[0].tokens.size();
    for (std::size_t i = 1; i < questions.size(); ++i)
        limit = std::min(limit, questions[i].tokens.size());
    if (limit <= 1)
        return 0;
    // Leave at least one suffix token so every question runs prefill_logits on its ending.
    const std::size_t max_shared = limit - 1;
    std::size_t shared = 0;
    while (shared < max_shared) {
        const std::uint32_t tok = questions[0].tokens[shared];
        bool all_match = true;
        for (std::size_t i = 1; i < questions.size(); ++i) {
            if (questions[i].tokens[shared] != tok) {
                all_match = false;
                break;
            }
        }
        if (!all_match)
            break;
        ++shared;
    }
    return shared;
}

std::vector<PreparedDecisionQuestion> prepare_inkling_decisions(
    std::string_view state,
    std::span<const DecisionQuestion> questions,
    Tokenizer& tokenizer,
    std::size_t max_context_tokens,
    std::size_t max_total_tokens,
    const std::atomic_bool& cancel
) {
    check(!questions.empty() && questions.size() <= 8, "decision_question_count");
    check(max_context_tokens > 0 && max_total_tokens > 0, "decision_context_limit");

    std::vector<PreparedDecisionQuestion> prepared;
    prepared.reserve(questions.size());
    std::size_t total_tokens = 0;

    for (const auto& q : questions) {
        check_cancel(cancel);
        std::vector<std::pair<std::string, std::uint32_t>> aliases;
        std::vector<std::string> choice_codes;
        if (q.kind == DecisionQuestion::Kind::choice) {
            std::vector<std::string> labels;
            labels.reserve(q.options.size());
            for (const auto& [k, _] : q.options)
                labels.push_back(k);
            aliases = resolve_choice_aliases(tokenizer, labels, cancel);
            choice_codes.reserve(aliases.size());
            for (const auto& [code, _] : aliases)
                choice_codes.push_back(code);
        }
        auto prompt = render_inkling_decision_prompt(state, q, choice_codes);
        const std::size_t byte_limit = std::max<std::size_t>(65536, prompt.text.size() + 64);
        std::vector<std::uint32_t> tokens(max_context_tokens);
        const std::size_t count = tokenizer.encode(
            prompt.text, tokens, cancel, byte_limit, prompt.literal_ranges
        );
        check(
            count > 0 && count <= max_context_tokens,
            "Decision input exceeds the language model context limit; nothing was truncated."
        );
        tokens.resize(count);
        total_tokens += count;
        check(
            total_tokens <= max_total_tokens,
            "Decision input exceeds 65,536 encoded tokens; nothing was truncated."
        );
        auto groups = decision_readout_groups(tokenizer, q, aliases, cancel);
        prepared.push_back({std::move(tokens), std::move(groups)});
    }
    return prepared;
}

}  // namespace kadan::text
