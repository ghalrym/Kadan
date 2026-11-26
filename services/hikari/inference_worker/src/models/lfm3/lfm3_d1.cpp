#include "kadan/lfm3.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <limits>
#include <set>
#include <stdexcept>

namespace kadan::lfm3 {
namespace {
constexpr std::string_view kImStart = "<|im_start|>";
constexpr std::string_view kImEnd = "<|im_end|>";
constexpr std::string_view kZeroWidthSpace = "\xE2\x80\x8B";

void check(bool ok, const char* message) {
    if (!ok)
        throw std::runtime_error(message);
}

void check_cancel(const std::atomic_bool& cancel) {
    if (cancel.load(std::memory_order_relaxed))
        throw std::runtime_error("d1_cancelled");
}

std::string replace_underscores(std::string_view text) {
    std::string out(text);
    for (char& c : out)
        if (c == '_')
            c = ' ';
    return out;
}

std::vector<std::uint32_t> encode_str(
    text::Tokenizer& tokenizer, const std::string& text, const std::atomic_bool& cancel
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
    text::Tokenizer& tokenizer, const std::string& raw, const std::atomic_bool& cancel
) {
    try {
        auto ids = encode_str(tokenizer, raw, cancel);
        if (ids.size() == 1)
            return ids[0];
    } catch (...) {
    }
    return std::nullopt;
}

std::vector<std::pair<std::string, std::uint32_t>> resolve_choice_aliases(
    text::Tokenizer& tokenizer,
    const std::vector<std::string>& labels,
    int code_scheme,
    const std::atomic_bool& cancel
) {
    auto codes = d1_option_codes(labels, code_scheme);
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
    auto take = [&](const std::string& raw) -> bool {
        auto id = single_token_id(tokenizer, raw, cancel);
        if (!id || used.contains(*id))
            return false;
        used.insert(*id);
        out.emplace_back(raw, *id);
        return true;
    };
    for (const auto& code : codes) {
        if (take(code))
            continue;
        bool ok = false;
        for (const auto& cand : fallback) {
            if (take(cand)) {
                ok = true;
                break;
            }
        }
        check(ok, "d1_no_single_token_alias");
    }
    return out;
}

std::vector<std::vector<std::uint32_t>> readout_token_groups(
    text::Tokenizer& tokenizer,
    const D1Question& q,
    int code_scheme,
    const std::atomic_bool& cancel
) {
    std::vector<std::vector<std::uint32_t>> groups;
    if (q.kind == D1Question::Kind::noul) {
        std::vector<std::uint32_t> yes_ids, no_ids;
        for (const char* form : {"yes", "Yes", "YES"}) {
            if (auto id = single_token_id(tokenizer, form, cancel)) {
                if (std::find(yes_ids.begin(), yes_ids.end(), *id) == yes_ids.end())
                    yes_ids.push_back(*id);
            }
        }
        for (const char* form : {"no", "No", "NO"}) {
            if (auto id = single_token_id(tokenizer, form, cancel)) {
                if (std::find(no_ids.begin(), no_ids.end(), *id) == no_ids.end())
                    no_ids.push_back(*id);
            }
        }
        check(!yes_ids.empty() && !no_ids.empty(), "d1_missing_yes_no_tokens");
        groups.push_back(std::move(yes_ids));
        groups.push_back(std::move(no_ids));
        return groups;
    }
    if (q.kind == D1Question::Kind::score) {
        check(q.levels.size() >= 2 && q.levels.size() <= 10, "d1_score_level_bounds");
        for (std::size_t i = 0; i < q.levels.size(); ++i) {
            auto id = single_token_id(tokenizer, std::to_string(i), cancel);
            check(id.has_value(), "d1_missing_digit_token");
            groups.push_back({*id});
        }
        return groups;
    }
    std::vector<std::string> labels;
    labels.reserve(q.options.size());
    for (const auto& [k, _] : q.options)
        labels.push_back(k);
    auto aliases = resolve_choice_aliases(tokenizer, labels, code_scheme, cancel);
    for (const auto& [code, tid] : aliases) {
        std::vector<std::uint32_t> g{tid};
        if (auto spaced = single_token_id(tokenizer, " " + code, cancel);
            spaced && *spaced != tid) {
            g.push_back(*spaced);
        }
        groups.push_back(std::move(g));
    }
    return groups;
}

std::vector<double> softmax_scores(std::span<const double> scores, double temperature) {
    check(!scores.empty() && temperature > 0.0, "d1_softmax_params");
    double mx = -std::numeric_limits<double>::infinity();
    std::vector<double> scaled(scores.size());
    for (std::size_t i = 0; i < scores.size(); ++i) {
        scaled[i] = scores[i] / temperature;
        mx = std::max(mx, scaled[i]);
    }
    double sum = 0.0;
    for (double& s : scaled) {
        s = std::exp(s - mx);
        sum += s;
    }
    for (double& s : scaled)
        s /= sum;
    return scaled;
}
}  // namespace

D1Calibration D1Calibration::parse(const Json& doc) {
    D1Calibration cal;
    cal.default_temperature = doc.value("default", 1.0);
    if (doc.contains("temperatures") && doc.at("temperatures").is_object()) {
        for (auto it = doc.at("temperatures").begin();
             it != doc.at("temperatures").end();
             ++it) {
            cal.temperatures[it.key()] = it.value().get<double>();
        }
    }
    return cal;
}

double D1Calibration::temperature_for(const D1Question& q) const {
    std::string bucket;
    if (q.kind == D1Question::Kind::noul) {
        bucket = "noul";
    } else if (q.kind == D1Question::Kind::score) {
        bucket = "score:" + std::to_string(q.levels.size());
    } else {
        const std::size_t k = q.options.size();
        if (k <= 3)
            bucket = "choice:small";
        else if (k <= 8)
            bucket = "choice:mid";
        else if (k <= 26)
            bucket = "choice:large";
        else
            bucket = "choice:inventory";
    }
    const auto it = temperatures.find(bucket);
    return it != temperatures.end() ? it->second : default_temperature;
}

std::string d1_escape_specials(
    std::string_view text, const std::vector<std::string>& specials
) {
    std::vector<std::string> sorted;
    for (const auto& s : specials) {
        if (s.size() > 1 && s.find(kZeroWidthSpace) == std::string::npos)
            sorted.push_back(s);
    }
    if (sorted.empty())
        return std::string(text);
    std::sort(sorted.begin(), sorted.end(), [](const auto& a, const auto& b) {
        return a.size() > b.size();
    });
    std::string result(text);
    bool changed = true;
    while (changed) {
        changed = false;
        std::size_t best_pos = result.size();
        std::size_t best_len = 0;
        for (const auto& s : sorted) {
            const auto pos = result.find(s);
            if (pos != std::string::npos
                && (pos < best_pos || (pos == best_pos && s.size() > best_len))) {
                best_pos = pos;
                best_len = s.size();
            }
        }
        if (best_len > 0) {
            result.insert(best_pos + 1, kZeroWidthSpace);
            changed = true;
        }
    }
    return result;
}

std::vector<std::string> d1_option_codes(const std::vector<std::string>& labels, int scheme) {
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
    if (all_single_alpha)
        return trimmed;
    std::vector<std::string> out;
    if (labels.size() <= 26) {
        for (std::size_t i = 0; i < labels.size(); ++i)
            out.emplace_back(1, static_cast<char>('A' + i));
        return out;
    }
    if (scheme == 2) {
        for (char c = 'A'; c <= 'Z' && out.size() < labels.size(); ++c)
            out.emplace_back(1, c);
        for (char a = 'A'; a <= 'Z' && out.size() < labels.size(); ++a)
            for (char b = 'A'; b <= 'Z' && out.size() < labels.size(); ++b)
                out.push_back(std::string{a, b});
        return out;
    }
    for (std::size_t i = 0; i < labels.size(); ++i) {
        char buf[32];
        std::snprintf(buf, sizeof(buf), "%02zu", i);
        out.emplace_back(buf);
    }
    return out;
}

std::string d1_prefix_text(
    std::string_view state, const std::vector<std::string>& specials
) {
    // Matches d1/prompt.py: bos="<|startoftext|>", system="none", state_style="json_only".
    std::string block(state);
    block += "\n\n";
    return std::string("<|startoftext|>") + std::string(kImStart) + "user\n"
           + d1_escape_specials(block, specials) + "\nQUESTION:\n";
}

std::string d1_suffix_text(
    const D1Question& q, const std::vector<std::string>& specials, int code_scheme
) {
    std::string body;
    if (q.kind == D1Question::Kind::choice) {
        std::vector<std::string> labels;
        labels.reserve(q.options.size());
        for (const auto& [k, _] : q.options)
            labels.push_back(k);
        const auto codes = d1_option_codes(labels, code_scheme);
        body = q.instructions + "\n\nOptions:\n";
        for (std::size_t i = 0; i < q.options.size(); ++i) {
            if (i > 0)
                body += '\n';
            const std::string desc = q.options[i].second.empty()
                                         ? replace_underscores(q.options[i].first)
                                         : q.options[i].second;
            body += codes[i] + " " + desc;
        }
        body += "\n\nReply with the option code only.";
    } else if (q.kind == D1Question::Kind::score) {
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
    // Non-thinking ChatML lead for LFM3-VL is "<think></think>".
    return d1_escape_specials(body, specials) + std::string(kImEnd) + "\n"
           + std::string(kImStart) + "assistant\n<think></think>";
}

Json evaluate_d1_decisions(
    std::string_view state,
    const std::vector<D1Question>& questions,
    const Backbone& backbone,
    text::Tokenizer& tokenizer,
    const D1Calibration& calibration,
    std::size_t set_size,
    std::size_t max_tokens,
    const std::atomic_bool& cancel,
    const std::function<void(std::size_t, std::size_t)>& on_encoded
) {
    check(!questions.empty() && questions.size() <= 8, "d1_question_count");
    const auto specials = tokenizer.special_tokens();
    const std::string prefix = d1_prefix_text(state, specials);
    std::vector<std::string> suffixes;
    suffixes.reserve(questions.size());
    for (const auto& q : questions)
        suffixes.push_back(d1_suffix_text(q, specials, 2));

    const std::size_t step = std::max<std::size_t>(1, set_size);
    struct PackedSeq {
        std::size_t first_q = 0;
        std::size_t count_q = 0;
        std::vector<std::uint32_t> ids;
        std::size_t state_len = 0;
        std::vector<int> groups;
        std::vector<std::size_t> answer_slots;
    };
    std::vector<PackedSeq> sequences;
    std::size_t total_tokens = 0;
    for (std::size_t k = 0; k < questions.size(); k += step) {
        check_cancel(cancel);
        const std::size_t count = std::min(step, questions.size() - k);
        PackedSeq seq;
        seq.first_q = k;
        seq.count_q = count;
        seq.ids = encode_str(tokenizer, prefix + suffixes[k], cancel);
        const auto s0_ids = encode_str(tokenizer, suffixes[k], cancel);
        check(seq.ids.size() > s0_ids.size(), "d1_state_length");
        seq.state_len = seq.ids.size() - s0_ids.size();
        seq.groups.assign(seq.state_len, -1);
        seq.groups.insert(seq.groups.end(), seq.ids.size() - seq.state_len, 1);
        seq.answer_slots.push_back(seq.ids.size() - 1);
        for (std::size_t j = 1; j < count; ++j) {
            auto own = encode_str(tokenizer, "QUESTION:\n" + suffixes[k + j], cancel);
            seq.ids.insert(seq.ids.end(), own.begin(), own.end());
            seq.groups.insert(seq.groups.end(), own.size(), static_cast<int>(j + 1));
            seq.answer_slots.push_back(seq.ids.size() - 1);
        }
        total_tokens += seq.ids.size();
        sequences.push_back(std::move(seq));
    }
    check(
        total_tokens > 0 && total_tokens <= max_tokens,
        "The d1 input exceeds 65,536 encoded tokens; nothing was truncated."
    );
    if (on_encoded)
        on_encoded(total_tokens, max_tokens);

    Json answers = Json::array();
    const std::size_t H = backbone.config().hidden_size;
    const std::size_t V = backbone.config().vocab_size;

    for (const auto& seq : sequences) {
        check_cancel(cancel);
        const std::size_t T = seq.ids.size();
        std::vector<float> embeds(T * H);
        backbone.embed_tokens(seq.ids, embeds, cancel);
        std::vector<float> hidden(T * H);
        SetAttentionMask mask{seq.state_len, seq.groups};
        backbone.forward(embeds, T, hidden, nullptr, &mask, cancel, T);

        // Gather only the answer slot hidden vectors and project through lm_head.
        const std::size_t n_slots = seq.answer_slots.size();
        std::vector<float> slot_hidden(n_slots * H);
        for (std::size_t s = 0; s < n_slots; ++s) {
            std::copy_n(
                hidden.data() + seq.answer_slots[s] * H, H, slot_hidden.data() + s * H
            );
        }
        std::vector<float> logits(n_slots * V);
        backbone.lm_head(slot_hidden, n_slots, logits, cancel);

        for (std::size_t s = 0; s < n_slots; ++s) {
            const auto& q = questions[seq.first_q + s];
            const float* row = logits.data() + s * V;
            // Compute log-softmax over vocabulary.
            double max_z = -std::numeric_limits<double>::infinity();
            for (std::size_t v = 0; v < V; ++v)
                max_z = std::max(max_z, static_cast<double>(row[v]));
            double lse = 0.0;
            for (std::size_t v = 0; v < V; ++v)
                lse += std::exp(static_cast<double>(row[v]) - max_z);
            const double log_denom = max_z + std::log(lse);

            const auto groups = readout_token_groups(tokenizer, q, 2, cancel);
            std::vector<double> scores(groups.size());
            for (std::size_t g = 0; g < groups.size(); ++g) {
                double best = -std::numeric_limits<double>::infinity();
                for (std::uint32_t tid : groups[g]) {
                    check(tid < V, "d1_readout_token_range");
                    best = std::max(best, static_cast<double>(row[tid]) - log_denom);
                }
                scores[g] = best;
            }
            const auto probs = softmax_scores(scores, calibration.temperature_for(q));

            if (q.kind == D1Question::Kind::choice) {
                Json prob_map = Json::object();
                std::size_t best_idx = 0;
                for (std::size_t i = 0; i < q.options.size(); ++i) {
                    prob_map[q.options[i].first] = probs[i];
                    if (probs[i] > probs[best_idx])
                        best_idx = i;
                }
                answers.push_back(
                    {{"key", q.key},
                     {"type", "Choice"},
                     {"value", q.options[best_idx].first},
                     {"probabilities", prob_map},
                     {"confidence", probs[best_idx]}}
                );
            } else if (q.kind == D1Question::Kind::score) {
                Json prob_map = Json::object();
                double mean = 0.0;
                for (std::size_t i = 0; i < q.levels.size(); ++i) {
                    prob_map[std::to_string(i)] = probs[i];
                    mean += static_cast<double>(i) * probs[i];
                }
                answers.push_back(
                    {{"key", q.key},
                     {"type", "Score"},
                     {"value", mean},
                     {"probabilities", prob_map}}
                );
            } else {
                answers.push_back(
                    {{"key", q.key},
                     {"type", "Noul"},
                     {"value", probs[0]},
                     {"probabilities", {{"true", probs[0]}, {"false", probs[1]}}}}
                );
            }
        }
    }
    return answers;
}

}  // namespace kadan::lfm3
