#include "decision_tokenizer.hpp"
#define PCRE2_CODE_UNIT_WIDTH 8
#include <fstream>
#include <limits>
#include <pcre2.h>
#include <set>
#include <unicode/normalizer2.h>
#include <unicode/unistr.h>
#include <unicode/ustring.h>
namespace kadan::decision {
void check(bool ok, const char *error) {
    if (!ok)
        throw std::runtime_error(error);
}
void cancel_check(const std::atomic_bool &flag) { check(!flag.load(), "decision_cancelled"); }
Json parse_json(std::string_view text) {
    // Bound parser recursion and reject ambiguous duplicate fields, including
    // checkpoint metadata. Limits are checked before the DOM allocates a child.
    std::array<std::set<std::string>, 33> keys;
    std::size_t events = 0;
    return Json::parse(text, [&](int depth, Json::parse_event_t event, Json &value) {
        check(++events <= 400000, "decision_json_nodes");
        check(depth >= 0 && depth < 32, "decision_json_depth");
        if (event == Json::parse_event_t::object_start)
            keys[depth + 1].clear();
        if (event == Json::parse_event_t::key)
            check(keys[depth].insert(value.get<std::string>()).second, "decision_duplicate_field");
        return true;
    });
}
Json read_json(const std::string &path, std::size_t limit) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    check(bool(in) && in.tellg() > 0 && std::uint64_t(in.tellg()) <= limit, "decision_json_size");
    std::string text(std::size_t(in.tellg()), '\0');
    in.seekg(0);
    in.read(text.data(), text.size());
    check(bool(in), "decision_json_read");
    return parse_json(text);
}
Tokenizer::Tokenizer(const Json &j) {
    check(j.at("normalizer").at("type") == "NFC" &&
              j.at("pre_tokenizer").at("type") == "ByteLevel" &&
              !j.at("pre_tokenizer").at("add_prefix_space").get<bool>() &&
              j.at("pre_tokenizer").at("use_regex").get<bool>(),
          "decision_tokenizer_pipeline");
    auto &m = j.at("model");
    check(m.at("type") == "BPE" && m.at("dropout").is_null() && m.at("unk_token").is_null() &&
              !m.at("byte_fallback").get<bool>() && !m.at("ignore_merges").get<bool>() &&
              m.at("continuing_subword_prefix").is_null() && m.at("end_of_word_suffix").is_null(),
          "decision_tokenizer_model");
    check(m.at("vocab").size() <= 65536 && m.at("merges").size() <= 65536,
          "decision_tokenizer_size");
    for (auto it = m.at("vocab").begin(); it != m.at("vocab").end(); ++it) {
        int id = it.value().get<int>();
        check(id >= 0 && id < 65536 && it.key().size() <= 1024, "decision_vocab");
        vocab_.emplace(it.key(), id);
    }
    int next = 256;
    for (int b = 0; b < 256; ++b) {
        int c = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174) ? b : next++;
        icu::UnicodeString s;
        s.append(UChar32(c));
        s.toUTF8String(bytes_[b]);
        // Invalid UTF-8 leading bytes never occur after strict NFC decoding.
        if (b < 192 || (b >= 194 && b <= 244))
            check(vocab_.contains(bytes_[b]), "decision_byte_vocab");
    }
    int rank = 0;
    for (auto &pair : m.at("merges")) {
        check(pair.is_array() && pair.size() == 2, "decision_merge");
        auto a = pair[0].get<std::string>(), b = pair[1].get<std::string>();
        check(vocab_.contains(a) && vocab_.contains(b) && vocab_.contains(a + b),
              "decision_merge_vocab");
        check(merges_.emplace(a + '\0' + b, std::pair{rank++, vocab_.at(a + b)}).second,
              "decision_duplicate_merge");
    }
    check(j.at("added_tokens").size() <= 256, "decision_added_tokens");
    for (auto &t : j.at("added_tokens")) {
        check(!t.at("single_word").get<bool>() &&
                  (!t.at("lstrip").get<bool>() || t.at("content") == "[MASK]") &&
                  !t.at("rstrip").get<bool>(),
              "decision_added_flags");
        auto s = t.at("content").get<std::string>();
        int id = t.at("id").get<int>();
        check(!s.empty() && s.size() <= 128 && id >= 0 && id < 65536 &&
                  (!vocab_.contains(s) || vocab_.at(s) == id),
              "decision_added_vocab");
        vocab_[s] = id;
        added_.push_back({s, id, t.at("normalized").get<bool>()});
    }
    cls = vocab_.at("[CLS]");
    sep = vocab_.at("[SEP]");
    mask = vocab_.at("[MASK]");
    int error;
    PCRE2_SIZE offset;
    const char *regex =
        "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+";
    pattern_ = pcre2_compile(reinterpret_cast<PCRE2_SPTR>(regex), PCRE2_ZERO_TERMINATED,
                             PCRE2_UTF | PCRE2_UCP, &error, &offset, nullptr);
    check(pattern_, "decision_tokenizer_regex");
}
Tokenizer::~Tokenizer() {
    if (pattern_)
        pcre2_code_free(static_cast<pcre2_code *>(pattern_));
}
void Tokenizer::ordinary(std::string_view text, std::vector<int> &out,
                         const std::atomic_bool &cancel) const {
    auto *code = static_cast<pcre2_code *>(pattern_);
    std::unique_ptr<pcre2_match_data, decltype(&pcre2_match_data_free)> match(
        pcre2_match_data_create_from_pattern(code, nullptr), pcre2_match_data_free);
    check(bool(match), "decision_regex_memory");
    std::size_t offset = 0;
    while (offset < text.size()) {
        cancel_check(cancel);
        check(pcre2_match(code, reinterpret_cast<PCRE2_SPTR>(text.data()), text.size(), offset,
                          PCRE2_ANCHORED, match.get(), nullptr) > 0,
              "decision_tokenizer_match");
        auto *bounds = pcre2_get_ovector_pointer(match.get());
        check(bounds[1] > offset, "decision_tokenizer_empty");
        std::vector<std::string> pieces;
        for (std::size_t i = offset; i < bounds[1]; ++i)
            pieces.push_back(bytes_[static_cast<unsigned char>(text[i])]);
        while (pieces.size() > 1) {
            cancel_check(cancel);
            int best = std::numeric_limits<int>::max();
            std::size_t at = 0;
            for (std::size_t i = 0; i + 1 < pieces.size(); ++i) {
                auto it = merges_.find(pieces[i] + '\0' + pieces[i + 1]);
                if (it != merges_.end() && it->second.first < best) {
                    best = it->second.first;
                    at = i;
                }
            }
            if (best == std::numeric_limits<int>::max())
                break;
            pieces[at] += pieces[at + 1];
            pieces.erase(pieces.begin() + at + 1);
        }
        for (auto &s : pieces)
            out.push_back(vocab_.at(s));
        offset = bounds[1];
    }
}
std::vector<int> Tokenizer::encode(const std::string &input, const std::atomic_bool &cancel) const {
    check(input.size() <= 65536, "decision_text_size");
    check(input.find("[MASK]") == std::string::npos, "decision_literal_mask");
    cancel_check(cancel);
    UErrorCode status = U_ZERO_ERROR;
    int32_t required = 0;
    u_strFromUTF8(nullptr, 0, &required, input.data(), int32_t(input.size()), &status);
    check(status == U_BUFFER_OVERFLOW_ERROR || U_SUCCESS(status), "decision_utf8");
    status = U_ZERO_ERROR;
    auto *nfc = icu::Normalizer2::getNFCInstance(status);
    check(U_SUCCESS(status), "decision_nfc");
    auto normalize = [&](std::string_view value) {
        icu::UnicodeString normalized;
        nfc->normalize(icu::UnicodeString::fromUTF8(value), normalized, status);
        check(U_SUCCESS(status), "decision_nfc");
        std::string result;
        normalized.toUTF8String(result);
        return result;
    };
    std::vector<int> out;
    // Extract non-normalized added tokens from the original input first. NFC
    // may create their spelling, but must never create a special-token match.
    auto split = [&](std::string_view text, bool normalized, auto ordinary_part) {
        std::size_t start = 0;
        while (start < text.size()) {
            cancel_check(cancel);
            std::size_t found = text.size(), length = 0;
            const Added *selected = nullptr;
            for (const auto &token : added_) {
                if (token.normalized != normalized)
                    continue;
                auto spelling = normalized ? normalize(token.text) : token.text;
                auto pos = text.find(spelling, start);
                if (pos < found || (pos == found && selected && spelling.size() > length)) {
                    found = pos;
                    length = spelling.size();
                    selected = &token;
                }
            }
            ordinary_part(text.substr(start, found - start));
            if (!selected)
                break;
            check(selected->id != mask, "decision_literal_mask");
            out.push_back(selected->id);
            start = found + length;
        }
    };
    split(input, false, [&](std::string_view raw) {
        auto normalized = normalize(raw);
        split(normalized, true, [&](std::string_view text) { ordinary(text, out, cancel); });
    });
    return out;
}
} // namespace kadan::decision
