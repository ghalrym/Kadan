#include "kadan/text_tokenizer.hpp"
#define PCRE2_CODE_UNIT_WIDTH 8
#include <algorithm>
#include <array>
#include <fstream>
#include <limits>
#include <nlohmann/json.hpp>
#include <pcre2.h>
#include <unicode/normalizer2.h>
#include <unicode/unistr.h>
#include <unicode/ustring.h>
#include <unordered_map>
#include <unordered_set>
namespace kadan::text {
namespace {
void check(bool ok, const char *why) {
  if (!ok)
    throw std::runtime_error(why);
}
void stop(const std::atomic_bool &flag) {
  check(!flag.load(), "text_tokenizer_cancelled");
}
Footprint host(Resources &r, Bytes bytes) {
  auto p = r.snapshot().capacity;
  std::fill(p.begin(), p.end(), 0);
  p[0] = bytes;
  return p;
}
struct Admission {
  Resources &r;
  Handle id;
  Admission(Resources &ledger, Workload workload, Bytes bytes)
      : r(ledger), id(r.reserve(workload, host(r, bytes))) {}
  ~Admission() {
    if (id)
      r.released(id);
  }
};
struct Active {
  bool &busy;
  explicit Active(bool &b) : busy(b) {
    check(!b, "busy");
    busy = true;
  }
  ~Active() { busy = false; }
};
using Json = nlohmann::json;
Json read_json(const std::string &path, const std::atomic_bool &cancel) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  check(bool(file) && file.tellg() > 0 &&
            std::uint64_t(file.tellg()) <= 32 * 1024 * 1024,
        "text_tokenizer_file_size");
  std::string text(std::size_t(file.tellg()), '\0');
  file.seekg(0);
  file.read(text.data(), text.size());
  check(bool(file), "text_tokenizer_file_read");
  std::array<std::unordered_set<std::string>, 17> keys;
  std::size_t events = 0;
  return Json::parse(
      text, [&](int depth, Json::parse_event_t event, Json &value) {
        stop(cancel);
        check(++events < 2000000 && depth >= 0 && depth < 16,
              "text_tokenizer_json_limit");
        if (event == Json::parse_event_t::object_start)
          keys[depth + 1].clear();
        if (event == Json::parse_event_t::key)
          check(keys[depth].insert(value.get<std::string>()).second,
                "text_tokenizer_duplicate_key");
        return true;
      });
}
} // namespace
struct Tokenizer::Impl {
  std::array<std::string, 256> bytes;
  std::vector<std::string> decoded;
  bool normalize = false;
  bool ignore_merges = false;
  std::unordered_map<std::string, std::uint32_t> vocab;
  std::unordered_map<std::string, std::size_t> merges;
  std::vector<std::pair<std::string, std::uint32_t>> added;
  std::unique_ptr<pcre2_code, decltype(&pcre2_code_free)> pattern{
      nullptr, pcre2_code_free};
  Impl(const std::string &path, const std::atomic_bool &cancel) {
    const auto j = read_json(path, cancel);
    configure_text(j);
    const auto &model = j.at("model");
    validate_model(model);
    ignore_merges = model.value("ignore_merges", false);
    auto ids = load_vocabulary(model, cancel);
    build_byte_alphabet();
    load_merges(model, cancel);
    load_added_tokens(j, ids);
    build_decoder();
    compile_pattern(j.at("pre_tokenizer").at("pretokenizers")[0]);
  }
  void configure_text(const Json &j) {
    normalize = !j.at("normalizer").is_null();
    if (normalize)
      check(j.at("normalizer").at("type") == "NFC",
            "text_tokenizer_normalizer");
    const auto &pre = j.at("pre_tokenizer");
    check(pre.at("type") == "Sequence" && pre.at("pretokenizers").size() == 2,
          "text_tokenizer_pre");
    const auto &split = pre.at("pretokenizers")[0];
    const auto &byte = pre.at("pretokenizers")[1];
    check(split.at("type") == "Split" && split.at("behavior") == "Isolated" &&
              !split.at("invert").get<bool>(),
          "text_tokenizer_regex_contract");
    check(byte.at("type") == "ByteLevel" &&
              !byte.at("add_prefix_space").get<bool>() &&
              !byte.at("use_regex").get<bool>(),
          "text_tokenizer_byte_contract");
    const auto &post = j.at("post_processor");
    check(post.at("type") == "ByteLevel", "text_tokenizer_post");
  }
  static void validate_model(const Json &m) {
    check(m.at("type") == "BPE" && m.at("dropout").is_null() &&
              m.at("unk_token").is_null() &&
              !m.at("byte_fallback").get<bool>() &&
              (m.at("continuing_subword_prefix").is_null() ||
               m.at("continuing_subword_prefix") == "") &&
              (m.at("end_of_word_suffix").is_null() ||
               m.at("end_of_word_suffix") == ""),
          "text_tokenizer_model");
  }
  std::vector<bool> load_vocabulary(const Json &m,
                                    const std::atomic_bool &cancel) {
    check(m.at("vocab").size() >= 256 && m.at("vocab").size() <= 262144 &&
              m.at("merges").size() <= 300000,
          "text_tokenizer_vocab_size");
    std::vector<bool> ids(262144);
    decoded.resize(262144);
    for (auto it = m.at("vocab").begin(); it != m.at("vocab").end(); ++it) {
      stop(cancel);
      check(it.value().is_number_unsigned() && it.value() < 262144,
            "text_tokenizer_vocab_id");
      const auto id = it.value().get<std::uint32_t>();
      check(id < 262144 && !ids[id] && it.key().size() <= 1024,
            "text_tokenizer_vocab");
      ids[id] = true;
      vocab.emplace(it.key(), id);
    }
    return ids;
  }
  void build_byte_alphabet() {
    int next = 256;
    for (int b = 0; b < 256; ++b) {
      int c = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || b >= 174
                  ? b
                  : next++;
      icu::UnicodeString s;
      s.append(UChar32(c));
      s.toUTF8String(bytes[b]);
      check(vocab.contains(bytes[b]), "text_tokenizer_byte_vocab");
    }
  }
  void load_merges(const Json &m, const std::atomic_bool &cancel) {
    std::size_t rank = 0;
    for (const auto &item : m.at("merges")) {
      stop(cancel);
      std::string a, b;
      if (item.is_string()) {
        const auto s = item.get<std::string>();
        const auto space = s.find(' ');
        check(space != std::string::npos &&
                  s.find(' ', space + 1) == std::string::npos,
              "text_tokenizer_merge");
        a = s.substr(0, space);
        b = s.substr(space + 1);
      } else {
        check(item.is_array() && item.size() == 2, "text_tokenizer_merge");
        a = item[0].get<std::string>();
        b = item[1].get<std::string>();
      }
      check(vocab.contains(a) && vocab.contains(b) && vocab.contains(a + b),
            "text_tokenizer_merge_vocab");
      check(merges.emplace(a + '\0' + b, rank++).second,
            "text_tokenizer_duplicate_merge");
    }
  }
  void load_added_tokens(const Json &j, std::vector<bool> &ids) {
    check(j.at("added_tokens").size() <= 2048, "text_tokenizer_added_limit");
    for (const auto &item : j.at("added_tokens")) {
      check(!item.at("single_word").get<bool>() &&
                !item.at("lstrip").get<bool>() &&
                !item.at("rstrip").get<bool>() &&
                !item.at("normalized").get<bool>(),
            "text_tokenizer_added_flags");
      auto text = item.at("content").get<std::string>();
      check(item.at("id").is_number_unsigned() && item.at("id") < 262144,
            "text_tokenizer_added_id");
      auto id = item.at("id").get<std::uint32_t>();
      check(!text.empty() && text.size() <= 256 && id < 262144 &&
                (!ids[id] || (vocab.contains(text) && vocab.at(text) == id)),
            "text_tokenizer_added");
      ids[id] = true;
      added.emplace_back(std::move(text), id);
    }
  }
  void build_decoder() {
    std::unordered_map<UChar32, unsigned char> reverse;
    for (int b = 0; b < 256; ++b)
      reverse.emplace(icu::UnicodeString::fromUTF8(bytes[b]).char32At(0),
                      static_cast<unsigned char>(b));
    for (const auto &[token, id] : vocab) {
      const auto unicode = icu::UnicodeString::fromUTF8(token);
      for (int32_t offset = 0; offset < unicode.length();) {
        const auto code = unicode.char32At(offset);
        check(reverse.contains(code), "text_tokenizer_decode_byte");
        decoded[id] += char(reverse.at(code));
        offset += U16_LENGTH(code);
      }
    }
    for (const auto &token : added)
      decoded[token.second].clear();
  }
  void compile_pattern(const Json &split) {
    const auto regex = split.at("pattern").at("Regex").get<std::string>();
    check(!regex.empty() && regex.size() <= 4096, "text_tokenizer_regex_size");
    int error;
    PCRE2_SIZE offset;
    pattern.reset(pcre2_compile(reinterpret_cast<PCRE2_SPTR>(regex.data()),
                                regex.size(), PCRE2_UTF | PCRE2_UCP, &error,
                                &offset, nullptr));
    check(bool(pattern), "text_tokenizer_regex");
  }
  template <class Emit>
  void encode_piece(std::string_view text, Emit emit,
                    const std::atomic_bool &cancel) {
    if (ignore_merges) {
      std::string encoded;
      for (unsigned char byte : text)
        encoded += bytes[byte];
      const auto found = vocab.find(encoded);
      if (found != vocab.end()) {
        emit(found->second);
        return;
      }
    }
    // Stable node indices preserve the
    // leftmost tie rule. Only the two
    // neighbours of a merge need new heap entries; generations reject
    // obsolete candidates without repeated full-sequence scans.
    constexpr auto none = std::numeric_limits<std::size_t>::max();
    struct Node {
      std::string value;
      std::size_t previous, next, generation = 0;
    };
    struct Candidate {
      std::size_t rank, left, left_generation, right_generation;
    };
    struct Later {
      bool operator()(const Candidate &a, const Candidate &b) const {
        return a.rank > b.rank || (a.rank == b.rank && a.left > b.left);
      }
    };
    const auto count = text.size();
    std::vector<Node> nodes;
    nodes.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
      stop(cancel);
      nodes.push_back({bytes[static_cast<unsigned char>(text[i])],
                       i ? i - 1 : none, i + 1 < count ? i + 1 : none});
    }
    // At most N initial and two entries per merge; this storage and
    // all nodes fit the input-sized admitted scratch envelope.
    std::vector<Candidate> heap;
    heap.reserve(3 * count);
    auto offer = [&](std::size_t left) {
      if (left == none || nodes[left].next == none)
        return;
      const auto right = nodes[left].next;
      const auto found =
          merges.find(nodes[left].value + '\0' + nodes[right].value);
      if (found != merges.end()) {
        heap.push_back({found->second, left, nodes[left].generation,
                        nodes[right].generation});
        std::push_heap(heap.begin(), heap.end(), Later{});
      }
    };
    for (std::size_t i = 0; i < count; ++i)
      offer(i);
    while (!heap.empty()) {
      stop(cancel);
      const auto candidate = heap.front();
      std::pop_heap(heap.begin(), heap.end(), Later{});
      heap.pop_back();
      auto &left = nodes[candidate.left];
      if (left.value.empty() || left.generation != candidate.left_generation ||
          left.next == none)
        continue;
      auto &right = nodes[left.next];
      if (right.generation != candidate.right_generation)
        continue;
      left.value += right.value;
      left.next = right.next;
      ++left.generation;
      if (left.next != none)
        nodes[left.next].previous = candidate.left;
      std::string{}.swap(right.value);
      ++right.generation;
      offer(left.previous);
      offer(candidate.left);
    }
    for (std::size_t i = 0; i != none; i = nodes[i].next)
      emit(vocab.at(nodes[i].value));
  }
  std::size_t encode(const std::string &input, std::span<std::uint32_t> output,
                     const std::atomic_bool &cancel) {
    UErrorCode status = U_ZERO_ERROR;
    int32_t needed = 0;
    u_strFromUTF8(nullptr, 0, &needed, input.data(), int32_t(input.size()),
                  &status);
    check(status == U_BUFFER_OVERFLOW_ERROR || U_SUCCESS(status),
          "text_tokenizer_utf8");
    status = U_ZERO_ERROR;
    const auto *nfc = icu::Normalizer2::getNFCInstance(status);
    check(U_SUCCESS(status), "text_tokenizer_nfc");
    std::unique_ptr<pcre2_match_data, decltype(&pcre2_match_data_free)> match(
        pcre2_match_data_create_from_pattern(pattern.get(), nullptr),
        pcre2_match_data_free);
    check(bool(match), "text_tokenizer_memory");
    std::unique_ptr<pcre2_match_context, decltype(&pcre2_match_context_free)>
        context(pcre2_match_context_create(nullptr), pcre2_match_context_free);
    check(bool(context), "text_tokenizer_memory");
    pcre2_set_match_limit(context.get(), 1000000);
    pcre2_set_depth_limit(context.get(), 1000);
    std::size_t written = 0;
    auto emit = [&](std::uint32_t id) {
      check(written < output.size(), "text_tokenizer_token_limit");
      output[written++] = id;
    };
    auto ordinary = [&](std::string_view raw) {
      icu::UnicodeString unicode = icu::UnicodeString::fromUTF8(raw);
      if (normalize) {
        icu::UnicodeString normalized;
        nfc->normalize(unicode, normalized, status);
        unicode = std::move(normalized);
      }
      check(U_SUCCESS(status), "text_tokenizer_nfc");
      std::string text;
      unicode.toUTF8String(text);
      check(text.size() <= 4 * input.size(), "text_tokenizer_normalized_size");
      std::size_t at = 0;
      while (at < text.size()) {
        stop(cancel);
        check(pcre2_match(pattern.get(),
                          reinterpret_cast<PCRE2_SPTR>(text.data()),
                          text.size(), at, PCRE2_ANCHORED, match.get(),
                          context.get()) > 0,
              "text_tokenizer_match");
        auto *bounds = pcre2_get_ovector_pointer(match.get());
        check(bounds[0] == at && bounds[1] > at, "text_tokenizer_match_range");
        encode_piece(std::string_view(text).substr(at, bounds[1] - at), emit,
                     cancel);
        at = bounds[1];
      }
    };
    std::size_t at = 0;
    while (at < input.size()) {
      stop(cancel);
      std::size_t found = input.size(), length = 0;
      std::uint32_t selected = 0;
      for (const auto &token : added) {
        const auto where = input.find(token.first, at);
        if (where < found || (where == found && token.first.size() > length)) {
          found = where;
          length = token.first.size();
          selected = token.second;
        }
      }
      ordinary(std::string_view(input).substr(at, found - at));
      if (!length)
        break;
      emit(selected);
      at = found + length;
    }
    check(written > 0, "text_tokenizer_empty");
    return written;
  }
};
Tokenizer::Tokenizer(std::shared_ptr<Resources> r, Workload workload)
    : resources_(std::move(r)), workload_(workload) {
  check(bool(resources_), "text_tokenizer_resources");
}
Tokenizer::~Tokenizer() { unload(); }
void Tokenizer::load(const std::string &path, const std::atomic_bool &cancel) {
  stop(cancel);
  check(!executing_, "busy");
  check(!impl_, "text_tokenizer_already_loaded");
  Admission admitted(*resources_, workload_, 768ULL * 1024 * 1024);
  auto next = std::make_unique<Impl>(path, cancel);
  stop(cancel);
  resources_->resize_loading(admitted.id,
                             host(*resources_, 384ULL * 1024 * 1024));
  impl_ = std::move(next);
  reservation_ = admitted.id;
  admitted.id = 0;
}
void Tokenizer::unload() {
  check(!executing_, "busy");
  impl_.reset();
  if (reservation_) {
    resources_->released(reservation_);
    reservation_ = 0;
  }
}
std::size_t Tokenizer::encode(const std::string &text,
                              std::span<std::uint32_t> output,
                              const std::atomic_bool &cancel,
                              std::size_t input_limit) {
  stop(cancel);
  check(bool(impl_), "text_tokenizer_not_loaded");
  check(!text.empty() && text.size() <= input_limit,
        "text_tokenizer_input_limit");
  Active active(executing_);
  check(text.size() <= std::numeric_limits<Bytes>::max() / 1024,
        "text_tokenizer_scratch_overflow");
  Admission temporary(*resources_, workload_,
                      std::max<Bytes>(32 * 1024 * 1024, text.size() * 1024));
  return impl_->encode(text, output, cancel);
}
std::string Tokenizer::decode(std::uint32_t token) const {
  check(bool(impl_) && token < impl_->decoded.size(),
        "text_tokenizer_decode_id");
  return impl_->decoded[token];
}
} // namespace kadan::text
