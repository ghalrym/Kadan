#include "kadan/text_tokenizer.hpp"
#include <array>
#include <cassert>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <unicode/unistr.h>
#include <unistd.h>

// Synthetic metadata only: no checkpoint weights, GPU, or inference.
int main() {
  using Json = nlohmann::json;
  const auto directory = std::filesystem::temp_directory_path() /
                         ("kadan-tokenizer-unit-" + std::to_string(getpid()));
  std::filesystem::create_directory(directory);
  const auto path = directory / "tokenizer.json";
  Json vocabulary = Json::object();
  int next = 256;
  for (int byte = 0; byte < 256; ++byte) {
    const auto code = (byte >= 33 && byte <= 126) ||
                              (byte >= 161 && byte <= 172) || byte >= 174
                          ? byte
                          : next++;
    icu::UnicodeString unicode;
    unicode.append(UChar32(code));
    std::string token;
    unicode.toUTF8String(token);
    vocabulary[token] = byte;
  }
  vocabulary["abc"] = 256;
  Json metadata = {
      {"normalizer", nullptr},
      {"pre_tokenizer",
       {{"type", "Sequence"},
        {"pretokenizers", Json::array({{{"type", "Split"},
                                        {"behavior", "Isolated"},
                                        {"invert", false},
                                        {"pattern", {{"Regex", ".+"}}}},
                                       {{"type", "ByteLevel"},
                                        {"add_prefix_space", false},
                                        {"use_regex", false}}})}}},
      {"post_processor", {{"type", "ByteLevel"}}},
      {"model",
       {{"type", "BPE"},
        {"dropout", nullptr},
        {"unk_token", nullptr},
        {"continuing_subword_prefix", nullptr},
        {"end_of_word_suffix", nullptr},
        {"byte_fallback", false},
        {"ignore_merges", true},
        {"vocab", vocabulary},
        {"merges", Json::array()}}},
      {"added_tokens", Json::array()}};
  auto resources = std::make_shared<kadan::Resources>(
      kadan::Footprint{1024ULL * 1024 * 1024});
  std::atomic_bool cancel{false};
  std::array<std::uint32_t, 10> output{};
  {
    std::ofstream(path) << metadata;
    kadan::text::Tokenizer tokenizer(resources, kadan::Workload::llm);
    tokenizer.load(path.string(), cancel);
    assert(tokenizer.encode("abc", output, cancel) == 1 && output[0] == 256);
    assert(tokenizer.decode(256) == "abc");
    tokenizer.unload();
    metadata["model"]["ignore_merges"] = false;
    std::ofstream(path) << metadata;
    tokenizer.load(path.string(), cancel);
    assert(tokenizer.encode("abc", output, cancel) == 3);
    assert(output[0] == 'a' && output[1] == 'b' && output[2] == 'c');
    std::string history(33000, 'a');
    std::vector<std::uint32_t> history_tokens(history.size());
    bool default_limit_rejected = false;
    try {
      tokenizer.encode(history, history_tokens, cancel);
    } catch (const std::runtime_error &) {
      default_limit_rejected = true;
    }
    assert(default_limit_rejected);
    assert(tokenizer.encode(history, history_tokens, cancel, 40000) ==
           history.size());
  }
  assert(resources->snapshot().used[0] == 0);
  std::filesystem::remove_all(directory);
  std::cout << "Synthetic BPE direct-vocabulary and merge fallback passed\n";
}
