#include "kadan/text_tokenizer.hpp"

#include <array>
#include <cassert>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <unistd.h>

#include <nlohmann/json.hpp>
#include <unicode/unistr.h>

// Synthetic metadata only: no checkpoint weights, GPU, or inference.
int main() {
    using Json = nlohmann::json;
    const auto directory = std::filesystem::temp_directory_path()
                           / ("kadan-tokenizer-unit-" + std::to_string(getpid()));
    std::filesystem::create_directory(directory);
    const auto path = directory / "tokenizer.json";
    Json vocabulary = Json::object();
    int next = 256;
    for (int byte = 0; byte < 256; ++byte) {
        const auto code =
            (byte >= 33 && byte <= 126) || (byte >= 161 && byte <= 172) || byte >= 174
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
          {"pretokenizers",
           Json::array(
               {{{"type", "Split"},
                 {"behavior", "Isolated"},
                 {"invert", false},
                 {"pattern", {{"Regex", ".+"}}}},
                {{"type", "ByteLevel"},
                 {"add_prefix_space", false},
                 {"use_regex", false}}}
           )}}},
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
        {"added_tokens", Json::array()}
    };
    auto resources =
        std::make_shared<kadan::Resources>(kadan::Footprint{1024ULL * 1024 * 1024});
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
        } catch (const std::runtime_error&) {
            default_limit_rejected = true;
        }
        assert(default_limit_rejected);
        assert(
            tokenizer.encode(history, history_tokens, cancel, 40000) == history.size()
        );
    }
    assert(resources->snapshot().used[0] == 0);
    // Drama needs its 4,172 added tokens, without widening other formats.
    {
        auto drama = metadata;
        drama["normalizer"] = {{"type", "NFC"}};
        for (unsigned index = 0; index < 4172; ++index)
            drama["added_tokens"].push_back(
                {{"id", 248044u + index},
                 {"content", "<|drama:" + std::to_string(index) + "|>"},
                 {"single_word", false},
                 {"lstrip", false},
                 {"rstrip", false},
                 {"normalized", false},
                 {"special", true}}
            );
        std::ofstream(path) << drama;
        kadan::text::Tokenizer tokenizer(
            resources, kadan::Workload::tts, kadan::text::Tokenizer::Format::drama
        );
        tokenizer.load(path.string(), cancel);
        assert(tokenizer.encode("<|drama:4171|>", output, cancel) == 1);
        assert(output[0] == 252215);
        tokenizer.unload();
        kadan::text::Tokenizer ordinary(resources, kadan::Workload::llm);
        bool rejected = false;
        try {
            ordinary.load(path.string(), cancel);
        } catch (const std::runtime_error& error) {
            rejected = std::string(error.what()) == "text_tokenizer_added_limit";
        }
        assert(rejected && resources->snapshot().used[0] == 0);
    }
    // Muse Spark's published tokenizer contract, using a compact synthetic vocab.
    // Its ByteLevel postprocessor controls offsets; encode still presents explicit
    // special tokens verbatim and never inserts an implicit BOS.
    {
        auto muse = metadata;
        muse["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"] =
            R"MUSE([^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+)MUSE";
        muse["pre_tokenizer"]["pretokenizers"][1]["trim_offsets"] = true;
        muse["post_processor"] = {
            {"type", "ByteLevel"},
            {"add_prefix_space", true},
            {"trim_offsets", false},
            {"use_regex", true}
        };
        muse["decoder"] = {
            {"type", "ByteLevel"},
            {"add_prefix_space", true},
            {"trim_offsets", true},
            {"use_regex", true}
        };
        muse["model"]["ignore_merges"] = true;
        muse["model"]["fuse_unk"] = false;
        muse["added_tokens"] = Json::array();
        for (const auto& [token, id] : std::array<std::pair<const char*, unsigned>, 6>{
                 {{"<|begin_of_text|>", 200000},
                  {"<|end_of_text|>", 200001},
                  {"<|eom|>", 200007},
                  {"<|eot|>", 200008},
                  {"<|start|>", 200022},
                  {"<|message|>", 200023}}
             })
            muse["added_tokens"].push_back(
                {{"id", id},
                 {"content", token},
                 {"single_word", false},
                 {"lstrip", false},
                 {"rstrip", false},
                 {"normalized", false},
                 {"special", true}}
            );
        std::ofstream(path) << muse;
        kadan::text::Tokenizer tokenizer(resources, kadan::Workload::llm);
        tokenizer.load(path.string(), cancel);
        assert(tokenizer.encode("abc", output, cancel) == 1 && output[0] == 256);
        assert(tokenizer.encode("<|begin_of_text|>abc<|eom|>", output, cancel) == 3);
        assert(output[0] == 200000 && output[1] == 256 && output[2] == 200007);
        tokenizer.unload();
        auto rejects_muse = [&](const Json& invalid, const char* expected) {
            std::ofstream(path) << invalid;
            bool rejected = false;
            try {
                tokenizer.load(path.string(), cancel);
            } catch (const std::runtime_error& error) {
                rejected = std::string(error.what()) == expected;
            }
            assert(rejected && resources->snapshot().used[0] == 0);
        };
        auto invalid = muse;
        invalid["pre_tokenizer"]["pretokenizers"][1]["add_prefix_space"] = true;
        rejects_muse(invalid, "text_tokenizer_byte_contract");
        invalid = muse;
        invalid["post_processor"]["type"] = "TemplateProcessing";
        rejects_muse(invalid, "text_tokenizer_post");
        invalid = muse;
        invalid["normalizer"] = {{"type", "NFKC"}};
        rejects_muse(invalid, "text_tokenizer_normalizer");
    }
    // Match the fixed vocabulary dimensions with synthetic byte strings only.
    // No model files or network execution are involved.
    vocabulary.erase("abc");
    std::vector<std::string> tokens(256);
    for (auto it = vocabulary.begin(); it != vocabulary.end(); ++it)
        tokens[it.value().get<std::size_t>()] = it.key();
    Json merges = Json::array();
    for (std::size_t left = 0; tokens.size() < 151643; ++left) {
        for (std::size_t right = 0; right < 256 && tokens.size() < 151643; ++right) {
            auto merged = tokens[left] + tokens[right];
            vocabulary[merged] = tokens.size();
            merges.push_back(Json::array({tokens[left], tokens[right]}));
            tokens.push_back(std::move(merged));
        }
    }
    metadata["normalizer"] = {{"type", "NFC"}};
    metadata["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"] =
        R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)";
    metadata["model"]["vocab"] = vocabulary;
    metadata["model"]["merges"] = std::move(merges);
    metadata["model"]["continuing_subword_prefix"] = "";
    metadata["model"]["end_of_word_suffix"] = "";
    metadata["added_tokens"] = Json::array(
        {{{"content", "<tag>"},
          {"id", 151643},
          {"single_word", false},
          {"lstrip", false},
          {"rstrip", false},
          {"normalized", false}},
         {{"content", "<tag>more"},
          {"id", 151644},
          {"single_word", false},
          {"lstrip", false},
          {"rstrip", false},
          {"normalized", false}}}
    );
    using Tokenizer = kadan::text::Tokenizer;
    for (const auto format : {Tokenizer::Format::h3, Tokenizer::Format::tts}) {
        metadata["post_processor"] =
            format == Tokenizer::Format::h3
                ? Json{{"type", "ByteLevel"}, {"add_prefix_space", false}}
                : Json{
                      {"type", "TemplateProcessing"},
                      {"special_tokens", Json::object()},
                      {"single",
                       Json::array({{{"Sequence", {{"id", "A"}, {"type_id", 0}}}}})}
                  };
        auto bounded =
            std::make_shared<kadan::Resources>(kadan::Footprint{512ULL * 1024 * 1024});
        Tokenizer tokenizer(
            bounded,
            format == Tokenizer::Format::h3 ? kadan::Workload::video
                                            : kadan::Workload::tts,
            format
        );
        std::ofstream(path) << metadata;
        tokenizer.load(path.string(), cancel);
        assert(bounded->snapshot().used[0] == 192ULL * 1024 * 1024);
        assert(tokenizer.encode("abcd", output, cancel) == 2);
        assert(output[0] == vocabulary.at("ab") && output[1] == vocabulary.at("cd"));
        // Digits are separate regex pieces even though their merge exists.
        assert(
            tokenizer.encode("12", output, cancel) == 2 && output[0] == '1'
            && output[1] == '2'
        );
        assert(tokenizer.encode("e\xcc\x81", output, cancel) == 1);
        assert(output[0] == vocabulary.at(tokens[0xc3] + tokens[0xa9]));
        assert(
            tokenizer.encode("<tag>more", output, cancel) == 1 && output[0] == 151644
        );
        auto rejects = [](auto action, const char* expected) {
            bool rejected = false;
            try {
                action();
            } catch (const std::runtime_error& error) {
                rejected = std::string(error.what()) == expected;
            }
            assert(rejected);
        };
        rejects(
            [&] { tokenizer.encode("abcd", std::span(output).first(1), cancel); },
            "text_tokenizer_token_limit"
        );
        rejects(
            [&] { tokenizer.encode("\xff", output, cancel); }, "text_tokenizer_utf8"
        );
        rejects(
            [&] { tokenizer.encode(std::string(32001, 'a'), output, cancel, 40000); },
            "text_tokenizer_input_limit"
        );
        cancel = true;
        rejects(
            [&] { tokenizer.encode("abc", output, cancel); }, "text_tokenizer_cancelled"
        );
        cancel = false;
        assert(bounded->snapshot().used[0] == 192ULL * 1024 * 1024);
        tokenizer.unload();
        assert(bounded->snapshot().used[0] == 0);
        auto rejects_metadata = [&](const Json& invalid, const char* expected) {
            std::ofstream(path) << invalid;
            rejects([&] { tokenizer.load(path.string(), cancel); }, expected);
            assert(bounded->snapshot().used[0] == 0);
        };
        // Early format checks do not need a full vocabulary on disk.
        auto minimal = metadata;
        minimal["model"]["vocab"] = Json::object();
        minimal["model"]["merges"] = Json::array();
        auto invalid = minimal;
        invalid["normalizer"] = nullptr;
        rejects_metadata(invalid, "text_tokenizer_normalizer");
        invalid = minimal;
        invalid["post_processor"]["type"] = "Unsupported";
        rejects_metadata(invalid, "text_tokenizer_post");
        invalid = minimal;
        invalid["model"]["ignore_merges"] = true;
        rejects_metadata(invalid, "text_tokenizer_model");
        rejects_metadata(minimal, "text_tokenizer_vocab_size");
        invalid = metadata;
        invalid["added_tokens"][0]["id"] = 97;
        invalid["added_tokens"][0]["content"] = "a";
        rejects_metadata(invalid, "text_tokenizer_added");
    }
    std::filesystem::remove_all(directory);
    std::cout << "Shared BPE and H3/TTS format unit tests passed\n";
}
