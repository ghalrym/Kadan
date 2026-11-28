#include <fstream>
#include <sstream>
#include <string>

#include "kadan/preparation.hpp"

namespace kadan::serving {
void export_speech_tokenizer(
    const fs::path& checkpoint,
    const fs::path& destination,
    const std::atomic_bool& cancel
) {
    preparation_cancel(cancel);
    preparation_check(!fs::exists(destination), "Speech tokenizer destination exists");
    const auto vocab = read_document(checkpoint / "vocab.json"),
               config = read_document(checkpoint / "tokenizer_config.json");
    Json merges = Json::array();
    std::istringstream input(read_bytes(checkpoint / "merges.txt", 32 * 1024 * 1024));
    std::string line;
    while (std::getline(input, line)) {
        preparation_cancel(cancel);
        if (!line.empty() && line.back() == '\r')
            line.pop_back();
        if (line.empty() || line.starts_with("#version:"))
            continue;
        const auto split = line.find(' ');
        preparation_check(
            split != std::string::npos
                && line.find(' ', split + 1) == std::string::npos,
            "Invalid speech tokenizer merge"
        );
        merges.push_back({line.substr(0, split), line.substr(split + 1)});
    }
    Json added = Json::array();
    for (auto it = config.at("added_tokens_decoder").begin();
         it != config.at("added_tokens_decoder").end();
         ++it) {
        preparation_check(
            !it.key().empty()
                && it.key().find_first_not_of("0123456789") == std::string::npos,
            "Invalid speech token ID"
        );
        auto token = it.value();
        token["id"] = std::stoul(it.key());
        added.push_back(token);
    }
    const std::string pattern =
        R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)";
    Json tokenizer = {
        {"version", "1.0"},
        {"truncation", nullptr},
        {"padding", nullptr},
        {"added_tokens", added},
        {"normalizer", {{"type", "NFC"}}},
        {"pre_tokenizer",
         {{"type", "Sequence"},
          {"pretokenizers",
           Json::array(
               {{{"type", "Split"},
                 {"pattern", {{"Regex", pattern}}},
                 {"behavior", "Isolated"},
                 {"invert", false}},
                {{"type", "ByteLevel"},
                 {"add_prefix_space", false},
                 {"trim_offsets", true},
                 {"use_regex", false}}}
           )}}},
        {"post_processor",
         {{"type", "TemplateProcessing"},
          {"single", Json::array({{{"Sequence", {{"id", "A"}, {"type_id", 0}}}}})},
          {"pair",
           Json::array(
               {{{"Sequence", {{"id", "A"}, {"type_id", 0}}}},
                {{"Sequence", {{"id", "B"}, {"type_id", 1}}}}}
           )},
          {"special_tokens", Json::object()}}},
        {"decoder",
         {{"type", "ByteLevel"},
          {"add_prefix_space", true},
          {"trim_offsets", true},
          {"use_regex", true}}},
        {"model",
         {{"type", "BPE"},
          {"dropout", nullptr},
          {"unk_token", nullptr},
          {"continuing_subword_prefix", ""},
          {"end_of_word_suffix", ""},
          {"fuse_unk", false},
          {"byte_fallback", false},
          {"ignore_merges", false},
          {"vocab", vocab},
          {"merges", merges}}}
    };
    fs::create_directories(destination.parent_path());
    const auto stage = fs::path(destination.string() + ".partial");
    fs::remove_all(stage);
    fs::create_directory(stage);
    try {
        write_document(stage / "tokenizer.json", tokenizer);
        preparation_cancel(cancel);
        fs::rename(stage, destination);
    } catch (...) {
        fs::remove_all(stage);
        throw;
    }
}
}  // namespace kadan::serving
