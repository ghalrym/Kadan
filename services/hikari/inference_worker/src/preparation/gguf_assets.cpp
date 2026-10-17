#include "kadan/gguf_checkpoint.hpp"
#include "kadan/gguf_text.hpp"
#include "kadan/preparation.hpp"

namespace kadan::serving {
namespace {
// Byte-level BPE pre-tokenizers published with each checkpoint family. The
// vocabulary, merges and control tokens come from the GGUF tokenizer arrays.
struct Pretokenizer {
    const char* regex;
    bool nfc, ignore_merges;
};

Pretokenizer pretokenizer(const std::string& name) {
    if (name == "qwen35")
        return {
            R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)",
            true,
            false
        };
    if (name == "glm4")
        return {
            R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)",
            false,
            true
        };
    throw RequestError("Unsupported GGUF tokenizer", 422);
}

Json tokenizer(
    const checkpoint::GgufMetadata& metadata, const std::atomic_bool& cancel
) {
    const auto pre = pretokenizer(metadata.string("tokenizer.ggml.pre"));
    const auto tokens = metadata.strings("tokenizer.ggml.tokens");
    const auto types = metadata.integers("tokenizer.ggml.token_type");
    const auto merges = metadata.strings("tokenizer.ggml.merges");
    preparation_check(
        metadata.string("tokenizer.ggml.model") == "gpt2"
            && types.size() == tokens.size() && !tokens.empty()
            && tokens.size() <= 262144 && merges.size() <= 500000,
        "Unsupported GGUF tokenizer dimensions"
    );
    // Normal tokens form the BPE vocabulary; control and user-defined tokens
    // are added tokens; unused padding slots are left out.
    Json vocabulary = Json::object(), added = Json::array();
    for (std::size_t id = 0; id < tokens.size(); ++id) {
        preparation_cancel(cancel);
        preparation_check(
            tokens[id].size() <= 1024, "GGUF tokenizer entry exceeds limit"
        );
        if (types[id] == 1) {
            preparation_check(
                added.empty() && !vocabulary.contains(tokens[id]),
                "Invalid GGUF vocabulary order"
            );
            vocabulary[tokens[id]] = id;
        } else if (types[id] == 3 || types[id] == 4)
            added.push_back(
                {{"id", id},
                 {"content", tokens[id]},
                 {"single_word", false},
                 {"lstrip", false},
                 {"rstrip", false},
                 {"normalized", false},
                 {"special", types[id] == 3}}
            );
        else
            preparation_check(types[id] == 5, "Invalid GGUF token type");
    }
    const Json byte_level = {
        {"type", "ByteLevel"},
        {"add_prefix_space", false},
        {"trim_offsets", false},
        {"use_regex", false}
    };
    // The worker presents explicit prompt tokens: no BOS and no chat template.
    return {
        {"normalizer", pre.nfc ? Json{{"type", "NFC"}} : Json(nullptr)},
        {"added_tokens", added},
        {"pre_tokenizer",
         {{"type", "Sequence"},
          {"pretokenizers",
           Json::array(
               {{{"type", "Split"},
                 {"pattern", {{"Regex", pre.regex}}},
                 {"behavior", "Isolated"},
                 {"invert", false}},
                byte_level}
           )}}},
        {"post_processor", byte_level},
        {"model",
         {{"type", "BPE"},
          {"dropout", nullptr},
          {"unk_token", nullptr},
          {"continuing_subword_prefix", nullptr},
          {"end_of_word_suffix", nullptr},
          {"byte_fallback", false},
          {"ignore_merges", pre.ignore_merges},
          {"vocab", vocabulary},
          {"merges", merges}}}
    };
}
}  // namespace

void prepare_gguf_assets(
    const fs::path& root, const std::string& first_file, const std::atomic_bool& cancel
) {
    preparation_cancel(cancel);
    preparation_check(
        fs::path(first_file).filename() == first_file
            && !fs::is_symlink(root / first_file),
        "Invalid GGUF metadata filename"
    );
    checkpoint::GgufMetadata metadata((root / first_file).string(), cancel);
    std::string type;
    try {
        type = text::validate_metadata(metadata);
    } catch (const std::exception& error) {
        throw RequestError(
            std::string("Unsupported GGUF checkpoint: ") + error.what(), 422
        );
    }
    const auto architecture = metadata.string("general.architecture");
    const Json config = {
        {"model_type", type},
        {"kadan_checkpoint_format", "gguf"},
        {"max_position_embeddings", metadata.integer(architecture + ".context_length")}
    };
    const auto publish = [&](const fs::path& path, const Json& value) {
        preparation_check(!fs::is_symlink(path), "Derived GGUF asset is a symlink");
        if (fs::exists(path)) {
            preparation_check(
                read_document(path) == value,
                "Derived GGUF asset changed; inspect model storage.",
                503
            );
            return;
        }
        const auto temporary = fs::path(path.string() + ".worker.tmp");
        preparation_check(
            !fs::is_symlink(temporary), "Derived GGUF asset temporary is a symlink"
        );
        write_document(temporary, value);
        preparation_cancel(cancel);
        fs::rename(temporary, path);
    };
    // Derived assets sit beside the hashed downloads; the exact architecture
    // contract is validated before any remaining split is acquired.
    publish(root / "config.json", config);
    if (!fs::exists(root / "tokenizer.json"))
        publish(root / "tokenizer.json", tokenizer(metadata, cancel));
}
}  // namespace kadan::serving
