#include <algorithm>
#include <cmath>

#include "kadan/gguf_checkpoint.hpp"
#include "kadan/muse_spark.hpp"
#include "kadan/preparation.hpp"

namespace kadan::serving {
void prepare_muse_gguf_assets(
    const fs::path& root, const std::string& first_split, const std::atomic_bool& cancel
) {
    preparation_cancel(cancel);
    preparation_check(
        fs::path(first_split).filename() == first_split
            && !fs::is_symlink(root / first_split),
        "Invalid Muse GGUF metadata filename"
    );
    checkpoint::GgufMetadata metadata((root / first_split).string(), cancel);
    preparation_check(
        metadata.tensor_count() == 0 && metadata.integer("split.no") == 0
            && metadata.string("general.architecture") == "muse-spark"
            && metadata.string("tokenizer.ggml.model") == "gpt2"
            && metadata.string("tokenizer.ggml.pre") == "gpt-4o",
        "Unsupported Muse GGUF metadata"
    );
    const auto number = [&](const char* key, double expected) {
        preparation_check(
            std::abs(metadata.number(key) - expected)
                <= std::max(1e-12, std::abs(expected) * 1e-6),
            "Unsupported Muse GGUF arithmetic"
        );
        return expected;
    };
    Json text = {
        {"model_type", "muse_spark_text"},
        {"hidden_size", metadata.integer("muse-spark.embedding_length")},
        {"num_hidden_layers", metadata.integer("muse-spark.block_count")},
        {"num_attention_heads", metadata.integer("muse-spark.attention.head_count")},
        {"num_key_value_heads", metadata.integer("muse-spark.attention.head_count_kv")},
        {"head_dim", metadata.integer("muse-spark.attention.key_length")},
        {"vocab_size", 202048},
        {"intermediate_size", metadata.integer("muse-spark.feed_forward_length")},
        {"moe_hidden_size", metadata.integer("muse-spark.moe_latent_size")},
        {"moe_intermediate_size",
         metadata.integer("muse-spark.expert_feed_forward_length")},
        {"num_local_experts", metadata.integer("muse-spark.expert_count")},
        {"num_experts_per_tok", metadata.integer("muse-spark.expert_used_count")},
        {"num_shared_experts", 0},
        {"sliding_window", metadata.integer("muse-spark.attention.sliding_window")},
        {"max_position_embeddings", metadata.integer("muse-spark.context_length")},
        {"bos_token_id", metadata.integer("tokenizer.ggml.bos_token_id")},
        {"eos_token_id", metadata.integer("tokenizer.ggml.eos_token_id")},
        {"pad_token_id", metadata.integer("tokenizer.ggml.padding_token_id")},
        {"attention_bias", false},
        {"attention_dropout", 0},
        {"tie_word_embeddings", false},
        {"use_cache", true},
        {"hidden_act", "silu"},
        {"rms_norm_eps", number("muse-spark.attention.layer_norm_rms_epsilon", 1e-5)},
        {"post_norm_eps", number("muse-spark.post_norm_epsilon", 1e-8)},
        {"residual_gate_temperature", 0.3},
        {"routed_scaling_factor", number("muse-spark.expert_weights_scale", 1)},
        {"final_logit_softcapping", number("muse-spark.final_logit_softcapping", 20)},
        {"output_multiplier", number("muse-spark.logit_scale", 0.17677669529663687)},
        {"qk_scale_factor", 5.4730064863875},
        {"rope_parameters",
         {{"rope_type", "default"},
          {"rope_theta", number("muse-spark.rope.freq_base_swa", 500000)}}},
        {"layer_types", Json::array()},
        {"layer_rope_theta", Json::array()},
        {"mlp_layer_types", Json::array()}
    };
    const auto sliding =
        metadata.booleans("muse-spark.attention.sliding_window_pattern");
    preparation_check(sliding.size() == 62, "Unsupported Muse GGUF layer schedule");
    for (const bool local : sliding) {
        text["layer_types"].push_back(local ? "sliding_attention" : "full_attention");
        text["layer_rope_theta"].push_back(local ? 500000 : 0);
        text["mlp_layer_types"].push_back("sparse");
    }
    preparation_check(
        metadata.integer("tokenizer.ggml.eot_token_id") == 200008
            && metadata.boolean("muse-spark.expert_weights_norm")
            && metadata.integer("muse-spark.expert_gating_func") == 2,
        "Unsupported Muse GGUF routing or stop tokens"
    );
    const Json config = {
        {"model_type", "muse_spark"},
        {"dtype", "float32"},
        {"architectures", {"MuseSparkForConditionalGeneration"}},
        {"kadan_checkpoint_format", "gguf"},
        {"text_config", text}
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
    // Derived metadata is kept separate from the hashed downloaded splits.
    // Validate the exact decoder contract before any weight split is acquired.
    publish(root / "config.json", config);
    auto budget = std::make_shared<checkpoint::MemoryBudget>(4 << 20);
    kadan::muse_spark::validate_config(root.c_str(), budget);
    publish(root / "generation_config.json", {{"eos_token_id", {200001, 200008}}});

    preparation_check(
        !fs::is_symlink(root / "tokenizer.json"), "Derived GGUF tokenizer is a symlink"
    );
    if (fs::exists(root / "tokenizer.json"))
        return;
    const auto tokens = metadata.strings("tokenizer.ggml.tokens");
    const auto types = metadata.integers("tokenizer.ggml.token_type");
    const auto merges = metadata.strings("tokenizer.ggml.merges");
    preparation_check(
        tokens.size() == 202048 && types.size() == tokens.size()
            && merges.size() == 439802,
        "Unsupported Muse GGUF tokenizer dimensions"
    );
    Json vocabulary = Json::object(), added = Json::array();
    for (std::size_t id = 0; id < tokens.size(); ++id) {
        preparation_cancel(cancel);
        preparation_check(
            tokens[id].size() <= 1024, "Muse GGUF tokenizer entry exceeds limit"
        );
        if (id < 200000) {
            preparation_check(
                types[id] == 1 && !vocabulary.contains(tokens[id]),
                "Invalid Muse GGUF vocabulary"
            );
            vocabulary[tokens[id]] = id;
        } else if (id < 201818) {
            preparation_check(types[id] == 3, "Invalid Muse GGUF control token");
            added.push_back(
                {{"id", id},
                 {"content", tokens[id]},
                 {"single_word", false},
                 {"lstrip", false},
                 {"rstrip", false},
                 {"normalized", false},
                 {"special", true}}
            );
        } else
            preparation_check(types[id] == 5, "Invalid Muse GGUF unused token");
    }
    for (const auto& [id, token] :
         std::initializer_list<std::pair<std::size_t, const char*>>{
             {200000, "<|begin_of_text|>"},
             {200001, "<|end_of_text|>"},
             {200007, "<|eom|>"},
             {200008, "<|eot|>"},
             {200018, "<|finetune_right_pad|>"},
             {200022, "<|start|>"},
             {200023, "<|message|>"}
         })
        preparation_check(tokens[id] == token, "Unsupported Muse GGUF control tokens");
    // Published Muse/gpt-4o byte-level BPE contract. The worker presents its
    // explicit prompt tokens; this tokenizer inserts no BOS or chat template.
    constexpr auto pattern =
        R"MUSE([^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+)MUSE";
    const Json tokenizer = {
        {"normalizer", nullptr},
        {"added_tokens", added},
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
          {"merges", merges}}}
    };
    publish(root / "tokenizer.json", tokenizer);
}
}  // namespace kadan::serving
