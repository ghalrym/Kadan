// Configuration validation only: no checkpoint tensors, Generator::load, model
// forward, generation, codec decoding, or GPU access.
#include "kadan/drama_model.hpp"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <source_location>
#include <stdexcept>
#include <string>

#include <nlohmann/json.hpp>

namespace {
using Json = nlohmann::json;
namespace fs = std::filesystem;

void check(bool ok, std::source_location at = std::source_location::current()) {
    if (!ok)
        throw std::runtime_error("drama_metadata_test:" + std::to_string(at.line()));
}

// Minimal metadata describing the published original Drama 3 geometry; no model
// weights are synthesized. Unknown shapes must be rejected before inference.
Json original_config() {
    Json text{
        {"model_type", "fish_omni_text"},
        {"hidden_size", 2048},
        {"num_hidden_layers", 40},
        {"num_attention_heads", 16},
        {"num_key_value_heads", 2},
        {"head_dim", 256},
        {"vocab_size", 252288},
        {"linear_conv_kernel_dim", 4},
        {"linear_key_head_dim", 128},
        {"linear_value_head_dim", 128},
        {"linear_num_key_heads", 16},
        {"linear_num_value_heads", 32},
        {"num_experts", 256},
        {"num_experts_per_tok", 8},
        {"moe_intermediate_size", 512},
        {"shared_expert_intermediate_size", 512},
        {"rms_norm_eps", 1e-6},
        {"hidden_act", "silu"},
        {"tie_word_embeddings", false},
        {"rotary_interleaved", false},
        {"use_qk_norm", true},
        {"attn_output_gate", true},
        {"layernorm_zero_centered_gamma", true},
        {"shared_expert_gate", true},
        {"attention_bias", false},
        {"max_position_embeddings", 262144},
        {"rope_parameters",
         {{"rope_type", "default"},
          {"rope_theta", 1e7},
          {"partial_rotary_factor", .25}}},
        {"layer_types", Json::array()},
        {"mlp_layer_types", Json::array()}
    };
    for (std::size_t i = 0; i < 40; ++i) {
        text["layer_types"].push_back(
            (i + 1) % 4 ? "linear_attention" : "full_attention"
        );
        text["mlp_layer_types"].push_back("moe");
    }
    Json decoder{
        {"model_type", "fish_qwen3_audio_decoder"},
        {"decoder_type", "codebook_mtp_shared"},
        {"dim", 2048},
        {"text_dim", 2048},
        {"n_head", 16},
        {"n_local_heads", 2},
        {"head_dim", 256},
        {"n_layer", 4},
        {"shared_enrichment_layers", 4},
        {"intermediate_size", 6144},
        {"num_codebooks", 10},
        {"vocab_size", 4096},
        {"max_seq_len", 11},
        {"vq_backbone_codebooks", 1},
        {"norm_eps", 1e-6},
        {"rope_base", 1e7},
        {"rotary_percent", .25},
        {"layernorm_zero_centered_gamma", true},
        {"attention_output_gate", true},
        {"vq_feature_scope", "all_vq"},
        {"vq_feature_placement", "backbone_partial"},
        {"mtp_spec", {{"attention_variant", "full_attention"}}}
    };
    return {
        {"model_type", "fish_omni"},
        {"fish_schema_version", 1},
        {"speech_template_version", "reference-bracket-v1"},
        {"semantic_start_token_id", 248099},
        {"semantic_end_token_id", 252194},
        {"eos_token_id", 248046},
        {"text_config", text},
        {"audio_decoder_config", decoder},
        {"quantization_config",
         {{"quant_method", "bsgemm-moe"},
          {"weight_block_size", Json::array({128, 128})},
          {"activation_scheme", "dynamic"}}}
    };
}

struct ConfigDirectory {
    fs::path root;

    ConfigDirectory() {
        auto pattern =
            (fs::temp_directory_path() / "kadan-drama-metadata-XXXXXX").string();
        auto* made = mkdtemp(pattern.data());
        check(made != nullptr);
        root = made;
    }

    ~ConfigDirectory() {
        std::error_code ignored;
        fs::remove_all(root, ignored);
    }

    void save(const Json& config) const {
        std::ofstream file(root / "config.json");
        file << config;
        check(bool(file));
    }

    void reject(const Json& config, const char* message) const {
        save(config);
        try {
            (void)kadan::drama::inspect(root);
        } catch (const std::runtime_error& error) {
            check(std::string(error.what()) == message);
            return;
        }
        check(false);
    }
};
}  // namespace

int main() {
    ConfigDirectory directory;
    const auto original = original_config();
    directory.save(original);
    const auto info = kadan::drama::inspect(directory.root);
    check(info.fp8_experts && info.codebooks == 10 && info.codebook_size == 4096);
    check(
        info.context_limit == 4096 && info.semantic_begin == 248099
        && info.eos == 248046
    );
    // Independent admission arithmetic: fixed metadata+scratch, 30 recurrent
    // layers (state+convolution), and 23 full-attention KV histories.
    check(info.host_bytes == 720175104);
    const auto small = kadan::drama::inspect(directory.root, 1024);
    check(info.host_bytes - small.host_bytes == 23ULL * 2 * (4096 - 1024) * 512 * 4);

    auto changed = original;
    changed["quantization_config"] = nullptr;
    directory.save(changed);
    check(!kadan::drama::inspect(directory.root).fp8_experts);

    changed = original;
    changed["quantization_config"]["quant_method"] = "drama3-int4";
    directory.reject(changed, "drama_int4_gptq_marlin_unsupported");
    changed = original;
    changed["text_config"]["rope_parameters"]["partial_rotary_factor"] = .5;
    directory.reject(changed, "drama_configuration_unsupported");
    changed = original;
    changed["quantization_config"]["weight_block_size"] = {64, 128};
    directory.reject(changed, "drama_configuration_unsupported");
    changed = original;
    changed["text_config"]["layer_types"][0] = "full_attention";
    directory.reject(changed, "drama_layer_schedule");
    changed = original;
    changed["audio_decoder_config"]["use_moe"] = true;
    directory.reject(changed, "drama_decoder_options_unsupported");

    directory.save(original);
    for (const auto context : {std::size_t(0), std::size_t(4097)}) {
        bool rejected = false;
        try {
            (void)kadan::drama::inspect(directory.root, context);
        } catch (const std::runtime_error& error) {
            rejected = std::string(error.what()) == "drama_context_limit";
        }
        check(rejected);
    }
}
