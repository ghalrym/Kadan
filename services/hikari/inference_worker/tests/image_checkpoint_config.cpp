// Configuration and scalar schedule units only; no weights or model forward.
#include "kadan/image_checkpoint_config.hpp"

#include <filesystem>
#include <fstream>
#include <source_location>
#include <string_view>
#include <unistd.h>

#include <nlohmann/json.hpp>

namespace {
using Json = nlohmann::json;
namespace fs = std::filesystem;

void check(bool ok, std::source_location at = std::source_location::current()) {
    if (!ok)
        throw std::runtime_error("image_config_test:" + std::to_string(at.line()));
}

struct Checkpoint {
    fs::path root;
    Json pipeline{{"_class_name", "QwenImage21Pipeline"}},
        scheduler{
            {"_class_name", "FlowMatchEulerDiscreteScheduler"},
            {"base_image_seq_len", 256},
            {"max_image_seq_len", 8192},
            {"base_shift", .5},
            {"max_shift", .9},
            {"shift", 1},
            {"shift_terminal", .02},
            {"time_shift_type", "exponential"},
            {"use_dynamic_shifting", true},
            {"num_train_timesteps", 1000},
            {"invert_sigmas", false},
            {"stochastic_sampling", false},
            {"use_beta_sigmas", false},
            {"use_exponential_sigmas", false},
            {"use_karras_sigmas", false}
        };

    Json text{
        {"model_type", "qwen3_vl"},
        {"image_token_id", 151655},
        {"vision_start_token_id", 151652},
        {"vision_end_token_id", 151653},
        {"text_config",
         {{"vocab_size", 151936},
          {"hidden_size", 4096},
          {"num_hidden_layers", 36},
          {"num_attention_heads", 32},
          {"num_key_value_heads", 8},
          {"head_dim", 128},
          {"intermediate_size", 12288},
          {"rope_theta", 5000000},
          {"rms_norm_eps", 1e-6},
          {"attention_bias", false},
          {"rope_scaling",
           {{"mrope_interleaved", true},
            {"mrope_section", {24, 20, 20}},
            {"rope_type", "default"}}}}},
        {"vision_config",
         {{"model_type", "qwen3_vl"},
          {"depth", 27},
          {"hidden_size", 1152},
          {"num_heads", 16},
          {"intermediate_size", 4304},
          {"out_hidden_size", 4096},
          {"in_channels", 3},
          {"patch_size", 16},
          {"spatial_merge_size", 2},
          {"temporal_patch_size", 2},
          {"num_position_embeddings", 2304},
          {"hidden_act", "gelu_pytorch_tanh"},
          {"deepstack_visual_indexes", {8, 16, 24}}}}
    };

    Checkpoint() {
        char name[] = "/tmp/kadan-image-config-XXXXXX";
        const auto* dir = mkdtemp(name);
        check(dir != nullptr);
        root = dir;
        write(
            "vae/config.json",
            {{"decoder_base_dim", 144},
             {"base_dim", 96},
             {"z_dim", 64},
             {"dim_mult", {1, 2, 4, 8, 8}},
             {"num_res_blocks", 2},
             {"out_channels", 4},
             {"in_channels", 4},
             {"is_residual", true},
             {"patch_size", nullptr},
             {"temperal_downsample", {false, true, true, true}},
             {"latents_mean", std::vector<float>(64, 0)},
             {"latents_std", std::vector<float>(64, 1)}}
        );
        write(
            "transformer/config.json",
            {{"num_layers", 32},
             {"num_attention_heads", 32},
             {"attention_head_dim", 128},
             {"in_channels", 64},
             {"context_in_dim", 4096},
             {"axes_dims_rope", {16, 56, 56}},
             {"patch_size", 1},
             {"mlp_ratio", 3},
             {"eps", 1e-6},
             {"causal_condition", true},
             {"out_channels", 64}}
        );
    }

    ~Checkpoint() {
        std::error_code error;
        fs::remove_all(root, error);
    }

    void write(const std::string& path, const Json& data) const {
        fs::create_directories((root / path).parent_path());
        std::ofstream(root / path) << data;
    }

    kadan::image::ImageCheckpointConfig load() const {
        write("text_encoder/config.json", text);
        write("model_index.json", pipeline);
        write("scheduler/scheduler_config.json", scheduler);
        return kadan::image::validate_checkpoint_config(root.string());
    }

    void turbo() {
        // Official grid at fc13ba1986fcb905854dc8de2f52eee395f49b5b.
        pipeline["sample_sigmas"] = {
            1.0, .978453, .95418, .926626, .89508, .845148, .704534, .414568
        };
        scheduler["use_dynamic_shifting"] = false;
        scheduler["shift_terminal"] = nullptr;
    }

    void rejects(std::string_view expected) const {
        bool rejected = false;
        try {
            (void)load();
        } catch (const std::runtime_error& error) {
            rejected = error.what() == expected;
        }
        check(rejected);
    }
};

void base_profile() {
    Checkpoint checkpoint;
    for (bool explicit_null : {false, true}) {
        if (explicit_null)
            checkpoint.pipeline["sample_sigmas"] = nullptr;
        const auto config = checkpoint.load();
        check(config.text_files.size() == 4);
        check(config.text_files.front() == "model-00001-of-00004.safetensors");
        check(config.sample_schedule.steps == 0);
        for (std::size_t steps : {2u, 40u, 100u}) {
            const auto actual = config.sampling_schedule(16384, steps);
            const auto original = kadan::image::schedule(16384, steps);
            check(actual.steps == steps && actual.sigma == original.sigma);
        }
    }
}

void turbo_profile() {
    Checkpoint checkpoint;
    checkpoint.turbo();
    const auto config = checkpoint.load();
    check(config.text_files == std::vector<std::string>{"model.safetensors"});
    const std::array<float, 9> expected{
        1, .978453f, .95418f, .926626f, .89508f, .845148f, .704534f, .414568f, 0
    };
    // Neither requested steps nor canvas size transforms the saved Turbo grid.
    for (std::size_t tokens : {16384u, 18432u})
        for (std::size_t steps : {2u, 8u, 50u, 100u}) {
            const auto actual = config.sampling_schedule(tokens, steps);
            check(actual.steps == 8);
            for (std::size_t i = 0; i < expected.size(); ++i)
                check(actual.sigma[i] == expected[i]);
            check(actual.sigma[7] != .02f);
        }
}

void rejected_multimodal_profiles() {
    for (const auto& field :
         {"depth",
          "hidden_size",
          "num_heads",
          "intermediate_size",
          "out_hidden_size",
          "in_channels",
          "patch_size",
          "spatial_merge_size",
          "temporal_patch_size",
          "num_position_embeddings"}) {
        Checkpoint checkpoint;
        checkpoint.text["vision_config"][field] = 1;
        checkpoint.rejects("image_vision_config_contract");
    }
    for (const auto& field : {"model_type", "hidden_act", "deepstack_visual_indexes"}) {
        Checkpoint checkpoint;
        checkpoint.text["vision_config"][field] = "unsupported";
        checkpoint.rejects("image_vision_config_contract");
    }
    for (const auto& field :
         {"image_token_id", "vision_start_token_id", "vision_end_token_id"}) {
        Checkpoint checkpoint;
        checkpoint.text[field] = 0;
        checkpoint.rejects("image_multimodal_config_contract");
    }
    for (const auto& field : {"mrope_interleaved", "mrope_section", "rope_type"}) {
        Checkpoint checkpoint;
        checkpoint.text["text_config"]["rope_scaling"][field] = nullptr;
        checkpoint.rejects("image_multimodal_config_contract");
    }
}

void rejected_profiles() {
    for (const auto& sigmas : std::array<Json, 7>{
             Json::array(),
             Json::array({1, .9}),
             Json::array({1, .9, .8, .7, .6, .5, .4, 0}),
             Json::array({.99, .9, .8, .7, .6, .5, .4, .3}),
             Json::array({1, .9, .8, .7, .7, .5, .4, .3}),
             Json::array({1, .9, .8, .7, .6, .5, .4, "bad"}),
             Json::array({1, .9, .8, .7, .6, .5, .4, 1e100})
         }) {
        Checkpoint checkpoint;
        checkpoint.turbo();
        checkpoint.pipeline["sample_sigmas"] = sigmas;
        checkpoint.rejects("image_sample_sigmas");
    }
    for (const auto& field :
         {"use_dynamic_shifting", "shift_terminal", "shift", "use_karras_sigmas"}) {
        Checkpoint checkpoint;
        checkpoint.turbo();
        checkpoint.scheduler[field] =
            std::string_view(field) == "shift_terminal" ? Json(.02) : Json(true);
        checkpoint.rejects("image_scheduler_contract");
    }
    Checkpoint base;
    base.scheduler["use_dynamic_shifting"] = false;
    base.rejects("image_scheduler_contract");
}
}  // namespace

int main() {
    base_profile();
    turbo_profile();
    rejected_profiles();
    rejected_multimodal_profiles();
}
