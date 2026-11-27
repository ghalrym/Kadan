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
    Json pipeline{
        {"_class_name", "Flux2KleinPipeline"},
        {"is_distilled", true}
    };
    Json scheduler{
        {"_class_name", "FlowMatchEulerDiscreteScheduler"},
        {"shift", 1.0},
        {"shift_terminal", nullptr},
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
        {"model_type", "qwen3"},
        {"vocab_size", 151936},
        {"hidden_size", 2560},
        {"num_hidden_layers", 36},
        {"num_attention_heads", 32},
        {"num_key_value_heads", 8},
        {"head_dim", 128},
        {"intermediate_size", 9728},
        {"rope_theta", 1000000},
        {"rms_norm_eps", 1e-6},
        {"attention_bias", false}
    };
    Json vae{
        {"_class_name", "AutoencoderKLFlux2"},
        {"in_channels", 3},
        {"out_channels", 3},
        {"latent_channels", 32},
        {"block_out_channels", {128, 256, 512, 512}},
        {"layers_per_block", 2},
        {"norm_num_groups", 32},
        {"mid_block_add_attention", true},
        {"use_quant_conv", true},
        {"use_post_quant_conv", true},
        {"patch_size", {2, 2}},
        {"batch_norm_eps", 1e-4}
    };
    Json transformer{
        {"num_layers", 5},
        {"num_single_layers", 20},
        {"num_attention_heads", 24},
        {"attention_head_dim", 128},
        {"in_channels", 128},
        {"joint_attention_dim", 7680},
        {"axes_dims_rope", {32, 32, 32, 32}},
        {"rope_theta", 2000},
        {"guidance_embeds", false},
        {"mlp_ratio", 3.0},
        {"eps", 1e-6}
    };

    Checkpoint() {
        char name[] = "/tmp/kadan-image-config-XXXXXX";
        const auto* dir = mkdtemp(name);
        check(dir != nullptr);
        root = dir;
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
        write("vae/config.json", vae);
        write("transformer/config.json", transformer);
        return kadan::image::validate_checkpoint_config(root.string());
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

void klein_profile() {
    Checkpoint checkpoint;
    const auto config = checkpoint.load();
    check(config.text_files.size() == 2);
    check(config.text_files.front() == "model-00001-of-00002.safetensors");
    check(config.denoiser_files == std::vector<std::string>{"diffusion_pytorch_model.safetensors"});
    check(config.default_steps == 4);
    check(std::abs(config.batch_norm_eps - 1e-4f) < 1e-9f);
    const auto actual = config.sampling_schedule(4096, 4);
    const auto expected = kadan::image::schedule(4096, 4);
    check(actual.steps == 4 && actual.sigma == expected.sigma);
}

void rejected_profiles() {
    {
        Checkpoint cp;
        cp.pipeline["_class_name"] = "QwenImage21Pipeline";
        cp.rejects("image_pipeline_contract");
    }
    {
        Checkpoint cp;
        cp.pipeline["is_distilled"] = false;
        cp.rejects("image_pipeline_contract");
    }
    for (const auto& field : {"hidden_size", "num_hidden_layers", "intermediate_size", "rope_theta"}) {
        Checkpoint cp;
        cp.text[field] = 1;
        cp.rejects("image_text_config_contract");
    }
    for (const auto& field : {"latent_channels", "layers_per_block", "batch_norm_eps"}) {
        Checkpoint cp;
        cp.vae[field] = 0;
        cp.rejects("image_vae_config_contract");
    }
    for (const auto& field : {"num_layers", "num_single_layers", "in_channels", "joint_attention_dim"}) {
        Checkpoint cp;
        cp.transformer[field] = 1;
        cp.rejects("image_transformer_contract");
    }
    {
        Checkpoint cp;
        cp.transformer["guidance_embeds"] = true;
        cp.rejects("image_transformer_contract");
    }
    for (const auto& field : {"use_dynamic_shifting", "shift_terminal", "use_karras_sigmas"}) {
        Checkpoint cp;
        cp.scheduler[field] = std::string_view(field) == "shift_terminal" ? Json(.02)
                            : std::string_view(field) == "use_dynamic_shifting" ? Json(false)
                            : Json(true);
        cp.rejects("image_scheduler_contract");
    }
}
}  // namespace

int main() {
    klein_profile();
    rejected_profiles();
}
