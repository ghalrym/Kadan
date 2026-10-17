#include "kadan/image_checkpoint_config.hpp"

#include <cmath>
#include <fstream>
#include <stdexcept>

#include <nlohmann/json.hpp>

namespace kadan::image {
namespace {
void need(bool b, const char* error) {
    if (!b)
        throw std::runtime_error(error);
}

using Json = nlohmann::json;

Json read(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    need(bool(f) && f.tellg() > 0 && f.tellg() <= 65536, "image_config_size");
    std::string s(std::size_t(f.tellg()), '\0');
    f.seekg(0);
    f.read(s.data(), s.size());
    need(bool(f), "image_config_read");
    return Json::parse(s);
}

}  // namespace

ImageCheckpointConfig validate_checkpoint_config(const std::string& root) {
    ImageCheckpointConfig result;
    const auto pipeline = read(root + "/model_index.json");
    need(
        pipeline.at("_class_name") == "QwenImage21Pipeline", "image_pipeline_contract"
    );
    const auto sigmas = pipeline.value("sample_sigmas", Json());
    const bool turbo = !sigmas.is_null();
    if (turbo) {
        need(sigmas.is_array() && sigmas.size() == 8, "image_sample_sigmas");
        result.sample_schedule.steps = sigmas.size();
        for (std::size_t i = 0; i < sigmas.size(); ++i) {
            need(sigmas[i].is_number(), "image_sample_sigmas");
            const float sigma = sigmas[i];
            need(
                std::isfinite(sigma) && sigma > 0 && sigma <= 1
                    && (i == 0 ? sigma == 1
                               : sigma < result.sample_schedule.sigma[i - 1]),
                "image_sample_sigmas"
            );
            result.sample_schedule.sigma[i] = sigma;
        }
        // Euler's terminal zero is appended by Diffusers, not saved in the grid.
        result.sample_schedule.sigma[sigmas.size()] = 0;
        result.text_files = {"model.safetensors"};
    } else {
        result.text_files = {
            "model-00001-of-00004.safetensors",
            "model-00002-of-00004.safetensors",
            "model-00003-of-00004.safetensors",
            "model-00004-of-00004.safetensors"
        };
    }
    auto text = read(root + "/text_encoder/config.json").at("text_config");
    need(
        text.at("vocab_size") == 151936 && text.at("hidden_size") == 4096
            && text.at("num_hidden_layers") == 36
            && text.at("num_attention_heads") == 32
            && text.at("num_key_value_heads") == 8 && text.at("head_dim") == 128
            && text.at("intermediate_size") == 12288 && text.at("rope_theta") == 5000000
            && text.at("rms_norm_eps") == 1e-6
            && !text.at("attention_bias").get<bool>(),
        "image_text_config_contract"
    );
    auto v = read(root + "/vae/config.json");
    need(
        v.at("decoder_base_dim") == 144 && v.at("base_dim") == 96 && v.at("z_dim") == 64
            && v.at("dim_mult") == Json::array({1, 2, 4, 8, 8})
            && v.at("num_res_blocks") == 2 && v.at("out_channels") == 4
            && v.at("in_channels") == 4 && v.at("is_residual") == true
            && v.at("patch_size").is_null()
            && v.at("temperal_downsample") == Json::array({false, true, true, true}),
        "image_vae_config_contract"
    );
    need(
        v.at("latents_mean").size() == 64 && v.at("latents_std").size() == 64,
        "image_latent_normalization"
    );
    for (std::size_t i = 0; i < 64; ++i) {
        result.mean[i] = v.at("latents_mean")[i];
        result.stddev[i] = v.at("latents_std")[i];
        need(
            std::isfinite(result.mean[i]) && std::isfinite(result.stddev[i])
                && result.stddev[i] > 0,
            "image_latent_normalization"
        );
    }
    auto s = read(root + "/scheduler/scheduler_config.json");
    need(
        s.at("_class_name") == "FlowMatchEulerDiscreteScheduler"
            && s.at("base_image_seq_len") == 256 && s.at("max_image_seq_len") == 8192
            && s.at("base_shift") == .5 && s.at("max_shift") == .9 && s.at("shift") == 1
            && s.at("time_shift_type") == "exponential"
            && s.at("use_dynamic_shifting") == !turbo
            && (turbo ? s.at("shift_terminal").is_null()
                      : s.at("shift_terminal") == .02)
            && s.at("num_train_timesteps") == 1000,
        "image_scheduler_contract"
    );
    for (auto key :
         {"invert_sigmas",
          "stochastic_sampling",
          "use_beta_sigmas",
          "use_exponential_sigmas",
          "use_karras_sigmas"})
        need(s.at(key) == false, "image_scheduler_contract");
    auto t = read(root + "/transformer/config.json");
    need(
        t.at("num_layers") == 32 && t.at("num_attention_heads") == 32
            && t.at("attention_head_dim") == 128 && t.at("in_channels") == 64
            && t.at("context_in_dim") == 4096
            && t.at("axes_dims_rope") == Json::array({16, 56, 56})
            && t.at("patch_size") == 1 && t.at("mlp_ratio") == 3 && t.at("eps") == 1e-6
            && t.at("causal_condition") == true && t.at("out_channels") == 64,
        "image_transformer_contract"
    );
    return result;
}
}  // namespace kadan::image
