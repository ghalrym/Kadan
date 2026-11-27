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
        pipeline.at("_class_name") == "Flux2KleinPipeline"
            && pipeline.value("is_distilled", false) == true,
        "image_pipeline_contract"
    );
    result.text_files = {
        "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"
    };
    result.denoiser_files = {"diffusion_pytorch_model.safetensors"};

    const auto text = read(root + "/text_encoder/config.json");
    need(
        text.at("model_type") == "qwen3" && text.at("vocab_size") == 151936
            && text.at("hidden_size") == 2560 && text.at("num_hidden_layers") == 36
            && text.at("num_attention_heads") == 32
            && text.at("num_key_value_heads") == 8 && text.at("head_dim") == 128
            && text.at("intermediate_size") == 9728 && text.at("rope_theta") == 1000000
            && text.at("rms_norm_eps") == 1e-6
            && !text.at("attention_bias").get<bool>(),
        "image_text_config_contract"
    );

    const auto v = read(root + "/vae/config.json");
    need(
        v.at("_class_name") == "AutoencoderKLFlux2" && v.at("in_channels") == 3
            && v.at("out_channels") == 3 && v.at("latent_channels") == 32
            && v.at("block_out_channels") == Json::array({128, 256, 512, 512})
            && v.at("layers_per_block") == 2 && v.at("norm_num_groups") == 32
            && v.at("mid_block_add_attention") == true
            && v.at("use_quant_conv") == true && v.at("use_post_quant_conv") == true
            && v.at("patch_size") == Json::array({2, 2})
            && v.at("batch_norm_eps").is_number(),
        "image_vae_config_contract"
    );
    const float eps = v.at("batch_norm_eps");
    need(std::isfinite(eps) && eps > 0.f && eps <= 1e-2f, "image_vae_config_contract");
    result.batch_norm_eps = eps;

    const auto s = read(root + "/scheduler/scheduler_config.json");
    need(
        s.at("_class_name") == "FlowMatchEulerDiscreteScheduler"
            && s.at("time_shift_type") == "exponential"
            && s.at("use_dynamic_shifting") == true && s.at("shift_terminal").is_null()
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

    const auto t = read(root + "/transformer/config.json");
    need(
        t.at("num_layers") == 5 && t.at("num_single_layers") == 20
            && t.at("num_attention_heads") == 24 && t.at("attention_head_dim") == 128
            && t.at("in_channels") == 128 && t.at("joint_attention_dim") == 7680
            && t.at("axes_dims_rope") == Json::array({32, 32, 32, 32})
            && t.at("rope_theta") == 2000 && t.at("guidance_embeds") == false
            && t.at("mlp_ratio") == 3.0 && t.at("eps") == 1e-6,
        "image_transformer_contract"
    );
    return result;
}
}  // namespace kadan::image
