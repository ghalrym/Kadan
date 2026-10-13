#include "kadan/image_generation.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <random>

#include <nlohmann/json.hpp>

#include "kadan/bf16_projection.hpp"
#include "kadan/cleanup_error.hpp"
#include "kadan/image_attention.hpp"
#include "kadan/image_denoiser.hpp"
#include "kadan/image_schedule.hpp"
#include "kadan/image_text.hpp"
#include "kadan/image_vae.hpp"
#include "kadan/text_tokenizer.hpp"
#include "kadan/weight_progress.hpp"

namespace kadan::image {
namespace {
void need(bool b, const char* e) {
    if (!b)
        throw std::runtime_error(e);
}

void stop(const std::atomic_bool& c) {
    need(!c.load(), "image_cancelled");
}

Footprint host(Resources& r, Bytes n) {
    auto p = r.snapshot().capacity;
    std::fill(p.begin(), p.end(), 0);
    p[0] = n;
    return p;
}

struct Lease {
    Resources& r;
    Handle h;

    Lease(Resources& a, Bytes n) : r(a), h(r.reserve(Workload::image, host(r, n))) {
    }

    ~Lease() {
        r.released(h);
    }
};

struct Busy {
    bool& b;

    Busy(bool& f) : b(f) {
        need(!b, "busy");
        b = true;
    }

    ~Busy() {
        b = false;
    }
};

#ifdef KADAN_EXPERIMENTAL_IMAGE_PV_TF32
constexpr auto attention_precision = AttentionPrecision::experimental_pv_tf32;
#else
constexpr auto attention_precision = AttentionPrecision::pedantic_fp32;
#endif

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

#ifdef KADAN_WORKER_CUDA
// All operations run on the existing request owner. The session is declared first
// so its streams outlive every launcher, including constructor/error unwinding.
struct ResidentDenoising {
    std::unique_ptr<MediaSession> session;
    std::vector<std::unique_ptr<video::CudaBf16Projection>> owned_launchers;
    std::vector<video::ImageFeedForwardLauncher*> launchers;

    ResidentDenoising(
        const std::shared_ptr<Resources>& resources,
        const std::shared_ptr<DenseCompute>& compute,
        const std::shared_ptr<MediaSessionActivity>& activity,
        const std::vector<int>& devices,
        const std::atomic_bool& cancel
    ) {
        compute->release_scratch();
        std::vector<MediaDeviceLimit> limits;
        for (int device : devices) {
            stop(cancel);
            owned_launchers.push_back(
                std::make_unique<video::CudaBf16Projection>(
                    compute,
                    activity,
                    resources,
                    Workload::image,
                    device,
                    attention_precision
                )
            );
            launchers.push_back(owned_launchers.back().get());
            limits.push_back({device});
        }
        session = std::make_unique<MediaSession>(
            resources,
            activity,
            cuda_media_device_operations(compute, devices),
            Workload::image,
            std::move(limits),
            cancel
        );
    }

    std::vector<video::ImageFeedForwardLauncher*> selected_launchers() const {
        std::vector<video::ImageFeedForwardLauncher*> selected;
        for (const auto& device : session->plan().devices) {
            const auto owner = std::find_if(
                owned_launchers.begin(),
                owned_launchers.end(),
                [&](const auto& launcher) {
                    return launcher->device() == device.device;
                }
            );
            need(owner != owned_launchers.end(), "image_resident_launcher");
            selected.push_back(owner->get());
        }
        return selected;
    }

    void close(bool failed = false) {
        std::exception_ptr first;
        // Close every handle even when one device cannot confirm cleanup.
        for (auto& launcher : owned_launchers) {
            try {
                launcher->close();
            } catch (...) {
                if (!first)
                    first = std::current_exception();
            }
        }
        try {
            if (failed)
                session->abort();
            else
                session->finish();
        } catch (...) {
            if (!first)
                first = std::current_exception();
        }
        if (first)
            std::rethrow_exception(first);
    }
};

const char* fallback_phase(PlanFailure reason) {
    switch (reason) {
    case PlanFailure::capacity:
        return "denoiser_host_capacity";
    case PlanFailure::topology:
        return "denoiser_host_topology";
    case PlanFailure::unsupported_layout:
        return "denoiser_host_unsupported";
    case PlanFailure::invalid_shape:
        return "denoiser_host_invalid_shape";
    }
    throw std::invalid_argument("image_resident_plan_failure");
}
#endif

constexpr const char* prefix =
    "<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n";
}  // namespace

struct Generator::Impl {
    std::shared_ptr<DenseCompute> compute;
    Lease metadata;
    text::Tokenizer tokenizer;
    TextEncoder text;
    Denoiser denoiser;
    VaeDecoder vae;
    std::array<float, 64> mean, stddev;

    Impl(std::shared_ptr<Resources> r, std::shared_ptr<DenseCompute> c)
        : compute(c), metadata(*r, 32 * 1024 * 1024),
          tokenizer(r, kadan::Workload::image), text(r, c), denoiser(r, c), vae(r, c) {
    }
};

Generator::Generator(
    std::shared_ptr<Resources> r,
    std::shared_ptr<DenseCompute> compute,
    std::shared_ptr<MediaSessionActivity> activity,
    std::vector<int> devices
)
    : resources_(std::move(r)), compute_(std::move(compute)),
      media_activity_(std::move(activity)), devices_(std::move(devices)) {
    need(bool(resources_), "image_resources");
}

Generator::~Generator() {
    unload();
}

void Generator::load(
    const std::string& root, const std::atomic_bool& cancel, const Hook& hook
) {
    Busy active(busy_);
    stop(cancel);
    need(!model_, "image_generator_loaded");
    auto m = std::make_unique<Impl>(resources_, compute_);
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
        v.at("decoder_base_dim") == 144 && v.at("z_dim") == 64
            && v.at("dim_mult") == Json::array({1, 2, 4, 8, 8})
            && v.at("num_res_blocks") == 2 && v.at("out_channels") == 4
            && v.at("is_residual") == true && v.at("patch_size").is_null()
            && v.at("temperal_downsample") == Json::array({false, true, true, true}),
        "image_vae_config_contract"
    );
    need(
        v.at("latents_mean").size() == 64 && v.at("latents_std").size() == 64,
        "image_latent_normalization"
    );
    for (std::size_t i = 0; i < 64; ++i) {
        m->mean[i] = v.at("latents_mean")[i];
        m->stddev[i] = v.at("latents_std")[i];
        need(
            std::isfinite(m->mean[i]) && std::isfinite(m->stddev[i])
                && m->stddev[i] > 0,
            "image_latent_normalization"
        );
    }
    auto s = read(root + "/scheduler/scheduler_config.json");
    need(
        s.at("_class_name") == "FlowMatchEulerDiscreteScheduler"
            && s.at("base_image_seq_len") == 256 && s.at("max_image_seq_len") == 8192
            && s.at("base_shift") == .5 && s.at("max_shift") == .9
            && s.at("shift_terminal") == .02 && s.at("time_shift_type") == "exponential"
            && s.at("use_dynamic_shifting") == true
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
    if (compute_) {
        Lease parser(*resources_, 32 * 1024 * 1024);
        auto budget = std::make_shared<checkpoint::MemoryBudget>(32 * 1024 * 1024);
        checkpoint::WeightInventory inventory;
        for (const auto* relative :
             {"text_encoder/model-00001-of-00004.safetensors",
              "text_encoder/model-00002-of-00004.safetensors",
              "text_encoder/model-00003-of-00004.safetensors",
              "text_encoder/model-00004-of-00004.safetensors",
              "transformer/diffusion_pytorch_model-00001-of-00002.safetensors",
              "transformer/diffusion_pytorch_model-00002-of-00002.safetensors",
              "vae/diffusion_pytorch_model.safetensors"}) {
            stop(cancel);
            const std::string path(relative);
            const auto slash = path.find_last_of('/');
            checkpoint::Shard shard(
                (root + "/" + path.substr(0, slash)).c_str(),
                path.substr(slash + 1),
                budget
            );
            inventory.add(shard);
        }
        const auto execution_bytes =
            inventory.floating_execution_bytes(compute_->supports_bf16_weights());
        compute_->prepare_weights(execution_bytes);
        if (hook)
            hook("execution_weight_inventory_bytes", execution_bytes);
        report_weight_inventory(inventory, hook);
        report_weight_placement(*compute_, hook);
    }
    m->tokenizer.load(root + "/processor/tokenizer.json", cancel);
    if (hook)
        hook("tokenizer_loaded", 0);
    const std::array<std::string, 4> text_files{
        "model-00001-of-00004.safetensors",
        "model-00002-of-00004.safetensors",
        "model-00003-of-00004.safetensors",
        "model-00004-of-00004.safetensors"
    };
    auto tr = root + "/text_encoder";
    m->text.load(tr.c_str(), text_files, {}, cancel, [&](std::size_t bytes) {
        if (hook)
            hook("text_checkpoint_read_bytes", bytes);
    });
    if (hook)
        hook("text_loaded", 0);
    const std::array<std::string, 2> denoiser_files{
        "diffusion_pytorch_model-00001-of-00002.safetensors",
        "diffusion_pytorch_model-00002-of-00002.safetensors"
    };
    auto dr = root + "/transformer";
    m->denoiser.load(dr.c_str(), denoiser_files, {}, cancel, hook);
    if (hook)
        hook("denoiser_loaded", 0);
    auto vr = root + "/vae";
    m->vae.load(
        vr.c_str(),
        "diffusion_pytorch_model.safetensors",
        {},
        cancel,
        [&](std::size_t bytes) {
            if (hook)
                hook("vae_checkpoint_read_bytes", bytes);
        }
    );
    if (hook)
        hook("vae_loaded", 0);
    stop(cancel);
    model_ = std::move(m);
}

void Generator::unload() {
    need(!busy_, "busy");
    model_.reset();
}

void Generator::validate(const ImageRequest& request) {
    need(
        !request.prompt.empty() && request.prompt.size() <= 8000 && request.width >= 32
            && request.height >= 32 && request.width <= 3072 && request.height <= 3072
            && request.width * request.height <= 4608 * 1024 && request.width % 32 == 0
            && request.height % 32 == 0 && request.steps >= 2 && request.steps <= 100,
        "image_request_bounds"
    );
}

void Generator::generate(
    const ImageRequest& request,
    std::span<float> output,
    const std::atomic_bool& cancel,
    const Hook& hook,
    [[maybe_unused]] const DenoisingTimings::Sink& timing_sink
) {
    Busy active(busy_);
    stop(cancel);
    need(bool(model_), "image_generator_not_loaded");
    validate(request);
    need(output.size() == request.width * request.height * 4, "image_output_shape");
    auto& m = *model_;
    auto h = request.height / 16, w = request.width / 16, points = h * w;
    Lease admitted(*resources_, 64 * 1024 * 1024);
    std::array<std::uint32_t, 2048> tokens;
    auto raw = std::string(prefix) + "<|im_start|>user\n" + request.prompt
               + "<|im_end|>\n<|im_start|>assistant\n";
    auto count = m.tokenizer.encode(raw, tokens, cancel);
    std::array<std::uint32_t, 64> system;
    auto drop = m.tokenizer.encode(prefix, system, cancel);
    need(drop == 14 && count > drop, "image_prompt_prefix");
    std::vector<float> features(count * 4096);
    try {
        m.text.execute(
            std::span(tokens).first(count), features, cancel, [&](std::size_t i) {
                if (hook)
                    hook("text_layer", i);
            }
        );
    } catch (...) {
        rethrow_after_cleanup(std::current_exception(), [&] {
            m.text.release_device_weights();
        });
    }
    // Conditioning is host-owned now. Retire only text tiles on every image,
    // leaving a warm denoiser bank intact across images and later requests.
    const auto released = m.text.release_device_weights();
    if (hook)
        hook("text_gpu_weights_released_bytes", released);
    if (compute_)
        report_weight_placement(*compute_, hook);
    auto condition = std::span(features).subspan(drop * 4096);
    const auto text_count = count - drop;
    std::vector<float> latent(points * 64), decoded(64 * points);
    std::mt19937_64 rng(request.seed);
    auto uniform = [&]() {
        return (double(rng() >> 11) + .5) / 9007199254740992.;
    };
    for (std::size_t i = 0; i < latent.size(); i += 2) {
        double radius = std::sqrt(-2 * std::log(uniform())),
               angle = 6.2831853071795864769 * uniform();
        latent[i] = float(radius * std::cos(angle));
        if (i + 1 < latent.size())
            latent[i + 1] = float(radius * std::sin(angle));
    }
    auto times = schedule(points, request.steps);
    bool resident_complete = false;
#ifdef KADAN_WORKER_CUDA
    if (compute_ && compute_->supports_bf16_weights() && !devices_.empty()) {
        ResidentDenoising resident(
            resources_, compute_, media_activity_, devices_, cancel
        );
        auto& session = *resident.session;
        std::optional<DenoisingTimings> timing;
        if (timing_sink) {
            timing.emplace(timing_sink);
            timing->reset(
                1,
                points + text_count,
                text_count,
                times.sigma[0],
                times.sigma[1],
                session.transfers(),
                attention_precision
            );
        }
        auto* timings = timing ? &*timing : nullptr;
        try {
            std::optional<MediaPlanError> error;
            {
                DenoisingStageTimer timer(timings, DenoisingStage::preflight);
                error = m.denoiser.preflight(session, text_count, h, w, cancel);
            }
            if (!error) {
                DenoisingStageTimer timer(timings, DenoisingStage::preparation);
                error = m.denoiser.prepare_feed_forward(
                    session,
                    0,
                    points + text_count,
                    HiddenBoundary::pre_qkv,
                    cancel,
                    text_count
                );
            }
            if (error) {
                if (timings)
                    timings->publish(
                        DenoisingOutcome::not_selected, session.transfers()
                    );
                resident.close();
                if (hook)
                    hook(fallback_phase(error->reason), 0);
            } else {
                if (hook)
                    hook("denoiser_resident_selected", 0);
                const auto rows = points + text_count;
                for (std::size_t step = 0; step < request.steps; ++step) {
                    if (step && timings)
                        timings->reset(
                            step + 1,
                            rows,
                            text_count,
                            times.sigma[step],
                            times.sigma[step + 1],
                            session.transfers(),
                            attention_precision
                        );
                    stop(cancel);
                    if (step) {
                        DenoisingStageTimer timer(timings, DenoisingStage::preparation);
                        if (auto rejected = m.denoiser.prepare_feed_forward(
                                session,
                                0,
                                rows,
                                HiddenBoundary::pre_qkv,
                                cancel,
                                text_count
                            ))
                            throw std::runtime_error(
                                "image_resident_admission: " + rejected->detail
                            );
                    }
                    m.denoiser.execute(
                        session,
                        step ? std::span<const float>{}
                             : std::span<const float>(latent),
                        step ? std::span<const float>{}
                             : std::span<const float>(condition),
                        text_count,
                        h,
                        w,
                        times.sigma[step],
                        resident.launchers,
                        cancel,
                        hook,
                        timings
                    );
                    const auto selected = resident.selected_launchers();
                    {
                        DenoisingStageTimer timer(timings, DenoisingStage::scheduler);
                        m.denoiser.execute_scheduler(
                            session,
                            rows,
                            text_count,
                            times.sigma[step],
                            times.sigma[step + 1],
                            selected,
                            cancel
                        );
                    }
                    if (timings)
                        timings->publish(
                            DenoisingOutcome::completed, session.transfers()
                        );
                    if (hook)
                        hook("step_completed", step + 1);
                }
                prepare_decoder_input(
                    session, latent, m.mean, m.stddev, decoded, cancel
                );
                if (hook) {
                    hook(
                        "denoiser_input_upload_bytes", session.transfers().input_upload
                    );
                    hook(
                        "denoiser_output_download_bytes",
                        session.transfers().output_download
                    );
                    hook(
                        "denoiser_peer_activation_bytes",
                        session.transfers().peer_activation
                    );
                }
                resident.close();
                resident_complete = true;
            }
        } catch (...) {
            if (timings)
                timings->publish(
                    cancel.load() ? DenoisingOutcome::cancelled
                                  : DenoisingOutcome::failed,
                    session.transfers()
                );
            rethrow_after_cleanup(std::current_exception(), [&] {
                resident.close(true);
            });
        }
    } else if (hook) {
        hook("denoiser_host_unsupported", 0);
    }
#else
    if (hook)
        hook("denoiser_host_unsupported", 0);
#endif
    if (!resident_complete) {
        need(
            attention_precision == AttentionPrecision::pedantic_fp32,
            "image_pv_tf32_requires_resident_attention"
        );
        std::vector<float> velocity(points * 64);
        for (std::size_t i = 0; i < request.steps; ++i) {
            stop(cancel);
            m.denoiser.execute(
                latent,
                condition,
                text_count,
                h,
                w,
                times.sigma[i],
                velocity,
                cancel,
                [&](const char* phase, std::size_t j) {
                    if (hook)
                        hook(phase, j);
                }
            );
            const float dt = times.sigma[i + 1] - times.sigma[i];
            for (std::size_t j = 0; j < latent.size(); ++j) {
                latent[j] += dt * velocity[j];
                need(std::isfinite(latent[j]), "image_nonfinite_latent");
            }
            if (hook)
                hook("step_completed", i + 1);
        }
        prepare_decoder_input(latent, m.mean, m.stddev, decoded, cancel);
    }
    if (compute_)
        report_weight_placement(*compute_, hook);
    m.vae.decode(decoded, h, w, output, cancel, [&](std::size_t i) {
        if (hook)
            hook("vae_up_block", i);
    });
    stop(cancel);
    if (compute_)
        report_weight_placement(*compute_, hook);
}
}  // namespace kadan::image
