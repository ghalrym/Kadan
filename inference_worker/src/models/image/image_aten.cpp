#include "kadan/image_aten.hpp"

#include <ATen/ATen.h>
#include <ATen/Context.h>
#include <ATen/cuda/CUDAContextLight.h>
#include <algorithm>
#include <c10/core/InferenceMode.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAGuard.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <map>

#include "image_aten_modulation.hpp"
#include "kadan/checkpoint.hpp"
#include "kadan/image_aten_plan.hpp"
#include "kadan/image_checkpoint_config.hpp"
#include "kadan/image_edit.hpp"
#include "kadan/image_input.hpp"
#include "kadan/image_schedule.hpp"
#include "kadan/text_tokenizer.hpp"

namespace kadan::image {
namespace {
using Tensor = at::Tensor;
using Clock = std::chrono::steady_clock;
constexpr Bytes mib = 1024 * 1024;
constexpr const char* text_prefix = "model.language_model.";

void need(bool value, const char* error) {
    if (!value)
        throw std::runtime_error(error);
}

void stop(const std::atomic_bool& cancel) {
    need(!cancel.load(), "image_cancelled");
}

struct Weight {
    checkpoint::Shard* shard;
    checkpoint::TensorInfo info;
};

// Shard owns O_NOFOLLOW descriptors and validates exact byte ranges. Preserve the
// existing checkpoint reader; neither Python nor a new deserializer is involved.
struct Inventory {
    std::vector<std::unique_ptr<checkpoint::Shard>> shards;
    std::map<std::string, Weight, std::less<>> weights;
    Bytes bytes = 0, largest = 0;
    AtenImageStage stage;
    at::ScalarType dtype;

    explicit Inventory(
        const std::string& root,
        std::span<const std::string> files,
        AtenImageStage component,
        const std::atomic_bool& cancel
    )
        : stage(component), dtype(
                                component == AtenImageStage::vae
                                        || component == AtenImageStage::vae_encoder
                                    ? at::kFloat
                                    : at::kBFloat16
                            ) {
        const std::string_view prefix =
            component == AtenImageStage::text ? text_prefix : "";
        for (const auto& file : files) {
            stop(cancel);
            auto shard = std::make_unique<checkpoint::Shard>(
                root.c_str(), file, std::make_shared<checkpoint::MemoryBudget>(16 * mib)
            );
            for (std::size_t i = 0; i < shard->tensor_count(); ++i) {
                auto info = shard->tensor_at(i);
                if (!info.name.starts_with(prefix))
                    continue;
                if (stage == AtenImageStage::vae && !info.name.starts_with("decoder.")
                    && !info.name.starts_with("post_quant_conv."))
                    continue;
                if (stage == AtenImageStage::vae_encoder
                    && ((!info.name.starts_with("encoder.")
                         && !info.name.starts_with("quant_conv."))
                        || info.name.find(".time_conv.") != std::string_view::npos))
                    continue;
                need(
                    info.dtype == checkpoint::Dtype::bf16
                        || info.dtype == checkpoint::Dtype::fp32,
                    "image_aten_weight_dtype"
                );
                std::string name(info.name.substr(prefix.size()));
                // Text conditioning stops before the final language-model norm.
                if (stage == AtenImageStage::text && name != "embed_tokens.weight"
                    && !name.starts_with("layers."))
                    continue;
                need(
                    weights.emplace(name, Weight{shard.get(), info}).second,
                    "image_aten_duplicate_tensor"
                );
                Bytes elements = 1;
                for (std::size_t d = 0; d < info.rank; ++d)
                    elements *= info.shape[d];
                bytes += elements * (dtype == at::kFloat ? 4 : 2);
                largest = std::max(largest, info.bytes);
            }
            shards.push_back(std::move(shard));
        }
        need(!weights.empty(), "image_aten_empty_inventory");
    }

    void
    shape(const std::string& name, std::initializer_list<std::uint64_t> shape) const {
        const auto& info = weights.at(name).info;
        need(
            info.rank == shape.size()
                && std::equal(shape.begin(), shape.end(), info.shape.begin()),
            "image_aten_weight_shape"
        );
    }

    void validate_text() const {
        shape("embed_tokens.weight", {151936, 4096});
        for (int i = 0; i < 36; ++i) {
            auto p = "layers." + std::to_string(i) + ".";
            for (auto n : {"input_layernorm.weight", "post_attention_layernorm.weight"})
                shape(p + n, {4096});
            for (auto n : {"q_norm.weight", "k_norm.weight"})
                shape(p + "self_attn." + n, {128});
            for (auto n : {"q_proj.weight", "o_proj.weight"})
                shape(p + "self_attn." + n, {4096, 4096});
            for (auto n : {"k_proj.weight", "v_proj.weight"})
                shape(p + "self_attn." + n, {1024, 4096});
            for (auto n : {"gate_proj.weight", "up_proj.weight"})
                shape(p + "mlp." + n, {12288, 4096});
            shape(p + "mlp.down_proj.weight", {4096, 12288});
        }
    }

    void validate_denoiser() const {
        shape("img_in.weight", {4096, 64});
        shape("txt_in.text_norm.weight", {4096});
        for (auto n :
             {"txt_in.in_layer.weight",
              "txt_in.out_layer.weight",
              "time_text_embed.timestep_embedder.linear_2.weight",
              "norm_out.linear.weight"})
            shape(n, {4096, 4096});
        shape("time_text_embed.timestep_embedder.linear_1.weight", {4096, 256});
        shape("modulation.1.weight", {16384, 4096});
        shape("proj_out.weight", {64, 4096});
        for (int i = 0; i < 32; ++i) {
            auto p = "transformer_blocks." + std::to_string(i) + ".";
            for (auto n :
                 {"to_q.weight", "to_k.weight", "to_v.weight", "to_out.0.weight"})
                shape(p + "attn." + n, {4096, 4096});
            for (auto n : {"norm_q.weight", "norm_k.weight"})
                shape(p + "attn." + n, {128});
            for (auto n : {"proj.weight", "gate_layer.weight"})
                shape(p + "img_mlp." + n, {12288, 4096});
            shape(p + "img_mlp.out.weight", {4096, 12288});
        }
    }

    void validate_vae() const {
        auto conv = [&](const std::string& p,
                        std::uint64_t in,
                        std::uint64_t out,
                        std::uint64_t kernel) {
            shape(p + ".weight", {out, in, kernel, kernel});
            shape(p + ".bias", {out});
        };
        auto residual = [&](const std::string& p, std::uint64_t in, std::uint64_t out) {
            shape(p + ".norm1.gamma", {in, 1, 1, 1});
            shape(p + ".norm2.gamma", {out, 1, 1, 1});
            conv(p + ".conv1", in, out, 3);
            conv(p + ".conv2", out, out, 3);
            if (in != out)
                conv(p + ".conv_shortcut", in, out, 1);
        };
        if (stage == AtenImageStage::vae_encoder) {
            conv("encoder.conv_in", 4, 96, 3);
            const std::array<std::uint64_t, 6> dims{1, 1, 2, 4, 8, 8};
            for (int i = 0; i < 5; ++i) {
                const auto p = "encoder.down_blocks." + std::to_string(i);
                for (int j = 0; j < 2; ++j)
                    residual(
                        p + ".resnets." + std::to_string(j),
                        96 * dims[j ? i + 1 : i],
                        96 * dims[i + 1]
                    );
                if (i < 4)
                    conv(
                        p + ".downsampler.resample.1",
                        96 * dims[i + 1],
                        96 * dims[i + 1],
                        3
                    );
            }
            residual("encoder.mid_block.resnets.0", 768, 768);
            residual("encoder.mid_block.resnets.1", 768, 768);
            shape("encoder.mid_block.attentions.0.norm.gamma", {768, 1, 1});
            conv("encoder.mid_block.attentions.0.to_qkv", 768, 2304, 1);
            conv("encoder.mid_block.attentions.0.proj", 768, 768, 1);
            shape("encoder.norm_out.gamma", {768, 1, 1, 1});
            conv("encoder.conv_out", 768, 128, 3);
            conv("quant_conv", 128, 128, 1);
            return;
        }
        conv("post_quant_conv", 64, 64, 1);
        conv("decoder.conv_in", 64, 1152, 3);
        residual("decoder.mid_block.resnets.0", 1152, 1152);
        residual("decoder.mid_block.resnets.1", 1152, 1152);
        shape("decoder.mid_block.attentions.0.norm.gamma", {1152, 1, 1});
        conv("decoder.mid_block.attentions.0.to_qkv", 1152, 3456, 1);
        conv("decoder.mid_block.attentions.0.proj", 1152, 1152, 1);
        const std::array<std::uint64_t, 6> dims{8, 8, 8, 4, 2, 1};
        for (int i = 0; i < 5; ++i) {
            auto p = "decoder.up_blocks." + std::to_string(i);
            for (int j = 0; j < 3; ++j)
                residual(
                    p + ".resnets." + std::to_string(j),
                    144 * dims[j ? i + 1 : i],
                    144 * dims[i + 1]
                );
            if (i < 4)
                conv(
                    p + ".upsampler.resample.1", 144 * dims[i + 1], 144 * dims[i + 1], 3
                );
        }
        shape("decoder.norm_out.gamma", {144, 1, 1, 1});
        conv("decoder.conv_out", 144, 4, 3);
    }
};

int stage_owner(Resources& resources, const std::vector<int>& devices, Bytes bytes) {
    at::globalContext().lazyInitDevice(c10::DeviceType::CUDA);
    auto snapshot = resources.snapshot();
    for (int device : devices) {
        need(
            device >= 0 && (std::size_t(device) + 1) < snapshot.capacity.size(),
            "image_aten_device"
        );
        c10::cuda::CUDAGuard guard(device);
        std::size_t free = 0, total = 0;
        C10_CUDA_CHECK(cudaMemGetInfo(&free, &total));
        auto ledger_free = snapshot.capacity[device + 1] - snapshot.used[device + 1];
        snapshot.capacity[device + 1] =
            snapshot.used[device + 1] + std::min<Bytes>(free, ledger_free);
    }
    return aten_stage_device(snapshot, devices, bytes);
}

// Admission precedes allocation. ATen's allocator is capped at this ledger
// envelope; clear its cached blocks before acknowledging release to the worker.
class Stage {
    Resources& resources_;
    Handle handle_ = 0;

public:
    int device;
    std::map<std::string, Tensor, std::less<>> bank;

    Stage(
        Resources& r,
        const std::vector<int>& devices,
        Inventory& inventory,
        Bytes workspace,
        const std::atomic_bool& cancel
    )
        : resources_(r), device(stage_owner(r, devices, inventory.bytes + workspace)) {
        auto footprint = r.snapshot().capacity;
        std::fill(footprint.begin(), footprint.end(), 0);
        footprint[0] = 2 * inventory.largest + 32 * mib;
        footprint[device + 1] = inventory.bytes + workspace;
        handle_ = r.reserve(Workload::image, footprint);
        try {
            at::globalContext().lazyInitDevice(c10::DeviceType::CUDA);
            c10::cuda::CUDAGuard guard(device);
            std::size_t free = 0, total = 0;
            C10_CUDA_CHECK(cudaMemGetInfo(&free, &total));
            need(free >= footprint[device + 1], "image_aten_physical_headroom");
            c10::cuda::CUDACachingAllocator::setMemoryFraction(
                double(footprint[device + 1]) / double(total), device
            );
            for (auto& [name, weight] : inventory.weights) {
                stop(cancel);
                std::vector<std::int64_t> shape(
                    weight.info.shape.begin(),
                    weight.info.shape.begin() + weight.info.rank
                );
                auto stored = weight.info.dtype == checkpoint::Dtype::bf16
                                  ? at::kBFloat16
                                  : at::kFloat;
                auto cpu = at::empty(
                    shape, at::TensorOptions().dtype(stored).device(at::kCPU)
                );
                need(cpu.nbytes() == weight.info.bytes, "image_aten_weight_bytes");
                auto* destination = static_cast<std::uint8_t*>(cpu.data_ptr());
                for (std::size_t offset = 0; offset < cpu.nbytes(); offset += 8 * mib) {
                    stop(cancel);
                    auto bytes = std::min<std::size_t>(8 * mib, cpu.nbytes() - offset);
                    weight.shard->read_tensor(
                        weight.info.name, offset, {destination + offset, bytes}
                    );
                }
                bank.emplace(
                    name,
                    cpu.to(at::Device(at::kCUDA, device), inventory.dtype, false, true)
                );
                weight.shard->check_unchanged();
            }
            sync();
        } catch (...) {
            auto error = std::current_exception();
            close();
            std::rethrow_exception(error);
        }
    }

    ~Stage() noexcept {
        try {
            close();
        } catch (...) { /* Keep admission when cleanup is unconfirmed. */
        }
    }

    void sync() const {
        c10::cuda::CUDAGuard guard(device);
        c10::cuda::device_synchronize();
    }

    void close() {
        if (!handle_)
            return;
        c10::cuda::CUDAGuard guard(device);
        sync();
        bank.clear();
        at::cuda::clearCublasWorkspaces();
        c10::cuda::CUDACachingAllocator::emptyCache();
        // Locals in the caller's component scope must be gone before close().
        auto stats = c10::cuda::CUDACachingAllocator::getDeviceStats(device);
        need(
            stats.allocated_bytes[0].current == 0
                && stats.reserved_bytes[0].current == 0,
            "image_aten_cleanup_unconfirmed"
        );
        resources_.released(handle_);
        handle_ = 0;
    }

    Tensor weight(const std::string& name) const {
        return bank.at(name);
    }

    Tensor linear(const std::string& p, const Tensor& x) const {
        // Text keeps BF16 weights resident but uses the reference FP32 forward.
        return at::linear(x, weight(p + ".weight").to(x.scalar_type()));
    }

    Tensor rms(const std::string& p, const Tensor& x, bool centered = false) const {
        auto gamma = weight(p + ".weight").to(at::kFloat);
        if (centered)
            gamma = gamma + 1;
        return at::rms_norm(x.to(at::kFloat), {x.size(-1)}, gamma, 1e-6)
            .to(x.scalar_type());
    }
};

Tensor cpu_float(const Tensor& x) {
    auto cpu = x.to(at::kCPU, at::kFloat).contiguous();
    need(at::isfinite(cpu).all().item<bool>(), "image_aten_nonfinite");
    return cpu;
}

Tensor rms_heads(Stage& stage, const std::string& name, const Tensor& x) {
    return stage.rms(name, x);
}

Tensor heads(const Tensor& x, std::int64_t count) {
    return x.reshape({1, x.size(0), count, 128}).transpose(1, 2);
}

Tensor unheads(const Tensor& x) {
    return x.transpose(1, 2).reshape({x.size(2), -1});
}

Tensor text_rope(const Tensor& x, const Tensor& cosine, const Tensor& sine) {
    auto a = x.slice(-1, 0, 64), b = x.slice(-1, 64, 128);
    return at::cat({a * cosine - b * sine, b * cosine + a * sine}, -1);
}

Tensor encode_text(
    Stage& stage,
    std::span<const std::uint32_t> tokens,
    const std::atomic_bool& cancel,
    const Generator::Hook& hook
) {
    auto options =
        at::TensorOptions().device(at::kCUDA, stage.device).dtype(at::kFloat);
    std::vector<std::int64_t> ids(tokens.begin(), tokens.end());
    auto indices = at::from_blob(ids.data(), {std::int64_t(ids.size())}, at::kLong)
                       .to(options.device());
    auto x = at::embedding(stage.weight("embed_tokens.weight"), indices).to(at::kFloat);
    auto positions = at::arange(x.size(0), options).view({1, 1, -1, 1});
    auto frequencies =
        at::exp(at::arange(64, options) * (-2 * std::log(5000000.) / 128));
    auto angle = positions * frequencies;
    auto cosine = angle.cos(), sine = angle.sin();
    for (int layer = 0; layer < 36; ++layer) {
        stop(cancel);
        auto p = "layers." + std::to_string(layer) + ".";
        auto normalized = stage.rms(p + "input_layernorm", x);
        auto q = rms_heads(
            stage,
            p + "self_attn.q_norm",
            heads(stage.linear(p + "self_attn.q_proj", normalized), 32)
        );
        auto k = rms_heads(
            stage,
            p + "self_attn.k_norm",
            heads(stage.linear(p + "self_attn.k_proj", normalized), 8)
        );
        auto v = heads(stage.linear(p + "self_attn.v_proj", normalized), 8);
        q = text_rope(q, cosine, sine);
        k = text_rope(k, cosine, sine);
        auto mix = at::scaled_dot_product_attention(q, k, v, {}, 0., true, {}, true);
        x = x + stage.linear(p + "self_attn.o_proj", unheads(mix));
        normalized = stage.rms(p + "post_attention_layernorm", x);
        auto gate = at::silu(stage.linear(p + "mlp.gate_proj", normalized));
        auto up = stage.linear(p + "mlp.up_proj", normalized);
        x = x + stage.linear(p + "mlp.down_proj", gate * up);
        if (hook)
            hook("text_layer", layer);
    }
    stop(cancel);
    return cpu_float(x);
}

struct Rotary {
    Tensor cosine, sine;
};

Rotary image_rotary(
    std::int64_t text, std::int64_t h, std::int64_t w, const at::TensorOptions& options
) {
    auto t = at::arange(text, options), y = at::arange(h, options) - (h - h / 2),
         z = at::arange(w, options) - (w - w / 2);
    std::array<Tensor, 3> positions{
        at::cat({t, at::full({h * w}, text, options)}),
        at::cat({t, y.view({h, 1}).expand({h, w}).reshape({-1})}),
        at::cat({t, z.view({1, w}).expand({h, w}).reshape({-1})})
    };
    const std::array<int, 3> widths{16, 56, 56};
    std::vector<Tensor> angles;
    for (int axis = 0; axis < 3; ++axis) {
        auto frequency = at::exp(
            at::arange(0, widths[axis], 2, options) * (-std::log(10000.) / widths[axis])
        );
        angles.push_back(positions[axis].unsqueeze(1) * frequency);
    }
    auto angle = at::cat(angles, 1).unsqueeze(0).unsqueeze(0);
    return {angle.cos(), angle.sin()};
}

Tensor rotate_image(const Tensor& input, const Rotary& rotary) {
    auto pairs = input.to(at::kFloat).reshape({1, 32, input.size(2), 64, 2});
    auto a = pairs.select(-1, 0), b = pairs.select(-1, 1);
    return at::stack(
               {a * rotary.cosine - b * rotary.sine,
                b * rotary.cosine + a * rotary.sine},
               -1
    )
        .reshape_as(input)
        .to(input.scalar_type());
}

Tensor denoise(
    Stage& stage,
    Tensor latent_cpu,
    const Tensor& source_cpu,
    const Tensor& condition_cpu,
    std::int64_t h,
    std::int64_t w,
    const ImageRequest& request,
    const ImageCheckpointConfig& config,
    const std::atomic_bool& cancel,
    const Generator::Hook& hook
) {
    auto options =
        at::TensorOptions().device(at::kCUDA, stage.device).dtype(at::kFloat);
    auto latent = latent_cpu.to(options.device());
    auto times = config.sampling_schedule(h * w, request.steps);
    if (!request.source.empty())
        times = edit_schedule(times, request.strength);
    Tensor source, noise, mask, inverse_mask;
    if (source_cpu.defined()) {
        source = source_cpu.to(options.device());
        noise = latent;
        latent = source * (1.f - times.sigma[0]) + noise * times.sigma[0];
        if (!request.mask.empty()) {
            // Match the native pipeline's area average, including soft masks.
            auto mask_cpu = at::empty({h * w, 1}, at::kFloat);
            auto* values = mask_cpu.data_ptr<float>();
            for (std::int64_t y = 0; y < h; ++y) {
                stop(cancel);
                for (std::int64_t x = 0; x < w; ++x) {
                    float sum = 0;
                    for (int dy = 0; dy < 16; ++dy)
                        for (int dx = 0; dx < 16; ++dx)
                            sum +=
                                request
                                    .mask[(y * 16 + dy) * request.width + x * 16 + dx];
                    values[y * w + x] = sum / 256.f;
                }
            }
            mask = mask_cpu.to(options.device());
            inverse_mask = 1.f - mask;
        }
    }
    auto condition = condition_cpu.to(options.device(), at::kBFloat16);
    auto projected = stage.linear(
        "txt_in.out_layer",
        at::gelu(
            stage.linear(
                "txt_in.in_layer", stage.rms("txt_in.text_norm", condition, true)
            ),
            "tanh"
        )
    );
    const auto text = condition.size(0);
    auto all_rotary = image_rotary(text, h, w, options);
    Rotary image_only{all_rotary.cosine.slice(2, text), all_rotary.sine.slice(2, text)};
    std::array<Tensor, 32> prefix_k, prefix_v;
    auto frequency = at::exp(at::arange(128, options) * (-std::log(10000.) / 128));
    for (std::size_t step = 0; step < times.steps; ++step) {
        stop(cancel);
        // Match the reference BF16 rounding at the model timestep boundary.
        const auto sigma =
            (at::scalar_tensor(times.sigma[step] * 1000, options).to(at::kBFloat16)
             / 1000)
                .to(at::kBFloat16)
                .item<float>();
        auto angle = at::tensor({sigma * 1000, 0.f}, options).unsqueeze(1) * frequency;
        auto embedding = at::cat({angle.cos(), angle.sin()}, 1).to(at::kBFloat16);
        auto time = stage.linear(
            "time_text_embed.timestep_embedder.linear_2",
            at::silu(
                stage.linear("time_text_embed.timestep_embedder.linear_1", embedding)
            )
        );
        auto activated_time = at::silu(time);
        detail::AtenModulation modulation(stage.linear("modulation.1", activated_time));
        auto image = stage.linear("img_in", latent.to(at::kBFloat16));
        Tensor x = step ? image : at::cat({projected, image}, 0);
        const auto prefix = step ? 0 : text;
        const auto& rotary = step ? image_only : all_rotary;
        if (hook)
            hook("conditioning", step);
        for (int layer = 0; layer < 32; ++layer) {
            stop(cancel);
            if (hook)
                hook("block_started", layer);
            auto p = "transformer_blocks." + std::to_string(layer) + ".";
            auto n = detail::aten_modulated_norm(x, modulation.attention_scale, prefix);
            auto q = rotate_image(
                stage.rms(
                    p + "attn.norm_q", heads(stage.linear(p + "attn.to_q", n), 32)
                ),
                rotary
            );
            auto k = rotate_image(
                stage.rms(
                    p + "attn.norm_k", heads(stage.linear(p + "attn.to_k", n), 32)
                ),
                rotary
            );
            auto v = heads(stage.linear(p + "attn.to_v", n), 32);
            Tensor mix;
            if (!step) {
                prefix_k[layer] = k.slice(2, 0, text).contiguous();
                prefix_v[layer] = v.slice(2, 0, text).contiguous();
                auto prefix_attention = at::scaled_dot_product_attention(
                    q.slice(2, 0, text), prefix_k[layer], prefix_v[layer], {}, 0., true
                );
                auto image_attention =
                    at::scaled_dot_product_attention(q.slice(2, text), k, v);
                mix = at::cat({prefix_attention, image_attention}, 2);
            } else {
                mix = at::scaled_dot_product_attention(
                    q,
                    at::cat({prefix_k[layer], k}, 2),
                    at::cat({prefix_v[layer], v}, 2)
                );
            }
            x = detail::aten_gated_residual(
                x,
                stage.linear(p + "attn.to_out.0", unheads(mix)),
                modulation.attention_gate,
                prefix
            );
            n = detail::aten_modulated_norm(x, modulation.mlp_scale, prefix);
            auto gate = at::silu(stage.linear(p + "img_mlp.gate_layer", n));
            auto up = stage.linear(p + "img_mlp.proj", n);
            x = detail::aten_gated_residual(
                x,
                stage.linear(p + "img_mlp.out", gate * up),
                modulation.mlp_gate,
                prefix
            );
            if (hook)
                hook("block", layer);
        }
        auto scale = stage.linear("norm_out.linear", activated_time.slice(0, 0, 1))
                         .to(at::kFloat)
                         .add(1);
        auto normalized = detail::aten_modulated_norm(x.slice(0, prefix), scale, 0);
        auto velocity = stage.linear("proj_out", normalized).to(at::kFloat);
        latent = latent + velocity * (times.sigma[step + 1] - times.sigma[step]);
        if (mask.defined()) {
            const auto next_sigma = times.sigma[step + 1];
            latent =
                mask * latent
                + inverse_mask * (source * (1.f - next_sigma) + noise * next_sigma);
        }
        if (hook)
            hook("step_completed", step + 1);
    }
    stop(cancel);
    return cpu_float(latent);
}

Tensor vae_norm(Stage& stage, const std::string& p, const Tensor& x, bool activate) {
    auto denominator = x.square().sum(1, true).sqrt().clamp_min(1e-12);
    auto gamma = stage.weight(p + ".gamma").reshape({1, x.size(1), 1, 1});
    // Division owns this output; reuse it without mutating the residual input.
    auto y = x / denominator;
    y.mul_(std::sqrt(double(x.size(1)))).mul_(gamma);
    if (activate)
        at::silu_(y);
    return y;
}

Tensor convolution(
    Stage& stage, const std::string& p, const Tensor& x, bool downsample = false
) {
    const auto weight = stage.weight(p + ".weight");
    if (downsample)
        return at::conv2d(
            at::constant_pad_nd(x, {0, 1, 0, 1}, 0),
            weight,
            stage.weight(p + ".bias"),
            {2, 2},
            at::IntArrayRef{0, 0}
        );
    return at::conv2d(
        x,
        weight,
        stage.weight(p + ".bias"),
        {1, 1},
        {weight.size(2) / 2, weight.size(3) / 2}
    );
}

Tensor residual(
    Stage& stage, const std::string& p, const Tensor& x, const std::atomic_bool& cancel
) {
    stop(cancel);
    auto y = convolution(stage, p + ".conv1", vae_norm(stage, p + ".norm1", x, true));
    stop(cancel);
    y = convolution(stage, p + ".conv2", vae_norm(stage, p + ".norm2", y, true));
    return y.add_(
        x.size(1) == y.size(1) ? x : convolution(stage, p + ".conv_shortcut", x)
    );
}

Tensor vae_attention(
    Stage& stage, const std::string& p, const Tensor& x, const std::atomic_bool& cancel
) {
    auto qkv = convolution(stage, p + ".to_qkv", vae_norm(stage, p + ".norm", x, false))
                   .flatten(2);
    auto c = x.size(1), points = qkv.size(2);
    auto q = qkv.slice(1, 0, c).transpose(1, 2).unsqueeze(1);
    auto k = qkv.slice(1, c, 2 * c).transpose(1, 2).unsqueeze(1);
    auto v = qkv.slice(1, 2 * c, 3 * c).transpose(1, 2).unsqueeze(1);
    // The VAE has one wide head. Query tiling bounds the math backend's
    // scratch when this shape cannot use Flash/efficient SDPA.
    auto mix = at::empty_like(q);
    for (std::int64_t row = 0; row < points; row += 256) {
        stop(cancel);
        auto end = std::min(row + 256, points);
        mix.slice(2, row, end)
            .copy_(at::scaled_dot_product_attention(q.slice(2, row, end), k, v));
    }
    return x
           + convolution(
               stage, p + ".proj", mix.squeeze(1).transpose(1, 2).reshape_as(x)
           );
}

Tensor encode_source(
    Stage& stage,
    const ImageRequest& request,
    const ImageCheckpointConfig& config,
    const std::atomic_bool& cancel,
    const Generator::Hook& hook
) {
    const auto device = at::Device(at::kCUDA, stage.device);
    auto x = at::from_blob(
                 const_cast<float*>(request.source.data()),
                 {1, 4, std::int64_t(request.height), std::int64_t(request.width)},
                 at::kFloat
    )
                 .to(device);
    x = convolution(stage, "encoder.conv_in", x);
    for (int i = 0; i < 5; ++i) {
        stop(cancel);
        const auto p = "encoder.down_blocks." + std::to_string(i);
        auto original = x;
        for (int j = 0; j < 2; ++j)
            x = residual(stage, p + ".resnets." + std::to_string(j), x, cancel);
        if (i < 4)
            x = convolution(stage, p + ".downsampler.resample.1", x, true);
        // AvgDown3D orders [channel,time,dy,dx] before averaging channel groups.
        // A single real frame follows temporal zeros; the first chunk never
        // executes the checkpoint's time_conv. This also adds block4's identity.
        const auto temporal = i > 0 && i < 4 ? 2 : 1;
        const auto spatial = i < 4 ? 2 : 1;
        const auto group = original.size(1) * temporal * spatial * spatial / x.size(1);
        auto shortcut = original.unsqueeze(2);
        if (temporal == 2)
            shortcut = at::constant_pad_nd(shortcut, {0, 0, 0, 0, 1, 0}, 0);
        shortcut = shortcut.reshape(
            {1, original.size(1), 1, temporal, x.size(2), spatial, x.size(3), spatial}
        );
        original = Tensor{};
        shortcut = shortcut.permute({0, 1, 3, 5, 7, 2, 4, 6})
                       .reshape({1, x.size(1), group, 1, x.size(2), x.size(3)})
                       .mean(2)
                       .squeeze(2);
        x.add_(shortcut);
        if (hook)
            hook("vae_down_block", i);
    }
    x = residual(stage, "encoder.mid_block.resnets.0", x, cancel);
    x = vae_attention(stage, "encoder.mid_block.attentions.0", x, cancel);
    x = residual(stage, "encoder.mid_block.resnets.1", x, cancel);
    x = convolution(
        stage, "encoder.conv_out", vae_norm(stage, "encoder.norm_out", x, true)
    );
    x = convolution(stage, "quant_conv", x).slice(1, 0, 64);
    auto mean =
        at::from_blob(const_cast<float*>(config.mean.data()), {1, 64, 1, 1}, at::kFloat)
            .to(device);
    auto stddev =
        at::from_blob(
            const_cast<float*>(config.stddev.data()), {1, 64, 1, 1}, at::kFloat
        )
            .to(device);
    stop(cancel);
    return cpu_float(((x - mean) / stddev).flatten(2).squeeze(0).transpose(0, 1));
}

Tensor decode(
    Stage& stage,
    Tensor latent,
    std::int64_t h,
    std::int64_t w,
    const ImageCheckpointConfig& config,
    const std::atomic_bool& cancel,
    const Generator::Hook& hook
) {
    auto options =
        at::TensorOptions().device(at::kCUDA, stage.device).dtype(at::kFloat);
    auto mean =
        at::from_blob(const_cast<float*>(config.mean.data()), {1, 64, 1, 1}, at::kFloat)
            .to(options.device());
    auto stddev =
        at::from_blob(
            const_cast<float*>(config.stddev.data()), {1, 64, 1, 1}, at::kFloat
        )
            .to(options.device());
    auto x = latent.to(options.device()).transpose(0, 1).reshape({1, 64, h, w}) * stddev
             + mean;
    x = convolution(stage, "post_quant_conv", x);
    x = convolution(stage, "decoder.conv_in", x);
    x = residual(stage, "decoder.mid_block.resnets.0", x, cancel);
    x = vae_attention(stage, "decoder.mid_block.attentions.0", x, cancel);
    x = residual(stage, "decoder.mid_block.resnets.1", x, cancel);
    for (int i = 0; i < 5; ++i) {
        stop(cancel);
        const auto p = "decoder.up_blocks." + std::to_string(i);
        auto original = i < 4 ? x : Tensor{};
        for (int j = 0; j < 3; ++j)
            x = residual(stage, p + ".resnets." + std::to_string(j), x, cancel);
        if (i < 4) {
            x = at::upsample_nearest2d(x, {x.size(2) * 2, x.size(3) * 2}, {}, {});
            x = convolution(stage, p + ".upsampler.resample.1", x);
            auto factor = i < 3 ? 2 : 1;
            auto repeats = x.size(1) * factor * 4 / original.size(1);
            need(
                repeats > 0 && x.size(1) * factor * 4 % original.size(1) == 0,
                "image_aten_vae_shortcut"
            );
            const auto original_h = original.size(2), original_w = original.size(3);
            auto shortcut = original.repeat_interleave(repeats, 1);
            // The reshape below copies a permuted view. Retire the source first
            // so the peak is x + repeated shortcut + reshaped shortcut.
            original = Tensor{};
            shortcut =
                shortcut.reshape({1, x.size(1), factor, 2, 2, original_h, original_w})
                    .select(2, factor - 1)
                    .permute({0, 1, 4, 2, 5, 3})
                    .reshape_as(x);
            x.add_(shortcut);
        }
        if (hook)
            hook("vae_up_block", i);
    }
    x = convolution(
        stage, "decoder.conv_out", vae_norm(stage, "decoder.norm_out", x, true)
    );
    stop(cancel);
    return cpu_float(x.clamp(-1, 1));
}

// Backend flags are process-wide, so restore them before another workload runs.
struct ArithmeticScope {
    at::Context& context = at::globalContext();
    bool blas = context.allowTF32CuBLAS(), cudnn = context.allowTF32CuDNN();
    at::CuBLASReductionOption reduction = context.allowBF16ReductionCuBLAS();

    ArithmeticScope() {
        context.setAllowTF32CuBLAS(false);
        context.setAllowTF32CuDNN(false);
        context.setAllowBF16ReductionCuBLAS(false);
    }

    ~ArithmeticScope() {
        context.setAllowTF32CuBLAS(blas);
        context.setAllowTF32CuDNN(cudnn);
        context.setAllowBF16ReductionCuBLAS(
            reduction == at::CuBLASReductionOption::AllowReducedPrecisionWithSplitK,
            reduction
                != at::CuBLASReductionOption::DisallowReducedPrecisionDisallowSplitK
        );
    }
};

void stage_timing(
    const char* phase, const char* stage, int device, Clock::time_point started
) {
    auto seconds = std::chrono::duration<double>(Clock::now() - started).count();
    std::fprintf(
        stderr,
        "KADAN_ATEN_IMAGE_TIMING "
        "{\"phase\":\"%s\",\"stage\":\"%s\",\"device\":%d,\"seconds\":%.9f}\n",
        phase,
        stage,
        device,
        seconds
    );
    std::fflush(stderr);
}
}  // namespace

struct AtenGenerator::Impl {
    // Parser/tokenizer allocations are accounted for even while GPU stages idle.
    Resources& resources;
    Handle metadata;
    text::Tokenizer tokenizer;
    ImageCheckpointConfig config;
    std::string root;
    std::unique_ptr<Inventory> text, denoiser, vae, vae_encoder;

    explicit Impl(const std::shared_ptr<Resources>& r)
        : resources(*r), metadata(0), tokenizer(r, Workload::image) {
        auto bytes = r->snapshot().capacity;
        std::fill(bytes.begin(), bytes.end(), 0);
        bytes[0] = 192 * mib;
        metadata = r->reserve(Workload::image, bytes);
    }

    ~Impl() {
        vae_encoder.reset();
        vae.reset();
        denoiser.reset();
        text.reset();
        resources.released(metadata);
    }

    Inventory& encoder(const std::atomic_bool& cancel) {
        if (!vae_encoder) {
            const std::array<std::string, 1> files{
                "diffusion_pytorch_model.safetensors"
            };
            auto inventory = std::make_unique<Inventory>(
                root + "/vae", files, AtenImageStage::vae_encoder, cancel
            );
            inventory->validate_vae();
            vae_encoder = std::move(inventory);
        }
        return *vae_encoder;
    }
};

AtenGenerator::AtenGenerator(
    std::shared_ptr<Resources> resources, std::vector<int> devices
)
    : resources_(std::move(resources)), devices_(std::move(devices)) {
    need(bool(resources_) && !devices_.empty(), "image_aten_resources");
}

AtenGenerator::~AtenGenerator() = default;

void AtenGenerator::load(
    const std::string& root, const std::atomic_bool& cancel, const Generator::Hook& hook
) {
    need(!model_, "image_generator_loaded");
    auto model = std::make_unique<Impl>(resources_);
    model->root = root;
    model->config = validate_checkpoint_config(root);
    model->tokenizer.load(root + "/processor/tokenizer.json", cancel);
    const std::array<std::string, 2> denoiser_files{
        "diffusion_pytorch_model-00001-of-00002.safetensors",
        "diffusion_pytorch_model-00002-of-00002.safetensors"
    };
    const std::array<std::string, 1> vae_files{"diffusion_pytorch_model.safetensors"};
    model->text = std::make_unique<Inventory>(
        root + "/text_encoder", model->config.text_files, AtenImageStage::text, cancel
    );
    model->denoiser = std::make_unique<Inventory>(
        root + "/transformer", denoiser_files, AtenImageStage::denoiser, cancel
    );
    model->vae = std::make_unique<Inventory>(
        root + "/vae", vae_files, AtenImageStage::vae, cancel
    );
    model->text->validate_text();
    model->denoiser->validate_denoiser();
    model->vae->validate_vae();
    stop(cancel);
    model_ = std::move(model);
    if (hook)
        hook("aten_checkpoint_loaded", 0);
}

void AtenGenerator::unload() {
    model_.reset();
}

void AtenGenerator::prepare(
    const ImageRequest& request, const std::atomic_bool& cancel
) {
    need(bool(model_), "image_generator_not_loaded");
    Generator::validate(request);
    stop(cancel);
    if (!request.source.empty() && request.strength == 0)
        return;
    auto& m = *model_;
    auto prompt = prepare_image_prompt(m.tokenizer, request.prompt, cancel);
    const auto points = (request.height / 16) * (request.width / 16);
    std::vector<const Inventory*> inventories{
        m.text.get(), m.denoiser.get(), m.vae.get()
    };
    if (!request.source.empty() && (request.strength < 1 || !request.mask.empty()))
        inventories.push_back(&m.encoder(cancel));
    // All required component plans must fit before the FIFO owner commits to
    // running. Components execute sequentially, so their GPU budgets do not sum.
    for (const auto* inventory : inventories) {
        stop(cancel);
        const auto tokens = inventory->stage == AtenImageStage::text
                                ? prompt.count
                                : prompt.count - prompt.prefix;
        const auto workspace = aten_stage_workspace(
            inventory->stage, points, tokens, !request.source.empty()
        );
        (void)stage_owner(*resources_, devices_, inventory->bytes + workspace);
        const auto snapshot = resources_->snapshot();
        if (2 * inventory->largest + 32 * mib > snapshot.capacity[0] - snapshot.used[0])
            throw ResourceExhausted("image_aten_host_headroom");
    }
}

void AtenGenerator::generate(
    const ImageRequest& request,
    std::span<float> output,
    const std::atomic_bool& cancel,
    const Generator::Hook& hook
) {
    need(bool(model_), "image_generator_not_loaded");
    Generator::validate(request);
    need(output.size() == request.width * request.height * 4, "image_output_shape");
    stop(cancel);
    if (!request.source.empty() && request.strength == 0) {
        std::copy(request.source.begin(), request.source.end(), output.begin());
        return;
    }
    c10::InferenceMode inference;
    ArithmeticScope arithmetic;
    auto& m = *model_;
    auto prompt = prepare_image_prompt(m.tokenizer, request.prompt, cancel);
    const auto count = prompt.count, drop = prompt.prefix;
    const auto h = request.height / 16, w = request.width / 16, points = h * w;
    Tensor source;
    if (!request.source.empty() && (request.strength < 1 || !request.mask.empty())) {
        auto started = Clock::now();
        Stage stage(
            *resources_,
            devices_,
            m.encoder(cancel),
            aten_stage_workspace(AtenImageStage::vae_encoder, points, 0),
            cancel
        );
        stage_timing("load", "vae_encoder", stage.device, started);
        c10::cuda::CUDAGuard guard(stage.device);
        started = Clock::now();
        source = encode_source(stage, request, m.config, cancel, hook);
        stage.sync();
        stage_timing("generation", "vae_encoder", stage.device, started);
        started = Clock::now();
        stage.close();
        stage_timing("unload", "vae_encoder", stage.device, started);
    }
    Tensor condition;
    {
        auto started = Clock::now();
        Stage stage(
            *resources_,
            devices_,
            *m.text,
            aten_stage_workspace(AtenImageStage::text, points, count),
            cancel
        );
        stage_timing("load", "text", stage.device, started);
        c10::cuda::CUDAGuard guard(stage.device);
        started = Clock::now();
        condition =
            encode_text(stage, std::span(prompt.tokens).first(count), cancel, hook)
                .slice(0, drop)
                .contiguous();
        stage.sync();
        stage_timing("generation", "text", stage.device, started);
        started = Clock::now();
        stage.close();
        stage_timing("unload", "text", stage.device, started);
    }
    // Preserve Kadan's seed/noise contract. A shared replay tensor is needed for
    // elementwise comparison to PyTorch CPU CHW noise in the reference harness.
    auto latent = at::empty({std::int64_t(points), 64}, at::kFloat);
    seed_image_latent(
        {latent.data_ptr<float>(), std::size_t(latent.numel())}, request.seed, cancel
    );
    {
        auto started = Clock::now();
        Stage stage(
            *resources_,
            devices_,
            *m.denoiser,
            aten_stage_workspace(
                AtenImageStage::denoiser, points, count - drop, !request.source.empty()
            ),
            cancel
        );
        stage_timing("load", "denoiser", stage.device, started);
        c10::cuda::CUDAGuard guard(stage.device);
        started = Clock::now();
        latent = denoise(
            stage, latent, source, condition, h, w, request, m.config, cancel, hook
        );
        stage.sync();
        stage_timing("generation", "denoiser", stage.device, started);
        started = Clock::now();
        stage.close();
        stage_timing("unload", "denoiser", stage.device, started);
    }
    {
        auto started = Clock::now();
        Stage stage(
            *resources_,
            devices_,
            *m.vae,
            aten_stage_workspace(AtenImageStage::vae, points, count - drop),
            cancel
        );
        stage_timing("load", "vae", stage.device, started);
        c10::cuda::CUDAGuard guard(stage.device);
        started = Clock::now();
        auto decoded = decode(stage, latent, h, w, m.config, cancel, hook);
        stage.sync();
        stage_timing("generation", "vae", stage.device, started);
        std::copy_n(decoded.data_ptr<float>(), output.size(), output.data());
        started = Clock::now();
        stage.close();
        stage_timing("unload", "vae", stage.device, started);
    }
    if (!request.mask.empty())
        composite_image_edit(output, request.source, request.mask);
    stop(cancel);
}
}  // namespace kadan::image
