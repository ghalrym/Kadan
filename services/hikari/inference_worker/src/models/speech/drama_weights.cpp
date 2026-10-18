// SPDX-License-Identifier: Apache-2.0
// Drama 3 checkpoint schema and block-FP8 interpretation follow Fish Audio's
// published checkpoint and Apache-2.0 SGLang-Omni fishaudio_drama3/weights.py and
// quantization.py at be34dd84ed9e0971f72fe5e3d9ba8245cdad780a.
// https://github.com/sgl-project/sglang-omni/tree/be34dd84ed9e0971f72fe5e3d9ba8245cdad780a/sglang_omni/models/fishaudio_drama3
// https://github.com/sgl-project/sglang-omni/blob/be34dd84ed9e0971f72fe5e3d9ba8245cdad780a/LICENSE
#include "drama_weights.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace kadan::drama {
void require(bool condition, const char* message) {
    if (!condition)
        throw std::runtime_error(message);
}

void check_cancel(const std::atomic_bool& cancel) {
    if (cancel.load())
        throw std::runtime_error("cancelled");
}

Weights::Weights(
    const std::filesystem::path& root,
    const ModelInfo& info,
    const std::atomic_bool& cancel
)
    : metadata_(std::make_shared<checkpoint::MemoryBudget>(128ULL << 20)), info_(info) {
    for (const auto& entry : std::filesystem::directory_iterator(root)) {
        check_cancel(cancel);
        const auto name = entry.path().filename().string();
        if (name != "model.safetensors"
            && !(name.starts_with("model-") && name.ends_with(".safetensors")))
            continue;
        require(shards_.size() < 16, "drama_shard_count_limit");
        auto shard = std::make_unique<checkpoint::Shard>(root.c_str(), name, metadata_);
        for (std::size_t index = 0; index < shard->tensor_count(); ++index) {
            const auto tensor_info = shard->tensor_at(index);
            require(tensors_.size() < 64000, "drama_tensor_count_limit");
            require(tensor_info.name.size() <= 256, "drama_tensor_name_limit");
            const auto [_, inserted] = tensors_.emplace(
                std::string(tensor_info.name), Tensor{shard.get(), tensor_info}
            );
            require(inserted, "drama_duplicate_tensor");
        }
        shards_.push_back(std::move(shard));
    }
    require(!shards_.empty(), "drama_checkpoint_missing");
}

const Weights::Tensor& Weights::tensor(const std::string& name) const {
    const auto found = tensors_.find(name);
    if (found == tensors_.end())
        throw std::runtime_error("drama_missing_tensor: " + name);
    return found->second;
}

void Weights::shape(
    const std::string& name, std::initializer_list<std::uint64_t> expected, bool fp8
) {
    const auto& t = tensor(name).info;
    require(t.rank == expected.size(), "drama_tensor_rank");
    require(
        std::equal(expected.begin(), expected.end(), t.shape.begin()),
        "drama_tensor_shape"
    );
    if (fp8) {
        require(t.dtype == checkpoint::Dtype::fp8, "drama_expert_fp8_dtype");
        require(
            t.rank == 2 && t.shape[0] % 128 == 0 && t.shape[1] % 128 == 0,
            "drama_fp8_block_shape"
        );
        const auto scale_name = name.substr(0, name.size() - 6) + "weight_scale_inv";
        shape(scale_name, {t.shape[0] / 128, t.shape[1] / 128});
        require(
            tensor(scale_name).info.dtype == checkpoint::Dtype::fp32,
            "drama_fp8_scale_dtype"
        );
    } else {
        require(
            t.dtype == checkpoint::Dtype::bf16 || t.dtype == checkpoint::Dtype::fp32,
            "drama_unquantized_dtype"
        );
    }
    bound_.insert(name);
}

void Weights::validate(const std::atomic_bool& cancel) {
    constexpr std::uint64_t hidden = 2048, width = 4096, kv = 512;
    shape("model.language_model.embed_tokens.weight", {info_.vocabulary, hidden});
    shape("model.language_model.norm.weight", {hidden});
    shape("lm_head.weight", {info_.vocabulary, hidden});
    for (std::size_t i = 0; i < 40; ++i) {
        const auto p = "model.language_model.layers." + std::to_string(i) + ".";
        shape(p + "input_layernorm.weight", {hidden});
        shape(p + "post_attention_layernorm.weight", {hidden});
        if ((i + 1) % 4 == 0) {
            const auto a = p + "self_attn.";
            shape(a + "q_proj.weight", {2 * width, hidden});
            shape(a + "k_proj.weight", {kv, hidden});
            shape(a + "v_proj.weight", {kv, hidden});
            shape(a + "o_proj.weight", {hidden, width});
            shape(a + "q_norm.weight", {256});
            shape(a + "k_norm.weight", {256});
        } else {
            const auto a = p + "linear_attn.";
            shape(a + "in_proj_qkv.weight", {8192, hidden});
            shape(a + "in_proj_z.weight", {4096, hidden});
            shape(a + "in_proj_a.weight", {32, hidden});
            shape(a + "in_proj_b.weight", {32, hidden});
            shape(a + "out_proj.weight", {hidden, 4096});
            shape(a + "conv1d.weight", {8192, 1, 4});
            shape(a + "A_log", {32});
            shape(a + "dt_bias", {32});
            shape(a + "norm.weight", {128});
        }
        shape(p + "mlp.gate.weight", {256, hidden});
        shape(p + "mlp.shared_expert_gate.weight", {1, hidden});
        for (std::size_t expert = 0; expert <= 256; ++expert) {
            check_cancel(cancel);
            const auto e =
                p
                + (expert == 256 ? "mlp.shared_expert."
                                 : "mlp.experts." + std::to_string(expert) + ".");
            const auto quantized = expert < 256 && info_.fp8_experts;
            shape(e + "gate_proj.weight", {512, hidden}, quantized);
            shape(e + "up_proj.weight", {512, hidden}, quantized);
            shape(e + "down_proj.weight", {hidden, 512}, quantized);
        }
    }
    shape("audio_decoder.codebook_embeddings.weight", {40960, hidden});
    for (std::size_t i = 0; i < 13; ++i) {
        const bool shared = i < 4;
        const auto p = std::string(
                           shared ? "audio_decoder_mtp_shared.layers."
                                  : "audio_decoder_mtp.layers."
                       )
                       + std::to_string(shared ? i : i - 4) + ".";
        shape(
            p + (shared ? "semantic_embedding.weight" : "codebook_embedding.weight"),
            {4096, hidden}
        );
        shape(p + "eh_proj.weight", {hidden, 2 * hidden});
        for (const auto* name :
             {"enorm.weight",
              "hnorm.weight",
              "final_layernorm.weight",
              "transformer_layer.attention_norm.weight",
              "transformer_layer.ffn_norm.weight"})
            shape(p + name, {hidden});
        const auto a = p + "transformer_layer.attention.";
        shape(a + "wqkv.weight", {2 * width + 2 * kv, hidden});
        shape(a + "wo.weight", {hidden, width});
        shape(a + "q_norm.weight", {256});
        shape(a + "k_norm.weight", {256});
        shape(p + "transformer_layer.feed_forward.w1.weight", {6144, hidden});
        shape(p + "transformer_layer.feed_forward.w3.weight", {6144, hidden});
        shape(p + "transformer_layer.feed_forward.w2.weight", {hidden, 6144});
        if (!shared)
            shape(
                "audio_decoder_mtp.heads." + std::to_string(i - 4) + ".weight",
                {4096, hidden}
            );
    }
    require(bound_.size() == tensors_.size(), "drama_unrecognized_checkpoint_tensors");
    bound_.clear();
}

Bytes Weights::execution_bytes() const {
    // Conservative decoded inventory for the existing capacity-weighted GPU
    // planner. This is a ceiling, not a demand to make every expert resident.
    Bytes total = 0;
    for (const auto& [name, tensor] : tensors_)
        if (tensor.info.rank == 2 && name.ends_with(".weight"))
            total += tensor.info.shape[0] * tensor.info.shape[1] * sizeof(float);
    return total;
}

void Weights::check_unchanged() const {
    for (const auto& shard : shards_)
        shard->check_unchanged();
}

void Weights::decode(
    const std::string& name,
    std::size_t first,
    std::span<float> out,
    const std::atomic_bool& cancel
) const {
    check_cancel(cancel);
    const auto& t = tensor(name);
    if (t.info.dtype != checkpoint::Dtype::fp8) {
        t.shard->read_float_tensor(name, first, out, cancel);
        return;
    }
    const auto columns = t.info.shape[1];
    const auto& scale = tensor(name.substr(0, name.size() - 6) + "weight_scale_inv");
    // Bound staging independently of tensor size. Read scales once per source
    // callback, then packed tiles; never issue one syscall per quantization block.
    const auto scale_count = (t.info.shape[0] / 128) * (columns / 128);
    require(scale_count <= 1024, "drama_fp8_scale_limit");
    std::array<float, 1024> multipliers{};
    scale.shard->read_float_tensor(
        scale.info.name, 0, std::span(multipliers).first(scale_count), cancel
    );
    for (float multiplier : std::span(multipliers).first(scale_count))
        require(std::isfinite(multiplier) && multiplier > 0, "drama_fp8_scale_value");
    std::array<std::uint8_t, 65536> packed;
    for (std::size_t at = 0; at < out.size();) {
        check_cancel(cancel);
        const auto count = std::min(out.size() - at, packed.size());
        t.shard->read_tensor(name, first + at, std::span(packed).first(count));
        for (std::size_t i = 0; i < count; ++i) {
            const auto element = first + at + i;
            const auto scale_index = ((element / columns) / 128) * (columns / 128)
                                     + (element % columns) / 128;
            require(scale_index < scale_count, "drama_fp8_read_range");
            const auto bits = packed[i];
            const auto exponent = (bits >> 3) & 15, mantissa = bits & 7;
            require(exponent != 15 || mantissa != 7, "drama_fp8_nonfinite");
            const float magnitude =
                exponent ? std::ldexp(1.f + float(mantissa) / 8.f, exponent - 7)
                         : float(mantissa) / 512.f;
            out[at + i] =
                (bits & 128 ? -magnitude : magnitude) * multipliers[scale_index];
            require(std::isfinite(out[at + i]), "drama_fp8_nonfinite");
        }
        at += count;
    }
}

Vector Weights::read(const std::string& name, const std::atomic_bool& cancel) const {
    const auto& t = tensor(name).info;
    std::size_t count = 1;
    for (std::size_t i = 0; i < t.rank; ++i)
        count *= t.shape[i];
    require(count <= 32768, "drama_small_tensor_limit");
    Vector out(count);
    decode(name, 0, out, cancel);
    return out;
}

Vector Weights::row(
    const std::string& name, std::size_t row, const std::atomic_bool& cancel
) const {
    const auto& t = tensor(name).info;
    require(
        t.rank == 2 && row < t.shape[0] && t.shape[1] <= 4096, "drama_embedding_row"
    );
    Vector out(t.shape[1]);
    decode(name, row * t.shape[1], out, cancel);
    return out;
}

Vector Weights::project(
    DenseCompute& compute,
    const std::string& name,
    std::span<const float> input,
    const std::atomic_bool& cancel,
    std::size_t first_row,
    std::size_t rows
) {
    check_cancel(cancel);
    check_unchanged();
    const auto& t = tensor(name);
    require(
        t.info.rank == 2 && input.size() == t.info.shape[1], "drama_projection_input"
    );
    if (!rows)
        rows = t.info.shape[0];
    require(
        first_row <= t.info.shape[0] && rows <= t.info.shape[0] - first_row,
        "drama_projection_rows"
    );
    require(rows <= 16384, "drama_projection_output_limit");
    const auto columns = input.size();
    auto identity = t.shard->tensor_identity(name);
    const auto element_bytes = t.info.bytes / (t.info.shape[0] * t.info.shape[1]);
    identity.payload_begin += first_row * columns * element_bytes;
    identity.payload_end = identity.payload_begin + rows * columns * element_bytes;
    used_.insert(identity);
    Vector output(rows);
    ProjectionResult result;
    if (t.info.dtype == checkpoint::Dtype::bf16 && compute.supports_bf16_weights()) {
        const auto source = [&](std::size_t first,
                                std::span<std::uint16_t> destination) {
            check_cancel(cancel);
            t.shard->read_tensor(
                name,
                (first_row * columns + first) * 2,
                {reinterpret_cast<std::uint8_t*>(destination.data()),
                 destination.size_bytes()}
            );
            check_cancel(cancel);
        };
        result = compute.dense_bf16_source(
            identity, source, {}, input, columns, rows, output, cancel
        );
    } else {
        const auto source = [&](std::size_t first, std::span<float> destination) {
            decode(name, first_row * columns + first, destination, cancel);
        };
        result = compute.dense_source(
            identity, source, {}, input, columns, rows, output, cancel
        );
    }
    require(
        result != ProjectionResult::unsupported, "drama_streamed_projection_unsupported"
    );
    require(projection_output_finite(result, output), "drama_projection_nonfinite");
    check_unchanged();
    return output;
}

void Weights::release(DenseCompute& compute) {
    const std::vector<WeightIdentity> identities(used_.begin(), used_.end());
    compute.release_weights(identities);
    used_.clear();
}
}  // namespace kadan::drama
