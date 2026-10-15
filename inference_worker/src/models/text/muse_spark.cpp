// Text equations: public Transformers PR 49644 and MLX Muse implementation.
// GGUF transformations: llama.cpp 443fc4f6d733d5cd52d151e102390c61f25547eb.
// MLX uses BF16 activation boundaries; GGUF uses FP32 graph arithmetic.
// Projections stream bounded decoded rows through the existing CUDA worker.
#include "kadan/muse_spark.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <numeric>
#include <set>
#include <string>

#include "../../weights/metadata_json.hpp"
#include "kadan/gguf_checkpoint.hpp"
#include "kadan/quantization.hpp"
#include "kadan/weight_inventory.hpp"

namespace kadan::muse_spark {
namespace {
constexpr std::size_t hidden = 8192, layers = 62, heads = 128, kv_heads = 16, dim = 64,
                      kv_width = kv_heads * dim, experts = 256, top_k = 8,
                      expert_width = 4096, vocabulary = 202048, window = 2048,
                      metadata_bytes = 512ULL << 20, scratch_bytes = 128ULL << 20;
constexpr float epsilon = 1e-5f, post_epsilon = 1e-8f, q_scale = 5.4730064863875f,
                output_multiplier = 0.17677669529663687f;
using Vec = std::vector<float>;
using Value = checkpoint::json::Value;

void need(bool condition, const char* error) {
    if (!condition)
        throw std::runtime_error(error);
}

float bf(float value) {
    need(std::isfinite(value), "muse_spark_nonfinite");
    auto bits = std::bit_cast<std::uint32_t>(value);
    bits += 0x7fff + ((bits >> 16) & 1);
    value = std::bit_cast<float>(bits & 0xffff0000U);
    need(std::isfinite(value), "muse_spark_nonfinite");
    return value;
}

void cast(Vec& values) {
    for (auto& value : values)
        value = bf(value);
}

float sigmoid(float value) {
    return value >= 0 ? 1 / (1 + std::exp(-value))
                      : std::exp(value) / (1 + std::exp(value));
}

bool full_attention(std::size_t layer) {
    return (layers - 1 - layer) % 4 == 0;
}

// The two learned normalization variants differ in where BF16 rounding occurs.
void norm(
    Vec& values,
    std::span<const float> weight = {},
    float eps = epsilon,
    bool centered = false,
    bool input_scaled = false,
    bool rounded = true
) {
    need(weight.empty() || weight.size() == values.size(), "muse_spark_norm_shape");
    if (input_scaled) {
        need(!weight.empty(), "muse_spark_norm_weight");
        for (std::size_t i = 0; i < values.size(); ++i) {
            values[i] = values[i] * (centered ? bf(1 + weight[i]) : weight[i]);
            if (rounded)
                values[i] = bf(values[i]);
        }
    }
    float sum = 0;
    for (float value : values)
        sum += value * value;
    need(std::isfinite(sum), "muse_spark_norm_nonfinite");
    const float factor = std::pow(sum / float(values.size()) + eps, -0.5f);
    for (std::size_t i = 0; i < values.size(); ++i) {
        float value = values[i] * factor;
        if (!input_scaled && !weight.empty())
            value *= centered ? bf(1 + weight[i]) : weight[i];
        need(std::isfinite(value), "muse_spark_norm_nonfinite");
        values[i] = rounded ? bf(value) : value;
    }
}

unsigned quantization_bits(const Value& config) {
    const auto* quant = config.find("quantization_config");
    if (!quant || quant->kind == Value::Kind::null)
        return 0;
    const auto bits = quant->at("bits").integer();
    need(
        (bits == 3 || bits == 4) && quant->at("group_size").integer() == 64
            && quant->at("mode").string() == "affine",
        "muse_spark_quantization"
    );
    if (const auto* duplicate = config.find("quantization"))
        need(
            duplicate->at("bits").integer() == bits
                && duplicate->at("group_size").integer() == 64
                && duplicate->at("mode").string() == "affine",
            "muse_spark_quantization"
        );
    return unsigned(bits);
}

struct Tensor {
    checkpoint::Shard* shard;
    checkpoint::TensorInfo info;
    bool used = false;
};

struct Matrix {
    Tensor *weight, *scales = nullptr, *biases = nullptr;
    std::size_t rows, columns, count;
    std::string gguf_name;
};

struct Store {
    std::shared_ptr<checkpoint::MemoryBudget> budget =
        std::make_shared<checkpoint::MemoryBudget>(metadata_bytes);
    std::vector<std::unique_ptr<checkpoint::Shard>> shards;
    std::map<std::string, Tensor, std::less<>> tensors;
    std::map<std::string, Matrix, std::less<>> matrices;
    std::size_t packed = 0, execution = 0;
    std::unique_ptr<checkpoint::GgufCheckpoint> gguf;
    std::map<std::string, std::string, std::less<>> gguf_vectors;
    unsigned bits = 0;
    std::string base;
    std::atomic_bool idle_cancel{false};
    const std::atomic_bool* cancel = &idle_cancel;

    void check_cancel() const {
        need(!cancel->load(), "muse_spark_cancelled");
    }

    Tensor& tensor(const std::string& name) {
        auto it = tensors.find(name);
        need(it != tensors.end(), "muse_spark_missing_tensor");
        return it->second;
    }

    Tensor& bind(const std::string& name, std::initializer_list<std::size_t> dims) {
        auto& value = tensor(name);
        need(value.info.rank == dims.size(), "muse_spark_tensor_shape");
        std::size_t i = 0;
        for (auto expected : dims)
            need(value.info.shape[i++] == expected, "muse_spark_tensor_shape");
        need(!value.used, "muse_spark_duplicate_binding");
        value.used = true;
        return value;
    }

    void vector(const std::string& name, std::size_t count) {
        auto& value = bind(name, {count});
        need(
            value.info.dtype == checkpoint::Dtype::bf16
                || value.info.dtype == checkpoint::Dtype::fp32,
            "muse_spark_vector_dtype"
        );
    }

    void matrix(
        const std::string& prefix,
        std::size_t rows,
        std::size_t columns,
        std::size_t count = 1,
        bool dense = false
    ) {
        const bool quantized = bits && !dense;
        const auto stored_columns = quantized ? columns * bits / 32 : columns;
        Tensor* weight = count == 1
                             ? &bind(prefix + ".weight", {rows, stored_columns})
                             : &bind(prefix + ".weight", {count, rows, stored_columns});
        Matrix descriptor{weight, nullptr, nullptr, rows, columns, count, {}};
        if (quantized) {
            need(
                columns % 64 == 0 && weight->info.dtype == checkpoint::Dtype::u32,
                "muse_spark_affine_weight"
            );
            descriptor.scales =
                count == 1 ? &bind(prefix + ".scales", {rows, columns / 64})
                           : &bind(prefix + ".scales", {count, rows, columns / 64});
            descriptor.biases =
                count == 1 ? &bind(prefix + ".biases", {rows, columns / 64})
                           : &bind(prefix + ".biases", {count, rows, columns / 64});
            for (auto* companion : {descriptor.scales, descriptor.biases})
                need(
                    companion->info.dtype == checkpoint::Dtype::bf16
                        || companion->info.dtype == checkpoint::Dtype::fp16,
                    "muse_spark_affine_scale_dtype"
                );
        } else
            need(
                weight->info.dtype == checkpoint::Dtype::fp32, "muse_spark_router_dtype"
            );
        need(
            matrices.emplace(prefix, descriptor).second, "muse_spark_duplicate_matrix"
        );
        execution = checkpoint::weight_bytes_add(
            execution, count * rows * columns * sizeof(float)
        );
    }

    void load_gguf(const char* root) {
        gguf = std::make_unique<checkpoint::GgufCheckpoint>(root, *cancel);
        need(
            gguf->string("general.architecture") == "muse-spark",
            "muse_spark_gguf_architecture"
        );
        for (const auto& [key, expected] :
             std::initializer_list<std::pair<const char*, std::uint64_t>>{
                 {"block_count", layers},
                 {"context_length", 1048576},
                 {"embedding_length", hidden},
                 {"attention.head_count", heads},
                 {"attention.head_count_kv", kv_heads},
                 {"attention.key_length", dim},
                 {"attention.value_length", dim},
                 {"expert_count", experts},
                 {"expert_used_count", top_k},
                 {"moe_latent_size", expert_width},
                 {"expert_feed_forward_length", expert_width},
                 {"attention.sliding_window", window},
                 {"expert_gating_func", 2}
             })
            need(
                gguf->integer(std::string("muse-spark.") + key) == expected,
                "muse_spark_gguf_dimensions"
            );
        for (const auto& [key, expected] :
             std::initializer_list<std::pair<const char*, double>>{
                 {"rope.freq_base", 500000},
                 {"rope.freq_base_swa", 500000},
                 {"attention.layer_norm_rms_epsilon", double(epsilon)},
                 {"post_norm_epsilon", double(post_epsilon)},
                 {"logit_scale", double(output_multiplier)},
                 {"final_logit_softcapping", 20},
                 {"expert_weights_scale", 1}
             })
            need(
                std::abs(gguf->number(std::string("muse-spark.") + key) - expected)
                    <= std::abs(expected) * 1e-6,
                "muse_spark_gguf_math"
            );
        const auto sliding =
            gguf->booleans("muse-spark.attention.sliding_window_pattern");
        need(sliding.size() == layers, "muse_spark_gguf_schedule");
        for (std::size_t i = 0; i < layers; ++i)
            need(sliding[i] == !full_attention(i), "muse_spark_gguf_schedule");
        need(
            gguf->boolean("muse-spark.expert_weights_norm"),
            "muse_spark_gguf_router_normalization"
        );
        std::set<std::string> used;
        const auto bind_vector = [&](const std::string& role,
                                     const std::string& actual,
                                     std::size_t size) {
            const auto& tensor = gguf->tensor(actual);
            need(
                tensor.rank == 1 && tensor.shape[0] == size && tensor.type == 0,
                "muse_spark_gguf_vector"
            );
            need(
                used.insert(actual).second && gguf_vectors.emplace(role, actual).second,
                "muse_spark_duplicate_binding"
            );
        };
        const auto bind_matrix = [&](const std::string& role,
                                     const std::string& actual,
                                     std::size_t rows,
                                     std::size_t columns,
                                     std::size_t count,
                                     std::uint32_t type) {
            const auto& tensor = gguf->tensor(actual);
            need(
                tensor.rank == (count == 1 ? 2 : 3) && tensor.shape[0] == columns
                    && tensor.shape[1] == rows
                    && (count == 1 || tensor.shape[2] == count) && tensor.type == type,
                "muse_spark_gguf_matrix"
            );
            need(
                used.insert(actual).second
                    && matrices
                           .emplace(
                               role,
                               Matrix{
                                   nullptr,
                                   nullptr,
                                   nullptr,
                                   rows,
                                   columns,
                                   count,
                                   actual
                               }
                           )
                           .second,
                "muse_spark_duplicate_binding"
            );
            execution = checkpoint::weight_bytes_add(
                execution, count * rows * columns * sizeof(float)
            );
        };
        bind_matrix(
            base + "embed_tokens", "token_embd.weight", vocabulary, hidden, 1, 8
        );
        bind_matrix("lm_head", "output.weight", vocabulary, hidden, 1, 8);
        bind_vector(base + "norm.weight", "output_norm.weight", hidden);
        for (std::size_t i = 0; i < layers; ++i) {
            const auto role = base + "layers." + std::to_string(i) + ".";
            const auto actual = "blk." + std::to_string(i) + ".";
            for (const auto& [name, tensor] :
                 std::initializer_list<std::pair<const char*, const char*>>{
                     {"input_layernorm.weight", "attn_norm.weight"},
                     {"pre_feedforward_layernorm.weight", "ffn_norm.weight"},
                     {"post_feedforward_layernorm.weight", "post_ffw_norm.weight"},
                     {"post_attention_residual_gate.gate.alpha",
                      "attn_resid_alpha.weight"},
                     {"post_attention_residual_gate.gate.beta",
                      "attn_resid_beta.weight"},
                     {"post_feedforward_residual_gate.gate.alpha",
                      "ffn_resid_alpha.weight"},
                     {"post_feedforward_residual_gate.gate.beta",
                      "ffn_resid_beta.weight"}
                 })
                bind_vector(role + name, actual + tensor, hidden);
            bind_vector(
                role + "self_attn.q_norm.weight", actual + "attn_q_norm.weight", dim
            );
            bind_vector(
                role + "self_attn.k_norm.weight", actual + "attn_k_norm.weight", dim
            );
            bind_matrix(
                role + "self_attn.q_proj",
                actual + "attn_q.weight",
                hidden,
                hidden,
                1,
                8
            );
            bind_matrix(
                role + "self_attn.k_proj",
                actual + "attn_k.weight",
                kv_width,
                hidden,
                1,
                8
            );
            bind_matrix(
                role + "self_attn.v_proj",
                actual + "attn_v.weight",
                kv_width,
                hidden,
                1,
                8
            );
            bind_matrix(
                role + "self_attn.gate_proj",
                actual + "attn_gate.weight",
                hidden,
                hidden,
                1,
                8
            );
            bind_matrix(
                role + "self_attn.o_proj",
                actual + "attn_output.weight",
                hidden,
                hidden,
                1,
                8
            );
            bind_matrix(
                role + "mlp.gate", actual + "ffn_gate_inp.weight", experts, hidden, 1, 0
            );
            bind_vector(
                role + "mlp.gate.e_score_correction_bias",
                actual + "exp_probs_b.bias",
                experts
            );
            bind_matrix(
                role + "mlp.pre_expert_proj",
                actual + "ffn_latent_down.weight",
                expert_width,
                hidden,
                1,
                8
            );
            bind_matrix(
                role + "mlp.post_expert_proj",
                actual + "ffn_latent_up.weight",
                hidden,
                expert_width,
                1,
                8
            );
            bind_vector(
                role + "mlp.pre_expert_norm.weight",
                actual + "ffn_pre_exp_norm.weight",
                expert_width
            );
            bind_vector(
                role + "mlp.post_expert_norm.weight",
                actual + "ffn_post_exp_norm.weight",
                expert_width
            );
            bind_matrix(
                role + "mlp.switch_mlp.gate_proj",
                actual + "ffn_gate_exps.weight",
                expert_width,
                expert_width,
                experts,
                11
            );
            bind_matrix(
                role + "mlp.switch_mlp.up_proj",
                actual + "ffn_up_exps.weight",
                expert_width,
                expert_width,
                experts,
                11
            );
            bind_matrix(
                role + "mlp.switch_mlp.down_proj",
                actual + "ffn_down_exps.weight",
                expert_width,
                expert_width,
                experts,
                12
            );
        }
        need(used.size() == gguf->tensor_count(), "muse_spark_gguf_unbound_tensor");
        for (std::size_t i = 0; i < gguf->tensor_count(); ++i) {
            const auto& tensor = gguf->tensor_at(i);
            need(used.contains(tensor.name), "muse_spark_gguf_unbound_tensor");
            packed = checkpoint::weight_bytes_add(packed, tensor.bytes);
        }
    }

    Store(const char* root, const std::atomic_bool* requested) {
        if (requested)
            cancel = requested;
        check_cancel();
        validate_config(root, budget);
        auto config = checkpoint::json::read(root, "config.json", 16 << 20, budget);
        base = "model.";
        if (const auto* format = config.find("kadan_checkpoint_format")) {
            need(format->string() == "gguf", "muse_spark_checkpoint_format");
            load_gguf(root);
            return;
        }
        bits = quantization_bits(config);
        const auto generation =
            checkpoint::json::read(root, "generation_config.json", 65536, budget);
        need(
            generation.at("eos_token_id").kind == Value::Kind::array,
            "muse_spark_eos_contract"
        );
        const auto& eos = generation.at("eos_token_id").children;
        need(
            eos.size() == 2 && eos[0].integer() == 200001 && eos[1].integer() == 200008,
            "muse_spark_eos_contract"
        );
        const auto index = checkpoint::json::read(
            root, "model.safetensors.index.json", 32 << 20, budget
        );
        const auto& mapping = index.at("weight_map");
        need(
            mapping.kind == Value::Kind::object && !mapping.children.empty()
                && mapping.children.size() <= 200000,
            "muse_spark_weight_index"
        );
        std::map<std::string, std::string, std::less<>> expected;
        std::set<std::string> files;
        for (const auto& entry : mapping.children) {
            const std::string file(entry.string());
            need(file.ends_with(".safetensors"), "muse_spark_shard_name");
            need(
                expected.emplace(std::string(entry.name), file).second,
                "muse_spark_duplicate_index_tensor"
            );
            files.insert(file);
        }
        need(files.size() <= 512, "muse_spark_shard_limit");
        std::size_t found = 0;
        for (const auto& file : files) {
            check_cancel();
            auto shard = std::make_unique<checkpoint::Shard>(root, file, budget);
            for (std::size_t i = 0; i < shard->tensor_count(); ++i) {
                const auto info = shard->tensor_at(i);
                auto it = expected.find(info.name);
                need(
                    it != expected.end() && it->second == file,
                    "muse_spark_index_coverage"
                );
                ++found;
                const bool text = info.name.starts_with(base + "layers.")
                                  || info.name == base + "embed_tokens.weight"
                                  || info.name == base + "embed_tokens.scales"
                                  || info.name == base + "embed_tokens.biases"
                                  || info.name == base + "norm.weight"
                                  || info.name.starts_with("lm_head.");
                need(text, "muse_spark_unrecognized_tensor");
                need(
                    tensors.emplace(std::string(info.name), Tensor{shard.get(), info})
                        .second,
                    "muse_spark_duplicate_tensor"
                );
                packed = checkpoint::weight_bytes_add(packed, info.bytes);
            }
            shards.push_back(std::move(shard));
        }
        need(found == expected.size(), "muse_spark_index_coverage");
        matrix(base + "embed_tokens", vocabulary, hidden);
        matrix("lm_head", vocabulary, hidden);
        vector(base + "norm.weight", hidden);
        for (std::size_t layer = 0; layer < layers; ++layer) {
            const auto prefix = base + "layers." + std::to_string(layer) + ".";
            for (const auto name :
                 {"input_layernorm.weight",
                  "pre_feedforward_layernorm.weight",
                  "post_feedforward_layernorm.weight",
                  "post_attention_residual_gate.gate",
                  "post_feedforward_residual_gate.gate"})
                vector(prefix + name, hidden);
            matrix(prefix + "self_attn.q_proj", hidden, hidden);
            matrix(prefix + "self_attn.k_proj", kv_width, hidden);
            matrix(prefix + "self_attn.v_proj", kv_width, hidden);
            matrix(prefix + "self_attn.gate_proj", hidden, hidden);
            matrix(prefix + "self_attn.o_proj", hidden, hidden);
            matrix(prefix + "mlp.gate", experts, hidden, 1, true);
            vector(prefix + "mlp.gate.e_score_correction_bias", experts);
            matrix(prefix + "mlp.pre_expert_proj", expert_width, hidden);
            matrix(prefix + "mlp.post_expert_proj", hidden, expert_width);
            vector(prefix + "mlp.pre_expert_norm.weight", expert_width);
            vector(prefix + "mlp.post_expert_norm.weight", expert_width);
            for (const auto name : {"gate_proj", "up_proj", "down_proj"})
                matrix(
                    prefix + "mlp.switch_mlp." + name,
                    expert_width,
                    expert_width,
                    experts
                );
        }
        for (const auto& [name, value] : tensors) {
            (void)name;
            need(value.used, "muse_spark_unbound_tensor");
        }
    }

    Vec read(const std::string& name) {
        if (gguf) {
            const auto& actual = gguf_vectors.at(name);
            const auto& descriptor = gguf->tensor(actual);
            Vec output(descriptor.shape[0]);
            gguf->read_vector(actual, output, *cancel);
            return output;
        }
        auto& value = tensor(name);
        need(
            value.info.rank == 1 && value.info.shape[0] <= hidden,
            "muse_spark_small_tensor"
        );
        Vec output(value.info.shape[0]);
        value.shard->read_float_tensor(name, 0, output, *cancel);
        for (auto v : output)
            need(std::isfinite(v), "muse_spark_nonfinite_weight");
        return output;
    }

    void source(
        const Matrix& matrix,
        std::size_t expert,
        std::size_t first,
        std::span<float> destination
    ) {
        check_cancel();
        need(
            expert < matrix.count && first % matrix.columns == 0
                && destination.size() % matrix.columns == 0
                && first <= matrix.rows * matrix.columns
                && destination.size() <= matrix.rows * matrix.columns - first,
            "muse_spark_source_bounds"
        );
        if (gguf) {
            gguf->read_rows(
                matrix.gguf_name, expert, first / matrix.columns, destination, *cancel
            );
            return;
        }
        const auto name = std::string(matrix.weight->info.name);
        if (!matrix.scales) {
            matrix.weight->shard->read_float_tensor(
                name,
                expert * matrix.rows * matrix.columns + first,
                destination,
                *cancel
            );
        } else {
            const auto rows = destination.size() / matrix.columns;
            const auto row = expert * matrix.rows + first / matrix.columns;
            const auto row_bytes = matrix.columns * bits / 8;
            std::vector<std::uint8_t> packed_rows(rows * row_bytes);
            matrix.weight->shard->read_tensor(name, row * row_bytes, packed_rows);
            Vec scales(rows * matrix.columns / 64), biases(scales.size());
            matrix.scales->shard->read_float_tensor(
                matrix.scales->info.name, row * matrix.columns / 64, scales, *cancel
            );
            matrix.biases->shard->read_float_tensor(
                matrix.biases->info.name, row * matrix.columns / 64, biases, *cancel
            );
            quantization::decode_mlx_affine_rows(
                {rows, matrix.columns, bits, 64, packed_rows, scales, biases},
                0,
                destination
            );
        }
        for (float value : destination)
            need(std::isfinite(value), "muse_spark_nonfinite_weight");
        check_cancel();
    }

    Vec embedding(unsigned token) {
        auto& descriptor = matrices.at(base + "embed_tokens");
        Vec result(hidden);
        source(descriptor, 0, std::size_t(token) * hidden, result);
        if (!gguf)
            cast(result);
        norm(result, {}, epsilon, false, false, !gguf);
        return result;
    }

    Vec project(
        DenseCompute& compute,
        const std::string& name,
        std::span<const float> input,
        std::size_t expert = 0,
        bool rounded = true
    ) {
        const auto& descriptor = matrices.at(name);
        need(
            input.size() == descriptor.columns && expert < descriptor.count,
            "muse_spark_projection_shape"
        );
        const auto unchanged = [&] {
            if (gguf) {
                gguf->check_unchanged();
                return;
            }
            descriptor.weight->shard->check_unchanged();
            if (descriptor.scales) {
                descriptor.scales->shard->check_unchanged();
                descriptor.biases->shard->check_unchanged();
            }
        };
        unchanged();
        Vec result(descriptor.rows);
        auto identity = gguf ? gguf->identity(descriptor.gguf_name, expert)
                             : descriptor.weight->shard->tensor_identity(
                                   descriptor.weight->info.name
                               );
        if (!gguf) {
            const auto expert_bytes = descriptor.weight->info.bytes / descriptor.count;
            identity.payload_begin += expert * expert_bytes;
            identity.payload_end = identity.payload_begin + expert_bytes;
        }
        ProjectionResult projected;
        if (gguf && gguf->tensor(descriptor.gguf_name).type != 0) {
            const auto type = gguf->tensor(descriptor.gguf_name).type;
            need(type == 8 || type == 11 || type == 12, "muse_spark_gguf_type");
            const auto encoding = type == 8    ? gguf::Encoding::q8_0
                                  : type == 11 ? gguf::Encoding::q3_k
                                               : gguf::Encoding::q4_k;
            const auto packed_reader = [&](std::size_t first,
                                           std::span<std::byte> destination) {
                gguf->read_packed_rows(
                    descriptor.gguf_name, expert, first, destination, *cancel
                );
            };
            projected = compute.dense_gguf_source(
                identity,
                encoding,
                packed_reader,
                {},
                input,
                descriptor.columns,
                descriptor.rows,
                result,
                *cancel,
                !rounded
            );
        } else {
            const auto reader = [&](std::size_t first, std::span<float> destination) {
                source(descriptor, expert, first, destination);
            };
            projected = compute.dense_source(
                identity,
                reader,
                {},
                input,
                descriptor.columns,
                descriptor.rows,
                result,
                *cancel,
                !rounded
            );
        }
        need(
            projected != ProjectionResult::unsupported,
            "muse_spark_cuda_source_required"
        );
        // Resident decoded rows still depend on both immutable scale tensors.
        unchanged();
        if (rounded && !gguf)
            cast(result);
        else
            for (float value : result)
                need(std::isfinite(value), "muse_spark_nonfinite");
        return result;
    }
};
}  // namespace

void validate_config(
    const char* root, std::shared_ptr<checkpoint::MemoryBudget> budget
) {
    const auto config = checkpoint::json::read(root, "config.json", 16 << 20, budget);
    need(config.at("model_type").string() == "muse_spark", "muse_spark_model_type");
    const auto& architectures = config.at("architectures");
    const auto& names = architectures.children;
    need(
        architectures.kind == Value::Kind::array && names.size() == 1
            && names[0].string() == "MuseSparkForConditionalGeneration",
        "muse_spark_architecture"
    );
    // A native-generated bridge identifies GGUF graph arithmetic explicitly.
    // MLX affine tensors retain their checkpoint's BF16 activation boundaries.
    if (const auto* format = config.find("kadan_checkpoint_format")) {
        need(format->string() == "gguf", "muse_spark_checkpoint_format");
        need(config.at("dtype").string() == "float32", "muse_spark_activation_dtype");
        need(
            !config.find("quantization") && !config.find("quantization_config"),
            "muse_spark_gguf_quantization_bridge"
        );
    } else {
        need(config.at("dtype").string() == "bfloat16", "muse_spark_activation_dtype");
        need(quantization_bits(config) != 0, "muse_spark_quantization_required");
    }
    const auto& text = config.at("text_config");
    need(text.at("model_type").string() == "muse_spark_text", "muse_spark_text_type");
    for (const auto& [key, value] :
         std::initializer_list<std::pair<const char*, std::size_t>>{
             {"hidden_size", hidden},
             {"num_hidden_layers", layers},
             {"num_attention_heads", heads},
             {"num_key_value_heads", kv_heads},
             {"head_dim", dim},
             {"vocab_size", vocabulary},
             {"intermediate_size", 16384},
             {"moe_hidden_size", expert_width},
             {"moe_intermediate_size", expert_width},
             {"num_local_experts", experts},
             {"num_experts_per_tok", top_k},
             {"num_shared_experts", 0},
             {"sliding_window", window},
             {"max_position_embeddings", 1048576},
             {"bos_token_id", 200000},
             {"eos_token_id", 200001},
             {"pad_token_id", 200018}
         })
        need(text.at(key).integer() == value, "muse_spark_config_shape");
    need(
        !text.at("attention_bias").boolean()
            && text.at("attention_dropout").number() == 0
            && !text.at("tie_word_embeddings").boolean()
            && text.at("use_cache").boolean()
            && text.at("hidden_act").string() == "silu",
        "muse_spark_config_flags"
    );
    need(
        text.at("rms_norm_eps").number() == 1e-5
            && text.at("post_norm_eps").number() == 1e-8
            && text.at("residual_gate_temperature").number() == 0.3
            && text.at("routed_scaling_factor").number() == 1
            && text.at("final_logit_softcapping").number() == 20
            && std::abs(text.at("output_multiplier").number() - 0.17677669529663687)
                   < 1e-15
            && std::abs(text.at("qk_scale_factor").number() - 5.4730064863875) < 1e-12,
        "muse_spark_config_math"
    );
    const auto& rope = text.at("rope_parameters");
    need(
        rope.at("rope_type").string() == "default"
            && rope.at("rope_theta").number() == 500000,
        "muse_spark_rope"
    );
    const auto& types = text.at("layer_types").children;
    const auto& theta = text.at("layer_rope_theta").children;
    const auto& mlp = text.at("mlp_layer_types").children;
    need(
        text.at("layer_types").kind == Value::Kind::array
            && text.at("layer_rope_theta").kind == Value::Kind::array
            && text.at("mlp_layer_types").kind == Value::Kind::array
            && types.size() == layers && theta.size() == layers && mlp.size() == layers,
        "muse_spark_layer_count"
    );
    for (std::size_t i = 0; i < layers; ++i)
        need(
            types[i].string()
                    == (full_attention(i) ? "full_attention" : "sliding_attention")
                && theta[i].number() == (full_attention(i) ? 0 : 500000)
                && mlp[i].string() == "sparse",
            "muse_spark_layer_types"
        );
}

std::size_t state_bytes(std::size_t context) {
    need(
        context > 0 && context <= serving::max_resident_capacity,
        "muse_spark_context_bound"
    );
    // Sixteen global/NoPE layers and 46 sliding-window layers. K/V entries use
    // F32 host storage for either checkpoint's activation arithmetic.
    return (16 * context + 46 * std::min(context, window)) * 2 * kv_width
           * sizeof(float);
}

Plan plan(const char* root, std::size_t context, const std::atomic_bool* cancel) {
    const auto bytes = state_bytes(context);
    Store store(root, cancel);
    return {
        context,
        metadata_bytes + scratch_bytes + bytes,
        store.packed,
        store.gguf ? store.gguf->tensor_count() : store.tensors.size(),
        store.execution
    };
}

struct Engine::Impl {
    struct State {
        Vec key, value;
        std::size_t capacity;
    };

    Plan p;
    WorkerCompute execution;
    Handle memory = 0;
    std::unique_ptr<Store> weights;
    std::array<State, layers> states;
    std::size_t tokens = 0;
    bool active = false, poisoned = false;

    Impl(
        const char* root,
        Plan selected,
        const ComputePlan& compute_plan,
        std::shared_ptr<Resources> owner,
        const std::atomic_bool* cancel
    )
        : p(selected), execution(compute_plan, Workload::llm, std::move(owner)) {
        need(bool(execution.compute), "muse_spark_cuda_required");
        need(
            p.host == metadata_bytes + scratch_bytes + state_bytes(p.context),
            "muse_spark_plan_state"
        );
        memory = execution.resources->reserve(
            Workload::llm, host_footprint(*execution.resources, p.host)
        );
        try {
            weights = std::make_unique<Store>(root, cancel);
            need(
                weights->packed == p.packed
                    && (weights->gguf ? weights->gguf->tensor_count()
                                      : weights->tensors.size())
                           == p.tensors
                    && weights->execution == p.execution_weights,
                "muse_spark_preflight_changed"
            );
            for (std::size_t i = 0; i < layers; ++i) {
                weights->check_cancel();
                auto& state = states[i];
                state.capacity =
                    full_attention(i) ? p.context : std::min(p.context, window);
                state.key.resize(state.capacity * kv_width);
                state.value.resize(state.key.size());
            }
            // False selects the existing bounded disk-source projection path.
            execution.compute->prepare_weights(p.execution_weights);
            execution.resources->loaded(memory);
        } catch (...) {
            weights.reset();
            for (auto& state : states)
                state = {};
            execution.resources->released(memory);
            memory = 0;
            throw;
        }
    }

    ~Impl() {
        weights.reset();
        for (auto& state : states)
            state = {};
        if (memory) {
            execution.resources->begin_eviction(memory);
            execution.resources->released(memory);
        }
    }

    Vec project(
        const std::string& name,
        std::span<const float> input,
        std::size_t expert = 0,
        bool rounded = true
    ) {
        return weights->project(*execution.compute, name, input, expert, rounded);
    }

    Vec learned_norm(const std::string& name, Vec input, bool input_scaled = false) {
        const auto weight = weights->read(name);
        norm(
            input,
            weight,
            input_scaled ? post_epsilon : epsilon,
            !weights->gguf,
            input_scaled,
            !weights->gguf
        );
        return input;
    }

    Vec residual(const std::string& name, const Vec& input, const Vec& branch) {
        if (weights->gguf) {
            const auto alpha = weights->read(name + ".alpha");
            const auto beta = weights->read(name + ".beta");
            Vec result(hidden);
            for (std::size_t i = 0; i < hidden; ++i) {
                result[i] = alpha[i] * input[i] + beta[i] * branch[i];
                need(std::isfinite(result[i]), "muse_spark_residual_nonfinite");
            }
            return result;
        }
        const auto gate = weights->read(name);
        Vec result(hidden);
        for (std::size_t i = 0; i < hidden; ++i) {
            // The MLX reference computes gate coefficients and the residual
            // blend in FP32, then rounds only their sum to the activation dtype.
            const auto w = gate[i] / 0.3f;
            const auto beta = sigmoid(w);
            const auto alpha = std::sqrt(std::max(1e-3f, sigmoid(-w) * (1 + beta)));
            result[i] = bf(alpha * input[i] + beta * branch[i]);
        }
        return result;
    }

    float round(float value) const {
        need(std::isfinite(value), "muse_spark_nonfinite");
        return weights->gguf ? value : bf(value);
    }

    void round(Vec& values) const {
        for (auto& value : values)
            value = round(value);
    }

    void
    normalize_heads(Vec& values, float scale = 1, std::span<const float> weight = {}) {
        for (std::size_t i = 0; i < values.size(); i += dim) {
            Vec head(values.begin() + i, values.begin() + i + dim);
            norm(head, weight, epsilon, false, false, !weights->gguf);
            for (std::size_t j = 0; j < dim; ++j)
                values[i + j] = round(head[j] * scale);
        }
    }

    void rotate(Vec& values) {
        for (std::size_t h = 0; h < values.size() / dim; ++h)
            for (std::size_t i = 0; i < dim / 2; ++i) {
                const auto angle =
                    float(tokens) / std::pow(500000.f, 2 * float(i) / float(dim));
                const auto cosine = std::cos(angle), sine = std::sin(angle);
                const auto a = values[h * dim + i], b = values[h * dim + dim / 2 + i];
                values[h * dim + i] = round(a * cosine - b * sine);
                values[h * dim + dim / 2 + i] = round(b * cosine + a * sine);
            }
    }

    Vec attention(const std::string& prefix, std::size_t layer, const Vec& input) {
        auto query = project(prefix + "q_proj", input);
        auto key = project(prefix + "k_proj", input);
        auto value = project(prefix + "v_proj", input);
        if (weights->gguf) {
            // GGUF conversion interleaves Q/K projection rows on every layer,
            // including NoPE layers. Restore the shared split-half convention.
            for (auto* values : {&query, &key}) {
                const auto original = *values;
                for (std::size_t h = 0; h < values->size() / dim; ++h)
                    for (std::size_t i = 0; i < dim / 2; ++i) {
                        (*values)[h * dim + i] = original[h * dim + 2 * i];
                        (*values)[h * dim + dim / 2 + i] =
                            original[h * dim + 2 * i + 1];
                    }
            }
            normalize_heads(query, 1, weights->read(prefix + "q_norm.weight"));
            normalize_heads(key, 1, weights->read(prefix + "k_norm.weight"));
        } else {
            normalize_heads(query, q_scale);
            normalize_heads(key);
        }
        if (!full_attention(layer)) {
            rotate(query);
            rotate(key);
        }
        auto& state = states[layer];
        const auto slot = tokens % state.capacity;
        std::copy(key.begin(), key.end(), state.key.begin() + slot * kv_width);
        std::copy(value.begin(), value.end(), state.value.begin() + slot * kv_width);
        const auto count = std::min(tokens + 1, state.capacity);
        const auto first = tokens + 1 - count;
        Vec output(hidden), scores(count);
        for (std::size_t head = 0; head < heads; ++head) {
            weights->check_cancel();
            const auto kv = head / (heads / kv_heads);
            float maximum = -INFINITY;
            for (std::size_t t = 0; t < count; ++t) {
                if (t % 256 == 0)
                    weights->check_cancel();
                const auto at = ((first + t) % state.capacity) * kv_width + kv * dim;
                float score = 0;
                for (std::size_t j = 0; j < dim; ++j)
                    score += query[head * dim + j] * state.key[at + j];
                scores[t] = score / 8;
                need(std::isfinite(scores[t]), "muse_spark_attention_nonfinite");
                maximum = std::max(maximum, scores[t]);
            }
            float sum = 0;
            for (auto& score : scores) {
                score = std::exp(score - maximum);
                sum += score;
            }
            need(std::isfinite(sum) && sum > 0, "muse_spark_attention_softmax");
            for (std::size_t t = 0; t < count; ++t) {
                if (t % 256 == 0)
                    weights->check_cancel();
                const auto probability = scores[t] / sum;
                const auto at = ((first + t) % state.capacity) * kv_width + kv * dim;
                for (std::size_t j = 0; j < dim; ++j)
                    output[head * dim + j] += probability * state.value[at + j];
            }
        }
        round(output);
        normalize_heads(output);
        auto gate = project(prefix + "gate_proj", input);
        for (std::size_t j = 0; j < hidden; ++j)
            output[j] = round(output[j] * round(sigmoid(gate[j])));
        return project(prefix + "o_proj", output);
    }

    Vec moe(const std::string& prefix, const Vec& input) {
        auto scores = project(prefix + "gate", input, 0, false);
        const auto bias = weights->read(prefix + "gate.e_score_correction_bias");
        for (auto& score : scores)
            score = sigmoid(score * (weights->gguf ? 1 : output_multiplier));
        std::array<std::size_t, experts> order;
        std::iota(order.begin(), order.end(), 0);
        std::partial_sort(
            order.begin(), order.begin() + top_k, order.end(), [&](auto a, auto b) {
                const auto sa = scores[a] + bias[a], sb = scores[b] + bias[b];
                return sa == sb ? a < b : sa > sb;
            }
        );
        float sum = 1e-15f;
        for (std::size_t i = 0; i < top_k; ++i)
            sum += scores[order[i]];
        need(std::isfinite(sum) && sum > 0, "muse_spark_router_nonfinite");
        auto pooled = learned_norm(
            prefix + "pre_expert_norm.weight",
            project(prefix + "pre_expert_proj", input)
        );
        Vec output(expert_width);
        // The reference accumulates selected experts in ascending expert order.
        std::sort(order.begin(), order.begin() + top_k);
        const auto norm_weight = weights->read(prefix + "post_expert_norm.weight");
        for (std::size_t i = 0; i < top_k; ++i) {
            weights->check_cancel();
            const auto expert = order[i];
            auto gate = project(prefix + "switch_mlp.gate_proj", pooled, expert);
            auto up = project(prefix + "switch_mlp.up_proj", pooled, expert);
            for (std::size_t j = 0; j < expert_width; ++j)
                gate[j] = round(round(gate[j] * sigmoid(gate[j])) * up[j]);
            auto down = project(prefix + "switch_mlp.down_proj", gate, expert);
            norm(down, norm_weight, post_epsilon, !weights->gguf, true, !weights->gguf);
            const auto weight = scores[expert] / sum;
            for (std::size_t j = 0; j < expert_width; ++j)
                output[j] += down[j] * weight;
        }
        round(output);
        return project(prefix + "post_expert_proj", output);
    }

    serving::Token step(unsigned token, bool logits) {
        need(
            active && !poisoned && token < vocabulary && tokens < p.context,
            "muse_spark_step_state"
        );
        weights->check_cancel();
        auto stream = weights->embedding(token);
        for (std::size_t i = 0; i < layers; ++i) {
            weights->check_cancel();
            const auto prefix = weights->base + "layers." + std::to_string(i) + ".";
            auto input = learned_norm(prefix + "input_layernorm.weight", stream);
            auto output = attention(prefix + "self_attn.", i, input);
            norm(output, {}, post_epsilon, false, false, !weights->gguf);
            stream =
                residual(prefix + "post_attention_residual_gate.gate", stream, output);
            input = learned_norm(prefix + "pre_feedforward_layernorm.weight", stream);
            output = learned_norm(
                prefix + "post_feedforward_layernorm.weight",
                moe(prefix + "mlp.", input),
                true
            );
            stream = residual(
                prefix + "post_feedforward_residual_gate.gate", stream, output
            );
        }
        ++tokens;
        if (!logits)
            return {0, false, tokens};
        norm(
            stream,
            weights->read(weights->base + "norm.weight"),
            epsilon,
            false,
            false,
            !weights->gguf
        );
        auto values = project("lm_head", stream, 0, false);
        // FP32 logits retain ranking through the reference's monotonic softcap.
        for (auto& value : values)
            value = 20 * std::tanh(value * output_multiplier / 20);
        const auto chosen =
            unsigned(std::max_element(values.begin(), values.end()) - values.begin());
        return {chosen, chosen == 200001 || chosen == 200008, tokens};
    }
};

Engine::Engine(
    const char* root,
    Plan p,
    const ComputePlan& cp,
    std::shared_ptr<Resources> owner,
    const std::atomic_bool* cancel
)
    : impl_(std::make_unique<Impl>(root, p, cp, std::move(owner), cancel)) {
}

Engine::~Engine() = default;

void Engine::set_cancel(const std::atomic_bool& cancel) {
    impl_->weights->cancel = &cancel;
}

std::shared_ptr<Resources> Engine::resources() const {
    return impl_->execution.resources;
}

serving::Info Engine::info() const {
    return {vocabulary, impl_->p.context, impl_->p.execution_weights, impl_->p.host};
}

void Engine::reset() {
    need(!impl_->poisoned, "muse_spark_reset_required");
    impl_->tokens = 0;
    // Each position is overwritten before attention can read it.
}

void Engine::begin_request() {
    need(!impl_->active && !impl_->poisoned, "muse_spark_request_state");
    if (impl_->weights->gguf)
        impl_->weights->gguf->check_unchanged();
    for (const auto& shard : impl_->weights->shards)
        shard->check_unchanged();
    impl_->execution.resume();
    reset();
    impl_->active = true;
}

void Engine::end_request() {
    impl_->execution.idle();
    impl_->active = false;
}

void Engine::park() {
    need(!impl_->active, "muse_spark_active_park");
    impl_->execution.park();
}

serving::Token Engine::step(unsigned token, bool logits) {
    try {
        return impl_->step(token, logits);
    } catch (...) {
        impl_->poisoned = true;
        throw;
    }
}

void Engine::close() {
    auto owner = resources();
    impl_->active = false;
    impl_->execution.park();
    impl_.reset();
    for (auto bytes : owner->snapshot().used)
        need(bytes == 0, "muse_spark_cleanup_unconfirmed");
}
}  // namespace kadan::muse_spark
