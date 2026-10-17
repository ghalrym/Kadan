#include "kadan/s1_mini.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <utility>

#ifdef KADAN_WHISPER_BLAS
#include <cblas.h>
#endif

#include "../../weights/metadata_json.hpp"
#include "kadan/float_weights.hpp"
#include "kadan/layer_math.hpp"

namespace kadan::s1 {
namespace {
constexpr std::size_t metadata_bytes = 16 * 1024 * 1024, batch_size = 64;

void require(bool value, const char* message) {
    if (!value)
        throw std::invalid_argument(message);
}

void cancelled(const std::atomic_bool& cancel) {
    if (cancel.load(std::memory_order_relaxed))
        throw std::runtime_error("s1_mini_cancelled");
}

void finite(std::span<const float> values) {
    for (float value : values)
        if (!std::isfinite(value))
            throw std::runtime_error("s1_mini_nonfinite");
}

// The reservation outlives its metadata allocator and every parsed header.
struct MetadataAdmission {
    std::shared_ptr<Resources> resources;
    Handle handle;

    explicit MetadataAdmission(std::shared_ptr<Resources> owner)
        : resources(std::move(owner)),
          handle(resources->reserve(
              Workload::formatting, host_footprint(*resources, metadata_bytes)
          )) {
    }

    ~MetadataAdmission() {
        resources->released(handle);
    }
};

struct Layer {
    std::span<float> input_norm, post_norm, q_norm, k_norm;
    std::span<float> query, key, value, attention_out, gate, up, down;
};

// Resident row-major weights avoid rereading the entire checkpoint per token.
// Tile output rows so cancellation remains bounded even for the vocabulary head.
void project(
    std::span<const float> weights,
    std::span<const float> input,
    std::size_t in,
    std::size_t out,
    std::span<float> output,
    const std::atomic_bool& cancel
) {
    const auto rows = input.size() / in;
    require(
        input.size() == rows * in && rows > 0 && rows <= batch_size
            && weights.size() == in * out && output.size() == rows * out,
        "s1_mini_projection_shape"
    );
    for (std::size_t first = 0; first < out; first += 64) {
        cancelled(cancel);
        const auto count = std::min<std::size_t>(64, out - first);
#if defined(KADAN_WHISPER_BLAS)
        cblas_sgemm(
            CblasRowMajor,
            CblasNoTrans,
            CblasTrans,
            int(rows),
            int(count),
            int(in),
            1,
            input.data(),
            int(in),
            weights.data() + first * in,
            int(in),
            0,
            output.data() + first,
            int(out)
        );
#else
        for (std::size_t row = 0; row < rows; ++row)
            for (std::size_t o = first; o < first + count; ++o) {
                double sum = 0;
                for (std::size_t i = 0; i < in; ++i)
                    sum += double(input[row * in + i]) * weights[o * in + i];
                output[row * out + o] = static_cast<float>(sum);
            }
#endif
    }
    cancelled(cancel);
    finite(output);
}

void residual(std::span<float> hidden, std::span<const float> update) {
    for (std::size_t i = 0; i < hidden.size(); ++i)
        hidden[i] += update[i];
    finite(hidden);
}
}  // namespace

Config
read_config(const char* root, const std::shared_ptr<checkpoint::MemoryBudget>& budget) {
    require(bool(budget), "s1_mini_metadata_budget");
    const auto json = checkpoint::json::read(root, "config.json", 65536, budget);
    const auto& architectures = json.at("architectures");
    require(
        architectures.kind == checkpoint::json::Value::Kind::array
            && architectures.children.size() == 1
            && architectures.children.front().string() == "Qwen3ForCausalLM"
            && json.at("model_type").string() == "qwen3"
            && json.at("hidden_act").string() == "silu"
            && json.at("torch_dtype").string() == "bfloat16"
            && !json.at("attention_bias").boolean()
            && !json.at("use_sliding_window").boolean()
            && json.at("tie_word_embeddings").boolean()
            && json.at("rope_scaling").kind == checkpoint::json::Value::Kind::null
            && json.at("sliding_window").kind == checkpoint::json::Value::Kind::null,
        "s1_mini_unsupported_architecture"
    );
    Config config{
        json.at("hidden_size").integer(),
        json.at("intermediate_size").integer(),
        json.at("num_hidden_layers").integer(),
        json.at("num_attention_heads").integer(),
        json.at("num_key_value_heads").integer(),
        json.at("head_dim").integer(),
        json.at("vocab_size").integer(),
        json.at("max_position_embeddings").integer(),
        static_cast<float>(json.at("rms_norm_eps").number()),
        json.at("rope_theta").number()
    };
    require(
        config.hidden == 1024 && config.intermediate == 3072 && config.layers == 28
            && config.heads == 16 && config.kv_heads == 8 && config.head_dim == 128
            && config.vocabulary == 151936 && config.max_positions == 40960
            && config.epsilon == 1e-6f && config.rope_theta == 1000000
            && json.at("eos_token_id").integer() == 151645,
        "s1_mini_unsupported_config"
    );
    return config;
}

struct Model::Impl {
    MetadataAdmission admission;
    std::shared_ptr<checkpoint::MemoryBudget> metadata =
        std::make_shared<checkpoint::MemoryBudget>(metadata_bytes);
    Config c;
    std::array<Layer, 28> layers;
    std::unique_ptr<checkpoint::FloatSlice> weights, history, scratch;
    std::span<float> embedding, final_norm;
    std::size_t capacity = 0, position = 0;

    Impl(
        std::shared_ptr<Resources> resources,
        const std::string& root,
        const std::atomic_bool& cancel
    )
        : admission(std::move(resources)), c(read_config(root.c_str(), metadata)) {
        cancelled(cancel);
        const auto generation = checkpoint::json::read(
            root.c_str(), "generation_config.json", 65536, metadata
        );
        const auto& eos = generation.at("eos_token_id");
        require(
            eos.kind == checkpoint::json::Value::Kind::array && eos.children.size() == 2
                && eos.children[0].integer() == 151645
                && eos.children[1].integer() == 151643
                && !generation.at("do_sample").boolean(),
            "s1_mini_unsupported_generation"
        );
        checkpoint::Shard shard(root.c_str(), "model.safetensors", metadata);
        const auto q_width = c.heads * c.head_dim, kv_width = c.kv_heads * c.head_dim;
        const auto elements = c.vocabulary * c.hidden + c.hidden
                              + c.layers
                                    * (2 * c.hidden + 2 * c.head_dim
                                       + (2 * q_width + 2 * kv_width) * c.hidden
                                       + 3 * c.intermediate * c.hidden);

        auto shape = [&](std::string_view name, std::size_t rows, std::size_t columns) {
            const auto info = shard.tensor(name);
            require(
                info.dtype == checkpoint::Dtype::bf16
                    && info.rank == (columns ? 2U : 1U) && info.shape[0] == rows
                    && (!columns || info.shape[1] == columns),
                "s1_mini_weight_shape"
            );
        };
        // Validate every binding before admitting a multi-gigabyte weight bank.
        auto bindings = [&](auto bind) {
            bind(embedding, "model.embed_tokens.weight", c.vocabulary, c.hidden);
            bind(final_norm, "model.norm.weight", c.hidden, 0);
            for (std::size_t i = 0; i < layers.size(); ++i) {
                auto& layer = layers[i];
                const auto prefix = "model.layers." + std::to_string(i);
                bind(layer.input_norm, prefix + ".input_layernorm.weight", c.hidden, 0);
                bind(
                    layer.post_norm,
                    prefix + ".post_attention_layernorm.weight",
                    c.hidden,
                    0
                );
                bind(layer.q_norm, prefix + ".self_attn.q_norm.weight", c.head_dim, 0);
                bind(layer.k_norm, prefix + ".self_attn.k_norm.weight", c.head_dim, 0);
                bind(
                    layer.query, prefix + ".self_attn.q_proj.weight", q_width, c.hidden
                );
                bind(
                    layer.key, prefix + ".self_attn.k_proj.weight", kv_width, c.hidden
                );
                bind(
                    layer.value, prefix + ".self_attn.v_proj.weight", kv_width, c.hidden
                );
                bind(
                    layer.attention_out,
                    prefix + ".self_attn.o_proj.weight",
                    c.hidden,
                    q_width
                );
                bind(
                    layer.gate,
                    prefix + ".mlp.gate_proj.weight",
                    c.intermediate,
                    c.hidden
                );
                bind(
                    layer.up, prefix + ".mlp.up_proj.weight", c.intermediate, c.hidden
                );
                bind(
                    layer.down,
                    prefix + ".mlp.down_proj.weight",
                    c.hidden,
                    c.intermediate
                );
            }
        };
        bindings([&](auto&, auto name, auto rows, auto columns) {
            shape(name, rows, columns);
        });
        // The upstream checkpoint stores a redundant output head. Tied embedding
        // configuration makes the input embedding the execution source for both.
        shape("lm_head.weight", c.vocabulary, c.hidden);
        require(shard.tensor_count() == 3 + 11 * c.layers, "s1_mini_weight_inventory");
        weights = std::make_unique<checkpoint::FloatSlice>(
            *admission.resources, Workload::formatting, elements
        );
        std::size_t offset = 0;
        bindings([&](auto& destination, auto name, auto rows, auto columns) {
            cancelled(cancel);
            const auto count = rows * (columns ? columns : 1);
            destination = weights->span().subspan(offset, count);
            shard.read_float_tensor(name, 0, destination, cancel);
            offset += count;
        });
        require(offset == elements, "s1_mini_weight_layout");
        shard.check_unchanged();
    }

    void reset() {
        scratch.reset();
        history.reset();
        capacity = position = 0;
    }

    void begin(std::size_t requested) {
        require(requested > 0 && requested <= Model::max_capacity, "s1_mini_capacity");
        reset();
        const auto q_width = c.heads * c.head_dim, kv_width = c.kv_heads * c.head_dim;
        history = std::make_unique<checkpoint::FloatSlice>(
            *admission.resources,
            Workload::formatting,
            2 * c.layers * requested * kv_width
        );
        scratch = std::make_unique<checkpoint::FloatSlice>(
            *admission.resources,
            Workload::formatting,
            batch_size
                    * (3 * c.hidden + 2 * q_width + 2 * kv_width + 2 * c.intermediate)
                + requested + c.vocabulary + c.head_dim / 2
        );
        capacity = requested;
    }

    std::uint32_t forward(
        std::span<const std::uint32_t> tokens,
        const std::atomic_bool& cancel,
        bool predict
    ) {
        const auto rows = tokens.size(), q_width = c.heads * c.head_dim,
                   kv_width = c.kv_heads * c.head_dim;
        require(
            capacity && rows && rows <= batch_size && rows <= capacity - position,
            "s1_mini_context_limit"
        );
        cancelled(cancel);
        std::size_t offset = 0;
        auto region = [&](std::size_t count) {
            auto result = scratch->span().subspan(offset, count);
            offset += count;
            return result;
        };
        auto hidden = region(rows * c.hidden), normalized = region(rows * c.hidden),
             update = region(rows * c.hidden), query = region(rows * q_width),
             attended = region(rows * q_width), key = region(rows * kv_width),
             value = region(rows * kv_width), gate = region(rows * c.intermediate),
             up = region(rows * c.intermediate), probability = region(capacity),
             logits = region(c.vocabulary), frequencies = region(c.head_dim / 2);
        math::inverse_frequencies(c.rope_theta, frequencies);
        for (std::size_t row = 0; row < rows; ++row) {
            require(tokens[row] < c.vocabulary, "s1_mini_token_range");
            std::copy_n(
                embedding.begin() + tokens[row] * c.hidden,
                c.hidden,
                hidden.begin() + row * c.hidden
            );
        }
        for (std::size_t index = 0; index < layers.size(); ++index) {
            cancelled(cancel);
            const auto& layer = layers[index];
            math::rms_norm(
                {rows, c.hidden, c.epsilon, math::NormScale::direct},
                hidden,
                layer.input_norm,
                normalized
            );
            project(layer.query, normalized, c.hidden, q_width, query, cancel);
            project(layer.key, normalized, c.hidden, kv_width, key, cancel);
            project(layer.value, normalized, c.hidden, kv_width, value, cancel);
            auto keys = history->span().subspan(
                2 * index * capacity * kv_width, capacity * kv_width
            );
            auto values = history->span().subspan(
                (2 * index + 1) * capacity * kv_width, capacity * kv_width
            );
            for (std::size_t row = 0; row < rows; ++row) {
                auto q = query.subspan(row * q_width, q_width);
                auto k = key.subspan(row * kv_width, kv_width);
                math::rms_norm(
                    {c.heads, c.head_dim, c.epsilon, math::NormScale::direct},
                    q,
                    layer.q_norm,
                    q
                );
                math::rms_norm(
                    {c.kv_heads, c.head_dim, c.epsilon, math::NormScale::direct},
                    k,
                    layer.k_norm,
                    k
                );
                math::rope(
                    {c.heads, c.head_dim, c.head_dim, capacity},
                    position + row,
                    frequencies,
                    q,
                    q
                );
                math::rope(
                    {c.kv_heads, c.head_dim, c.head_dim, capacity},
                    position + row,
                    frequencies,
                    k,
                    k
                );
                std::copy(
                    k.begin(), k.end(), keys.begin() + (position + row) * kv_width
                );
                std::copy_n(
                    value.begin() + row * kv_width,
                    kv_width,
                    values.begin() + (position + row) * kv_width
                );
            }
            const auto scale = 1 / std::sqrt(double(c.head_dim));
            for (std::size_t row = 0; row < rows; ++row) {
                for (std::size_t head = 0; head < c.heads; ++head) {
                    cancelled(cancel);
                    const auto kv_head = head / (c.heads / c.kv_heads);
                    const auto q =
                        query.subspan(row * q_width + head * c.head_dim, c.head_dim);
                    float maximum = -std::numeric_limits<float>::infinity();
                    for (std::size_t token = 0; token <= position + row; ++token) {
                        double dot = 0;
                        const auto k = keys.subspan(
                            token * kv_width + kv_head * c.head_dim, c.head_dim
                        );
                        for (std::size_t d = 0; d < c.head_dim; ++d)
                            dot += double(q[d]) * k[d];
                        probability[token] = static_cast<float>(dot * scale);
                        maximum = std::max(maximum, probability[token]);
                    }
                    auto scores = probability.first(position + row + 1);
                    finite(scores);
                    double denominator = 0;
                    for (auto& score : scores) {
                        score = std::exp(score - maximum);
                        denominator += score;
                    }
                    for (auto& score : scores)
                        score = static_cast<float>(score / denominator);
                    auto result =
                        attended.subspan(row * q_width + head * c.head_dim, c.head_dim);
                    for (std::size_t d = 0; d < c.head_dim; ++d) {
                        double sum = 0;
                        for (std::size_t token = 0; token < scores.size(); ++token)
                            sum +=
                                double(scores[token])
                                * values[token * kv_width + kv_head * c.head_dim + d];
                        result[d] = static_cast<float>(sum);
                    }
                }
            }
            finite(attended);
            project(layer.attention_out, attended, q_width, c.hidden, update, cancel);
            residual(hidden, update);
            math::rms_norm(
                {rows, c.hidden, c.epsilon, math::NormScale::direct},
                hidden,
                layer.post_norm,
                normalized
            );
            project(layer.gate, normalized, c.hidden, c.intermediate, gate, cancel);
            project(layer.up, normalized, c.hidden, c.intermediate, up, cancel);
            math::gate(math::Gate::silu, up, gate, up);
            project(layer.down, up, c.intermediate, c.hidden, update, cancel);
            residual(hidden, update);
        }
        position += rows;
        if (!predict)
            return 0;
        auto final = normalized.first(c.hidden);
        math::rms_norm(
            {1, c.hidden, c.epsilon, math::NormScale::direct},
            hidden.last(c.hidden),
            final_norm,
            final
        );
        project(embedding, final, c.hidden, c.vocabulary, logits, cancel);
        return static_cast<std::uint32_t>(
            std::max_element(logits.begin(), logits.end()) - logits.begin()
        );
    }
};

Model::Model(std::shared_ptr<Resources> resources) : resources_(std::move(resources)) {
    require(bool(resources_), "s1_mini_resources_required");
}

Model::~Model() = default;

void Model::load(const std::string& root, const std::atomic_bool& cancel) {
    cancelled(cancel);
#ifdef KADAN_WHISPER_BLAS
    require(openblas_get_num_threads() == 1, "s1_mini_blas_thread_limit");
#endif
    close();
    impl_ = std::make_unique<Impl>(resources_, root, cancel);
}

void Model::begin_request(std::size_t capacity) {
    require(bool(impl_), "s1_mini_not_loaded");
    try {
        impl_->begin(capacity);
    } catch (...) {
        impl_->reset();
        throw;
    }
}

std::uint32_t
Model::prefill(std::span<const std::uint32_t> prompt, const std::atomic_bool& cancel) {
    require(
        bool(impl_) && impl_->capacity && impl_->position == 0, "s1_mini_request_state"
    );
    require(
        !prompt.empty() && prompt.size() <= impl_->capacity, "s1_mini_prompt_limit"
    );
    try {
        std::uint32_t result = 0;
        for (std::size_t at = 0; at < prompt.size(); at += batch_size) {
            const auto count = std::min(batch_size, prompt.size() - at);
            result = impl_->forward(
                prompt.subspan(at, count), cancel, at + count == prompt.size()
            );
        }
        return result;
    } catch (...) {
        impl_->reset();
        throw;
    }
}

std::uint32_t Model::step(std::uint32_t token, const std::atomic_bool& cancel) {
    require(bool(impl_) && impl_->position > 0, "s1_mini_request_state");
    try {
        return impl_->forward({&token, 1}, cancel, true);
    } catch (...) {
        impl_->reset();
        throw;
    }
}

void Model::reset_request() {
    if (impl_)
        impl_->reset();
}

void Model::close() {
    impl_.reset();
}

bool Model::is_eos(std::uint32_t token) const {
    return token == 151645 || token == 151643;
}

const Config& Model::config() const {
    require(bool(impl_), "s1_mini_not_loaded");
    return impl_->c;
}
}  // namespace kadan::s1
