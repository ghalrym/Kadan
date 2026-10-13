#include "kadan/gpt_oss_manifest.hpp"

#include <algorithm>
#include <string>

#include "kadan/weight_inventory.hpp"
#include "metadata_json.hpp"

namespace kadan::gpt_oss {
namespace {
using checkpoint::Dtype;
using checkpoint::json::Value;

void require(bool value, const char* error) {
    if (!value)
        throw std::invalid_argument(error);
}

Config validate_config(const Value& c) {
    const auto& names = c.at("architectures");
    require(
        c.at("model_type").string() == "gpt_oss" && names.kind == Value::Kind::array
            && names.children.size() == 1
            && names.children[0].string() == "GptOssForCausalLM",
        "gpt_oss_architecture"
    );
    const Config config{
        c.at("num_hidden_layers").integer(), c.at("num_local_experts").integer()
    };
    require(
        (config.layers == 24 && config.experts == 32)
            || (config.layers == 36 && config.experts == 128),
        "gpt_oss_dimensions"
    );
    for (const auto& [key, expected] :
         std::initializer_list<std::pair<const char*, std::size_t>>{
             {"hidden_size", Config::hidden},
             {"intermediate_size", Config::intermediate},
             {"vocab_size", Config::vocab},
             {"num_attention_heads", Config::heads},
             {"num_key_value_heads", Config::kv_heads},
             {"head_dim", Config::head_dim},
             {"num_experts_per_tok", Config::experts_per_token},
             {"experts_per_token", Config::experts_per_token},
             {"max_position_embeddings", Config::max_context},
             {"initial_context_length", 4096},
             {"sliding_window", Config::sliding_window},
             {"eos_token_id", 200002},
             {"pad_token_id", 199999}
         })
        require(c.at(key).integer() == expected, "gpt_oss_dimensions");
    require(
        c.at("attention_bias").boolean() && c.at("use_cache").boolean()
            && !c.at("tie_word_embeddings").boolean()
            && c.at("attention_dropout").number() == 0
            && c.at("hidden_act").string() == "silu"
            && c.at("rms_norm_eps").number() == 1e-5
            && c.at("swiglu_limit").number() == 7
            && c.at("rope_theta").number() == 150000,
        "gpt_oss_math"
    );
    const auto& rope = c.at("rope_scaling");
    require(
        !c.find("rope_parameters") && rope.children.size() == 6
            && rope.at("rope_type").string() == "yarn"
            && rope.at("factor").number() == 32 && rope.at("beta_fast").number() == 32
            && rope.at("beta_slow").number() == 1
            && rope.at("original_max_position_embeddings").integer() == 4096
            && !rope.at("truncate").boolean(),
        "gpt_oss_rope"
    );
    const auto& layers = c.at("layer_types");
    require(
        layers.kind == Value::Kind::array && layers.children.size() == config.layers,
        "gpt_oss_layer_types"
    );
    for (std::size_t i = 0; i < config.layers; ++i)
        require(
            layers.children[i].string()
                == (i % 2 ? "full_attention" : "sliding_attention"),
            "gpt_oss_layer_types"
        );
    const auto& quant = c.at("quantization_config");
    require(
        quant.children.size() == 2 && quant.at("quant_method").string() == "mxfp4",
        "gpt_oss_quantization"
    );
    const auto& excluded = quant.at("modules_to_not_convert");
    require(
        excluded.kind == Value::Kind::array && excluded.children.size() == 4,
        "gpt_oss_quantization"
    );
    for (auto name :
         {"model.layers.*.self_attn",
          "model.layers.*.mlp.router",
          "model.embed_tokens",
          "lm_head"})
        require(
            std::any_of(
                excluded.children.begin(),
                excluded.children.end(),
                [&](const auto& item) { return item.string() == name; }
            ),
            "gpt_oss_quantization"
        );
    return config;
}

std::pmr::string expert_prefix(
    std::size_t layer, ExpertProjection projection, std::pmr::memory_resource* memory
) {
    require(
        projection == ExpertProjection::gate_up || projection == ExpertProjection::down,
        "gpt_oss_projection"
    );
    std::pmr::string name("model.layers.", memory);
    name += std::to_string(layer);
    name += projection == ExpertProjection::gate_up ? ".mlp.experts.gate_up_proj"
                                                    : ".mlp.experts.down_proj";
    return name;
}
}  // namespace

Config read_config(const char* root, std::shared_ptr<checkpoint::MemoryBudget> budget) {
    require(bool(budget), "gpt_oss_metadata_budget");
    return validate_config(checkpoint::json::read(root, "config.json", 65536, budget));
}

struct Manifest::Impl {
    struct Entry {
        checkpoint::TensorInfo info;
        std::size_t shard;
        bool used = false;
    };

    std::shared_ptr<checkpoint::MemoryBudget> budget;
    Config config;
    std::pmr::vector<std::unique_ptr<checkpoint::Shard>> shards;
    std::pmr::vector<Entry> entries;
    std::uint64_t bytes = 0;

    Impl(const char* root, std::shared_ptr<checkpoint::MemoryBudget> memory)
        : budget(std::move(memory)), config(read_config(root, budget)),
          shards(budget.get()), entries(budget.get()) {
        const auto index = checkpoint::json::read(
            root, "model.safetensors.index.json", 1024 * 1024, budget
        );
        const auto& map = index.at("weight_map");
        require(
            map.kind == Value::Kind::object
                && map.children.size() == 3 + 19 * config.layers,
            "gpt_oss_index_coverage"
        );
        std::pmr::vector<std::pmr::string> names(budget.get());
        for (const auto& item : map.children) {
            auto name = item.string();
            require(name.ends_with(".safetensors"), "gpt_oss_shard_name");
            if (std::find(names.begin(), names.end(), name) == names.end()) {
                require(names.size() < 64, "gpt_oss_shard_limit");
                names.emplace_back(name);
                shards.push_back(
                    std::make_unique<checkpoint::Shard>(root, name, budget)
                );
            }
            const auto shard = std::size_t(
                std::find(names.begin(), names.end(), name) - names.begin()
            );
            auto info = shards[shard]->tensor(item.name);
            entries.push_back({info, shard});
            bytes = checkpoint::weight_bytes_add(bytes, info.bytes);
        }
        std::size_t count = 0;
        for (const auto& shard : shards)
            count += shard->tensor_count();
        require(count == entries.size(), "gpt_oss_index_coverage");
        require(
            index.at("metadata").at("total_size").integer() == bytes,
            "gpt_oss_index_bytes"
        );
        std::sort(entries.begin(), entries.end(), [](const auto& a, const auto& b) {
            return a.info.name < b.info.name;
        });
        bind();
        for (const auto& shard : shards)
            shard->check_unchanged();
    }

    Entry& entry(std::string_view name) {
        auto it = std::lower_bound(
            entries.begin(), entries.end(), name, [](const auto& e, auto n) {
                return e.info.name < n;
            }
        );
        require(it != entries.end() && it->info.name == name, "gpt_oss_missing_tensor");
        return *it;
    }

    void expect(
        std::string_view name, Dtype dtype, std::initializer_list<std::uint64_t> shape
    ) {
        auto& e = entry(name);
        require(
            !e.used && e.info.dtype == dtype && e.info.rank == shape.size()
                && std::equal(shape.begin(), shape.end(), e.info.shape.begin()),
            "gpt_oss_tensor_layout"
        );
        e.used = true;
    }

    std::pmr::string
    tensor_name(std::string_view prefix, std::string_view suffix) const {
        std::pmr::string result(prefix, budget.get());
        result += suffix;
        return result;
    }

    void bind() {
        constexpr auto attention_width = Config::heads * Config::head_dim;
        constexpr auto kv_width = Config::kv_heads * Config::head_dim;
        expect(
            "model.embed_tokens.weight", Dtype::bf16, {Config::vocab, Config::hidden}
        );
        expect("lm_head.weight", Dtype::bf16, {Config::vocab, Config::hidden});
        expect("model.norm.weight", Dtype::bf16, {Config::hidden});
        for (std::size_t layer = 0; layer < config.layers; ++layer) {
            std::pmr::string p("model.layers.", budget.get());
            p += std::to_string(layer);
            p += ".";
            auto dense = [&](std::string_view suffix,
                             std::initializer_list<std::uint64_t> shape) {
                expect(tensor_name(p, suffix), Dtype::bf16, shape);
            };
            dense("input_layernorm.weight", {Config::hidden});
            dense("post_attention_layernorm.weight", {Config::hidden});
            dense("self_attn.q_proj.weight", {attention_width, Config::hidden});
            dense("self_attn.q_proj.bias", {attention_width});
            for (auto projection : {"k", "v"}) {
                std::pmr::string name("self_attn.", budget.get());
                name += projection;
                dense(tensor_name(name, "_proj.weight"), {kv_width, Config::hidden});
                dense(tensor_name(name, "_proj.bias"), {kv_width});
            }
            dense("self_attn.o_proj.weight", {Config::hidden, attention_width});
            dense("self_attn.o_proj.bias", {Config::hidden});
            dense("self_attn.sinks", {Config::heads});
            dense("mlp.router.weight", {config.experts, Config::hidden});
            dense("mlp.router.bias", {config.experts});
            for (auto projection :
                 {ExpertProjection::gate_up, ExpertProjection::down}) {
                auto name = expert_prefix(layer, projection, budget.get());
                const auto rows = projection == ExpertProjection::gate_up
                                      ? 2 * Config::intermediate
                                      : Config::hidden;
                expect(tensor_name(name, "_bias"), Dtype::bf16, {config.experts, rows});
                expect(
                    tensor_name(name, "_blocks"),
                    Dtype::u8,
                    {config.experts, rows, 90, 16}
                );
                expect(
                    tensor_name(name, "_scales"), Dtype::u8, {config.experts, rows, 90}
                );
                require(
                    entry(tensor_name(name, "_blocks")).shard
                        == entry(tensor_name(name, "_scales")).shard,
                    "gpt_oss_expert_shards"
                );
            }
        }
        for (const auto& e : entries)
            require(e.used, "gpt_oss_unbound_tensor");
    }
};

Manifest::Manifest(const char* root, std::shared_ptr<checkpoint::MemoryBudget> budget)
    : impl_(std::make_unique<Impl>(root, std::move(budget))) {
}

Manifest::~Manifest() = default;

const Config& Manifest::config() const {
    return impl_->config;
}

std::uint64_t Manifest::stored_bytes() const {
    return impl_->bytes;
}

std::size_t Manifest::tensor_count() const {
    return impl_->entries.size();
}

checkpoint::TensorInfo Manifest::tensor(std::string_view name) const {
    return impl_->entry(name).info;
}

void Manifest::read_tensor(
    std::string_view name, std::size_t offset, std::span<std::uint8_t> out
) const {
    const auto& e = impl_->entry(name);
    impl_->shards[e.shard]->read_tensor(name, offset, out);
}

checkpoint::Projection Manifest::load_expert_rows(
    std::size_t layer,
    ExpertProjection projection,
    std::size_t expert,
    std::size_t first,
    std::size_t count,
    std::size_t payload_bytes,
    std::shared_ptr<checkpoint::MemoryBudget> payload_memory,
    const checkpoint::TensorReader& reader
) const {
    require(layer < impl_->config.layers, "gpt_oss_layer");
    require(bool(payload_memory), "gpt_oss_payload_budget");
    const auto prefix = expert_prefix(layer, projection, impl_->budget.get());
    const auto& e = impl_->entry(impl_->tensor_name(prefix, "_blocks"));
    return impl_->shards[e.shard]->load_mxfp4_expert_rows(
        prefix, expert, first, count, payload_bytes, std::move(payload_memory), reader
    );
}

void Manifest::check_unchanged() const {
    for (const auto& shard : impl_->shards)
        shard->check_unchanged();
}
}  // namespace kadan::gpt_oss
