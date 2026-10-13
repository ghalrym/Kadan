#include "kadan/gpt_oss_manifest.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <source_location>
#include <unistd.h>

#include <nlohmann/json.hpp>

using namespace kadan::gpt_oss;
using Json = nlohmann::json;
namespace fs = std::filesystem;

namespace {
void check(bool ok, std::source_location at = std::source_location::current()) {
    if (!ok)
        throw std::runtime_error("check_line_" + std::to_string(at.line()));
}

template <class F> void fails(F fn, std::string_view expected) {
    try {
        fn();
    } catch (const std::exception& e) {
        check(e.what() == expected);
        return;
    }
    throw std::runtime_error("expected_failure");
}

auto memory() {
    return std::make_shared<kadan::checkpoint::MemoryBudget>(4 << 20);
}

Json config(std::size_t layers = 24, std::size_t experts = 32) {
    Json types = Json::array();
    for (std::size_t i = 0; i < layers; ++i)
        types.push_back(i % 2 ? "full_attention" : "sliding_attention");
    return {
        {"model_type", "gpt_oss"},
        {"architectures", {"GptOssForCausalLM"}},
        {"num_hidden_layers", layers},
        {"num_local_experts", experts},
        {"hidden_size", 2880},
        {"intermediate_size", 2880},
        {"vocab_size", 201088},
        {"num_attention_heads", 64},
        {"num_key_value_heads", 8},
        {"head_dim", 64},
        {"num_experts_per_tok", 4},
        {"experts_per_token", 4},
        {"max_position_embeddings", 131072},
        {"initial_context_length", 4096},
        {"sliding_window", 128},
        {"eos_token_id", 200002},
        {"pad_token_id", 199999},
        {"attention_bias", true},
        {"use_cache", true},
        {"tie_word_embeddings", false},
        {"attention_dropout", 0},
        {"hidden_act", "silu"},
        {"rms_norm_eps", 1e-5},
        {"swiglu_limit", 7},
        {"rope_theta", 150000},
        {"layer_types", types},
        {"rope_scaling",
         {{"rope_type", "yarn"},
          {"factor", 32},
          {"beta_fast", 32},
          {"beta_slow", 1},
          {"original_max_position_embeddings", 4096},
          {"truncate", false}}},
        {"quantization_config",
         {{"quant_method", "mxfp4"},
          {"modules_to_not_convert",
           {"model.layers.*.self_attn",
            "model.layers.*.mlp.router",
            "model.embed_tokens",
            "lm_head"}}}}
    };
}

struct Fixture {
    fs::path root;
    Json tensors = Json::object(), index = {{"weight_map", Json::object()}};
    std::uint64_t bytes = 0, payload_start = 0;

    Fixture() {
        std::string pattern = "/tmp/kadan-gpt-oss-metadata-XXXXXX";
        auto* path = mkdtemp(pattern.data());
        check(path);
        root = path;
    }

    ~Fixture() {
        fs::remove_all(root);
    }

    void document(const char* name, const Json& value) {
        std::ofstream(root / name) << value.dump();
    }

    void
    add(const std::string& name,
        std::initializer_list<std::uint64_t> shape,
        const char* dtype = "BF16") {
        std::uint64_t size = std::string_view(dtype) == "U8" ? 1 : 2;
        for (auto d : shape)
            size *= d;
        tensors[name] = {
            {"dtype", dtype}, {"shape", shape}, {"data_offsets", {bytes, bytes + size}}
        };
        index["weight_map"][name] = "model.safetensors";
        bytes += size;
    }

    void build(std::size_t layers, std::size_t experts) {
        document("config.json", config(layers, experts));
        add("model.embed_tokens.weight", {201088, 2880});
        add("lm_head.weight", {201088, 2880});
        add("model.norm.weight", {2880});
        for (std::size_t i = 0; i < layers; ++i) {
            const auto p = "model.layers." + std::to_string(i) + ".";
            add(p + "input_layernorm.weight", {2880});
            add(p + "post_attention_layernorm.weight", {2880});
            add(p + "self_attn.q_proj.weight", {4096, 2880});
            add(p + "self_attn.q_proj.bias", {4096});
            for (auto kind : {"k", "v"}) {
                add(p + "self_attn." + kind + "_proj.weight", {512, 2880});
                add(p + "self_attn." + kind + "_proj.bias", {512});
            }
            add(p + "self_attn.o_proj.weight", {2880, 4096});
            add(p + "self_attn.o_proj.bias", {2880});
            add(p + "self_attn.sinks", {64});
            add(p + "mlp.router.weight", {experts, 2880});
            add(p + "mlp.router.bias", {experts});
            add(p + "mlp.experts.gate_up_proj_bias", {experts, 5760});
            add(p + "mlp.experts.gate_up_proj_blocks", {experts, 5760, 90, 16}, "U8");
            add(p + "mlp.experts.gate_up_proj_scales", {experts, 5760, 90}, "U8");
            add(p + "mlp.experts.down_proj_bias", {experts, 2880});
            add(p + "mlp.experts.down_proj_blocks", {experts, 2880, 90, 16}, "U8");
            add(p + "mlp.experts.down_proj_scales", {experts, 2880, 90}, "U8");
        }
        index["metadata"]["total_size"] = bytes;
        save();
    }

    void save() {
        document("model.safetensors.index.json", index);
        payload_start = save_shard("model.safetensors", tensors);
    }

    std::uint64_t save_shard(const char* name, Json& contents) {
        std::uint64_t size = 0;
        for (auto& tensor : contents) {
            const auto length = tensor["data_offsets"][1].get<std::uint64_t>()
                                - tensor["data_offsets"][0].get<std::uint64_t>();
            tensor["data_offsets"] = {size, size + length};
            size += length;
        }
        const auto header = contents.dump();
        {
            std::ofstream file(root / name, std::ios::binary | std::ios::trunc);
            for (unsigned i = 0; i < 8; ++i)
                file.put(static_cast<char>(std::uint64_t(header.size()) >> (8 * i)));
            file << header;
            check(bool(file));
        }
        // Sparse byte-format fixture: headers have official dimensions, but no
        // model weights are allocated or executed. Holes exercise metadata only.
        fs::resize_file(root / name, header.size() + 8 + size);
        return header.size() + 8;
    }

    void separate(std::initializer_list<std::string> names) {
        Json other = Json::object();
        for (const auto& name : names) {
            other[name] = tensors[name];
            tensors.erase(name);
            index["weight_map"][name] = "other.safetensors";
        }
        save_shard("other.safetensors", other);
        save();
    }

    void byte(const std::string& name, std::uint64_t offset, unsigned char value) {
        std::fstream file(
            root / "model.safetensors", std::ios::binary | std::ios::in | std::ios::out
        );
        file.seekp(
            payload_start + tensors[name]["data_offsets"][0].get<std::uint64_t>()
            + offset
        );
        file.put(static_cast<char>(value));
        check(bool(file));
    }
};

void config_only() {
    Fixture f;
    auto budget = memory();
    for (auto [layers, experts] : {std::pair{24, 32}, std::pair{36, 128}}) {
        f.document("config.json", config(layers, experts));
        const auto c = read_config(f.root.c_str(), budget);
        check(c.layers == std::size_t(layers) && c.experts == std::size_t(experts));
        check(budget->used() == 0 && !fs::exists(f.root / "model.safetensors"));
    }
    auto reject = [&](Json c, const char* error) {
        f.document("config.json", c);
        fails([&] { read_config(f.root.c_str(), budget); }, error);
        // Constructor shares exactly the same validator, before looking for
        // index/shards.
        fails([&] { Manifest m(f.root.c_str(), budget); }, error);
        check(budget->used() == 0);
    };
    auto c = config();
    c["model_type"] = "qwen3_5_moe";
    reject(c, "gpt_oss_architecture");
    for (auto key :
         {"hidden_size",
          "num_hidden_layers",
          "num_local_experts",
          "experts_per_token",
          "head_dim"}) {
        c = config();
        c[key] = 1;
        reject(c, "gpt_oss_dimensions");
    }
    c = config();
    c["tie_word_embeddings"] = true;
    reject(c, "gpt_oss_math");
    c = config();
    c["swiglu_limit"] = 10;
    reject(c, "gpt_oss_math");
    c = config();
    c["rope_scaling"]["truncate"] = true;
    reject(c, "gpt_oss_rope");
    c = config();
    c["rope_scaling"]["factor"] = 16;
    reject(c, "gpt_oss_rope");
    c = config();
    c["rope_parameters"] = c["rope_scaling"];
    reject(c, "gpt_oss_rope");
    c = config();
    c["layer_types"][0] = "full_attention";
    reject(c, "gpt_oss_layer_types");
    c = config();
    c["quantization_config"]["quant_method"] = "nvfp4";
    reject(c, "gpt_oss_quantization");
    c = config();
    c["quantization_config"]["modules_to_not_convert"][0] = "lm_head";
    reject(c, "gpt_oss_quantization");
}

void complete_manifest(std::size_t layers, std::size_t experts, std::uint64_t bytes) {
    Fixture f;
    f.build(layers, experts);
    // A payload NaN scale must not be read by the metadata-only constructor.
    const std::string prefix = "model.layers.0.mlp.experts.gate_up_proj";
    f.byte(prefix + "_scales", 0, 255);
    auto budget = memory(), payload = memory();
    {
        Manifest m(f.root.c_str(), budget);
        check(m.stored_bytes() == bytes && m.tensor_count() == 3 + 19 * layers);
        check(m.config().experts == experts);
        check(m.tensor("model.layers.0.self_attn.sinks").shape[0] == 64);
        fails(
            [&] {
                m.load_expert_rows(
                    0, ExpertProjection::gate_up, 0, 0, 1, 1530, payload
                );
            },
            "invalid_block_scale"
        );
        check(payload->used() == 0);
        unsigned reads = 0;
        auto row = m.load_expert_rows(
            0,
            ExpertProjection::gate_up,
            experts - 1,
            5759,
            1,
            1530,
            payload,
            [&](std::string_view name, std::size_t at, auto out) {
                const auto index = (experts * 5760 - 1) * 90;
                check(at == index * (reads == 0 ? 16 : 1));
                check(out.size() == (reads == 0 ? 1440 : 90));
                check(name == prefix + (reads == 0 ? "_blocks" : "_scales"));
                std::fill(out.begin(), out.end(), reads++ == 0 ? 0x22 : 127);
            }
        );
        check(reads == 2 && row.view().columns == 2880 && row.view().rows == 1);
        check(kadan::quantization::decode_rows(row.view(), 0, 1, 11520)[0] == 1);
        std::array<std::uint8_t, 2> dense{};
        m.read_tensor("model.norm.weight", 0, dense);
        check(dense[0] == 0 && dense[1] == 0);
        fails(
            [&] {
                m.load_expert_rows(
                    layers, ExpertProjection::down, 0, 0, 1, 1530, payload
                );
            },
            "gpt_oss_layer"
        );
        fails(
            [&] { m.load_expert_rows(0, ExpertProjection::down, 0, 0, 1, 1530, {}); },
            "gpt_oss_payload_budget"
        );
        fs::resize_file(f.root / "model.safetensors", f.payload_start);
        fails([&] { m.check_unchanged(); }, "checkpoint_changed");
        fails(
            [&] { m.read_tensor("model.norm.weight", 0, dense); }, "checkpoint_changed"
        );
    }
    check(budget->used() == 0 && payload->used() == 0);
}

void manifest_rejections() {
    Fixture f;
    f.build(24, 32);
    auto budget = memory();
    auto reject = [&](const char* message) {
        f.save();
        fails([&] { Manifest m(f.root.c_str(), budget); }, message);
        check(budget->used() == 0);
    };
    auto original = f.index;
    f.index["metadata"]["total_size"] = 1;
    reject("gpt_oss_index_bytes");
    f.index = original;
    f.index["weight_map"].erase("model.norm.weight");
    reject("gpt_oss_index_coverage");
    f.index = original;
    f.index["weight_map"]["model.norm.weight"] = "../model.safetensors";
    f.save();
    // Shard itself rejects nonlocal paths before any file is opened.
    bool rejected = false;
    try {
        Manifest m(f.root.c_str(), budget);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    check(rejected && budget->used() == 0);
    f.index = original;
    const std::string key = "model.layers.0.self_attn.q_proj.weight";
    f.tensors[key]["shape"] = {2880, 4096};
    reject("gpt_oss_tensor_layout");
    f.tensors[key]["shape"] = {4096, 2880};
    f.tensors[key]["dtype"] = "F16";
    reject("gpt_oss_tensor_layout");
    f.tensors[key]["dtype"] = "BF16";
    const std::string extra = "unsupported.weight";
    f.tensors[extra] = f.tensors["model.norm.weight"];
    f.tensors.erase("model.norm.weight");
    f.index["weight_map"].erase("model.norm.weight");
    f.index["weight_map"][extra] = "model.safetensors";
    reject("gpt_oss_missing_tensor");
}

void multiple_shards() {
    const std::string prefix = "model.layers.23.mlp.experts.down_proj";
    auto budget = memory(), payload = memory();
    {
        Fixture f;
        f.build(24, 32);
        f.separate({"model.norm.weight", prefix + "_blocks", prefix + "_scales"});
        Manifest m(f.root.c_str(), budget);
        auto row =
            m.load_expert_rows(23, ExpertProjection::down, 31, 2879, 1, 1530, payload);
        check(row.view().rows == 1 && row.view().columns == 2880);
        std::array<std::uint8_t, 2> dense{};
        m.read_tensor("model.norm.weight", 0, dense);
        check(dense[0] == 0 && dense[1] == 0);
    }
    check(budget->used() == 0 && payload->used() == 0);
    {
        Fixture f;
        f.build(24, 32);
        f.separate({prefix + "_scales"});
        fails([&] { Manifest m(f.root.c_str(), budget); }, "gpt_oss_expert_shards");
    }
    {
        Fixture f;
        f.build(24, 32);
        f.add("unindexed.weight", {1});
        f.index["weight_map"].erase("unindexed.weight");
        f.save();
        fails([&] { Manifest m(f.root.c_str(), budget); }, "gpt_oss_index_coverage");
    }
    check(budget->used() == 0);
}
}  // namespace

int main() {
    try {
        config_only();
        complete_manifest(24, 32, 13761264768ULL);
        complete_manifest(36, 128, 65248815744ULL);
        manifest_rejections();
        multiple_shards();
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
