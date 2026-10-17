#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <unistd.h>
#include <vector>

#include <nlohmann/json.hpp>

#include "kadan/s1_mini.hpp"

namespace {
void check(bool value) {
    if (!value)
        throw std::runtime_error("test_failed");
}

template <class F> void fails(F operation) {
    try {
        operation();
    } catch (const std::exception&) {
        return;
    }
    throw std::runtime_error("expected_failure");
}

struct Directory {
    std::string path;

    Directory() {
        char name[] = "/tmp/kadan-s1-metadata-XXXXXX";
        const auto result = ::mkdtemp(name);
        if (!result)
            throw std::runtime_error("temporary_directory");
        path = result;
    }

    ~Directory() {
        std::filesystem::remove_all(path);
    }

    void write(const nlohmann::json& value) const {
        std::ofstream stream(path + "/config.json");
        stream << value.dump();
        if (!stream)
            throw std::runtime_error("metadata_write");
    }
};
}  // namespace

int main() {
    // Metadata and rejected-request lifecycle only. No weights or forward passes.
    const auto config = nlohmann::json::parse(R"({
        "architectures": ["Qwen3ForCausalLM"],
        "model_type": "qwen3", "hidden_act": "silu", "torch_dtype": "bfloat16",
        "attention_bias": false, "use_sliding_window": false,
        "tie_word_embeddings": true, "rope_scaling": null, "sliding_window": null,
        "hidden_size": 1024, "intermediate_size": 3072, "num_hidden_layers": 28,
        "num_attention_heads": 16, "num_key_value_heads": 8, "head_dim": 128,
        "vocab_size": 151936, "max_position_embeddings": 40960,
        "rms_norm_eps": 0.000001, "rope_theta": 1000000, "eos_token_id": 151645
    })");
    Directory directory;
    directory.write(config);
    auto budget = std::make_shared<kadan::checkpoint::MemoryBudget>(1024 * 1024);
    const auto parsed = kadan::s1::read_config(directory.path.c_str(), budget);
    check(parsed.hidden == 1024 && parsed.heads * parsed.head_dim == 2048);
    check(parsed.kv_heads == 8 && parsed.layers == 28 && parsed.vocabulary == 151936);
    check(budget->used() == 0);

    for (const auto& change : std::vector<nlohmann::json>{
             {{"model_type", "qwen3_moe"}},
             {{"architectures", {"Qwen3MoeForCausalLM"}}},
             {{"head_dim", 64}},
             {{"hidden_size", 2048}},
             {{"num_key_value_heads", 16}},
             {{"tie_word_embeddings", false}},
             {{"use_sliding_window", true}},
             {{"attention_bias", true}},
             {{"rope_scaling", {{"rope_type", "yarn"}}}},
             {{"sliding_window", 1024}},
             {{"rms_norm_eps", 0}},
             {{"eos_token_id", 0}}
         }) {
        auto unsupported = config;
        unsupported.update(change);
        directory.write(unsupported);
        fails([&] { (void)kadan::s1::read_config(directory.path.c_str(), budget); });
        check(budget->used() == 0);
    }
    directory.write(config);
    auto resources =
        std::make_shared<kadan::Resources>(kadan::Footprint{32 * 1024 * 1024});
    kadan::s1::Model model(resources);
    std::atomic_bool cancel = false;
    fails([&] { model.begin_request(1); });
    fails([&] { (void)model.config(); });
    check(model.is_eos(151645) && model.is_eos(151643) && !model.is_eos(42));
    // Missing generation metadata cannot leave an admission or partially loaded model.
    fails([&] { model.load(directory.path, cancel); });
    check(resources->snapshot().residents == 0 && resources->snapshot().used[0] == 0);
    cancel = true;
    fails([&] { model.load(directory.path, cancel); });
    check(resources->snapshot().residents == 0);
    model.reset_request();
    model.close();
    check(resources->snapshot().used[0] == 0);
    // Reject metadata before allocation when the worker cannot admit its envelope.
    auto small = std::make_shared<kadan::Resources>(kadan::Footprint{1024});
    kadan::s1::Model denied(small);
    cancel = false;
    fails([&] { denied.load(directory.path, cancel); });
    check(small->snapshot().residents == 0);
    std::cout << "s1 mini metadata ok\n";
}
