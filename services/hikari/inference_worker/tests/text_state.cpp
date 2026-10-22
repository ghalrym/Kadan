// State ownership only: constructor parameters are scalar placeholders and no
// projection tensors exist. No forward pass, generation or device execution.
#include <algorithm>
#include <array>
#include <filesystem>
#include <ggml.h>
#include <gguf.h>
#include <iostream>
#include <numeric>
#include <type_traits>
#include <unistd.h>

#include "../src/models/text/gguf_model.hpp"

namespace {
using namespace kadan::text;

void check(bool condition, const char* message) {
    if (!condition)
        throw std::runtime_error(message);
}

class Parameters {
public:
    std::filesystem::path root;
    std::vector<std::string> names;

    explicit Parameters(bool qwen4) {
        char pattern[] = "/tmp/kadan-text-state-XXXXXX";
        const auto* directory = mkdtemp(pattern);
        check(directory, "state_parameters_directory");
        root = directory;
        const std::unique_ptr<ggml_context, decltype(&ggml_free)> tensors(
            ggml_init({1 << 20, nullptr, false}), ggml_free
        );
        const std::unique_ptr<gguf_context, decltype(&gguf_free)> metadata(
            gguf_init_empty(), gguf_free
        );
        const auto add = [&](const std::string& name) {
            auto* tensor = ggml_new_tensor_1d(tensors.get(), GGML_TYPE_F32, 1);
            ggml_set_name(tensor, name.c_str());
            *static_cast<float*>(tensor->data) = 0;
            gguf_add_tensor(metadata.get(), tensor);
            names.push_back(name);
        };
        add(qwen4 ? "output_hc_norm.weight" : "output_norm.weight");
        for (std::size_t i = 0; i < (qwen4 ? 48 : 40); ++i) {
            const auto prefix = "blk." + std::to_string(i) + ".";
            add(prefix + "ffn_gate_inp_shexp.weight");
            if (qwen4) {
                for (const auto* suffix : {"hc_attn_", "hc_ffn_"}) {
                    add(prefix + suffix + "norm.weight");
                    add(prefix + suffix + "inject.weight");
                }
            } else {
                add(prefix + "attn_norm.weight");
                add(prefix + "post_attention_norm.weight");
            }
            if ((i + 1) % 4 == 0) {
                add(prefix + "attn_q_norm.weight");
                add(prefix + "attn_k_norm.weight");
                if (qwen4) {
                    add(prefix + "indexer.q_norm.weight");
                    add(prefix + "indexer.k_norm.weight");
                }
            } else {
                for (const auto* name :
                     {"ssm_conv1d.weight", "ssm_dt.bias", "ssm_a", "ssm_norm.weight"})
                    add(prefix + name);
            }
        }
        if (qwen4) {
            for (const auto* name :
                 {"ple_norm_key.weight",
                  "ple_norm_query.weight",
                  "ple_norm_conv.weight",
                  "ple_conv1d.weight"})
                add(std::string("blk.1.") + name);
            const std::array<std::int64_t, 3> multipliers{1, 2, 3};
            const std::array<std::int64_t, 16> offsets{},
                sizes{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
            for (const auto& [name, values] : std::initializer_list<
                     std::pair<const char*, std::span<const std::int64_t>>>{
                     {"qwen4exp.ple.layer_multipliers", multipliers},
                     {"qwen4exp.ple.head_offsets", offsets},
                     {"qwen4exp.ple.head_vocab_sizes", sizes}
                 })
                gguf_set_arr_data(
                    metadata.get(), name, GGUF_TYPE_INT64, values.data(), values.size()
                );
        }
        check(
            gguf_write_to_file(metadata.get(), (root / "state.gguf").c_str(), false),
            "state_parameters_write"
        );
    }

    ~Parameters() {
        std::error_code error;
        std::filesystem::remove_all(root, error);
    }
};

struct NoCompute final : kadan::DenseCompute {
    void dense(
        std::span<const float>,
        std::span<const float>,
        std::span<const float>,
        std::size_t,
        std::size_t,
        std::span<float>,
        const std::atomic_bool&,
        bool
    ) override {
        throw std::runtime_error("unexpected_state_projection");
    }

    void
    copy_recurrent_state(std::uint64_t, std::span<float>, std::span<float>) override {
        throw std::runtime_error("unexpected_state_device_copy");
    }
};

template <class F> void buffers(ModelState& state, F fn) {
    for (auto& layer : state.layers)
        std::visit(
            [&](auto& saved) {
                using T = std::decay_t<decltype(saved)>;
                if constexpr (std::is_same_v<T, RecurrentState>) {
                    fn(saved.convolution);
                    fn(saved.recurrent);
                } else if constexpr (std::is_same_v<T, AttentionState>) {
                    fn(saved.keys);
                    fn(saved.values);
                } else {
                    fn(saved.attention.keys);
                    fn(saved.attention.values);
                    fn(saved.raw_keys);
                    fn(saved.pooled);
                }
            },
            layer
        );
    fn(state.ple);
}

void round_trip(bool qwen4) {
    Parameters parameters(qwen4);
    GgufWeights weights(parameters.root.c_str(), nullptr);
    for (const auto& name : parameters.names)
        weights.vector(name, {1});
    NoCompute compute;
    auto model = qwen4 ? qwen4exp_model(weights, compute, 9)
                       : qwen35moe_model(weights, compute, 9);
    auto state = model->take_state();
    std::vector<std::size_t> sizes;
    float value = 0;
    buffers(state, [&](Vec& buffer) {
        sizes.push_back(buffer.size());
        std::fill(buffer.begin(), buffer.end(), ++value);
    });
    if (qwen4)
        state.recent = {17, 29};
    for (const bool device : {false, true}) {
        for (auto& layer : state.layers)
            if (auto* recurrent = std::get_if<RecurrentState>(&layer))
                recurrent->device = device;
        // Repeat on the same emptied instance, as checkpoint save/restore do.
        for (int round = 0; round < 2; ++round) {
            model->restore_state(std::move(state));
            state = model->take_state();
            check(state.layers.size() == (qwen4 ? 48 : 40), "state_layer_count");
            std::size_t index = 0;
            buffers(state, [&](Vec& buffer) {
                check(buffer.size() == sizes.at(index++), "state_buffer_size");
                check(
                    std::all_of(
                        buffer.begin(),
                        buffer.end(),
                        [&](float v) { return v == float(index); }
                    ),
                    "state_buffer_values"
                );
            });
            check(index == sizes.size(), "state_buffer_count");
            check(
                state.recent
                    == (qwen4 ? std::vector<unsigned>{17, 29}
                              : std::vector<unsigned>{}),
                "state_recent"
            );
            for (const auto& layer : state.layers)
                if (const auto* recurrent = std::get_if<RecurrentState>(&layer))
                    check(recurrent->device == device, "state_device_flag");
        }
    }
}

void checkpoint_copy() {
    // Nine context positions hold two complete pools of four. All sixteen
    // pooled values belong to the eight-token prefix, including the last one.
    ModelState state;
    state.layers.emplace_back(QsaState{{Vec(18, 1), Vec(18, 2)}, Vec(72, 3), Vec(16)});
    state.layers.emplace_back(RecurrentState{{10, 11}, {12, 13}, true});
    auto& source = std::get<QsaState>(state.layers[0]);
    std::iota(source.pooled.begin(), source.pooled.end(), 4.f);
    state.ple = {14, 15};
    state.recent = {17, 29};
    ModelState checkpoint;
    copy_state(state, checkpoint, 8, 9);
    // Exercise reuse of an already allocated checkpoint as well as first save.
    for (int round = 0; round < 2; ++round) {
        auto& qsa = std::get<QsaState>(checkpoint.layers[0]);
        check(qsa.pooled == source.pooled, "checkpoint_complete_pools");
        check(qsa.attention.keys[15] == 1, "checkpoint_key_prefix");
        check(qsa.attention.values[15] == 2, "checkpoint_value_prefix");
        check(qsa.raw_keys[63] == 3, "checkpoint_index_prefix");
        const auto& recurrent = std::get<RecurrentState>(checkpoint.layers[1]);
        check(
            recurrent.convolution == Vec({10, 11})
                && recurrent.recurrent == Vec({12, 13}) && recurrent.device,
            "checkpoint_recurrence"
        );
        check(
            checkpoint.ple == state.ple && checkpoint.recent == state.recent,
            "checkpoint_ngram"
        );
        buffers(checkpoint, [](Vec& buffer) {
            std::fill(buffer.begin(), buffer.end(), -1);
        });
        copy_state(state, checkpoint, 8, 9);
        check(
            qsa.attention.keys.back() == -1 && qsa.attention.values.back() == -1
                && qsa.raw_keys.back() == -1,
            "checkpoint_suffix_not_copied"
        );
    }
}
}  // namespace

int main() {
    try {
        checkpoint_copy();
        round_trip(false);
        round_trip(true);
        std::cout << "Same-instance text state round trips passed; no inference\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
