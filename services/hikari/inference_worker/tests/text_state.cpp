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
    std::vector<std::string> vectors;
    std::vector<std::string> tables;

    Parameters() {
        char pattern[] = "/tmp/kadan-text-state-XXXXXX";
        const auto* directory = mkdtemp(pattern);
        check(directory, "state_parameters_directory");
        root = directory;
        const std::unique_ptr<ggml_context, decltype(&ggml_free)> tensors(
            ggml_init({4 << 20, nullptr, false}), ggml_free
        );
        const std::unique_ptr<gguf_context, decltype(&gguf_free)> metadata(
            gguf_init_empty(), gguf_free
        );
        const auto add_vec = [&](const std::string& name) {
            auto* tensor = ggml_new_tensor_1d(tensors.get(), GGML_TYPE_F32, 1);
            ggml_set_name(tensor, name.c_str());
            *static_cast<float*>(tensor->data) = 0;
            gguf_add_tensor(metadata.get(), tensor);
            vectors.push_back(name);
        };
        const auto add_table = [&](const std::string& name) {
            auto* tensor = ggml_new_tensor_2d(tensors.get(), GGML_TYPE_F32, 2, 2);
            ggml_set_name(tensor, name.c_str());
            std::fill_n(static_cast<float*>(tensor->data), 4, 0.f);
            gguf_add_tensor(metadata.get(), tensor);
            tables.push_back(name);
        };
        add_vec("token_embd_norm.weight");
        add_vec("output_norm.weight");
        add_vec("output_res_query.weight");
        for (std::size_t i = 0; i < 24; ++i) {
            const auto prefix = "blk." + std::to_string(i) + ".";
            const bool is_swa = (i % 6) != 5;
            if (is_swa) {
                add_vec(prefix + "attn_norm.weight");
            } else {
                add_vec(prefix + "attn_q_inp_norm.weight");
                add_vec(prefix + "attn_k_inp_norm.weight");
                add_vec(prefix + "attn_v_inp_norm.weight");
            }
            add_table(prefix + "attn_q_clip.weight");
            add_table(prefix + "attn_rel_bias.weight");
            add_table(prefix + "attn_rel_clip.weight");
            add_table(prefix + "attn_k_conv.weight");
            add_table(prefix + "attn_v_conv.weight");
            add_table(prefix + "attn_post_conv.weight");
            add_table(prefix + "ffn_post_conv.weight");
            if (i > 0) {
                if (is_swa)
                    add_vec(prefix + "attn_res_query.weight");
                else
                    add_table(prefix + "attn_res_query.weight");
            }
            add_vec(prefix + "ffn_res_query.weight");
            add_vec(prefix + "ffn_norm.weight");
            add_vec(prefix + "exp_probs_b.bias");
            if (i >= 6 && i < 18)
                add_table(prefix + "loop_msg_query.weight");
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
                if constexpr (std::is_same_v<T, AttentionState>) {
                    fn(saved.keys);
                    fn(saved.values);
                } else {
                    fn(saved.attention.keys);
                    fn(saved.attention.values);
                    fn(saved.conv);
                }
            },
            layer
        );
}

void round_trip() {
    Parameters parameters;
    GgufWeights weights(parameters.root.c_str(), nullptr);
    for (const auto& name : parameters.vectors)
        weights.vector(name, {1});
    for (const auto& name : parameters.tables)
        weights.table(name, 2, 2);
    NoCompute compute;
    // Context 2048 > SWA window 1024 verifies 40 ring-buffered SWA slots and
    // 8 full-context slots across the 48 logical Inkling 2 passes.
    auto model = inkling2_loop_model(weights, compute, 2048);
    const std::size_t expected_device_state =
        (40ULL * 2 * (2 * 1024) + 8ULL * 2 * 2048) * 1024 * sizeof(float)
        + 24ULL * 1024 * 16 * sizeof(float);
    check(model->device_state_bytes() == expected_device_state, "inkling_device_state_bytes");
    auto state = model->take_state();
    check(state.layers.size() == 48, "inkling_slot_count");
    std::size_t swa_slots = 0, full_slots = 0;
    for (const auto& layer : state.layers) {
        const auto& slot = std::get<InklingLayerState>(layer);
        if (slot.sliding_window) {
            ++swa_slots;
            check(slot.attention.keys.size() == 1024 * 1024, "swa_kv_capacity");
            check(slot.attention.values.size() == 1024 * 1024, "swa_v_capacity");
        } else {
            ++full_slots;
            check(slot.attention.keys.size() == 2048 * 1024, "full_kv_capacity");
            check(slot.attention.values.size() == 2048 * 1024, "full_v_capacity");
        }
        check(slot.conv.size() == 6656 * 3, "conv_state_size");
    }
    check(swa_slots == 40 && full_slots == 8, "swa_full_partition");

    std::vector<std::size_t> sizes;
    float value = 0;
    buffers(state, [&](Vec& buffer) {
        sizes.push_back(buffer.size());
        std::fill(buffer.begin(), buffer.end(), ++value);
    });
    // Repeat on the same emptied instance, as checkpoint save/restore do.
    for (int round = 0; round < 2; ++round) {
        model->restore_state(std::move(state));
        state = model->take_state();
        check(state.layers.size() == 48, "state_layer_count");
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
    }
}

void checkpoint_copy() {
    ModelState state;
    // Full-attention slot: 9 positions, 2 floats/position -> prefix 8 copies 16 floats.
    state.layers.emplace_back(
        InklingLayerState{{Vec(18, 1.f), Vec(18, 2.f)}, Vec{10.f, 11.f}, false}
    );
    // Sliding-window slot: ring-buffered, copies whole KV + conv.
    state.layers.emplace_back(
        InklingLayerState{{Vec(8, 3.f), Vec(8, 4.f)}, Vec{12.f, 13.f}, true}
    );
    ModelState checkpoint;
    copy_state(state, checkpoint, 8, 9);
    // Exercise reuse of an already allocated checkpoint as well as first save.
    for (int round = 0; round < 2; ++round) {
        const auto& full = std::get<InklingLayerState>(checkpoint.layers[0]);
        check(!full.sliding_window, "checkpoint_full_flag");
        check(full.attention.keys[15] == 1.f, "checkpoint_full_key_prefix");
        check(full.attention.values[15] == 2.f, "checkpoint_full_value_prefix");
        check(full.conv == Vec({10.f, 11.f}), "checkpoint_full_conv");
        const auto& swa = std::get<InklingLayerState>(checkpoint.layers[1]);
        check(swa.sliding_window, "checkpoint_swa_flag");
        check(swa.attention.keys == Vec(8, 3.f), "checkpoint_swa_keys");
        check(swa.attention.values == Vec(8, 4.f), "checkpoint_swa_values");
        check(swa.conv == Vec({12.f, 13.f}), "checkpoint_swa_conv");
        buffers(checkpoint, [](Vec& buffer) {
            std::fill(buffer.begin(), buffer.end(), -1.f);
        });
        copy_state(state, checkpoint, 8, 9);
        const auto& reused_full = std::get<InklingLayerState>(checkpoint.layers[0]);
        check(
            reused_full.attention.keys.back() == -1.f
                && reused_full.attention.values.back() == -1.f,
            "checkpoint_suffix_not_copied"
        );
    }
}
}  // namespace

int main() {
    try {
        checkpoint_copy();
        round_trip();
        std::cout << "Same-instance text state round trips passed; no inference\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
