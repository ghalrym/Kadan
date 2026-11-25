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
#include "kadan/inkling_gpu.hpp"

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
        (40ULL * 2 * (2 * 1024) + 8ULL * 2 * 2048) * 1024 * sizeof(std::uint16_t)
        + 48ULL * 3 * 6656 * sizeof(float)
        + 2ULL * kadan::inkling_gpu_boundary_bytes(2048)
        + 24ULL * 1024 * 16 * sizeof(float)
        + (16ULL * 1024 * 1024);
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
    // Full-attention slot: 9 positions, 2 floats/position -> prefix 8 compacts to 16 floats.
    state.layers.emplace_back(
        InklingLayerState{{Vec(18, 1.f), Vec(18, 2.f)}, Vec{10.f, 11.f}, false}
    );
    // Sliding-window slot: ring-buffered, copies whole KV + conv.
    state.layers.emplace_back(
        InklingLayerState{{Vec(8, 3.f), Vec(8, 4.f)}, Vec{12.f, 13.f}, true}
    );
    const auto full_bytes = state_memory_bytes(state);
    auto compact = compact_state(state, 8, 9);
    check(state_memory_bytes(compact) < full_bytes, "compact_reduces_bytes");
    const auto& c_full = std::get<InklingLayerState>(compact.layers[0]);
    check(!c_full.sliding_window, "checkpoint_full_flag");
    check(c_full.attention.keys.size() == 16 && c_full.attention.values.size() == 16,
          "checkpoint_full_compact_size");
    check(c_full.attention.keys[15] == 1.f, "checkpoint_full_key_prefix");
    check(c_full.attention.values[15] == 2.f, "checkpoint_full_value_prefix");
    check(c_full.conv == Vec({10.f, 11.f}), "checkpoint_full_conv");
    const auto& c_swa = std::get<InklingLayerState>(compact.layers[1]);
    check(c_swa.sliding_window, "checkpoint_swa_flag");
    check(c_swa.attention.keys == Vec(8, 3.f), "checkpoint_swa_keys");
    check(c_swa.attention.values == Vec(8, 4.f), "checkpoint_swa_values");
    check(c_swa.conv == Vec({12.f, 13.f}), "checkpoint_swa_conv");

    ModelState target;
    target.layers.emplace_back(
        InklingLayerState{{Vec(18, -1.f), Vec(18, -1.f)}, Vec{0.f, 0.f}, false}
    );
    target.layers.emplace_back(
        InklingLayerState{{Vec(8, -1.f), Vec(8, -1.f)}, Vec{0.f, 0.f}, true}
    );
    restore_compact_state(compact, target, 8, 9);
    const auto& restored_full = std::get<InklingLayerState>(target.layers[0]);
    check(restored_full.attention.keys[15] == 1.f, "restored_full_key_prefix");
    check(restored_full.attention.values[15] == 2.f, "restored_full_value_prefix");
    check(
        restored_full.attention.keys.back() == -1.f
            && restored_full.attention.values.back() == -1.f,
        "checkpoint_suffix_not_copied"
    );
    check(restored_full.conv == Vec({10.f, 11.f}), "restored_full_conv");
    const auto& restored_swa = std::get<InklingLayerState>(target.layers[1]);
    check(restored_swa.attention.keys == Vec(8, 3.f), "restored_swa_keys");
    check(restored_swa.attention.values == Vec(8, 4.f), "restored_swa_values");
    check(restored_swa.conv == Vec({12.f, 13.f}), "restored_swa_conv");
}

void inkling_gpu_geometry_and_scratch() {
    using G = kadan::InklingGeometry;
    static_assert(G::n_phys == 24);
    static_assert(G::n_slots == 48);
    static_assert(G::n_embd == 2304 && G::n_embd % 256 == 0);
    static_assert(G::n_ff_exp == 1152 && G::n_ff_exp % 32 == 0);
    std::size_t swa = 0, full = 0;
    for (std::size_t s = 0; s < G::n_slots; ++s) {
        if (G::is_swa_phys(G::slot_phys(s)))
            ++swa;
        else
            ++full;
    }
    check(swa == 40 && full == 8, "inkling_geometry_slot_split");
    // Both decode (1 row) and max prefill chunk (512 rows) must fit well inside
    // the 1 GiB LLM scratch envelope.
    const std::size_t decode_scratch = kadan::inkling_gpu_scratch_bytes(1);
    const std::size_t prefill_scratch = kadan::inkling_gpu_scratch_bytes(512);
    check(decode_scratch > 0 && decode_scratch < prefill_scratch, "inkling_scratch_monotonic");
    check(prefill_scratch < (512ULL << 20), "inkling_prefill_scratch_bounded");
    // At 65,536 context, FP16 KV mirrors + F32 conv + 2 GPU boundary snapshots
    // + params must stay well below 3 GiB so IQ4_NL (16.4 GiB) fits on 24 GiB.
    const std::size_t bnd_2k = kadan::inkling_gpu_boundary_bytes(2048);
    const std::size_t bnd_65k = kadan::inkling_gpu_boundary_bytes(65536);
    check(bnd_2k == bnd_65k && bnd_2k % sizeof(float) == 0, "inkling_gpu_boundary_bytes");
    const std::size_t dev_65k = inkling2_loop_device_state_bytes(65536);
    check(dev_65k < (3ULL << 30), "inkling_65k_device_state_fits_single_24g");
    check(dev_65k < inkling2_loop_state_bytes(65536), "inkling_fp16_smaller_than_host_f32");
}
}  // namespace

int main() {
    try {
        checkpoint_copy();
        round_trip();
        inkling_gpu_geometry_and_scratch();
        std::cout << "Same-instance text state round trips passed; no inference\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
