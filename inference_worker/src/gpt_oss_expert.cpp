#include "kadan/gpt_oss_expert.hpp"

#include <algorithm>
#include <cmath>
#include <string>

#include "kadan/checkpoint_floats.hpp"
#include "kadan/linear_attention.hpp"
#include "layer_math_validation.hpp"

namespace kadan::gpt_oss {
namespace {
using math::detail::require;

void bf16_input(std::span<const float> input) {
    for (float value : input)
        require(linear::bf16_round(value) == value, "gpt_oss_input_bf16");
}

void cancelled(const std::atomic_bool& cancel) {
    if (cancel.load())
        throw std::runtime_error("gpt_oss_expert_cancelled");
}
}  // namespace

void swiglu(std::span<const float> input, std::span<float> output) {
    require(
        !output.empty() && output.size() <= Config::intermediate
            && input.size() == 2 * output.size(),
        "gpt_oss_swiglu_shape"
    );
    math::detail::output_alias(input, output, false);
    bf16_input(input);
    for (std::size_t i = 0; i < output.size(); ++i) {
        const float gate = std::min(input[2 * i], 7.F);
        const float up = std::clamp(input[2 * i + 1], -7.F, 7.F);
        // A negative product overflowing BF16 still has a sigmoid of zero.
        // This also avoids overflowing the scalar multiply for finite BF16 input.
        float probability = 0;
        if (gate >= -128.F) {
            const float z = linear::bf16_round(1.702F * gate);
            const float e = std::exp(-std::abs(z));
            probability = linear::bf16_round(z >= 0 ? 1 / (1 + e) : e / (1 + e));
        }
        const float gated = linear::bf16_round(gate * probability);
        output[i] = linear::bf16_round(gated * linear::bf16_round(up + 1));
    }
}

void expert(
    const Manifest& manifest,
    std::size_t layer,
    std::size_t selected,
    std::span<const float> input,
    std::span<float> output,
    std::shared_ptr<checkpoint::MemoryBudget> workspace,
    const std::atomic_bool& cancel,
    const checkpoint::TensorReader& reader
) {
    require(layer < manifest.config().layers, "gpt_oss_layer");
    require(selected < manifest.config().experts, "gpt_oss_expert");
    require(
        input.size() == Config::hidden && output.size() == Config::hidden,
        "gpt_oss_expert_shape"
    );
    require(bool(workspace), "gpt_oss_payload_budget");
    math::detail::output_alias(input, output, false);
    bf16_input(input);
    cancelled(cancel);

    constexpr std::size_t tile = 64;
    std::pmr::vector<float> activation(Config::intermediate, workspace.get());
    std::pmr::vector<float> projected(tile, workspace.get()),
        bias(tile, workspace.get());
    std::array<std::uint8_t, tile * 2> scratch{};
    std::pmr::string prefix("model.layers.", workspace.get());
    prefix += std::to_string(layer);
    prefix += ".mlp.experts.";
    auto read = [&](std::string_view name, std::size_t offset, auto bytes) {
        cancelled(cancel);
        if (reader)
            reader(name, offset, bytes);
        else
            manifest.read_tensor(name, offset, bytes);
        cancelled(cancel);
    };
    // Reference target keeps std::function from allocating outside the budget.
    const checkpoint::TensorReader tensor_reader = std::ref(read);
    manifest.check_unchanged();
    for (auto projection : {ExpertProjection::gate_up, ExpertProjection::down}) {
        const auto rows = projection == ExpertProjection::gate_up
                              ? 2 * Config::intermediate
                              : Config::hidden;
        std::pmr::string name(prefix, workspace.get());
        name += projection == ExpertProjection::gate_up ? "gate_up_proj_bias"
                                                        : "down_proj_bias";
        const std::span<const float> x = projection == ExpertProjection::gate_up
                                             ? input
                                             : std::span<const float>(activation);
        for (std::size_t first = 0; first < rows; first += tile) {
            cancelled(cancel);
            const auto count = std::min(tile, rows - first);
            const auto weights = manifest.load_expert_rows(
                layer,
                projection,
                selected,
                first,
                count,
                count * (x.size() / 2 + x.size() / 32),
                workspace,
                tensor_reader
            );
            checkpoint::read_floats(
                checkpoint::Dtype::bf16,
                std::span(bias).first(count),
                scratch,
                cancel,
                [&](std::size_t offset, auto bytes) {
                    read(name, (selected * rows + first) * 2 + offset, bytes);
                },
                [](std::size_t) {}
            );
            auto y = std::span(projected).first(count);
            quantization::matvec(weights.view(), x, y);
            for (std::size_t i = 0; i < count; ++i)
                y[i] = linear::bf16_round(linear::bf16_round(y[i]) + bias[i]);
            if (projection == ExpertProjection::gate_up)
                swiglu(y, std::span(activation).subspan(first / 2, count / 2));
            else
                std::copy(y.begin(), y.end(), output.begin() + first);
        }
    }
    cancelled(cancel);
    manifest.check_unchanged();
}
}  // namespace kadan::gpt_oss
