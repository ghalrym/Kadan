#include "kadan/gpt_oss_expert.hpp"

#include <bit>
#include <cmath>
#include <iostream>
#include <limits>
#include <source_location>

namespace {
void check(bool value, std::source_location at = std::source_location::current()) {
    if (!value)
        throw std::runtime_error("check_line_" + std::to_string(at.line()));
}

template <class F> void fails(F fn, std::string_view message) {
    try {
        fn();
    } catch (const std::invalid_argument& e) {
        check(e.what() == message);
        return;
    }
    throw std::runtime_error("expected_failure");
}

void swiglu_values() {
    // Scalar numerical fixtures only: no projection, layer or model forward.
    // Frozen eager BF16 values cover interleaving, the asymmetric gate clamp,
    // both linear clamps, +1, and the separate BF16 intermediate roundings.
    const std::array<float, 18> input{
        0, 0, 1, 3, -1, 2, -10, 0, 10, 10, 10, -10, 1, 0.00390625F, 1, -1, -100, 7
    };
    const std::array<float, 9> expected{
        0, 3.390625F, -0.462890625F, -0x1.bcp-22F, 56, -42, 0.84765625F, 0, -0.F
    };
    std::array<float, 9> output{};
    kadan::gpt_oss::swiglu(input, output);
    check(output == expected && std::signbit(output.back()));
    const float large = std::bit_cast<float>(0x7f7f0000U);
    const std::array<float, 4> extremes{-large, large, -large, -large};
    std::array<float, 2> zero{};
    kadan::gpt_oss::swiglu(extremes, zero);
    check(zero[0] == 0 && std::signbit(zero[0]));
    check(zero[1] == 0 && !std::signbit(zero[1]));
}

void swiglu_rejects() {
    std::array<float, 4> input{1, 1, 1, 1};
    std::array<float, 2> output{9, 9};
    fails([&] { kadan::gpt_oss::swiglu({}, {}); }, "gpt_oss_swiglu_shape");
    fails(
        [&] { kadan::gpt_oss::swiglu(input, std::span(output).first(1)); },
        "gpt_oss_swiglu_shape"
    );
    fails(
        [&] { kadan::gpt_oss::swiglu(input, std::span(input).first(2)); },
        "math_overlap"
    );
    input.back() = 1.001F;
    fails([&] { kadan::gpt_oss::swiglu(input, output); }, "gpt_oss_input_bf16");
    check(output == std::array<float, 2>{9, 9});
    input.back() = std::numeric_limits<float>::infinity();
    fails([&] { kadan::gpt_oss::swiglu(input, output); }, "linear_nonfinite");
    input.back() = std::numeric_limits<float>::quiet_NaN();
    fails([&] { kadan::gpt_oss::swiglu(input, output); }, "linear_nonfinite");
}

void routing() {
    using kadan::gpt_oss::route;
    std::array<float, 128> logits{};
    const std::array<std::size_t, 4> indices{127, 63, 1, 0};
    for (std::size_t rank = 0; rank < indices.size(); ++rank)
        logits[indices[rank]] = float(4 - rank);
    auto selected = route(logits);
    const std::array<float, 4> expected{
        0.64453125F, 0.2373046875F, 0.0869140625F, 0.031982421875F
    };
    check(selected.indices == indices && selected.weights == expected);
    // BF16 rounding is final; a second normalization changes these weights.
    check(
        selected.weights[0] + selected.weights[1] + selected.weights[2]
            + selected.weights[3]
        == 1.000732421875F
    );
    for (auto& logit : logits)
        logit += 128;
    check(route(logits).weights == expected);

    std::array<float, 32> equal{};
    selected = route(equal);
    check(selected.indices == std::array<std::size_t, 4>{0, 1, 2, 3});
    check(selected.weights == std::array<float, 4>{.25F, .25F, .25F, .25F});
    equal[31] = equal[30] = 1;
    check(route(equal).indices == std::array<std::size_t, 4>{30, 31, 0, 1});

    const float large = std::bit_cast<float>(0x7f7f0000U);
    equal.fill(-large);
    equal[31] = large;
    selected = route(equal);
    check(selected.indices == std::array<std::size_t, 4>{31, 0, 1, 2});
    check(selected.weights == std::array<float, 4>{1, 0, 0, 0});
    fails([&] { route(std::span(equal).first(4)); }, "gpt_oss_router_shape");
    equal[0] = 1.001F;
    fails([&] { route(equal); }, "gpt_oss_input_bf16");
    equal[0] = std::numeric_limits<float>::quiet_NaN();
    fails([&] { route(equal); }, "linear_nonfinite");
}

void summation() {
    using kadan::gpt_oss::weighted_sum;
    // Scalar/vector arithmetic only; no projection or expert/model forward.
    // First channel distinguishes FP32 reduction from per-expert BF16 rounding.
    std::array<float, 8> values{256, 1, 1, 2, -256, 3, 1, 4};
    std::array<float, 4> weights{.25F, .25F, .25F, .25F};
    std::array<float, 2> output{};
    weighted_sum(values, weights, output);
    check(output == std::array<float, 2>{.5F, 2.5F});
    // Unequal weights expose premature BF16 product rounding.
    const std::array<float, 4> products{1.0078125F, -1, 0, 0};
    const std::array<float, 4> fractional{.33203125F, .33203125F, 0, 0};
    std::array<float, 1> scalar{};
    weighted_sum(products, fractional, scalar);
    check(scalar[0] == .002593994140625F);
    // Cancellation distinguishes rank-ordered FP32 from double/regrouped sums.
    const std::array<float, 4> cancellation{16777216, 1, -16777216, 1};
    weighted_sum(cancellation, weights, scalar);
    check(scalar[0] == .25F);
    weights = {1, 0, 0, 0};
    weighted_sum(values, weights, output);
    check(output == std::array<float, 2>{256, 1});
    fails([&] { weighted_sum({}, weights, {}); }, "gpt_oss_sum_shape");
    fails(
        [&] { weighted_sum(values, std::span(weights).first(3), output); },
        "gpt_oss_sum_shape"
    );
    fails(
        [&] { weighted_sum(values, weights, std::span(values).first(2)); },
        "math_overlap"
    );
    fails(
        [&] { weighted_sum(values, weights, std::span(weights).first(2)); },
        "math_overlap"
    );
    weights[0] = -0.25F;
    fails([&] { weighted_sum(values, weights, output); }, "gpt_oss_routing_weight");
    weights[0] = 2;
    fails([&] { weighted_sum(values, weights, output); }, "gpt_oss_routing_weight");
    weights[0] = 0.1F;
    fails([&] { weighted_sum(values, weights, output); }, "gpt_oss_input_bf16");
    weights[0] = 1;
    values.back() = 1.001F;
    fails([&] { weighted_sum(values, weights, output); }, "gpt_oss_input_bf16");
    check(output == std::array<float, 2>{256, 1});
    values.fill(std::bit_cast<float>(0x7f7f0000U));
    weights.fill(1);
    fails([&] { weighted_sum(values, weights, output); }, "linear_nonfinite");
}
}  // namespace

int main() {
    try {
        swiglu_values();
        swiglu_rejects();
        routing();
        summation();
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
