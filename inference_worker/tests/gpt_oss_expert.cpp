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
}  // namespace

int main() {
    try {
        swiglu_values();
        swiglu_rejects();
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
