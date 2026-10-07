#include "kadan/cuda_plan.hpp"
#include "../src/fp8_numeric.hpp"
#include <array>
#include <bit>
#include <climits>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace {
void check(bool ok) { if (!ok) throw std::runtime_error("check_failed"); }
template<class Error = std::invalid_argument, class F>
void fails(F fn, std::string_view expected) {
    bool caught = false;
    try { fn(); } catch (const Error& error) { check(expected == error.what()); caught = true; }
    check(caught);
}
void decoder() {
    for (unsigned code = 0; code < 256; ++code) {
        if ((code & 127) == 127) continue; // NaN codes are rejected at admission.
        const float actual = kadan::cuda::detail::finite_e4m3fn(code);
        const float reference = kadan::quantization::e4m3fn(code);
        check(actual == reference && std::signbit(actual) == std::signbit(reference));
    }
}
void admission() {
    using namespace kadan;
    std::vector<std::uint8_t> weights(51, 0x38);
    std::array<float, 3> scales{0.1F, 2, 3};
    quantization::Matrix m{quantization::Encoding::modelopt_fp8, 3, 17, weights, {}, scales};
    const auto p = cuda::plan_fp8(m, 1280);
    check(p.rows == 3 && p.columns == 17 && p.weight_bytes == 51 && p.scale_bytes == 12 && p.scale_count == 3);
    check(p.scale_offset == 256 && p.input_offset == 512 && p.output_offset == 768);
    check(p.status_offset == 1024 && p.device_bytes == 1280);
    fails<std::length_error>([&] { cuda::plan_fp8(m, 1279); }, "cuda_device_budget");
    auto bad = m; bad.encoding = quantization::Encoding::modelopt_nvfp4;
    fails([&] { cuda::plan_fp8(bad, 1280); }, "cuda_requires_modelopt_fp8");
    for (auto size : {std::size_t{0}, std::size_t{INT_MAX} + 1}) {
        bad = m; bad.rows = size;
        fails([&] { cuda::plan_fp8(bad, 1280); }, "cuda_shape_limit");
        bad = m; bad.columns = size;
        fails([&] { cuda::plan_fp8(bad, 1280); }, "cuda_shape_limit");
    }
    bad = m; bad.weights = std::span(weights).first(50);
    fails([&] { cuda::plan_fp8(bad, 1280); }, "weight_shape");
    bad = m; bad.block_scales = std::span(weights).first(1);
    fails([&] { cuda::plan_fp8(bad, 1280); }, "unexpected_block_scales");
    for (std::size_t count : {0, 2}) {
        bad = m; bad.multipliers = std::span(scales).first(count);
        fails([&] { cuda::plan_fp8(bad, 1280); }, "fp8_scale_shape");
    }
    for (float scale : {0.0F, -0.0F, -1.0F, std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN()}) {
        scales[2] = scale;
        fails([&] { cuda::plan_fp8(m, 1280); }, "invalid_multiplier");
    }
    scales[2] = 3;
    for (auto code : {0x7f, 0xff}) {
        weights.back() = code;
        fails([&] { cuda::plan_fp8(m, 1280); }, "nonfinite_weight");
    }
    weights.back() = 0x38;
    bad = m; bad.multipliers = std::span(scales).first(1);
    check(cuda::plan_fp8(bad, 1280).scale_bytes == 4);
    std::vector<std::uint8_t> large(17 * 1025, 0x38);
    m = {quantization::Encoding::modelopt_fp8, 17, 1025, large, {}, std::span(scales).first(1)};
    const auto large_plan = cuda::plan_fp8(m, 65536);
    check(large_plan.device_bytes == 22784 && large_plan.input_offset >= large.size() + 4);
    check(large_plan.output_offset >= large_plan.input_offset + 1025 * sizeof(float));
    check(large_plan.status_offset >= large_plan.output_offset + 17 * sizeof(float));
}
void numerical_contract() {
    using namespace kadan;
    std::array<std::uint8_t, 2> weights{0x3b, 0xbb}; // +/-1.375
    std::array<float, 2> scales{1, std::bit_cast<float>(0x7f3a2e8bU)};
    quantization::Matrix m{quantization::Encoding::modelopt_fp8, 2, 1, weights, {}, scales};
    const double exact = 1.375 * static_cast<double>(scales[1]);
    check(exact > std::numeric_limits<float>::max());
    check(static_cast<float>(exact) == std::numeric_limits<float>::max());
    const std::array<float, 1> zero{};
    for (auto code : {0x3b, 0xbb}) {
        weights[1] = code;
        fails<std::overflow_error>([&] { cuda::plan_fp8(m, 1280); }, "nonfinite_result");
        fails<std::overflow_error>([&] { quantization::matvec(m, zero, 8); }, "nonfinite_result");
    }
    scales[1] = std::nextafter(scales[1], 0.0F);
    cuda::plan_fp8(m, 1280);
    check(quantization::matvec(m, zero, 8)[1] == 0);
    weights = {0x38, 0xb8}; scales = {std::numeric_limits<float>::max(), std::numeric_limits<float>::max()};
    cuda::plan_fp8(m, 1280); // Exactly representable +/-FLT_MAX is valid.
    weights = {0x38, 0x01}; scales = {std::numeric_limits<float>::denorm_min(), std::numeric_limits<float>::denorm_min()};
    cuda::plan_fp8(m, 1280); // Subnormal and underflow-to-zero are valid.
    const auto values = quantization::decode_rows(m, 0, 2, 8);
    check(values[0] == scales[0] && values[1] == 0);
    weights = {0x00, 0x80}; scales = {std::numeric_limits<float>::max(), std::numeric_limits<float>::max()};
    cuda::plan_fp8(m, 1280);
    const auto zeros = quantization::decode_rows(m, 0, 2, 8);
    check(zeros[0] == 0 && !std::signbit(zeros[0]) && zeros[1] == 0 && std::signbit(zeros[1]));
}
}
int main() {
    try { decoder(); admission(); numerical_contract(); }
    catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
