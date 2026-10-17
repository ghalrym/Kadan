#include <array>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <vector>

#include "kadan/quantization.hpp"

using namespace kadan::quantization;

namespace {
void check(bool ok) {
    if (!ok)
        throw std::runtime_error("check_failed");
}

template <class Error = std::invalid_argument, class F>
void fails(F fn, std::string_view expected) {
    bool failed = false;
    try {
        fn();
    } catch (const Error& error) {
        check(error.what() == expected);
        failed = true;
    }
    check(failed);
}

void three_bit() {
    // Golden bytes from eight explicit codes 0,1,2,3,4,5,6,7, repeated.
    // Every third triplet crosses a U32 boundary.
    std::vector<std::uint8_t> packed;
    for (unsigned i = 0; i < 32; ++i)
        packed.insert(packed.end(), {0x88, 0xc6, 0xfa});
    const std::array<float, 4> scales{.5f, -2.f, 0.f, 3.f};
    const std::array<float, 4> biases{-1.f, 4.f, 9.f, -8.f};
    const MlxAffineMatrix matrix{2, 128, 3, 64, packed, scales, biases};
    std::array<float, 128> output{};
    decode_mlx_affine_rows(matrix, 0, output);
    for (std::size_t i = 0; i < output.size(); ++i)
        check(
            output[i] == (i < 64 ? .5f * float(i % 8) - 1.f : -2.f * float(i % 8) + 4.f)
        );
    decode_mlx_affine_rows(matrix, 1, output);
    for (std::size_t i = 0; i < output.size(); ++i)
        check(output[i] == (i < 64 ? 9.f : 3.f * float(i % 8) - 8.f));
    decode_mlx_affine_rows(matrix, 2, {});
}

void four_bit() {
    const std::array<std::uint8_t, 8> pattern{
        0x10, 0x32, 0x54, 0x76, 0x98, 0xba, 0xdc, 0xfe
    };
    std::vector<std::uint8_t> packed;
    for (unsigned i = 0; i < 4; ++i)
        packed.insert(packed.end(), pattern.begin(), pattern.end());
    const std::array<float, 1> scales{2.f}, biases{-3.f};
    const MlxAffineMatrix matrix{1, 64, 4, 64, packed, scales, biases};
    std::array<float, 64> output{};
    decode_mlx_affine_rows(matrix, 0, output);
    for (std::size_t i = 0; i < output.size(); ++i)
        check(output[i] == 2.f * float(i % 16) - 3.f);
}

void invalid_layouts() {
    std::array<std::uint8_t, 24> packed{};
    std::array<float, 1> scales{1}, biases{0};
    MlxAffineMatrix matrix{1, 64, 3, 64, packed, scales, biases};
    std::array<float, 64> output{};
    auto decode = [&] {
        decode_mlx_affine_rows(matrix, 0, output);
    };
    matrix.bits = 2;
    fails(decode, "mlx_affine_bits");
    matrix.bits = 3;
    matrix.group_size = 32;
    fails(decode, "mlx_affine_group_alignment");
    matrix.group_size = 64;
    matrix.weights = std::span(packed).first(23);
    fails(decode, "weight_shape");
    matrix.weights = packed;
    matrix.biases = {};
    fails(decode, "mlx_affine_scale_shape");
    matrix.biases = biases;
    fails(
        [&] { decode_mlx_affine_rows(matrix, 0, std::span(output).first(63)); },
        "row_range"
    );
    fails([&] { decode_mlx_affine_rows(matrix, 1, output); }, "row_range");
    scales[0] = std::numeric_limits<float>::infinity();
    fails(decode, "mlx_affine_nonfinite_scale");
    scales[0] = 1;
    biases[0] = std::numeric_limits<float>::quiet_NaN();
    fails(decode, "mlx_affine_nonfinite_bias");
    biases[0] = 0;
    matrix.scales = std::span(output).first(1);
    fails(decode, "output_overlap");
    matrix.scales = scales;
    scales[0] = std::numeric_limits<float>::max();
    packed[0] = 7;
    fails<std::overflow_error>(decode, "nonfinite_result");
}
}  // namespace

int main() {
    try {
        three_bit();
        four_bit();
        invalid_layouts();
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
