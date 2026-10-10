#include "kadan/quantization.hpp"
#include "../src/nvfp4_numeric.hpp"

#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string_view>

using namespace kadan::quantization;
namespace {
void check(bool ok) { if (!ok) throw std::runtime_error("check_failed"); }
template<class Error = std::invalid_argument, class F> void fails(F fn, std::string_view expected) {
    bool failed = false;
    try { fn(); }
    catch (const Error& error) { check(error.what() == expected); failed = true; }
    check(failed);
}
void bounded_global() {
    using kadan::cuda::detail::bounded_nvfp4_global;
    using kadan::cuda::detail::weight_overflows;
    const float bound=std::numeric_limits<float>::max()/4096.f;
    check(bounded_nvfp4_global(bound)&&bounded_nvfp4_global(-bound));
    check(!bounded_nvfp4_global(std::nextafter(bound,INFINITY)));
    check(!bounded_nvfp4_global(std::nextafter(-bound,-INFINITY)));
    check(!bounded_nvfp4_global(INFINITY)&&!bounded_nvfp4_global(-INFINITY));
    check(!bounded_nvfp4_global(std::numeric_limits<float>::quiet_NaN()));
    for(unsigned code=0;code<16;++code)for(unsigned scale=0;scale<127;++scale){
        const float local=e2m1(code)*e4m3fn(scale);
        check(!weight_overflows(local,bound)&&!weight_overflows(local,-bound));
    }
}
void primitives() {
    // Explicit golden values, not production exponent arithmetic.
    const std::array<float, 16> fp4{0, .5F, 1, 1.5F, 2, 3, 4, 6, -0.F, -.5F, -1, -1.5F, -2, -3, -4, -6};
    for (unsigned i = 0; i < fp4.size(); ++i) {
        check(e2m1(i) == fp4[i]);
        check(std::signbit(e2m1(i)) == (i >= 8));
    }
    fails([] { e2m1(16); }, "invalid_fp4_code");
    // Exhaustively construct E4M3 positive bins from explicit bin bases/steps,
    // then apply sign. The top bin ends at 448; 0x7f and 0xff are NaNs, not inf.
    const std::array<float, 16> base{0, .015625F, .03125F, .0625F, .125F, .25F,
                                   .5F, 1, 2, 4, 8, 16, 32, 64, 128, 256};
    const std::array<float, 16> step{.001953125F, .001953125F, .00390625F, .0078125F,
                                   .015625F, .03125F, .0625F, .125F, .25F, .5F, 1, 2, 4, 8, 16, 32};
    for (unsigned code = 0; code < 256; ++code) {
        if ((code & 127) == 127) { check(std::isnan(e4m3fn(code))); continue; }
        float expected = base[(code & 127) / 8] + step[(code & 127) / 8] * (code % 8);
        if (code >= 128) expected = -expected;
        check(e4m3fn(code) == expected);
        check(std::signbit(e4m3fn(code)) == (code >= 128));
    }
    const std::array<std::uint8_t, 4> two{0, 0, 0, 64}, minus_zero{0, 0, 0, 128};
    check(fp32_le(two) == 2);
    check(std::signbit(fp32_le(minus_zero)));
    fails([&] { fp32_le(std::span(two).first(3)); }, "fp32_byte_count");
}
void packed_byte_order() {
    std::array<std::uint8_t, 256> bytes{};
    for (unsigned i = 0; i < bytes.size(); ++i) bytes[i] = i;
    std::array<std::uint8_t, 32> scales{}; scales.fill(0x38);
    const std::array<float, 1> global{1};
    const std::array<float, 16> golden{0, .5F, 1, 1.5F, 2, 3, 4, 6, -0.F, -.5F, -1, -1.5F, -2, -3, -4, -6};
    Matrix m{Encoding::modelopt_nvfp4, 1, 512, bytes, scales, global};
    const auto decoded = decode_rows(m, 0, 1, 2048);
    for (unsigned i = 0; i < bytes.size(); ++i) {
        check(decoded[2 * i] == golden[i % 16]);
        check(decoded[2 * i + 1] == golden[i / 16]);
        check(std::signbit(decoded[2 * i]) == (i % 16 >= 8));
        check(std::signbit(decoded[2 * i + 1]) == (i / 16 >= 8));
    }
}
void nvfp4() {
    // Two rows, two blocks each. Low nibble comes first; scales distinguish
    // both the block and row axes. Entire fixture is 32 packed bytes.
    std::vector<std::uint8_t> packed(32, 0xA2); // +1, -1 alternating.
    packed[0] = 0x71; // +0.5, +6; catches nibble reversal.
    const std::array<std::uint8_t, 4> scales{0x38, 0x40, 0x30, 0x44}; // 1, 2, .5, 3
    const std::array<float, 1> global{2};
    Matrix m{Encoding::modelopt_nvfp4, 2, 32, packed, scales, global};
    const auto rows = decode_rows(m, 0, 2, 256);
    std::vector<float> expected;
    for (int i = 0; i < 64; ++i) {
        const float factor = i < 16 ? 2.F : i < 32 ? 4.F : i < 48 ? 1.F : 6.F;
        expected.push_back(i % 2 ? -factor : factor);
    }
    expected[0] = 1; expected[1] = 12;
    check(rows == expected);
    check(decode_rows(m, 1, 1, 128) == std::vector<float>(expected.begin() + 32, expected.end()));
    check(decode_rows(m, 2, 0, 0).empty());
    fails<std::length_error>([&] { decode_rows(m, 0, 2, 255); }, "output_budget");
    fails([&] { decode_rows(m, 1, 2, 256); }, "row_range");
    auto bad = m; bad.columns = 31;
    fails([&] { decode_rows(bad, 0, 1, 256); }, "nvfp4_block_alignment");
    bad = m; bad.weights = std::span(packed).first(31);
    fails([&] { decode_rows(bad, 0, 1, 256); }, "weight_shape");
    bad = m; bad.block_scales = std::span(scales).first(3);
    fails([&] { decode_rows(bad, 0, 1, 256); }, "block_scale_shape");
    const std::array<float, 2> row_scales{1, 1}; bad = m; bad.multipliers = row_scales;
    fails([&] { decode_rows(bad, 0, 1, 256); }, "nvfp4_global_scale_shape");
    for (auto byte : {0x7f, 0xff, 0x80, 0xb8}) {
        auto invalid = scales; invalid[3] = byte; bad = m; bad.block_scales = invalid;
        fails([&] { decode_rows(bad, 0, 1, 256); }, "invalid_block_scale");
    }
    auto zero_scales = scales; zero_scales[1] = 0; bad = m; bad.block_scales = zero_scales;
    check(decode_rows(bad, 0, 1, 128)[16] == 0);
    for (float invalid : {0.F, -1.F, std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN()}) {
        const std::array<float, 1> g{invalid}; bad = m; bad.multipliers = g;
        fails([&] { decode_rows(bad, 0, 1, 256); }, "invalid_multiplier");
    }
    const std::array<float, 1> huge{std::numeric_limits<float>::max()}; bad = m; bad.multipliers = huge;
    fails<std::overflow_error>([&] { decode_rows(bad, 0, 1, 256); }, "nonfinite_result");
    bad = m; bad.rows = std::numeric_limits<std::size_t>::max();
    fails([&] { decode_rows(bad, 0, 1, 256); }, "shape_overflow");
    bad = m; bad.encoding = static_cast<Encoding>(255);
    fails([&] { decode_rows(bad, 0, 1, 256); }, "unsupported_encoding");
}
void fp8() {
    const std::array<std::uint8_t, 6> weights{0x38, 0xc0, 0x01, 0x7e, 0x80, 0x30};
    const std::array<float, 2> scales{2, .5F};
    Matrix m{Encoding::modelopt_fp8, 2, 3, weights, {}, scales};
    check(decode_rows(m, 0, 2, 24) == std::vector<float>({2, -4, .00390625F, 224, -0.F, .25F}));

}
} // namespace
int main() {
    try { bounded_global(); primitives(); packed_byte_order(); nvfp4(); fp8(); }
    catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
