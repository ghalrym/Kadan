#include "kadan/gguf_quantization.hpp"

#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <ggml.h>
#include <iostream>
#include <stdexcept>

namespace {
using kadan::gguf::decode;
using kadan::gguf::Encoding;
using Block = std::array<unsigned char, 144>;

void check(bool condition) {
    if (!condition)
        throw std::runtime_error("gguf_quantization_check_failed");
}

void same(float actual, float expected) {
    if (std::isnan(expected))
        check(std::isnan(actual));
    else
        check(
            std::bit_cast<std::uint32_t>(actual)
            == std::bit_cast<std::uint32_t>(expected)
        );
}

void half_bits(Block& block, std::size_t offset, unsigned bits) {
    block[offset] = bits & 255u;
    block[offset + 1] = bits >> 8;
}

void literal_codes() {
    Block block{};
    half_bits(block, 0, 0x3800);  // 0.5
    for (unsigned i = 0; i < 32; ++i) {
        block[2 + i] = i < 16 ? i : 256u - i;
        same(
            decode(Encoding::q8_0, block.data(), i), .5f * (i < 16 ? int(i) : -int(i))
        );
    }

    block.fill(0);
    for (unsigned i = 0; i < 32; ++i)
        block[i] = 0x55;
    for (unsigned i = 32; i < 96; ++i)
        block[i] = 0xe4;  // 0,1,2,3 in consecutive 32-element groups.
    for (unsigned i = 96; i < 104; ++i)
        block[i] = 0x11;
    for (unsigned i = 104; i < 108; ++i)
        block[i] = 0xaa;            // Every six-bit scale is 33, representing +1.
    half_bits(block, 108, 0xc000);  // -2
    for (unsigned i = 0; i < 256; ++i) {
        const int code = int((i / 32) % 4) - ((i / 32) % 2 ? 4 : 0);
        same(decode(Encoding::q3_k, block.data(), i), -2.f * float(code));
    }

    block.fill(0);
    half_bits(block, 0, 0x4000);  // 2
    half_bits(block, 2, 0x3800);  // 0.5
    // Literal packed scales 1..8 and minimum multipliers 8..1.
    const std::array<unsigned char, 12> scales{
        1, 2, 3, 4, 8, 7, 6, 5, 0x45, 0x36, 0x27, 0x18
    };
    for (unsigned i = 0; i < scales.size(); ++i)
        block[4 + i] = scales[i];
    for (unsigned i = 16; i < 144; ++i)
        block[i] = 0xc3;
    for (unsigned i = 0; i < 256; ++i) {
        const unsigned group = i / 32;
        same(
            decode(Encoding::q4_k, block.data(), i),
            2.f * float(group + 1) * (group % 2 ? 12.f : 3.f) - .5f * float(8 - group)
        );
    }
}

void all_half_patterns() {
    Block block{};
    block[2] = 1;
    for (unsigned bits = 0; bits < 65536; ++bits) {
        half_bits(block, 0, bits);
        same(
            decode(Encoding::q8_0, block.data(), 0),
            ggml_fp16_to_fp32(ggml_fp16_t(bits))
        );
    }
}

std::uint32_t random_bits(std::uint32_t& state) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

void reference_blocks(Encoding encoding, ggml_type type) {
    const auto values = kadan::gguf::block_values(encoding);
    check(ggml_blck_size(type) == std::int64_t(values));
    check(ggml_type_size(type) == kadan::gguf::block_bytes(encoding));
    const auto to_float = ggml_get_type_traits(type)->to_float;
    std::uint32_t state = 0x31415926;
    alignas(16) Block block{};
    std::array<float, 256> expected{};
    // Includes both signs, smallest/largest subnormal and normal values,
    // infinities and quiet/signaling NaNs. Every code/scale byte is randomized.
    const std::array<unsigned, 16> halves{
        0,
        0x8000,
        1,
        0x8001,
        0x03ff,
        0x83ff,
        0x0400,
        0x8400,
        0x3c00,
        0xbc00,
        0x7bff,
        0xfbff,
        0x7c00,
        0xfc00,
        0x7e01,
        0x7c01
    };
    for (unsigned trial = 0; trial < 2048; ++trial) {
        for (auto& byte : block)
            byte = random_bits(state) & 255u;
        const std::size_t offset = encoding == Encoding::q3_k ? 108 : 0;
        const unsigned scale =
            trial < halves.size()
                ? halves[trial]
                : (random_bits(state) & 0x83ffu) | ((random_bits(state) % 31) << 10);
        half_bits(block, offset, scale);
        if (encoding == Encoding::q4_k) {
            const unsigned minimum = trial < halves.size()
                                         ? halves[halves.size() - 1 - trial]
                                         : (random_bits(state) & 0x83ffu)
                                               | ((random_bits(state) % 31) << 10);
            half_bits(block, 2, minimum);
        }
        to_float(block.data(), expected.data(), values);
        for (std::size_t i = 0; i < values; ++i)
            same(decode(encoding, block.data(), i), expected[i]);
    }
}
}  // namespace

int main() {
    try {
        literal_codes();
        all_half_patterns();
        reference_blocks(Encoding::q3_k, GGML_TYPE_Q3_K);
        reference_blocks(Encoding::q4_k, GGML_TYPE_Q4_K);
        reference_blocks(Encoding::q8_0, GGML_TYPE_Q8_0);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
