#pragma once

#include <bit>
#include <cstddef>
#include <cstdint>

#if defined(__CUDACC__)
#define KADAN_GGUF_HD __host__ __device__
#else
#define KADAN_GGUF_HD
#endif

namespace kadan::gguf {

enum class Encoding {
    q3_k,
    q4_k,
    q8_0
};

KADAN_GGUF_HD constexpr std::size_t block_values(Encoding encoding) {
    return encoding == Encoding::q8_0 ? 32 : 256;
}

KADAN_GGUF_HD constexpr std::size_t block_bytes(Encoding encoding) {
    return encoding == Encoding::q3_k ? 110 : encoding == Encoding::q4_k ? 144 : 34;
}

namespace detail {
KADAN_GGUF_HD inline float float_bits(std::uint32_t bits) {
#if defined(__CUDA_ARCH__)
    return __uint_as_float(bits);
#else
    return std::bit_cast<float>(bits);
#endif
}

KADAN_GGUF_HD inline float half(const unsigned char* bytes) {
    const unsigned bits = unsigned(bytes[0]) | (unsigned(bytes[1]) << 8);
    const unsigned sign = (bits & 0x8000u) << 16;
    unsigned exponent = (bits >> 10) & 31u;
    unsigned fraction = bits & 1023u;
    if (exponent == 31)
        return float_bits(sign | 0x7f800000u | (fraction << 13));
    if (exponent != 0)
        return float_bits(sign | ((exponent + 112u) << 23) | (fraction << 13));
    if (fraction == 0)
        return float_bits(sign);
    unsigned shift = 0;
    while ((fraction & 1024u) == 0) {
        fraction <<= 1;
        ++shift;
    }
    return float_bits(sign | ((113u - shift) << 23) | ((fraction & 1023u) << 13));
}

// Match the reference's separately rounded products, including Q4's final
// subtraction. CUDA must not contract these operations into an FMA.
KADAN_GGUF_HD inline float multiply(float left, float right) {
#if defined(__CUDA_ARCH__)
    return __fmul_rn(left, right);
#else
    volatile float result = left * right;
    return result;
#endif
}

KADAN_GGUF_HD inline float subtract(float left, float right) {
#if defined(__CUDA_ARCH__)
    return __fsub_rn(left, right);
#else
    return left - right;
#endif
}
}  // namespace detail

// The caller validates the encoding, whole-block layout and element range.
// Byte loads support unaligned packed staging without aliasing packed structs.
KADAN_GGUF_HD inline float
decode(Encoding encoding, const unsigned char* block, std::size_t element) {
    if (encoding == Encoding::q8_0) {
        const unsigned byte = block[2 + element];
        const int code = byte < 128 ? int(byte) : int(byte) - 256;
        return detail::multiply(detail::half(block), float(code));
    }
    if (encoding == Encoding::q3_k) {
        const unsigned group = unsigned(element / 16);
        const auto* scales = block + 96;
        const unsigned low = group < 8 ? scales[group] & 15u : scales[group - 8] >> 4;
        const unsigned high = (scales[8 + group % 4] >> (2 * (group / 4))) & 3u;
        const int scale = int(low | (high << 4)) - 32;
        const unsigned lane = unsigned(element % 32);
        const unsigned shift = 2 * unsigned((element % 128) / 32);
        const unsigned low_code =
            (block[32 + (element / 128) * 32 + lane] >> shift) & 3u;
        const unsigned high_mask = 1u << unsigned(element / 32);
        const int code = int(low_code) - ((block[lane] & high_mask) ? 0 : 4);
        return detail::multiply(
            detail::multiply(detail::half(block + 108), float(scale)), float(code)
        );
    }
    const unsigned group = unsigned(element / 32);
    const auto* scales = block + 4;
    const unsigned scale =
        group < 4 ? scales[group] & 63u
                  : (scales[group + 4] & 15u) | ((scales[group - 4] >> 6) << 4);
    const unsigned minimum =
        group < 4 ? scales[group + 4] & 63u
                  : (scales[group + 4] >> 4) | ((scales[group] >> 6) << 4);
    const unsigned packed = block[16 + (element / 64) * 32 + element % 32];
    const unsigned code = (group & 1u) ? packed >> 4 : packed & 15u;
    const float scaled = detail::multiply(detail::half(block), float(scale));
    const float offset = detail::multiply(detail::half(block + 2), float(minimum));
    return detail::subtract(detail::multiply(scaled, float(code)), offset);
}

}  // namespace kadan::gguf

#undef KADAN_GGUF_HD
