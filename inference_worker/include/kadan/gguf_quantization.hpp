#pragma once

#include <bit>
#include <cstddef>
#include <cstdint>
#include <optional>

#if defined(__CUDACC__)
#define KADAN_GGUF_HD __host__ __device__
#else
#define KADAN_GGUF_HD
#endif

namespace kadan::gguf {

enum class Encoding {
    q3_k,
    q4_k,
    q8_0,
    q6_k,
    iq3_xxs,
    iq4_nl,
    iq4_xs,
    bf16
};

// GGML type IDs of the packed encodings a projection can decode on device.
inline std::optional<Encoding> encoding(std::uint32_t ggml_type) {
    switch (ggml_type) {
    case 8:
        return Encoding::q8_0;
    case 11:
        return Encoding::q3_k;
    case 12:
        return Encoding::q4_k;
    case 14:
        return Encoding::q6_k;
    case 18:
        return Encoding::iq3_xxs;
    case 20:
        return Encoding::iq4_nl;
    case 23:
        return Encoding::iq4_xs;
    case 30:
        return Encoding::bf16;
    default:
        return std::nullopt;
    }
}

KADAN_GGUF_HD constexpr bool supported(Encoding encoding) {
    return encoding == Encoding::q3_k || encoding == Encoding::q4_k
           || encoding == Encoding::q8_0 || encoding == Encoding::q6_k
           || encoding == Encoding::iq3_xxs || encoding == Encoding::iq4_nl
           || encoding == Encoding::iq4_xs || encoding == Encoding::bf16;
}

KADAN_GGUF_HD constexpr std::size_t block_values(Encoding encoding) {
    return encoding == Encoding::bf16                                   ? 1
           : encoding == Encoding::q8_0 || encoding == Encoding::iq4_nl ? 32
                                                                        : 256;
}

KADAN_GGUF_HD constexpr std::size_t block_bytes(Encoding encoding) {
    switch (encoding) {
    case Encoding::q3_k:
        return 110;
    case Encoding::q4_k:
        return 144;
    case Encoding::q8_0:
        return 34;
    case Encoding::q6_k:
        return 210;
    case Encoding::iq3_xxs:
        return 98;
    case Encoding::iq4_nl:
        return 18;
    case Encoding::iq4_xs:
        return 136;
    case Encoding::bf16:
        return 2;
    }
    return 0;
}

namespace detail {
// Lookup tables from ggml-common.h (MIT). Device code reads its own copies.
#define KADAN_GGUF_IQ3XXS_GRID                                                         \
    {                                                                                  \
        0x04040404, 0x04040414, 0x04040424, 0x04040c0c, 0x04040c1c, 0x04040c3e,        \
        0x04041404, 0x04041414, 0x04041c0c, 0x04042414, 0x04043e1c, 0x04043e2c,        \
        0x040c040c, 0x040c041c, 0x040c0c04, 0x040c0c14, 0x040c140c, 0x040c142c,        \
        0x040c1c04, 0x040c1c14, 0x040c240c, 0x040c2c24, 0x040c3e04, 0x04140404,        \
        0x04140414, 0x04140424, 0x04140c0c, 0x04141404, 0x04141414, 0x04141c0c,        \
        0x04141c1c, 0x04141c3e, 0x04142c0c, 0x04142c3e, 0x04143e2c, 0x041c040c,        \
        0x041c043e, 0x041c0c04, 0x041c0c14, 0x041c142c, 0x041c3e04, 0x04240c1c,        \
        0x04241c3e, 0x04242424, 0x04242c3e, 0x04243e1c, 0x04243e2c, 0x042c040c,        \
        0x042c043e, 0x042c1c14, 0x042c2c14, 0x04341c2c, 0x04343424, 0x043e0c04,        \
        0x043e0c24, 0x043e0c34, 0x043e241c, 0x043e340c, 0x0c04040c, 0x0c04041c,        \
        0x0c040c04, 0x0c040c14, 0x0c04140c, 0x0c04141c, 0x0c041c04, 0x0c041c14,        \
        0x0c041c24, 0x0c04243e, 0x0c042c04, 0x0c0c0404, 0x0c0c0414, 0x0c0c0c0c,        \
        0x0c0c1404, 0x0c0c1414, 0x0c14040c, 0x0c14041c, 0x0c140c04, 0x0c140c14,        \
        0x0c14140c, 0x0c141c04, 0x0c143e14, 0x0c1c0404, 0x0c1c0414, 0x0c1c1404,        \
        0x0c1c1c0c, 0x0c1c2434, 0x0c1c3434, 0x0c24040c, 0x0c24042c, 0x0c242c04,        \
        0x0c2c1404, 0x0c2c1424, 0x0c2c2434, 0x0c2c3e0c, 0x0c34042c, 0x0c3e1414,        \
        0x0c3e2404, 0x14040404, 0x14040414, 0x14040c0c, 0x14040c1c, 0x14041404,        \
        0x14041414, 0x14041434, 0x14041c0c, 0x14042414, 0x140c040c, 0x140c041c,        \
        0x140c042c, 0x140c0c04, 0x140c0c14, 0x140c140c, 0x140c1c04, 0x140c341c,        \
        0x140c343e, 0x140c3e04, 0x14140404, 0x14140414, 0x14140c0c, 0x14140c3e,        \
        0x14141404, 0x14141414, 0x14141c3e, 0x14142404, 0x14142c2c, 0x141c040c,        \
        0x141c0c04, 0x141c0c24, 0x141c3e04, 0x141c3e24, 0x14241c2c, 0x14242c1c,        \
        0x142c041c, 0x142c143e, 0x142c240c, 0x142c3e24, 0x143e040c, 0x143e041c,        \
        0x143e0c34, 0x143e242c, 0x1c04040c, 0x1c040c04, 0x1c040c14, 0x1c04140c,        \
        0x1c04141c, 0x1c042c04, 0x1c04342c, 0x1c043e14, 0x1c0c0404, 0x1c0c0414,        \
        0x1c0c1404, 0x1c0c1c0c, 0x1c0c2424, 0x1c0c2434, 0x1c14040c, 0x1c14041c,        \
        0x1c140c04, 0x1c14142c, 0x1c142c14, 0x1c143e14, 0x1c1c0c0c, 0x1c1c1c1c,        \
        0x1c241c04, 0x1c24243e, 0x1c243e14, 0x1c2c0404, 0x1c2c0434, 0x1c2c1414,        \
        0x1c2c2c2c, 0x1c340c24, 0x1c341c34, 0x1c34341c, 0x1c3e1c1c, 0x1c3e3404,        \
        0x24040424, 0x24040c3e, 0x24041c2c, 0x24041c3e, 0x24042c1c, 0x24042c3e,        \
        0x240c3e24, 0x24141404, 0x24141c3e, 0x24142404, 0x24143404, 0x24143434,        \
        0x241c043e, 0x241c242c, 0x24240424, 0x24242c0c, 0x24243424, 0x242c142c,        \
        0x242c241c, 0x242c3e04, 0x243e042c, 0x243e0c04, 0x243e0c14, 0x243e1c04,        \
        0x2c040c14, 0x2c04240c, 0x2c043e04, 0x2c0c0404, 0x2c0c0434, 0x2c0c1434,        \
        0x2c0c2c2c, 0x2c140c24, 0x2c141c14, 0x2c143e14, 0x2c1c0414, 0x2c1c2c1c,        \
        0x2c240c04, 0x2c24141c, 0x2c24143e, 0x2c243e14, 0x2c2c0414, 0x2c2c1c0c,        \
        0x2c342c04, 0x2c3e1424, 0x2c3e2414, 0x34041424, 0x34042424, 0x34042434,        \
        0x34043424, 0x340c140c, 0x340c340c, 0x34140c3e, 0x34143424, 0x341c1c04,        \
        0x341c1c34, 0x34242424, 0x342c042c, 0x342c2c14, 0x34341c1c, 0x343e041c,        \
        0x343e140c, 0x3e04041c, 0x3e04042c, 0x3e04043e, 0x3e040c04, 0x3e041c14,        \
        0x3e042c14, 0x3e0c1434, 0x3e0c2404, 0x3e140c14, 0x3e14242c, 0x3e142c14,        \
        0x3e1c0404, 0x3e1c0c2c, 0x3e1c1c1c, 0x3e1c3404, 0x3e24140c, 0x3e24240c,        \
        0x3e2c0404, 0x3e2c0414, 0x3e2c1424, 0x3e341c04                                 \
    }
#define KADAN_GGUF_IQ4NL_VALUES                                                        \
    {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113}

#if defined(__CUDACC__)
static __device__ const std::uint32_t iq3xxs_grid_device[256] = KADAN_GGUF_IQ3XXS_GRID;
static __device__ const signed char iq4nl_values_device[16] = KADAN_GGUF_IQ4NL_VALUES;
#endif
inline constexpr std::uint32_t iq3xxs_grid_host[256] = KADAN_GGUF_IQ3XXS_GRID;
inline constexpr signed char iq4nl_values_host[16] = KADAN_GGUF_IQ4NL_VALUES;
#undef KADAN_GGUF_IQ3XXS_GRID
#undef KADAN_GGUF_IQ4NL_VALUES

KADAN_GGUF_HD inline std::uint32_t iq3xxs_grid(unsigned index) {
#if defined(__CUDA_ARCH__)
    return iq3xxs_grid_device[index];
#else
    return iq3xxs_grid_host[index];
#endif
}

KADAN_GGUF_HD inline float iq4nl_value(unsigned index) {
#if defined(__CUDA_ARCH__)
    return float(iq4nl_values_device[index]);
#else
    return float(iq4nl_values_host[index]);
#endif
}

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

KADAN_GGUF_HD inline float add(float left, float right) {
#if defined(__CUDA_ARCH__)
    return __fadd_rn(left, right);
#else
    volatile float result = left + right;
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

KADAN_GGUF_HD inline std::uint32_t word(const unsigned char* bytes) {
    return std::uint32_t(bytes[0]) | (std::uint32_t(bytes[1]) << 8)
           | (std::uint32_t(bytes[2]) << 16) | (std::uint32_t(bytes[3]) << 24);
}
}  // namespace detail

// The caller validates the encoding, whole-block layout and element range.
// Byte loads support unaligned packed staging without aliasing packed structs.
KADAN_GGUF_HD inline float
decode(Encoding encoding, const unsigned char* block, std::size_t element) {
    if (encoding == Encoding::bf16)
        return detail::float_bits(
            (std::uint32_t(block[0]) | (std::uint32_t(block[1]) << 8)) << 16
        );
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
    if (encoding == Encoding::q6_k) {
        // ql[128], qh[64], int8 scales[16], half d; two 128-value halves.
        const unsigned half_index = unsigned(element / 128), offset = element % 128;
        const unsigned quarter = offset / 32, lane = offset % 32;
        const unsigned packed = block[half_index * 64 + lane + (quarter & 1u) * 32];
        const unsigned low = quarter < 2 ? packed & 15u : packed >> 4;
        const unsigned high =
            (block[128 + half_index * 32 + lane] >> (2 * quarter)) & 3u;
        const int code = int(low | (high << 4)) - 32;
        const unsigned scale_byte =
            block[192 + half_index * 8 + lane / 16 + 2 * quarter];
        const int scale = scale_byte < 128 ? int(scale_byte) : int(scale_byte) - 256;
        return detail::multiply(
            detail::multiply(detail::half(block + 208), float(scale)), float(code)
        );
    }
    if (encoding == Encoding::iq3_xxs) {
        // half d, 64 grid indices, then eight 32-bit scale/sign words.
        const unsigned group = unsigned(element / 32), within = element % 32;
        const unsigned pair = within / 8, lane = within % 8;
        const auto aux = detail::word(block + 2 + 64 + 4 * group);
        const float scale = detail::multiply(
            detail::multiply(detail::half(block), detail::add(.5f, float(aux >> 28))),
            .5f
        );
        // ksigns_iq2xs stores seven explicit signs plus their odd parity bit.
        unsigned signs = (aux >> (7 * pair)) & 127u, parity = signs;
        parity ^= parity >> 4;
        parity ^= parity >> 2;
        parity ^= parity >> 1;
        signs |= (parity & 1u) << 7;
        const auto grid =
            detail::iq3xxs_grid(block[2 + group * 8 + 2 * pair + lane / 4]);
        const float magnitude =
            detail::multiply(scale, float((grid >> (8 * (lane % 4))) & 255u));
        return detail::multiply(magnitude, (signs >> lane) & 1u ? -1.f : 1.f);
    }
    if (encoding == Encoding::iq4_nl) {
        const unsigned packed = block[2 + element % 16];
        return detail::multiply(
            detail::half(block),
            detail::iq4nl_value(element < 16 ? packed & 15u : packed >> 4)
        );
    }
    if (encoding == Encoding::iq4_xs) {
        // half d, 16-bit high scales, four low scale bytes, 128 code bytes.
        const unsigned group = unsigned(element / 32), within = element % 32;
        const unsigned high_scales = unsigned(block[2]) | (unsigned(block[3]) << 8);
        const int scale = int(((block[4 + group / 2] >> (4 * (group % 2))) & 15u)
                              | (((high_scales >> (2 * group)) & 3u) << 4))
                          - 32;
        const unsigned packed = block[8 + group * 16 + within % 16];
        return detail::multiply(
            detail::multiply(detail::half(block), float(scale)),
            detail::iq4nl_value(within < 16 ? packed & 15u : packed >> 4)
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
