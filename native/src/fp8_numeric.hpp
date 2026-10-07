#pragma once
#include <bit>
#include <cstdint>

namespace kadan::cuda::detail {
// Precondition: byte is finite E4M3FN (not 0x7f/0xff), checked at admission.
// Shared host/device decoder permits exhaustive CPU tests of the kernel path.
#ifdef __CUDACC__
__host__ __device__
#endif
inline float finite_e4m3fn(std::uint8_t byte) {
    const unsigned exponent = (byte >> 3) & 15;
    const unsigned fraction = byte & 7;
    const unsigned bits = ((exponent + 120U) << 23) | (fraction << 20);
#ifdef __CUDA_ARCH__
    const float normal = __uint_as_float(bits);
#else
    const float normal = std::bit_cast<float>(bits);
#endif
    const float magnitude = exponent == 0 ? float(fraction) * 0.001953125F : normal;
    return byte & 128 ? -magnitude : magnitude;
}
}
