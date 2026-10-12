// Scalar arithmetic only: no model, forward pass, checkpoint or device.
#include "../src/image_activation.hpp"

#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {
void check(bool ok) {
    if (!ok)
        throw std::runtime_error("image_activation_test_failed");
}

void values_and_signed_zero() {
    struct Example {
        float input;
        std::uint32_t output;
    };

    // Rounded float32 results, including underflow and the negative-tail subnormal.
    const std::array examples{
        Example{-100.f, 0x80000a5f},
        Example{-20.f, 0xb3310d3f},
        Example{-1.f, 0xbe89b2b1},
        Example{-.5f, 0xbe414d03},
        Example{.5f, 0x3e9f597f},
        Example{1.f, 0x3f3b26a8},
        Example{20.f, 0x41a00000},
        Example{0.f, 0},
        Example{-0.f, 0x80000000},
        Example{std::numeric_limits<float>::max(), 0x7f7fffff},
        Example{-std::numeric_limits<float>::max(), 0x80000000},
        Example{std::numeric_limits<float>::min(), 0x00400000},
        Example{-std::numeric_limits<float>::min(), 0x80400000},
        Example{std::numeric_limits<float>::denorm_min(), 0},
        Example{-std::numeric_limits<float>::denorm_min(), 0x80000000},
    };
    for (const auto& example : examples)
        check(
            std::bit_cast<std::uint32_t>(kadan::image::detail::silu(example.input))
            == example.output
        );
}

void preserves_previous_rounding() {
    // Cover both signs, every finite exponent, and several mantissas against the
    // previous expression. This catches altered precision or operation ordering.
    for (std::uint32_t sign : {0u, 0x80000000u})
        for (std::uint32_t exponent = 0; exponent < 255; ++exponent)
            for (std::uint32_t mantissa : {0u, 1u, 0x123456u, 0x7fffffu}) {
                const float x =
                    std::bit_cast<float>(sign | (exponent << 23) | mantissa);
                const double v = x;
                const float previous = float(
                    v >= 0 ? v / (1 + std::exp(-v))
                           : v * std::exp(v) / (1 + std::exp(v))
                );
                check(
                    std::bit_cast<std::uint32_t>(kadan::image::detail::silu(x))
                    == std::bit_cast<std::uint32_t>(previous)
                );
            }
}

void nonfinite_values() {
    // The primitive preserves IEEE results; the existing projection boundary
    // rejects nonfinite activations before this function is used by the block.
    const float infinity = std::numeric_limits<float>::infinity();
    check(kadan::image::detail::silu(infinity) == infinity);
    check(std::isnan(kadan::image::detail::silu(-infinity)));
    check(
        std::isnan(kadan::image::detail::silu(std::numeric_limits<float>::quiet_NaN()))
    );
}
}  // namespace

int main() {
    values_and_signed_zero();
    preserves_previous_rounding();
    nonfinite_values();
    std::cout << "image SiLU scalar arithmetic passed (no inference)\n";
}
