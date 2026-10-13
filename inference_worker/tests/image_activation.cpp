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

void frequency_rounding_and_order() {
    using kadan::image::detail::timestep_frequency;
    for (float timestep :
         {0.f, -0.f, std::numeric_limits<float>::denorm_min(), .001f, .1f, .625f, 1.f})
        for (std::size_t column = 0; column < 128; ++column) {
            const float angle = timestep * 1000.f
                                * std::exp(-std::log(10000.f) * float(column) / 128.f);
            check(
                std::bit_cast<std::uint32_t>(timestep_frequency(timestep, column))
                == std::bit_cast<std::uint32_t>(std::cos(angle))
            );
            check(
                std::bit_cast<std::uint32_t>(timestep_frequency(timestep, 128 + column))
                == std::bit_cast<std::uint32_t>(std::sin(angle))
            );
            check(timestep_frequency(0.f, column) == 1.f);
            check(
                std::bit_cast<std::uint32_t>(timestep_frequency(0.f, 128 + column)) == 0
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

void vae_normalization_rounding() {
    using kadan::image::detail::vae_norm_scale;
    using kadan::image::detail::vae_norm_value;
    for (std::size_t channels : {1u, 3u, 144u, 1152u})
        for (double sum : {0., 1e-30, 1e-24, 1e-12, 1., 3., 1e78}) {
            const float previous =
                float(std::sqrt(double(channels)) / std::max(std::sqrt(sum), 1e-12));
            check(
                std::bit_cast<std::uint32_t>(vae_norm_scale(sum, channels))
                == std::bit_cast<std::uint32_t>(previous)
            );
        }
    for (float x :
         {0.f,
          -0.f,
          std::numeric_limits<float>::denorm_min(),
          -100.f,
          -.5f,
          1.00001f,
          12345.f})
        for (float scale : {.12345f, 1.f, 123.f})
            for (float gamma : {-3.14159f, 0.f, 1.f})
                for (bool activate : {false, true}) {
                    const float value = x * scale * gamma;
                    const double v = value;
                    const float previous =
                        activate ? float(
                                       v >= 0 ? v / (1 + std::exp(-v))
                                              : v * std::exp(v) / (1 + std::exp(v))
                                   )
                                 : value;
                    check(
                        std::bit_cast<std::uint32_t>(
                            vae_norm_value(x, scale, gamma, activate)
                        )
                        == std::bit_cast<std::uint32_t>(previous)
                    );
                }
}

void product_rounding() {
    using kadan::image::detail::silu_product;
    check(std::bit_cast<std::uint32_t>(silu_product(-0.f, 1.f)) == 0x80000000);
    check(std::bit_cast<std::uint32_t>(silu_product(-0.f, -1.f)) == 0);
    check(std::bit_cast<std::uint32_t>(silu_product(-100.f, 1.f)) == 0x80000a5f);
    check(std::isinf(silu_product(std::numeric_limits<float>::max(), 2.f)));
    check(
        silu_product(
            -std::numeric_limits<float>::max(), std::numeric_limits<float>::max()
        )
        == 0.f
    );
    bool distinct_rounding = false;
    for (float gate : {-.5f, -1.f, .5f, 1.f, 2.f, 3.f})
        for (float up : {1.00001f, 3.14159f, 1234567.f, -87654.f}) {
            const double v = gate;
            const double exponential = std::exp(v >= 0 ? -v : v);
            const double activated =
                v >= 0 ? v / (1 + exponential) : v * exponential / (1 + exponential);
            const float rounded = float(activated);
            const float expected = rounded * up;
            check(
                std::bit_cast<std::uint32_t>(silu_product(gate, up))
                == std::bit_cast<std::uint32_t>(expected)
            );
            distinct_rounding |= std::bit_cast<std::uint32_t>(expected)
                                 != std::bit_cast<std::uint32_t>(float(activated * up));
        }
    check(distinct_rounding);
}
}  // namespace

int main() {
    values_and_signed_zero();
    preserves_previous_rounding();
    nonfinite_values();
    frequency_rounding_and_order();
    product_rounding();
    vae_normalization_rounding();
    std::cout << "image SiLU scalar arithmetic passed (no inference)\n";
}
