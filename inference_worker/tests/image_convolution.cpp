// Indexing, byte-preserving copies, plans and an unsupported boundary only.
// No weights, model, convolution, projection, decoding or GPU execution.
#include <array>
#include <bit>
#include <iostream>
#include <source_location>
#include <vector>

#include "kadan/dense_compute.hpp"
#include "kadan/device_placement.hpp"
using namespace kadan;
using namespace kadan::image;

namespace {
void check(bool ok, std::source_location at = std::source_location::current()) {
    if (!ok)
        throw std::runtime_error(
            "image_convolution_layout_test:" + std::to_string(at.line())
        );
}

template <class F> void fails(F fn) {
    try {
        fn();
    } catch (const std::invalid_argument&) {
        return;
    }
    throw std::runtime_error("expected_convolution_shape_rejection");
}

void patch_layout_and_halo() {
    const std::array bits{0u, 0x80000000u, 1u, 0x80000001u, 0x3f800001u, 0xff7fffffu};
    for (std::size_t height : {1u, 2u, 3u, 17u})
        for (std::size_t width : {1u, 2u, 3u, 17u, 257u})
            for (std::size_t kernel : {1u, 3u}) {
                const VaeConvolution shape{2, height, width, 3, kernel};
                std::vector<float> input(shape.channels * shape.points());
                for (std::size_t i = 0; i < input.size(); ++i)
                    input[i] = std::bit_cast<float>(bits[i % bits.size()]);
                for (std::size_t first :
                     {std::size_t(0),
                      std::size_t(1),
                      width - 1,
                      width,
                      shape.points() - 1}) {
                    if (first >= shape.points())
                        continue;
                    for (std::size_t requested : {1u, 2u, 255u, 256u}) {
                        const auto count = std::min(requested, shape.points() - first);
                        const auto region = vae_patch_region(shape, first, count);
                        check(
                            region.top + region.height <= height
                            && region.left + region.width <= width
                        );
                        std::vector<float> raw(shape.channels * region.points());
                        // Emulate the 3D copy's distinct row/slice pitches.
                        for (std::size_t c = 0; c < shape.channels; ++c)
                            for (std::size_t y = 0; y < region.height; ++y)
                                for (std::size_t x = 0; x < region.width; ++x)
                                    raw[c * region.points() + y * region.width
                                        + x] = input
                                        [c * shape.points() + (region.top + y) * width
                                         + region.left + x];
                        for (std::size_t row = 0; row < count; ++row)
                            for (std::size_t c = 0; c < shape.channels; ++c)
                                for (std::size_t ky = 0; ky < kernel; ++ky)
                                    for (std::size_t kx = 0; kx < kernel; ++kx) {
                                        const auto col =
                                            (c * kernel + ky) * kernel + kx;
                                        const auto py =
                                            std::ptrdiff_t((first + row) / width + ky)
                                            - std::ptrdiff_t(kernel / 2);
                                        const auto px =
                                            std::ptrdiff_t((first + row) % width + kx)
                                            - std::ptrdiff_t(kernel / 2);
                                        const bool valid =
                                            py >= 0 && px >= 0
                                            && py < std::ptrdiff_t(height)
                                            && px < std::ptrdiff_t(width);
                                        const auto index =
                                            vae_patch_index(shape, first + row, col);
                                        check((index != vae_patch_padding) == valid);
                                        float actual = 0, expected = 0;
                                        if (valid) {
                                            check(
                                                index
                                                == c * shape.points()
                                                       + std::size_t(py) * width
                                                       + std::size_t(px)
                                            );
                                            check(
                                                std::size_t(py) >= region.top
                                                && std::size_t(py)
                                                       < region.top + region.height
                                                && std::size_t(px) >= region.left
                                                && std::size_t(px)
                                                       < region.left + region.width
                                            );
                                            actual =
                                                raw[c * region.points()
                                                    + (std::size_t(py) - region.top)
                                                          * region.width
                                                    + std::size_t(px) - region.left];
                                            expected = input
                                                [c * shape.points()
                                                 + std::size_t(py) * width
                                                 + std::size_t(px)];
                                        }
                                        check(
                                            std::bit_cast<std::uint32_t>(actual)
                                            == std::bit_cast<std::uint32_t>(expected)
                                        );
                                    }
                    }
                }
            }
}

void planar_output_tails() {
    for (std::size_t rows : {1u, 17u, 256u})
        for (std::size_t columns : {1u, 3u, 404u}) {
            const std::size_t points = rows + 7, first = 3, first_column = 2,
                              outputs = columns + 5;
            std::vector<float> matrix(rows * columns), planar(rows * columns),
                output(outputs * points, -99.f);
            for (std::size_t i = 0; i < matrix.size(); ++i)
                matrix[i] = std::bit_cast<float>(std::uint32_t(0x80000000u + i));
            for (std::size_t i = 0; i < matrix.size(); ++i)
                planar[vae_planar_tile_index(i, rows, columns)] = matrix[i];
            for (std::size_t c = 0; c < columns; ++c)
                for (std::size_t row = 0; row < rows; ++row)
                    output[(first_column + c) * points + first + row] =
                        planar[c * rows + row];
            for (std::size_t c = 0; c < outputs; ++c)
                for (std::size_t p = 0; p < points; ++p) {
                    const auto expected =
                        (c >= first_column && c < first_column + columns && p >= first
                         && p < first + rows)
                            ? matrix[(p - first) * columns + c - first_column]
                            : -99.f;
                    check(
                        std::bit_cast<std::uint32_t>(output[c * points + p])
                        == std::bit_cast<std::uint32_t>(expected)
                    );
                }
        }
}

void planning_and_validation() {
    constexpr Bytes limit = 2ULL * 1024 * 1024 * 1024;
    auto aligned = [](Bytes n) {
        return (n + 255) / 256 * 256;
    };
    for (const auto shape :
         {VaeConvolution{64, 128, 128, 1152, 3},
          VaeConvolution{1152, 256, 256, 576, 3},
          VaeConvolution{144, 2048, 2048, 4, 3},
          VaeConvolution{3, 3, 257, 5, 1}})
        for (Bytes budget :
             std::array<Bytes, 3>{16 * 1024 * 1024, 64 * 1024 * 1024, limit}) {
            const auto plan = image_vae_convolution_plan(shape, shape.outputs, budget);
            if (!plan)
                continue;
            const auto& p = plan->projection;
            const auto in = shape.columns();
            check(
                p.rows > 0 && p.rows <= 256 && p.columns > 0
                && p.bytes + plan->patch_bytes <= budget
            );
            check(p.columns * in * 4 <= 16 * 1024 * 1024);
            const auto layout =
                aligned(4) + aligned(p.rows * in * 4) + plan->patch_bytes
                + aligned(p.columns * in * 4) + 2 * aligned(p.columns * 4)
                + aligned(p.rows * in * 4) + aligned(p.rows * in) + aligned(p.rows * 4)
                + 2 * aligned(p.rows * p.columns * 4) + aligned(8 * 1024 * 1024);
            check(layout <= p.bytes + plan->patch_bytes);
            check(image_vae_convolution_plan(
                      shape, shape.outputs, p.bytes + plan->patch_bytes
            )
                      .has_value());
            std::size_t covered = 0;
            for (std::size_t first = 0; first < shape.points(); first += 256) {
                const auto rows = std::min<std::size_t>(256, shape.points() - first);
                for (std::size_t at = 0; at < rows; at += p.rows) {
                    const auto count = std::min(p.rows, rows - at);
                    const auto region = vae_patch_region(shape, first + at, count);
                    check(
                        aligned(shape.channels * region.points() * 4)
                        <= plan->patch_bytes
                    );
                    covered += count;
                }
            }
            check(covered == shape.points());
        }
    check(!image_vae_convolution_plan({1, 1, 1, 1, 5}, 1, limit));
    check(!image_vae_convolution_plan({1, 1, 2147483647ULL / 4 + 1, 1, 1}, 1, limit));
    check(!image_vae_convolution_plan({4194305, 1, 1, 1, 1}, 1, limit));
    check(!image_vae_convolution_plan({1, 1, 1, 1, 1}, 1, 1));
    fails([] { VaeConvolution{0, 1, 1, 1, 1}.validate(); });
    fails([] { VaeConvolution{1, SIZE_MAX, 2, 1, 1}.validate(); });
    fails([] { VaeConvolution{1, 1, 1, 1, SIZE_MAX}.validate(); });
    fails([] { vae_patch_region({1, 1, 1, 1, 3}, 0, 0); });
    fails([] { vae_patch_region({1, 1, 1, 1, 3}, 1, 1); });
    std::array<float, 6> input{}, output{};
    std::array<float, 2> bias{};
    validate_vae_convolution({2, 1, 3, 2, 3}, input, bias, output);
    fails([&] { validate_vae_convolution({2, 1, 3, 2, 3}, input, bias, input); });
    fails([&] {
        validate_vae_convolution(
            {2, 1, 3, 2, 3}, input, std::span(output).first(2), output
        );
    });
    fails([&] { validate_vae_convolution({2, 1, 4, 2, 3}, input, bias, output); });
}

struct Unsupported final : DenseCompute {
    void dense(
        std::span<const float>,
        std::span<const float>,
        std::span<const float>,
        std::size_t,
        std::size_t,
        std::span<float>,
        const std::atomic_bool&,
        bool
    ) override {
        throw std::runtime_error("unexpected_projection");
    }
};

void unsupported_boundary() {
    Unsupported device;
    std::array<float, 1> input{1}, bias{0}, output{17};
    std::atomic_bool cancel = false;
    unsigned reads = 0;
    DenseCompute::FloatWeightSource source = [&](std::size_t, std::span<float>) {
        ++reads;
    };
    check(!device.image_vae_convolution_source(
        {}, source, bias, input, {1, 1, 1, 1, 1}, output, cancel
    ));
    check(reads == 0 && output[0] == 17);
}
}  // namespace

int main() {
    patch_layout_and_halo();
    planar_output_tails();
    planning_and_validation();
    unsupported_boundary();
    std::cout << "image convolution layout units passed (no inference)\n";
}
