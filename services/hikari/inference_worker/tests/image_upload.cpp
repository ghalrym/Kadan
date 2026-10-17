// PNG transport and scalar pixel conversion only; no model or inference.
#include "kadan/image_upload.hpp"

#include <array>
#include <cmath>
#include <iostream>
#include <png.h>
#include <source_location>

#include "kadan/preparation.hpp"

using namespace kadan::serving;

namespace {
void check(bool ok, std::source_location at = std::source_location::current()) {
    if (!ok)
        throw std::runtime_error("image_upload_test:" + std::to_string(at.line()));
}

template <class F> void rejects(F action, std::string_view message = {}) {
    try {
        action();
    } catch (const RequestError& error) {
        check(error.status == 422);
        check(message.empty() || error.what() == message);
        return;
    }
    check(false);
}

std::string png(unsigned width, unsigned height, std::array<unsigned char, 4> pixel) {
    std::vector<unsigned char> rgba(std::size_t(width) * height * 4);
    for (std::size_t i = 0; i < rgba.size(); ++i)
        rgba[i] = pixel[i % 4];
    png_image image{};
    image.version = PNG_IMAGE_VERSION;
    image.width = width;
    image.height = height;
    image.format = PNG_FORMAT_RGBA;
    png_alloc_size_t length = 0;
    check(
        png_image_write_to_memory(&image, nullptr, &length, 0, rgba.data(), 0, nullptr)
    );
    std::string bytes(length, '\0');
    check(png_image_write_to_memory(
        &image, bytes.data(), &length, 0, rgba.data(), 0, nullptr
    ));
    bytes.resize(length);
    png_image_free(&image);
    return "data:image/png;base64," + base64_encode(bytes);
}

void transport_bounds() {
    auto source = prepare_image_upload(png(64, 32, {255, 0, 127, 255}));
    check(source.width == 64 && source.height == 32);
    rejects([] { prepare_image_upload("https://example.com/source.png"); });
    rejects([] { prepare_image_upload("data:image/jpeg;base64,AAAA"); });
    rejects([] { prepare_image_upload("data:image/png;base64,@@@@"); });
    rejects([] { prepare_image_upload("data:image/png;base64,AAAA"); });
    rejects([] {
        prepare_image_upload(std::string(image_upload_encoded_maximum + 1, 'A'));
    });
    for (const auto& size : std::array{
             std::array{31u, 32u},
             std::array{32u, 33u},
             std::array{3104u, 32u},
             std::array{32u, 3104u},
             std::array{3072u, 1568u}
         })
        rejects([&] { prepare_image_upload(png(size[0], size[1], {0, 0, 0, 255})); });
    check(base64_decode("YWJj", 4) == "abc");
    rejects([] { base64_decode("YWJj", 3); });
    const std::string large(3 * 1024 * 1024 + 4, 'A');
    rejects([&] { base64_decode(large); });
    check(base64_decode(large, large.size()).size() == large.size() / 4 * 3);
}

void transparent_source_pixels() {
    auto source = prepare_image_upload(png(32, 32, {255, 0, 127, 64}));
    std::vector<std::uint8_t> scratch(32 * 32 * 4);
    std::vector<float> planar(scratch.size());
    decode_image_upload(source, scratch, planar);
    check(planar[0] == 1.f);
    check(planar[1024] == -1.f);
    check(std::abs(planar[2048] - (254.f / 255.f - 1.f)) < 1e-6f);
    check(std::abs(planar[3072] - (128.f / 255.f - 1.f)) < 1e-6f);
}

void mask_pixels() {
    std::vector<std::uint8_t> scratch(32 * 32 * 4);
    std::vector<float> mask(32 * 32);
    for (const auto& [pixel, expected] : std::array{
             std::pair{std::array<unsigned char, 4>{0, 0, 0, 255}, 0.f},
             std::pair{std::array<unsigned char, 4>{255, 255, 255, 255}, 1.f},
             std::pair{std::array<unsigned char, 4>{255, 0, 0, 255}, 54.f / 255.f}
         }) {
        auto source = prepare_image_upload(png(32, 32, pixel));
        decode_image_mask(source, scratch, mask);
        check(std::abs(mask[0] - expected) < 1e-6f);
    }
    auto transparent = prepare_image_upload(png(32, 32, {255, 255, 255, 254}));
    rejects(
        [&] { decode_image_mask(transparent, scratch, mask); },
        "Image mask must be opaque"
    );
}

void corrupt_pixels() {
    auto source = prepare_image_upload(png(32, 32, {255, 255, 255, 255}));
    source.png.resize(source.png.size() / 2);
    std::vector<std::uint8_t> scratch(32 * 32 * 4);
    std::vector<float> planar(scratch.size());
    rejects([&] { decode_image_upload(source, scratch, planar); });
}
}  // namespace

int main() {
    transport_bounds();
    transparent_source_pixels();
    mask_pixels();
    corrupt_pixels();
    std::cout << "image upload units passed\n";
}
