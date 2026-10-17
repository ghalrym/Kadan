#include "kadan/image_upload.hpp"

#include <cmath>
#include <png.h>

#include "kadan/preparation.hpp"

namespace kadan::serving {
namespace {
struct PngReader {
    png_image image{};

    explicit PngReader(std::string_view bytes) {
        image.version = PNG_IMAGE_VERSION;
        if (!png_image_begin_read_from_memory(&image, bytes.data(), bytes.size())) {
            png_image_free(&image);
            throw RequestError("Invalid PNG image", 422);
        }
    }

    ~PngReader() {
        png_image_free(&image);
    }
};

void decode_rgba(const ImageUpload& input, std::span<std::uint8_t> rgba) {
    PngReader reader(input.png);
    auto& image = reader.image;
    preparation_check(
        image.width == input.width && image.height == input.height,
        "PNG dimensions changed",
        422
    );
    preparation_check(
        rgba.size() == std::size_t(input.width) * input.height * 4,
        "Invalid PNG decode buffer"
    );
    image.format = PNG_FORMAT_RGBA;
    preparation_check(
        png_image_finish_read(&image, nullptr, rgba.data(), 0, nullptr) != 0,
        "Invalid PNG pixels",
        422
    );
}
}  // namespace

ImageUpload prepare_image_upload(std::string_view data_url) {
    constexpr std::string_view prefix = "data:image/png;base64,";
    preparation_check(
        data_url.size() <= image_upload_encoded_maximum && data_url.starts_with(prefix),
        "Expected a PNG data URL up to 20 MiB",
        422
    );
    ImageUpload input;
    input.png = base64_decode(
        data_url.substr(prefix.size()), image_upload_encoded_maximum - prefix.size()
    );
    preparation_check(
        !input.png.empty() && input.png.size() <= image_upload_maximum,
        "PNG image exceeds 20 MiB",
        422
    );
    PngReader reader(input.png);
    input.width = reader.image.width;
    input.height = reader.image.height;
    preparation_check(
        input.width >= 32 && input.height >= 32 && input.width <= 3072
            && input.height <= 3072 && input.width % 32 == 0 && input.height % 32 == 0
            && std::size_t(input.width) * input.height <= 4608 * 1024,
        "PNG dimensions must be multiples of 32 between 32 and 3072, up to 4718592 "
        "pixels",
        422
    );
    return input;
}

void decode_image_upload(
    const ImageUpload& input, std::span<std::uint8_t> rgba, std::span<float> planar
) {
    const auto pixels = std::size_t(input.width) * input.height;
    preparation_check(planar.size() == pixels * 4, "Invalid image pixel buffer");
    decode_rgba(input, rgba);
    for (std::size_t i = 0; i < pixels; ++i)
        for (std::size_t channel = 0; channel < 4; ++channel)
            planar[channel * pixels + i] = 2.f * rgba[i * 4 + channel] / 255.f - 1.f;
}

void decode_image_mask(
    const ImageUpload& input, std::span<std::uint8_t> rgba, std::span<float> mask
) {
    const auto pixels = std::size_t(input.width) * input.height;
    preparation_check(mask.size() == pixels, "Invalid image mask buffer");
    decode_rgba(input, rgba);
    for (std::size_t i = 0; i < pixels; ++i) {
        preparation_check(rgba[i * 4 + 3] == 255, "Image mask must be opaque", 422);
        mask[i] = std::round(
                      .2126f * rgba[i * 4] + .7152f * rgba[i * 4 + 1]
                      + .0722f * rgba[i * 4 + 2]
                  )
                  / 255.f;
    }
}
}  // namespace kadan::serving
