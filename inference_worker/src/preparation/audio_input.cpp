#include "kadan/preparation.hpp"

namespace kadan::serving {
namespace {
std::uint32_t little(std::string_view bytes, std::size_t offset, std::size_t size) {
    preparation_check(
        offset <= bytes.size() && size <= bytes.size() - offset, "Truncated WAV", 422
    );
    std::uint32_t value = 0;
    for (std::size_t i = 0; i < size; ++i)
        value |= std::uint32_t(static_cast<unsigned char>(bytes[offset + i]))
                 << (8 * i);
    return value;
}

}  // namespace

Wave wave(std::string_view bytes) {
    preparation_check(
        bytes.size() >= 12 && bytes.substr(0, 4) == "RIFF"
            && bytes.substr(8, 4) == "WAVE",
        "Expected PCM WAV",
        422
    );
    preparation_check(
        std::uint64_t(little(bytes, 4, 4)) + 8 == bytes.size(),
        "Truncated WAV container",
        422
    );
    Wave result;
    bool format = false, data = false;
    for (std::size_t at = 12; at + 8 <= bytes.size();) {
        const auto size = little(bytes, at + 4, 4);
        preparation_check(size <= bytes.size() - at - 8, "Truncated WAV chunk", 422);
        if (bytes.substr(at, 4) == "fmt ") {
            preparation_check(
                !format && size >= 16 && little(bytes, at + 8, 2) == 1,
                "Expected uncompressed PCM WAV",
                422
            );
            format = true;
            result.channels = little(bytes, at + 10, 2);
            result.rate = little(bytes, at + 12, 4);
            result.alignment = little(bytes, at + 20, 2);
            result.bits = little(bytes, at + 22, 2);
        } else if (bytes.substr(at, 4) == "data") {
            preparation_check(!data, "Duplicate WAV data", 422);
            data = true;
            result.offset = at + 8;
            result.bytes = size;
        }
        at += 8 + size + (size % 2);
    }
    preparation_check(
        format && data && result.channels && result.rate && result.bits
            && result.bits % 8 == 0
            && result.alignment == result.channels * (result.bits / 8)
            && result.bytes % result.alignment == 0,
        "Invalid WAV format",
        422
    );
    return result;
}

std::vector<float> pcm_audio(std::string_view audio) {
    constexpr std::string_view prefix = "data:audio/wav;base64,";
    preparation_check(
        audio.starts_with(prefix) && audio.size() <= 2 * 1024 * 1024,
        "Supply at most 30 seconds of mono 16-kHz PCM WAV.",
        422
    );
    const auto bytes = base64_decode(audio.substr(prefix.size()));
    const auto info = wave(bytes);
    preparation_check(
        info.channels == 1 && info.rate == 16000 && info.bits == 16 && info.bytes > 0
            && info.bytes <= 480000 * 2,
        "Supply at most 30 seconds of mono 16-kHz PCM WAV.",
        422
    );
    std::vector<float> result;
    result.reserve(info.bytes / 2);
    for (std::size_t i = 0; i < info.bytes; i += 2) {
        const auto raw = little(bytes, info.offset + i, 2);
        const auto sample =
            static_cast<float>(raw < 32768 ? int(raw) : int(raw) - 65536) / 32768.0f;
        result.push_back(sample);
    }
    return result;
}

}  // namespace kadan::serving
