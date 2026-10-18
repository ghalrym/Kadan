#include "kadan/drama_codec.hpp"

#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
void check(bool value) {
    if (!value)
        throw std::runtime_error("drama_codec_validation_test");
}

template <class F> void rejects(F action) {
    try {
        action();
    } catch (const std::invalid_argument&) {
        return;
    }
    throw std::runtime_error("expected_invalid_codec_window");
}

void frame_boundaries() {
    using namespace kadan::drama;
    rejects([] { codec_window_samples({}); });
    std::vector<std::uint32_t> codes(Codec::codebooks, 0);
    check(codec_window_samples(codes) == 2048);
    codes.resize(Codec::max_frames * Codec::codebooks);
    check(codec_window_samples(codes) == Codec::max_frames * 2048);
    codes.push_back(0);
    rejects([&] { codec_window_samples(codes); });
    codes.resize((Codec::max_frames + 1) * Codec::codebooks);
    rejects([&] { codec_window_samples(codes); });
}

void codebook_boundaries() {
    using namespace kadan::drama;
    std::vector<std::uint32_t> codes(2 * Codec::codebooks, 1023);
    codes[0] = 4095;
    codes[Codec::codebooks] = 4095;
    check(codec_window_samples(codes) == 4096);
    codes[Codec::codebooks] = 4096;
    rejects([&] { codec_window_samples(codes); });
    codes[Codec::codebooks] = 4095;
    codes.back() = 1024;
    check(codec_window_samples(codes) == 4096);
    codes.back() = 4095;
    const auto original_codes = codes;
    check(codec_window_samples(codes) == 4096);
    check(codes == original_codes);
    codes.back() = 4096;
    rejects([&] { codec_window_samples(codes); });
}
}  // namespace

int main() {
    frame_boundaries();
    codebook_boundaries();
    std::cout << "drama codec window validation passed\n";
}
