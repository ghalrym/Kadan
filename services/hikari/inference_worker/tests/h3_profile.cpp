#include "kadan/h3_profile.hpp"

#include <array>
#include <iostream>
#include <stdexcept>

#include "kadan/h3_memory.hpp"
using namespace kadan::video;

void check(bool b) {
    if (!b)
        throw std::runtime_error("h3_profile_test_failed");
}

void owned_input_fits_existing_execution_allowance() {
    // Count the actual denoiser arrays independently of the estimator:
    // current/normalized/projected, QKV/Q/K/V/attended, two FF planes,
    // F64 rotary coordinates, time IDs and resident modality groups.
    constexpr kadan::Bytes hidden = 5376, inner = 7168, ff = 14336;
    constexpr kadan::Bytes scratch_per_token =
        (3 * hidden + 7 * inner + 2 * ff) * 4 + 96 * 8 + 8 + 4;
    constexpr kadan::Bytes input_per_token = hidden * 4;
    for (auto edge : {480u, 768u})
        for (auto aspect : {"16:9", "9:16", "1:1"})
            for (std::size_t seconds = 1; seconds <= 15; ++seconds)
                for (std::size_t prompt_bytes : {0u, 128u, 32000u}) {
                    const auto profile = h3_api_profile(edge, aspect, seconds);
                    const auto text = std::min<std::size_t>(128000, 4 * prompt_bytes);
                    const auto tokens =
                        text + profile.video_tokens + profile.audio_tokens;
                    const auto upload_bytes = tokens * input_per_token;
                    // Even the smallest accepted clip's hidden upload dominates
                    // the fixed maximum 3 timesteps * 3 modalities * 6 AdaLN rows.
                    check(upload_bytes >= 3 * 3 * 6 * hidden * 4);
                    const auto execution = h3_denoiser_host_bytes_per_token * tokens;
                    kadan::Resources host({execution});
                    const auto scratch = host.reserve(
                        kadan::Workload::video, {scratch_per_token * tokens}
                    );
                    const auto input =
                        host.reserve(kadan::Workload::video, {upload_bytes});
                    check(host.snapshot().used[0] <= execution);
                    check(h3_request_memory(profile, prompt_bytes) >= execution);
                    host.released(input);
                    host.released(scratch);
                    check(host.snapshot().used == kadan::Footprint({0}));
                }
}

int main() {
    try {
        owned_input_fits_existing_execution_allowance();
        // Independent expected values from the installed H3 shape resolver, for
        // every advertised duration and each API resolution/aspect combination.
        constexpr std::array<std::size_t, 15> frames{
            39, 56, 73, 107, 124, 158, 175, 192, 226, 243, 277, 294, 328, 345, 362
        };
        constexpr std::array<std::size_t, 15> times{
            12, 17, 22, 32, 37, 47, 52, 57, 67, 72, 82, 87, 97, 102, 107
        };
        for (auto edge : {480u, 768u})
            for (auto aspect : {"16:9", "9:16", "1:1"})
                for (std::size_t seconds = 1; seconds <= 15; ++seconds) {
                    const auto p = h3_api_profile(edge, aspect, seconds);
                    const auto long_edge = edge == 480 ? 864u : 1344u;
                    check(
                        p.width
                        == (std::string_view(aspect) == "16:9" ? long_edge : edge)
                    );
                    check(
                        p.height
                        == (std::string_view(aspect) == "9:16" ? long_edge : edge)
                    );
                    check(
                        p.frames == frames[seconds - 1]
                        && p.video_time == times[seconds - 1]
                    );
                    check(p.output_frames == seconds * 24);
                    check(p.frames >= seconds * 24 && p.frames - seconds * 24 < 17);
                    check(
                        p.video_tokens
                        == p.video_time * (p.width / 32) * (p.height / 32)
                    );
                }
        constexpr kadan::Bytes gib = 1024ULL * 1024 * 1024;
        const auto go = h3_request_memory(h3_api_profile(480, "16:9", 3), 128);
        const auto long_video = h3_request_memory(h3_api_profile(768, "16:9", 15), 128);
        check(go == 9 * gib);
        check(long_video == 56 * gib);
        check(h3_cache_memory(32 * gib, go) == 23 * gib / 2);
        check(h3_cache_memory(512 * gib, long_video) == 228 * gib);
        check(h3_cache_memory(go, go) == 0);
        check(h3_cache_memory(go - 1, go) == 0);
        // A larger request must be readmitted when the existing cache envelope
        // leaves too little execution room, even if its pages are still untouched.
        check(32 * gib - h3_cache_memory(32 * gib, go) < long_video);
        const auto maximum = h3_api_profile(768, "16:9", 15);
        check(
            maximum.video_tokens == 107856 && maximum.audio_tokens == 1206
            && maximum.vae_tokens == 28224
        );
        for (auto edge : {0u, 128u, 720u, 1080u}) {
            bool rejected = false;
            try {
                h3_api_profile(edge, "1:1", 4);
            } catch (const std::invalid_argument&) {
                rejected = true;
            }
            check(rejected);
        }
        for (auto seconds : {0u, 16u, ~0u}) {
            bool rejected = false;
            try {
                h3_api_profile(768, "1:1", seconds);
            } catch (const std::invalid_argument&) {
                rejected = true;
            }
            check(rejected);
        }
        std::cout << "PASS all 90 advertised H3 profiles, temporal alignment and "
                     "maximum token counts\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
