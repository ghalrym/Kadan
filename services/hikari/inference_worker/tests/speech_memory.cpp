#include "kadan/speech_memory.hpp"

#include <cassert>
#include <iostream>
#include <limits>

using namespace kadan;
using namespace kadan::serving;

namespace {
template <class Error, class Action> void rejects(Action action) {
    try {
        action();
    } catch (const Error&) {
        return;
    }
    throw std::runtime_error("speech_memory_expected_rejection");
}

void whisper_buffers_and_device_staging() {
    const stt::WhisperDimensions medium{
        80, 1500, 1024, 16, 24, 51865, 448, 1024, 16, 24
    };
    const auto host = whisper_execution_host_bytes(medium, 4, 224, 0);
    // Encoder's first/second/position, normalization and attention buffers alone.
    assert(host >= 10ULL * 1500 * 1024 * sizeof(float));
    const auto gpu = whisper_execution_host_bytes(medium, 4, 224, 1);
    const auto two = whisper_execution_host_bytes(medium, 4, 224, 2);
    assert(gpu > host + DeviceWorkspace::host_staging_capacity);
    assert(two - gpu == gpu - host);
    assert(whisper_execution_host_bytes(medium, 4, 445, 1) >= gpu);
    rejects<std::invalid_argument>([&] {
        whisper_execution_host_bytes(medium, 4, 446, 1);
    });
    auto oversized = medium;
    oversized.audio_state = std::numeric_limits<Bytes>::max();
    rejects<std::overflow_error>([&] {
        whisper_execution_host_bytes(oversized, 4, 224, 1);
    });
}

void speech_cache_and_waveform_peaks() {
    const auto short_audio = speech_execution_host_bytes({}, {}, 1, 0, 1, 1);
    const auto long_audio = speech_execution_host_bytes({}, {}, 1, 0, 300, 1);
    // Last waveform stage owns audio plus both sides of a residual convolution.
    assert(long_audio >= 3ULL * 300 * 1920 * 96 * sizeof(float));
    assert(long_audio > short_audio);
    const auto longest = speech_execution_host_bytes({}, {}, 2048, 2048, 300, 1);
    const tts::GeneratorConfig d;
    const auto kv = 2ULL * d.talker.layers * (2048 + 2048 + 11 + 300)
                    * d.talker.kv_heads * d.talker.head_dim * sizeof(float);
    assert(longest >= kv && longest > long_audio);
    assert(speech_execution_host_bytes({}, {}, 2048, 2048, 300, 3) > longest);
    rejects<std::invalid_argument>([] {
        speech_execution_host_bytes({}, {}, 2049, 0, 1, 1);
    });
    rejects<std::invalid_argument>([] {
        speech_execution_host_bytes({}, {}, 1, 2049, 1, 1);
    });
    rejects<std::invalid_argument>([] {
        speech_execution_host_bytes({}, {}, 1, 0, 301, 1);
    });
    rejects<std::invalid_argument>([] {
        speech_execution_host_bytes({}, {}, 0, 0, 1, 1);
    });
    auto oversized = d;
    oversized.talker.layers = std::numeric_limits<Bytes>::max();
    rejects<std::overflow_error>([&] {
        speech_execution_host_bytes(oversized, {}, 1, 0, 1, 1);
    });
    tts::AudioConfig audio;
    audio.latent = std::numeric_limits<Bytes>::max();
    rejects<std::overflow_error>([&] {
        speech_execution_host_bytes({}, audio, 1, 0, 1, 1);
    });
}

void admission_checks_scope_and_physical_memory() {
    constexpr Bytes gib = 1024ULL * 1024 * 1024;
    Resources resources({4 * gib, 4 * gib, 4 * gib});
    auto scope = resources.scope({2 * gib, 3 * gib, 3 * gib});
    auto resident = scope->reserve(Workload::tts, {gib, gib, gib});
    const std::vector<int> devices{0, 1};
    assert(speech_request_fits(
        scope->snapshot(), {4 * gib, 4 * gib, 4 * gib}, gib, devices
    ));
    // A request can pass a load-only estimate and still need sequential fallback.
    assert(!speech_request_fits(
        scope->snapshot(), {4 * gib, 4 * gib, 4 * gib}, gib + 1, devices
    ));
    scope->admit({4 * gib, 3 * gib, 3 * gib});
    assert(speech_request_fits(
        scope->snapshot(), {4 * gib, 4 * gib, 4 * gib}, gib + 1, devices
    ));
    assert(!speech_request_fits(
        scope->snapshot(), {gib, 4 * gib, 4 * gib}, gib + 1, devices
    ));
    assert(
        !speech_request_fits(scope->snapshot(), {4 * gib, 4 * gib, gib}, gib, devices)
    );
    assert(!speech_request_fits(
        Snapshot{{4 * gib, 3 * gib}, {0, gib + 1}, 0},
        {4 * gib, 4 * gib},
        gib,
        std::array{0}
    ));
    rejects<std::invalid_argument>([&] {
        speech_request_fits(scope->snapshot(), {gib}, 1, devices);
    });
    rejects<std::invalid_argument>([&] {
        speech_request_fits(
            scope->snapshot(), {4 * gib, 4 * gib, 4 * gib}, 1, std::array{2}
        );
    });
    scope->released(resident);
}
}  // namespace

int main() {
    whisper_buffers_and_device_staging();
    speech_cache_and_waveform_peaks();
    admission_checks_scope_and_physical_memory();
    std::cout << "Speech memory planning units passed; no model execution.\n";
}
