#pragma once

#include <algorithm>
#include <initializer_list>
#include <limits>

#include "kadan/checkpoint_floats.hpp"
#include "kadan/device_workspace.hpp"
#include "kadan/stt.hpp"
#include "kadan/tts_audio.hpp"
#include "kadan/tts_generator.hpp"
#include "kadan/whisper.hpp"

namespace kadan::serving {
namespace speech_memory {
inline Bytes sum(std::initializer_list<Bytes> terms) {
    Bytes result = 0;
    for (auto term : terms) {
        if (term > std::numeric_limits<Bytes>::max() - result)
            throw std::overflow_error("speech_memory_size");
        result += term;
    }
    return result;
}

inline Bytes product(std::initializer_list<Bytes> factors) {
    Bytes result = 1;
    for (auto factor : factors) {
        if (factor && result > std::numeric_limits<Bytes>::max() / factor)
            throw std::overflow_error("speech_memory_size");
        result *= factor;
    }
    return result;
}

// Native projections bound each device's source tile to 16 MiB. Attention
// additionally packs one head's Q/K/V/output in host doubles; full heads bound
// every tiled plan. Checkpoint decoding adds its independently bounded buffer.
inline Bytes compute_host(Bytes queries, Bytes keys, Bytes head, Bytes devices) {
    return sum(
        {checkpoint::float_read_buffer_bytes,
         product(
             {devices,
              sum(
                  {DeviceWorkspace::host_staging_capacity,
                   4096,
                   product({2, sum({queries, keys}), head, sizeof(double)})}
              )}
         )}
    );
}
}  // namespace speech_memory

// Extra lazy allocations beyond the caller's input/output reservation and the
// resident checkpoint. Whisper recomputes decoder prefixes rather than retaining KV.
inline Bytes whisper_execution_host_bytes(
    stt::WhisperDimensions d, Bytes prompt, Bytes output, Bytes devices
) {
    using speech_memory::product;
    using speech_memory::sum;
    if (!prompt || !output || prompt > d.text_context
        || output > d.text_context - prompt + 1 || !d.audio_heads || !d.text_heads)
        throw std::invalid_argument("whisper_memory_shape");
    const auto tokens = sum({prompt, output - 1});
    const auto audio = product({d.audio_context, d.audio_state});
    const auto text = product({tokens, d.text_state});
    // Encoder first/second/position survive its blocks. Decoder position and
    // logits survive self/cross-attention and the expanded feed-forward buffer.
    const auto encoder = sum({product({10, audio}), d.audio_context});
    const auto decoder = sum(
        {product({d.text_context, d.text_state}),
         d.vocabulary,
         d.text_context,
         std::max(
             sum({product({8, text}), tokens}),
             sum({product({6, text}), product({2, audio}), d.audio_context})
         )}
    );
    const auto weights = std::max(
        {product({3, d.audio_state, std::max(d.audio_state, d.mels)}),
         product({256, std::max(d.audio_state, d.text_state)})}
    );
    const auto model_peak = product(
        {4,
         sum(
             {std::max(encoder, decoder),
              weights,
              product({192, d.audio_state}),
              product({8, std::max(d.audio_state, d.text_state)})}
         )}
    );
    const auto frontend =
        sum({product({stt::LogMel::max_samples, 4}), stt::LogMel::scratch_bytes});
    return sum(
        {std::max(model_peak, frontend),
         speech_memory::compute_host(
             std::max<Bytes>(d.audio_context, tokens),
             std::max<Bytes>(d.audio_context, tokens),
             std::max(d.audio_state / d.audio_heads, d.text_state / d.text_heads),
             devices
         ),
         product({output, 128})}
    );
}

inline Bytes speech_execution_host_bytes(
    tts::GeneratorConfig generator,
    tts::AudioConfig audio,
    Bytes text,
    Bytes instruction,
    Bytes frames,
    Bytes devices
) {
    using speech_memory::product;
    using speech_memory::sum;
    if (!text || text > 2048 || instruction > 2048 || !frames
        || frames > tts::AudioDecoder::max_frames)
        throw std::invalid_argument("speech_memory_shape");
    const auto context = sum({text, instruction, 11, frames});
    const auto small_context = sum({generator.codebooks, 1});
    const auto cache = [](tts::StackConfig d, Bytes length) {
        return product({2, d.layers, length, d.kv_heads, d.head_dim});
    };
    const auto step = [](tts::StackConfig d, Bytes length) {
        return sum(
            {product({3, d.state}),
             product({2, d.heads, d.head_dim}),
             product({2, d.kv_heads, d.head_dim}),
             product({2, d.intermediate}),
             length}
        );
    };
    const auto generator_peak = sum(
        {cache(generator.talker, context),
         cache(generator.predictor, small_context),
         product({7, generator.talker.state}),
         product({2, generator.predictor.state}),
         std::max(generator.codec_vocabulary, generator.entries),
         std::max(
             {step(generator.talker, context),
              step(generator.predictor, small_context),
              product({2, generator.talker.state})}
         ),
         product(
             {64,
              std::max(generator.talker.intermediate, generator.predictor.intermediate)}
         ),
         product({8, std::max(generator.talker.state, generator.predictor.state)})}
    );
    // Audio transformer, two latent upsamplers, then four waveform upsamplers.
    // During a residual convolution both old and new delta coexist with audio.
    Bytes audio_peak = std::max(
        {sum(
             {product({frames, audio.code_dim, 2}),
              product({audio.code_dim, audio.code_dim}),
              audio.entries}
         ),
         product(
             {frames,
              sum(
                  {product({4, audio.state}),
                   product({4, audio.heads, audio.head_dim}),
                   product({2, audio.intermediate}),
                   1}
              )}
         ),
         product({frames, 4, 6, audio.latent})}
    );
    Bytes rows = product({frames, 4}), columns = audio.decoder;
    for (Bytes rate : {8, 5, 4, 3}) {
        const auto previous = product({rows, columns});
        rows = product({rows, rate});
        columns /= 2;
        audio_peak = std::max(
            {audio_peak,
             sum({previous, product({rows, columns})}),
             product({3, rows, columns})}
        );
    }
    // Largest convolution packing/projection tile, small weight slices and
    // transpose-source rows can coexist with the live activation peak.
    audio_peak = sum(
        {audio_peak,
         product({1024, std::max(audio.latent, audio.decoder)}),
         product({8, std::max(audio.state, audio.latent)})}
    );
    const auto head = std::max(
        {generator.talker.head_dim, generator.predictor.head_dim, audio.head_dim}
    );
    return sum(
        {product({4, std::max(generator_peak, audio_peak)}),
         generator.codec_vocabulary,
         speech_memory::compute_host(
             std::max(frames, context), std::max(frames, context), head, devices
         )}
    );
}

// DenseCompute budgets up to 2 GiB for adaptive projection/attention scratch;
// its resident weight cache is bounded to leave this space free.
inline constexpr Bytes speech_device_workspace_bytes = 2ULL * 1024 * 1024 * 1024;

inline bool speech_request_fits(
    const Snapshot& scope,
    const Footprint& physical,
    Bytes host,
    std::span<const int> devices
) {
    if (scope.capacity.empty() || physical.size() != scope.capacity.size()
        || scope.used.size() != scope.capacity.size())
        throw std::invalid_argument("speech_memory_budget_shape");
    const auto fits = [&](std::size_t index, Bytes bytes) {
        return scope.used[index] <= scope.capacity[index]
               && bytes <= scope.capacity[index] - scope.used[index]
               && bytes <= physical[index];
    };
    if (!fits(0, host))
        return false;
    for (int device : devices) {
        if (device < 0 || std::size_t(device) + 1 >= physical.size())
            throw std::invalid_argument("speech_memory_device");
        if (!fits(std::size_t(device) + 1, speech_device_workspace_bytes))
            return false;
    }
    return true;
}
}  // namespace kadan::serving
