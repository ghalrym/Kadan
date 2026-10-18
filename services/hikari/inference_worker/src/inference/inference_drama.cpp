#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>

#include "kadan/drama_codec.hpp"
#include "kadan/drama_model.hpp"
#include "kadan/drama_prompt.hpp"
#include "kadan/inference_engines.hpp"
#include "kadan/preparation.hpp"
#include "kadan/text_tokenizer.hpp"

namespace kadan::serving {
namespace {
using Clock = std::chrono::steady_clock;

double milliseconds(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

class DramaEngine final : public InferenceEngine {
public:
    DramaEngine(
        std::shared_ptr<Resources> resources, Progress progress, AudioChunk audio
    )
        : resources_(std::move(resources)), progress_(std::move(progress)),
          audio_(std::move(audio)) {
    }

    void load(
        const InferenceRequest& request,
        const Footprint& available,
        const std::atomic_bool& cancel
    ) override {
        const auto started = Clock::now();
        check_cancel(cancel);
        require(request.model == "drama-3", "drama_native_model_unsupported");
        const auto info = drama::inspect(request.configuration.checkpoint);
        // RAM holds the admitted decoder and generator state. FP8 expert weights
        // stay file-backed and are read on demand, not counted as a 3B model.
        if (available[0] < info.host_bytes + 3ULL * 1024 * 1024 * 1024)
            throw AdmissionWait("drama_host_headroom");
        compute_ = std::make_unique<WorkerCompute>(
            media_plan(available), Workload::tts, resources_
        );
        tokenizer_ = std::make_unique<text::Tokenizer>(
            resources_, Workload::tts, text::Tokenizer::Format::drama
        );
        tokenizer_->load(
            std::get<SpeechRequest>(request.input).tokenizer.string(), cancel
        );
        generator_ = std::make_unique<drama::Generator>(resources_, compute_->compute);
        generator_->load(request.configuration.checkpoint, cancel);
        // Validate tokenizer/model identity before the first generation.
        std::array<std::uint32_t, 2> token{};
        require(
            tokenizer_->encode("<|semantic:0|>", token, cancel) == 1
                && token[0] == generator_->info().semantic_begin,
            "drama_tokenizer_semantic_id"
        );
        codec_ = std::make_unique<drama::Codec>(resources_, compute_->compute);
        codec_->load((request.configuration.checkpoint / "codec").c_str(), cancel);
        load_ms_ = milliseconds(started);
    }

    void prepare(
        const InferenceRequest& request,
        const Footprint& available,
        const std::atomic_bool& cancel
    ) override {
        check_cancel(cancel);
        constexpr Bytes scratch = 8 * 1024 * 1024;
        if (available[0] < scratch)
            throw AdmissionWait("drama_request_headroom");
        request_memory_ =
            std::make_unique<HostAllocation>(resources_, Workload::tts, scratch);
        prompt_.resize(3072);
        std::size_t used = 0;
        for (const auto& segment :
             drama::prompt_segments(std::get<SpeechRequest>(request.input).script))
            used +=
                tokenizer_->encode(segment, std::span(prompt_).subspan(used), cancel);
        require(
            used > 1 && used + max_frames <= generator_->info().context_limit,
            "drama_prompt_context_limit"
        );
        prompt_.resize(used);
    }

    Json
    execute(const InferenceRequest& request, const std::atomic_bool& cancel) override {
        const auto started = Clock::now();
        std::optional<double> first_audio_ms;
        std::size_t frames = 0, emitted_frames = 0, window_index = 0, samples = 0,
                    chunks = 0;
        constexpr std::array<std::size_t, 4> ramp{10, 10, 12, 32};
        std::vector<std::uint32_t> codes;
        codes.reserve((64 + overlap_frames) * drama::Codec::codebooks);
        auto emit_window = [&] {
            check_cancel(cancel);
            const auto overlap = std::min(emitted_frames, overlap_frames);
            const auto decoded = codec_->decode_window(codes, cancel);
            const auto begin = overlap * drama::Codec::samples_per_frame;
            require(decoded.size() >= begin, "drama_codec_window_size");
            // Bounded wire events keep cancellation responsive and avoid oversized
            // IPC lines; they are consecutive PCM samples, not codec windows.
            for (std::size_t offset = begin; offset < decoded.size(); offset += 8192) {
                check_cancel(cancel);
                const auto count = std::min<std::size_t>(8192, decoded.size() - offset);
                std::string pcm(count * 2, '\0');
                for (std::size_t i = 0; i < count; ++i) {
                    require(
                        std::isfinite(decoded[offset + i]), "drama_nonfinite_audio"
                    );
                    const auto sample = std::uint16_t(
                        std::clamp(
                            std::lround(decoded[offset + i] * 32768), -32768L, 32767L
                        )
                    );
                    pcm[2 * i] = char(sample & 255);
                    pcm[2 * i + 1] = char(sample >> 8);
                }
                if (!first_audio_ms)
                    first_audio_ms = milliseconds(started);
                check_cancel(cancel);
                audio_(
                    {{"audio", base64_encode(pcm)},
                     {"sample_rate", drama::Codec::sample_rate},
                     {"model", request.model},
                     {"format", "s16le"},
                     {"sequence", chunks++}}
                );
                samples += count;
            }
            emitted_frames = frames;
            const auto retained =
                std::min(frames, overlap_frames) * drama::Codec::codebooks;
            codes.erase(codes.begin(), codes.end() - retained);
            ++window_index;
        };
        const bool stopped = generator_->generate(
            prompt_,
            max_frames,
            cancel,
            [&](std::span<const std::uint32_t> frame) {
                check_cancel(cancel);
                require(frame.size() == drama::Codec::codebooks, "drama_codebooks");
                codes.insert(codes.end(), frame.begin(), frame.end());
                ++frames;
                progress_("speech_frames", frames);
                const auto stride =
                    window_index < ramp.size() ? ramp[window_index] : 64;
                if (frames - emitted_frames >= stride)
                    emit_window();
            },
            drama::Sampling{1.0f, .9f, 30, 0}
        );
        if (frames > emitted_frames)
            emit_window();
        check_cancel(cancel);
        require(samples > 0, "drama_empty_audio");
        const auto elapsed = milliseconds(started);
        const auto seconds = double(samples) / drama::Codec::sample_rate;
        compute_->idle();
        prompt_.clear();
        request_memory_.reset();
        const auto load_ms = load_ms_;
        load_ms_ = 0;
        return {
            {"finish_reason", stopped ? "stop" : "length"},
            {"model", request.model},
            {"sample_rate", drama::Codec::sample_rate},
            {"timing",
             {{"audio_seconds", seconds},
              {"first_audio_ms", first_audio_ms.value()},
              {"generation_ms", elapsed},
              {"load_ms", load_ms},
              {"real_time_factor", elapsed / 1000 / seconds},
              {"audio_chunks", chunks},
              {"frames", frames}}}
        };
    }

    void close() override {
        prompt_.clear();
        request_memory_.reset();
        codec_.reset();
        generator_.reset();
        tokenizer_.reset();
        if (compute_)
            compute_->park();
        compute_.reset();
    }

private:
    static constexpr std::size_t max_frames = 640, overlap_frames = 32;
    std::shared_ptr<Resources> resources_;
    Progress progress_;
    AudioChunk audio_;
    std::unique_ptr<WorkerCompute> compute_;
    std::unique_ptr<HostAllocation> request_memory_;
    std::unique_ptr<text::Tokenizer> tokenizer_;
    std::unique_ptr<drama::Generator> generator_;
    std::unique_ptr<drama::Codec> codec_;
    std::vector<std::uint32_t> prompt_;
    double load_ms_ = 0;
};
}  // namespace

std::unique_ptr<InferenceEngine> drama_engine(
    std::shared_ptr<Resources> resources, Progress progress, AudioChunk audio
) {
    return std::make_unique<DramaEngine>(
        std::move(resources), std::move(progress), std::move(audio)
    );
}
}  // namespace kadan::serving
