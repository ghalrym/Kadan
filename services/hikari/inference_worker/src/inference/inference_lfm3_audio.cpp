#include "kadan/inference_engines.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "kadan/checkpoint.hpp"
#include "kadan/lfm3.hpp"
#include "kadan/preparation.hpp"
#include "kadan/text_tokenizer.hpp"

namespace kadan::serving {
namespace {

constexpr std::size_t kAudioMaxContext = 32768;
constexpr std::size_t kMaxAudioSamples = 60 * 24000;
constexpr std::size_t kCodecFrameSamples = 1920;
constexpr std::size_t kPrefillChunkTokens = 512;

ComputePlan audio_plan(const Footprint& available) {
#ifdef KADAN_WORKER_CUDA
    if (available.size() > 1)
        return media_plan(available);
#endif
    return ComputePlan{available, {}};
}

std::vector<float> parse_audio_call_wav(const std::string& uri) {
    const std::string prefix = "data:audio/wav;base64,";
    preparation_check(
        uri.size() <= 900000 && uri.starts_with(prefix),
        "Audio calls require a bounded WAV data URI",
        422
    );
    const std::string raw = base64_decode(
        std::string_view(uri).substr(prefix.size()), 900000
    );
    const Wave info = wave(raw);
    preparation_check(
        info.channels == 1 && info.bits == 16 && info.rate == 16000,
        "Audio calls require mono 16-kHz PCM16 WAV",
        422
    );
    const std::size_t frames = info.bytes / 2;
    preparation_check(
        frames >= 1 && frames <= 20 * 16000,
        "Audio utterance must be at most twenty seconds",
        422
    );
    preparation_check(
        info.offset + frames * 2 <= raw.size(), "Audio utterance is truncated", 422
    );
    std::vector<float> pcm(frames);
    const auto* bytes = reinterpret_cast<const std::uint8_t*>(raw.data() + info.offset);
    for (std::size_t i = 0; i < frames; ++i) {
        const std::int16_t s = static_cast<std::int16_t>(
            static_cast<std::uint16_t>(bytes[2 * i])
            | (static_cast<std::uint16_t>(bytes[2 * i + 1]) << 8)
        );
        pcm[i] = static_cast<float>(s) / 32768.0f;
    }
    return pcm;
}

std::string render_audio_prompt(
    const std::vector<lfm3::Lfm3ChatMessage>& messages, bool add_generation_prompt
) {
    std::string out = "<|startoftext|>";
    for (std::size_t i = 0; i < messages.size(); ++i) {
        const auto& m = messages[i];
        const bool is_last = (i + 1 == messages.size());
        out += "<|im_start|>";
        out += m.role;
        out += '\n';
        out += m.content;
        if (!is_last || add_generation_prompt)
            out += "<|im_end|>\n";
    }
    if (add_generation_prompt)
        out += "<|im_start|>assistant\n";
    return out;
}

class Lfm3AudioEngine final : public InferenceEngine {
public:
    Lfm3AudioEngine(
        std::shared_ptr<Resources> resources,
        Progress progress,
        std::function<void(const Json&)> emit_event
    )
        : resources_(std::move(resources)), progress_(std::move(progress)),
          emit_event_(std::move(emit_event)) {
    }

    void load(
        const InferenceRequest& request,
        const Footprint& available,
        const std::atomic_bool& cancel
    ) override {
        check_cancel(cancel);
        const auto plan = audio_plan(available);
        compute_ =
            std::make_unique<WorkerCompute>(plan, Workload::lfm3_audio, resources_);

        const auto config_doc =
            read_document(request.configuration.checkpoint / "config.json");
        audio_config_ = lfm3::AudioModelConfig::parse(config_doc);
        auto model_budget =
            std::make_shared<checkpoint::MemoryBudget>(64ULL * 1024 * 1024);
        auto model_shard = std::make_shared<checkpoint::Shard>(
            request.configuration.checkpoint.string().c_str(),
            "model.safetensors",
            std::move(model_budget)
        );
        audio_model_ = std::make_unique<lfm3::AudioModel>(
            audio_config_, model_shard, resources_, compute_->compute
        );
        model_shard_ = std::move(model_shard);

        const auto codec_dir = request.configuration.checkpoint / "audio_tokenizer";
        const auto codec_config_doc = read_document(codec_dir / "config.json");
        codec_config_ = lfm3::LiquidCodecConfig::parse(codec_config_doc);
        auto codec_budget =
            std::make_shared<checkpoint::MemoryBudget>(32ULL * 1024 * 1024);
        auto codec_shard = std::make_shared<checkpoint::Shard>(
            codec_dir.string().c_str(),
            "model.safetensors",
            std::move(codec_budget)
        );
        codec_ = std::make_unique<lfm3::LiquidCodecDecoder>(
            codec_config_, codec_shard, resources_, compute_->compute
        );
        codec_shard_ = std::move(codec_shard);

        tokenizer_ =
            std::make_unique<text::Tokenizer>(resources_, Workload::lfm3_audio);
        tokenizer_->load(
            (request.configuration.checkpoint / "tokenizer.json").string(), cancel
        );
        im_end_id_ = tokenizer_->find_token("<|im_end|>");
        if (auto id = tokenizer_->find_token("<|audio|>"))
            audio_config_.audio_token_id = *id;
        if (auto id = tokenizer_->find_token("<|audio_out|>"))
            audio_config_.audio_out_token_id = *id;
        if (auto id = tokenizer_->find_token("<|audio_start|>"))
            audio_config_.audio_start_token_id = *id;
        if (auto id = tokenizer_->find_token("<|text_end|>"))
            audio_config_.text_end_token_id = *id;
        if (auto id = tokenizer_->find_token("<|mixed_start|>"))
            audio_config_.mixed_start_token_id = *id;
        if (auto id = tokenizer_->find_token("<|mixed_end|>"))
            audio_config_.mixed_end_token_id = *id;
    }

    void prepare(
        const InferenceRequest& request,
        const Footprint&,
        const std::atomic_bool& cancel
    ) override {
        check_cancel(cancel);
        const auto& input = std::get<AudioCallRequest>(request.input);
        if (input.mode == "transcribe" || input.mode == "turn") {
            preparation_check(
                input.audio.has_value(),
                "Audio calls require a bounded WAV data URI",
                422
            );
            input_pcm_ = parse_audio_call_wav(*input.audio);
        } else {
            input_pcm_.clear();
        }
        request_memory_ = std::make_unique<HostAllocation>(
            resources_, Workload::lfm3_audio, 64ULL * 1024 * 1024
        );
    }

    Json
    execute(const InferenceRequest& request, const std::atomic_bool& cancel) override {
        const auto& input = std::get<AudioCallRequest>(request.input);
        const auto started = std::chrono::steady_clock::now();
        Json stage_timing = Json::object();
        Json result;

        if (input.mode == "transcribe" || input.mode == "turn") {
            const auto asr_start = std::chrono::steady_clock::now();
            Json asr = run_submode(
                "transcribe", {}, {}, input_pcm_, kMaxAudioSamples, cancel
            );
            const double asr_sec =
                std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - asr_start
                )
                    .count();
            if (input.mode == "turn") {
                stage_timing["asr_operation_seconds"] = asr_sec;
                stage_timing["reply_operation_seconds"] = 0.0;
                const std::string transcript = asr.at("text").get<std::string>();
                if (emit_event_) {
                    emit_event_(
                        {{"id", request.id},
                         {"state", "audio_transcript"},
                         {"transcript", transcript}}
                    );
                }
                if (!transcript.empty()) {
                    std::vector<lfm3::Lfm3ChatMessage> tail;
                    for (std::size_t i = 1; i < input.messages.size(); ++i)
                        tail.push_back(input.messages[i]);
                    auto history = lfm3::append_call_history(tail, transcript, 12);
                    std::vector<lfm3::Lfm3ChatMessage> full_msgs;
                    full_msgs.push_back(input.messages.front());
                    full_msgs.insert(full_msgs.end(), history.begin(), history.end());
                    const auto reply_start = std::chrono::steady_clock::now();
                    if (input.sentence_tts) {
                        result = run_sentence_tts(request.id, full_msgs, cancel);
                    } else {
                        result = run_submode(
                            "reply", full_msgs, {}, {}, kMaxAudioSamples, cancel
                        );
                    }
                    stage_timing["reply_operation_seconds"] =
                        std::chrono::duration<double>(
                            std::chrono::steady_clock::now() - reply_start
                        )
                            .count();
                } else {
                    result = {{"text", ""}, {"audio_base64", nullptr}};
                }
                result["transcript"] = transcript;
            } else {
                result = std::move(asr);
            }
        } else {
            result = run_submode(
                input.mode,
                input.messages,
                input.script.value_or(""),
                {},
                kMaxAudioSamples,
                cancel
            );
        }

        const double total_sec =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - started)
                .count();
        Json timing = result.contains("timing") && result["timing"].is_object()
                          ? result["timing"]
                          : Json::object();
        timing["audio_import_seconds"] = 0.0;
        timing["model_load_seconds"] = 0.0;
        timing["processor_load_seconds"] = 0.0;
        timing["audio_operation_seconds"] = total_sec;
        for (const auto& [k, val] : stage_timing.items())
            timing[k] = val;
        result["timing"] = std::move(timing);

        input_pcm_.clear();
        input_pcm_.shrink_to_fit();
        request_memory_.reset();
        if (compute_)
            compute_->idle();
        return result;
    }

    void close() override {
        input_pcm_.clear();
        input_pcm_.shrink_to_fit();
        request_memory_.reset();
        tokenizer_.reset();
        codec_.reset();
        codec_shard_.reset();
        audio_model_.reset();
        model_shard_.reset();
        if (compute_) {
            compute_->park();
            compute_.reset();
        }
    }

private:
    Json run_sentence_tts(
        const std::string& request_id,
        const std::vector<lfm3::Lfm3ChatMessage>& messages,
        const std::atomic_bool& cancel
    ) {
        const auto started = std::chrono::steady_clock::now();
        const Json ans =
            run_submode("answer", messages, {}, {}, kMaxAudioSamples, cancel);
        const auto answered = std::chrono::steady_clock::now();
        const std::string answer_text = ans.at("text").get<std::string>();
        const auto scripts = lfm3::split_speech_scripts(answer_text);
        if (emit_event_) {
            emit_event_(
                {{"id", request_id},
                 {"state", "audio_answer"},
                 {"answer", answer_text}}
            );
        }
        std::size_t total_samples = 0;
        double total_tts = 0.0;
        std::optional<double> first_segment;
        for (std::size_t idx = 0; idx < scripts.size(); ++idx) {
            check_cancel(cancel);
            const auto seg_start = std::chrono::steady_clock::now();
            const Json seg = run_submode(
                "tts",
                {},
                scripts[idx],
                {},
                kMaxAudioSamples - total_samples,
                cancel
            );
            const auto seg_end = std::chrono::steady_clock::now();
            total_tts += std::chrono::duration<double>(seg_end - seg_start).count();
            const std::size_t sample_count = seg.at("sample_count").get<std::size_t>();
            const std::string b64 = seg.at("audio_base64").get<std::string>();
            preparation_check(
                sample_count >= 1 && !b64.empty(),
                "TTS returned an invalid known-script segment",
                502
            );
            total_samples += sample_count;
            preparation_check(
                total_samples <= kMaxAudioSamples,
                "Sentence TTS exceeded sixty seconds for this turn",
                502
            );
            if (!first_segment) {
                first_segment =
                    std::chrono::duration<double>(seg_end - started).count();
            }
            if (emit_event_) {
                emit_event_(
                    {{"id", request_id},
                     {"state", "audio_segment"},
                     {"audio_segment",
                      {{"index", idx},
                       {"script", scripts[idx]},
                       {"audio_base64", b64}}}}
                );
            }
        }
        return {
            {"text", answer_text},
            {"scripts", scripts},
            {"segment_count", scripts.size()},
            {"timing",
             {{"public_text_answer_seconds",
               std::chrono::duration<double>(answered - started).count()},
              {"first_tts_segment_seconds",
               first_segment ? Json(*first_segment) : Json(nullptr)},
              {"total_tts_seconds", total_tts},
              {"tts_segment_count", scripts.size()},
              {"tts_sample_count", total_samples}}}
        };
    }

    Json run_submode(
        const std::string& mode,
        const std::vector<lfm3::Lfm3ChatMessage>& messages,
        const std::string& script,
        std::span<const float> pcm_16k,
        std::size_t max_audio_samples,
        const std::atomic_bool& cancel
    ) {
        const bool is_transcribe = (mode == "transcribe");
        const bool is_answer = (mode == "answer");
        const bool is_tts = (mode == "tts");
        const bool is_reply = (mode == "reply");

        std::vector<float> audio_embeds;
        std::size_t audio_token_count = 0;
        std::vector<lfm3::Lfm3ChatMessage> conv;

        if (is_transcribe) {
            const auto mel = lfm3::extract_lfm3_audio_features(pcm_16k);
            audio_embeds = audio_model_->encode_audio(mel, cancel);
            audio_token_count = audio_embeds.size() / audio_config_.text_config.hidden_size;
            std::string placeholders;
            placeholders.reserve(audio_token_count * 10);
            for (std::size_t i = 0; i < audio_token_count; ++i)
                placeholders += "<|audio|>";
            conv.push_back({"system", "Perform ASR in English.", Json::array(), ""});
            conv.push_back({"user", placeholders, Json::array(), ""});
        } else if (is_tts) {
            conv.push_back(
                {"system",
                 "Perform TTS in American English using a female voice.",
                 Json::array(),
                 ""}
            );
            conv.push_back({"user", script, Json::array(), ""});
            conv.push_back({"assistant", "<|audio_start|>", Json::array(), ""});
        } else {
            conv = messages;
            if (is_reply) {
                conv.front().content =
                    "Respond with interleaved text and audio.\n\n" + conv.front().content;
                conv.push_back({"assistant", "<|mixed_start|>", Json::array(), ""});
            }
        }

        const bool add_gen_prompt = is_transcribe || is_answer;
        const std::string prompt_text = render_audio_prompt(conv, add_gen_prompt);

        std::vector<std::uint32_t> prompt_tokens(kAudioMaxContext + 1, 0);
        const std::size_t count = tokenizer_->encode(
            prompt_text, prompt_tokens, cancel,
            std::max<std::size_t>(prompt_text.size() + 64, 4 * 1024 * 1024)
        );
        prompt_tokens.resize(count);
        preparation_check(
            count >= 1 && count < kAudioMaxContext,
            "Audio call exceeds the 32768-token model context; start a new call",
            422
        );
        const std::size_t budget = kAudioMaxContext - count;
        const std::size_t max_frames = max_audio_samples / kCodecFrameSamples;
        if (!is_transcribe && !is_answer) {
            preparation_check(
                max_frames >= 1,
                "No speech duration remains within this call turn's sixty-second limit",
                502
            );
        }

        const auto& lm = audio_model_->language_model();
        const std::size_t h = audio_config_.text_config.hidden_size;
        const std::size_t v = audio_config_.text_config.vocab_size;

        std::vector<float> prompt_embeds(count * h);
        lm.embed_tokens(prompt_tokens, prompt_embeds, cancel);
        if (is_transcribe) {
            std::size_t a_idx = 0;
            for (std::size_t t = 0; t < count; ++t) {
                if (prompt_tokens[t] == audio_config_.audio_token_id) {
                    require(a_idx < audio_token_count, "audio_token_mismatch");
                    std::copy_n(
                        audio_embeds.data() + a_idx * h, h, prompt_embeds.data() + t * h
                    );
                    ++a_idx;
                }
            }
            require(a_idx == audio_token_count, "audio_token_count_mismatch");
        }

        lfm3::BackboneState state;
        state.reset(audio_config_.text_config);
        std::vector<float> last_hidden(h, 0.0f);
        for (std::size_t offset = 0; offset < count; offset += kPrefillChunkTokens) {
            check_cancel(cancel);
            const std::size_t chunk = std::min(kPrefillChunkTokens, count - offset);
            std::vector<float> out(chunk * h);
            lm.forward(
                std::span<const float>(prompt_embeds.data() + offset * h, chunk * h),
                chunk,
                out,
                &state,
                nullptr,
                cancel,
                kPrefillChunkTokens
            );
            if (offset + chunk == count)
                std::copy_n(out.data() + (chunk - 1) * h, h, last_hidden.data());
        }

        std::vector<std::uint32_t> eos_ids;
        if (audio_config_.text_config.eos_token_id)
            eos_ids.push_back(*audio_config_.text_config.eos_token_id);
        if (im_end_id_
            && std::find(eos_ids.begin(), eos_ids.end(), *im_end_id_) == eos_ids.end())
            eos_ids.push_back(*im_end_id_);
        auto is_turn_eos = [&](std::uint32_t tok) {
            return std::find(eos_ids.begin(), eos_ids.end(), tok) != eos_ids.end();
        };

        // LFM3-Audio generation mode state machine:
        // - transcribe / answer: pure text
        // - tts: pure audio frames until audio_eos_code -> emits text_end/im_end
        // - reply: interleaved (n_text=6 text tokens then n_audio=12 audio frames;
        //          once text emits text_end_token_id, pure audio tail until audio_eos)
        enum class GenMode { text, audio, mixed_text, mixed_audio };
        GenMode cur_mode = is_tts      ? GenMode::audio
                           : is_reply  ? GenMode::mixed_text
                                       : GenMode::text;
        std::size_t mode_counter = 0;
        bool text_finished = is_tts;
        bool audio_finished = (is_transcribe || is_answer);

        lfm3::SamplingConfig text_sampling;
        text_sampling.do_sample = false;  // greedy for ASR/answer/interleaved text stability
        if (is_transcribe || is_answer || is_tts) {
            text_sampling.suppress_ids = {
                audio_config_.mixed_start_token_id,
                audio_config_.mixed_end_token_id,
                audio_config_.audio_out_token_id,
                audio_config_.audio_start_token_id
            };
        }

        std::uint64_t rng_state = 0x243f6a8885a308d3ULL ^ count;
        std::vector<std::uint32_t> generated;
        std::vector<std::uint32_t> history = prompt_tokens;
        std::vector<std::int32_t> audio_codes;  // flat [frames * n_q]
        std::size_t num_audio_frames = 0;
        std::vector<float> logits(v);
        std::vector<float> step_embed(h);
        std::vector<float> step_out(h);

        for (std::size_t step = 0; step < budget; ++step) {
            check_cancel(cancel);
            progress_("decode", step + 1);
            if (cur_mode == GenMode::audio || cur_mode == GenMode::mixed_audio) {
                auto frame = audio_model_->sample_audio_frame(
                    last_hidden,
                    1.0f,
                    128,
                    1.1f,
                    4096,
                    audio_codes,
                    num_audio_frames,
                    rng_state,
                    cancel
                );
                const auto eos_code =
                    static_cast<std::int32_t>(audio_config_.audio_eos_code);
                const bool is_audio_eos =
                    !frame.empty()
                    && std::any_of(
                        frame.begin(), frame.end(),
                        [&](std::int32_t code) { return code == eos_code; }
                    );
                if (is_audio_eos) {
                    audio_finished = true;
                    const std::uint32_t end_tok =
                        is_tts ? audio_config_.text_end_token_id
                               : (im_end_id_.value_or(
                                     audio_config_.text_config.eos_token_id.value_or(7)
                                 ));
                    generated.push_back(end_tok);
                    break;
                }
                preparation_check(
                    num_audio_frames < max_frames,
                    "Audio generation exceeded the call turn's sixty-second speech limit",
                    502
                );
                audio_codes.insert(audio_codes.end(), frame.begin(), frame.end());
                ++num_audio_frames;
                generated.push_back(audio_config_.audio_out_token_id);

                if (step + 1 < budget) {
                    audio_model_->embed_audio_frame(frame, step_embed, cancel);
                    lm.forward(step_embed, 1, step_out, &state, nullptr, cancel, 1);
                    last_hidden.swap(step_out);
                }
                if (cur_mode == GenMode::mixed_audio) {
                    ++mode_counter;
                    if (!text_finished
                        && mode_counter >= audio_config_.interleaved_n_audio) {
                        cur_mode = GenMode::mixed_text;
                        mode_counter = 0;
                    }
                }
            } else {
                lm.lm_head(last_hidden, 1, logits, cancel);
                const std::uint32_t next_tok =
                    lfm3::sample_logits(logits, text_sampling, history, rng_state);
                generated.push_back(next_tok);
                history.push_back(next_tok);

                if (is_turn_eos(next_tok)
                    || ((is_transcribe || is_answer)
                        && next_tok == audio_config_.text_end_token_id)) {
                    break;
                }
                if (cur_mode == GenMode::mixed_text
                    && next_tok == audio_config_.text_end_token_id) {
                    text_finished = true;
                    cur_mode = GenMode::audio;
                    mode_counter = 0;
                }
                if (step + 1 < budget) {
                    std::uint32_t one[1] = {next_tok};
                    lm.embed_tokens(one, step_embed, cancel);
                    lm.forward(step_embed, 1, step_out, &state, nullptr, cancel, 1);
                    last_hidden.swap(step_out);
                }
                if (cur_mode == GenMode::mixed_text) {
                    ++mode_counter;
                    if (mode_counter >= audio_config_.interleaved_n_text) {
                        cur_mode = GenMode::mixed_audio;
                        mode_counter = 0;
                    }
                }
            }
        }

        const bool completed =
            !generated.empty()
            && (is_turn_eos(generated.back())
                || ((is_transcribe || is_answer || is_tts)
                    && generated.back() == audio_config_.text_end_token_id));
        preparation_check(
            completed,
            "Audio generation reached the context limit before completing the reply",
            502
        );

        std::string raw_decoded;
        for (std::uint32_t tok : generated)
            raw_decoded += tokenizer_->decode(tok, false);

        const std::string text =
            lfm3::validate_audio_public_text(raw_decoded, is_transcribe || is_answer);
        if (is_transcribe || is_answer) {
            if (is_answer) {
                preparation_check(
                    !text.empty(), "Audio model returned no public answer", 502
                );
            }
            return {{"text", text}};
        }
        if (is_tts) {
            preparation_check(text.empty(), "TTS returned unexpected public text", 502);
        }
        preparation_check(
            (is_tts || !text.empty()) && num_audio_frames > 0 && audio_finished,
            "Audio model did not return the required speech output",
            502
        );
        preparation_check(
            num_audio_frames * kCodecFrameSamples <= max_audio_samples,
            "Audio output exceeds the call turn's sixty-second speech limit",
            502
        );

        const std::vector<float> samples =
            codec_->decode(audio_codes, num_audio_frames, cancel);
        preparation_check(
            !samples.empty() && samples.size() <= max_audio_samples,
            "Audio model returned invalid or oversized speech",
            502
        );
        for (float s : samples) {
            preparation_check(
                std::isfinite(s), "Audio model returned invalid or oversized speech", 502
            );
        }
        const std::string b64 = lfm3::encode_wav_pcm16_base64(samples, 24000);
        Json out = {{"text", text}, {"audio_base64", b64}};
        if (is_tts) {
            out["script"] = script;
            out["sample_count"] = samples.size();
        }
        return out;
    }

    std::shared_ptr<Resources> resources_;
    Progress progress_;
    std::function<void(const Json&)> emit_event_;
    std::unique_ptr<WorkerCompute> compute_;
    lfm3::AudioModelConfig audio_config_;
    lfm3::LiquidCodecConfig codec_config_;
    std::shared_ptr<checkpoint::Shard> model_shard_;
    std::shared_ptr<checkpoint::Shard> codec_shard_;
    std::unique_ptr<lfm3::AudioModel> audio_model_;
    std::unique_ptr<lfm3::LiquidCodecDecoder> codec_;
    std::unique_ptr<text::Tokenizer> tokenizer_;
    std::optional<std::uint32_t> im_end_id_;
    std::unique_ptr<HostAllocation> request_memory_;
    std::vector<float> input_pcm_;
};

}  // namespace

std::unique_ptr<InferenceEngine> lfm3_audio_engine(
    std::shared_ptr<Resources> resources,
    Progress progress,
    std::function<void(const Json&)> emit_event
) {
    return std::make_unique<Lfm3AudioEngine>(
        std::move(resources), std::move(progress), std::move(emit_event)
    );
}

}  // namespace kadan::serving
