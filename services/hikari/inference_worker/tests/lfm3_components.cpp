// Pure unit tests for LFM3 + d1 C++ components: math, config parsing, prompt
// rendering, vision resize, audio feature extraction, WAV encoding, d1 formatting,
// and calibration. No checkpoint files, no network, no inference runs.
#include "kadan/lfm3.hpp"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

using namespace kadan::lfm3;

namespace {

void test_backbone_config_parsing() {
    const Json doc = {
        {"model_type", "lfm3"},
        {"vocab_size", 128000},
        {"hidden_size", 2048},
        {"intermediate_size", 7680},
        {"num_hidden_layers", 4},
        {"num_attention_heads", 16},
        {"num_key_value_heads", 4},
        {"head_dim", 128},
        {"max_position_embeddings", 262144},
        {"conv_L_cache", 7},
        {"sliding_window", 256},
        {"norm_eps", 1e-5},
        {"conv_bias", false},
        {"use_sandwich_norm", true},
        {"tie_word_embeddings", true},
        {"eos_token_id", 7},
        {"layer_types", {"conv", "sliding_attention", "conv", "full_attention"}},
        {"rope_parameters",
         {{"full_attention", {{"rope_theta", 1000000.0}, {"partial_rotary_factor", 0.5}}},
          {"sliding_attention", {{"rope_theta", 10000.0}, {"partial_rotary_factor", 1.0}}}}}
    };
    const auto cfg = BackboneConfig::parse(doc);
    assert(cfg.vocab_size == 128000);
    assert(cfg.hidden_size == 2048);
    assert(cfg.num_hidden_layers == 4);
    assert(cfg.layer_types.size() == 4);
    assert(cfg.layer_types[0] == LayerKind::conv);
    assert(cfg.layer_types[1] == LayerKind::sliding_attention);
    assert(cfg.layer_types[3] == LayerKind::full_attention);
    assert(cfg.full_rope.theta == 1000000.0);
    assert(cfg.full_rope.partial_rotary_factor == 0.5);
    assert(cfg.sliding_rope.theta == 10000.0);
    assert(cfg.eos_token_id == 7u);

    const auto codec_cfg = BackboneConfig::liquid_codec_decoder_config(512, 30);
    assert(codec_cfg.hidden_size == 512);
    assert(codec_cfg.num_hidden_layers == 8);
    assert(codec_cfg.sliding_window == 30);
    assert(codec_cfg.is_causal);
}

void test_norms_and_rope() {
    const std::vector<float> x = {3.0f, 4.0f};
    const std::vector<float> w = {1.0f, 2.0f};
    std::vector<float> out(2, 0.0f);
    rms_norm_lfm3(x, w, 1, 2, 1e-5f, out);
    const float expected_inv = 1.0f / std::sqrt(12.5f + 1e-5f);
    assert(std::fabs(out[0] - 3.0f * expected_inv) < 1e-4f);
    assert(std::fabs(out[1] - 8.0f * expected_inv) < 1e-4f);

    const std::vector<float> b = {0.5f, -0.5f};
    layer_norm_1d(x, w, b, 1, 2, 1e-5f, out);
    // mean=3.5, var=0.25, std=0.5 -> (-1, +1) -> * w + b = (-0.5, +1.5)
    assert(std::fabs(out[0] - (-0.5f)) < 1e-3f);
    assert(std::fabs(out[1] - 1.5f) < 1e-3f);

    // Partial RoPE (factor 0.5 on head_dim 8 -> 2 active freqs + 2 zero freqs)
    std::vector<float> inv_freq(4, 0.0f);
    compute_partial_rope_inv_freq(10000.0, 0.5, 8, inv_freq);
    assert(std::fabs(inv_freq[0] - 1.0f) < 1e-6f);
    assert(inv_freq[1] > 0.0f && inv_freq[1] < 1.0f);
    assert(inv_freq[2] == 0.0f && inv_freq[3] == 0.0f);

    std::vector<float> vec = {1, 2, 3, 4, 5, 6, 7, 8};
    apply_rope_lfm3(vec, 1, 1, 8, 3, inv_freq);
    // Zero-frequency lanes (indices 2,3 and 6,7) remain untouched.
    assert(vec[2] == 3.0f && vec[3] == 4.0f && vec[6] == 7.0f && vec[7] == 8.0f);
}

void test_causal_conv1d_streaming_equivalence() {
    const std::size_t rows = 5;
    const std::size_t ch = 2;
    const std::size_t k = 3;
    const std::vector<float> input = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    const std::vector<float> weight = {0.5f, 1.0f, 2.0f, 1.0f, -1.0f, 0.5f};
    const std::vector<float> bias = {0.1f, -0.2f};

    std::vector<float> full_out(rows * ch, 0.0f);
    std::vector<float> hist_full(ch * (k - 1), 0.0f);
    depthwise_causal_conv1d(input, weight, bias, rows, ch, k, hist_full, full_out);

    std::vector<float> stream_out(rows * ch, 0.0f);
    std::vector<float> hist_stream(ch * (k - 1), 0.0f);
    for (std::size_t r = 0; r < rows; ++r) {
        depthwise_causal_conv1d(
            std::span<const float>(input.data() + r * ch, ch),
            weight,
            bias,
            1,
            ch,
            k,
            hist_stream,
            std::span<float>(stream_out.data() + r * ch, ch)
        );
    }
    for (std::size_t i = 0; i < rows * ch; ++i)
        assert(std::fabs(full_out[i] - stream_out[i]) < 1e-5f);
}

void test_sampling_and_reasoning_budget() {
    SamplingConfig greedy;
    greedy.do_sample = false;
    std::vector<float> logits = {1.0f, 5.0f, 2.0f, 3.0f};
    std::uint64_t rng = 42;
    assert(sample_logits(logits, greedy, {}, rng) == 1u);

    greedy.suppress_ids = {1};
    logits = {1.0f, 5.0f, 2.0f, 3.0f};
    assert(sample_logits(logits, greedy, {}, rng) == 3u);

    ReasoningBudget budget;
    budget.active = true;
    budget.max_reasoning_tokens = 2;
    budget.end_think_token = 2;
    logits = {1.0f, 5.0f, 0.0f, 3.0f};
    budget.apply(2, 1u, logits);
    assert(logits[2] == 0.0f && std::isinf(logits[1]) && logits[1] < 0.0f);
    budget.apply(3, 2u, logits);
    assert(budget.closed);
}

void test_prompts_and_output_parsing() {
    std::vector<Lfm3ChatMessage> msgs = {
        {"system", "Be helpful.", Json::array(), ""},
        {"user", "Hello", Json::array(), ""}
    };
    const std::string prompt = render_lfm3_chat_prompt(msgs, Json(), true, false);
    assert(prompt.find("<|startoftext|>") == 0);
    assert(prompt.find("<|im_start|>system\nBe helpful.<|im_end|>\n") != std::string::npos);
    assert(prompt.find("<|im_start|>assistant\n<think>") != std::string::npos);

    assert(extract_lfm3_reply_text("plan</think>\nFinal answer.<|im_end|>") == "Final answer.");
    bool rejected = false;
    try {
        (void)extract_lfm3_reply_text("unfinished thinking without close");
    } catch (const std::exception&) {
        rejected = true;
    }
    assert(rejected);

    assert(
        extract_lfm3_vision_description("<think>look</think>A green button.<|im_end|>")
        == "A green button."
    );

    assert(validate_audio_public_text("Hello there.<|im_end|>", false) == "Hello there.");
    bool audio_think_rejected = false;
    try {
        (void)validate_audio_public_text("<think>oops</think>Hi", false);
    } catch (const std::exception&) {
        audio_think_rejected = true;
    }
    assert(audio_think_rejected);

    const auto scripts = split_speech_scripts("First sentence. Second sentence! Third?");
    assert(scripts.size() == 3);
    assert(scripts[0] == "First sentence.");
    assert(scripts[1] == "Second sentence!");
    assert(scripts[2] == "Third?");

    const auto hist = append_call_history({}, "Hi there", 12);
    assert(hist.size() == 1 && hist[0].role == "user" && hist[0].content == "Hi there");
}

void test_vision_resize_and_audio_features() {
    const auto [rh, rw] = smart_resize_lfm3_vl(1080, 1920, 32, 65536, 16777216);
    assert(rh % 32 == 0 && rw % 32 == 0);
    assert(rh * rw >= 65536 && rh * rw <= 16777216);

    // 0.5 seconds of 440 Hz sine wave at 16 kHz.
    std::vector<float> pcm(8000, 0.0f);
    for (std::size_t i = 0; i < pcm.size(); ++i)
        pcm[i] = 0.25f * std::sin(2.0f * 3.14159265f * 440.0f * float(i) / 16000.0f);
    const auto mel = extract_lfm3_audio_features(pcm);
    assert(mel.n_mels == 128);
    assert(mel.num_frames > 0);
    assert(mel.data.size() == mel.n_mels * mel.num_frames);
    assert(lfm3_audio_encoder_tokens(mel.num_frames, 8) > 0);

    const std::string wav_b64 = encode_wav_pcm16_base64(pcm, 24000);
    assert(!wav_b64.empty() && wav_b64.size() % 4 == 0);
}

void test_d1_helpers() {
    const Json cal_doc = {
        {"default", 1.337398453966105},
        {"temperatures", {{"score:3", 1.5}, {"choice:small", 1.2}}}
    };
    const auto cal = D1Calibration::parse(cal_doc);
    D1Question q_score;
    q_score.kind = D1Question::Kind::score;
    q_score.levels = {"low", "mid", "high"};
    assert(std::fabs(cal.temperature_for(q_score) - 1.5) < 1e-6);

    D1Question q_noul;
    q_noul.kind = D1Question::Kind::noul;
    assert(std::fabs(cal.temperature_for(q_noul) - 1.337398453966105) < 1e-6);

    const std::vector<std::string> specials = {"<|im_start|>", "<|im_end|>"};
    const std::string escaped = d1_escape_specials("a <|im_start|> b", specials);
    assert(escaped.find("<|im_start|>") == std::string::npos);

    const auto codes = d1_option_codes({"alpha", "beta", "gamma"}, 2);
    assert(codes.size() == 3);
    assert(codes[0] != codes[1] && codes[1] != codes[2]);

    const std::string prefix = d1_prefix_text("Context text.", specials);
    assert(prefix.find("Context text.") != std::string::npos);

    D1Question q_choice;
    q_choice.key = "pick";
    q_choice.kind = D1Question::Kind::choice;
    q_choice.instructions = "Pick one.";
    q_choice.options = {{"a", "Option A"}, {"b", "Option B"}};
    const std::string suffix = d1_suffix_text(q_choice, specials, 2);
    assert(suffix.find("Pick one.") != std::string::npos);
    assert(suffix.find("Option A") != std::string::npos);
}

}  // namespace

int main() {
    test_backbone_config_parsing();
    test_norms_and_rope();
    test_causal_conv1d_streaming_equivalence();
    test_sampling_and_reasoning_budget();
    test_prompts_and_output_parsing();
    test_vision_resize_and_audio_features();
    test_d1_helpers();
    std::cout << "LFM3 and d1 C++ component units passed\n";
    return 0;
}
