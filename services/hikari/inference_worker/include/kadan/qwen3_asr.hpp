#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "kadan/checkpoint.hpp"
#include "kadan/dense_compute.hpp"
#include "kadan/resources.hpp"
#include "kadan/text_tokenizer.hpp"

namespace kadan::stt {

struct Qwen3AsrAudioConfig {
    std::size_t d_model = 1024;
    std::size_t downsample_hidden_size = 480;
    std::size_t encoder_attention_heads = 16;
    std::size_t encoder_ffn_dim = 4096;
    std::size_t encoder_layers = 24;
    std::size_t num_mel_bins = 128;
    std::size_t max_source_positions = 1500;
    std::size_t n_window = 50;
    std::size_t n_window_infer = 800;
    std::size_t output_dim = 2048;
};

struct Qwen3AsrTextConfig {
    std::size_t vocab_size = 151936;
    std::size_t hidden_size = 2048;
    std::size_t intermediate_size = 6144;
    std::size_t num_hidden_layers = 28;
    std::size_t num_attention_heads = 16;
    std::size_t num_key_value_heads = 8;
    std::size_t head_dim = 128;
    std::size_t max_position_embeddings = 65536;
    float rms_norm_eps = 1e-6f;
    double rope_theta = 1000000.0;
};

struct Qwen3AsrConfig {
    Qwen3AsrAudioConfig audio;
    Qwen3AsrTextConfig text;
    std::uint32_t audio_start_token_id = 151669;
    std::uint32_t audio_end_token_id = 151670;
    std::uint32_t audio_token_id = 151676;
    std::vector<std::uint32_t> eos_token_ids = {151645, 151643};
    std::vector<std::string> support_languages;
};

struct Qwen3AsrResult {
    std::string text;
    std::string language;
};

// Pure helpers exposed for unit testing (no model loading or inference).
Qwen3AsrConfig read_qwen3_asr_config(
    const char* root, const std::shared_ptr<checkpoint::MemoryBudget>& budget
);
std::size_t feat_extract_output_length(std::size_t mel_frames);
std::vector<float> sinusoids_position_embedding(
    std::size_t length, std::size_t channels, double max_timescale = 10000.0
);
std::optional<std::string> normalize_language_hint(std::string_view hint);
std::string detect_and_fix_repetitions(std::string_view text, std::size_t threshold = 20);
Qwen3AsrResult parse_asr_output(
    std::string_view raw, std::string_view forced_language = {}
);

// Native C++20/CUDA Qwen3-ASR 1.7B model (AuT audio encoder + Qwen3 causal decoder).
class Qwen3Asr {
public:
    static constexpr std::size_t max_new_tokens = 512;

    Qwen3Asr(
        std::shared_ptr<Resources> resources,
        std::shared_ptr<DenseCompute> compute = {}
    );
    ~Qwen3Asr();
    Qwen3Asr(const Qwen3Asr&) = delete;
    Qwen3Asr& operator=(const Qwen3Asr&) = delete;

    void load(const char* root, const std::atomic_bool& cancel);
    void unload();
    bool loaded() const;
    const Qwen3AsrConfig& config() const;

    // mel is band-major [num_mel_bins, mel_frames] produced by LogMel::execute.
    Qwen3AsrResult transcribe(
        std::span<const float> mel,
        std::size_t mel_frames,
        text::Tokenizer& tokenizer,
        std::string_view forced_language,
        const std::atomic_bool& cancel,
        const std::function<void(const char*, std::size_t)>& progress = {}
    );

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace kadan::stt
