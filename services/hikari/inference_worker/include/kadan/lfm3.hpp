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
#include <vector>

#include <nlohmann/json.hpp>

#include "kadan/checkpoint.hpp"
#include "kadan/dense_compute.hpp"
#include "kadan/resources.hpp"
#include "kadan/text_tokenizer.hpp"

namespace kadan::lfm3 {

using Json = nlohmann::json;

enum class LayerKind {
    conv,
    full_attention,
    sliding_attention
};

struct RopeParams {
    double theta = 1000000.0;
    double partial_rotary_factor = 1.0;
};

struct BackboneConfig {
    std::size_t vocab_size = 128000;
    std::size_t hidden_size = 2048;
    std::size_t intermediate_size = 7680;
    std::size_t num_hidden_layers = 24;
    std::size_t num_attention_heads = 16;
    std::size_t num_key_value_heads = 4;
    std::size_t head_dim = 128;
    std::size_t max_position_embeddings = 262144;
    std::size_t conv_l_cache = 7;
    std::size_t sliding_window = 256;
    float norm_eps = 1e-5f;
    bool conv_bias = false;
    bool use_sandwich_norm = true;
    bool tie_word_embeddings = true;
    bool is_causal = true;
    std::optional<std::uint32_t> bos_token_id;
    std::optional<std::uint32_t> eos_token_id;
    std::optional<std::uint32_t> pad_token_id;
    std::vector<LayerKind> layer_types;
    RopeParams full_rope{1000000.0, 0.5};
    RopeParams sliding_rope{10000.0, 1.0};

    static BackboneConfig parse(const Json& config);
    static BackboneConfig liquid_codec_decoder_config(
        std::size_t encoder_dim, std::size_t decoder_window
    );
};

// Optional custom attention visibility for d1 `set` attention on full_attention layers.
struct SetAttentionMask {
    std::size_t state_len = 0;
    // Per-token group ID: -1 for state tokens, 1..N for question tokens.
    std::span<const int> question_groups;
};

struct SamplingConfig {
    bool do_sample = true;
    float temperature = 0.1f;
    std::size_t top_k = 50;
    float top_p = 1.0f;
    float repetition_penalty = 1.1f;
    std::size_t repetition_window = 0;  // 0 = full history
    std::uint64_t seed = 0;
    std::vector<std::uint32_t> suppress_ids;
};

// Per-request recurrent conv + KV cache for Lfm3Backbone.
struct BackboneState {
    struct LayerState {
        // Depthwise conv history: [hidden_size, conv_l_cache - 1]
        std::vector<float> conv_history;
        // KV cache: [seq_len, kv_heads * head_dim]
        std::vector<float> keys;
        std::vector<float> values;
        std::size_t kv_len = 0;
    };
    std::vector<LayerState> layers;
    std::size_t position = 0;

    void reset(const BackboneConfig& config);
};

// Pure C++20/CUDA LFM3/LFM2 transformer+shortconv backbone (no Python/LibTorch).
// Uses DenseCompute (CUDA BF16/F32 weight cache + attention) when `compute` is provided,
// and tiled OpenBLAS/CPU fallback otherwise.
class Backbone {
public:
    Backbone(
        BackboneConfig config,
        std::shared_ptr<checkpoint::Shard> shard,
        std::string weight_prefix,
        std::shared_ptr<Resources> resources,
        Workload workload,
        std::shared_ptr<DenseCompute> compute = {},
        bool load_embeddings = true,
        bool load_lm_head = true
    );
    ~Backbone();
    Backbone(const Backbone&) = delete;
    Backbone& operator=(const Backbone&) = delete;

    const BackboneConfig& config() const noexcept {
        return config_;
    }

    // Look up embeddings for `tokens` into `out` ([tokens.size(), hidden_size]).
    void embed_tokens(
        std::span<const std::uint32_t> tokens,
        std::span<float> out,
        const std::atomic_bool& cancel
    ) const;

    // Run backbone forward over `inputs_embeds` ([rows, hidden_size]) in place or into
    // `output` ([rows, hidden_size]). When `state` is non-null, updates conv/KV state
    // incrementally starting at `state->position`. When `set_mask` is non-null, full
    // attention layers use d1's set-attention visibility and state is not updated.
    void forward(
        std::span<const float> inputs_embeds,
        std::size_t rows,
        std::span<float> output,
        BackboneState* state,
        const SetAttentionMask* set_mask,
        const std::atomic_bool& cancel,
        std::size_t prefill_chunk_size = 512
    ) const;

    // Project selected hidden rows ([rows, hidden_size]) through lm_head into
    // `logits` ([rows, vocab_size]).
    void lm_head(
        std::span<const float> hidden,
        std::size_t rows,
        std::span<float> logits,
        const std::atomic_bool& cancel
    ) const;

private:
    BackboneConfig config_;
    std::shared_ptr<checkpoint::Shard> shard_;
    std::string prefix_;
    std::string lm_head_name_;
    std::shared_ptr<Resources> resources_;
    Workload workload_;
    std::shared_ptr<DenseCompute> compute_;
    bool has_embeddings_ = true;
    bool has_lm_head_ = true;
    std::vector<float> full_inv_freq_;
    std::vector<float> sliding_inv_freq_;
};

// Pure math helpers exposed for unit testing.
void rms_norm_lfm3(
    std::span<const float> input,
    std::span<const float> weight,
    std::size_t rows,
    std::size_t width,
    float eps,
    std::span<float> output
);

void layer_norm_1d(
    std::span<const float> input,
    std::span<const float> weight,
    std::span<const float> bias,
    std::size_t rows,
    std::size_t width,
    float eps,
    std::span<float> output
);

void compute_partial_rope_inv_freq(
    double theta, double partial_rotary_factor, std::size_t head_dim, std::span<float> out
);

void apply_rope_lfm3(
    std::span<float> tensor,  // [rows, heads, head_dim]
    std::size_t rows,
    std::size_t heads,
    std::size_t head_dim,
    std::size_t start_position,
    std::span<const float> inv_freq
);

void depthwise_causal_conv1d(
    std::span<const float> input,  // [rows, channels]
    std::span<const float> weight,  // [channels, kernel_size]
    std::span<const float> bias,  // empty or [channels]
    std::size_t rows,
    std::size_t channels,
    std::size_t kernel_size,
    std::span<float> history,  // empty or [channels, kernel_size - 1]
    std::span<float> output
);

std::uint32_t sample_logits(
    std::span<float> logits,
    const SamplingConfig& config,
    std::span<const std::uint32_t> history,
    std::uint64_t& rng_state
);

// ---------------------------------------------------------------------------
// LFM3 Chat & Audio Prompt Formatting + Output Validation
// ---------------------------------------------------------------------------

struct Lfm3ChatMessage {
    std::string role;
    std::string content;
    Json tool_calls = Json::array();
    std::string tool_call_id;
};

struct ReasoningBudget {
    std::size_t max_reasoning_tokens = 0;
    std::uint32_t end_think_token = 0;
    bool active = false;
    bool closed = false;

    void apply(
        std::size_t generated_so_far,
        std::optional<std::uint32_t> last_token,
        std::span<float> logits
    );
};

std::string render_lfm3_chat_prompt(
    const std::vector<Lfm3ChatMessage>& messages,
    const Json& tools,
    bool enable_thinking = true,
    bool preserve_thinking = false
);

std::string extract_lfm3_reply_text(std::string_view raw_generated);
std::string extract_lfm3_vision_description(std::string_view raw_generated);
std::string validate_audio_public_text(std::string_view raw, bool transcribe);
std::vector<std::string> split_speech_scripts(std::string_view text);
std::vector<Lfm3ChatMessage> append_call_history(
    const std::vector<Lfm3ChatMessage>& history,
    std::string_view user_text,
    std::size_t max_messages = 12
);

// ---------------------------------------------------------------------------
// LFM3-VL Vision Preprocessor & Encoder
// ---------------------------------------------------------------------------

struct RgbImage {
    std::size_t width = 0;
    std::size_t height = 0;
    std::vector<std::uint8_t> rgb;  // [height * width * 3]
};

struct VisionConfig {
    std::size_t hidden_size = 1024;
    std::size_t intermediate_size = 2560;
    std::size_t num_hidden_layers = 14;
    std::size_t num_attention_heads = 16;
    std::size_t num_key_value_heads = 8;
    std::size_t patch_size = 16;
    std::size_t temporal_patch_size = 2;
    std::size_t spatial_merge_size = 2;
    std::size_t conv_kernel_size = 3;
    std::size_t in_channels = 3;
    std::size_t downsample_factor = 2;
    std::size_t projector_hidden_size = 4096;
    std::size_t text_hidden_size = 2048;
    float norm_eps = 1e-5f;
    float projector_layer_norm_eps = 1e-6f;
    bool projector_bias = true;
    double rope_theta = 1000000.0;
    std::uint32_t image_token_id = 125030;
    std::vector<LayerKind> layer_types;

    static VisionConfig parse(const Json& config);
};

std::pair<std::size_t, std::size_t> smart_resize_lfm3_vl(
    std::size_t height,
    std::size_t width,
    std::size_t factor = 32,
    std::size_t min_pixels = 65536,
    std::size_t max_pixels = 16777216
);

// Decodes PNG or baseline JPEG bytes, validating format matches `media_type`
// ("image/png" or "image/jpeg") and total pixels <= max_pixels.
RgbImage decode_vision_image(
    std::string_view raw_bytes,
    std::string_view media_type,
    std::size_t max_pixels = 4194304
);

struct VisionPatches {
    std::size_t grid_h = 0;
    std::size_t grid_w = 0;
    std::size_t num_patches = 0;
    std::size_t patch_dim = 0;
    std::vector<float> pixel_values;  // [num_patches, patch_dim]
};

VisionPatches preprocess_vision_image(const RgbImage& image, const VisionConfig& config);

class VisionEncoder {
public:
    VisionEncoder(
        VisionConfig config,
        std::shared_ptr<checkpoint::Shard> shard,
        std::shared_ptr<Resources> resources,
        std::shared_ptr<DenseCompute> compute = {}
    );

    // Encodes preprocessed patches into [merged_tokens, text_hidden_size] embeddings.
    std::vector<float> encode(
        const VisionPatches& patches, const std::atomic_bool& cancel
    ) const;

private:
    VisionConfig config_;
    std::shared_ptr<checkpoint::Shard> shard_;
    std::shared_ptr<Resources> resources_;
    std::shared_ptr<DenseCompute> compute_;
};

// ---------------------------------------------------------------------------
// LFM3-Audio Feature Extractor, Depth Decoder & LiquidCodec
// ---------------------------------------------------------------------------

struct AudioFeatureConfig {
    std::size_t n_mels = 128;
    std::size_t sample_rate = 16000;
    std::size_t n_fft = 400;
    std::size_t win_length = 400;
    std::size_t hop_length = 160;
    float log_floor = -12.0f;
    float norm_offset = 4.0f;
    float norm_divisor = 4.0f;
    std::size_t encoder_stride = 8;
};

struct MelSpectrogram {
    std::size_t n_mels = 0;
    std::size_t num_frames = 0;
    std::vector<float> data;  // [n_mels, num_frames]
};

MelSpectrogram extract_lfm3_audio_features(
    std::span<const float> pcm_16k, const AudioFeatureConfig& config = {}
);

std::size_t lfm3_audio_encoder_tokens(
    std::size_t num_frames, std::size_t stride = 8
);

struct LiquidCodecConfig {
    std::size_t encoder_dim = 512;
    std::size_t stft_n_fft = 1280;
    std::size_t stft_hop_length = 320;
    std::size_t stft_win_length = 1280;
    std::size_t stft_frame_stack = 6;
    std::size_t quantizer_dim = 256;
    std::vector<int> fsq_levels{8, 7, 6, 6};
    std::size_t n_q = 12;
    std::size_t decoder_window = 30;
    std::size_t sampling_rate = 24000;

    static LiquidCodecConfig parse(const Json& config);
    std::size_t codebook_size() const;
    std::size_t frame_samples() const {
        return stft_hop_length * stft_frame_stack;
    }
};

class LiquidCodecDecoder {
public:
    LiquidCodecDecoder(
        LiquidCodecConfig config,
        std::shared_ptr<checkpoint::Shard> shard,
        std::shared_ptr<Resources> resources,
        std::shared_ptr<DenseCompute> compute = {}
    );

    // Decodes codes [num_frames, n_q] into 24 kHz mono float samples.
    std::vector<float> decode(
        std::span<const std::int32_t> codes,
        std::size_t num_frames,
        const std::atomic_bool& cancel
    ) const;

private:
    LiquidCodecConfig config_;
    std::shared_ptr<checkpoint::Shard> shard_;
    std::shared_ptr<Resources> resources_;
    std::shared_ptr<DenseCompute> compute_;
    std::unique_ptr<Backbone> decoder_lfm_;
    std::vector<int> basis_;
    std::vector<float> hann_window_;
};

std::string encode_wav_pcm16_base64(
    std::span<const float> samples, std::uint32_t sample_rate = 24000
);

struct AudioModelConfig {
    BackboneConfig text_config;
    BackboneConfig encoder_lfm_config;
    BackboneConfig depth_config;
    std::size_t encoder_dim_in = 128;
    std::size_t encoder_dim_conv = 256;
    std::size_t encoder_dim_out = 768;
    std::size_t encoder_sliding_window = 32;
    std::size_t num_codebooks = 12;
    std::size_t codebook_size = 2017;
    std::size_t audio_eos_code = 2016;
    std::size_t maskgit_steps = 3;
    std::size_t interleaved_n_text = 6;
    std::size_t interleaved_n_audio = 12;
    std::uint32_t audio_token_id = 125031;
    std::uint32_t audio_out_token_id = 125032;
    std::uint32_t audio_start_token_id = 125012;
    std::uint32_t text_end_token_id = 125014;
    std::uint32_t mixed_start_token_id = 125015;
    std::uint32_t mixed_end_token_id = 125016;

    static AudioModelConfig parse(const Json& config);
};

class AudioModel {
public:
    AudioModel(
        AudioModelConfig config,
        std::shared_ptr<checkpoint::Shard> shard,
        std::shared_ptr<Resources> resources,
        std::shared_ptr<DenseCompute> compute = {}
    );

    const AudioModelConfig& config() const noexcept {
        return config_;
    }

    // Encode log-mel [n_mels, num_frames] -> projected audio embeddings [n_tokens, hidden_size].
    std::vector<float> encode_audio(
        const MelSpectrogram& mel, const std::atomic_bool& cancel
    ) const;

    // Sample one 12-codebook speech frame from LM hidden state via 3-step MaskGIT depth decoder.
    std::vector<std::int32_t> sample_audio_frame(
        std::span<const float> lm_hidden,
        float temperature,
        std::size_t top_k,
        float repetition_penalty,
        std::size_t repetition_window,
        std::span<const std::int32_t> past_frames,
        std::size_t past_frame_count,
        std::uint64_t& rng_state,
        const std::atomic_bool& cancel
    ) const;

    // Sum codebook embeddings for a speech frame to feed back into LM.
    void embed_audio_frame(
        std::span<const std::int32_t> frame_codes,
        std::span<float> out_hidden,
        const std::atomic_bool& cancel
    ) const;

    const Backbone& language_model() const noexcept {
        return *language_model_;
    }

private:
    AudioModelConfig config_;
    std::shared_ptr<checkpoint::Shard> shard_;
    std::shared_ptr<Resources> resources_;
    std::shared_ptr<DenseCompute> compute_;
    std::unique_ptr<Backbone> language_model_;
    std::unique_ptr<Backbone> encoder_lfm_;
    std::unique_ptr<Backbone> depth_lfm_;
};

}  // namespace kadan::lfm3
