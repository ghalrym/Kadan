#include "kadan/qwen3_asr.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <limits>
#include <map>
#include <stdexcept>
#include <unordered_map>
#include <utility>

#include "../../weights/metadata_json.hpp"
#include "kadan/float_weights.hpp"
#include "kadan/layer_math.hpp"
#include "kadan/worker_compute.hpp"

namespace kadan::stt {
namespace {
constexpr std::size_t metadata_bytes = 32 * 1024 * 1024;
constexpr float layer_norm_eps = 1e-5f;

void need(bool ok, const char* error) {
    if (!ok)
        throw std::runtime_error(error);
}

void check_cancel(const std::atomic_bool& cancel) {
    need(!cancel.load(std::memory_order_relaxed), "qwen3_asr_cancelled");
}

void check_finite(std::span<const float> values) {
    for (float v : values)
        need(std::isfinite(v), "qwen3_asr_nonfinite");
}

float gelu_exact(float x) {
    constexpr float inv_sqrt2 = 0.70710678118654752440f;
    return 0.5f * x * (1.0f + std::erff(x * inv_sqrt2));
}

Footprint host_bytes(Resources& r, Bytes bytes) {
    return host_footprint(r, bytes);
}

struct Lease {
    Resources* resources;
    Handle handle;

    Lease(Resources& r, Bytes bytes)
        : resources(&r), handle(r.reserve(Workload::speech, host_bytes(r, bytes))) {
    }

    ~Lease() {
        if (resources && handle)
            resources->released(handle);
    }

    Lease(const Lease&) = delete;
    Lease& operator=(const Lease&) = delete;
    Lease(Lease&& other) noexcept
        : resources(std::exchange(other.resources, nullptr)),
          handle(std::exchange(other.handle, 0)) {
    }
    Lease& operator=(Lease&&) = delete;
};

struct FloatBuffer {
    Lease lease;
    std::vector<float> values;

    FloatBuffer(Resources& r, std::size_t count)
        : lease(r, count * sizeof(float)), values(count, 0.0f) {
    }

    FloatBuffer(FloatBuffer&&) noexcept = default;
    FloatBuffer(const FloatBuffer&) = delete;
    FloatBuffer& operator=(const FloatBuffer&) = delete;
    FloatBuffer& operator=(FloatBuffer&&) = delete;

    std::span<float> span() {
        return values;
    }

    std::span<const float> cspan() const {
        return values;
    }
};

struct BusyGuard {
    bool& flag;

    explicit BusyGuard(bool& b) : flag(b) {
        need(!flag, "qwen3_asr_busy");
        flag = true;
    }

    ~BusyGuard() {
        flag = false;
    }
};

struct PinGuard {
    Resources& r;
    Handle h;

    PinGuard(Resources& ledger, Handle handle) : r(ledger), h(handle) {
        r.pin(h);
    }

    ~PinGuard() {
        r.unpin(h);
    }
};

struct ScratchGuard {
    DenseCompute* compute;

    explicit ScratchGuard(DenseCompute* c) : compute(c) {
    }

    ~ScratchGuard() {
        if (compute)
            compute->release_scratch(false);
    }
};

void layer_norm_rows(
    std::span<const float> input,
    std::span<const float> weight,
    std::span<const float> bias,
    std::size_t rows,
    std::size_t width,
    float eps,
    std::span<float> output
) {
    need(
        width > 0 && input.size() == rows * width && output.size() == rows * width
            && weight.size() == width && bias.size() == width,
        "qwen3_asr_layernorm_shape"
    );
    for (std::size_t r = 0; r < rows; ++r) {
        const float* src = input.data() + r * width;
        float* dst = output.data() + r * width;
        double sum = 0.0;
        for (std::size_t c = 0; c < width; ++c)
            sum += src[c];
        const double mean = sum / double(width);
        double var = 0.0;
        for (std::size_t c = 0; c < width; ++c) {
            const double d = double(src[c]) - mean;
            var += d * d;
        }
        const float inv = static_cast<float>(1.0 / std::sqrt(var / double(width) + eps));
        for (std::size_t c = 0; c < width; ++c)
            dst[c] = (src[c] - static_cast<float>(mean)) * inv * weight[c] + bias[c];
    }
    check_finite(output);
}

void add_residual(std::span<float> hidden, std::span<const float> update) {
    need(hidden.size() == update.size(), "qwen3_asr_residual_shape");
    for (std::size_t i = 0; i < hidden.size(); ++i)
        hidden[i] += update[i];
    check_finite(hidden);
}

void cpu_bidirectional_attention(
    std::span<const float> q,
    std::span<const float> k,
    std::span<const float> v,
    std::size_t seq,
    std::size_t heads,
    std::size_t head_dim,
    std::span<float> out,
    const std::atomic_bool& cancel
) {
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    const std::size_t stride = heads * head_dim;
    std::vector<float> scores(seq);
    for (std::size_t qi = 0; qi < seq; ++qi) {
        check_cancel(cancel);
        for (std::size_t h = 0; h < heads; ++h) {
            const float* qh = q.data() + qi * stride + h * head_dim;
            float max_s = -std::numeric_limits<float>::infinity();
            for (std::size_t ki = 0; ki < seq; ++ki) {
                const float* kh = k.data() + ki * stride + h * head_dim;
                double dot = 0.0;
                for (std::size_t d = 0; d < head_dim; ++d)
                    dot += double(qh[d]) * kh[d];
                const float s = static_cast<float>(dot) * scale;
                scores[ki] = s;
                if (s > max_s)
                    max_s = s;
            }
            double denom = 0.0;
            for (std::size_t ki = 0; ki < seq; ++ki) {
                scores[ki] = std::exp(scores[ki] - max_s);
                denom += scores[ki];
            }
            const float inv = static_cast<float>(1.0 / denom);
            float* oh = out.data() + qi * stride + h * head_dim;
            std::fill_n(oh, head_dim, 0.0f);
            for (std::size_t ki = 0; ki < seq; ++ki) {
                const float w = scores[ki] * inv;
                const float* vh = v.data() + ki * stride + h * head_dim;
                for (std::size_t d = 0; d < head_dim; ++d)
                    oh[d] += w * vh[d];
            }
        }
    }
    check_finite(out);
}

void cpu_causal_gqa_attention(
    std::span<const float> q,
    std::span<const float> keys,
    std::span<const float> values,
    std::size_t rows,
    std::size_t position,
    std::size_t heads,
    std::size_t kv_heads,
    std::size_t head_dim,
    float scale,
    std::span<float> out,
    const std::atomic_bool& cancel
) {
    const std::size_t q_stride = heads * head_dim;
    const std::size_t kv_stride = kv_heads * head_dim;
    const std::size_t group = heads / kv_heads;
    std::vector<float> scores(position + rows);
    for (std::size_t r = 0; r < rows; ++r) {
        check_cancel(cancel);
        const std::size_t limit = position + r + 1;
        for (std::size_t h = 0; h < heads; ++h) {
            const std::size_t kvh = h / group;
            const float* qh = q.data() + r * q_stride + h * head_dim;
            float max_s = -std::numeric_limits<float>::infinity();
            for (std::size_t t = 0; t < limit; ++t) {
                const float* kh = keys.data() + t * kv_stride + kvh * head_dim;
                double dot = 0.0;
                for (std::size_t d = 0; d < head_dim; ++d)
                    dot += double(qh[d]) * kh[d];
                const float s = static_cast<float>(dot) * scale;
                scores[t] = s;
                if (s > max_s)
                    max_s = s;
            }
            double denom = 0.0;
            for (std::size_t t = 0; t < limit; ++t) {
                scores[t] = std::exp(scores[t] - max_s);
                denom += scores[t];
            }
            const float inv = static_cast<float>(1.0 / denom);
            float* oh = out.data() + r * q_stride + h * head_dim;
            std::fill_n(oh, head_dim, 0.0f);
            for (std::size_t t = 0; t < limit; ++t) {
                const float w = scores[t] * inv;
                const float* vh = values.data() + t * kv_stride + kvh * head_dim;
                for (std::size_t d = 0; d < head_dim; ++d)
                    oh[d] += w * vh[d];
            }
        }
    }
    check_finite(out);
}

std::string trim(std::string_view s) {
    std::size_t first = 0;
    while (first < s.size() && std::isspace(static_cast<unsigned char>(s[first])))
        ++first;
    std::size_t last = s.size();
    while (last > first && std::isspace(static_cast<unsigned char>(s[last - 1])))
        --last;
    return std::string(s.substr(first, last - first));
}

std::string lowercase(std::string_view s) {
    std::string out(s);
    std::transform(out.begin(), out.end(), out.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return out;
}

std::string canonical_case(std::string_view s) {
    auto out = lowercase(trim(s));
    if (!out.empty())
        out[0] = static_cast<char>(std::toupper(static_cast<unsigned char>(out[0])));
    return out;
}

std::string fix_char_repeats(std::string_view s, std::size_t threshold) {
    std::string out;
    out.reserve(s.size());
    std::size_t i = 0;
    while (i < s.size()) {
        std::size_t count = 1;
        while (i + count < s.size() && s[i + count] == s[i])
            ++count;
        if (count > threshold)
            out.push_back(s[i]);
        else
            out.append(s.substr(i, count));
        i += count;
    }
    return out;
}

std::string fix_pattern_repeats(
    std::string_view s, std::size_t threshold, std::size_t max_len = 20
) {
    const std::size_t min_chars = threshold * 2;
    if (s.size() < min_chars)
        return std::string(s);
    std::string out;
    out.reserve(s.size());
    std::size_t i = 0;
    while (i <= s.size() - min_chars) {
        bool found = false;
        for (std::size_t k = 1; k <= max_len; ++k) {
            if (i + k * threshold > s.size())
                break;
            const auto pattern = s.substr(i, k);
            bool valid = true;
            for (std::size_t rep = 1; rep < threshold; ++rep) {
                if (s.substr(i + rep * k, k) != pattern) {
                    valid = false;
                    break;
                }
            }
            if (valid) {
                std::size_t end_index = i + threshold * k;
                while (end_index + k <= s.size() && s.substr(end_index, k) == pattern)
                    end_index += k;
                out.append(pattern);
                out.append(fix_pattern_repeats(s.substr(end_index), threshold, max_len));
                return out;
            }
        }
        if (!found) {
            out.push_back(s[i]);
            ++i;
        }
    }
    out.append(s.substr(i));
    return out;
}

void validate_config(const Qwen3AsrConfig& cfg) {
    const auto& a = cfg.audio;
    need(
        a.d_model > 0 && a.d_model <= 4096 && a.downsample_hidden_size > 0
            && a.downsample_hidden_size <= 2048 && a.encoder_attention_heads > 0
            && a.d_model % a.encoder_attention_heads == 0 && a.encoder_ffn_dim > 0
            && a.encoder_ffn_dim <= 16384 && a.encoder_layers > 0
            && a.encoder_layers <= 64 && a.num_mel_bins == 128
            && a.max_source_positions > 0 && a.max_source_positions <= 8192
            && a.n_window > 0 && a.n_window_infer >= a.n_window * 2
            && a.n_window_infer % (a.n_window * 2) == 0 && a.output_dim > 0,
        "qwen3_asr_audio_config"
    );
    const auto& t = cfg.text;
    need(
        t.vocab_size > 0 && t.vocab_size <= 262144 && t.hidden_size == a.output_dim
            && t.hidden_size > 0 && t.hidden_size <= 8192 && t.intermediate_size > 0
            && t.intermediate_size <= 32768 && t.num_hidden_layers > 0
            && t.num_hidden_layers <= 64 && t.num_attention_heads > 0
            && t.num_key_value_heads > 0
            && t.num_attention_heads % t.num_key_value_heads == 0 && t.head_dim > 0
            && t.head_dim <= 256 && t.head_dim % 2 == 0
            && t.max_position_embeddings > 0 && t.rms_norm_eps > 0.0f
            && std::isfinite(t.rms_norm_eps) && t.rope_theta > 0.0
            && std::isfinite(t.rope_theta),
        "qwen3_asr_text_config"
    );
    need(
        cfg.audio_start_token_id < t.vocab_size && cfg.audio_end_token_id < t.vocab_size
            && cfg.audio_token_id < t.vocab_size && !cfg.eos_token_ids.empty(),
        "qwen3_asr_special_tokens"
    );
}
}  // namespace

Qwen3AsrConfig read_qwen3_asr_config(
    const char* root, const std::shared_ptr<checkpoint::MemoryBudget>& budget
) {
    need(root != nullptr && bool(budget), "qwen3_asr_config_args");
    const auto json = checkpoint::json::read(root, "config.json", 262144, budget);
    need(json.at("model_type").string() == "qwen3_asr", "qwen3_asr_model_type");
    const auto& thinker = json.at("thinker_config");
    const auto& ac = thinker.at("audio_config");
    const auto& tc = thinker.at("text_config");

    Qwen3AsrConfig cfg;
    cfg.audio.d_model = static_cast<std::size_t>(ac.at("d_model").integer());
    cfg.audio.downsample_hidden_size =
        static_cast<std::size_t>(ac.at("downsample_hidden_size").integer());
    cfg.audio.encoder_attention_heads =
        static_cast<std::size_t>(ac.at("encoder_attention_heads").integer());
    cfg.audio.encoder_ffn_dim =
        static_cast<std::size_t>(ac.at("encoder_ffn_dim").integer());
    cfg.audio.encoder_layers =
        static_cast<std::size_t>(ac.at("encoder_layers").integer());
    cfg.audio.num_mel_bins =
        static_cast<std::size_t>(ac.at("num_mel_bins").integer());
    cfg.audio.max_source_positions =
        static_cast<std::size_t>(ac.at("max_source_positions").integer());
    cfg.audio.n_window = static_cast<std::size_t>(ac.at("n_window").integer());
    cfg.audio.n_window_infer =
        static_cast<std::size_t>(ac.at("n_window_infer").integer());
    cfg.audio.output_dim = static_cast<std::size_t>(ac.at("output_dim").integer());

    cfg.text.vocab_size = static_cast<std::size_t>(tc.at("vocab_size").integer());
    cfg.text.hidden_size = static_cast<std::size_t>(tc.at("hidden_size").integer());
    cfg.text.intermediate_size =
        static_cast<std::size_t>(tc.at("intermediate_size").integer());
    cfg.text.num_hidden_layers =
        static_cast<std::size_t>(tc.at("num_hidden_layers").integer());
    cfg.text.num_attention_heads =
        static_cast<std::size_t>(tc.at("num_attention_heads").integer());
    cfg.text.num_key_value_heads =
        static_cast<std::size_t>(tc.at("num_key_value_heads").integer());
    if (const auto* hd = tc.find("head_dim");
        hd != nullptr && hd->kind == checkpoint::json::Value::Kind::number) {
        cfg.text.head_dim = static_cast<std::size_t>(hd->integer());
    } else {
        cfg.text.head_dim = cfg.text.hidden_size / cfg.text.num_attention_heads;
    }
    cfg.text.max_position_embeddings =
        static_cast<std::size_t>(tc.at("max_position_embeddings").integer());
    cfg.text.rms_norm_eps = static_cast<float>(tc.at("rms_norm_eps").number());
    cfg.text.rope_theta = tc.at("rope_theta").number();

    cfg.audio_start_token_id =
        static_cast<std::uint32_t>(thinker.at("audio_start_token_id").integer());
    cfg.audio_end_token_id =
        static_cast<std::uint32_t>(thinker.at("audio_end_token_id").integer());
    cfg.audio_token_id =
        static_cast<std::uint32_t>(thinker.at("audio_token_id").integer());

    if (const auto* langs = json.find("support_languages");
        langs != nullptr && langs->kind == checkpoint::json::Value::Kind::array) {
        for (const auto& item : langs->children)
            cfg.support_languages.emplace_back(item.string());
    }
    try {
        const auto gen =
            checkpoint::json::read(root, "generation_config.json", 65536, budget);
        if (const auto* eos = gen.find("eos_token_id"); eos != nullptr) {
            cfg.eos_token_ids.clear();
            if (eos->kind == checkpoint::json::Value::Kind::number)
                cfg.eos_token_ids.push_back(static_cast<std::uint32_t>(eos->integer()));
            else if (eos->kind == checkpoint::json::Value::Kind::array)
                for (const auto& id : eos->children)
                    cfg.eos_token_ids.push_back(static_cast<std::uint32_t>(id.integer()));
        }
    } catch (...) {
        // Default EOS token IDs remain {151645, 151643}.
    }
    validate_config(cfg);
    return cfg;
}

std::size_t feat_extract_output_length(std::size_t mel_frames) {
    const std::size_t leave = mel_frames % 100;
    std::size_t tail = 0;
    if (leave > 0) {
        const std::size_t t1 = (leave - 1) / 2 + 1;
        const std::size_t t2 = (t1 - 1) / 2 + 1;
        tail = (t2 - 1) / 2 + 1;
    }
    return tail + (mel_frames / 100) * 13;
}

std::vector<float> sinusoids_position_embedding(
    std::size_t length, std::size_t channels, double max_timescale
) {
    need(channels > 0 && channels % 2 == 0 && max_timescale > 1.0, "qwen3_asr_sinusoid");
    const std::size_t half = channels / 2;
    const double log_inc = std::log(max_timescale) / double(half - 1);
    std::vector<double> inv(half);
    for (std::size_t i = 0; i < half; ++i)
        inv[i] = std::exp(-log_inc * double(i));
    std::vector<float> out(length * channels);
    for (std::size_t pos = 0; pos < length; ++pos) {
        for (std::size_t i = 0; i < half; ++i) {
            const double angle = double(pos) * inv[i];
            out[pos * channels + i] = static_cast<float>(std::sin(angle));
            out[pos * channels + half + i] = static_cast<float>(std::cos(angle));
        }
    }
    return out;
}

std::optional<std::string> normalize_language_hint(std::string_view hint) {
    const auto clean = lowercase(trim(hint));
    if (clean.empty() || clean == "auto")
        return std::nullopt;
    static const std::unordered_map<std::string_view, std::string_view> aliases = {
        {"zh", "Chinese"},       {"chinese", "Chinese"},
        {"en", "English"},       {"english", "English"},
        {"yue", "Cantonese"},    {"cantonese", "Cantonese"},
        {"ar", "Arabic"},        {"arabic", "Arabic"},
        {"de", "German"},        {"german", "German"},
        {"fr", "French"},        {"french", "French"},
        {"es", "Spanish"},       {"spanish", "Spanish"},
        {"pt", "Portuguese"},    {"portuguese", "Portuguese"},
        {"id", "Indonesian"},    {"indonesian", "Indonesian"},
        {"it", "Italian"},       {"italian", "Italian"},
        {"ko", "Korean"},        {"korean", "Korean"},
        {"ru", "Russian"},       {"russian", "Russian"},
        {"th", "Thai"},          {"thai", "Thai"},
        {"vi", "Vietnamese"},    {"vietnamese", "Vietnamese"},
        {"ja", "Japanese"},      {"japanese", "Japanese"},
        {"tr", "Turkish"},       {"turkish", "Turkish"},
        {"hi", "Hindi"},         {"hindi", "Hindi"},
        {"ms", "Malay"},         {"malay", "Malay"},
        {"nl", "Dutch"},         {"dutch", "Dutch"},
        {"sv", "Swedish"},       {"swedish", "Swedish"},
        {"da", "Danish"},        {"danish", "Danish"},
        {"fi", "Finnish"},       {"finnish", "Finnish"},
        {"pl", "Polish"},        {"polish", "Polish"},
        {"cs", "Czech"},         {"czech", "Czech"},
        {"tl", "Filipino"},      {"fil", "Filipino"},      {"filipino", "Filipino"},
        {"fa", "Persian"},       {"persian", "Persian"},
        {"el", "Greek"},         {"greek", "Greek"},
        {"ro", "Romanian"},      {"romanian", "Romanian"},
        {"hu", "Hungarian"},     {"hungarian", "Hungarian"},
        {"mk", "Macedonian"},    {"macedonian", "Macedonian"},
    };
    const auto it = aliases.find(clean);
    if (it == aliases.end())
        return std::nullopt;
    return std::string(it->second);
}

std::string detect_and_fix_repetitions(std::string_view text, std::size_t threshold) {
    if (threshold == 0)
        return std::string(text);
    const auto chars_fixed = fix_char_repeats(text, threshold);
    return fix_pattern_repeats(chars_fixed, threshold);
}

Qwen3AsrResult parse_asr_output(
    std::string_view raw, std::string_view forced_language
) {
    auto cleaned = trim(raw);
    if (cleaned.empty())
        return {"", std::string(forced_language)};
    cleaned = detect_and_fix_repetitions(cleaned);
    if (!forced_language.empty())
        return {cleaned, std::string(forced_language)};

    constexpr std::string_view tag = "<asr_text>";
    const auto pos = cleaned.find(tag);
    if (pos == std::string::npos)
        return {trim(cleaned), ""};

    const auto meta = cleaned.substr(0, pos);
    const auto text = trim(cleaned.substr(pos + tag.size()));
    const auto meta_lower = lowercase(meta);
    if (meta_lower.find("language none") != std::string::npos)
        return {text, ""};

    std::string lang;
    std::size_t start = 0;
    while (start <= meta.size()) {
        const auto nl = meta.find('\n', start);
        const auto line = trim(
            nl == std::string::npos ? std::string_view(meta).substr(start)
                                    : std::string_view(meta).substr(start, nl - start)
        );
        const auto low = lowercase(line);
        constexpr std::string_view prefix = "language ";
        if (low.starts_with(prefix)) {
            const auto val = trim(std::string_view(line).substr(prefix.size()));
            if (!val.empty()) {
                lang = canonical_case(val);
                break;
            }
        }
        if (nl == std::string::npos)
            break;
        start = nl + 1;
    }
    return {text, lang};
}

struct Qwen3Asr::Impl {
    std::shared_ptr<Resources> resources;
    std::shared_ptr<DenseCompute> compute;
    Qwen3AsrConfig config;
    std::shared_ptr<checkpoint::MemoryBudget> budget;
    std::vector<std::unique_ptr<checkpoint::Shard>> shards;
    std::unordered_map<std::string, std::size_t> tensor_shard;
    std::vector<float> pos_emb;
    std::vector<float> inv_freq;
    Handle resident = 0;
    bool busy = false;

    const checkpoint::Shard& shard_for(std::string_view name) const {
        const auto it = tensor_shard.find(std::string(name));
        need(it != tensor_shard.end(), "qwen3_asr_missing_tensor");
        return *shards[it->second];
    }

    checkpoint::FloatSlice read_vec(
        std::string_view name, std::size_t count, const std::atomic_bool& cancel
    ) const {
        return checkpoint::float_slice(
            *resources, Workload::speech, shard_for(name), name, 0, count, cancel
        );
    }

    void project(
        std::string_view name,
        std::span<const float> bias,
        std::span<const float> input,
        std::size_t in,
        std::size_t out,
        std::span<float> output,
        const std::atomic_bool& cancel
    ) const {
        const auto res = checkpoint::project_source(
            *resources,
            Workload::speech,
            compute.get(),
            shard_for(name),
            name,
            bias,
            input,
            in,
            out,
            output,
            cancel
        );
        need(projection_output_finite(res, output), "qwen3_asr_nonfinite");
    }

    // Conv2d(kernel=3, stride=2, pad=1) + bias + exact GELU on one CHW chunk.
    void conv2d_s2_p1_gelu(
        std::string_view prefix,
        std::span<const float> input,
        std::size_t c_in,
        std::size_t h_in,
        std::size_t w_in,
        std::size_t c_out,
        std::span<float> output,
        const std::atomic_bool& cancel
    ) const {
        const std::size_t h_out = (h_in + 1) / 2;
        const std::size_t w_out = (w_in + 1) / 2;
        const std::size_t rows = h_out * w_out;
        const std::size_t cols = c_in * 9;
        need(
            input.size() == c_in * h_in * w_in && output.size() == c_out * h_out * w_out,
            "qwen3_asr_conv_shape"
        );
        FloatBuffer im2col(*resources, rows * cols);
        for (std::size_t oy = 0; oy < h_out; ++oy) {
            check_cancel(cancel);
            for (std::size_t ox = 0; ox < w_out; ++ox) {
                float* row = im2col.values.data() + (oy * w_out + ox) * cols;
                for (std::size_t ci = 0; ci < c_in; ++ci) {
                    const float* plane = input.data() + ci * h_in * w_in;
                    for (std::size_t kh = 0; kh < 3; ++kh) {
                        const std::ptrdiff_t iy =
                            static_cast<std::ptrdiff_t>(oy * 2 + kh) - 1;
                        for (std::size_t kw = 0; kw < 3; ++kw) {
                            const std::ptrdiff_t ix =
                                static_cast<std::ptrdiff_t>(ox * 2 + kw) - 1;
                            float val = 0.0f;
                            if (iy >= 0 && iy < static_cast<std::ptrdiff_t>(h_in)
                                && ix >= 0 && ix < static_cast<std::ptrdiff_t>(w_in)) {
                                val = plane[std::size_t(iy) * w_in + std::size_t(ix)];
                            }
                            row[ci * 9 + kh * 3 + kw] = val;
                        }
                    }
                }
            }
        }
        const auto bias = read_vec(std::string(prefix) + ".bias", c_out, cancel);
        FloatBuffer gemm_out(*resources, rows * c_out);
        project(
            std::string(prefix) + ".weight",
            bias,
            im2col.cspan(),
            cols,
            c_out,
            gemm_out.span(),
            cancel
        );
        // Transpose [h_out * w_out, c_out] -> [c_out, h_out, w_out] and apply GELU.
        for (std::size_t p = 0; p < rows; ++p) {
            const float* src = gemm_out.values.data() + p * c_out;
            for (std::size_t co = 0; co < c_out; ++co)
                output[co * rows + p] = gelu_exact(src[co]);
        }
        check_finite(output);
    }

    FloatBuffer encode_audio(
        std::span<const float> mel,
        std::size_t mel_frames,
        const std::atomic_bool& cancel,
        const std::function<void(const char*, std::size_t)>& progress
    ) const {
        const auto& a = config.audio;
        need(
            mel_frames > 0 && mel.size() == a.num_mel_bins * mel_frames,
            "qwen3_asr_mel_shape"
        );
        const std::size_t chunk_cap = a.n_window * 2;  // 100
        const std::size_t num_chunks = (mel_frames + chunk_cap - 1) / chunk_cap;
        const std::size_t max_chunk_len =
            num_chunks > 1 ? chunk_cap : mel_frames;
        const std::size_t t1 = (max_chunk_len + 1) / 2;
        const std::size_t t2 = (t1 + 1) / 2;
        const std::size_t max_t3 = (t2 + 1) / 2;
        const std::size_t f1 = (a.num_mel_bins + 1) / 2;
        const std::size_t f2 = (f1 + 1) / 2;
        const std::size_t f3 = (f2 + 1) / 2;
        const std::size_t total_tokens = feat_extract_output_length(mel_frames);
        need(max_t3 <= a.max_source_positions, "qwen3_asr_pos_limit");

        FloatBuffer hidden(*resources, total_tokens * a.d_model);
        FloatBuffer chunk_in(*resources, a.num_mel_bins * max_chunk_len);
        FloatBuffer c1(*resources, a.downsample_hidden_size * f1 * t1);
        FloatBuffer c2(*resources, a.downsample_hidden_size * f2 * t2);
        FloatBuffer c3(*resources, a.downsample_hidden_size * f3 * max_t3);
        FloatBuffer perm(*resources, max_t3 * a.downsample_hidden_size * f3);
        FloatBuffer proj(*resources, max_t3 * a.d_model);

        std::size_t out_cursor = 0;
        for (std::size_t b = 0; b < num_chunks; ++b) {
            check_cancel(cancel);
            const std::size_t src_start = b * chunk_cap;
            const std::size_t chunk_len =
                std::min(chunk_cap, mel_frames - src_start);
            const std::size_t valid_t3 = feat_extract_output_length(chunk_len);
            std::fill(chunk_in.values.begin(), chunk_in.values.end(), 0.0f);
            for (std::size_t m = 0; m < a.num_mel_bins; ++m) {
                std::copy_n(
                    mel.data() + m * mel_frames + src_start,
                    chunk_len,
                    chunk_in.values.data() + m * max_chunk_len
                );
            }
            conv2d_s2_p1_gelu(
                "thinker.audio_tower.conv2d1",
                chunk_in.cspan(),
                1,
                a.num_mel_bins,
                max_chunk_len,
                a.downsample_hidden_size,
                c1.span(),
                cancel
            );
            conv2d_s2_p1_gelu(
                "thinker.audio_tower.conv2d2",
                c1.cspan(),
                a.downsample_hidden_size,
                f1,
                t1,
                a.downsample_hidden_size,
                c2.span(),
                cancel
            );
            conv2d_s2_p1_gelu(
                "thinker.audio_tower.conv2d3",
                c2.cspan(),
                a.downsample_hidden_size,
                f2,
                t2,
                a.downsample_hidden_size,
                c3.span(),
                cancel
            );
            // Permute [C, F, T] -> [T, C * F]
            const std::size_t cf = a.downsample_hidden_size * f3;
            for (std::size_t t = 0; t < max_t3; ++t) {
                float* dst = perm.values.data() + t * cf;
                for (std::size_t c = 0; c < a.downsample_hidden_size; ++c)
                    for (std::size_t f = 0; f < f3; ++f)
                        dst[c * f3 + f] =
                            c3.values[(c * f3 + f) * max_t3 + t];
            }
            project(
                "thinker.audio_tower.conv_out.weight",
                {},
                perm.cspan(),
                cf,
                a.d_model,
                proj.span(),
                cancel
            );
            // Add sinusoidal positional embedding per chunk and gather valid frames.
            for (std::size_t t = 0; t < valid_t3; ++t) {
                float* dst = hidden.values.data() + (out_cursor + t) * a.d_model;
                const float* src = proj.values.data() + t * a.d_model;
                const float* pe = pos_emb.data() + t * a.d_model;
                for (std::size_t d = 0; d < a.d_model; ++d)
                    dst[d] = src[d] + pe[d];
            }
            out_cursor += valid_t3;
        }
        need(out_cursor == total_tokens, "qwen3_asr_chunk_count");

        // Block-diagonal window boundaries for encoder self-attention.
        const std::size_t window_aftercnn =
            max_t3 * (a.n_window_infer / (a.n_window * 2));
        std::vector<std::size_t> cu_seqlens = {0};
        for (std::size_t pos = 0; pos < total_tokens;) {
            const std::size_t step = std::min(window_aftercnn, total_tokens - pos);
            pos += step;
            cu_seqlens.push_back(pos);
        }

        const std::size_t head_dim = a.d_model / a.encoder_attention_heads;
        FloatBuffer norm(*resources, total_tokens * a.d_model);
        FloatBuffer q(*resources, total_tokens * a.d_model);
        FloatBuffer k(*resources, total_tokens * a.d_model);
        FloatBuffer v(*resources, total_tokens * a.d_model);
        FloatBuffer attn(*resources, total_tokens * a.d_model);
        FloatBuffer ffn(*resources, total_tokens * a.encoder_ffn_dim);

        for (std::size_t layer = 0; layer < a.encoder_layers; ++layer) {
            check_cancel(cancel);
            if (progress)
                progress("encode", layer + 1);
            const std::string p =
                "thinker.audio_tower.layers." + std::to_string(layer);
            {
                const auto w = read_vec(p + ".self_attn_layer_norm.weight", a.d_model, cancel);
                const auto b = read_vec(p + ".self_attn_layer_norm.bias", a.d_model, cancel);
                layer_norm_rows(
                    hidden.cspan(), w, b, total_tokens, a.d_model, layer_norm_eps, norm.span()
                );
            }
            {
                const auto qb = read_vec(p + ".self_attn.q_proj.bias", a.d_model, cancel);
                const auto kb = read_vec(p + ".self_attn.k_proj.bias", a.d_model, cancel);
                const auto vb = read_vec(p + ".self_attn.v_proj.bias", a.d_model, cancel);
                project(p + ".self_attn.q_proj.weight", qb, norm.cspan(), a.d_model, a.d_model, q.span(), cancel);
                project(p + ".self_attn.k_proj.weight", kb, norm.cspan(), a.d_model, a.d_model, k.span(), cancel);
                project(p + ".self_attn.v_proj.weight", vb, norm.cspan(), a.d_model, a.d_model, v.span(), cancel);
            }
            for (std::size_t w = 0; w + 1 < cu_seqlens.size(); ++w) {
                const std::size_t start = cu_seqlens[w];
                const std::size_t len = cu_seqlens[w + 1] - start;
                const std::size_t elem_off = start * a.d_model;
                const std::size_t elem_cnt = len * a.d_model;
                auto qw = std::span<const float>(q.values.data() + elem_off, elem_cnt);
                auto kw = std::span<const float>(k.values.data() + elem_off, elem_cnt);
                auto vw = std::span<const float>(v.values.data() + elem_off, elem_cnt);
                auto ow = std::span<float>(attn.values.data() + elem_off, elem_cnt);
                if (!compute
                    || !compute->attend(
                        qw,
                        kw,
                        vw,
                        len,
                        len,
                        a.encoder_attention_heads,
                        a.encoder_attention_heads,
                        head_dim,
                        0,
                        0,
                        0,
                        ow,
                        cancel
                    )) {
                    cpu_bidirectional_attention(
                        qw, kw, vw, len, a.encoder_attention_heads, head_dim, ow, cancel
                    );
                }
            }
            {
                const auto ob = read_vec(p + ".self_attn.out_proj.bias", a.d_model, cancel);
                project(p + ".self_attn.out_proj.weight", ob, attn.cspan(), a.d_model, a.d_model, norm.span(), cancel);
            }
            add_residual(hidden.span(), norm.cspan());

            {
                const auto w = read_vec(p + ".final_layer_norm.weight", a.d_model, cancel);
                const auto b = read_vec(p + ".final_layer_norm.bias", a.d_model, cancel);
                layer_norm_rows(
                    hidden.cspan(), w, b, total_tokens, a.d_model, layer_norm_eps, norm.span()
                );
            }
            {
                const auto b1 = read_vec(p + ".fc1.bias", a.encoder_ffn_dim, cancel);
                project(p + ".fc1.weight", b1, norm.cspan(), a.d_model, a.encoder_ffn_dim, ffn.span(), cancel);
            }
            for (float& val : ffn.values)
                val = gelu_exact(val);
            {
                const auto b2 = read_vec(p + ".fc2.bias", a.d_model, cancel);
                project(p + ".fc2.weight", b2, ffn.cspan(), a.encoder_ffn_dim, a.d_model, norm.span(), cancel);
            }
            add_residual(hidden.span(), norm.cspan());
        }

        // ln_post -> proj1 -> GELU -> proj2 -> [total_tokens, output_dim]
        {
            const auto w = read_vec("thinker.audio_tower.ln_post.weight", a.d_model, cancel);
            const auto b = read_vec("thinker.audio_tower.ln_post.bias", a.d_model, cancel);
            layer_norm_rows(
                hidden.cspan(), w, b, total_tokens, a.d_model, layer_norm_eps, norm.span()
            );
        }
        {
            const auto b1 = read_vec("thinker.audio_tower.proj1.bias", a.d_model, cancel);
            project("thinker.audio_tower.proj1.weight", b1, norm.cspan(), a.d_model, a.d_model, hidden.span(), cancel);
        }
        for (float& val : hidden.values)
            val = gelu_exact(val);

        FloatBuffer audio_embeds(*resources, total_tokens * a.output_dim);
        {
            const auto b2 = read_vec("thinker.audio_tower.proj2.bias", a.output_dim, cancel);
            project(
                "thinker.audio_tower.proj2.weight",
                b2,
                hidden.cspan(),
                a.d_model,
                a.output_dim,
                audio_embeds.span(),
                cancel
            );
        }
        return audio_embeds;
    }

    void embed_token(
        std::uint32_t token, std::span<float> dst, const std::atomic_bool& cancel
    ) const {
        const auto h = config.text.hidden_size;
        need(token < config.text.vocab_size && dst.size() == h, "qwen3_asr_embed_token");
        const auto& shard = shard_for("thinker.model.embed_tokens.weight");
        shard.read_float_tensor(
            "thinker.model.embed_tokens.weight", std::size_t(token) * h, dst, cancel
        );
    }

    bool is_eos(std::uint32_t token) const {
        return std::find(
                   config.eos_token_ids.begin(), config.eos_token_ids.end(), token
               )
               != config.eos_token_ids.end();
    }
};

Qwen3Asr::Qwen3Asr(
    std::shared_ptr<Resources> resources, std::shared_ptr<DenseCompute> compute
)
    : impl_(std::make_unique<Impl>()) {
    need(bool(resources), "qwen3_asr_resources");
    impl_->resources = std::move(resources);
    impl_->compute = std::move(compute);
}

Qwen3Asr::~Qwen3Asr() {
    unload();
}

bool Qwen3Asr::loaded() const {
    return impl_->resident != 0;
}

const Qwen3AsrConfig& Qwen3Asr::config() const {
    need(loaded(), "qwen3_asr_not_loaded");
    return impl_->config;
}

void Qwen3Asr::load(const char* root, const std::atomic_bool& cancel) {
    check_cancel(cancel);
    need(!loaded() && root != nullptr, "qwen3_asr_load_state");
    Handle handle = impl_->resources->reserve(
        Workload::speech, host_bytes(*impl_->resources, metadata_bytes)
    );
    try {
        auto budget = std::make_shared<checkpoint::MemoryBudget>(metadata_bytes);
        auto cfg = read_qwen3_asr_config(root, budget);

        std::vector<std::string> shard_names;
        try {
            const auto index = checkpoint::json::read(
                root, "model.safetensors.index.json", 4 * 1024 * 1024, budget
            );
            const auto& wm = index.at("weight_map");
            need(wm.kind == checkpoint::json::Value::Kind::object, "qwen3_asr_weight_map");
            std::map<std::string, bool> unique;
            for (const auto& child : wm.children)
                unique[std::string(child.string())] = true;
            for (const auto& [name, _] : unique)
                shard_names.push_back(name);
        } catch (...) {
            shard_names.push_back("model.safetensors");
        }
        need(!shard_names.empty(), "qwen3_asr_no_shards");

        std::vector<std::unique_ptr<checkpoint::Shard>> shards;
        std::unordered_map<std::string, std::size_t> tensor_shard;
        Bytes total_weight_bytes = 0;
        for (std::size_t s = 0; s < shard_names.size(); ++s) {
            check_cancel(cancel);
            auto shard = std::make_unique<checkpoint::Shard>(
                root, shard_names[s], budget
            );
            for (std::size_t i = 0; i < shard->tensor_count(); ++i) {
                const auto info = shard->tensor_at(i);
                need(
                    tensor_shard.emplace(std::string(info.name), s).second,
                    "qwen3_asr_duplicate_tensor"
                );
                total_weight_bytes += info.bytes;
            }
            shards.push_back(std::move(shard));
        }

        // Verify required tensors exist with expected dimensions (header-only).
        auto expect_tensor = [&](const std::string& name,
                                 std::initializer_list<std::uint64_t> shape) {
            const auto it = tensor_shard.find(name);
            need(it != tensor_shard.end(), "qwen3_asr_missing_tensor");
            const auto info = shards[it->second]->tensor(name);
            need(info.rank == shape.size(), "qwen3_asr_tensor_rank");
            std::size_t idx = 0;
            for (auto dim : shape)
                need(info.shape[idx++] == dim, "qwen3_asr_tensor_shape");
        };
        const auto& a = cfg.audio;
        const auto& t = cfg.text;
        const std::size_t f_after =
            (((a.num_mel_bins + 1) / 2 + 1) / 2 + 1) / 2;
        expect_tensor("thinker.audio_tower.conv2d1.weight", {a.downsample_hidden_size, 1, 3, 3});
        expect_tensor("thinker.audio_tower.conv2d1.bias", {a.downsample_hidden_size});
        expect_tensor("thinker.audio_tower.conv2d2.weight", {a.downsample_hidden_size, a.downsample_hidden_size, 3, 3});
        expect_tensor("thinker.audio_tower.conv2d2.bias", {a.downsample_hidden_size});
        expect_tensor("thinker.audio_tower.conv2d3.weight", {a.downsample_hidden_size, a.downsample_hidden_size, 3, 3});
        expect_tensor("thinker.audio_tower.conv2d3.bias", {a.downsample_hidden_size});
        expect_tensor("thinker.audio_tower.conv_out.weight", {a.d_model, a.downsample_hidden_size * f_after});
        for (std::size_t l = 0; l < a.encoder_layers; ++l) {
            const std::string p = "thinker.audio_tower.layers." + std::to_string(l);
            expect_tensor(p + ".self_attn_layer_norm.weight", {a.d_model});
            expect_tensor(p + ".self_attn_layer_norm.bias", {a.d_model});
            expect_tensor(p + ".self_attn.q_proj.weight", {a.d_model, a.d_model});
            expect_tensor(p + ".self_attn.q_proj.bias", {a.d_model});
            expect_tensor(p + ".self_attn.k_proj.weight", {a.d_model, a.d_model});
            expect_tensor(p + ".self_attn.k_proj.bias", {a.d_model});
            expect_tensor(p + ".self_attn.v_proj.weight", {a.d_model, a.d_model});
            expect_tensor(p + ".self_attn.v_proj.bias", {a.d_model});
            expect_tensor(p + ".self_attn.out_proj.weight", {a.d_model, a.d_model});
            expect_tensor(p + ".self_attn.out_proj.bias", {a.d_model});
            expect_tensor(p + ".final_layer_norm.weight", {a.d_model});
            expect_tensor(p + ".final_layer_norm.bias", {a.d_model});
            expect_tensor(p + ".fc1.weight", {a.encoder_ffn_dim, a.d_model});
            expect_tensor(p + ".fc1.bias", {a.encoder_ffn_dim});
            expect_tensor(p + ".fc2.weight", {a.d_model, a.encoder_ffn_dim});
            expect_tensor(p + ".fc2.bias", {a.d_model});
        }
        expect_tensor("thinker.audio_tower.ln_post.weight", {a.d_model});
        expect_tensor("thinker.audio_tower.ln_post.bias", {a.d_model});
        expect_tensor("thinker.audio_tower.proj1.weight", {a.d_model, a.d_model});
        expect_tensor("thinker.audio_tower.proj1.bias", {a.d_model});
        expect_tensor("thinker.audio_tower.proj2.weight", {a.output_dim, a.d_model});
        expect_tensor("thinker.audio_tower.proj2.bias", {a.output_dim});

        expect_tensor("thinker.model.embed_tokens.weight", {t.vocab_size, t.hidden_size});
        expect_tensor("thinker.lm_head.weight", {t.vocab_size, t.hidden_size});
        expect_tensor("thinker.model.norm.weight", {t.hidden_size});
        const std::size_t q_dim = t.num_attention_heads * t.head_dim;
        const std::size_t kv_dim = t.num_key_value_heads * t.head_dim;
        for (std::size_t l = 0; l < t.num_hidden_layers; ++l) {
            const std::string p = "thinker.model.layers." + std::to_string(l);
            expect_tensor(p + ".input_layernorm.weight", {t.hidden_size});
            expect_tensor(p + ".post_attention_layernorm.weight", {t.hidden_size});
            expect_tensor(p + ".self_attn.q_proj.weight", {q_dim, t.hidden_size});
            expect_tensor(p + ".self_attn.k_proj.weight", {kv_dim, t.hidden_size});
            expect_tensor(p + ".self_attn.v_proj.weight", {kv_dim, t.hidden_size});
            expect_tensor(p + ".self_attn.o_proj.weight", {t.hidden_size, q_dim});
            expect_tensor(p + ".self_attn.q_norm.weight", {t.head_dim});
            expect_tensor(p + ".self_attn.k_norm.weight", {t.head_dim});
            expect_tensor(p + ".mlp.gate_proj.weight", {t.intermediate_size, t.hidden_size});
            expect_tensor(p + ".mlp.up_proj.weight", {t.intermediate_size, t.hidden_size});
            expect_tensor(p + ".mlp.down_proj.weight", {t.hidden_size, t.intermediate_size});
        }

        auto pos_emb =
            sinusoids_position_embedding(a.max_source_positions, a.d_model);
        std::vector<float> inv_freq(t.head_dim / 2);
        math::inverse_frequencies(t.rope_theta, inv_freq);

        if (impl_->compute) {
            // Reserve modest device state for KV cache mirroring during decode.
            const Bytes kv_state =
                Bytes(t.num_hidden_layers) * 2 * 1024 * kv_dim * sizeof(float);
            impl_->compute->reserve_device_state(kv_state);
            impl_->compute->prepare_weights(total_weight_bytes);
        }

        check_cancel(cancel);
        impl_->resources->loaded(handle);
        impl_->config = std::move(cfg);
        impl_->budget = std::move(budget);
        impl_->shards = std::move(shards);
        impl_->tensor_shard = std::move(tensor_shard);
        impl_->pos_emb = std::move(pos_emb);
        impl_->inv_freq = std::move(inv_freq);
        impl_->resident = handle;
    } catch (...) {
        impl_->resources->released(handle);
        throw;
    }
}

void Qwen3Asr::unload() {
    if (!impl_->resident)
        return;
    impl_->resources->begin_eviction(impl_->resident);
    if (impl_->compute) {
        impl_->compute->release_scratch(false);
        impl_->compute->release_devices();
    }
    impl_->shards.clear();
    impl_->tensor_shard.clear();
    impl_->pos_emb.clear();
    impl_->inv_freq.clear();
    impl_->budget.reset();
    impl_->resources->released(impl_->resident);
    impl_->resident = 0;
}

Qwen3AsrResult Qwen3Asr::transcribe(
    std::span<const float> mel,
    std::size_t mel_frames,
    text::Tokenizer& tokenizer,
    std::string_view forced_language,
    const std::atomic_bool& cancel,
    const std::function<void(const char*, std::size_t)>& progress
) {
    check_cancel(cancel);
    need(loaded(), "qwen3_asr_not_loaded");
    BusyGuard busy(impl_->busy);
    PinGuard pin(*impl_->resources, impl_->resident);
    ScratchGuard scratch(impl_->compute.get());

    const auto& t = impl_->config.text;

    // 1. Encode audio frames into [audio_tokens, hidden_size] embeddings.
    auto audio_embeds = impl_->encode_audio(mel, mel_frames, cancel, progress);
    const std::size_t audio_tokens = audio_embeds.values.size() / t.hidden_size;

    // 2. Build prompt token IDs using the official Qwen3-ASR chat template:
    //    <|im_start|>system\n<|im_end|>\n
    //    <|im_start|>user\n<|audio_start|><|audio_pad|>*N<|audio_end|><|im_end|>\n
    //    <|im_start|>assistant\n[language X<asr_text>]
    const auto im_start = tokenizer.added_token_id("<|im_start|>").value_or(151644);
    const auto im_end = tokenizer.added_token_id("<|im_end|>").value_or(151645);
    const auto asr_text = tokenizer.added_token_id("<asr_text>").value_or(151704);

    auto encode_str = [&](const std::string& text) {
        std::vector<std::uint32_t> buf(text.size() + 16);
        const auto n = tokenizer.encode(text, buf, cancel);
        buf.resize(n);
        return buf;
    };

    std::vector<std::uint32_t> prompt;
    auto append = [&](const std::vector<std::uint32_t>& ids) {
        prompt.insert(prompt.end(), ids.begin(), ids.end());
    };
    prompt.push_back(im_start);
    append(encode_str("system\n"));
    prompt.push_back(im_end);
    append(encode_str("\n"));
    prompt.push_back(im_start);
    append(encode_str("user\n"));
    prompt.push_back(impl_->config.audio_start_token_id);
    const std::size_t audio_start_idx = prompt.size();
    prompt.insert(prompt.end(), audio_tokens, impl_->config.audio_token_id);
    prompt.push_back(impl_->config.audio_end_token_id);
    prompt.push_back(im_end);
    append(encode_str("\n"));
    prompt.push_back(im_start);
    append(encode_str("assistant\n"));
    if (!forced_language.empty()) {
        append(encode_str("language " + std::string(forced_language)));
        prompt.push_back(asr_text);
    }

    const std::size_t prompt_len = prompt.size();
    const std::size_t max_seq = prompt_len + max_new_tokens;
    need(max_seq <= t.max_position_embeddings, "qwen3_asr_context_limit");

    // 3. Allocate KV cache [layers, max_seq, kv_dim] for keys and values.
    const std::size_t q_dim = t.num_attention_heads * t.head_dim;
    const std::size_t kv_dim = t.num_key_value_heads * t.head_dim;
    const std::size_t layer_kv = max_seq * kv_dim;
    FloatBuffer k_cache(*impl_->resources, t.num_hidden_layers * layer_kv);
    FloatBuffer v_cache(*impl_->resources, t.num_hidden_layers * layer_kv);

    // 4. Prefill embeddings: lookup text tokens, then splice audio_embeds.
    FloatBuffer hidden(*impl_->resources, prompt_len * t.hidden_size);
    for (std::size_t i = 0; i < prompt_len; ++i) {
        check_cancel(cancel);
        auto dst = std::span<float>(
            hidden.values.data() + i * t.hidden_size, t.hidden_size
        );
        if (i >= audio_start_idx && i < audio_start_idx + audio_tokens) {
            std::copy_n(
                audio_embeds.values.data() + (i - audio_start_idx) * t.hidden_size,
                t.hidden_size,
                dst.data()
            );
        } else {
            impl_->embed_token(prompt[i], dst, cancel);
        }
    }

    // Working buffers sized for max(prompt_len, 1) rows; reused across steps.
    FloatBuffer norm(*impl_->resources, prompt_len * t.hidden_size);
    FloatBuffer q_buf(*impl_->resources, prompt_len * q_dim);
    FloatBuffer k_buf(*impl_->resources, prompt_len * kv_dim);
    FloatBuffer v_buf(*impl_->resources, prompt_len * kv_dim);
    FloatBuffer attn_buf(*impl_->resources, prompt_len * q_dim);
    FloatBuffer gate_buf(*impl_->resources, prompt_len * t.intermediate_size);
    FloatBuffer up_buf(*impl_->resources, prompt_len * t.intermediate_size);
    FloatBuffer logits(*impl_->resources, t.vocab_size);

    const float attn_scale = 1.0f / std::sqrt(static_cast<float>(t.head_dim));
    const math::Rotary q_rope{
        t.num_attention_heads, t.head_dim, t.head_dim, t.max_position_embeddings
    };
    const math::Rotary k_rope{
        t.num_key_value_heads, t.head_dim, t.head_dim, t.max_position_embeddings
    };

    auto forward_step = [&](std::size_t rows, std::size_t position) -> std::uint32_t {
        auto h_span = std::span<float>(hidden.values.data(), rows * t.hidden_size);
        auto n_span = std::span<float>(norm.values.data(), rows * t.hidden_size);
        auto q_span = std::span<float>(q_buf.values.data(), rows * q_dim);
        auto k_span = std::span<float>(k_buf.values.data(), rows * kv_dim);
        auto v_span = std::span<float>(v_buf.values.data(), rows * kv_dim);
        auto a_span = std::span<float>(attn_buf.values.data(), rows * q_dim);
        auto g_span = std::span<float>(gate_buf.values.data(), rows * t.intermediate_size);
        auto u_span = std::span<float>(up_buf.values.data(), rows * t.intermediate_size);

        for (std::size_t l = 0; l < t.num_hidden_layers; ++l) {
            check_cancel(cancel);
            const std::string p = "thinker.model.layers." + std::to_string(l);
            {
                const auto w = impl_->read_vec(p + ".input_layernorm.weight", t.hidden_size, cancel);
                math::rms_norm(
                    {rows, t.hidden_size, t.rms_norm_eps, math::NormScale::direct},
                    h_span,
                    w,
                    n_span
                );
            }
            impl_->project(p + ".self_attn.q_proj.weight", {}, n_span, t.hidden_size, q_dim, q_span, cancel);
            impl_->project(p + ".self_attn.k_proj.weight", {}, n_span, t.hidden_size, kv_dim, k_span, cancel);
            impl_->project(p + ".self_attn.v_proj.weight", {}, n_span, t.hidden_size, kv_dim, v_span, cancel);

            {
                const auto qw = impl_->read_vec(p + ".self_attn.q_norm.weight", t.head_dim, cancel);
                const auto kw = impl_->read_vec(p + ".self_attn.k_norm.weight", t.head_dim, cancel);
                math::rms_norm(
                    {rows * t.num_attention_heads, t.head_dim, t.rms_norm_eps, math::NormScale::direct},
                    q_span,
                    qw,
                    q_span
                );
                math::rms_norm(
                    {rows * t.num_key_value_heads, t.head_dim, t.rms_norm_eps, math::NormScale::direct},
                    k_span,
                    kw,
                    k_span
                );
            }
            for (std::size_t r = 0; r < rows; ++r) {
                auto qr = std::span<float>(q_span.data() + r * q_dim, q_dim);
                auto kr = std::span<float>(k_span.data() + r * kv_dim, kv_dim);
                math::rope(q_rope, position + r, impl_->inv_freq, qr, qr);
                math::rope(k_rope, position + r, impl_->inv_freq, kr, kr);
            }

            float* lk = k_cache.values.data() + l * layer_kv;
            float* lv = v_cache.values.data() + l * layer_kv;
            std::copy_n(k_span.data(), rows * kv_dim, lk + position * kv_dim);
            std::copy_n(v_span.data(), rows * kv_dim, lv + position * kv_dim);

            auto lk_span = std::span<const float>(lk, (position + rows) * kv_dim);
            auto lv_span = std::span<const float>(lv, (position + rows) * kv_dim);
            if (!impl_->compute
                || !impl_->compute->attend_cached(
                    l,
                    q_span,
                    lk_span,
                    lv_span,
                    rows,
                    position,
                    t.num_attention_heads,
                    t.num_key_value_heads,
                    t.head_dim,
                    attn_scale,
                    a_span,
                    cancel
                )) {
                cpu_causal_gqa_attention(
                    q_span,
                    lk_span,
                    lv_span,
                    rows,
                    position,
                    t.num_attention_heads,
                    t.num_key_value_heads,
                    t.head_dim,
                    attn_scale,
                    a_span,
                    cancel
                );
            }

            impl_->project(p + ".self_attn.o_proj.weight", {}, a_span, q_dim, t.hidden_size, n_span, cancel);
            add_residual(h_span, n_span);

            {
                const auto w = impl_->read_vec(p + ".post_attention_layernorm.weight", t.hidden_size, cancel);
                math::rms_norm(
                    {rows, t.hidden_size, t.rms_norm_eps, math::NormScale::direct},
                    h_span,
                    w,
                    n_span
                );
            }
            impl_->project(p + ".mlp.gate_proj.weight", {}, n_span, t.hidden_size, t.intermediate_size, g_span, cancel);
            impl_->project(p + ".mlp.up_proj.weight", {}, n_span, t.hidden_size, t.intermediate_size, u_span, cancel);
            math::gate(math::Gate::silu, g_span, u_span, g_span);
            impl_->project(p + ".mlp.down_proj.weight", {}, g_span, t.intermediate_size, t.hidden_size, n_span, cancel);
            add_residual(h_span, n_span);
        }

        // Final norm + lm_head on the last row only.
        auto last_in = std::span<const float>(
            h_span.data() + (rows - 1) * t.hidden_size, t.hidden_size
        );
        auto last_norm = std::span<float>(n_span.data(), t.hidden_size);
        {
            const auto w = impl_->read_vec("thinker.model.norm.weight", t.hidden_size, cancel);
            math::rms_norm(
                {1, t.hidden_size, t.rms_norm_eps, math::NormScale::direct},
                last_in,
                w,
                last_norm
            );
        }
        impl_->project(
            "thinker.lm_head.weight",
            {},
            last_norm,
            t.hidden_size,
            t.vocab_size,
            logits.span(),
            cancel
        );
        const auto best = std::max_element(logits.values.begin(), logits.values.end());
        return static_cast<std::uint32_t>(std::distance(logits.values.begin(), best));
    };

    // Prefill pass -> first generated token.
    std::uint32_t next_token = forward_step(prompt_len, 0);
    std::size_t pos = prompt_len;
    std::string raw_decoded;
    for (std::size_t step = 0; step < max_new_tokens; ++step) {
        check_cancel(cancel);
        if (impl_->is_eos(next_token))
            break;
        raw_decoded += tokenizer.decode(next_token, false);
        if (progress)
            progress("decode", step + 1);
        if (step + 1 == max_new_tokens)
            break;
        auto dst = std::span<float>(hidden.values.data(), t.hidden_size);
        impl_->embed_token(next_token, dst, cancel);
        next_token = forward_step(1, pos);
        ++pos;
    }

    return parse_asr_output(raw_decoded, forced_language);
}

}  // namespace kadan::stt
