#include "kadan/lfm3.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>

#include "kadan/float_weights.hpp"

namespace kadan::lfm3 {
namespace {
constexpr double kPi = 3.14159265358979323846;

void check(bool ok, const char* message) {
    if (!ok)
        throw std::runtime_error(message);
}

void check_cancel(const std::atomic_bool& cancel) {
    if (cancel.load(std::memory_order_relaxed))
        throw std::runtime_error("lfm3_audio_cancelled");
}

double silu_f64(double x) {
    const double e = std::exp(-std::abs(x));
    const double sig = x >= 0.0 ? 1.0 / (1.0 + e) : e / (1.0 + e);
    return x * sig;
}

double gelu_f64(double x) {
    return 0.5 * x * (1.0 + std::erf(x * 0.70710678118654752440));
}

// Slaney mel scale matching transformers.audio_utils.mel_filter_bank(..., mel_scale="slaney", norm="slaney").
double hz_to_mel_slaney(double hz) {
    constexpr double f_sp = 200.0 / 3.0;
    constexpr double min_log_hz = 1000.0;
    constexpr double min_log_mel = min_log_hz / f_sp;
    const double logstep = std::log(6.4) / 27.0;
    if (hz >= min_log_hz)
        return min_log_mel + std::log(hz / min_log_hz) / logstep;
    return hz / f_sp;
}

double mel_to_hz_slaney(double mel) {
    constexpr double f_sp = 200.0 / 3.0;
    constexpr double min_log_hz = 1000.0;
    constexpr double min_log_mel = min_log_hz / f_sp;
    const double logstep = std::log(6.4) / 27.0;
    if (mel >= min_log_mel)
        return min_log_hz * std::exp(logstep * (mel - min_log_mel));
    return mel * f_sp;
}

std::vector<float> slaney_mel_filterbank(
    std::size_t n_fft, std::size_t n_mels, double sample_rate
) {
    const std::size_t n_bins = n_fft / 2 + 1;
    std::vector<double> fft_freqs(n_bins);
    for (std::size_t i = 0; i < n_bins; ++i)
        fft_freqs[i] = (sample_rate * 0.5) * static_cast<double>(i)
                       / static_cast<double>(n_bins - 1);

    const double mel_min = hz_to_mel_slaney(0.0);
    const double mel_max = hz_to_mel_slaney(sample_rate * 0.5);
    std::vector<double> filter_freqs(n_mels + 2);
    for (std::size_t i = 0; i < n_mels + 2; ++i) {
        const double m =
            mel_min + (mel_max - mel_min) * static_cast<double>(i)
                          / static_cast<double>(n_mels + 1);
        filter_freqs[i] = mel_to_hz_slaney(m);
    }

    std::vector<float> fb(n_mels * n_bins, 0.0f);
    for (std::size_t m = 0; m < n_mels; ++m) {
        const double f_left = filter_freqs[m];
        const double f_center = filter_freqs[m + 1];
        const double f_right = filter_freqs[m + 2];
        const double enorm = 2.0 / (f_right - f_left);
        for (std::size_t b = 0; b < n_bins; ++b) {
            const double f = fft_freqs[b];
            const double down = (f - f_left) / (f_center - f_left);
            const double up = (f_right - f) / (f_right - f_center);
            const double w = std::max(0.0, std::min(down, up)) * enorm;
            fb[m * n_bins + b] = static_cast<float>(w);
        }
    }
    return fb;
}

void conv2d_nchw(
    std::span<const float> input,  // [in_c, in_h, in_w]
    std::span<const float> weight,  // [out_c, in_c/groups, k, k]
    std::span<const float> bias,  // [out_c]
    std::size_t in_c,
    std::size_t in_h,
    std::size_t in_w,
    std::size_t out_c,
    std::size_t groups,
    std::size_t k,
    std::size_t stride,
    std::size_t pad,
    std::vector<float>& output,
    std::size_t& out_h,
    std::size_t& out_w
) {
    out_h = (in_h + 2 * pad - k) / stride + 1;
    out_w = (in_w + 2 * pad - k) / stride + 1;
    output.assign(out_c * out_h * out_w, 0.0f);
    const std::size_t in_per_group = in_c / groups;
    const std::size_t out_per_group = out_c / groups;
    for (std::size_t oc = 0; oc < out_c; ++oc) {
        const std::size_t g = oc / out_per_group;
        const std::size_t ic_start = g * in_per_group;
        const float b = bias.empty() ? 0.0f : bias[oc];
        for (std::size_t oy = 0; oy < out_h; ++oy) {
            for (std::size_t ox = 0; ox < out_w; ++ox) {
                double acc = b;
                for (std::size_t ic = 0; ic < in_per_group; ++ic) {
                    const float* in_plane =
                        input.data() + (ic_start + ic) * in_h * in_w;
                    const float* w_plane =
                        weight.data() + (oc * in_per_group + ic) * k * k;
                    for (std::size_t ky = 0; ky < k; ++ky) {
                        const std::ptrdiff_t iy =
                            static_cast<std::ptrdiff_t>(oy * stride + ky)
                            - static_cast<std::ptrdiff_t>(pad);
                        if (iy < 0 || iy >= static_cast<std::ptrdiff_t>(in_h))
                            continue;
                        for (std::size_t kx = 0; kx < k; ++kx) {
                            const std::ptrdiff_t ix =
                                static_cast<std::ptrdiff_t>(ox * stride + kx)
                                - static_cast<std::ptrdiff_t>(pad);
                            if (ix < 0 || ix >= static_cast<std::ptrdiff_t>(in_w))
                                continue;
                            acc += static_cast<double>(
                                       in_plane[static_cast<std::size_t>(iy) * in_w
                                                + static_cast<std::size_t>(ix)]
                                   )
                                   * static_cast<double>(w_plane[ky * k + kx]);
                        }
                    }
                }
                output[(oc * out_h + oy) * out_w + ox] = static_cast<float>(acc);
            }
        }
    }
}

void rms_norm_fp32_scale(
    std::span<const float> input,
    std::span<const float> weight,
    std::size_t rows,
    std::size_t width,
    float eps,
    std::span<float> output
) {
    // Matches Lfm3AudioRMSNorm: (x * rsqrt(mean(x^2) + eps) * weight)
    rms_norm_lfm3(input, weight, rows, width, eps, output);
}

std::uint64_t next_rand(std::uint64_t& state) {
    std::uint64_t z = (state += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

bool shard_has(const checkpoint::Shard& shard, std::string_view name) {
    try {
        (void)shard.tensor(name);
        return true;
    } catch (...) {
        return false;
    }
}
}  // namespace

std::size_t lfm3_audio_encoder_tokens(std::size_t num_frames, std::size_t stride) {
    return (num_frames + stride - 1) / stride;
}

MelSpectrogram extract_lfm3_audio_features(
    std::span<const float> pcm_16k, const AudioFeatureConfig& config
) {
    check(!pcm_16k.empty(), "lfm3_audio_empty_pcm");
    std::size_t padded = config.n_fft;
    if (pcm_16k.size() >= config.n_fft) {
        const std::size_t rem = (pcm_16k.size() - config.n_fft) % config.hop_length;
        padded = pcm_16k.size() + (rem ? (config.hop_length - rem) : 0);
    }
    std::vector<float> samples(padded, 0.0f);
    std::copy(pcm_16k.begin(), pcm_16k.end(), samples.begin());

    const std::size_t num_frames = 1 + (padded - config.n_fft) / config.hop_length;
    const std::size_t n_bins = config.n_fft / 2 + 1;

    // Periodic Hann window matching torch.hann_window(win_length).
    std::vector<double> window(config.win_length);
    for (std::size_t i = 0; i < config.win_length; ++i) {
        window[i] = 0.5 * (1.0 - std::cos(2.0 * kPi * static_cast<double>(i)
                                          / static_cast<double>(config.win_length)));
    }
    // Precompute DFT trig table for n_fft=400 (fast, exact, no FFTW dependency).
    std::vector<double> cos_t(n_bins * config.n_fft);
    std::vector<double> sin_t(n_bins * config.n_fft);
    for (std::size_t k = 0; k < n_bins; ++k) {
        const double factor = -2.0 * kPi * static_cast<double>(k)
                              / static_cast<double>(config.n_fft);
        for (std::size_t n = 0; n < config.n_fft; ++n) {
            cos_t[k * config.n_fft + n] = std::cos(factor * static_cast<double>(n));
            sin_t[k * config.n_fft + n] = std::sin(factor * static_cast<double>(n));
        }
    }
    const auto mel_fb = slaney_mel_filterbank(
        config.n_fft, config.n_mels, static_cast<double>(config.sample_rate)
    );

    MelSpectrogram out;
    out.n_mels = config.n_mels;
    out.num_frames = num_frames;
    out.data.resize(config.n_mels * num_frames);

    std::vector<double> windowed(config.n_fft);
    std::vector<float> power(n_bins);
    for (std::size_t f = 0; f < num_frames; ++f) {
        const std::size_t base = f * config.hop_length;
        for (std::size_t n = 0; n < config.n_fft; ++n)
            windowed[n] = static_cast<double>(samples[base + n]) * window[n];
        for (std::size_t k = 0; k < n_bins; ++k) {
            const double* ct = cos_t.data() + k * config.n_fft;
            const double* st = sin_t.data() + k * config.n_fft;
            double re = 0.0, im = 0.0;
            for (std::size_t n = 0; n < config.n_fft; ++n) {
                re += windowed[n] * ct[n];
                im += windowed[n] * st[n];
            }
            power[k] = static_cast<float>(re * re + im * im);
        }
        for (std::size_t m = 0; m < config.n_mels; ++m) {
            const float* fb = mel_fb.data() + m * n_bins;
            double mel_e = 0.0;
            for (std::size_t k = 0; k < n_bins; ++k)
                mel_e += static_cast<double>(fb[k]) * static_cast<double>(power[k]);
            double lm = mel_e > 0.0 ? std::log10(mel_e) : static_cast<double>(config.log_floor);
            if (lm < static_cast<double>(config.log_floor))
                lm = static_cast<double>(config.log_floor);
            out.data[m * num_frames + f] = static_cast<float>(
                (lm + static_cast<double>(config.norm_offset))
                / static_cast<double>(config.norm_divisor)
            );
        }
    }
    return out;
}

LiquidCodecConfig LiquidCodecConfig::parse(const Json& doc) {
    LiquidCodecConfig c;
    c.encoder_dim = doc.value("encoder_dim", c.encoder_dim);
    c.stft_n_fft = doc.value("stft_n_fft", c.stft_n_fft);
    c.stft_hop_length = doc.value("stft_hop_length", c.stft_hop_length);
    c.stft_win_length = doc.value("stft_win_length", c.stft_win_length);
    c.stft_frame_stack = doc.value("stft_frame_stack", c.stft_frame_stack);
    c.quantizer_dim = doc.value("quantizer_dim", c.quantizer_dim);
    c.n_q = doc.value("n_q", c.n_q);
    c.decoder_window = doc.value("decoder_window", c.decoder_window);
    c.sampling_rate = doc.value("sampling_rate", c.sampling_rate);
    if (doc.contains("fsq_levels") && doc.at("fsq_levels").is_array())
        c.fsq_levels = doc.at("fsq_levels").get<std::vector<int>>();
    return c;
}

std::size_t LiquidCodecConfig::codebook_size() const {
    std::size_t prod = 1;
    for (int l : fsq_levels)
        prod *= static_cast<std::size_t>(l);
    return prod;
}

LiquidCodecDecoder::LiquidCodecDecoder(
    LiquidCodecConfig config,
    std::shared_ptr<checkpoint::Shard> shard,
    std::shared_ptr<Resources> resources,
    std::shared_ptr<DenseCompute> compute
)
    : config_(std::move(config)), shard_(std::move(shard)),
      resources_(std::move(resources)), compute_(std::move(compute)) {
    check(bool(shard_) && bool(resources_), "liquid_codec_deps");
    basis_.resize(config_.fsq_levels.size());
    int prod = 1;
    for (std::size_t i = 0; i < config_.fsq_levels.size(); ++i) {
        basis_[i] = prod;
        prod *= config_.fsq_levels[i];
    }
    hann_window_.resize(config_.stft_win_length);
    for (std::size_t i = 0; i < config_.stft_win_length; ++i) {
        hann_window_[i] = static_cast<float>(
            0.5 * (1.0 - std::cos(2.0 * kPi * static_cast<double>(i)
                                  / static_cast<double>(config_.stft_win_length)))
        );
    }
    auto dec_cfg = BackboneConfig::liquid_codec_decoder_config(
        config_.encoder_dim, config_.decoder_window
    );
    decoder_lfm_ = std::make_unique<Backbone>(
        dec_cfg,
        shard_,
        "decoder.lfm.",
        resources_,
        Workload::lfm3_audio,
        compute_,
        false,
        false
    );
}

std::vector<float> LiquidCodecDecoder::decode(
    std::span<const std::int32_t> codes,
    std::size_t num_frames,
    const std::atomic_bool& cancel
) const {
    if (num_frames == 0)
        return {};
    const std::size_t n_q = config_.n_q;
    const std::size_t cb_size = config_.codebook_size();
    const std::size_t cdim = config_.fsq_levels.size();
    const std::size_t eff_dim = n_q * cdim;
    check(codes.size() == num_frames * n_q, "liquid_codec_codes_shape");

    // 1. FSQ quantizer decode: [num_frames, eff_dim]
    std::vector<float> fsq_latents(num_frames * eff_dim);
    for (std::size_t f = 0; f < num_frames; ++f) {
        for (std::size_t q = 0; q < n_q; ++q) {
            const int code = codes[f * n_q + q];
            check(code >= 0 && static_cast<std::size_t>(code) < cb_size, "liquid_codec_code_range");
            for (std::size_t d = 0; d < cdim; ++d) {
                const int level = config_.fsq_levels[d];
                const int half_w = level / 2;
                const int digit = (code / basis_[d]) % level;
                fsq_latents[f * eff_dim + q * cdim + d] =
                    static_cast<float>(digit - half_w) / static_cast<float>(half_w);
            }
        }
    }

    // 2. Optional quantizer.project_out -> [num_frames, quantizer_dim]
    std::vector<float> q_out;
    const std::string q_proj_w = shard_has(*shard_, "quantizer.project_out.weight")
                                     ? "quantizer.project_out.weight"
                                     : "quantizer.fsq.project_out.weight";
    const std::string q_proj_b = shard_has(*shard_, "quantizer.project_out.bias")
                                     ? "quantizer.project_out.bias"
                                     : "quantizer.fsq.project_out.bias";
    if (config_.quantizer_dim != eff_dim && shard_has(*shard_, q_proj_w)) {
        q_out.resize(num_frames * config_.quantizer_dim);
        auto b = checkpoint::float_slice(
            *resources_, Workload::lfm3_audio, *shard_, q_proj_b, 0, config_.quantizer_dim, cancel
        );
        checkpoint::project_source(
            *resources_, Workload::lfm3_audio, compute_.get(), *shard_,
            q_proj_w, b, fsq_latents, eff_dim, config_.quantizer_dim, q_out, cancel
        );
    } else {
        q_out = std::move(fsq_latents);
    }

    // 3. from_quantizer -> [num_frames, encoder_dim]
    std::vector<float> embeddings(num_frames * config_.encoder_dim);
    {
        auto b = checkpoint::float_slice(
            *resources_, Workload::lfm3_audio, *shard_, "from_quantizer.bias", 0,
            config_.encoder_dim, cancel
        );
        checkpoint::project_source(
            *resources_, Workload::lfm3_audio, compute_.get(), *shard_,
            "from_quantizer.weight", b, q_out, config_.quantizer_dim,
            config_.encoder_dim, embeddings, cancel
        );
    }

    // 4. Nearest-exact upsample by stft_frame_stack (6x) -> [num_frames * 6, encoder_dim]
    const std::size_t stft_frames = num_frames * config_.stft_frame_stack;
    std::vector<float> upsampled(stft_frames * config_.encoder_dim);
    for (std::size_t f = 0; f < num_frames; ++f) {
        const float* src = embeddings.data() + f * config_.encoder_dim;
        for (std::size_t s = 0; s < config_.stft_frame_stack; ++s) {
            std::copy_n(
                src,
                config_.encoder_dim,
                upsampled.data() + (f * config_.stft_frame_stack + s) * config_.encoder_dim
            );
        }
    }

    // 5. Decoder 8-layer Lfm3Model -> [stft_frames, encoder_dim]
    std::vector<float> dec_hidden(stft_frames * config_.encoder_dim);
    decoder_lfm_->forward(upsampled, stft_frames, dec_hidden, nullptr, nullptr, cancel, 512);

    // 6. decoder.lin -> [stft_frames, 2 * n_bins] (log_magnitude, angle)
    const std::size_t n_bins = config_.stft_n_fft / 2 + 1;
    std::vector<float> spec_params(stft_frames * 2 * n_bins);
    {
        auto b = checkpoint::float_slice(
            *resources_, Workload::lfm3_audio, *shard_, "decoder.lin.bias", 0,
            2 * n_bins, cancel
        );
        checkpoint::project_source(
            *resources_, Workload::lfm3_audio, compute_.get(), *shard_,
            "decoder.lin.weight", b, dec_hidden, config_.encoder_dim,
            2 * n_bins, spec_params, cancel
        );
    }

    // 7. ISTFT: polar(exp(log_mag), angle) -> IRFFT (norm="backward") -> Hann overlap-add.
    std::vector<float> synth_window = hann_window_;
    if (shard_has(*shard_, "decoder.istft.window")) {
        auto w = checkpoint::float_slice(
            *resources_, Workload::lfm3_audio, *shard_, "decoder.istft.window", 0,
            config_.stft_win_length, cancel
        );
        synth_window.assign(w.begin(), w.end());
    }
    const std::size_t n_fft = config_.stft_n_fft;
    const std::size_t hop = config_.stft_hop_length;
    const std::size_t win = config_.stft_win_length;
    const std::size_t full_len = (stft_frames - 1) * hop + win;
    std::vector<double> audio_acc(full_len, 0.0);
    std::vector<double> env_acc(full_len, 0.0);

    // Precompute IRFFT cosine/sine table for n_fft=1280.
    std::vector<double> cos_t(n_fft * n_bins);
    std::vector<double> sin_t(n_fft * n_bins);
    for (std::size_t n = 0; n < n_fft; ++n) {
        const double factor =
            2.0 * kPi * static_cast<double>(n) / static_cast<double>(n_fft);
        for (std::size_t k = 0; k < n_bins; ++k) {
            cos_t[n * n_bins + k] = std::cos(factor * static_cast<double>(k));
            sin_t[n * n_bins + k] = std::sin(factor * static_cast<double>(k));
        }
    }

    std::vector<double> re(n_bins), im(n_bins);
    for (std::size_t f = 0; f < stft_frames; ++f) {
        if ((f & 7) == 0)
            check_cancel(cancel);
        const float* row = spec_params.data() + f * 2 * n_bins;
        for (std::size_t k = 0; k < n_bins; ++k) {
            const double mag = std::exp(static_cast<double>(row[k]));
            const double ang = static_cast<double>(row[n_bins + k]);
            re[k] = mag * std::cos(ang);
            im[k] = mag * std::sin(ang);
        }
        const std::size_t base = f * hop;
        for (std::size_t n = 0; n < win; ++n) {
            const double* ct = cos_t.data() + n * n_bins;
            const double* st = sin_t.data() + n * n_bins;
            // Real symmetric IRFFT: (X[0] + X[N/2]*(-1)^n + 2 * sum_{k=1..N/2-1} Re(X[k] e^{i2pikn/N})) / N
            double val = re[0] + re[n_bins - 1] * ((n & 1) ? -1.0 : 1.0);
            for (std::size_t k = 1; k + 1 < n_bins; ++k)
                val += 2.0 * (re[k] * ct[k] - im[k] * st[k]);
            val /= static_cast<double>(n_fft);
            const double w = static_cast<double>(synth_window[n]);
            audio_acc[base + n] += val * w;
            env_acc[base + n] += w * w;
        }
    }

    const std::size_t trim = (win - hop) / 2;
    check(full_len > 2 * trim, "liquid_codec_trim_underflow");
    const std::size_t out_len = full_len - 2 * trim;
    std::vector<float> samples(out_len);
    for (std::size_t i = 0; i < out_len; ++i) {
        const double env = env_acc[trim + i];
        samples[i] = env > 1e-12 ? static_cast<float>(audio_acc[trim + i] / env) : 0.0f;
    }
    return samples;
}

std::string encode_wav_pcm16_base64(
    std::span<const float> samples, std::uint32_t sample_rate
) {
    const std::uint32_t data_bytes = static_cast<std::uint32_t>(samples.size() * 2);
    const std::uint32_t riff_size = 36 + data_bytes;
    const std::uint32_t byte_rate = sample_rate * 2;
    std::string wav;
    wav.reserve(44 + data_bytes);
    auto push_u16 = [&](std::uint16_t v) {
        wav.push_back(static_cast<char>(v & 0xFF));
        wav.push_back(static_cast<char>((v >> 8) & 0xFF));
    };
    auto push_u32 = [&](std::uint32_t v) {
        wav.push_back(static_cast<char>(v & 0xFF));
        wav.push_back(static_cast<char>((v >> 8) & 0xFF));
        wav.push_back(static_cast<char>((v >> 16) & 0xFF));
        wav.push_back(static_cast<char>((v >> 24) & 0xFF));
    };
    wav.append("RIFF", 4);
    push_u32(riff_size);
    wav.append("WAVEfmt ", 8);
    push_u32(16);  // PCM fmt chunk size
    push_u16(1);   // PCM
    push_u16(1);   // mono
    push_u32(sample_rate);
    push_u32(byte_rate);
    push_u16(2);   // block align
    push_u16(16);  // bits per sample
    wav.append("data", 4);
    push_u32(data_bytes);
    for (float s : samples) {
        const float clamped = std::clamp(s, -1.0f, 1.0f);
        const auto pcm = static_cast<std::int16_t>(clamped * 32767.0f);
        push_u16(static_cast<std::uint16_t>(pcm));
    }
    static constexpr char kAlphabet[] =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string b64;
    b64.reserve(((wav.size() + 2) / 3) * 4);
    for (std::size_t i = 0; i < wav.size(); i += 3) {
        const std::uint32_t b0 = static_cast<std::uint8_t>(wav[i]);
        const std::uint32_t b1 =
            (i + 1 < wav.size()) ? static_cast<std::uint8_t>(wav[i + 1]) : 0u;
        const std::uint32_t b2 =
            (i + 2 < wav.size()) ? static_cast<std::uint8_t>(wav[i + 2]) : 0u;
        const std::uint32_t triple = (b0 << 16) | (b1 << 8) | b2;
        b64.push_back(kAlphabet[(triple >> 18) & 0x3Fu]);
        b64.push_back(kAlphabet[(triple >> 12) & 0x3Fu]);
        b64.push_back((i + 1 < wav.size()) ? kAlphabet[(triple >> 6) & 0x3Fu] : '=');
        b64.push_back((i + 2 < wav.size()) ? kAlphabet[triple & 0x3Fu] : '=');
    }
    return b64;
}

AudioModelConfig AudioModelConfig::parse(const Json& doc) {
    AudioModelConfig c;
    c.text_config = BackboneConfig::parse(doc.at("lfm"));
    const Json& enc = doc.at("encoder");
    c.encoder_dim_in = enc.value("dim_in", c.encoder_dim_in);
    c.encoder_dim_conv = enc.value("dim_conv", c.encoder_dim_conv);
    c.encoder_dim_out = enc.value("dim_out", c.encoder_dim_out);
    c.encoder_sliding_window = enc.value("sliding_window", c.encoder_sliding_window);
    c.encoder_lfm_config = BackboneConfig::parse(enc.at("lfm"));
    c.encoder_lfm_config.sliding_window = c.encoder_sliding_window;
    for (auto& lt : c.encoder_lfm_config.layer_types) {
        if (lt == LayerKind::full_attention)
            lt = LayerKind::sliding_attention;
    }
    c.encoder_lfm_config.sliding_rope = c.encoder_lfm_config.full_rope;
    c.depth_config = BackboneConfig::parse(doc.at("depthformer"));
    c.depth_config.is_causal = false;
    c.num_codebooks = doc.value("codebooks", c.num_codebooks);
    c.codebook_size = doc.value("audio_vocab_size", c.codebook_size);
    c.audio_eos_code = c.codebook_size - 1;
    c.maskgit_steps = doc.value("maskgit_steps", c.maskgit_steps);
    c.interleaved_n_text = doc.value("interleaved_n_text", c.interleaved_n_text);
    c.interleaved_n_audio = doc.value("interleaved_n_audio", c.interleaved_n_audio);
    return c;
}

AudioModel::AudioModel(
    AudioModelConfig config,
    std::shared_ptr<checkpoint::Shard> shard,
    std::shared_ptr<Resources> resources,
    std::shared_ptr<DenseCompute> compute
)
    : config_(std::move(config)), shard_(std::move(shard)),
      resources_(std::move(resources)), compute_(std::move(compute)) {
    check(bool(shard_) && bool(resources_), "lfm3_audio_model_deps");
    const std::string lm_prefix =
        shard_has(*shard_, "lfm.model.embed_tokens.weight")
            ? "lfm.model."
        : shard_has(*shard_, "model.language_model.embed_tokens.weight")
            ? "model.language_model."
            : "lfm.";
    language_model_ = std::make_unique<Backbone>(
        config_.text_config,
        shard_,
        lm_prefix,
        resources_,
        Workload::lfm3_audio,
        compute_,
        true,
        true
    );
    const std::string enc_prefix =
        shard_has(*shard_, "encoder.lfm.lfm.layers.0.operator_norm.weight")
            ? "encoder.lfm.lfm."
        : shard_has(*shard_, "model.audio_encoder.lfm.layers.0.operator_norm.weight")
            ? "model.audio_encoder.lfm."
            : "audio_encoder.lfm.";
    encoder_lfm_ = std::make_unique<Backbone>(
        config_.encoder_lfm_config,
        shard_,
        enc_prefix,
        resources_,
        Workload::lfm3_audio,
        compute_,
        false,
        false
    );
    const std::string depth_prefix =
        shard_has(*shard_, "depthformer.model.layers.0.operator_norm.weight")
            ? "depthformer.model."
        : shard_has(*shard_, "model.depth_decoder.model.layers.0.operator_norm.weight")
            ? "model.depth_decoder.model."
            : "depth_decoder.model.";
    depth_lfm_ = std::make_unique<Backbone>(
        config_.depth_config,
        shard_,
        depth_prefix,
        resources_,
        Workload::lfm3_audio,
        compute_,
        false,
        false
    );
}

std::vector<float> AudioModel::encode_audio(
    const MelSpectrogram& mel, const std::atomic_bool& cancel
) const {
    const std::string dp =
        shard_has(*shard_, "encoder.downsample.conv_in.weight")
            ? "encoder.downsample."
        : shard_has(*shard_, "model.audio_encoder.downsample.conv_in.weight")
            ? "model.audio_encoder.downsample."
            : "audio_encoder.downsample.";
    const std::size_t C = config_.encoder_dim_conv;
    std::vector<float> h1, h2a, h2, h3a, h3;
    std::size_t oh = 0, ow = 0;

    // conv_in: [1 -> C, k=3, s=2, p=1] + SiLU
    {
        auto w = checkpoint::float_slice(
            *resources_, Workload::lfm3_audio, *shard_, dp + "conv_in.weight", 0, C * 9, cancel
        );
        auto b = checkpoint::float_slice(
            *resources_, Workload::lfm3_audio, *shard_, dp + "conv_in.bias", 0, C, cancel
        );
        conv2d_nchw(mel.data, w, b, 1, mel.n_mels, mel.num_frames, C, 1, 3, 2, 1, h1, oh, ow);
        for (float& v : h1)
            v = static_cast<float>(silu_f64(v));
    }
    // conv_mid (depthwise k=3, s=2, p=1) -> lin_mid (1x1) + SiLU
    {
        auto w = checkpoint::float_slice(
            *resources_, Workload::lfm3_audio, *shard_, dp + "conv_mid.weight", 0, C * 9, cancel
        );
        auto b = checkpoint::float_slice(
            *resources_, Workload::lfm3_audio, *shard_, dp + "conv_mid.bias", 0, C, cancel
        );
        std::size_t mh = 0, mw = 0;
        conv2d_nchw(h1, w, b, C, oh, ow, C, C, 3, 2, 1, h2a, mh, mw);
        auto lw = checkpoint::float_slice(
            *resources_, Workload::lfm3_audio, *shard_, dp + "lin_mid.weight", 0, C * C, cancel
        );
        auto lb = checkpoint::float_slice(
            *resources_, Workload::lfm3_audio, *shard_, dp + "lin_mid.bias", 0, C, cancel
        );
        conv2d_nchw(h2a, lw, lb, C, mh, mw, C, 1, 1, 1, 0, h2, oh, ow);
        for (float& v : h2)
            v = static_cast<float>(silu_f64(v));
    }
    // conv_end (depthwise k=3, s=2, p=1) -> lin_end (1x1) + SiLU
    {
        auto w = checkpoint::float_slice(
            *resources_, Workload::lfm3_audio, *shard_, dp + "conv_end.weight", 0, C * 9, cancel
        );
        auto b = checkpoint::float_slice(
            *resources_, Workload::lfm3_audio, *shard_, dp + "conv_end.bias", 0, C, cancel
        );
        std::size_t eh = 0, ew = 0;
        conv2d_nchw(h2, w, b, C, oh, ow, C, C, 3, 2, 1, h3a, eh, ew);
        auto lw = checkpoint::float_slice(
            *resources_, Workload::lfm3_audio, *shard_, dp + "lin_end.weight", 0, C * C, cancel
        );
        auto lb = checkpoint::float_slice(
            *resources_, Workload::lfm3_audio, *shard_, dp + "lin_end.bias", 0, C, cancel
        );
        conv2d_nchw(h3a, lw, lb, C, eh, ew, C, 1, 1, 1, 0, h3, oh, ow);
        for (float& v : h3)
            v = static_cast<float>(silu_f64(v));
    }
    // Flatten (C, oh) -> lin_out (1x1 conv -> [ow, dim_out])
    const std::size_t flat_dim = C * oh;
    std::vector<float> flat_t(ow * flat_dim);
    for (std::size_t t = 0; t < ow; ++t) {
        for (std::size_t f = 0; f < flat_dim; ++f)
            flat_t[t * flat_dim + f] = h3[f * ow + t];
    }
    std::vector<float> enc_in(ow * config_.encoder_dim_out);
    {
        auto b = checkpoint::float_slice(
            *resources_, Workload::lfm3_audio, *shard_, dp + "lin_out.bias", 0,
            config_.encoder_dim_out, cancel
        );
        checkpoint::project_source(
            *resources_, Workload::lfm3_audio, compute_.get(), *shard_,
            dp + "lin_out.weight", b, flat_t, flat_dim, config_.encoder_dim_out, enc_in, cancel
        );
    }

    // 17-layer causal LFM2 audio encoder
    std::vector<float> encoded(ow * config_.encoder_dim_out);
    encoder_lfm_->forward(enc_in, ow, encoded, nullptr, nullptr, cancel, 512);

    // Audio projector: LayerNorm -> linear_1 -> GELU -> linear_2
    // Checkpoint uses `mlp.model.{0,1,3}.*` (Sequential) or `audio_projector.*`.
    const bool seq_mlp = shard_has(*shard_, "mlp.model.0.weight");
    const std::string pp = shard_has(*shard_, "model.audio_projector.layer_norm.weight")
                               ? "model.audio_projector."
                               : "audio_projector.";
    const std::string ln_w = seq_mlp ? "mlp.model.0.weight" : (pp + "layer_norm.weight");
    const std::string ln_b = seq_mlp ? "mlp.model.0.bias" : (pp + "layer_norm.bias");
    const std::string l1_w = seq_mlp ? "mlp.model.1.weight" : (pp + "linear_1.weight");
    const std::string l1_b = seq_mlp ? "mlp.model.1.bias" : (pp + "linear_1.bias");
    const std::string l2_w = seq_mlp ? "mlp.model.3.weight" : (pp + "linear_2.weight");
    const std::string l2_b = seq_mlp ? "mlp.model.3.bias" : (pp + "linear_2.bias");
    {
        auto nw = checkpoint::float_slice(
            *resources_, Workload::lfm3_audio, *shard_, ln_w, 0,
            config_.encoder_dim_out, cancel
        );
        auto nb = checkpoint::float_slice(
            *resources_, Workload::lfm3_audio, *shard_, ln_b, 0,
            config_.encoder_dim_out, cancel
        );
        layer_norm_1d(encoded, nw, nb, ow, config_.encoder_dim_out, 1e-5f, encoded);
    }
    const std::size_t H = config_.text_config.hidden_size;
    std::vector<float> mid(ow * H);
    {
        auto b = checkpoint::float_slice(
            *resources_, Workload::lfm3_audio, *shard_, l1_b, 0, H, cancel
        );
        checkpoint::project_source(
            *resources_, Workload::lfm3_audio, compute_.get(), *shard_,
            l1_w, b, encoded, config_.encoder_dim_out, H, mid, cancel
        );
    }
    for (float& v : mid)
        v = static_cast<float>(gelu_f64(v));
    std::vector<float> out(ow * H);
    {
        auto b = checkpoint::float_slice(
            *resources_, Workload::lfm3_audio, *shard_, l2_b, 0, H, cancel
        );
        checkpoint::project_source(
            *resources_, Workload::lfm3_audio, compute_.get(), *shard_,
            l2_w, b, mid, H, H, out, cancel
        );
    }
    return out;
}

std::vector<std::int32_t> AudioModel::sample_audio_frame(
    std::span<const float> lm_hidden,
    float temperature,
    std::size_t top_k,
    float repetition_penalty,
    std::size_t repetition_window,
    std::span<const std::int32_t> past_frames,
    std::size_t past_frame_count,
    std::uint64_t& rng_state,
    const std::atomic_bool& cancel
) const {
    const std::string dp =
        shard_has(*shard_, "depth_linear.weight")
            ? ""
        : shard_has(*shard_, "model.depth_decoder.depth_linear.weight")
            ? "model.depth_decoder."
            : "depth_decoder.";
    const std::size_t n_q = config_.num_codebooks;
    const std::size_t dH = config_.depth_config.hidden_size;
    const std::size_t lmH = config_.text_config.hidden_size;
    check(lm_hidden.size() == lmH, "lfm3_audio_depth_hidden_shape");

    // depth_linear(lm_hidden) -> [n_q, dH]
    std::vector<float> base_in(n_q * dH);
    {
        auto b = checkpoint::float_slice(
            *resources_, Workload::lfm3_audio, *shard_, dp + "depth_linear.bias", 0,
            n_q * dH, cancel
        );
        checkpoint::project_source(
            *resources_, Workload::lfm3_audio, compute_.get(), *shard_,
            dp + "depth_linear.weight", b, lm_hidden, lmH, n_q * dH, base_in, cancel
        );
    }
    auto mask_emb = checkpoint::float_slice(
        *resources_, Workload::lfm3_audio, *shard_, dp + "mask_embedding", 0, dH, cancel
    );

    // Random permutation of [0..n_q-1] split into maskgit_steps passes.
    std::vector<std::size_t> order(n_q);
    std::iota(order.begin(), order.end(), 0);
    for (std::size_t i = n_q; i > 1; --i) {
        const std::size_t j = static_cast<std::size_t>(next_rand(rng_state) % i);
        std::swap(order[i - 1], order[j]);
    }
    const std::size_t steps = std::max<std::size_t>(1, config_.maskgit_steps);
    const std::size_t per_step = (n_q + steps - 1) / steps;

    std::vector<std::int32_t> codes(n_q, 0);
    std::vector<bool> revealed(n_q, false);
    std::vector<float> depth_in(n_q * dH);
    std::vector<float> depth_out(n_q * dH);
    std::vector<float> normed_slot(dH);
    std::vector<float> logits(config_.codebook_size);

    for (std::size_t s = 0; s < n_q; s += per_step) {
        check_cancel(cancel);
        const std::size_t end = std::min(n_q, s + per_step);
        for (std::size_t q = 0; q < n_q; ++q) {
            float* dst = depth_in.data() + q * dH;
            const float* src = base_in.data() + q * dH;
            if (!revealed[q]) {
                for (std::size_t d = 0; d < dH; ++d)
                    dst[d] = src[d] + mask_emb[d];
            } else {
                const std::string ename =
                    dp + "depth_embeddings." + std::to_string(q) + ".embedding.weight";
                std::vector<float> emb(dH);
                shard_->read_float_tensor(
                    ename, static_cast<std::size_t>(codes[q]) * dH, emb, cancel
                );
                for (std::size_t d = 0; d < dH; ++d)
                    dst[d] = src[d] + emb[d];
            }
        }
        depth_lfm_->forward(depth_in, n_q, depth_out, nullptr, nullptr, cancel, n_q);

        SamplingConfig sc;
        sc.do_sample = temperature > 0.0f;
        sc.temperature = temperature;
        sc.top_k = top_k;
        sc.top_p = 1.0f;
        sc.repetition_penalty = repetition_penalty;
        sc.repetition_window = repetition_window;

        for (std::size_t idx = s; idx < end; ++idx) {
            const std::size_t q = order[idx];
            const std::string ep = dp + "depth_embeddings." + std::to_string(q) + ".";
            auto nw = checkpoint::float_slice(
                *resources_, Workload::lfm3_audio, *shard_, ep + "embedding_norm.weight",
                0, dH, cancel
            );
            rms_norm_fp32_scale(
                std::span(depth_out.data() + q * dH, dH),
                nw,
                1,
                dH,
                config_.depth_config.norm_eps,
                normed_slot
            );
            checkpoint::project_source(
                *resources_, Workload::lfm3_audio, compute_.get(), *shard_,
                ep + "embedding.weight", {}, normed_slot, dH, config_.codebook_size,
                logits, cancel
            );
            std::vector<std::uint32_t> cb_hist;
            if (past_frame_count > 0) {
                const std::size_t start_f =
                    (repetition_window > 0 && past_frame_count > repetition_window)
                        ? past_frame_count - repetition_window
                        : 0;
                cb_hist.reserve(past_frame_count - start_f);
                for (std::size_t f = start_f; f < past_frame_count; ++f)
                    cb_hist.push_back(static_cast<std::uint32_t>(past_frames[f * n_q + q]));
            }
            codes[q] = static_cast<std::int32_t>(
                sample_logits(logits, sc, cb_hist, rng_state)
            );
            revealed[q] = true;
        }
    }
    return codes;
}

void AudioModel::embed_audio_frame(
    std::span<const std::int32_t> frame_codes,
    std::span<float> out_hidden,
    const std::atomic_bool& cancel
) const {
    const std::size_t n_q = config_.num_codebooks;
    const std::size_t H = config_.text_config.hidden_size;
    check(frame_codes.size() == n_q && out_hidden.size() == H, "lfm3_audio_embed_shape");
    const std::string table = shard_has(*shard_, "model.codebook_embedding.weight")
                                  ? "model.codebook_embedding.weight"
                                  : "codebook_embedding.weight";
    std::fill(out_hidden.begin(), out_hidden.end(), 0.0f);
    std::vector<float> row(H);
    for (std::size_t q = 0; q < n_q; ++q) {
        check(
            frame_codes[q] >= 0
                && static_cast<std::size_t>(frame_codes[q]) < config_.codebook_size,
            "lfm3_audio_code_range"
        );
        const std::size_t idx =
            q * config_.codebook_size + static_cast<std::size_t>(frame_codes[q]);
        shard_->read_float_tensor(table, idx * H, row, cancel);
        for (std::size_t d = 0; d < H; ++d)
            out_hidden[d] += row[d];
    }
}

}  // namespace kadan::lfm3
