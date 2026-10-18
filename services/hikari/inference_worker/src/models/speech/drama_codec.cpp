// SPDX-License-Identifier: Apache-2.0 AND MIT
// Copyright 2025-2026 SGLang-Omni Team
// Copyright (c) 2023-present, Descript
// Native C++ adaptation of Fish's codec graph vendored in SGLang-Omni:
// https://github.com/sgl-project/sglang-omni/blob/be34dd84ed9e0971f72fe5e3d9ba8245cdad780a/sglang_omni/models/fishaudio_s2_pro/fish_speech/models/dac/modded_dac.py
// https://github.com/sgl-project/sglang-omni/blob/be34dd84ed9e0971f72fe5e3d9ba8245cdad780a/sglang_omni/models/fishaudio_s2_pro/fish_speech/models/dac/rvq.py
// Quantizer lookup/projection and Snake semantics checked against Descript DAC:
// https://github.com/descriptinc/descript-audio-codec/blob/c7cfc5d2647e26471dc394f95846a0830e7bec34/dac/nn/quantize.py
// https://github.com/descriptinc/descript-audio-codec/blob/c7cfc5d2647e26471dc394f95846a0830e7bec34/dac/nn/layers.py
// Changes: bounded native checkpoint projections, worker resource accounting,
// cancellation, and independent rolling-window PCM decoding; no Python runtime.
#include "kadan/drama_codec.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <map>
#include <vector>

#include "kadan/checkpoint.hpp"
#include "kadan/float_weights.hpp"

namespace kadan::drama {
namespace {
void need(bool b, const char* e) {
    if (!b)
        throw std::runtime_error(e);
}

void stop(const std::atomic_bool& c) {
    need(!c.load(), "drama_codec_cancelled");
}

Footprint host(Resources& r, Bytes n) {
    auto f = r.snapshot().capacity;
    std::fill(f.begin(), f.end(), 0);
    f[0] = n;
    return f;
}

struct Lease {
    Resources& r;
    Handle h;

    Lease(Resources& r_, Bytes n) : r(r_), h(r.reserve(Workload::tts, host(r, n))) {
    }

    ~Lease() {
        if (h)
            r.released(h);
    }
};

struct Matrix {
    Lease lease;
    std::size_t rows, cols;
    std::unique_ptr<float[]> data;

    Matrix(Resources& r, std::size_t t, std::size_t c)
        : lease(r, t * c * 4), rows(t), cols(c),
          data(std::make_unique<float[]>(t * c)) {
    }

    std::span<float> span() {
        return {data.get(), rows * cols};
    }
};

using M = std::unique_ptr<Matrix>;

M matrix(Resources& r, std::size_t t, std::size_t c) {
    return std::make_unique<Matrix>(r, t, c);
}

struct Active {
    bool& busy;

    Active(bool& b) : busy(b) {
        need(!b, "busy");
        b = true;
    }

    ~Active() {
        busy = false;
    }
};

struct Pin {
    Resources& r;
    Handle h;

    Pin(Resources& a, Handle b) : r(a), h(b) {
        r.pin(h);
    }

    ~Pin() {
        r.unpin(h);
    }
};

void finite(Matrix& x) {
    for (float v : x.span())
        need(std::isfinite(v), "drama_codec_nonfinite");
}

// Fish s2-pro 1de9996b6be38b745688de084d87a5633f714e4e decoder architecture.
// Weights retain upstream names after folding dim-0 weight normalization.
}  // namespace

struct Codec::Impl {
    Resources& resources;
    std::shared_ptr<DenseCompute> compute;
    std::unique_ptr<Lease> parser;
    std::unique_ptr<checkpoint::Shard> shard;
    Lease metadata;
    std::vector<CodecTensorSpec> specs;
    std::map<std::string, std::size_t, std::less<>> index;
    M weights;

    Impl(Resources& r, std::shared_ptr<DenseCompute> c)
        : resources(r), compute(std::move(c)), metadata(r, 8 * 1024 * 1024),
          specs(codec_tensor_specs()) {
        for (std::size_t i = 0; i < specs.size(); ++i)
            index.emplace(specs[i].name, i);
    }

    checkpoint::FloatSlice
    w(const std::string& p, const std::atomic_bool& cancel) const {
        auto& s = specs.at(index.at(p));
        return checkpoint::float_slice(
            resources, Workload::tts, *shard, p, 0, s.count, cancel
        );
    }

    M linear(
        Resources& r,
        const std::string& p,
        Matrix& x,
        std::size_t out,
        bool bias,
        const std::atomic_bool& cancel
    ) {
        auto y = matrix(r, x.rows, out);
        auto b = checkpoint::FloatSlice(r, Workload::tts, bias ? out : 0);
        if (bias)
            shard->read_float_tensor(p + ".bias", 0, b.span(), cancel);
        checkpoint::project_source(
            r,
            Workload::tts,
            compute.get(),
            *shard,
            p + ".weight",
            b,
            x.span(),
            x.cols,
            out,
            y->span(),
            cancel
        );
        finite(*y);
        return y;
    }

    M conv(
        Resources& r,
        const std::string& p,
        Matrix& x,
        std::size_t out,
        std::size_t kernel,
        std::size_t dilation,
        const std::atomic_bool& cancel,
        bool depth = false
    ) {
        auto y = matrix(r, x.rows, out);
        auto b = w(p + ".conv.bias", cancel);
        if (!depth) {
            constexpr std::size_t tile = 64;
            auto packed = matrix(r, tile, x.cols * kernel);
            for (std::size_t at = 0; at < x.rows; at += tile) {
                stop(cancel);
                const auto count = std::min(tile, x.rows - at);
                for (std::size_t t = 0; t < count; ++t)
                    for (std::size_t c = 0; c < x.cols; ++c)
                        for (std::size_t k = 0; k < kernel; ++k) {
                            const auto source =
                                std::ptrdiff_t(at + t)
                                - std::ptrdiff_t((kernel - 1 - k) * dilation);
                            packed->data[(t * x.cols + c) * kernel + k] =
                                source >= 0 ? x.data[std::size_t(source) * x.cols + c]
                                            : 0;
                        }
                checkpoint::project_source(
                    r,
                    Workload::tts,
                    compute.get(),
                    *shard,
                    p + ".conv.weight",
                    b,
                    packed->span().first(count * x.cols * kernel),
                    x.cols * kernel,
                    out,
                    y->span().subspan(at * out, count * out),
                    cancel
                );
            }
            finite(*y);
            return y;
        }
        auto a = w(p + ".conv.weight", cancel);
        for (std::size_t t = 0; t < x.rows; ++t) {
            stop(cancel);
            for (std::size_t o = 0; o < out; ++o) {
                float value = b[o];
                for (std::size_t c = 0; c < (depth ? 1 : x.cols); ++c)
                    for (std::size_t k = 0; k < kernel; ++k) {
                        const auto source =
                            std::ptrdiff_t(t)
                            - std::ptrdiff_t((kernel - 1 - k) * dilation);
                        if (source >= 0)
                            value +=
                                x.data[std::size_t(source) * x.cols + (depth ? o : c)]
                                * a[(o * (depth ? 1 : x.cols) + c) * kernel + k];
                    }
                y->data[t * out + o] = value;
            }
        }
        finite(*y);
        return y;
    }

    M upsample(
        Resources& r,
        const std::string& p,
        Matrix& x,
        std::size_t out,
        std::size_t kernel,
        std::size_t stride,
        const std::atomic_bool& cancel
    ) {
        auto y = matrix(r, x.rows * stride, out);
        auto b = w(p + ".conv.bias", cancel);
        for (std::size_t t = 0; t < y->rows; ++t)
            std::copy(b.begin(), b.end(), y->data.get() + t * out);
        const auto name = p + ".conv.weight";
        auto source = [&](std::size_t first, std::span<float> target) {
            need(
                first % x.cols == 0 && target.size() % x.cols == 0,
                "drama_codec_transpose_source_shape"
            );
            auto first_row = first / x.cols, rows = target.size() / x.cols;
            auto slice = checkpoint::FloatSlice(r, Workload::tts, rows);
            for (std::size_t c = 0; c < x.cols; ++c) {
                stop(cancel);
                shard->read_float_tensor(
                    name, c * out * kernel + first_row, slice.span(), cancel
                );
                for (std::size_t row = 0; row < rows; ++row)
                    target[row * x.cols + c] = slice[row];
            }
        };
        constexpr std::size_t tile = 64;
        auto projected = matrix(r, tile, out * kernel);
        for (std::size_t at = 0; at < x.rows; at += tile) {
            stop(cancel);
            auto count = std::min(tile, x.rows - at);
            auto input = x.span().subspan(at * x.cols, count * x.cols),
                 output = projected->span().first(count * out * kernel);
            shard->check_unchanged();
            if (!compute
                || compute->dense_source(
                       shard->tensor_identity(name),
                       source,
                       {},
                       input,
                       x.cols,
                       out * kernel,
                       output,
                       cancel
                   ) == ProjectionResult::unsupported) {
                for (std::size_t row = 0; row < out * kernel; row += 32) {
                    auto nr = std::min<std::size_t>(32, out * kernel - row);
                    auto weights =
                        checkpoint::FloatSlice(r, Workload::tts, nr * x.cols);
                    source(row * x.cols, weights.span());
                    for (std::size_t t = 0; t < count; ++t)
                        for (std::size_t o = 0; o < nr; ++o) {
                            float v = 0;
                            for (std::size_t c = 0; c < x.cols; ++c)
                                v += input[t * x.cols + c] * weights[o * x.cols + c];
                            output[t * out * kernel + row + o] = v;
                        }
                }
            }
            shard->check_unchanged();
            for (std::size_t t = 0; t < count; ++t)
                for (std::size_t o = 0; o < out; ++o)
                    for (std::size_t k = 0;
                         k < kernel && (at + t) * stride + k < y->rows;
                         ++k)
                        y->data[((at + t) * stride + k) * out + o] +=
                            projected->data[(t * out + o) * kernel + k];
        }
        finite(*y);
        return y;
    }

    M norm(
        Resources& r,
        const std::string& p,
        Matrix& x,
        bool rms,
        const std::atomic_bool& cancel
    ) {
        auto y = matrix(r, x.rows, x.cols);
        auto a = w(p + ".weight", cancel),
             b = rms ? checkpoint::FloatSlice(r, Workload::tts, 0)
                     : w(p + ".bias", cancel);
        for (std::size_t t = 0; t < x.rows; ++t) {
            stop(cancel);
            double mean = 0, square = 0;
            if (!rms) {
                for (std::size_t c = 0; c < x.cols; ++c)
                    mean += x.data[t * x.cols + c];
                mean /= x.cols;
            }
            for (std::size_t c = 0; c < x.cols; ++c) {
                double v = x.data[t * x.cols + c] - mean;
                square += v * v;
            }
            // Preserve FP32 reciprocal/multiply rounding. Shared math::rms_norm
            // divides and scales in double, changing Fish's operation order.
            const float scale =
                float(1 / std::sqrt(square / x.cols + (rms ? 1e-5 : 1e-6)));
            for (std::size_t c = 0; c < x.cols; ++c)
                y->data[t * x.cols + c] =
                    (x.data[t * x.cols + c] - float(mean)) * scale * a[c]
                    + (rms ? 0 : b[c]);
        }
        finite(*y);
        return y;
    }

    void snake(const std::string& p, Matrix& x, const std::atomic_bool& cancel) {
        auto a = w(p + ".alpha", cancel);
        for (std::size_t t = 0; t < x.rows; ++t) {
            stop(cancel);
            for (std::size_t c = 0; c < x.cols; ++c) {
                float& v = x.data[t * x.cols + c];
                const float sine = std::sin(v * a[c]);
                v += sine * sine / (a[c] + 1e-9f);
            }
        }
        finite(x);
    }

    M attention(
        Resources& r, const std::string& p, Matrix& x, const std::atomic_bool& cancel
    ) {
        constexpr std::size_t width = 1024, heads = 16, head_dim = 64, window = 128;
        auto qkv = linear(r, p + ".wqkv", x, 3 * width, false, cancel);
        auto q = matrix(r, x.rows, width), k = matrix(r, x.rows, width),
             v = matrix(r, x.rows, width), mix = matrix(r, x.rows, width),
             scores = matrix(r, 1, x.rows);
        for (std::size_t t = 0; t < x.rows; ++t) {
            stop(cancel);
            std::copy_n(
                qkv->data.get() + t * 3 * width, width, q->data.get() + t * width
            );
            std::copy_n(
                qkv->data.get() + t * 3 * width + width,
                width,
                k->data.get() + t * width
            );
            std::copy_n(
                qkv->data.get() + t * 3 * width + 2 * width,
                width,
                v->data.get() + t * width
            );
            for (std::size_t h = 0; h < heads; ++h)
                for (std::size_t c = 0; c < head_dim / 2; ++c) {
                    const float angle =
                        float(t) * std::pow(10000.f, -2.f * float(c) / float(head_dim));
                    // Upstream precompute_freqs_cis stores its cache in bfloat16,
                    // including when the decoder is subsequently cast to fp32.
                    auto bf16 = [](float value) {
                        auto bits = std::bit_cast<std::uint32_t>(value);
                        bits += 0x7fff + ((bits >> 16) & 1);
                        return std::bit_cast<float>(bits & 0xffff0000U);
                    };
                    const float co = bf16(std::cos(angle)), si = bf16(std::sin(angle));
                    for (auto* value : {q.get(), k.get()}) {
                        const auto offset = t * width + h * head_dim + c * 2;
                        const float a = value->data[offset],
                                    b = value->data[offset + 1];
                        value->data[offset] = a * co - b * si;
                        value->data[offset + 1] = b * co + a * si;
                    }
                }
        }
        qkv.reset();
        if (!(compute
              && compute->attend(
                  q->span(),
                  k->span(),
                  v->span(),
                  x.rows,
                  x.rows,
                  heads,
                  heads,
                  head_dim,
                  x.rows,
                  std::min(window, x.rows),
                  0,
                  mix->span(),
                  cancel
              ))) {
            for (std::size_t t = 0; t < x.rows; ++t)
                for (std::size_t h = 0; h < heads; ++h) {
                    stop(cancel);
                    const auto first = t + 1 > window ? t + 1 - window : 0;
                    float maximum = -INFINITY;
                    for (std::size_t j = first; j <= t; ++j) {
                        float score = 0;
                        for (std::size_t c = 0; c < head_dim; ++c)
                            score += q->data[t * width + h * head_dim + c]
                                     * k->data[j * width + h * head_dim + c];
                        scores->data[j] = score / std::sqrt(float(head_dim));
                        maximum = std::max(maximum, scores->data[j]);
                    }
                    double sum = 0;
                    for (std::size_t j = first; j <= t; ++j) {
                        scores->data[j] = std::exp(scores->data[j] - maximum);
                        sum += scores->data[j];
                    }
                    need(std::isfinite(sum) && sum > 0, "drama_codec_nonfinite");
                    for (std::size_t c = 0; c < head_dim; ++c) {
                        float value = 0;
                        for (std::size_t j = first; j <= t; ++j)
                            value += float(scores->data[j] / sum)
                                     * v->data[j * width + h * head_dim + c];
                        mix->data[t * width + h * head_dim + c] = value;
                    }
                }
        }
        return linear(r, p + ".wo", *mix, width, false, cancel);
    }

    void block(
        Resources& r, const std::string& p, Matrix& x, const std::atomic_bool& cancel
    ) {
        auto n = norm(r, p + ".attention_norm", x, true, cancel),
             a = attention(r, p + ".attention", *n, cancel);
        auto scale = w(p + ".attention_layer_scale.gamma", cancel);
        for (std::size_t i = 0; i < x.rows * x.cols; ++i)
            x.data[i] += a->data[i] * scale[i % x.cols];
        n = norm(r, p + ".ffn_norm", x, true, cancel);
        a.reset();
        auto gate = linear(r, p + ".feed_forward.w1", *n, 3072, false, cancel),
             up = linear(r, p + ".feed_forward.w3", *n, 3072, false, cancel);
        for (std::size_t i = 0; i < gate->rows * gate->cols; ++i) {
            const double v = gate->data[i];
            gate->data[i] = float(
                                v >= 0 ? v / (1 + std::exp(-v))
                                       : v * std::exp(v) / (1 + std::exp(v))
                            )
                            * up->data[i];
        }
        up.reset();
        n.reset();
        auto down = linear(r, p + ".feed_forward.w2", *gate, 1024, false, cancel);
        auto scale2 = w(p + ".ffn_layer_scale.gamma", cancel);
        for (std::size_t i = 0; i < x.rows * x.cols; ++i)
            x.data[i] += down->data[i] * scale2[i % x.cols];
        finite(x);
    }
};

Codec::Codec(std::shared_ptr<Resources> r, std::shared_ptr<DenseCompute> compute)
    : resources_(std::move(r)), compute_(std::move(compute)) {
    need(bool(resources_), "drama_codec_resources_required");
}

Codec::~Codec() {
    unload();
}

void Codec::load(const char* root, const std::atomic_bool& cancel) {
    Active active(busy_);
    stop(cancel);
    need(!model_, "drama_codec_loaded");
    auto m = std::make_unique<Impl>(*resources_, compute_);
    m->parser = std::make_unique<Lease>(*resources_, 8 * 1024 * 1024);
    m->shard = std::make_unique<checkpoint::Shard>(
        root,
        "model.safetensors",
        std::make_shared<checkpoint::MemoryBudget>(8 * 1024 * 1024),
        checkpoint::Limits{1024 * 1024, 4096, 4096}
    );
    for (const auto& s : m->specs) {
        stop(cancel);
        auto info = m->shard->tensor(s.name);
        need(
            info.dtype == checkpoint::Dtype::fp32 && info.rank == s.shape.size()
                && std::equal(s.shape.begin(), s.shape.end(), info.shape.begin()),
            "drama_codec_tensor_layout"
        );
    }
    need(m->shard->tensor_count() == m->specs.size(), "drama_codec_tensor_count");
    m->weights = matrix(*resources_, 1, 1);
    m->shard->check_unchanged();
    stop(cancel);
    resources_->loaded(m->weights->lease.h);
    model_ = std::move(m);
}

void Codec::unload() {
    need(!busy_, "busy");
    if (model_) {
        resources_->begin_eviction(model_->weights->lease.h);
        model_.reset();
    }
}

std::vector<float> Codec::decode_window(
    std::span<const std::uint32_t> codes, const std::atomic_bool& cancel
) {
    Active active(busy_);
    stop(cancel);
    const auto samples = codec_window_samples(codes);
    need(bool(model_), "drama_codec_not_loaded");
    auto& m = *model_;
    const auto frames = codes.size() / codebooks;
    Pin pin(*resources_, m.weights->lease.h);
    auto quantized = matrix(*resources_, frames, 1024);
    M semantic;
    for (std::size_t group = 0; group < codebooks; ++group) {
        stop(cancel);
        const auto p =
            group == 0 ? std::string("quantizer.semantic_quantizer.quantizers.0")
                       : "quantizer.quantizer.quantizers." + std::to_string(group - 1);
        auto embeddings = matrix(*resources_, frames, 8);
        for (std::size_t t = 0; t < frames; ++t) {
            // Drama samples 4096 logits for every codebook. Fish clamps the
            // residual lookup to its 1024-entry table; retain the original
            // generated token in the caller's frames/history.
            const auto index =
                std::min(codes[t * codebooks + group], group == 0 ? 4095U : 1023U);
            m.shard->read_float_tensor(
                p + ".codebook.weight",
                index * 8,
                embeddings->span().subspan(t * 8, 8),
                cancel
            );
        }
        auto projected = matrix(*resources_, frames, 1024);
        auto bias = m.w(p + ".out_proj.bias", cancel);
        checkpoint::project_source(
            *resources_,
            Workload::tts,
            compute_.get(),
            *m.shard,
            p + ".out_proj.weight",
            bias,
            embeddings->span(),
            8,
            1024,
            projected->span(),
            cancel
        );
        if (group == 0) {
            semantic = std::move(projected);
        } else {
            for (std::size_t i = 0; i < quantized->span().size(); ++i)
                quantized->data[i] += projected->data[i];
        }
    }
    for (std::size_t i = 0; i < quantized->span().size(); ++i)
        quantized->data[i] += semantic->data[i];
    semantic.reset();
    finite(*quantized);
    for (int i = 0; i < 8; ++i)
        m.block(
            *resources_,
            "quantizer.post_module.layers." + std::to_string(i),
            *quantized,
            cancel
        );
    auto latent =
        m.norm(*resources_, "quantizer.post_module.norm", *quantized, true, cancel);
    quantized.reset();
    for (int i = 0; i < 2; ++i) {
        const auto p = "quantizer.upsample." + std::to_string(i);
        auto x = m.upsample(*resources_, p + ".0", *latent, 1024, 2, 2, cancel);
        latent.reset();
        auto depth = m.conv(*resources_, p + ".1.dwconv", *x, 1024, 7, 1, cancel, true);
        auto n = m.norm(*resources_, p + ".1.norm", *depth, false, cancel);
        depth.reset();
        auto expanded = m.linear(*resources_, p + ".1.pwconv1", *n, 4096, true, cancel);
        n.reset();
        for (float& v : expanded->span())
            v = .5f * v * (1 + std::erf(v / float(std::sqrt(2.))));
        auto delta =
            m.linear(*resources_, p + ".1.pwconv2", *expanded, 1024, true, cancel);
        expanded.reset();
        auto gamma = m.w(p + ".1.gamma", cancel);
        for (std::size_t j = 0; j < x->rows * x->cols; ++j)
            x->data[j] += delta->data[j] * gamma[j % x->cols];
        finite(*x);
        latent = std::move(x);
    }
    auto audio = m.conv(*resources_, "decoder.model.0", *latent, 1536, 7, 1, cancel);
    latent.reset();
    constexpr std::array<std::size_t, 4> rates{8, 8, 4, 2};
    constexpr std::array<std::size_t, 3> dilations{1, 3, 9};
    for (std::size_t i = 0; i < rates.size(); ++i) {
        const auto p = "decoder.model." + std::to_string(i + 1) + ".block";
        m.snake(p + ".0", *audio, cancel);
        audio = m.upsample(
            *resources_,
            p + ".1",
            *audio,
            audio->cols / 2,
            2 * rates[i],
            rates[i],
            cancel
        );
        for (std::size_t j = 0; j < dilations.size(); ++j) {
            const auto q = p + "." + std::to_string(j + 2) + ".block";
            auto delta = matrix(*resources_, audio->rows, audio->cols);
            std::copy(audio->span().begin(), audio->span().end(), delta->data.get());
            m.snake(q + ".0", *delta, cancel);
            delta = m.conv(
                *resources_, q + ".1", *delta, delta->cols, 7, dilations[j], cancel
            );
            m.snake(q + ".2", *delta, cancel);
            delta = m.conv(*resources_, q + ".3", *delta, delta->cols, 1, 1, cancel);
            for (std::size_t k = 0; k < audio->rows * audio->cols; ++k)
                audio->data[k] += delta->data[k];
            finite(*audio);
        }
    }
    m.snake("decoder.model.5", *audio, cancel);
    audio = m.conv(*resources_, "decoder.model.6", *audio, 1, 7, 1, cancel);
    need(audio->rows == samples, "drama_codec_output_shape");
    Lease output_lease(*resources_, samples * sizeof(float));
    std::vector<float> pcm(samples);
    for (std::size_t i = 0; i < samples; ++i)
        pcm[i] = std::tanh(audio->data[i]);
    stop(cancel);
    m.shard->check_unchanged();
    return pcm;
}
}  // namespace kadan::drama
