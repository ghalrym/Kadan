#include "kadan/h3_denoiser.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <limits>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include "h3_feed_forward.hpp"
#include "h3_quant_marker.hpp"
#include "kadan/cleanup_error.hpp"

namespace kadan::video {
namespace {
using Shard = checkpoint::Shard;
using Dtype = checkpoint::Dtype;

void require(bool value, const char* why) {
    if (!value)
        throw std::runtime_error(why);
}

void stop(const std::atomic_bool& flag) {
    require(!flag.load(), "h3_denoiser_cancelled");
}

Footprint host(Resources& ledger, Bytes bytes) {
    auto p = ledger.snapshot().capacity;
    std::fill(p.begin(), p.end(), 0);
    p[0] = bytes;
    return p;
}

struct Admission {
    Resources& ledger;
    Handle handle;

    Admission(Resources& r, Bytes n)
        : ledger(r), handle(r.reserve(Workload::video, host(r, n))) {
    }

    ~Admission() {
        if (handle)
            ledger.released(handle);
    }
};

struct Active {
    bool& busy;

    explicit Active(bool& b) : busy(b) {
        require(!b, "busy");
        b = true;
    }

    ~Active() {
        busy = false;
    }
};

void shape(
    Shard& shard,
    const std::string& name,
    std::initializer_list<std::uint64_t> dimensions
) {
    const auto t = shard.tensor(name);
    require(
        t.rank == dimensions.size()
            && std::equal(dimensions.begin(), dimensions.end(), t.shape.begin()),
        "h3_denoiser_shape"
    );
}

void finite(std::span<const float> x) {
    for (float v : x)
        require(std::isfinite(v), "h3_denoiser_nonfinite");
}

void read_dense(
    Shard& shard,
    const std::string& name,
    std::span<float> output,
    const std::atomic_bool& cancel,
    std::size_t first = 0
) {
    const auto t = shard.tensor(name);
    require(
        t.dtype == Dtype::fp32 || t.dtype == Dtype::bf16, "h3_denoiser_dense_dtype"
    );
    shard.read_float_tensor(name, first, output, cancel);
    finite(output);
}

void rotate(std::span<double> values, std::size_t group) {
    require(
        (group == 64 || group == 256) && values.size() % group == 0,
        "h3_denoiser_rotation_shape"
    );
    for (std::size_t g = 0; g < values.size(); g += group)
        for (std::size_t stride = 1; stride < group; stride *= 4)
            for (std::size_t start = 0; start < group; start += stride * 4)
                for (std::size_t i = 0; i < stride; ++i) {
                    auto* p = values.data() + g + start + i;
                    double a = p[0], b = p[stride], c = p[2 * stride],
                           d = p[3 * stride];
                    p[0] = ((a + b) + (c - d)) * .5;
                    p[stride] = ((a + b) + (d - c)) * .5;
                    p[2 * stride] = ((a + c) + (d - b)) * .5;
                    p[3 * stride] = ((b + c) + (d - a)) * .5;
                }
}

std::int8_t quantize(float x) {
    const float lo = std::floor(x), fraction = x - lo;
    int n = int(lo);
    if (fraction > .5f || (fraction == .5f && n % 2))
        ++n;
    return std::int8_t(std::clamp(n, -127, 127));
}

std::size_t quant_group(Shard& shard, const std::string& prefix) {
    const auto info = shard.tensor(prefix + ".comfy_quant");
    require(
        info.dtype == Dtype::u8 && info.rank == 1 && info.bytes <= 256,
        "h3_denoiser_quant_marker"
    );
    std::array<std::uint8_t, 256> raw;
    shard.read_tensor(
        prefix + ".comfy_quant", 0, {raw.data(), std::size_t(info.bytes)}
    );
    const auto group = detail::convrot_group(
        {reinterpret_cast<const char*>(raw.data()), std::size_t(info.bytes)}
    );
    require(group != 0, "h3_denoiser_quant_marker");
    return group;
}

// One projection payload resident at a time, released before the admission.
void project_base(
    Shard& shard,
    const std::string& prefix,
    std::span<const float> input,
    std::size_t in,
    std::size_t out,
    std::span<float> output,
    bool bias,
    Resources& resources,
    const std::atomic_bool& cancel,
    const H3Denoiser::Hook& hook,
    H3Compute* compute
) {
    require(
        in > 0 && in <= 28672 && out > 0 && out <= 96768 && input.size() % in == 0
            && output.size() == input.size() / in * out,
        "h3_denoiser_projection_shape"
    );
    finite(input);
    shape(shard, prefix + ".weight", {out, in});
    const auto info = shard.tensor(prefix + ".weight");
    const bool quantized = info.dtype == Dtype::i8;
    require(
        quantized || info.dtype == Dtype::bf16 || info.dtype == Dtype::fp32,
        "h3_denoiser_projection_dtype"
    );
    const auto rows = input.size() / in;
    // Fixed read scratch and hook/control envelope included; tensor memory uses exact
    // counts.
    const Bytes bytes =
        quantized ? out * 8 + in * 9 + 4096 : in * 64 * 4 + out * 4 + 4096;
    Admission admitted(resources, bytes);
    auto bias_values = std::make_unique<float[]>(out);
    if (bias) {
        shape(shard, prefix + ".bias", {out});
        read_dense(shard, prefix + ".bias", {bias_values.get(), out}, cancel);
    }
    if (quantized) {
        shape(shard, prefix + ".weight_scale", {out, 1});
        require(
            shard.tensor(prefix + ".weight_scale").dtype == Dtype::fp32,
            "h3_denoiser_scale_dtype"
        );
        const auto group = quant_group(shard, prefix);
        if (compute) {
            auto scales = std::make_unique<float[]>(out);
            read_dense(shard, prefix + ".weight_scale", {scales.get(), out}, cancel);
            if (compute->convrot_source(
                    shard.tensor_identity(prefix + ".weight"),
                    [&](std::size_t first, std::span<std::uint8_t> target) {
                        for (std::size_t at = 0; at < target.size();) {
                            stop(cancel);
                            const auto count =
                                std::min<std::size_t>(1024 * 1024, target.size() - at);
                            shard.read_tensor(
                                prefix + ".weight",
                                first + at,
                                target.subspan(at, count)
                            );
                            at += count;
                        }
                    },
                    {scales.get(), out},
                    {bias_values.get(), out},
                    input,
                    in,
                    out,
                    group,
                    output,
                    cancel
                ))
                return;
        }
        require(in % group == 0, "h3_denoiser_rotation_shape");
        Admission payload(resources, in * out);
        auto weights = std::make_unique<std::uint8_t[]>(in * out);
        auto scales = std::make_unique<float[]>(out);
        auto rotated = std::make_unique<double[]>(in);
        auto codes = std::make_unique<std::int8_t[]>(in);
        for (std::size_t offset = 0; offset < in * out;) {
            stop(cancel);
            const auto count = std::min<std::size_t>(1024 * 1024, in * out - offset);
            shard.read_tensor(
                prefix + ".weight", offset, {weights.get() + offset, count}
            );
            offset += count;
        }
        read_dense(shard, prefix + ".weight_scale", {scales.get(), out}, cancel);
        for (std::size_t r = 0; r < out; ++r)
            require(scales[r] >= 0, "h3_denoiser_scale");
        if (compute) {
            compute->convrot_weight(
                shard.tensor_identity(prefix + ".weight"),
                {weights.get(), in * out},
                {scales.get(), out},
                {bias_values.get(), out},
                input,
                in,
                out,
                group,
                output,
                cancel
            );
            return;
        }
        for (std::size_t t = 0; t < rows; ++t) {
            stop(cancel);
            for (std::size_t c = 0; c < in; ++c)
                rotated[c] = input[t * in + c];
            rotate({rotated.get(), in}, group);
            float maximum = 1e-10f;
            for (std::size_t c = 0; c < in; ++c) {
                rotated[c] = float(rotated[c]);
                require(std::isfinite(rotated[c]), "h3_denoiser_nonfinite");
                maximum = std::max(maximum, std::abs(float(rotated[c])));
            }
            const float scale = maximum / 127;
            for (std::size_t c = 0; c < in; ++c)
                codes[c] = quantize(float(rotated[c]) / scale);
            for (std::size_t r = 0; r < out; ++r) {
                if (r % 32 == 0)
                    stop(cancel);
                std::int32_t sum = 0;
                for (std::size_t c = 0; c < in; ++c)
                    sum += std::int32_t(std::bit_cast<std::int8_t>(weights[r * in + c]))
                           * codes[c];
                output[t * out + r] = (float(sum) * scale) * scales[r] + bias_values[r];
                if ((r + 1) % 256 == 0 && hook)
                    hook("projection", r + 1);
            }
        }
    } else {
        if (compute
            && compute->dense_source(
                   shard.tensor_identity(prefix + ".weight"),
                   [&](std::size_t first, std::span<float> target) {
                       read_dense(shard, prefix + ".weight", target, cancel, first);
                   },
                   {bias_values.get(), out},
                   input,
                   in,
                   out,
                   output,
                   cancel,
                   true
               ) != ProjectionResult::unsupported)
            return;
        if (compute) {
            Admission dense_admission(resources, in * out * sizeof(float));
            auto weights = std::make_unique<float[]>(in * out);
            read_dense(shard, prefix + ".weight", {weights.get(), in * out}, cancel);
            compute->dense_weight(
                shard.tensor_identity(prefix + ".weight"),
                {weights.get(), in * out},
                {bias_values.get(), out},
                input,
                in,
                out,
                output,
                cancel,
                true
            );
            return;
        }
        auto block = std::make_unique<float[]>(in * 64);
        for (std::size_t r = 0; r < out; r += 64) {
            stop(cancel);
            const auto count = std::min<std::size_t>(64, out - r);
            read_dense(
                shard, prefix + ".weight", {block.get(), count * in}, cancel, r * in
            );
            for (std::size_t t = 0; t < rows; ++t)
                for (std::size_t j = 0; j < count; ++j) {
                    double sum = 0;
                    for (std::size_t c = 0; c < in; ++c)
                        sum += double(input[t * in + c]) * block[j * in + c];
                    output[t * out + r + j] = float(sum) + bias_values[r + j];
                }
            if (hook)
                hook("projection", r + count);
        }
    }
    finite(output);
}

void project(
    Shard& shard,
    Shard* turbo,
    const std::string& prefix,
    std::span<const float> x,
    std::size_t in,
    std::size_t out,
    std::span<float> y,
    bool bias,
    bool adapted,
    Resources& resources,
    const std::atomic_bool& cancel,
    const H3Denoiser::Hook& hook,
    H3Compute* compute
) {
    project_base(shard, prefix, x, in, out, y, bias, resources, cancel, hook, compute);
    if (!turbo || !adapted)
        return;
    const auto p = "diffusion_model." + prefix;
    const auto a = turbo->tensor(p + ".lora_A.weight");
    require(
        a.dtype == Dtype::bf16 && a.rank == 2 && a.shape[1] == in && a.shape[0] > 0
            && a.shape[0] <= 384,
        "h3_denoiser_lora_layout"
    );
    const auto rank = std::size_t(a.shape[0]);
    shape(*turbo, p + ".lora_B.weight", {out, rank});
    shape(*turbo, p + ".alpha", {});
    float alpha;
    read_dense(*turbo, p + ".alpha", {&alpha, 1}, cancel);
    require(alpha >= 0, "h3_denoiser_lora_alpha");
    const float scale = alpha / float(rank);
    const auto rows = x.size() / in;
    Admission admitted(resources, rows * (rank + out) * sizeof(float));
    auto tmp = std::make_unique<float[]>(rows * rank),
         delta = std::make_unique<float[]>(rows * out);
    project_base(
        *turbo,
        p + ".lora_A",
        x,
        in,
        rank,
        {tmp.get(), rows * rank},
        false,
        resources,
        cancel,
        hook,
        compute
    );
    project_base(
        *turbo,
        p + ".lora_B",
        {tmp.get(), rows * rank},
        rank,
        out,
        {delta.get(), rows * out},
        false,
        resources,
        cancel,
        hook,
        compute
    );
    for (std::size_t i = 0; i < y.size(); ++i)
        y[i] += delta[i] * scale;
    finite(y);
}

void rms(std::span<const float> x, std::span<const float> weight, std::span<float> y) {
    require(
        x.size() == y.size() && x.size() % weight.size() == 0, "h3_denoiser_norm_shape"
    );
    for (std::size_t at = 0; at < x.size(); at += weight.size()) {
        double squares = 0;
        for (std::size_t c = 0; c < weight.size(); ++c)
            squares += double(x[at + c]) * x[at + c];
        double scale = 1 / std::sqrt(squares / weight.size() + 1e-5);
        for (std::size_t c = 0; c < weight.size(); ++c)
            y[at + c] = double(x[at + c]) * scale * weight[c];
    }
    finite(y);
}

void norm(
    Shard& shard,
    const std::string& name,
    std::span<const float> x,
    std::span<float> y,
    std::span<float> scratch,
    const std::atomic_bool& cancel
) {
    shape(shard, name, {scratch.size()});
    read_dense(shard, name, scratch, cancel);
    rms(x, scratch, y);
}

// Angles depend only on token and channel, so one [token,channel,{cos,sin}] table
// serves every head, Q and K, and all blocks with unchanged values.
void rope_table(
    std::span<const float> positions,
    std::span<const float> frequencies,
    std::span<double> table
) {
    for (std::size_t t = 0; t < positions.size() / 3; ++t)
        for (std::size_t c = 0; c < 48; ++c) {
            const double angle =
                double(positions[t * 3 + c / 16]) * frequencies[c % 16];
            table[(t * 48 + c) * 2] = std::cos(angle);
            table[(t * 48 + c) * 2 + 1] = std::sin(angle);
        }
}

void rope(std::span<float> x, std::span<const double> table) {
    for (std::size_t t = 0; t < table.size() / 96; ++t)
        for (std::size_t h = 0; h < 56; ++h)
            for (std::size_t c = 0; c < 48; ++c) {
                const double cos = table[(t * 48 + c) * 2],
                             sin = table[(t * 48 + c) * 2 + 1];
                auto* p = x.data() + (t * 56 + h) * 128;
                const double a = p[c], b = p[c + 48];
                p[c] = a * cos - b * sin;
                p[c + 48] = b * cos + a * sin;
            }
}

void attention(
    std::span<const float> q,
    std::span<const float> k,
    std::span<const float> v,
    std::span<float> y,
    std::size_t tokens,
    const std::atomic_bool& cancel
) {
    std::array<double, H3Denoiser::max_tokens> probability;
    for (std::size_t t = 0; t < tokens; ++t)
        for (std::size_t h = 0; h < 56; ++h) {
            stop(cancel);
            double maximum = -INFINITY;
            for (std::size_t j = 0; j < tokens; ++j) {
                double sum = 0;
                for (std::size_t c = 0; c < 128; ++c)
                    sum +=
                        double(q[(t * 56 + h) * 128 + c]) * k[(j * 56 + h) * 128 + c];
                probability[j] = sum / std::sqrt(128.0);
                maximum = std::max(maximum, probability[j]);
            }
            double total = 0;
            for (std::size_t j = 0; j < tokens; ++j) {
                probability[j] = std::exp(probability[j] - maximum);
                total += probability[j];
            }
            for (std::size_t c = 0; c < 128; ++c) {
                double sum = 0;
                for (std::size_t j = 0; j < tokens; ++j)
                    sum += probability[j] / total * v[(j * 56 + h) * 128 + c];
                y[(t * 56 + h) * 128 + c] = sum;
            }
        }
    finite(y);
}

bool matrix(
    const Shard& shard,
    const std::string& name,
    Dtype dtype,
    std::uint64_t rows,
    std::uint64_t columns
) {
    const auto t = shard.tensor(name);
    return t.dtype == dtype && t.rank == 2 && t.shape[0] == rows
           && t.shape[1] == columns;
}

// The released INT8 convrot block layout with optional Turbo adapters. Any other
// layout keeps the host-staged path rather than adopting a different contract.
std::variant<detail::H3FeedForward, MediaPlanError> describe_feed_forward(
    Shard& shard, Shard* turbo, std::size_t block, const std::atomic_bool& cancel
) {
    detail::H3FeedForward ff;
    for (std::size_t fc = 0; fc < 2; ++fc) {
        const auto rows = fc ? ff.hidden : 2 * ff.width,
                   columns = fc ? ff.width : ff.hidden;
        const auto p =
            "blocks." + std::to_string(block) + ".mlp.fc" + std::to_string(fc + 1);
        ff.weight[fc] = p + ".weight";
        ff.scale[fc] = p + ".weight_scale";
        if (!matrix(shard, ff.weight[fc], Dtype::i8, rows, columns)
            || !matrix(shard, ff.scale[fc], Dtype::fp32, rows, 1))
            return MediaPlanError{
                PlanFailure::unsupported_layout, "h3_ff_convrot_layout", {}
            };
        ff.weight_id[fc] = shard.tensor_identity(ff.weight[fc]);
        ff.scale_id[fc] = shard.tensor_identity(ff.scale[fc]);
        ff.group[fc] = quant_group(shard, p);
        if (!turbo)
            continue;
        ff.lora_a[fc] = "diffusion_model." + p + ".lora_A.weight";
        ff.lora_b[fc] = "diffusion_model." + p + ".lora_B.weight";
        const auto rank = std::size_t(turbo->tensor(ff.lora_a[fc]).shape[0]);
        if (!rank || rank > h3_lora_rank_limit
            || !matrix(*turbo, ff.lora_a[fc], Dtype::bf16, rank, columns)
            || !matrix(*turbo, ff.lora_b[fc], Dtype::bf16, rows, rank))
            return MediaPlanError{
                PlanFailure::unsupported_layout, "h3_ff_lora_layout", {}
            };
        ff.rank[fc] = rank;
        ff.lora_a_id[fc] = turbo->tensor_identity(ff.lora_a[fc]);
        ff.lora_b_id[fc] = turbo->tensor_identity(ff.lora_b[fc]);
        const auto alpha_name = "diffusion_model." + p + ".alpha";
        shape(*turbo, alpha_name, {});
        float alpha;
        read_dense(*turbo, alpha_name, {&alpha, 1}, cancel);
        require(alpha >= 0, "h3_denoiser_lora_alpha");
        ff.lora_scale[fc] = alpha / float(rank);
    }
    return ff;
}

// Admit one block's feed-forward and upload its exact checkpoint bytes.
std::optional<MediaPlanError> prepare_feed_forward(
    MediaSession& session,
    Shard& shard,
    Shard* turbo,
    const detail::H3FeedForward& ff,
    std::size_t tokens,
    std::size_t groups,
    Resources& resources,
    const std::atomic_bool& cancel
) {
    if (auto error = session.admit(
            detail::h3_feed_forward_manifest(ff, tokens, groups),
            MediaShape{tokens, ff.hidden, 0, 0, 0, ff.width}
        ))
        return error;
    try {
        for (const auto& slice : session.plan().weights) {
            const bool adapter = slice.name.starts_with("diffusion_model.");
            if (adapter && !turbo)
                throw std::invalid_argument("h3_ff_missing_turbo");
            Shard& source = adapter ? *turbo : shard;
            const auto t = source.tensor(slice.name);
            const std::size_t width = t.dtype == Dtype::i8     ? 1
                                      : t.dtype == Dtype::bf16 ? 2
                                                               : 4;
            Admission staging(resources, slice.stored_bytes);
            std::vector<std::uint8_t> raw(slice.stored_bytes);
            const auto offset = slice.first_row * std::size_t(t.shape[1]) * width;
            for (std::size_t at = 0; at < raw.size();) {
                stop(cancel);
                const auto count = std::min<std::size_t>(16 << 20, raw.size() - at);
                source.read_tensor(slice.name, offset + at, {raw.data() + at, count});
                at += count;
            }
            source.check_unchanged();
            stop(cancel);
            if (t.dtype == Dtype::bf16)
                for (std::size_t i = 0; i < raw.size(); i += 2)
                    require(
                        ((std::uint16_t(raw[i + 1]) << 8 | raw[i]) & 0x7f80) != 0x7f80,
                        "h3_cuda_nonfinite_weight"
                    );
            else if (t.dtype == Dtype::fp32)
                for (std::size_t i = 0; i < raw.size(); i += 4) {
                    float value;
                    std::memcpy(&value, raw.data() + i, sizeof(value));
                    require(std::isfinite(value) && value >= 0, "h3_cuda_scale");
                }
            auto destination = session.weight(slice);
            session.upload_weight(destination, std::as_bytes(std::span(raw)));
        }
    } catch (...) {
        session.abort(std::current_exception());
        throw;
    }
    return {};
}
}  // namespace

namespace detail {
void close_h3_feed_forward(
    MediaSession& session, std::span<H3ResidentLauncher* const> launchers, bool abort
) {
    std::exception_ptr first;
    for (auto* launcher : launchers)
        try {
            launcher->release_stream();
        } catch (...) {
            if (!first)
                first = std::current_exception();
        }
    try {
        if (abort || first)
            session.abort();
        else
            session.finish();
    } catch (...) {
        if (first)
            throw OperationCleanupError(first, std::current_exception());
        throw;
    }
    if (first)
        std::rethrow_exception(first);
}

StageManifest h3_feed_forward_manifest(
    const H3FeedForward& ff, std::size_t tokens, std::size_t groups
) {
    StageManifest m;
    m.stage = MediaStage::denoise;
    m.schedule = MediaSchedule::feed_forward;
    m.blas_bytes = h3_resident_blas_bytes;
    m.status_bytes = sizeof(std::uint32_t);
    const bool lora = ff.rank[0] != 0;
    std::size_t lora_rank = 0, lora_elements = 0;
    for (std::size_t fc = 0; fc < 2; ++fc) {
        const auto rows = fc ? ff.hidden : 2 * ff.width,
                   columns = fc ? ff.width : ff.hidden;
        const std::size_t planes = fc ? 1 : 2;
        m.weights.push_back(
            {ff.weight[fc],
             ff.weight_id[fc],
             TensorStorage::i8_convrot,
             rows,
             columns,
             WeightPartition::channels,
             4,
             planes,
             ProjectionPrecision::convrot,
             ff.group[fc]}
        );
        m.weights.push_back(
            {ff.scale[fc],
             ff.scale_id[fc],
             TensorStorage::f32,
             rows,
             1,
             WeightPartition::channels,
             4,
             planes,
             ProjectionPrecision::f32}
        );
        if (!lora)
            continue;
        const auto rank = ff.rank[fc];
        m.weights.push_back(
            {ff.lora_a[fc],
             ff.lora_a_id[fc],
             TensorStorage::bf16_exact,
             rank,
             columns,
             WeightPartition::replicated,
             1,
             1,
             ProjectionPrecision::precise_f64}
        );
        m.weights.push_back(
            {ff.lora_b[fc],
             ff.lora_b_id[fc],
             TensorStorage::bf16_exact,
             rows,
             rank,
             WeightPartition::channels,
             4,
             planes,
             ProjectionPrecision::precise_f64}
        );
        lora_rank = std::max(lora_rank, rank);
        lora_elements = std::max(
            lora_elements, rank * columns + rows / planes * rank + tokens * columns
        );
    }
    m.tensors.push_back({"ff/residual-gates", groups * ff.hidden * sizeof(float)});
    m.tensors.push_back({"ff/token-groups", tokens * sizeof(std::uint32_t)});
    if (lora) {
        m.tensors.push_back(
            {"ff/lora-low",
             tokens * lora_rank * sizeof(double),
             MediaPhase::ff,
             MediaPhase::ff,
             AllocationKind::conversion}
        );
        m.tensors.push_back(
            {"ff/lora-weights",
             lora_elements * sizeof(double),
             MediaPhase::ff,
             MediaPhase::ff,
             AllocationKind::conversion}
        );
    }
    m.workspaces_complete = true;
    return m;
}

void upload_h3_feed_forward_inputs(
    MediaSession& session,
    const H3FeedForward& ff,
    std::span<const float> normalized,
    std::span<const float> hidden,
    std::span<const float> gates,
    std::span<const std::uint32_t> token_groups
) {
    const auto tokens = token_groups.size(), groups = gates.size() / ff.hidden;
    require(
        tokens && groups && gates.size() == groups * ff.hidden
            && normalized.size() == tokens * ff.hidden
            && hidden.size() == tokens * ff.hidden,
        "h3_ff_input_shape"
    );
    for (const auto& planned : session.plan().devices) {
        auto matrix = [&](const char* name, std::size_t rows) {
            return session.tensor(
                planned.device,
                name,
                DeviceDType::f32,
                {rows, ff.hidden},
                {ff.hidden, 1}
            );
        };
        auto n = matrix("resident/normalized", tokens);
        session.upload_input(n, std::as_bytes(normalized));
        auto h = matrix("resident/hidden", tokens);
        session.upload_input(h, std::as_bytes(hidden));
        auto g = matrix("ff/residual-gates", groups);
        session.upload_input(g, std::as_bytes(gates));
        auto t = session.tensor(
            planned.device, "ff/token-groups", DeviceDType::u32, {tokens}, {1}
        );
        session.upload_input(t, std::as_bytes(token_groups));
    }
}

// Each device holds a channel share of every projection and its own full hidden.
void execute_h3_feed_forward(
    MediaSession& session,
    std::span<H3ResidentLauncher* const> launchers,
    const H3FeedForward& ff,
    std::size_t tokens,
    std::size_t groups,
    std::span<float> hidden,
    const std::atomic_bool& cancel
) {
    const auto& plan = session.plan();
    require(
        plan.schedule == MediaSchedule::feed_forward && plan.rows && plan.rows <= tokens
            && !plan.devices.empty() && hidden.size() == tokens * ff.hidden,
        "h3_ff_plan"
    );
    const bool lora = ff.rank[0] != 0;

    // Projection 0/1 are fc1's gate/value planes; projection 2 is fc2.
    struct Device {
        H3ResidentLauncher* launcher = nullptr;
        int id = -1;
        std::size_t fc1_first = 0, fc1_rows = 0, fc2_first = 0, fc2_rows = 0;
        std::array<DeviceTensorView, 3> weight, scale, lora_b;
        std::array<DeviceTensorView, 2> lora_a;
        DeviceTensorView normalized, hidden, gates, groups, conversion, lora_weights,
            blas, status;
    };

    std::vector<Device> devices;
    devices.reserve(plan.devices.size());
    for (const auto& planned : plan.devices) {
        auto& d = devices.emplace_back();
        d.id = planned.device;
        for (auto* candidate : launchers) {
            require(candidate, "h3_ff_launcher");
            if (candidate->device() == d.id) {
                require(!d.launcher, "h3_ff_launcher");
                d.launcher = candidate;
            }
        }
        require(d.launcher, "h3_ff_launcher");
        auto slice = [&](const std::string& name, std::size_t j) -> const WeightSlice& {
            for (const auto& w : plan.weights)
                if (w.device == d.id && w.name == name
                    && (j == 2 || (w.first_row >= ff.width) == (j == 1)))
                    return w;
            throw std::runtime_error("h3_ff_weight");
        };
        for (std::size_t j = 0; j < 3; ++j) {
            const auto fc = j / 2;
            const auto& w = slice(ff.weight[fc], j);
            const auto& s = slice(ff.scale[fc], j);
            require(s.first_row == w.first_row && s.rows == w.rows, "h3_ff_partition");
            d.weight[j] = session.weight(w);
            d.scale[j] = session.weight(s);
            if (j == 0) {
                d.fc1_first = w.first_row;
                d.fc1_rows = w.rows;
            } else if (j == 1) {
                require(
                    w.first_row == ff.width + d.fc1_first && w.rows == d.fc1_rows,
                    "h3_ff_partition"
                );
            } else {
                d.fc2_first = w.first_row;
                d.fc2_rows = w.rows;
            }
            if (!lora)
                continue;
            const auto& b = slice(ff.lora_b[fc], j);
            require(b.first_row == w.first_row && b.rows == w.rows, "h3_ff_partition");
            d.lora_b[j] = session.weight(b);
            if (j != 1)
                d.lora_a[fc] = session.weight(slice(ff.lora_a[fc], j));
        }
        auto matrix = [&](const char* name, std::size_t rows) {
            return session.tensor(
                d.id, name, DeviceDType::f32, {rows, ff.hidden}, {ff.hidden, 1}
            );
        };
        d.normalized = matrix("resident/normalized", tokens);
        d.hidden = matrix("resident/hidden", tokens);
        d.gates = matrix("ff/residual-gates", groups);
        d.groups =
            session.tensor(d.id, "ff/token-groups", DeviceDType::u32, {tokens}, {1});
        std::vector<DeviceTensorView*> ready{
            &d.normalized, &d.hidden, &d.gates, &d.groups
        };
        for (std::size_t j = 0; j < 3; ++j) {
            ready.push_back(&d.weight[j]);
            ready.push_back(&d.scale[j]);
            if (lora)
                ready.push_back(&d.lora_b[j]);
        }
        if (lora) {
            ready.push_back(&d.lora_a[0]);
            ready.push_back(&d.lora_a[1]);
        }
        session.require_ready(ready);
    }
    // fc1 gate/value and fc2 have independently partitioned output channels.
    auto ordered = [&](bool down) {
        std::vector<Device*> result;
        result.reserve(devices.size());
        for (auto& d : devices)
            result.push_back(&d);
        std::sort(result.begin(), result.end(), [&](auto* a, auto* b) {
            return (down ? a->fc2_first : a->fc1_first)
                   < (down ? b->fc2_first : b->fc1_first);
        });
        std::size_t column = 0;
        for (auto* d : result) {
            require(
                (down ? d->fc2_first : d->fc1_first) == column
                    && (down ? d->fc2_rows : d->fc1_rows),
                "h3_ff_partition"
            );
            column += down ? d->fc2_rows : d->fc1_rows;
        }
        require(column == (down ? ff.hidden : ff.width), "h3_ff_partition");
        return result;
    };
    const auto gate_order = ordered(false), down_order = ordered(true);
    session.advance(MediaPhase::ff);
    for (auto& d : devices) {
        const auto conversion =
            std::find_if(plan.tensors.begin(), plan.tensors.end(), [&](const auto& t) {
                return t.device == d.id && t.name == "resident/projection-conversion";
            });
        require(conversion != plan.tensors.end(), "h3_ff_plan");
        d.conversion = session.tensor(
            d.id,
            "resident/projection-conversion",
            DeviceDType::u8,
            {std::size_t(conversion->bytes)},
            {1}
        );
        d.blas = session.tensor(
            d.id, "resident/blas", DeviceDType::u8, {h3_resident_blas_bytes}, {1}
        );
        d.status = session.tensor(d.id, "resident/status", DeviceDType::u32, {1}, {1});
        if (lora) {
            const auto elements = std::max(
                ff.rank[0] * ff.hidden + d.fc1_rows * ff.rank[0]
                    + plan.rows * ff.hidden,
                ff.rank[1] * ff.width + d.fc2_rows * ff.rank[1] + plan.rows * ff.width
            );
            d.lora_weights = session.tensor(
                d.id, "ff/lora-weights", DeviceDType::f64, {elements}, {1}
            );
        }
    }
    auto project = [&](Device& d,
                       std::size_t j,
                       DeviceTensorView& input,
                       DeviceTensorView& output,
                       std::size_t rows) {
        const auto fc = j / 2;
        std::optional<DeviceTensorView> low;
        std::vector<DeviceTensorView*> read{&input, &d.weight[j], &d.scale[j]};
        std::vector<DeviceTensorView*> write{&output, &d.conversion};
        if (lora) {
            low = session.tensor(
                d.id,
                "ff/lora-low",
                DeviceDType::f64,
                {rows, ff.rank[fc]},
                {ff.rank[fc], 1}
            );
            read.push_back(&d.lora_a[fc]);
            read.push_back(&d.lora_b[j]);
            write.push_back(&*low);
            write.push_back(&d.lora_weights);
        }
        write.push_back(&d.blas);
        write.push_back(&d.status);
        session.launch(d.id, read, write, [&](void* stream, auto in, auto out) {
            d.launcher->convrot(stream, in, out, ff.group[fc], ff.lora_scale[fc]);
        });
        session.complete(write, d.status);
    };

    struct Tile {
        Device* device = nullptr;
        DeviceTensorView input, pair, gate, value, ff_input, down, gathered, hidden;
    };

    for (std::size_t first = 0; first < tokens;) {
        stop(cancel);
        const auto rows = std::min(plan.rows, tokens - first);
        std::vector<Tile> tiles(devices.size());
        for (std::size_t i = 0; i < devices.size(); ++i) {
            auto& d = devices[i];
            auto& t = tiles[i];
            t.device = &d;
            t.input = session.rows(d.normalized, first, rows);
            t.pair = session.tensor(
                d.id,
                "resident/ff-gate-up",
                DeviceDType::f32,
                {2 * rows, d.fc1_rows},
                {d.fc1_rows, 1}
            );
            t.gate = session.rows(t.pair, 0, rows);
            t.value = session.rows(t.pair, rows, rows);
            project(d, 0, t.input, t.gate, rows);
            project(d, 1, t.input, t.value, rows);
            const std::array read{&t.gate, &t.value};
            const std::array write{&t.gate, &d.status};
            session.launch(d.id, read, write, [&](void* stream, auto in, auto out) {
                d.launcher->silu_gate(stream, in, out);
            });
            session.complete(write, d.status);
        }
        auto tile = [&](const Device* d) -> Tile& {
            return *std::find_if(tiles.begin(), tiles.end(), [&](const auto& t) {
                return t.device == d;
            });
        };
        for (auto& t : tiles) {
            auto& d = *t.device;
            auto* input = &t.gate;
            if (tiles.size() > 1) {
                t.ff_input = session.tensor(
                    d.id,
                    "resident/ff-input",
                    DeviceDType::f32,
                    {rows, ff.width},
                    {ff.width, 1}
                );
                for (auto* source : gate_order)
                    session.gather(tile(source).gate, t.ff_input, source->fc1_first);
                input = &t.ff_input;
            }
            t.down = session.tensor(
                d.id,
                "resident/residual-tile",
                DeviceDType::f32,
                {rows, d.fc2_rows},
                {d.fc2_rows, 1}
            );
            project(d, 2, *input, t.down, rows);
        }
        for (auto& t : tiles) {
            auto& d = *t.device;
            auto* projected = &t.down;
            if (tiles.size() > 1) {
                t.gathered = session.tensor(
                    d.id,
                    "resident/peer-tile",
                    DeviceDType::f32,
                    {rows, ff.hidden},
                    {ff.hidden, 1}
                );
                for (auto* source : down_order)
                    session.gather(tile(source).down, t.gathered, source->fc2_first);
                projected = &t.gathered;
            }
            t.hidden = session.rows(d.hidden, first, rows);
            const std::array read{&t.hidden, projected, &d.gates, &d.groups};
            const std::array write{&t.hidden, &d.status};
            session.launch(d.id, read, write, [&](void* stream, auto in, auto out) {
                d.launcher->gated_residual(stream, in, out, first);
            });
            session.complete(write, d.status);
        }
        first += rows;
    }
    // Every device holds the same updated hidden; return the first copy.
    auto result = std::move(devices.front().hidden);
    devices.clear();
    session.advance(MediaPhase::complete);
    session.download_output(result, std::as_writable_bytes(hidden));
}
}  // namespace detail

H3Denoiser::H3Denoiser(std::shared_ptr<Resources> r, std::shared_ptr<H3Compute> compute)
    : resources_(std::move(r)), compute_(std::move(compute)) {
    require(bool(resources_), "h3_denoiser_resources");
}

H3Denoiser::~H3Denoiser() {
    unload();
}

void H3Denoiser::load(
    const char* root,
    const std::string& name,
    const std::atomic_bool& cancel,
    std::shared_ptr<checkpoint::ReadCache> cache
) {
    stop(cancel);
    require(!executing_, "busy");
    require(!loaded(), "h3_denoiser_already_loaded");
    Admission admitted(*resources_, metadata_bytes);
    auto shard = std::make_unique<Shard>(
        root,
        name,
        std::make_shared<checkpoint::MemoryBudget>(metadata_bytes),
        checkpoint::Limits{1024 * 1024, 4096, 8192}
    );
    shard->cache_reads(std::move(cache), &cancel);
    shape(*shard, "condition_proj.weight", {hidden, text_width});
    shape(*shard, "video_patch_proj.weight", {hidden, video_width});
    shape(*shard, "audio_patch_proj.weight", {hidden, audio_width});
    shape(*shard, "rope.inv_freq", {16});
    for (std::size_t layer = 0; layer < layers; ++layer) {
        stop(cancel);
        auto p = "blocks." + std::to_string(layer) + ".";
        shape(*shard, p + "adaln_proj.linear.weight", {96768, 2688});
        shape(*shard, p + "attn.qkv_proj.weight", {21504, hidden});
        shape(*shard, p + "mlp.fc1.weight", {28672, hidden});
    }
    shard->check_unchanged();
    stop(cancel);
    shard_ = std::move(shard);
    metadata_ = admitted.handle;
    admitted.handle = 0;
}

void H3Denoiser::load_turbo(
    const char* root,
    const std::string& name,
    const std::atomic_bool& cancel,
    std::shared_ptr<checkpoint::ReadCache> cache
) {
    stop(cancel);
    require(!executing_, "busy");
    require(loaded() && !turbo_, "h3_denoiser_turbo_state");
    Admission admitted(*resources_, metadata_bytes);
    auto shard = std::make_unique<Shard>(
        root,
        name,
        std::make_shared<checkpoint::MemoryBudget>(metadata_bytes),
        checkpoint::Limits{1024 * 1024, 2048, 8192}
    );
    shard->cache_reads(std::move(cache), &cancel);
    for (std::size_t layer = 0; layer < layers; ++layer) {
        stop(cancel);
        auto p = "diffusion_model.blocks." + std::to_string(layer) + ".";
        shape(*shard, p + "attn.qkv_proj.lora_A.weight", {384, hidden});
        shape(*shard, p + "attn.qkv_proj.lora_B.weight", {21504, 384});
    }
    shard->check_unchanged();
    stop(cancel);
    turbo_ = std::move(shard);
    turbo_metadata_ = admitted.handle;
    admitted.handle = 0;
}

void H3Denoiser::unload() {
    require(!executing_, "busy");
    turbo_.reset();
    shard_.reset();
    if (turbo_metadata_) {
        resources_->released(turbo_metadata_);
        turbo_metadata_ = 0;
    }
    if (metadata_) {
        resources_->released(metadata_);
        metadata_ = 0;
    }
}

void H3Denoiser::execute(
    const Input& input,
    std::span<float> video_out,
    std::span<float> audio_out,
    const std::atomic_bool& cancel,
    const Hook& hook,
    Resident* resident
) {
    stop(cancel);
    require(loaded(), "h3_denoiser_not_loaded");
    Active active(executing_);
    require(
        !input.text.empty() && input.text.size() % text_width == 0
            && input.video.size() % video_width == 0
            && input.audio.size() % audio_width == 0,
        "h3_denoiser_input_shape"
    );
    const auto nt = input.text.size() / text_width,
               nv = input.video.size() / video_width,
               na = input.audio.size() / audio_width, n = nt + nv + na;
    require(
        nt <= (compute_ ? text::Tokenizer::max_tokens : 512)
            && n
                   <= (compute_ ? 107856 + 1206 + text::Tokenizer::max_tokens
                                : max_tokens)
            && n > nt && input.positions.size() == n * 3 && input.timesteps.size() == n
            && input.tags.size() == n && video_out.size() == input.video.size()
            && audio_out.size() == input.audio.size(),
        "h3_denoiser_token_limit"
    );
    finite(input.text);
    finite(input.video);
    finite(input.audio);
    finite(input.positions);
    finite(input.timesteps);
    std::array<float, 3> times{};
    std::size_t time_count = 0;
    Admission indices(*resources_, n * sizeof(std::size_t));
    auto time_ids = std::make_unique<std::size_t[]>(n);
    for (std::size_t i = 0; i < n; ++i) {
        require(
            input.tags[i] < 3 && input.timesteps[i] >= 0 && input.timesteps[i] <= 1,
            "h3_denoiser_timestep"
        );
        std::size_t id = 0;
        while (id < time_count && times[id] != input.timesteps[i])
            ++id;
        if (id == time_count) {
            require(time_count < 3, "h3_denoiser_time_count");
            times[time_count++] = input.timesteps[i];
        }
        time_ids[i] = id;
    }
    constexpr std::size_t inner = 7168, ff = 14336;
    // Three unique timesteps, all three AdaLN modalities; fixed scratch ledger
    // also covers stack arrays, checkpoint read chunks and norm vectors.
    constexpr std::size_t fixed = 3 * (256 + 5376 + 2688 + 96768) + hidden + 16;
    constexpr std::size_t per_token = hidden * 3 + inner * 7 + ff * 2;
    Admission admitted(
        *resources_,
        (n * per_token + fixed) * 4
            + max_tokens * (sizeof(double) + sizeof(std::size_t)) + 16384
            + n * 96 * sizeof(double)
    );
    auto memory = std::make_unique<float[]>(n * per_token + fixed);
    std::size_t offset = 0;
    auto span = [&](std::size_t count) {
        std::span<float> s(memory.get() + offset, count);
        offset += count;
        return s;
    };
    auto current = span(n * hidden), normalized = span(n * hidden),
         projected = span(n * hidden), qkv = span(n * inner * 3), q = span(n * inner),
         k = span(n * inner), v = span(n * inner), attended = span(n * inner),
         mlp = span(n * ff * 2);
    auto frequencies = span(16), norm_weight = span(hidden), t_freq = span(3 * 256),
         t_hidden = span(3 * 5376), t_embed = span(3 * 2688), adaln = span(3 * 96768);
    require(offset <= n * per_token + fixed, "h3_denoiser_scratch");
    shard_->check_unchanged();
    if (turbo_)
        turbo_->check_unchanged();
    read_dense(*shard_, "rope.inv_freq", frequencies, cancel);
    auto rotary = std::make_unique<double[]>(n * 96);
    rope_table(input.positions, frequencies, {rotary.get(), n * 96});
    auto linear = [&](const std::string& p,
                      std::span<const float> x,
                      std::size_t in,
                      std::size_t out,
                      std::span<float> y,
                      bool bias = false,
                      bool adapted = false) {
        project(
            *shard_,
            turbo_.get(),
            p,
            x,
            in,
            out,
            y,
            bias,
            adapted,
            *resources_,
            cancel,
            hook,
            compute_.get()
        );
    };
    auto attn = [&](const std::string& p, std::size_t count, bool positional) {
        linear(
            p + ".qkv_proj",
            normalized.first(count * hidden),
            hidden,
            inner * 3,
            qkv.first(count * inner * 3),
            false,
            true
        );
        // ComfyUI base weights and the ComfyUI Turbo adapter store [QKV,head,128].
        // The original Hugging Face grouped layout is a different checkpoint contract.
        for (std::size_t t = 0; t < count; ++t)
            for (std::size_t h = 0; h < 56; ++h)
                for (std::size_t c = 0; c < 128; ++c) {
                    auto at = (t * 56 + h) * 128 + c;
                    const auto row = t * inner * 3 + h * 128 + c;
                    q[at] = qkv[row];
                    k[at] = qkv[row + inner];
                    v[at] = qkv[row + 2 * inner];
                }
        norm(
            *shard_,
            p + ".q_norm.weight",
            q.first(count * inner),
            q.first(count * inner),
            norm_weight.first(128),
            cancel
        );
        norm(
            *shard_,
            p + ".k_norm.weight",
            k.first(count * inner),
            k.first(count * inner),
            norm_weight.first(128),
            cancel
        );
        if (positional) {
            rope(q.first(count * inner), {rotary.get(), n * 96});
            rope(k.first(count * inner), {rotary.get(), n * 96});
        }
        if (compute_)
            compute_->attention(
                q.first(count * inner),
                k.first(count * inner),
                v.first(count * inner),
                count,
                56,
                56,
                128,
                false,
                attended.first(count * inner),
                cancel,
                true
            );
        else
            attention(
                q.first(count * inner),
                k.first(count * inner),
                v.first(count * inner),
                attended.first(count * inner),
                count,
                cancel
            );
        linear(
            p + ".out_proj",
            attended.first(count * inner),
            inner,
            hidden,
            projected.first(count * hidden),
            false,
            true
        );
    };
    auto feed = [&](const std::string& p, std::size_t count) {
        linear(
            p + ".fc1",
            normalized.first(count * hidden),
            hidden,
            ff * 2,
            mlp.first(count * ff * 2),
            false,
            true
        );
        // Compact gate*value in place; destination never reaches a future row.
        for (std::size_t t = 0; t < count; ++t)
            for (std::size_t c = 0; c < ff; ++c) {
                const double gate = mlp[t * ff * 2 + c];
                mlp[t * ff + c] =
                    float(gate / (1 + std::exp(-gate))) * mlp[t * ff * 2 + ff + c];
            }
        linear(
            p + ".fc2",
            mlp.first(count * ff),
            ff,
            hidden,
            projected.first(count * hidden),
            false,
            true
        );
    };
    linear(
        "condition_proj",
        input.text,
        text_width,
        hidden,
        current.first(nt * hidden),
        true
    );
    for (std::size_t b = 0; b < 2; ++b) {
        auto p = "token_refiner.blocks." + std::to_string(b);
        norm(
            *shard_,
            p + ".norm1.weight",
            current.first(nt * hidden),
            normalized.first(nt * hidden),
            norm_weight,
            cancel
        );
        attn(p + ".attn", nt, false);
        for (std::size_t i = 0; i < nt * hidden; ++i)
            current[i] += projected[i];
        norm(
            *shard_,
            p + ".norm2.weight",
            current.first(nt * hidden),
            normalized.first(nt * hidden),
            norm_weight,
            cancel
        );
        feed(p + ".mlp", nt);
        for (std::size_t i = 0; i < nt * hidden; ++i)
            current[i] += projected[i];
        if (hook)
            hook("refiner_completed", b + 1);
    }
    norm(
        *shard_,
        "token_refiner.final_norm.weight",
        current.first(nt * hidden),
        current.first(nt * hidden),
        norm_weight,
        cancel
    );
    if (nv)
        linear(
            "video_patch_proj",
            input.video,
            video_width,
            hidden,
            current.subspan(nt * hidden, nv * hidden),
            true
        );
    if (na)
        linear(
            "audio_patch_proj",
            input.audio,
            audio_width,
            hidden,
            current.subspan((nt + nv) * hidden, na * hidden),
            true
        );
    for (std::size_t t = 0; t < time_count; ++t)
        for (std::size_t c = 0; c < 128; ++c) {
            const double angle =
                times[t] * std::exp(-std::log(10000.0) * double(c) / 128);
            t_freq[t * 256 + c] = std::cos(angle);
            t_freq[t * 256 + c + 128] = std::sin(angle);
        }
    linear(
        "time_embedder.proj_in",
        t_freq.first(time_count * 256),
        256,
        5376,
        t_hidden.first(time_count * 5376),
        true
    );
    for (std::size_t i = 0; i < time_count * 5376; ++i) {
        double a = t_hidden[i];
        t_hidden[i] = a / (1 + std::exp(-a));
    }
    linear(
        "time_embedder.proj_out",
        t_hidden.first(time_count * 5376),
        5376,
        2688,
        t_embed.first(time_count * 2688),
        true
    );
    for (std::size_t i = 0; i < time_count * 2688; ++i) {
        double a = t_embed[i];
        t_embed[i] = a / (1 + std::exp(-a));
    }
    auto modulate = [&](std::size_t group) {
        for (std::size_t t = 0; t < n; ++t) {
            const auto base = (time_ids[t] * 3 + input.tags[t]) * 6 * hidden;
            for (std::size_t c = 0; c < hidden; ++c)
                normalized[t * hidden + c] =
                    normalized[t * hidden + c]
                        * (1 + adaln[base + (group + 1) * hidden + c])
                    + adaln[base + group * hidden + c];
        }
    };
    auto residual = [&](std::size_t group) {
        for (std::size_t t = 0; t < n; ++t) {
            const auto base = (time_ids[t] * 3 + input.tags[t]) * 6 * hidden;
            for (std::size_t c = 0; c < hidden; ++c)
                current[t * hidden + c] +=
                    projected[t * hidden + c] * adaln[base + group * hidden + c];
        }
        finite(current);
    };
    const auto groups = time_count * 3;
    std::optional<Admission> resident_rows;
    std::vector<std::uint32_t> token_groups;
    std::vector<float> ff_gates;
    if (resident) {
        require(bool(resident->session), "h3_resident_session");
        resident_rows.emplace(*resources_, n * 4 + groups * hidden * 4);
        token_groups.resize(n);
        for (std::size_t t = 0; t < n; ++t)
            token_groups[t] = std::uint32_t(time_ids[t] * 3 + input.tags[t]);
        ff_gates.resize(groups * hidden);
    }
    // One fresh session per block. It finishes before the next host-staged
    // operation, so its device memory never competes with that operation.
    auto resident_feed_forward = [&](std::size_t block) {
        if (!resident || resident->rejected)
            return false;
        auto session = resident->session();
        require(bool(session), "h3_resident_session");
        try {
            auto described =
                describe_feed_forward(*shard_, turbo_.get(), block, cancel);
            std::optional<MediaPlanError> error;
            if (auto* rejected = std::get_if<MediaPlanError>(&described))
                error = std::move(*rejected);
            else
                error = prepare_feed_forward(
                    *session,
                    *shard_,
                    turbo_.get(),
                    std::get<detail::H3FeedForward>(described),
                    n,
                    groups,
                    *resources_,
                    cancel
                );
            if (error) {
                if (block)
                    throw std::runtime_error("h3_resident_admission: " + error->detail);
                resident->rejected = std::move(error);
                session->finish();
                return false;
            }
            const auto& ff = std::get<detail::H3FeedForward>(described);
            for (std::size_t g = 0; g < groups; ++g)
                std::copy_n(
                    adaln.begin() + (g * 6 + 5) * hidden,
                    hidden,
                    ff_gates.begin() + g * hidden
                );
            detail::upload_h3_feed_forward_inputs(
                *session, ff, normalized, current, ff_gates, token_groups
            );
            detail::execute_h3_feed_forward(
                *session, resident->launchers, ff, n, groups, current, cancel
            );
            detail::close_h3_feed_forward(*session, resident->launchers);
        } catch (...) {
            auto error = std::current_exception();
            rethrow_after_cleanup(error, [&] {
                detail::close_h3_feed_forward(*session, resident->launchers, true);
            });
        }
        finite(current);
        return true;
    };
    for (std::size_t b = 0; b < layers; ++b) {
        stop(cancel);
        auto p = "blocks." + std::to_string(b);
        linear(
            p + ".adaln_proj.linear",
            t_embed.first(time_count * 2688),
            2688,
            96768,
            adaln.first(time_count * 96768),
            true
        );
        norm(*shard_, p + ".norm1.weight", current, normalized, norm_weight, cancel);
        modulate(0);
        attn(p + ".attn", n, true);
        residual(2);
        norm(*shard_, p + ".norm2.weight", current, normalized, norm_weight, cancel);
        modulate(3);
        if (!resident_feed_forward(b)) {
            feed(p + ".mlp", n);
            residual(5);
        }
        if (hook)
            hook("block_completed", b + 1);
    }
    linear(
        "final_layer.adaln_proj.linear",
        t_embed.first(time_count * 2688),
        2688,
        hidden * 2,
        adaln.first(time_count * hidden * 2),
        true
    );
    norm(*shard_, "final_layer.norm.weight", current, normalized, norm_weight, cancel);
    for (std::size_t t = 0; t < n; ++t)
        for (std::size_t c = 0; c < hidden; ++c)
            normalized[t * hidden + c] =
                normalized[t * hidden + c]
                    * (1 + adaln[(time_ids[t] * 2 + 1) * hidden + c])
                + adaln[time_ids[t] * 2 * hidden + c];
    // Compute selected rows only; dense F64 reductions have no batch-dependent kernel
    // choice.
    if (nv)
        linear(
            "final_layer.video_out",
            normalized.subspan(nt * hidden, nv * hidden),
            hidden,
            video_width,
            video_out,
            true
        );
    if (na)
        linear(
            "final_layer.audio_out",
            normalized.subspan((nt + nv) * hidden, na * hidden),
            hidden,
            audio_width,
            audio_out,
            true
        );
    stop(cancel);
    shard_->check_unchanged();
    if (turbo_)
        turbo_->check_unchanged();
    if (hook)
        hook("completed", n);
    stop(cancel);
}
}  // namespace kadan::video
