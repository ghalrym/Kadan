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

#include "h3_quant_marker.hpp"
#include "h3_resident_block.hpp"
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

MediaPlanError unsupported(const char* why) {
    return {PlanFailure::unsupported_layout, why, {}};
}

// The released INT8 convrot layout with optional Turbo adapters. Any other layout
// keeps the host-staged path rather than adopting a different contract.
std::optional<MediaPlanError> describe_lora(
    Shard* turbo,
    const std::string& prefix,
    std::size_t rows,
    std::size_t columns,
    detail::H3Projection& p,
    const std::atomic_bool& cancel
);

std::optional<MediaPlanError> describe_norms(
    Shard& shard,
    const std::string& prefix,
    std::size_t hidden,
    std::size_t head_dim,
    std::array<std::string, 4>& names,
    std::array<WeightIdentity, 4>& ids
);

std::optional<MediaPlanError> describe_projection(
    Shard& shard,
    Shard* turbo,
    const std::string& prefix,
    std::size_t rows,
    std::size_t columns,
    detail::H3Projection& p,
    const std::atomic_bool& cancel
) {
    p.weight = prefix + ".weight";
    p.scale = prefix + ".weight_scale";
    if (!matrix(shard, p.weight, Dtype::i8, rows, columns)
        || !matrix(shard, p.scale, Dtype::fp32, rows, 1))
        return unsupported("h3_resident_convrot_layout");
    p.weight_id = shard.tensor_identity(p.weight);
    p.scale_id = shard.tensor_identity(p.scale);
    p.group = quant_group(shard, prefix);
    return describe_lora(turbo, prefix, rows, columns, p, cancel);
}

// Exact BF16 or F32 weights on the host's precise dense path, with an optional
// bias and, when Turbo is loaded, its adapter.
std::optional<MediaPlanError> describe_dense(
    Shard& shard,
    Shard* turbo,
    const std::string& prefix,
    std::size_t rows,
    std::size_t columns,
    bool bias,
    detail::H3Projection& p,
    const std::atomic_bool& cancel
) {
    auto storage = [](Dtype dtype) -> std::optional<TensorStorage> {
        if (dtype == Dtype::bf16)
            return TensorStorage::bf16_exact;
        if (dtype == Dtype::fp32)
            return TensorStorage::f32;
        return {};
    };
    p.weight = prefix + ".weight";
    const auto weight = shard.tensor(p.weight);
    const auto weight_storage = storage(weight.dtype);
    if (!weight_storage || !matrix(shard, p.weight, weight.dtype, rows, columns))
        return unsupported("h3_resident_dense_layout");
    p.storage = *weight_storage;
    p.weight_id = shard.tensor_identity(p.weight);
    if (bias) {
        p.bias = prefix + ".bias";
        const auto t = shard.tensor(p.bias);
        const auto bias_storage = storage(t.dtype);
        if (!bias_storage || t.rank != 1 || t.shape[0] != rows)
            return unsupported("h3_resident_dense_layout");
        p.bias_storage = *bias_storage;
        p.bias_id = shard.tensor_identity(p.bias);
    }
    return describe_lora(turbo, prefix, rows, columns, p, cancel);
}

std::optional<MediaPlanError> describe_lora(
    Shard* turbo,
    const std::string& prefix,
    std::size_t rows,
    std::size_t columns,
    detail::H3Projection& p,
    const std::atomic_bool& cancel
) {
    if (!turbo)
        return {};
    p.lora_a = "diffusion_model." + prefix + ".lora_A.weight";
    p.lora_b = "diffusion_model." + prefix + ".lora_B.weight";
    const auto rank = std::size_t(turbo->tensor(p.lora_a).shape[0]);
    if (!rank || rank > h3_lora_rank_limit
        || !matrix(*turbo, p.lora_a, Dtype::bf16, rank, columns)
        || !matrix(*turbo, p.lora_b, Dtype::bf16, rows, rank))
        return unsupported("h3_resident_lora_layout");
    p.rank = rank;
    p.lora_a_id = turbo->tensor_identity(p.lora_a);
    p.lora_b_id = turbo->tensor_identity(p.lora_b);
    const auto alpha_name = "diffusion_model." + prefix + ".alpha";
    shape(*turbo, alpha_name, {});
    float alpha;
    read_dense(*turbo, alpha_name, {&alpha, 1}, cancel);
    require(alpha >= 0, "h3_denoiser_lora_alpha");
    p.lora_scale = alpha / float(rank);
    return {};
}

std::variant<detail::H3FeedForward, MediaPlanError> describe_feed_forward(
    Shard& shard, Shard* turbo, std::size_t block, const std::atomic_bool& cancel
) {
    detail::H3FeedForward ff;
    const auto p = "blocks." + std::to_string(block) + ".mlp.fc";
    if (auto error = describe_projection(
            shard, turbo, p + "1", 2 * ff.width, ff.hidden, ff.fc[0], cancel
        ))
        return *error;
    if (auto error = describe_projection(
            shard, turbo, p + "2", ff.hidden, ff.width, ff.fc[1], cancel
        ))
        return *error;
    return ff;
}

std::variant<detail::H3Block, MediaPlanError> describe_block(
    Shard& shard, Shard* turbo, std::size_t block, const std::atomic_bool& cancel
) {
    detail::H3Block b;
    auto ff = describe_feed_forward(shard, turbo, block, cancel);
    if (auto* error = std::get_if<MediaPlanError>(&ff))
        return *error;
    b.ff = std::get<detail::H3FeedForward>(ff);
    const auto p = "blocks." + std::to_string(block) + ".";
    const auto inner = b.heads * b.head_dim;
    if (auto error = describe_projection(
            shard, turbo, p + "attn.qkv_proj", 3 * inner, b.ff.hidden, b.qkv, cancel
        ))
        return *error;
    if (auto error = describe_projection(
            shard, turbo, p + "attn.out_proj", b.ff.hidden, inner, b.out, cancel
        ))
        return *error;
    // AdaLN carries a BF16 bias and no Turbo adapter.
    const auto modulation = 18 * b.ff.hidden;
    if (auto error = describe_projection(
            shard,
            nullptr,
            p + "adaln_proj.linear",
            modulation,
            b.time_width,
            b.adaln,
            cancel
        ))
        return *error;
    b.adaln.bias = p + "adaln_proj.linear.bias";
    const auto bias = shard.tensor(b.adaln.bias);
    if (bias.dtype != Dtype::bf16 || bias.rank != 1 || bias.shape[0] != modulation)
        return unsupported("h3_resident_adaln_layout");
    b.adaln.bias_id = shard.tensor_identity(b.adaln.bias);
    if (auto error =
            describe_norms(shard, p, b.ff.hidden, b.head_dim, b.norm, b.norm_id))
        return *error;
    return b;
}

std::optional<MediaPlanError> describe_norms(
    Shard& shard,
    const std::string& prefix,
    std::size_t hidden,
    std::size_t head_dim,
    std::array<std::string, 4>& names,
    std::array<WeightIdentity, 4>& ids
) {
    const std::array<const char*, 4> norms{
        "norm1.weight", "norm2.weight", "attn.q_norm.weight", "attn.k_norm.weight"
    };
    for (std::size_t i = 0; i < norms.size(); ++i) {
        names[i] = prefix + norms[i];
        const auto t = shard.tensor(names[i]);
        if (t.dtype != Dtype::bf16 || t.rank != 1
            || t.shape[0] != (i < 2 ? hidden : head_dim))
            return unsupported("h3_resident_norm_layout");
        ids[i] = shard.tensor_identity(names[i]);
    }
    return {};
}

// The released BF16 token refiner with optional Turbo adapters. The last block
// also carries the refiner's final norm.
std::variant<detail::H3Refiner, MediaPlanError> describe_refiner(
    Shard& shard, Shard* turbo, std::size_t block, const std::atomic_bool& cancel
) {
    detail::H3Refiner r;
    const auto p = "token_refiner.blocks." + std::to_string(block) + ".";
    const auto inner = r.heads * r.head_dim;

    struct ProjectionDescription {
        const char* name;
        std::size_t rows, columns;
        detail::H3Projection* projection;
    };

    const std::array<ProjectionDescription, 4> projections{
        {{"attn.qkv_proj", 3 * inner, r.hidden, &r.qkv},
         {"attn.out_proj", r.hidden, inner, &r.out},
         {"mlp.fc1", 2 * r.width, r.hidden, &r.fc1},
         {"mlp.fc2", r.hidden, r.width, &r.fc2}}
    };
    for (const auto& [name, rows, columns, projection] : projections)
        if (auto error = describe_dense(
                shard, turbo, p + name, rows, columns, false, *projection, cancel
            ))
            return *error;
    if (auto error = describe_norms(shard, p, r.hidden, r.head_dim, r.norm, r.norm_id))
        return *error;
    if (block + 1 == 2) {
        r.final_norm = "token_refiner.final_norm.weight";
        const auto t = shard.tensor(r.final_norm);
        if (t.dtype != Dtype::bf16 || t.rank != 1 || t.shape[0] != r.hidden)
            return unsupported("h3_resident_norm_layout");
        r.final_norm_id = shard.tensor_identity(r.final_norm);
    }
    return r;
}

// Final AdaLN, norm and the selected latent output projections; no adapters.
std::variant<detail::H3Final, MediaPlanError>
describe_final(Shard& shard, bool video, bool audio, const std::atomic_bool& cancel) {
    detail::H3Final f;
    if (auto error = describe_dense(
            shard,
            nullptr,
            "final_layer.adaln_proj.linear",
            h3_final_modulation_rows * f.hidden,
            f.time_width,
            true,
            f.adaln,
            cancel
        ))
        return *error;
    if (video)
        if (auto error = describe_dense(
                shard,
                nullptr,
                "final_layer.video_out",
                f.video_width,
                f.hidden,
                true,
                f.video,
                cancel
            ))
            return *error;
    if (audio)
        if (auto error = describe_dense(
                shard,
                nullptr,
                "final_layer.audio_out",
                f.audio_width,
                f.hidden,
                true,
                f.audio,
                cancel
            ))
            return *error;
    f.norm = "final_layer.norm.weight";
    const auto t = shard.tensor(f.norm);
    if (t.dtype != Dtype::bf16 || t.rank != 1 || t.shape[0] != f.hidden)
        return unsupported("h3_resident_norm_layout");
    f.norm_id = shard.tensor_identity(f.norm);
    return f;
}

// Admit one block stage and upload its exact checkpoint bytes.
std::optional<MediaPlanError> prepare(
    MediaSession& session,
    Shard& shard,
    Shard* turbo,
    const StageManifest& manifest,
    const MediaShape& stage,
    Resources& resources,
    const std::atomic_bool& cancel
) {
    if (auto error = session.admit(manifest, stage))
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
            const auto columns = t.rank == 1 ? 1 : std::size_t(t.shape[1]);
            Admission staging(resources, slice.stored_bytes);
            std::vector<std::uint8_t> raw(slice.stored_bytes);
            const auto offset = slice.first_row * columns * width;
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
                    require(
                        std::isfinite(value)
                            && (value >= 0 || !slice.name.ends_with(".weight_scale")),
                        "h3_cuda_scale"
                    );
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
    // Host-staged refiner over the text rows, used when it is not resident.
    auto host_refiner = [&] {
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
    };
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
    if (resident) {
        require(bool(resident->session), "h3_resident_session");
        resident_rows.emplace(*resources_, n * sizeof(std::uint32_t));
        token_groups.resize(n);
        for (std::size_t t = 0; t < n; ++t)
            token_groups[t] = std::uint32_t(time_ids[t] * 3 + input.tags[t]);
    }
    // The refiner, full blocks and final layer share one session. Hidden and
    // step inputs stay resident across admissions; only output rows return to
    // the host after the final layer.
    std::unique_ptr<MediaSession> held;
    auto close_held = [&](bool abort) {
        if (!held)
            return;
        auto session = std::move(held);
        detail::close_h3_resident(*session, resident->launchers, abort);
    };
    // The first resident stage uploads hidden and the step's shared inputs.
    auto upload_step = [&](std::size_t time_width) {
        detail::upload_h3_step_inputs(
            *held,
            hidden,
            time_width,
            current,
            t_embed.first(time_count * time_width),
            token_groups,
            {rotary.get(), n * 96}
        );
    };

    auto resident_block = [&](std::size_t block) {
        if (resident->attention_rejected || !resident->attention)
            return false;
        const bool retained = bool(held);
        if (!held)
            held = resident->session();
        require(bool(held), "h3_resident_session");
        auto described = describe_block(*shard_, turbo_.get(), block, cancel);
        std::optional<MediaPlanError> error;
        if (auto* rejected = std::get_if<MediaPlanError>(&described))
            error = std::move(*rejected);
        else {
            const auto& layout = std::get<detail::H3Block>(described);
            error = prepare(
                *held,
                *shard_,
                turbo_.get(),
                detail::h3_block_manifest(layout, n, groups, retained),
                detail::h3_block_shape(layout, n),
                *resources_,
                cancel
            );
        }
        if (error) {
            // After a resident refiner, hidden lives only in the session.
            if (block || retained)
                throw std::runtime_error("h3_resident_admission: " + error->detail);
            resident->attention_rejected = std::move(error);
            close_held(false);
            return false;
        }
        const auto& layout = std::get<detail::H3Block>(described);
        if (!retained)
            upload_step(layout.time_width);
        detail::execute_h3_block(*held, resident->launchers, layout, n, cancel);
        return true;
    };
    // Both refiner blocks open the update's session, before block 0. Only the
    // first admission may fall back; the host refiner then runs instead.
    auto resident_refiner = [&] {
        if (resident->refiner_rejected || !resident->refiner
            || resident->attention_rejected || !resident->attention)
            return false;
        held = resident->session();
        require(bool(held), "h3_resident_session");
        for (std::size_t block = 0; block < 2; ++block) {
            auto described = describe_refiner(*shard_, turbo_.get(), block, cancel);
            std::optional<MediaPlanError> error;
            if (auto* rejected = std::get_if<MediaPlanError>(&described))
                error = std::move(*rejected);
            else {
                const auto& layout = std::get<detail::H3Refiner>(described);
                error = prepare(
                    *held,
                    *shard_,
                    turbo_.get(),
                    detail::h3_refiner_manifest(layout, n, nt, time_count, block != 0),
                    detail::h3_refiner_shape(layout, n),
                    *resources_,
                    cancel
                );
            }
            if (error) {
                if (block)
                    throw std::runtime_error("h3_resident_admission: " + error->detail);
                resident->refiner_rejected = std::move(error);
                close_held(false);
                return false;
            }
            const auto& layout = std::get<detail::H3Refiner>(described);
            if (!block)
                upload_step(layout.time_width);
            detail::execute_h3_refiner(
                *held, resident->launchers, layout, n, nt, cancel
            );
            if (hook)
                hook("refiner_completed", block + 1);
        }
        return true;
    };
    // After the last resident block, the final layer downloads only the output
    // rows; hidden never returns to the host.
    auto resident_final = [&] {
        auto described = describe_final(*shard_, nv != 0, na != 0, cancel);
        std::optional<MediaPlanError> error;
        if (auto* rejected = std::get_if<MediaPlanError>(&described))
            error = std::move(*rejected);
        else {
            const auto& layout = std::get<detail::H3Final>(described);
            error = prepare(
                *held,
                *shard_,
                nullptr,
                detail::h3_final_manifest(layout, n, nv, na, time_count),
                detail::h3_final_shape(layout, n),
                *resources_,
                cancel
            );
        }
        if (error)
            throw std::runtime_error("h3_resident_admission: " + error->detail);
        std::vector<std::uint32_t> ids(n);
        for (std::size_t t = 0; t < n; ++t)
            ids[t] = std::uint32_t(time_ids[t]);
        detail::upload_h3_final_inputs(*held, ids);
        detail::execute_h3_final(
            *held,
            resident->launchers,
            std::get<detail::H3Final>(described),
            n,
            nt,
            video_out,
            audio_out,
            cancel
        );
        close_held(false);
        finite(video_out);
        finite(audio_out);
    };
    // FF-only fallback: one fresh session per block, finished before the next
    // host-staged operation. The block's AdaLN rows come from the host.
    auto resident_feed_forward = [&](std::size_t block) {
        if (resident->rejected)
            return false;
        auto session = resident->session();
        require(bool(session), "h3_resident_session");
        try {
            auto described =
                describe_feed_forward(*shard_, turbo_.get(), block, cancel);
            std::optional<MediaPlanError> error;
            if (auto* rejected = std::get_if<MediaPlanError>(&described))
                error = std::move(*rejected);
            else {
                const auto& layout = std::get<detail::H3FeedForward>(described);
                error = prepare(
                    *session,
                    *shard_,
                    turbo_.get(),
                    detail::h3_feed_forward_manifest(layout, n, groups),
                    detail::h3_feed_forward_shape(layout, n),
                    *resources_,
                    cancel
                );
            }
            if (error) {
                if (block)
                    throw std::runtime_error("h3_resident_admission: " + error->detail);
                resident->rejected = std::move(error);
                session->finish();
                return false;
            }
            const auto& layout = std::get<detail::H3FeedForward>(described);
            detail::upload_h3_feed_forward_inputs(
                *session,
                layout,
                normalized,
                current,
                adaln.first(groups * 6 * hidden),
                token_groups
            );
            detail::execute_h3_feed_forward(
                *session, resident->launchers, layout, n, current, cancel
            );
            detail::close_h3_resident(*session, resident->launchers);
        } catch (...) {
            rethrow_after_cleanup(std::current_exception(), [&] {
                detail::close_h3_resident(*session, resident->launchers, true);
            });
        }
        finite(current);
        return true;
    };
    bool final_done = false;
    try {
        if (!resident || !resident_refiner())
            host_refiner();
        for (std::size_t b = 0; b < layers; ++b) {
            stop(cancel);
            auto p = "blocks." + std::to_string(b);
            if (resident && resident_block(b)) {
                if (hook)
                    hook("block_completed", b + 1);
                continue;
            }
            linear(
                p + ".adaln_proj.linear",
                t_embed.first(time_count * 2688),
                2688,
                96768,
                adaln.first(time_count * 96768),
                true
            );
            norm(
                *shard_, p + ".norm1.weight", current, normalized, norm_weight, cancel
            );
            modulate(0);
            attn(p + ".attn", n, true);
            residual(2);
            norm(
                *shard_, p + ".norm2.weight", current, normalized, norm_weight, cancel
            );
            modulate(3);
            if (!resident || !resident_feed_forward(b)) {
                feed(p + ".mlp", n);
                residual(5);
            }
            if (hook)
                hook("block_completed", b + 1);
        }
        if (held) {
            resident_final();
            final_done = true;
        }
    } catch (...) {
        rethrow_after_cleanup(std::current_exception(), [&] { close_held(true); });
    }
    if (!final_done) {
        linear(
            "final_layer.adaln_proj.linear",
            t_embed.first(time_count * 2688),
            2688,
            hidden * 2,
            adaln.first(time_count * hidden * 2),
            true
        );
        norm(
            *shard_, "final_layer.norm.weight", current, normalized, norm_weight, cancel
        );
        for (std::size_t t = 0; t < n; ++t)
            for (std::size_t c = 0; c < hidden; ++c)
                normalized[t * hidden + c] =
                    normalized[t * hidden + c]
                        * (1 + adaln[(time_ids[t] * 2 + 1) * hidden + c])
                    + adaln[time_ids[t] * 2 * hidden + c];
        // Compute selected rows only; dense F64 reductions have no batch-dependent
        // kernel choice.
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
    }
    stop(cancel);
    shard_->check_unchanged();
    if (turbo_)
        turbo_->check_unchanged();
    if (hook)
        hook("completed", n);
    stop(cancel);
}
}  // namespace kadan::video
