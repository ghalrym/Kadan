// GGUF text model plumbing; arithmetic follows llama.cpp
// 6420cf5dd588c68e48657a0bbd82bb3bd5ac7209 (ggml-cpu ops.cpp).
#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

#include "gguf_model.hpp"
#include "kadan/gguf_quantization.hpp"
#include "kadan/weight_inventory.hpp"

namespace kadan::text {
void need(bool condition, const char* error) {
    if (!condition)
        throw std::runtime_error(error);
}

GgufWeights::GgufWeights(const char* root, const std::atomic_bool* cancel) {
    if (cancel)
        cancel_ = cancel;
    check_cancel();
    gguf_ = std::make_unique<checkpoint::GgufCheckpoint>(root, *cancel_);
    need(!gguf_->has("prism.hadamard.sign_values"), "gguf_text_hadamard_unsupported");
    for (std::size_t i = 0; i < gguf_->tensor_count(); ++i)
        packed_ = checkpoint::weight_bytes_add(packed_, gguf_->tensor_at(i).bytes);
}

void GgufWeights::check_cancel() const {
    need(!cancel_->load(), "gguf_text_cancelled");
}

void GgufWeights::bind(const std::string& name) {
    need(bound_.insert(name).second, "gguf_text_duplicate_binding");
}

void GgufWeights::vector(
    const std::string& name, std::initializer_list<std::uint64_t> shape
) {
    const auto& tensor = gguf_->tensor(name);
    need(tensor.type == 0 && tensor.rank == shape.size(), "gguf_text_vector");
    std::size_t i = 0, count = 1;
    for (auto size : shape) {
        need(tensor.shape[i++] == size, "gguf_text_vector");
        count *= size;
    }
    need(count <= (64ULL << 20), "gguf_text_vector_bound");
    bind(name);
    vectors_.emplace(name, count);
}

void GgufWeights::matrix(
    const std::string& name, std::size_t rows, std::size_t columns, std::size_t experts
) {
    const auto& tensor = gguf_->tensor(name);
    need(
        tensor.rank == (experts == 1 ? 2 : 3) && tensor.shape[0] == columns
            && tensor.shape[1] == rows && tensor.shape[2] == experts
            && (tensor.type == 0 || gguf::encoding(tensor.type)),
        "gguf_text_matrix"
    );
    bind(name);
    matrices_.emplace(name, Matrix{rows, columns, experts});
    execution_ = checkpoint::weight_bytes_add(
        execution_, std::uint64_t(rows) * columns * experts * sizeof(float)
    );
}

void GgufWeights::table(
    const std::string& name, std::size_t rows, std::size_t columns, std::size_t experts
) {
    const auto& tensor = gguf_->tensor(name);
    need(
        tensor.rank == (experts == 1 ? 2 : 3) && tensor.shape[0] == columns
            && tensor.shape[1] == rows && tensor.shape[2] == experts
            && (tensor.type == 0 || gguf::encoding(tensor.type)),
        "gguf_text_table"
    );
    bind(name);
    matrices_.emplace(name, Matrix{rows, columns, experts});
}

Vec GgufWeights::decode(const std::string& name) const {
    const auto it = matrices_.find(name);
    need(it != matrices_.end(), "gguf_text_missing_matrix");
    const auto& m = it->second;
    const auto count = m.rows * m.columns;
    need(count * m.experts <= (64ULL << 20), "gguf_text_decode_bound");
    Vec output(count * m.experts);
    for (std::size_t e = 0; e < m.experts; ++e)
        gguf_->read_rows(
            name, e, 0, std::span(output).subspan(e * count, count), *cancel_
        );
    return output;
}

void GgufWeights::skip(std::string_view prefix) {
    skipped_.emplace_back(prefix);
}

void GgufWeights::finish() {
    for (std::size_t i = 0; i < gguf_->tensor_count(); ++i) {
        const auto& name = gguf_->tensor_at(i).name;
        const bool skipped =
            std::any_of(skipped_.begin(), skipped_.end(), [&](const auto& prefix) {
                return name.starts_with(prefix);
            });
        need(bound_.contains(name) != skipped, "gguf_text_unbound_tensor");
    }
}

Vec GgufWeights::read(const std::string& name) const {
    const auto it = vectors_.find(name);
    need(it != vectors_.end(), "gguf_text_missing_vector");
    Vec output(it->second);
    gguf_->read_vector(name, output, *cancel_);
    return output;
}

void GgufWeights::rows(
    const std::string& name, std::size_t first, std::span<float> output
) const {
    const auto it = matrices_.find(name);
    need(
        it != matrices_.end() && it->second.experts == 1
            && output.size() % it->second.columns == 0,
        "gguf_text_rows"
    );
    gguf_->read_rows(name, 0, first, output, *cancel_);
}

// Applies one placement decision: device evictions leave the banks, host
// evictions free RAM, and a newly RAM-placed slice is read once from disk.
void GgufWeights::place(
    DenseCompute& compute,
    const SliceKey& key,
    Bytes bytes,
    Placement::Decision& decision
) const {
    if (!budgeted_) {
        Bytes device = 0;
        for (const auto& placement : compute.weight_placement())
            device += placement.planned_bytes;
        // Without a device plan every slice is retained by the banks' own policy.
        placement_.budgets(device ? device / 10 * 9 : ~Bytes(0), ram_budget_);
        budgeted_ = true;
    }
    decision = placement_.use(key, bytes);
    if (!decision.device_evicted.empty()) {
        std::vector<WeightIdentity> released;
        released.reserve(decision.device_evicted.size());
        for (const auto& slice : decision.device_evicted)
            released.push_back(gguf_->identity(slice.tensor, slice.expert));
        compute.release_weights(released);
    }
    for (const auto& slice : decision.ram_evicted)
        ram_.erase(slice);
    if (decision.ram && !ram_.contains(key)) {
        const auto& m = matrices_.at(key.tensor);
        std::vector<std::byte> slice(bytes);
        gguf_->read_packed_rows(key.tensor, key.expert, 0, slice, *cancel_);
        if (gguf_->tensor(key.tensor).type == 0) {
            const std::span values(
                reinterpret_cast<const float*>(slice.data()), m.rows * m.columns
            );
            need(
                std::all_of(
                    values.begin(),
                    values.end(),
                    [](float v) { return std::isfinite(v); }
                ),
                "gguf_text_nonfinite_weight"
            );
        }
        ram_.emplace(key, std::move(slice));
    }
}

Vec GgufWeights::project(
    DenseCompute& compute,
    const std::string& name,
    std::span<const float> input,
    std::size_t expert
) const {
    const auto it = matrices_.find(name);
    need(it != matrices_.end(), "gguf_text_missing_matrix");
    const auto& m = it->second;
    need(
        !input.empty() && input.size() % m.columns == 0 && expert < m.experts,
        "gguf_text_projection_shape"
    );
    check_cancel();
    gguf_->check_unchanged();
    const auto& tensor = gguf_->tensor(name);
    const SliceKey key{name, expert};
    const Bytes bytes = tensor.bytes / m.experts, row_bytes = bytes / m.rows;
    Placement::Decision decision;
    place(compute, key, bytes, decision);
    const auto cached = ram_.find(key);
    const std::byte* host = cached == ram_.end() ? nullptr : cached->second.data();
    Vec output(input.size() / m.columns * m.rows);
    const auto identity = gguf_->identity(name, expert);
    ProjectionResult result;
    if (tensor.type == 0) {
        const auto reader = [&](std::size_t first, std::span<float> destination) {
            if (host)
                std::memcpy(
                    destination.data(),
                    host + first * sizeof(float),
                    destination.size_bytes()
                );
            else
                gguf_->read_rows(
                    name, expert, first / m.columns, destination, *cancel_
                );
        };
        result = compute.dense_source(
            identity, reader, {}, input, m.columns, m.rows, output, *cancel_
        );
    } else {
        const auto reader = [&](std::size_t first, std::span<std::byte> destination) {
            if (host)
                std::memcpy(
                    destination.data(), host + first * row_bytes, destination.size()
                );
            else
                gguf_->read_packed_rows(name, expert, first, destination, *cancel_);
        };
        const auto encoding = gguf::encoding(tensor.type);
        if (!encoding)
            throw std::runtime_error("gguf_text_encoding_unsupported");
        // Slices without device residency stream without displacing hot tiles.
        result = compute.dense_gguf_source(
            identity,
            *encoding,
            reader,
            {},
            input,
            m.columns,
            m.rows,
            output,
            *cancel_,
            false,
            decision.device
        );
    }
    need(result != ProjectionResult::unsupported, "gguf_text_cuda_required");
    need(projection_output_finite(result, output), "gguf_text_nonfinite");
    // Resident rows remain valid only while every file is unchanged.
    gguf_->check_unchanged();
    return output;
}

void GgufWeights::check_unchanged() const {
    gguf_->check_unchanged();
}

float sigmoid(float x) {
    return 1 / (1 + std::exp(-x));
}

float silu(float x) {
    return x / (1 + std::exp(-x));
}

float softplus(float x) {
    return x > 20 ? x : std::log1p(std::exp(x));
}

void rms_norm(std::span<float> x, std::span<const float> weight, float eps) {
    const auto width = weight.size();
    need(width && x.size() % width == 0, "gguf_text_norm_shape");
    for (std::size_t at = 0; at < x.size(); at += width) {
        double sum = 0;
        for (std::size_t i = 0; i < width; ++i)
            sum += double(x[at + i]) * x[at + i];
        const float scale = 1 / std::sqrt(float(sum / double(width)) + eps);
        for (std::size_t i = 0; i < width; ++i)
            x[at + i] = x[at + i] * scale * weight[i];
    }
    need(
        std::all_of(x.begin(), x.end(), [](float v) { return std::isfinite(v); }),
        "gguf_text_norm_nonfinite"
    );
}

void rms_norm(std::span<float> x, std::size_t width, float eps) {
    need(width && x.size() % width == 0, "gguf_text_norm_shape");
    for (std::size_t at = 0; at < x.size(); at += width) {
        double sum = 0;
        for (std::size_t i = 0; i < width; ++i)
            sum += double(x[at + i]) * x[at + i];
        const float scale = 1 / std::sqrt(float(sum / double(width)) + eps);
        for (std::size_t i = 0; i < width; ++i)
            x[at + i] *= scale;
    }
}

void l2_norm(std::span<float> x, std::size_t width, float eps) {
    // ggml_rms_norm(x, eps / n) / sqrt(n) == x / sqrt(sum(x^2) + eps).
    rms_norm(x, width, eps / float(width));
    const float scale = 1 / std::sqrt(float(width));
    for (auto& v : x)
        v *= scale;
}

void rope(
    std::span<float> x,
    std::size_t width,
    std::size_t rotary,
    std::size_t position,
    float base
) {
    need(
        width && rotary && rotary % 2 == 0 && rotary <= width && x.size() % width == 0,
        "gguf_text_rope_shape"
    );
    const float theta_scale = std::pow(base, -2.0f / float(rotary));
    const auto half = rotary / 2;
    for (std::size_t at = 0; at < x.size(); at += width) {
        float theta = float(position);
        for (std::size_t i = 0; i < half; ++i) {
            const float c = std::cos(theta), s = std::sin(theta);
            const float a = x[at + i], b = x[at + i + half];
            x[at + i] = a * c - b * s;
            x[at + i + half] = a * s + b * c;
            theta *= theta_scale;
        }
    }
}

void attend(
    DenseCompute& compute,
    std::span<const float> q,
    std::span<const float> k,
    std::span<const float> v,
    std::size_t queries,
    std::size_t keys,
    std::size_t heads,
    std::size_t kv_heads,
    std::size_t dim,
    std::size_t offset,
    float scale,
    std::span<float> out,
    const std::atomic_bool& cancel
) {
    need(
        queries && heads && kv_heads && heads % kv_heads == 0 && dim
            && offset + queries == keys && q.size() == queries * heads * dim
            && k.size() >= keys * kv_heads * dim && v.size() >= k.size()
            && out.size() == q.size(),
        "gguf_text_attention_shape"
    );
    k = k.first(keys * kv_heads * dim);
    v = v.first(keys * kv_heads * dim);
    // The device divides by sqrt(dim); fold the model's scale into the queries.
    Vec scaled(q.begin(), q.end());
    const float factor = scale * std::sqrt(float(dim));
    for (auto& x : scaled)
        x *= factor;
    if (compute.attend(
            scaled,
            k,
            v,
            queries,
            keys,
            heads,
            kv_heads,
            dim,
            queries,
            0,
            offset,
            out,
            cancel
        ))
        return;
    Vec scores(keys);
    for (std::size_t row = 0; row < queries; ++row)
        for (std::size_t h = 0; h < heads; ++h) {
            need(!cancel.load(), "gguf_text_cancelled");
            const auto kv = h / (heads / kv_heads);
            const auto* query = q.data() + (row * heads + h) * dim;
            const auto visible = offset + row + 1;
            float maximum = -INFINITY;
            for (std::size_t t = 0; t < visible; ++t) {
                const auto* key = k.data() + (t * kv_heads + kv) * dim;
                float dot = 0;
                for (std::size_t d = 0; d < dim; ++d)
                    dot += query[d] * key[d];
                scores[t] = dot * scale;
                maximum = std::max(maximum, scores[t]);
            }
            double total = 0;
            for (std::size_t t = 0; t < visible; ++t) {
                scores[t] = std::exp(scores[t] - maximum);
                total += scores[t];
            }
            auto* y = out.data() + (row * heads + h) * dim;
            std::fill(y, y + dim, 0.f);
            for (std::size_t t = 0; t < visible; ++t) {
                const float p = float(scores[t] / total);
                const auto* value = v.data() + (t * kv_heads + kv) * dim;
                for (std::size_t d = 0; d < dim; ++d)
                    y[d] += p * value[d];
            }
        }
}

void delta_rule(
    std::span<float> state,
    std::span<const float> q,
    std::span<const float> k,
    std::span<const float> v,
    std::span<const float> decay,
    float beta,
    std::span<float> out
) {
    const auto dim = k.size();
    need(
        dim && q.size() == dim && v.size() == dim && out.size() == dim
            && state.size() == dim * dim && (decay.size() == 1 || decay.size() == dim),
        "gguf_text_delta_shape"
    );
    for (std::size_t i = 0; i < dim; ++i) {
        const float g = decay[decay.size() == 1 ? 0 : i];
        for (std::size_t j = 0; j < dim; ++j)
            state[i * dim + j] *= g;
    }
    // delta[j] = (v[j] - sum_i S[i][j] k[i]) * beta, accumulated row by row.
    Vec delta(v.begin(), v.end());
    for (std::size_t i = 0; i < dim; ++i)
        for (std::size_t j = 0; j < dim; ++j)
            delta[j] -= state[i * dim + j] * k[i];
    for (auto& d : delta)
        d *= beta;
    std::fill(out.begin(), out.end(), 0.f);
    for (std::size_t i = 0; i < dim; ++i)
        for (std::size_t j = 0; j < dim; ++j) {
            state[i * dim + j] += k[i] * delta[j];
            out[j] += state[i * dim + j] * q[i];
        }
}
}  // namespace kadan::text
