// GGUF text model plumbing; arithmetic follows llama.cpp
// 6420cf5dd588c68e48657a0bbd82bb3bd5ac7209 (ggml-cpu ops.cpp).
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <stdexcept>

#include "cpu_matvec.hpp"
#include "gguf_model.hpp"
#include "kadan/gguf_quantization.hpp"
#include "kadan/weight_inventory.hpp"

namespace kadan::text {
void need(bool condition, const char* error) {
    if (!condition)
        throw std::runtime_error(error);
}

Workers::Workers(std::size_t threads) {
    try {
        for (std::size_t i = 0; i < threads; ++i)
            threads_.emplace_back([this] { work(); });
    } catch (...) {
        {
            std::lock_guard lock(mutex_);
            stop_ = true;
        }
        start_.notify_all();
        for (auto& thread : threads_)
            thread.join();
        throw;
    }
}

Workers::~Workers() {
    {
        std::lock_guard lock(mutex_);
        stop_ = true;
    }
    start_.notify_all();
    for (auto& thread : threads_)
        thread.join();
}

void Workers::drain() {
    for (std::size_t i; (i = next_.fetch_add(1)) < count_;)
        try {
            task_(i);
        } catch (...) {
            std::lock_guard lock(mutex_);
            if (!error_)
                error_ = std::current_exception();
        }
}

void Workers::work() {
    std::size_t seen = 0;
    for (;;) {
        {
            std::unique_lock lock(mutex_);
            start_.wait(lock, [&] { return stop_ || generation_ != seen; });
            if (stop_)
                return;
            seen = generation_;
        }
        drain();
        std::lock_guard lock(mutex_);
        if (--active_ == 0)
            done_.notify_one();
    }
}

void Workers::start(std::size_t count, std::function<void(std::size_t)> body) {
    need(!started_, "gguf_text_workers_busy");
    if (threads_.empty()) {
        for (std::size_t i = 0; i < count; ++i)
            body(i);
        return;
    }
    {
        std::lock_guard lock(mutex_);
        task_ = std::move(body);
        count_ = count;
        next_ = 0;
        error_ = nullptr;
        active_ = threads_.size();
        ++generation_;
        started_ = true;
    }
    start_.notify_all();
}

void Workers::wait() {
    if (!started_)
        return;
    std::unique_lock lock(mutex_);
    done_.wait(lock, [&] { return active_ == 0; });
    started_ = false;
    task_ = {};
    if (error_)
        std::rethrow_exception(std::exchange(error_, nullptr));
}

void Workers::run(std::size_t count, const std::function<void(std::size_t)>& body) {
    if (count < 2 || threads_.empty()) {
        for (std::size_t i = 0; i < count; ++i)
            body(i);
        return;
    }
    start(count, body);
    drain();
    wait();
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
    // Only routed experts move between tiers; every other matrix is pinned.
    decision = placement_.use(key, bytes, matrices_.at(key.tensor).experts == 1);
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
    const Projection request{name, input, expert};
    return std::move(project(compute, std::span(&request, 1)).front());
}

std::vector<Vec> GgufWeights::project(
    DenseCompute& compute, std::span<const Projection> requests
) const {
    check_cancel();
    const auto started = std::chrono::steady_clock::now();
    gguf_->check_unchanged();

    struct Prepared {
        const Matrix* matrix;
        const checkpoint::GgufTensor* tensor;
        SliceKey key;
        Bytes bytes, row_bytes;
        bool device;
        WeightIdentity identity;
        DenseCompute::GgufWeightSource source;
    };

    std::vector<Prepared> prepared;
    prepared.reserve(requests.size());
    std::vector<Vec> outputs(requests.size());
    for (std::size_t i = 0; i < requests.size(); ++i) {
        const auto& request = requests[i];
        const auto it = matrices_.find(request.name);
        need(it != matrices_.end(), "gguf_text_missing_matrix");
        const auto& m = it->second;
        need(
            !request.input.empty() && request.input.size() % m.columns == 0
                && request.expert < m.experts,
            "gguf_text_projection_shape"
        );
        const auto& tensor = gguf_->tensor(request.name);
        need(tensor.type == 0 || gguf::encoding(tensor.type), "gguf_text_encoding");
        SliceKey key{request.name, request.expert};
        const Bytes bytes = tensor.bytes / m.experts;
        Placement::Decision decision;
        place(compute, key, bytes, decision);
        // A later slice of this batch may displace an earlier one.
        for (const auto& evicted : decision.device_evicted)
            for (auto& earlier : prepared)
                if (earlier.key == evicted)
                    earlier.device = false;
        outputs[i].resize(request.input.size() / m.columns * m.rows);
        const auto row_bytes = bytes / m.rows;
        // RAM slices are looked up when read: later placements may evict them.
        auto source = [this,
                       key,
                       row_bytes](std::size_t first, std::span<std::byte> to) {
            const auto cached = ram_.find(key);
            if (cached != ram_.end())
                std::memcpy(
                    to.data(), cached->second.data() + first * row_bytes, to.size()
                );
            else
                gguf_->read_packed_rows(key.tensor, key.expert, first, to, *cancel_);
        };
        auto identity = identities_.find(key);
        if (identity == identities_.end())
            identity =
                identities_.emplace(key, gguf_->identity(request.name, request.expert))
                    .first;
        prepared.push_back(
            {&m,
             &tensor,
             std::move(key),
             bytes,
             row_bytes,
             decision.device,
             identity->second,
             std::move(source)}
        );
    }
    // Routed experts held in host RAM run on the host workers while the device
    // batch runs, as long as the host is expected to finish no later than the
    // device's PCIe stream (FreeToken's split of misses between CPU execution
    // and transfer). Resident, newly admitted and disk slices stay on the device.
    std::vector<bool> on_host(requests.size());
    Bytes host_bytes = 0, stream_bytes = 0;
    for (std::size_t i = 0; i < requests.size(); ++i) {
        const auto& p = prepared[i];
        if (p.device
            || requests[i].input.size() / p.matrix->columns
                   > DenseCompute::gguf_batch_rows)
            continue;
        const bool host = p.matrix->experts > 1 && ram_.contains(p.key)
                          && CpuMatvec::supported(p.tensor->type, p.matrix->columns);
        if (host
            && double(host_bytes + p.bytes) / host_rate_
                   <= double(stream_bytes + p.bytes) / stream_rate_) {
            on_host[i] = true;
            host_bytes += p.bytes;
        } else
            stream_bytes += p.bytes;
    }

    struct HostJob {
        std::size_t request;
        CpuMatvec product;
        const std::byte* weights;
    };

    std::vector<HostJob> host;
    std::vector<std::pair<std::size_t, std::size_t>> units;
    constexpr std::size_t unit_columns = 64;
    for (std::size_t i = 0; i < requests.size(); ++i)
        if (on_host[i]) {
            const auto& p = prepared[i];
            host.push_back(
                {i,
                 CpuMatvec(p.tensor->type, requests[i].input, p.matrix->columns),
                 ram_.at(p.key).data()}
            );
            for (std::size_t first = 0; first < p.matrix->rows; first += unit_columns)
                units.emplace_back(host.size() - 1, first);
        }
    std::atomic<std::size_t> remaining = units.size();
    std::chrono::steady_clock::time_point host_finished{};
    const auto host_started = std::chrono::steady_clock::now();

    // Every started unit finishes before this frame's data goes out of scope.
    struct Join {
        Workers* workers = nullptr;

        ~Join() {
            if (workers)
                try {
                    workers->wait();
                } catch (...) {
                }
        }
    } join;

    if (!units.empty()) {
        if (!workers_)
            parallel(0, [](std::size_t) {});
        workers_->start(units.size(), [&](std::size_t u) {
            const auto& [job, first] = units[u];
            const auto& h = host[job];
            const auto& p = prepared[h.request];
            h.product.run(
                h.weights,
                p.row_bytes,
                first,
                std::min(first + unit_columns, p.matrix->rows),
                outputs[h.request],
                p.matrix->rows
            );
            if (remaining.fetch_sub(1) == 1)
                host_finished = std::chrono::steady_clock::now();
        });
        join.workers = workers_.get();
    }
    std::vector<DenseCompute::GgufBatchJob> jobs;
    std::vector<std::size_t> batched;
    for (std::size_t i = 0; i < requests.size(); ++i) {
        const auto& p = prepared[i];
        if (on_host[i]
            || requests[i].input.size() / p.matrix->columns
                   > DenseCompute::gguf_batch_rows)
            continue;
        batched.push_back(i);
        jobs.push_back(
            {&p.identity,
             p.tensor->type == 0 ? std::nullopt : gguf::encoding(p.tensor->type),
             &p.source,
             requests[i].input,
             p.matrix->columns,
             p.matrix->rows,
             outputs[i],
             p.device}
        );
    }
    std::vector<bool> done(requests.size());
    if (!jobs.empty()) {
        const auto device_started = std::chrono::steady_clock::now();
        const auto result = compute.dense_gguf_batch(jobs, *cancel_);
        if (result != ProjectionResult::unsupported) {
            for (const auto i : batched) {
                need(
                    projection_output_finite(result, outputs[i]), "gguf_text_nonfinite"
                );
                done[i] = true;
            }
            // The whole batch time bounds the stream rate from below.
            if (stream_bytes >= (Bytes(1) << 20)) {
                const auto seconds =
                    std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - device_started
                    )
                        .count();
                stream_rate_ = std::clamp(
                    .8 * stream_rate_ + .2 * double(stream_bytes) / seconds, 1e8, 1e12
                );
            }
        }
    }
    if (join.workers) {
        join.workers = nullptr;
        workers_->wait();
        for (const auto& h : host) {
            need(
                std::all_of(
                    outputs[h.request].begin(),
                    outputs[h.request].end(),
                    [](float v) { return std::isfinite(v); }
                ),
                "gguf_text_nonfinite"
            );
            done[h.request] = true;
        }
        if (host_bytes >= (Bytes(1) << 20)) {
            const auto seconds =
                std::chrono::duration<double>(host_finished - host_started).count();
            if (seconds > 0)
                host_rate_ = std::clamp(
                    .8 * host_rate_ + .2 * double(host_bytes) / seconds, 1e8, 1e12
                );
        }
        timing_.host += host_bytes;
    }
    for (std::size_t i = 0; i < requests.size(); ++i) {
        if (done[i])
            continue;
        const auto& p = prepared[i];
        const auto& m = *p.matrix;
        const auto& request = requests[i];
        auto& output = outputs[i];
        ProjectionResult result;
        if (p.tensor->type == 0) {
            const auto reader = [&](std::size_t first, std::span<float> destination) {
                p.source(first / m.columns, std::as_writable_bytes(destination));
            };
            result = compute.dense_source(
                p.identity,
                reader,
                {},
                request.input,
                m.columns,
                m.rows,
                output,
                *cancel_
            );
        } else {
            const auto encoding = gguf::encoding(p.tensor->type);
            if (!encoding)
                throw std::runtime_error("gguf_text_encoding");
            // Slices without device residency stream without displacing hot tiles.
            result = compute.dense_gguf_source(
                p.identity,
                *encoding,
                p.source,
                {},
                request.input,
                m.columns,
                m.rows,
                output,
                *cancel_,
                false,
                p.device
            );
        }
        need(result != ProjectionResult::unsupported, "gguf_text_cuda_required");
        need(projection_output_finite(result, output), "gguf_text_nonfinite");
    }
    // Resident rows remain valid only while every file is unchanged.
    gguf_->check_unchanged();
    timing_.projection +=
        std::chrono::duration<double>(std::chrono::steady_clock::now() - started)
            .count();
    timing_.projections += requests.size();
    ++timing_.batches;
    for (std::size_t i = 0; i < prepared.size(); ++i)
        if (!prepared[i].device && !on_host[i])
            timing_.streamed += prepared[i].bytes;
    return outputs;
}

void GgufWeights::parallel(
    std::size_t count, const std::function<void(std::size_t)>& body
) const {
    if (!workers_) {
        const auto cores = std::max(1u, std::thread::hardware_concurrency());
        // About one thread per physical core, bounded; the caller is the last.
        workers_ =
            std::make_unique<Workers>(std::clamp<std::size_t>(cores / 2, 1, 16) - 1);
    }
    workers_->run(count, body);
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
    // The decayed state g_i * S[i][j] is recomputed with the same rounding in
    // both passes instead of being stored, so the state is read twice and
    // written once. delta[j] = (v[j] - sum_i S'[i][j] k[i]) * beta, row by row.
    float delta[256];
    need(dim <= std::size(delta), "gguf_text_delta_shape");
    std::copy(v.begin(), v.end(), delta);
    for (std::size_t i = 0; i < dim; ++i) {
        const float g = decay[decay.size() == 1 ? 0 : i], key = k[i];
        const float* row = state.data() + i * dim;
        for (std::size_t j = 0; j < dim; ++j)
            delta[j] -= row[j] * g * key;
    }
    for (std::size_t j = 0; j < dim; ++j)
        delta[j] *= beta;
    std::fill(out.begin(), out.end(), 0.f);
    for (std::size_t i = 0; i < dim; ++i) {
        const float g = decay[decay.size() == 1 ? 0 : i], key = k[i], query = q[i];
        float* row = state.data() + i * dim;
        for (std::size_t j = 0; j < dim; ++j) {
            row[j] = row[j] * g + key * delta[j];
            out[j] += row[j] * query;
        }
    }
}
}  // namespace kadan::text
