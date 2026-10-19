// Component units for the GGUF text layer math against independent naive
// formulas. No checkpoint is opened and no model runs.
#include <algorithm>
#include <cmath>
#include <cstring>
#include <ggml.h>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "../src/models/text/cpu_matvec.hpp"
#include "../src/models/text/gguf_model.hpp"

namespace {
using kadan::text::Vec;

void check(bool condition, const char* message) {
    if (!condition)
        throw std::runtime_error(message);
}

void near(float actual, float expected, const char* message, float tolerance = 1e-5f) {
    check(
        std::abs(actual - expected) <= tolerance * std::max(1.f, std::abs(expected)),
        message
    );
}

// Declines device attention so the host reference path runs.
struct HostCompute final : kadan::DenseCompute {
    void dense(
        std::span<const float>,
        std::span<const float>,
        std::span<const float>,
        std::size_t,
        std::size_t,
        std::span<float>,
        const std::atomic_bool&,
        bool
    ) override {
        throw std::runtime_error("unexpected_projection");
    }
};

void norms() {
    Vec x{3, 4, 6, 8};
    const Vec weight{1, 2};
    kadan::text::rms_norm(x, weight, 0);
    const float a = 1 / std::sqrt(12.5f), b = 1 / std::sqrt(50.f);
    near(x[0], 3 * a, "rms_norm_group0");
    near(x[1], 8 * a, "rms_norm_weight");
    near(x[2], 6 * b, "rms_norm_group1");
    Vec y{3, 4};
    kadan::text::l2_norm(y, 2, 0);
    near(y[0], .6f, "l2_norm");
    near(y[1], .8f, "l2_norm");
}

void rotation() {
    // Width 6 with four rotary values: pairs (0, 2) and (1, 3); 4 and 5 stay.
    Vec x{1, 2, 3, 4, 5, 6};
    kadan::text::rope(x, 6, 4, 3, 10000);
    const float t0 = 3, t1 = 3 * std::pow(10000.f, -.5f);
    near(x[0], 1 * std::cos(t0) - 3 * std::sin(t0), "rope_pair0");
    near(x[2], 1 * std::sin(t0) + 3 * std::cos(t0), "rope_pair0");
    near(x[1], 2 * std::cos(t1) - 4 * std::sin(t1), "rope_pair1");
    near(x[3], 2 * std::sin(t1) + 4 * std::cos(t1), "rope_pair1");
    check(x[4] == 5 && x[5] == 6, "rope_partial");
    Vec origin{1, 2, 3, 4};
    kadan::text::rope(origin, 4, 4, 0, 10000);
    check(origin == Vec({1, 2, 3, 4}), "rope_position_zero");
}

void delta() {
    constexpr std::size_t dim = 3;
    for (const bool per_key : {false, true}) {
        Vec state(dim * dim), naive(dim * dim);
        const std::array<Vec, 2> q{Vec{.5f, -.25f, 1}, Vec{.1f, .2f, -.3f}},
            k{Vec{.3f, .4f, -.2f}, Vec{-.6f, .1f, .5f}},
            v{Vec{1, -2, .5f}, Vec{.25f, .75f, -1}};
        const Vec decay = per_key ? Vec{.9f, .8f, .7f} : Vec{.95f};
        const float beta = .6f;
        for (std::size_t t = 0; t < 2; ++t) {
            Vec out(dim);
            kadan::text::delta_rule(state, q[t], k[t], v[t], decay, beta, out);
            // S[i][j] = g_i S[i][j]; S += k (beta (v - S^T k))^T; o = S^T q.
            for (std::size_t i = 0; i < dim; ++i)
                for (std::size_t j = 0; j < dim; ++j)
                    naive[i * dim + j] *= decay[per_key ? i : 0];
            Vec update(dim);
            for (std::size_t j = 0; j < dim; ++j) {
                float projected = 0;
                for (std::size_t i = 0; i < dim; ++i)
                    projected += naive[i * dim + j] * k[t][i];
                update[j] = beta * (v[t][j] - projected);
            }
            for (std::size_t i = 0; i < dim; ++i)
                for (std::size_t j = 0; j < dim; ++j)
                    naive[i * dim + j] += k[t][i] * update[j];
            for (std::size_t j = 0; j < dim; ++j) {
                float expected = 0;
                for (std::size_t i = 0; i < dim; ++i)
                    expected += naive[i * dim + j] * q[t][i];
                near(out[j], expected, "delta_output");
            }
        }
        for (std::size_t i = 0; i < state.size(); ++i)
            near(state[i], naive[i], "delta_state");
    }
}

// The fused two-pass recurrence equals the three-pass form bit for bit.
void delta_passes() {
    constexpr std::size_t dim = 128;
    std::uint32_t seed = 12345;
    auto next = [&] {
        seed = seed * 1664525u + 1013904223u;
        return float(int(seed >> 8) - (1 << 23)) / float(1 << 23);
    };
    for (const bool per_key : {false, true}) {
        Vec state(dim * dim), reference(dim * dim);
        for (std::size_t i = 0; i < state.size(); ++i)
            state[i] = reference[i] = next();
        for (int t = 0; t < 4; ++t) {
            Vec q(dim), k(dim), v(dim), decay(per_key ? dim : 1), out(dim),
                expected(dim);
            for (auto* x : {&q, &k, &v})
                for (auto& value : *x)
                    value = next();
            for (auto& g : decay)
                g = .5f + .5f * std::abs(next());
            const float beta = .5f + .25f * next();
            kadan::text::delta_rule(state, q, k, v, decay, beta, out);
            for (std::size_t i = 0; i < dim; ++i)
                for (std::size_t j = 0; j < dim; ++j)
                    reference[i * dim + j] *= decay[per_key ? i : 0];
            Vec delta(v);
            for (std::size_t i = 0; i < dim; ++i)
                for (std::size_t j = 0; j < dim; ++j)
                    delta[j] -= reference[i * dim + j] * k[i];
            for (auto& d : delta)
                d *= beta;
            for (std::size_t i = 0; i < dim; ++i)
                for (std::size_t j = 0; j < dim; ++j) {
                    reference[i * dim + j] += k[i] * delta[j];
                    expected[j] += reference[i * dim + j] * q[i];
                }
            check(out == expected && state == reference, "delta_passes");
        }
    }
}

void workers() {
    kadan::text::Workers pool(3);
    for (int round = 0; round < 50; ++round) {
        std::vector<int> seen(37);
        pool.run(seen.size(), [&](std::size_t i) { ++seen[i]; });
        check(
            std::all_of(seen.begin(), seen.end(), [](int n) { return n == 1; }),
            "workers_once"
        );
    }
    bool thrown = false;
    try {
        pool.run(9, [](std::size_t i) {
            if (i == 4)
                throw std::runtime_error("workers_error");
        });
    } catch (const std::runtime_error& error) {
        thrown = std::string(error.what()) == "workers_error";
    }
    check(thrown, "workers_rethrow");
    int total = 0;
    pool.run(1, [&](std::size_t) { ++total; });
    check(total == 1, "workers_after_error");
    // start() leaves the caller free until wait().
    std::vector<int> later(23);
    pool.start(later.size(), [&](std::size_t i) { ++later[i]; });
    pool.wait();
    check(
        std::all_of(later.begin(), later.end(), [](int n) { return n == 1; }),
        "workers_async"
    );
    pool.start(3, [](std::size_t i) {
        if (i == 1)
            throw std::runtime_error("workers_async_error");
    });
    thrown = false;
    try {
        pool.wait();
    } catch (const std::runtime_error&) {
        thrown = true;
    }
    check(thrown, "workers_async_rethrow");
}

// ggml-cpu products of packed rows against double-precision dot products of
// the same rows decoded by ggml.
void cpu_matvec() {
    using kadan::text::CpuMatvec;
    constexpr std::size_t in = 512, out = 37, rows = 3;
    std::uint32_t seed = 99;
    auto next = [&] {
        seed = seed * 1664525u + 1013904223u;
        return float(int(seed >> 8) - (1 << 23)) / float(1 << 23);
    };
    Vec input(rows * in);
    for (auto& v : input)
        v = next();
    for (const auto type :
         {GGML_TYPE_F32,
          GGML_TYPE_Q8_0,
          GGML_TYPE_Q4_K,
          GGML_TYPE_Q6_K,
          GGML_TYPE_IQ4_XS}) {
        check(CpuMatvec::supported(type, in), "cpu_matvec_supported");
        Vec values(out * in);
        for (auto& v : values)
            v = next() * .1f;
        const auto row_bytes = ggml_row_size(type, in);
        std::vector<std::byte> packed(out * row_bytes);
        if (type == GGML_TYPE_F32)
            std::memcpy(packed.data(), values.data(), packed.size());
        else
            ggml_quantize_chunk(
                type, values.data(), packed.data(), 0, out, in, nullptr
            );
        Vec decoded(values);
        if (type != GGML_TYPE_F32)
            ggml_get_type_traits(type)->to_float(
                packed.data(), decoded.data(), out * in
            );
        const CpuMatvec product(type, input, in);
        Vec output(rows * out);
        product.run(packed.data(), row_bytes, 0, 20, output, out);
        product.run(packed.data(), row_bytes, 20, out, output, out);
        double worst = 0, scale = 0;
        for (std::size_t r = 0; r < rows; ++r)
            for (std::size_t c = 0; c < out; ++c) {
                double expected = 0;
                for (std::size_t i = 0; i < in; ++i)
                    expected += double(decoded[c * in + i]) * input[r * in + i];
                scale = std::max(scale, std::abs(expected));
                worst = std::max(worst, std::abs(output[r * out + c] - expected));
            }
        // Activations are rounded to the 8-bit dot type, as llama.cpp does.
        check(
            worst <= (type == GGML_TYPE_F32 ? 1e-5 : 2e-2) * std::max(scale, 1.0),
            "cpu_matvec_value"
        );
    }
}

// Rows must be whole blocks of the weight type and of its dot type.
void cpu_matvec_blocks() {
    using kadan::text::CpuMatvec;
    check(CpuMatvec::supported(GGML_TYPE_IQ4_NL, 640), "cpu_matvec_iq4nl_640");
    check(!CpuMatvec::supported(GGML_TYPE_Q4_K, 640), "cpu_matvec_q4k_640");
    check(!CpuMatvec::supported(GGML_TYPE_COUNT, 256), "cpu_matvec_type_range");
}

void attention() {
    // Two query rows at positions 1 and 2, two heads sharing one KV head.
    constexpr std::size_t heads = 2, dim = 2, keys = 3;
    const Vec q{1, 0, 0, 1, .5f, .5f, -1, 2};
    const Vec k{1, 0, 0, 1, 1, 1};
    const Vec v{1, 2, 3, 4, 5, 6};
    Vec out(q.size());
    HostCompute compute;
    const std::atomic_bool cancel{false};
    const float scale = .7f;
    kadan::text::attend(
        compute, q, k, v, 2, keys, heads, 1, dim, 1, scale, out, cancel
    );
    for (std::size_t row = 0; row < 2; ++row)
        for (std::size_t h = 0; h < heads; ++h) {
            const auto visible = row + 2;
            Vec weight(visible);
            float total = 0;
            for (std::size_t t = 0; t < visible; ++t) {
                const float dot = q[(row * heads + h) * dim] * k[t * dim]
                                  + q[(row * heads + h) * dim + 1] * k[t * dim + 1];
                total += weight[t] = std::exp(dot * scale);
            }
            for (std::size_t d = 0; d < dim; ++d) {
                float expected = 0;
                for (std::size_t t = 0; t < visible; ++t)
                    expected += weight[t] / total * v[t * dim + d];
                near(out[(row * heads + h) * dim + d], expected, "attention_causal");
            }
        }
    bool rejected = false;
    try {
        kadan::text::attend(
            compute, q, k, v, 2, keys, heads, 1, dim, 0, scale, out, cancel
        );
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    check(rejected, "attention_offset_contract");
}

void placement() {
    using kadan::text::Placement;
    using kadan::text::SliceKey;
    const SliceKey a{"experts", 0}, b{"experts", 1}, c{"experts", 2},
        big{"embedding", 0};
    Placement tiers(8);
    tiers.budgets(100, 100);
    // Slices fill VRAM, then spill to RAM; one larger than both reads from disk.
    auto first = tiers.use(a, 60);
    check(first.device && !first.ram, "placement_device_fill");
    auto second = tiers.use(b, 60);
    check(!second.device && second.ram, "placement_ram_spill");
    auto oversized = tiers.use(big, 200);
    check(!oversized.device && !oversized.ram, "placement_disk_stream");
    // Equally used slices never displace each other.
    for (int i = 0; i < 4; ++i) {
        tiers.use(a, 60);
        check(!tiers.use(b, 60).device, "placement_no_thrash");
    }
    // A routinely routed slice displaces one that stopped being used.
    tiers.advance(64);
    Placement::Decision promoted;
    for (int i = 0; i < 4 && !promoted.device; ++i)
        promoted = tiers.use(b, 60);
    check(promoted.device && !promoted.ram, "placement_hot_promotion");
    check(
        promoted.device_evicted == std::vector<SliceKey>{a}, "placement_cold_eviction"
    );
    check(
        promoted.ram_evicted == std::vector<SliceKey>{b},
        "placement_promotion_frees_ram"
    );
    // The displaced slice returns through RAM on its next use.
    const auto returned = tiers.use(a, 60);
    check(!returned.device && returned.ram, "placement_cold_to_ram");
    tiers.use(c, 30);
    const auto stats = tiers.stats();
    check(stats.device == 90 && stats.ram == 60, "placement_accounting");
    check(stats.device_slices == 2 && stats.ram_slices == 1, "placement_slices");

    // Pinned slices displace any expert and are never displaced by one.
    Placement pinned(8);
    pinned.budgets(100, 100);
    for (int i = 0; i < 8; ++i)
        pinned.use(a, 60);
    const auto dense = pinned.use(big, 50, true);
    check(
        dense.device && dense.device_evicted == std::vector<SliceKey>{a},
        "placement_pin"
    );
    for (int i = 0; i < 16; ++i)
        check(!pinned.use(b, 60).device, "placement_pin_kept");
    check(pinned.use(c, 40).device, "placement_pin_room");

    // Admission cost does not grow with the number of cold slices.
    Placement many(8);
    many.budgets(1000, 0);
    for (std::size_t e = 0; e < 200000; ++e)
        many.use({"experts", e}, 10);
    const auto large = many.stats();
    check(large.device == 1000 && large.device_slices == 100, "placement_many");
}

void activations() {
    near(kadan::text::softplus(30), 30, "softplus_linear");
    near(kadan::text::softplus(0), std::log(2.f), "softplus");
    near(kadan::text::silu(1), 1 / (1 + std::exp(-1.f)), "silu");
    near(kadan::text::sigmoid(0), .5f, "sigmoid");
}
}  // namespace

void projection_timing_aggregates() {
    kadan::text::ProjectionTiming timing;
    using Path = kadan::text::ProjectionTiming::Path;
    timing.add(8, Path::packed);
    timing.add(1, Path::host);
    timing.add(9, Path::large_rows, .25);
    timing.add(64, Path::large_rows, .75);
    timing.add(8, Path::batch_unsupported, .5);
    check(
        timing.rows == 90 && timing.packed_calls == 1 && timing.host_calls == 1,
        "projection_rows_and_completed_paths"
    );
    check(
        timing.large_row_calls == 2 && timing.large_row_seconds == 1,
        "projection_large_row_cost"
    );
    check(
        timing.unsupported_calls == 1 && timing.unsupported_seconds == .5,
        "projection_eligible_batch_rejection_cost"
    );
}

int main() {
    projection_timing_aggregates();
    try {
        norms();
        rotation();
        delta();
        delta_passes();
        workers();
        cpu_matvec();
        cpu_matvec_blocks();
        attention();
        placement();
        activations();
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
