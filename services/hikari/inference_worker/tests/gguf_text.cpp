// Component units for the GGUF text layer math against independent naive
// formulas. No checkpoint is opened and no model runs.
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

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
}

void activations() {
    near(kadan::text::softplus(30), 30, "softplus_linear");
    near(kadan::text::softplus(0), std::log(2.f), "softplus");
    near(kadan::text::silu(1), 1 / (1 + std::exp(-1.f)), "silu");
    near(kadan::text::sigmoid(0), .5f, "sigmoid");
}
}  // namespace

int main() {
    try {
        norms();
        rotation();
        delta();
        attention();
        placement();
        activations();
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
