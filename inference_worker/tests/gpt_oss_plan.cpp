#include "kadan/gpt_oss_plan.hpp"

#include <cassert>
#include <cmath>
#include <iostream>
#include <stdexcept>

using namespace kadan::gpt_oss;

namespace {
constexpr std::size_t GiB = 1024ULL * 1024 * 1024;

bool close(float a, float b) {
    return std::abs(a - b) <= 1e-6F * std::abs(b);
}

// Layer ranges are contiguous and end on the head device.
void contiguous(const Placement& p) {
    assert(p.devices.back() == p.head);
    for (std::size_t i = 1; i < p.devices.size(); ++i)
        for (std::size_t j = i; j < p.devices.size(); ++j)
            assert(
                p.devices[i] == p.devices[i - 1] || p.devices[j] != p.devices[i - 1]
            );
}

void within(const Placement& p, std::size_t budget, std::size_t headroom) {
    for (const auto& [device, bytes] : p.allocations) {
        (void)device;
        assert(bytes.weights + bytes.state + headroom <= budget);
    }
}

std::size_t host_experts(const Config& c, const Placement& p) {
    std::size_t total = 0;
    for (auto resident : p.resident) {
        assert(resident <= c.experts);
        total += c.experts - resident;
    }
    return total;
}
}  // namespace

int main() {
    assert(expert_bytes == 13219200 && expert_bytes % 16 == 0);
    assert(qkv_rows == 5120 && mxfp4_row_bytes == 1530);
    assert(layer_slots(0, 65536) == 128 && layer_slots(1, 65536) == 65536);
    assert(layer_slots(0, 64) == 64);
    assert(layer_expert_bytes(3) % alignment == 0);
    assert(layer_expert_bytes(3) >= 3 * expert_bytes);

    // Independent FP32 evaluation of Transformers' YaRN parameters.
    const auto r = rope();
    assert(close(r.scaling, 1.34657359F));
    assert(r.inverse_frequency[0] == 1.F);
    assert(close(r.inverse_frequency[5], 0.155322984F));
    assert(close(r.inverse_frequency[10], 0.0193350017F));
    assert(close(r.inverse_frequency[20], 1.8188337e-05F));
    assert(close(r.inverse_frequency[31], 3.0235114e-07F));

    const Config small{24, 32}, large{36, 128};
    const std::size_t headroom = 512ULL << 20;

    // GPT-OSS 20B fits one 24 GiB device completely.
    const auto one = place(small, 65536, {{0, 24 * GiB}, {1, 24 * GiB}}, headroom);
    assert(one.allocations.size() == 1 && one.head == 0);
    assert(host_experts(small, one) == 0 && one.host_bytes == embedding_bytes());
    contiguous(one);
    within(one, 24 * GiB, headroom);

    // A larger second device becomes the head and keeps the whole model.
    const auto larger = place(small, 4096, {{0, 16 * GiB}, {1, 24 * GiB}}, headroom);
    assert(larger.head == 1 && larger.allocations.size() == 1);

    // Two small devices share complete residency in contiguous ranges.
    const auto split = place(small, 4096, {{0, 8 * GiB}, {1, 8 * GiB}}, headroom);
    assert(split.allocations.size() == 2 && host_experts(small, split) == 0);
    contiguous(split);
    within(split, 8 * GiB, headroom);

    // GPT-OSS 120B on two 24 GiB devices keeps the rest of its experts mapped.
    const auto offload = place(large, 32768, {{0, 24 * GiB}, {1, 24 * GiB}}, headroom);
    assert(offload.allocations.size() == 2);
    const auto mapped = host_experts(large, offload);
    assert(mapped > 0 && mapped < large.layers * large.experts);
    assert(offload.host_bytes == embedding_bytes() + mapped * expert_bytes);
    contiguous(offload);
    within(offload, 24 * GiB, headroom);
    for (auto resident : offload.resident)
        assert(resident > 0);

    // Enough memory makes 120B completely resident.
    const auto resident = place(large, 4096, {{0, 80 * GiB}}, headroom);
    assert(host_experts(large, resident) == 0);

    // Deterministic for identical inputs.
    const auto again = place(large, 32768, {{0, 24 * GiB}, {1, 24 * GiB}}, headroom);
    assert(again.devices == offload.devices && again.resident == offload.resident);

    bool rejected = false;
    try {
        place(large, 131072, {{0, 4 * GiB}}, headroom);
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    assert(rejected);
    rejected = false;
    try {
        place(small, 0, {{0, 24 * GiB}}, headroom);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    assert(rejected);
    std::cout << "GPT-OSS placement and rotary parameter checks passed\n";
}
