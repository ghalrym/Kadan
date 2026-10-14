// GPT-OSS execution sizes and rotary parameters. Rotary equations follow
// Transformers modeling_rope_utils._compute_yarn_parameters (truncate=False)
// and the OpenAI gpt_oss reference RotaryEmbedding.
#include "kadan/gpt_oss_plan.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <stdexcept>

#include "kadan/auto_placement.hpp"

namespace kadan::gpt_oss {
namespace {
using model::checked_add;

void require(bool value, const char* error) {
    if (!value)
        throw std::invalid_argument(error);
}

std::size_t bf16(std::size_t count) {
    return aligned(count * 2);
}

std::size_t floats(std::size_t count) {
    return aligned(count * 4);
}

std::size_t chunks(std::size_t capacity) {
    return (capacity + attention_chunk - 1) / attention_chunk;
}

void validate(const Config& c, std::size_t capacity) {
    require(
        (c.layers == 24 && c.experts == 32) || (c.layers == 36 && c.experts == 128),
        "gpt_oss_dimensions"
    );
    require(capacity > 0 && capacity <= Config::max_context, "gpt_oss_capacity");
}

struct Usage {
    int device;
    std::size_t room, weights = 0, state = 0, layers = 0;

    std::size_t used() const {
        return checked_add(weights, state);
    }
};
}  // namespace

std::size_t layer_slots(std::size_t layer, std::size_t capacity) {
    return sliding(layer) ? std::min(capacity, Config::sliding_window) : capacity;
}

std::size_t layer_dense_bytes(std::size_t experts) {
    return 2 * bf16(Config::hidden) + bf16(qkv_rows * Config::hidden) + bf16(qkv_rows)
           + bf16(Config::hidden * attention_width) + bf16(Config::hidden)
           + bf16(Config::heads) + bf16(experts * Config::hidden) + bf16(experts)
           + bf16(experts * gate_up_rows)
           + bf16(experts * Config::hidden)
           // Four packed-part pointers per expert, resident or host-mapped.
           + aligned(experts * 4 * sizeof(void*));
}

std::size_t layer_expert_bytes(std::size_t resident) {
    return aligned(resident * expert_bytes);
}

std::size_t layer_state_bytes(std::size_t layer, std::size_t capacity) {
    return 2 * bf16(layer_slots(layer, capacity) * kv_width);
}

std::size_t head_bytes() {
    return bf16(Config::hidden) + bf16(Config::vocab * Config::hidden);
}

std::size_t scratch_bytes(std::size_t capacity, std::size_t experts) {
    return 2 * floats(Config::hidden) + floats(qkv_rows) + floats(attention_width)
           + floats(experts) + floats(2 * Config::experts_per_token)
           + floats(Config::experts_per_token * Config::intermediate)
           + floats(Config::heads * chunks(capacity) * (Config::head_dim + 2))
           + floats(Config::head_dim / 2) + floats(1);
}

std::size_t head_state_bytes() {
    return floats(Config::vocab) + floats(1);
}

std::size_t embedding_bytes() {
    return Config::vocab * Config::hidden * 2;
}

Rope rope() {
    constexpr double base = 150000, factor = 32, original = 4096, fast = 32, slow = 1;
    constexpr double dim = Config::head_dim;
    const auto correction = [&](double rotations) {
        return dim * std::log(original / (rotations * 2 * std::numbers::pi))
               / (2 * std::log(base));
    };
    const double low = std::max(correction(fast), 0.0),
                 high = std::min(correction(slow), dim - 1);
    Rope result{};
    for (std::size_t i = 0; i < result.inverse_frequency.size(); ++i) {
        // Transformers evaluates these in FP32 before mixing.
        const auto position = float(std::pow(base, double(2 * i) / dim));
        const float extrapolation = 1.F / position,
                    interpolation = 1.F / (float(factor) * position);
        const float ramp =
            std::clamp(float((double(i) - low) / (high - low)), 0.F, 1.F);
        const float extrapolated = 1.F - ramp;
        result.inverse_frequency[i] =
            interpolation * (1.F - extrapolated) + extrapolation * extrapolated;
    }
    result.scaling = float(0.1 * std::log(factor) + 1.0);
    return result;
}

Placement place(
    const Config& c,
    std::size_t capacity,
    const std::map<int, std::size_t>& budgets,
    std::size_t headroom
) {
    validate(c, capacity);
    std::vector<Usage> order;
    for (auto [device, budget] : budgets) {
        require(device >= 0 && device < 64, "gpt_oss_device");
        if (budget > headroom)
            order.push_back({device, budget - headroom});
    }
    if (order.empty())
        throw std::runtime_error("GPT-OSS requires a CUDA device with free memory");
    std::stable_sort(order.begin(), order.end(), [](const auto& a, const auto& b) {
        return a.room > b.room;
    });
    const auto dense = layer_dense_bytes(c.experts),
               scratch = scratch_bytes(capacity, c.experts);
    const auto complete = checked_add(dense, layer_expert_bytes(c.experts));
    const auto fresh = [&](std::size_t count) {
        auto devices = order;
        devices.resize(count);
        for (auto& d : devices)
            d.state = scratch;
        devices[0].weights = head_bytes();
        devices[0].state = checked_add(scratch, head_state_bytes());
        return devices;
    };
    const auto finish = [&](const std::vector<Usage>& devices,
                            const std::vector<std::size_t>& assigned,
                            const std::vector<std::size_t>& resident) {
        Placement p;
        p.head = devices[0].device;
        p.host_bytes = embedding_bytes();
        for (std::size_t i = 0; i < c.layers; ++i) {
            const auto& d = devices[assigned[i]];
            p.devices.push_back(d.device);
            p.resident.push_back(resident[i]);
            p.host_bytes =
                checked_add(p.host_bytes, (c.experts - resident[i]) * expert_bytes);
        }
        for (const auto& d : devices)
            if (d.layers || d.device == p.head)
                p.allocations[d.device] = {d.weights, d.state};
        return p;
    };

    // Complete residency on the fewest devices, assigning from the last layer.
    for (std::size_t count = 1; count <= order.size(); ++count) {
        auto devices = fresh(count);
        std::vector<std::size_t> assigned(c.layers);
        std::size_t k = 0;
        bool fits = true;
        for (std::size_t i = c.layers; i-- > 0;) {
            const auto state = layer_state_bytes(i, capacity);
            while (k < count
                   && checked_add(devices[k].used(), checked_add(complete, state))
                          > devices[k].room)
                ++k;
            if (k == count) {
                fits = false;
                break;
            }
            devices[k].weights = checked_add(devices[k].weights, complete);
            devices[k].state = checked_add(devices[k].state, state);
            ++devices[k].layers;
            assigned[i] = k;
        }
        if (fits)
            return finish(devices, assigned, std::vector(c.layers, c.experts));
    }

    // Partial residency: share layers by free room, then fill experts evenly.
    auto devices = fresh(order.size());
    const auto shortage = [] {
        throw std::runtime_error(
            "GPT-OSS requires more GPU memory for its attention layers and context "
            "state; reduce the context limit"
        );
    };
    if (devices[0].used() > devices[0].room)
        shortage();
    std::size_t total = 0;
    for (const auto& d : devices)
        if (d.used() < d.room)
            total = checked_add(total, d.room - d.used());
    std::vector<std::size_t> assigned(c.layers);
    std::size_t next = c.layers, k = 0;
    for (; k < devices.size() && next; ++k) {
        auto& d = devices[k];
        const auto free = d.used() < d.room ? d.room - d.used() : 0;
        auto target =
            k + 1 == devices.size()
                ? next
                : std::min(
                      next,
                      std::size_t(
                          std::llround(double(c.layers) * double(free) / double(total))
                      )
                  );
        for (; target; --target, --next) {
            const auto core = checked_add(dense, layer_state_bytes(next - 1, capacity));
            if (checked_add(d.used(), core) > d.room)
                break;
            d.weights = checked_add(d.weights, dense);
            d.state = checked_add(d.state, layer_state_bytes(next - 1, capacity));
            ++d.layers;
            assigned[next - 1] = k;
        }
    }
    if (next)
        shortage();
    std::vector<std::size_t> resident(c.layers);
    for (auto& d : devices) {
        if (!d.layers)
            continue;
        const auto free = d.room - d.used();
        auto per_layer = std::min(c.experts, free / (d.layers * expert_bytes));
        while (per_layer && d.layers * layer_expert_bytes(per_layer) > free)
            --per_layer;
        d.weights = checked_add(d.weights, d.layers * layer_expert_bytes(per_layer));
        for (std::size_t i = 0; i < c.layers; ++i)
            if (devices[assigned[i]].device == d.device)
                resident[i] = per_layer;
    }
    return finish(devices, assigned, resident);
}
}  // namespace kadan::gpt_oss
