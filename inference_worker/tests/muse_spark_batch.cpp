#include "kadan/muse_spark_batch.hpp"

#include <bit>
#include <cstdint>
#include <deque>
#include <iostream>
#include <vector>

namespace {
namespace batch = kadan::muse_spark::batch;

void check(bool value) {
    if (!value)
        throw std::runtime_error("check_failed");
}

template <class Error, class F> void rejects(F action) {
    bool caught = false;
    try {
        action();
    } catch (const Error&) {
        caught = true;
    }
    check(caught);
}

void bounded_rows() {
    check(batch::plan(0, 1, 0) == 0);
    check(batch::plan(1000, 8 * 8192 * sizeof(float), 8u << 20) == 256);
    check(batch::plan(17, 1, 0) == 17);
    check(batch::plan(50, 1024, batch::scratch_limit - 3 * 1024) == 3);
    check(batch::plan(1, batch::scratch_limit, 0) == 1);
    rejects<std::invalid_argument>([] { batch::plan(1, 0, 0); });
    rejects<std::length_error>([] { batch::plan(1, 1, batch::scratch_limit); });
    rejects<std::length_error>([] { batch::plan(1, 1, SIZE_MAX); });
    rejects<std::length_error>([] { batch::plan(1, SIZE_MAX, 0); });
    std::array<float, 12> values{};
    auto middle = batch::row(std::span(values), 4, 1);
    check(middle.data() == values.data() + 4 && middle.size() == 4);
    middle[0] = 7;
    check(batch::row(std::span<const float>(values), 4, 1)[0] == 7);
    rejects<std::invalid_argument>([&] { batch::row(std::span(values), 0, 0); });
    rejects<std::invalid_argument>([&] { batch::row(std::span(values), 5, 0); });
    rejects<std::invalid_argument>([&] { batch::row(std::span(values), 4, 3); });
    check(batch::position(SIZE_MAX - 1, 1) == SIZE_MAX);
    rejects<std::overflow_error>([] { batch::position(SIZE_MAX, 1); });
    rejects<std::overflow_error>([] { batch::ring(SIZE_MAX, 1); });
    rejects<std::invalid_argument>([] { batch::ring(0, 0); });
}

// An attention component, not a model forward: scalar keys and values make the
// expected causal sum independently calculable from an append-only serial deque.
void causal_ring() {
    for (const std::size_t capacity : {1u, 3u, 17u, 2048u}) {
        for (const std::size_t chunk : {1u, 7u, 256u}) {
            std::vector<float> keys(capacity), values(capacity);
            std::deque<std::pair<float, float>> serial;
            constexpr std::size_t total = 4103;
            for (std::size_t base = 0; base < total; base += chunk) {
                const auto rows = std::min(chunk, total - base);
                for (std::size_t token = 0; token < rows; ++token) {
                    const auto absolute = batch::position(base, token);
                    const float key = float(int(absolute % 13) - 6) / 8;
                    const float value = float(int(absolute % 29) - 14) / 4;
                    serial.emplace_back(key, value);
                    if (serial.size() > capacity)
                        serial.pop_front();
                    const auto ring = batch::ring(absolute, capacity);
                    keys[ring.slot] = key;
                    values[ring.slot] = value;
                    check(ring.count == serial.size());
                    check(ring.first + ring.count == absolute + 1);
                    const float query = float(int(absolute % 7) - 3) / 4;
                    float expected_sum = 0, expected_weight = 0;
                    for (const auto& [k, v] : serial) {
                        const float weight = std::exp(query * k);
                        expected_sum += weight * v;
                        expected_weight += weight;
                    }
                    float actual_sum = 0, actual_weight = 0;
                    for (std::size_t i = 0; i < ring.count; ++i) {
                        const auto slot = (ring.first + i) % capacity;
                        check(keys[slot] == serial[i].first);
                        check(values[slot] == serial[i].second);
                        const float weight = std::exp(query * keys[slot]);
                        actual_sum += weight * values[slot];
                        actual_weight += weight;
                    }
                    check(actual_sum / actual_weight == expected_sum / expected_weight);
                }
            }
        }
    }
}

batch::Route serial_route(
    const std::array<float, 256>& probabilities, const std::array<float, 256>& bias
) {
    // Repeated maximum selection is independent of production's partial_sort.
    std::array<bool, 256> selected{};
    float sum = 1e-15f;
    for (std::size_t k = 0; k < 8; ++k) {
        std::size_t best = 256;
        for (std::size_t expert = 0; expert < 256; ++expert)
            if (!selected[expert]
                && (best == 256
                    || probabilities[expert] + bias[expert]
                           > probabilities[best] + bias[best]))
                best = expert;
        selected[best] = true;
        sum += probabilities[best];
    }
    batch::Route output{};
    std::size_t at = 0;
    for (std::size_t expert = 0; expert < 256; ++expert)
        if (selected[expert]) {
            output.experts[at] = expert;
            output.weights[at++] = probabilities[expert] / sum;
        }
    check(at == 8);
    return output;
}

void routing_and_scatter() {
    std::vector<batch::Route> routes;
    std::vector<float> serial_output;
    for (std::size_t token = 0; token < 257; ++token) {
        std::array<float, 256> probabilities{}, bias{};
        for (std::size_t expert = 0; expert < 256; ++expert) {
            // Includes all-equal ties, biased selection, and varied magnitudes
            // which distinguish ranked-denominator from expert-order summation.
            probabilities[expert] =
                token == 0 ? .5f
                           : std::ldexp(
                                 float((expert * 7919 + token * 31) % 997 + 1),
                                 -10 - int((expert + token) % 19)
                             );
            bias[expert] =
                token < 2 ? 0.f : float(int((expert * 17 + token) % 23) - 11) / 32;
        }
        const auto expected = serial_route(probabilities, bias);
        const auto actual = batch::route(probabilities, bias);
        check(actual.experts == expected.experts);
        check(actual.weights == expected.weights);
        if (!token)
            for (std::size_t i = 0; i < 8; ++i)
                check(actual.experts[i] == i && actual.weights[i] == .125f);
        routes.push_back(actual);
        float output = 0;
        for (std::size_t i = 0; i < 8; ++i) {
            const auto expert = expected.experts[i];
            const float contribution = float(int((expert * 7 + token) % 37) - 18) / 7;
            output += contribution * expected.weights[i];
        }
        serial_output.push_back(output);
    }
    std::vector<float> gathered_output(routes.size());
    for (std::size_t expert = 0; expert < 256; ++expert)
        for (std::size_t token = 0; token < routes.size(); ++token) {
            const auto selected = batch::route_index(routes[token], expert);
            if (selected == 8)
                continue;
            const float contribution = float(int((expert * 7 + token) % 37) - 18) / 7;
            gathered_output[token] += contribution * routes[token].weights[selected];
        }
    check(gathered_output == serial_output);
    // The selected experts must accumulate by ascending id even when selection
    // rank differs. Reassociation would lose the small final contribution.
    const auto tied = routes.front();
    float ordered = 0;
    for (std::size_t expert = 0; expert < 256; ++expert) {
        const auto at = batch::route_index(tied, expert);
        if (at == 8)
            continue;
        const float contribution = expert == 0   ? 1e20f
                                   : expert == 1 ? -1e20f
                                   : expert == 2 ? 3.f
                                                 : 0.f;
        ordered += contribution * tied.weights[at];
    }
    check(ordered == .375f);
    std::array<float, 256> scores{}, bias{};
    rejects<std::invalid_argument>([&] {
        batch::route(std::span(scores).first(255), bias);
    });
    for (const float invalid : {INFINITY, -INFINITY, NAN}) {
        scores[91] = invalid;
        rejects<std::invalid_argument>([&] { batch::route(scores, bias); });
        scores[91] = 0;
        bias[91] = invalid;
        rejects<std::invalid_argument>([&] { batch::route(scores, bias); });
        bias[91] = 0;
    }
}
}  // namespace

int main() {
    bounded_rows();
    causal_ring();
    routing_and_scatter();
    std::cout << "Muse batch component checks passed\n";
}
