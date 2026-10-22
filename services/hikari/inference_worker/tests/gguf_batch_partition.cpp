// Fake device downloads contain tags, never projection arithmetic or inference.
#include "kadan/gguf_batch_partition.hpp"

#include <array>
#include <iostream>
#include <source_location>

using namespace kadan;

namespace {
void check(bool ok, std::source_location location = std::source_location::current()) {
    if (!ok)
        throw std::runtime_error(
            "gguf_batch_partition_test:" + std::to_string(location.line())
        );
}

void eligibility_and_mixed_dispatch() {
    WeightIdentity identity;
    unsigned reads = 0, callbacks = 0;
    DenseCompute::GgufWeightSource source = [&](std::size_t, std::span<std::byte>) {
        ++reads;
    };
    std::vector<float> input(8 * 256, 1), first(8 * 19, -1), second(8, -1);
    std::array<DenseCompute::GgufBatchJob, 2> jobs{
        {{&identity, gguf::Encoding::q4_k, &source, input, 256, 19, first, true},
         {&identity, std::nullopt, &source, input, 256, 1, second, false}}
    };
    const std::array<Bytes, 2> capacities{1, 3};
    auto run = [&] {
        if (!gguf_batch_supported(jobs, capacities.size()))
            return false;
        for (std::size_t device = 0; device < capacities.size(); ++device)
            for (std::size_t j = 0; j < jobs.size(); ++j) {
                const auto& job = jobs[j];
                auto p = gguf_batch_partition(
                    job.in, job.out, job.encoding, capacities, device, 64 * 1024 * 1024
                );
                if (!p.count)
                    continue;
                ++callbacks;
                std::vector<std::byte> bytes;
                (*job.source)(p.first, bytes);
                std::vector<float> local(8 * p.count, float(j + device * 10));
                p.scatter(std::as_bytes(std::span(local)), job.output, job.out);
            }
        return true;
    };
    check(run());
    check(callbacks == 3 && reads == 3);
    for (std::size_t r = 0; r < 8; ++r) {
        for (std::size_t c = 0; c < 19; ++c)
            check(first[r * 19 + c] == (c < 4 ? 0 : 10));
        check(second[r] == 11);
    }
    auto rejected = [&](auto change) {
        auto saved = jobs;
        std::fill(first.begin(), first.end(), -1);
        std::fill(second.begin(), second.end(), -1);
        reads = callbacks = 0;
        change(jobs[1]);
        check(!run());
        check(!reads && !callbacks);
        check(std::all_of(first.begin(), first.end(), [](float v) { return v == -1; }));
        check(std::all_of(second.begin(), second.end(), [](float v) {
            return v == -1;
        }));
        jobs = saved;
    };
    // Prefill experts beyond one matvec pass are accepted up to the job bound.
    std::vector<float> prefill(9 * 256), prefill_output(9);
    {
        auto saved = jobs;
        jobs[1].input = prefill;
        jobs[1].output = prefill_output;
        check(gguf_batch_supported(jobs, capacities.size()));
        jobs = saved;
    }
    constexpr auto too_many = DenseCompute::gguf_batch_max_rows + 1;
    std::vector<float> large(too_many * 256), large_output(too_many);
    rejected([&](auto& job) {
        job.input = large;
        job.output = large_output;
    });
    rejected([](auto& job) { job.identity = nullptr; });
    rejected([](auto& job) { job.source = nullptr; });
    rejected([](auto& job) { job.in = 0; });
    rejected([](auto& job) { job.out = 0; });
    rejected([](auto& job) { job.input = {}; });
    rejected([](auto& job) { job.output = {}; });
    rejected([](auto& job) { job.in = 255; });
    std::vector<float> unaligned_input(8 * 128);
    rejected([&](auto& job) {
        job.encoding = gguf::Encoding::q4_k;
        job.in = 128;
        job.input = unaligned_input;
    });
    check(!gguf_batch_supported({}, 2));
    check(!gguf_batch_supported(jobs, 0));
}

void dispatch(std::span<const Bytes> capacities) {
    constexpr Bytes scratch = 64 * 1024 * 1024;
    const std::array<std::optional<gguf::Encoding>, 11> encodings{
        std::nullopt,
        gguf::Encoding::q3_k,
        gguf::Encoding::q4_k,
        gguf::Encoding::q8_0,
        gguf::Encoding::q6_k,
        gguf::Encoding::iq3_xxs,
        gguf::Encoding::iq4_nl,
        gguf::Encoding::iq4_xs,
        gguf::Encoding::iq2_s,
        gguf::Encoding::iq3_s,
        gguf::Encoding::bf16
    };
    // Mixed jobs, uneven partitions and a job narrower than the device count.
    for (auto rows : {1u, 8u}) {
        for (auto out : {1u, 19u, 10003u}) {
            for (auto encoding : encodings) {
                constexpr std::size_t in = 256;
                std::vector<float> output(rows * out, -1);
                std::vector<unsigned> owners(out);
                std::size_t previous = 0;
                for (std::size_t device = 0; device < capacities.size(); ++device) {
                    const auto p = gguf_batch_partition(
                        in, out, encoding, capacities, device, scratch
                    );
                    check(p.first == previous);
                    previous += p.count;
                    check(p.first == weight_boundary(out, capacities, device));
                    if (!p.count) {
                        check(!p.chunks());
                        p.scatter({}, output, out);
                        continue;
                    }
                    // Same keys/chunk shape as an individual projection's local plan.
                    const auto plan = encoding
                                          ? gguf_projection_plan(
                                                in, p.count, *encoding, false, scratch
                                            )
                                          : projection_plan(
                                                in,
                                                std::min(
                                                    p.count,
                                                    std::size_t(16 * 1024 * 1024)
                                                        / (in * sizeof(float))
                                                ),
                                                sizeof(float),
                                                false,
                                                1,
                                                scratch
                                            );
                    check(p.chunk_columns == plan.columns);
                    std::vector<float> local(rows * p.count, -1);
                    std::size_t chunks = 0;
                    for (std::size_t offset = 0; offset < p.count;
                         offset += p.chunk_columns) {
                        ++chunks;
                        const auto count = std::min(p.chunk_columns, p.count - offset);
                        // Fake source boundary observes absolute column and packed key
                        // units.
                        const auto first = p.first + offset;
                        const auto values =
                            encoding ? gguf::block_values(*encoding) : 1;
                        const auto key_first = first * (in / values);
                        check(key_first / (in / values) == first);
                        for (std::size_t c = 0; c < count; ++c) {
                            ++owners[first + c];
                            for (std::size_t r = 0; r < rows; ++r)
                                local[r * p.count + offset + c] =
                                    float(r * out + first + c);
                        }
                    }
                    check(chunks == p.chunks());
                    p.scatter(std::as_bytes(std::span(local)), output, out);
                }
                check(previous == out);
                for (auto owner : owners)
                    check(owner == 1);
                for (std::size_t i = 0; i < output.size(); ++i)
                    check(output[i] == float(i));
            }
        }
    }
}
}  // namespace

int main() {
    eligibility_and_mixed_dispatch();
    dispatch(std::array<Bytes, 1>{7});
    dispatch(std::array<Bytes, 2>{1, 1});
    dispatch(std::array<Bytes, 2>{1, 3});
    dispatch(std::array<Bytes, 3>{0, 1, 0});
    dispatch(std::array<Bytes, 3>{0, 0, 0});
    std::cout << "gguf batch partition tests passed\n";
}
