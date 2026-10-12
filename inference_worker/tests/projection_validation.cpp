// Validation/dispatch only: fakes copy outputs; CPU fallback has zero input rows.
// No model forward pass, projection arithmetic or CUDA device execution.
#include <array>
#include <bit>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <source_location>
#include <unistd.h>

#include "kadan/float_weights.hpp"

using namespace kadan;

namespace {
void check(bool ok, std::source_location location = std::source_location::current()) {
    if (!ok)
        throw std::runtime_error(
            "projection_validation_test:" + std::to_string(location.line())
        );
}

template <class F> void fails(F fn, std::string_view expected) {
    try {
        fn();
    } catch (const std::exception& error) {
        check(error.what() == expected);
        return;
    }
    throw std::runtime_error("expected_failure");
}

struct Fixture {
    char root[40] = "/tmp/kadan-projection-result-XXXXXX";

    Fixture() {
        check(mkdtemp(root));
        const std::string header =
            R"({"f32":{"dtype":"F32","shape":[1,1],"data_offsets":[0,4]},"bf16":{"dtype":"BF16","shape":[1,1],"data_offsets":[4,6]}})";
        std::ofstream file(path(), std::ios::binary);
        for (unsigned i = 0; i < 8; ++i)
            file.put(char(std::uint64_t(header.size()) >> (8 * i)));
        file << header;
        for (unsigned i = 0; i < 6; ++i)
            file.put(0);
        check(bool(file));
    }

    std::filesystem::path path() const {
        return std::filesystem::path(root) / "weights.safetensors";
    }

    ~Fixture() {
        std::filesystem::remove_all(root);
    }
};

struct UnavailableCompute : DenseCompute {
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
        throw std::runtime_error("unexpected_dense");
    }
};

struct FakeCompute final : UnavailableCompute {
    ProjectionResult f32_result = ProjectionResult::unvalidated;
    ProjectionResult bf16_result = ProjectionResult::unvalidated;
    float value = 1;
    unsigned f32_calls = 0, bf16_calls = 0;
    ProjectionPolicy policy = ProjectionPolicy::standard;

    bool supports_bf16_weights() const override {
        return true;
    }

    ProjectionResult finish(
        std::span<float> output, ProjectionResult result, const std::atomic_bool& cancel
    ) {
        if (cancel.load())
            throw std::runtime_error("fake_cancelled");
        if (result == ProjectionResult::unsupported)
            return result;
        std::fill(output.begin(), output.end(), value);
        if (result == ProjectionResult::finite
            && !projection_output_finite(ProjectionResult::unvalidated, output))
            throw std::runtime_error("fake_nonfinite_output");
        return result;
    }

    ProjectionResult dense_source(
        const WeightIdentity&,
        const FloatWeightSource&,
        std::span<const float>,
        std::span<const float>,
        std::size_t,
        std::size_t,
        std::span<float> output,
        const std::atomic_bool& cancel,
        bool precise,
        ProjectionPolicy p
    ) override {
        check(!precise);
        policy = p;
        ++f32_calls;
        return finish(output, f32_result, cancel);
    }

    ProjectionResult dense_bf16_source(
        const WeightIdentity&,
        const Bf16WeightSource&,
        std::span<const float>,
        std::span<const float>,
        std::size_t,
        std::size_t,
        std::span<float> output,
        const std::atomic_bool& cancel
    ) override {
        ++bf16_calls;
        return finish(output, bf16_result, cancel);
    }
};
}  // namespace

int main() {
    Fixture fixture;
    checkpoint::Shard shard(
        fixture.root,
        "weights.safetensors",
        std::make_shared<checkpoint::MemoryBudget>(1024 * 1024)
    );
    Resources resources({1024});
    std::atomic_bool cancel = false;
    FakeCompute compute;
    UnavailableCompute unavailable;
    std::array<float, 1> input{1}, output{};
    auto project =
        [&](DenseCompute* backend, std::string_view name, bool empty = false) {
            return checkpoint::project_source(
                resources,
                Workload::image,
                backend,
                shard,
                name,
                {},
                empty ? std::span<const float>{} : input,
                1,
                1,
                empty ? std::span<float>{} : output,
                cancel,
                ProjectionPolicy::image_feed_forward
            );
        };
    // Zero rows exercise CPU dispatch and storage loading without a forward pass.
    const auto cpu = project(nullptr, "f32", true);
    check(cpu == ProjectionResult::unvalidated);
    check(project(&unavailable, "bf16", true) == cpu);
    check(
        unavailable.dense_bf16_source({}, {}, {}, {}, 1, 1, {}, cancel)
        == ProjectionResult::unsupported
    );
    for (auto name : {"f32", "bf16"}) {
        for (std::uint32_t bits :
             {0U, 0x80000000U, 1U, 0x80000001U, 0x3f800000U, 0x7f7fffffU}) {
            compute.value = std::bit_cast<float>(bits);
            check(project(&compute, name) == cpu);
            check(std::bit_cast<std::uint32_t>(output[0]) == bits);
            check(projection_output_finite(cpu, output));
        }
        for (std::uint32_t bits : {0x7fc00000U, 0x7f800000U, 0xff800000U}) {
            compute.value = std::bit_cast<float>(bits);
            check(project(&compute, name) == cpu);
            check(!projection_output_finite(cpu, output));
        }
    }
    check(compute.policy == ProjectionPolicy::image_feed_forward);
    compute.f32_result = compute.bf16_result = ProjectionResult::finite;
    compute.value = 1;
    for (auto name : {"f32", "bf16"}) {
        check(project(&compute, name) == ProjectionResult::finite);
        check(projection_output_finite(ProjectionResult::finite, output));
        compute.value = std::bit_cast<float>(0x7f800000U);
        fails([&] { project(&compute, name); }, "fake_nonfinite_output");
        compute.value = 1;
    }
    // A sentinel proves a certificate short-circuits the host scan. Issuing a
    // certificate for this sentinel would violate the compute implementation contract.
    output[0] = std::bit_cast<float>(0x7fc00000U);
    check(projection_output_finite(ProjectionResult::finite, output));
    check(!projection_output_finite(ProjectionResult::unvalidated, output));
    check(!projection_output_finite(ProjectionResult::unsupported, output));

    compute.bf16_result = ProjectionResult::unsupported;
    const auto f32_calls = compute.f32_calls, bf16_calls = compute.bf16_calls;
    check(project(&compute, "bf16") == ProjectionResult::finite);
    check(compute.f32_calls == f32_calls + 1 && compute.bf16_calls == bf16_calls + 1);
    compute.f32_result = ProjectionResult::unsupported;
    check(project(&compute, "bf16", true) == cpu);
    compute.f32_result = ProjectionResult::finite;
    cancel = true;
    fails([&] { project(&compute, "f32"); }, "fake_cancelled");
    cancel = false;
    std::cout << "per-result certification and fallback validation passed\n";
}
