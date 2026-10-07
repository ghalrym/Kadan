#include "kadan/cuda_plan.hpp"
#include "../src/nvfp4_numeric.hpp"
#include <bit>
#include <cmath>
#include <limits>
#include <array>
#include <climits>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace {
void check(bool ok) { if (!ok) throw std::runtime_error("check_failed"); }
template<class Error = std::invalid_argument, class F>
void fails(F fn, std::string_view expected) {
    bool caught = false;
    try { fn(); } catch (const Error& error) { check(expected == error.what()); caught = true; }
    check(caught);
}
}
int main() {
    using namespace kadan;
    try {
        // Regression: exact decoded weight exceeds FLT_MAX but rounds to it.
        const float edge_global = std::bit_cast<float>(0x7f783e0fU);
        const float local = 6.0F * (11.0F / 64.0F);
        const double exact = static_cast<double>(local) * edge_global;
        check(exact > std::numeric_limits<float>::max());
        check(static_cast<float>(exact) == std::numeric_limits<float>::max());
        check(cuda::detail::weight_overflows(local, edge_global));
        check(cuda::detail::weight_overflows(-local, edge_global));
        check(!cuda::detail::weight_overflows(local, std::nextafter(edge_global, 0.0F)));
        check(!cuda::detail::weight_overflows(1, std::numeric_limits<float>::max()));
        check(!cuda::detail::weight_overflows(-1, std::numeric_limits<float>::max()));
        check(!cuda::detail::weight_overflows(0, edge_global));
        std::array<std::uint8_t, 8> edge_weights{}; edge_weights.fill(0x77);
        const std::array<std::uint8_t, 1> edge_blocks{0x23};
        const std::array<float, 1> edge_multiplier{edge_global};
        const std::array<float, 16> zeros{};
        quantization::Matrix edge{quantization::Encoding::modelopt_nvfp4, 1, 16,
                                  edge_weights, edge_blocks, edge_multiplier};
        fails<std::overflow_error>([&] { quantization::matvec(edge, zeros, 4); }, "nonfinite_result");
        edge_weights.fill(0xff);
        fails<std::overflow_error>([&] { quantization::matvec(edge, zeros, 4); }, "nonfinite_result");
        std::vector<std::uint8_t> weights(48, 0x22), blocks(6, 0x38);
        std::array<float, 1> global{1};
        quantization::Matrix m{quantization::Encoding::modelopt_nvfp4, 3, 32, weights, blocks, global};
        const auto plan = cuda::plan_nvfp4(m, 1280);
        check(plan.rows == 3 && plan.columns == 32 && plan.weight_bytes == 48 && plan.block_bytes == 6);
        check(plan.block_offset == 256 && plan.input_offset == 512 && plan.output_offset == 768);
        check(plan.status_offset == 1024 && plan.device_bytes == 1280);
        fails<std::length_error>([&] { cuda::plan_nvfp4(m, 1279); }, "cuda_device_budget");
        auto bad = m; bad.encoding = quantization::Encoding::modelopt_fp8;
        fails([&] { cuda::plan_nvfp4(bad, 1280); }, "cuda_requires_modelopt_nvfp4");
        bad = m; bad.rows = std::size_t{INT_MAX} + 1;
        fails([&] { cuda::plan_nvfp4(bad, 1280); }, "cuda_shape_limit");
        bad = m; bad.columns = 0;
        fails([&] { cuda::plan_nvfp4(bad, 1280); }, "cuda_shape_limit");
        bad = m; bad.columns = 31;
        fails([&] { cuda::plan_nvfp4(bad, 1280); }, "nvfp4_block_alignment");
        blocks[0] = 0x7f;
        fails([&] { cuda::plan_nvfp4(m, 1280); }, "invalid_block_scale");
        blocks[0] = 0x38; global[0] = 0;
        fails([&] { cuda::plan_nvfp4(m, 1280); }, "invalid_multiplier");
        global[0] = 1;
        weights.resize(33 * 1024); blocks.resize(33 * 128, 0x38);
        m = {quantization::Encoding::modelopt_nvfp4, 33, 2048, weights, blocks, global};
        const auto full = cuda::plan_nvfp4(m, 65536);
        check(full.device_bytes <= 65536 && full.input_offset >= weights.size() + blocks.size());
        check(full.output_offset >= full.input_offset + 2048 * sizeof(float));
        check(full.status_offset >= full.output_offset + 33 * sizeof(float));
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
