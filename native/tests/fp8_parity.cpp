// Compile only until explicitly authorized. Deliberately absent from CTest.
#include "kadan/cuda_projection.hpp"
#include <cuda_runtime_api.h>
#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace {
void check(bool ok, const char* error) { if (!ok) throw std::runtime_error(error); }
void released(const std::shared_ptr<kadan::Resources>& resources, int device) {
    check(resources->snapshot().residents == 0 && resources->snapshot().used[device + 1] == 0,
          "reservation_not_released");
}
void fixture(std::size_t rows, std::size_t columns, bool row_scales, bool repeat,
             int device, const std::shared_ptr<kadan::Resources>& resources) {
    using namespace kadan;
    std::vector<std::uint8_t> weights(rows * columns);
    for (std::size_t i = 0; i < weights.size(); ++i) {
        unsigned code = (i * 37 + 11) % 256;
        weights[i] = (code & 127) == 127 ? 0 : code;
    }
    std::vector<float> scales(row_scales ? rows : 1);
    for (std::size_t i = 0; i < scales.size(); ++i) scales[i] = 0.1F * (i % 3 + 1);
    quantization::Matrix m{quantization::Encoding::modelopt_fp8, rows, columns, weights, {}, scales};
    std::vector<float> input(columns), output(rows);
    for (std::size_t i = 0; i < columns; ++i) input[i] = (static_cast<int>(i % 17) - 8) / 16.0F;
    const auto reference = quantization::matvec(m, input, 1024 * 1024);
    const auto dense = quantization::decode_rows(m, 0, rows, 1024 * 1024);
    cuda::Fp8Projection projection(m, device, resources);
    const auto used = resources->snapshot().used[device + 1];
    check(used == projection.plan().device_bytes && used <= 65536, "bad_reservation");
    // Constructor owns a copy; later calls must never rescan these host spans.
    std::fill(weights.begin(), weights.end(), 0x7f);
    std::fill(scales.begin(), scales.end(), std::numeric_limits<float>::quiet_NaN());
    for (int token = 0; token < (repeat ? 2 : 1); ++token) {
        if (token) for (auto& value : input) value = -value;
        projection.matvec(input, output); // One fused launch per token; four ordinary launches total.
        for (std::size_t row = 0; row < rows; ++row) {
            double magnitude = 0;
            for (std::size_t col = 0; col < columns; ++col)
                magnitude += std::abs(static_cast<double>(dense[row * columns + col]) * input[col]);
            const double expected = token ? -reference[row] : reference[row];
            const double tolerance = 1e-5 + 64 * std::numeric_limits<float>::epsilon() * magnitude;
            check(std::isfinite(output[row]) && std::abs(static_cast<double>(output[row]) - expected) <= tolerance,
                  "fp8_cpu_parity_failed");
        }
        check(resources->snapshot().used[device + 1] == used, "persistent_storage_changed");
    }
    projection.close(); projection.close(); released(resources, device);
    std::cout << "passed FP8 rows=" << rows << " columns=" << columns << " launches=" << (repeat ? 2 : 1) << '\n';
}
void subnormal(int device, const std::shared_ptr<kadan::Resources>& resources) {
    using namespace kadan;
    std::array<std::uint8_t, 16> weights{}; weights.fill(0x38);
    const std::array<float, 1> scales{std::numeric_limits<float>::denorm_min()};
    quantization::Matrix m{quantization::Encoding::modelopt_fp8, 1, 16, weights, {}, scales};
    std::array<float, 16> input{}; input.fill(1);
    std::array<float, 1> output{};
    const auto reference = quantization::matvec(m, input, 4);
    cuda::Fp8Projection projection(m, device, resources);
    projection.matvec(input, output);
    check(output[0] == reference[0] && output[0] == 16 * scales[0], "fp8_subnormal_failed");
    projection.close(); released(resources, device);
    std::cout << "passed FP8 exact subnormal\n";
}
void reduction_overflow(int device, const std::shared_ptr<kadan::Resources>& resources) {
    using namespace kadan;
    const std::array<std::uint8_t, 2> weights{0x38, 0x38};
    const std::array<float, 1> scales{std::numeric_limits<float>::max()};
    quantization::Matrix m{quantization::Encoding::modelopt_fp8, 1, 2, weights, {}, scales};
    const std::array<float, 2> input{2, 2};
    std::array<float, 1> output{123};
    bool cpu_rejected = false;
    try { quantization::matvec(m, input, 4); }
    catch (const std::overflow_error&) { cpu_rejected = true; }
    check(cpu_rejected, "cpu_reduction_overflow_not_rejected");
    cuda::Fp8Projection projection(m, device, resources);
    bool gpu_rejected = false;
    try { projection.matvec(input, output); }
    catch (const std::overflow_error& error) {
        check(std::string_view(error.what()) == "nonfinite_cuda_projection", "unexpected_gpu_error");
        gpu_rejected = true;
    }
    check(gpu_rejected && output[0] == 123, "fp8_reduction_overflow_not_rejected");
    projection.close(); released(resources, device);
    std::cout << "passed FP8 reduction overflow\n";
}
}
int main(int argc, char** argv) {
    if (argc != 4 || std::string_view(argv[1]) != "--allow-gpu-validation" || std::string_view(argv[2]) != "--device") {
        std::cerr << "Explicit approval required: kadan-fp8-parity --allow-gpu-validation --device ORDINAL\n";
        return 2;
    }
    try {
        int device = -1; const std::string_view text(argv[3]);
        const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), device);
        check(error == std::errc{} && end == text.data() + text.size() && device >= 0 && device < 64, "invalid_device");
        check(cudaSetDevice(device) == cudaSuccess, "cudaSetDevice_failed");
        kadan::Footprint capacity(device + 2, 0); capacity[device + 1] = 65536;
        auto resources = std::make_shared<kadan::Resources>(capacity);
        fixture(1, 1, false, false, device, resources);
        fixture(3, 17, true, false, device, resources);
        fixture(17, 1025, true, true, device, resources);
        subnormal(device, resources);
        reduction_overflow(device, resources);
        std::cout << "Six FP8 synthetic launches passed; no throughput measurement.\n";
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
