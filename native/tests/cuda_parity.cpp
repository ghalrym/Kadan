// Intentionally not registered in CTest. Do not run without explicit permission.
#include "kadan/cuda_projection.hpp"
#include <cuda_runtime_api.h>

#include <array>
#include <bit>
#include <charconv>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace {
void check(bool ok, const char* error) { if (!ok) throw std::runtime_error(error); }
void fixture(std::size_t rows, std::size_t columns, int device,
             const std::shared_ptr<kadan::Resources>& resources) {
    using namespace kadan;
    std::vector<std::uint8_t> weights(rows * columns / 2), blocks(rows * columns / 16);
    constexpr std::array<std::uint8_t, 8> scales{0, 1, 0x30, 0x38, 0x3c, 0x40, 0x44, 0x7e};
    for (std::size_t i = 0; i < weights.size(); ++i) weights[i] = (i * 37 + 11) % 256;
    for (std::size_t i = 0; i < blocks.size(); ++i) blocks[i] = scales[i % scales.size()];
    std::array<float, 1> global{0.1F};
    quantization::Matrix matrix{quantization::Encoding::modelopt_nvfp4, rows, columns, weights, blocks, global};
    std::vector<float> input(columns), output(rows);
    for (std::size_t i = 0; i < columns; ++i) input[i] = (static_cast<int>(i % 17) - 8) / 16.0F;
    const auto reference = quantization::matvec(matrix, input, 1024 * 1024);
    const auto dense = quantization::decode_rows(matrix, 0, rows, 1024 * 1024);
    cuda::Nvfp4Projection projection(matrix, device, resources);
    check(resources->snapshot().used[device + 1] == projection.plan().device_bytes, "reservation_missing");
    projection.matvec(input, output); // Exactly one fused kernel per fixture.
    for (std::size_t row = 0; row < rows; ++row) {
        double magnitude = 0;
        for (std::size_t column = 0; column < columns; ++column)
            magnitude += std::abs(static_cast<double>(dense[row * columns + column]) * input[column]);
        // Small shapes have <=18 per-lane FMA terms plus two warp reductions.
        // A sum-of-magnitudes bound handles cancellation near a zero result.
        const double tolerance = 1e-5 + 64 * std::numeric_limits<float>::epsilon() * magnitude;
        check(std::isfinite(output[row]) && std::abs(static_cast<double>(output[row]) - reference[row]) <= tolerance,
              "gpu_cpu_parity_failed");
    }
    projection.close(); projection.close();
    check(resources->snapshot().residents == 0 && resources->snapshot().used[device + 1] == 0, "reservation_not_released");
    std::cout << "passed rows=" << rows << " columns=" << columns << '\n';
}
void overflow_fixture(int device, const std::shared_ptr<kadan::Resources>& resources) {
    using namespace kadan;
    std::array<std::uint8_t, 16> weights{};
    for (std::size_t i = 0; i < weights.size(); ++i) weights[i] = i < 8 ? 0x77 : 0xff;
    const std::array<std::uint8_t, 2> blocks{0x23, 0x23};
    const std::array<float, 1> global{std::bit_cast<float>(0x7f783e0fU)};
    quantization::Matrix matrix{quantization::Encoding::modelopt_nvfp4, 2, 16, weights, blocks, global};
    const std::array<float, 16> input{};
    std::array<float, 2> output{123, 456};
    bool cpu_rejected = false;
    try { quantization::matvec(matrix, input, 8); }
    catch (const std::overflow_error&) { cpu_rejected = true; }
    check(cpu_rejected, "cpu_overflow_not_rejected");
    cuda::Nvfp4Projection projection(matrix, device, resources);
    bool gpu_rejected = false;
    try { projection.matvec(input, output); } // Exactly one launch, both signs.
    catch (const std::overflow_error& error) {
        check(std::string_view(error.what()) == "nonfinite_cuda_projection", "unexpected_gpu_error");
        gpu_rejected = true;
    }
    check(gpu_rejected && output[0] == 123 && output[1] == 456, "gpu_overflow_not_rejected");
    projection.close();
    check(resources->snapshot().residents == 0 && resources->snapshot().used[device + 1] == 0,
          "overflow_reservation_not_released");
    std::cout << "passed pre-round overflow rows=2 columns=16\n";
}
void subnormal_fixture(int device, const std::shared_ptr<kadan::Resources>& resources) {
    using namespace kadan;
    const std::array<std::uint8_t, 8> weights{0x22,0x22,0x22,0x22,0x22,0x22,0x22,0x22};
    const std::array<std::uint8_t, 1> blocks{0x38};
    const std::array<float, 1> global{std::numeric_limits<float>::denorm_min()};
    quantization::Matrix matrix{quantization::Encoding::modelopt_nvfp4, 1, 16, weights, blocks, global};
    std::array<float, 16> input{}; input.fill(1);
    std::array<float, 1> output{};
    const auto reference = quantization::matvec(matrix, input, 4);
    cuda::Nvfp4Projection projection(matrix, device, resources);
    projection.matvec(input, output);
    // Exact representable result detects accidental flush-to-zero compilation.
    check(output[0] == reference[0] && output[0] == 16 * global[0], "gpu_subnormal_parity_failed");
    projection.close();
    check(resources->snapshot().residents == 0, "subnormal_reservation_not_released");
    std::cout << "passed subnormal rows=1 columns=16\n";
}
}
int main(int argc, char** argv) {
    // All executable validation is opt-in. Even this binary is not run during CI.
    if (argc != 4 || std::string_view(argv[1]) != "--allow-gpu-validation" || std::string_view(argv[2]) != "--device") {
        std::cerr << "Requires explicit approval before: kadan-cuda-parity --allow-gpu-validation --device ORDINAL\n";
        return 2;
    }
    try {
        int device = -1; const std::string_view text(argv[3]);
        const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), device);
        check(error == std::errc{} && end == text.data() + text.size() && device >= 0 && device < 64, "invalid_device");
        check(cudaSetDevice(device) == cudaSuccess, "cudaSetDevice_failed");
        kadan::Footprint capacity(device + 2, 0); capacity[device + 1] = 65536;
        auto resources = std::make_shared<kadan::Resources>(capacity);
        for (const auto shape : {std::array<std::size_t, 2>{3,32}, {5,256}, {33,2048}, {2,2064}})
            fixture(shape[0], shape[1], device, resources);
        subnormal_fixture(device, resources);
        overflow_fixture(device, resources);
        std::cout << "Six synthetic parity cases passed; no throughput measurement.\n";
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
