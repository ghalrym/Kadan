// Layout-only units. No CUDA calls, matrix arithmetic or model execution.
#include "kadan/bf16_projection.hpp"

#include <array>
#include <iostream>

using namespace kadan;
using namespace kadan::video;

int main() {
    alignas(256) static std::array<std::byte, bf16_projection_workspace_bytes + 0x6100>
        storage;
    auto span = [](std::size_t offset,
                   DeviceDType dtype,
                   std::vector<std::size_t> dimensions,
                   std::vector<std::size_t> strides,
                   Bytes bytes) {
        return MediaDeviceSpan{
            static_cast<void*>(storage.data() + offset),
            0,
            dtype,
            dimensions,
            strides,
            bytes,
            1,
            0
        };
    };
    std::array spans{
        span(0x1000, DeviceDType::f32, {3, 8}, {8, 1}, 96),
        span(0x2000, DeviceDType::bf16, {4, 8}, {8, 1}, 64),
        span(0x3000, DeviceDType::f32, {4}, {1}, 16),
        span(0x4000, DeviceDType::bf16, {3, 8}, {8, 1}, 48),
        span(0x5000, DeviceDType::f32, {3, 4}, {4, 1}, 48),
        span(
            0x6000,
            DeviceDType::u8,
            {bf16_projection_workspace_bytes},
            {1},
            bf16_projection_workspace_bytes
        ),
        span(bf16_projection_workspace_bytes + 0x6000, DeviceDType::u32, {1}, {1}, 4)
    };
    auto validate = [&] {
        return validate_bf16_projection(
            0, spans[0], spans[1], spans[2], spans[3], spans[4], spans[5], spans[6]
        );
    };
    auto check = [](bool ok) {
        if (!ok)
            throw std::runtime_error("bf16_projection_test");
    };
    const auto shape = validate();
    check(shape.rows == 3 && shape.in == 8 && shape.out == 4);
    const auto original = spans;
    auto rejects = [&] {
        bool rejected = false;
        try {
            validate();
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        check(rejected);
        spans = original;
    };
    for (std::size_t i = 0; i < spans.size(); ++i) {
        spans[i].device = 1;
        rejects();
        spans[i].data = nullptr;
        rejects();
        spans[i].bytes -= 1;
        rejects();
        spans[i].strides.back() = 2;
        rejects();
        spans[i].dtype = DeviceDType::f64;
        rejects();
        spans[i].data = storage.data() + 0x1001;
        rejects();
        for (std::size_t j = 0; j < i; ++j) {
            spans[i].data = spans[j].data;
            rejects();
        }
    }
    // Partially overlapping, naturally aligned output must also be rejected.
    spans[4].data = storage.data() + 0x1004;
    rejects();
    spans[5].data = storage.data() + 0x6004;
    rejects();
    spans[1].dimensions[1] = 7;
    rejects();
    spans[0].dimensions[0] = 0;
    rejects();
    spans[0].dimensions[0] = std::size_t(INT_MAX) + 1;
    rejects();
    spans[2] = {nullptr, 0, DeviceDType::f32, {}, {}, 0, 1, 0};
    validate();
    spans[2].bytes = 4;
    rejects();

    std::array pair{
        original[0],
        original[1],
        span(0x2100, DeviceDType::bf16, {4, 8}, {8, 1}, 64),
        original[3],
        original[4],
        span(0x5100, DeviceDType::f32, {3, 4}, {4, 1}, 48),
        original[5],
        original[6],
        span(bf16_projection_workspace_bytes + 0x6004, DeviceDType::u32, {1}, {1}, 4),
        span(bf16_projection_workspace_bytes + 0x6008, DeviceDType::u32, {1}, {1}, 4)
    };
    auto validate_pair = [&] {
        return validate_image_silu_pair(
            0,
            pair[0],
            pair[1],
            pair[2],
            pair[3],
            pair[4],
            pair[5],
            pair[6],
            pair[7],
            pair[8],
            pair[9]
        );
    };
    const auto pair_shape = validate_pair();
    check(pair_shape.rows == 3 && pair_shape.in == 8 && pair_shape.out == 4);
    const auto pair_original = pair;
    auto rejects_pair = [&] {
        bool rejected = false;
        try {
            validate_pair();
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        check(rejected);
        pair = pair_original;
    };
    for (std::size_t i = 0; i < pair.size(); ++i) {
        pair[i].device = 1;
        rejects_pair();
        pair[i].data = nullptr;
        rejects_pair();
        pair[i].bytes -= 1;
        rejects_pair();
        pair[i].strides.back() = 2;
        rejects_pair();
        pair[i].dtype = DeviceDType::f64;
        rejects_pair();
        pair[i].data = storage.data() + 0x1001;
        rejects_pair();
        // Include statuses, weights and both outputs across producer boundaries.
        for (std::size_t j = 0; j < i; ++j) {
            pair[i].data = pair[j].data;
            rejects_pair();
        }
    }
    pair[5].data = storage.data() + 0x5004;
    rejects_pair();
    pair[8].data = storage.data() + 0x5004;
    rejects_pair();
    // Both projections are valid individually, but their output widths differ.
    pair[2].dimensions[0] = 3;
    pair[2].bytes = 48;
    pair[5].dimensions = {3, 3};
    pair[5].strides = {3, 1};
    pair[5].bytes = 36;
    rejects_pair();
    std::cout << "BF16 projection layout validation passed\n";
}
