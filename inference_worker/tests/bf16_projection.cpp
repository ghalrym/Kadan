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
    auto envelope = span(0x5000, DeviceDType::f32, {6, 4}, {4, 1}, 96);
    const auto halves = split_image_silu_pair(envelope, 3);
    check(
        halves[0].bytes == 48 && halves[1].offset == 48
        && halves[1].data == storage.data() + 0x5030
    );
    const auto envelope_original = envelope;
    auto reject_envelope = [&] {
        bool rejected = false;
        try {
            split_image_silu_pair(envelope, 3);
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        check(rejected);
        envelope = envelope_original;
    };
    envelope.dimensions[0] = 5;
    reject_envelope();
    envelope.strides[0] = 5;
    reject_envelope();
    envelope.bytes = 95;
    reject_envelope();
    envelope.dtype = DeviceDType::bf16;
    reject_envelope();
    envelope.data = nullptr;
    reject_envelope();
    envelope.offset = SIZE_MAX;
    reject_envelope();

    std::array residual{
        original[0],
        span(0x5000, DeviceDType::f32, {3, 8}, {8, 1}, 96),
        span(0x3000, DeviceDType::f32, {2, 8}, {8, 1}, 64),
        original[6]
    };
    std::size_t first = 4;
    auto validate_residual = [&] {
        return validate_image_residual(
            0, residual[0], residual[1], residual[2], residual[3], first, 5
        );
    };
    check(validate_residual().rows == 3);
    const auto residual_original = residual;
    auto rejects_residual = [&] {
        bool rejected = false;
        try {
            validate_residual();
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        check(rejected);
        residual = residual_original;
        first = 4;
    };
    for (std::size_t i = 0; i < residual.size(); ++i) {
        residual[i].device = 1;
        rejects_residual();
        residual[i].data = nullptr;
        rejects_residual();
        residual[i].bytes -= 1;
        rejects_residual();
        residual[i].strides.back() = 2;
        rejects_residual();
        residual[i].dtype = DeviceDType::f64;
        rejects_residual();
        residual[i].data = storage.data() + 0x1001;
        rejects_residual();
        for (std::size_t j = 0; j < i; ++j) {
            residual[i].data = residual[j].data;
            rejects_residual();
        }
    }
    residual[1].data = storage.data() + 0x1004;
    rejects_residual();
    residual[2].dimensions[0] = 1;
    rejects_residual();
    residual[0].dimensions[0] = 0;
    rejects_residual();
    first = SIZE_MAX;
    rejects_residual();
    std::array preparation{
        original[0],
        span(0x2200, DeviceDType::f32, {2, 32}, {32, 1}, 256),
        span(0x3200, DeviceDType::f32, {3, 8}, {8, 1}, 96),
        span(0x4200, DeviceDType::f32, {2, 8}, {8, 1}, 64),
        original[6]
    };
    auto validate_preparation = [&] {
        return validate_image_ff_preparation(
            0,
            preparation[0],
            preparation[1],
            preparation[2],
            preparation[3],
            preparation[4],
            1
        );
    };
    const auto prep_shape = validate_preparation();
    check(prep_shape.rows == 3 && prep_shape.in == 8);
    const auto prep_original = preparation;
    auto rejects_preparation = [&] {
        bool rejected = false;
        try {
            validate_preparation();
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        check(rejected);
        preparation = prep_original;
    };
    for (std::size_t i = 0; i < preparation.size(); ++i) {
        preparation[i].device = 1;
        rejects_preparation();
        preparation[i].data = nullptr;
        rejects_preparation();
        preparation[i].bytes -= 1;
        rejects_preparation();
        preparation[i].strides.back() = 2;
        rejects_preparation();
        preparation[i].dtype = DeviceDType::f64;
        rejects_preparation();
        preparation[i].offset = SIZE_MAX;
        rejects_preparation();
        for (std::size_t j = 0; j < i; ++j) {
            preparation[i].data = preparation[j].data;
            rejects_preparation();
        }
    }
    preparation[1].dimensions = {8, 8};
    rejects_preparation();
    preparation[0].dimensions[0] = std::size_t(INT_MAX) + 1;
    rejects_preparation();
    bool bad_text = false;
    try {
        validate_image_ff_preparation(
            0,
            preparation[0],
            preparation[1],
            preparation[2],
            preparation[3],
            preparation[4],
            4
        );
    } catch (const std::invalid_argument&) {
        bad_text = true;
    }
    check(bad_text);
    std::cout << "BF16 projection layout validation passed\n";
}
