// Layout-only units. No CUDA calls, matrix arithmetic or model execution.
#include "kadan/bf16_projection.hpp"

#include <array>
#include <iostream>
#include <limits>

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

    std::array final_spans{
        original[0],
        span(0x3000, DeviceDType::f32, {1, 8}, {8, 1}, 32),
        span(0x5000, DeviceDType::f32, {3, 8}, {8, 1}, 96),
        original[6]
    };
    const auto final_original = final_spans;
    auto validate_final = [&] {
        return validate_image_final_normalization(
            0, final_spans[0], final_spans[1], final_spans[2], final_spans[3]
        );
    };
    check(validate_final().rows == 3 && validate_final().in == 8);
    auto rejects_final = [&] {
        bool rejected = false;
        try {
            validate_final();
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        check(rejected);
        final_spans = final_original;
    };
    for (std::size_t i = 0; i < final_spans.size(); ++i) {
        final_spans[i].device = 1;
        rejects_final();
        final_spans[i].data = nullptr;
        rejects_final();
        final_spans[i].bytes -= 1;
        rejects_final();
        final_spans[i].strides.back() = 2;
        rejects_final();
        final_spans[i].dtype = DeviceDType::f64;
        rejects_final();
        final_spans[i].offset = SIZE_MAX;
        rejects_final();
        final_spans[i].data = storage.data() + 0x1001;
        rejects_final();
        for (std::size_t j = 0; j < i; ++j) {
            final_spans[i].data = final_spans[j].data;
            rejects_final();
        }
    }
    final_spans[2].data = storage.data() + 0x1004;
    rejects_final();
    final_spans[0].dimensions[0] = 0;
    rejects_final();
    final_spans[0].dimensions[0] = std::size_t(INT_MAX) + 1;
    rejects_final();
    final_spans[1].dimensions[0] = 2;
    rejects_final();

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
        return validate_image_modulation(
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
        validate_image_modulation(
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
    auto updated = preparation[0];
    auto validate_attention = [&] {
        return validate_image_attention_residual(
            0,
            preparation[0],
            preparation[2],
            preparation[1],
            updated,
            preparation[3],
            preparation[4],
            1
        );
    };
    check(validate_attention().rows == 3);
    auto rejects_updated = [&] {
        bool rejected = false;
        try {
            validate_attention();
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        check(rejected);
        updated = preparation[0];
    };
    updated.data = preparation[2].data;
    rejects_updated();
    updated.device = 1;
    rejects_updated();
    updated.dtype = DeviceDType::bf16;
    rejects_updated();
    updated.dimensions[0] = 2;
    rejects_updated();
    updated.strides[0] = 1;
    rejects_updated();
    updated.bytes -= 4;
    rejects_updated();
    updated.offset = 4;
    rejects_updated();
    updated.generation += 1;
    rejects_updated();
    // The shared layout must fit the attention residual's one-dimensional grid.
    preparation[0].dimensions = {INT_MAX, 512};
    rejects_preparation();
    std::array frequency{
        span(0x1000, DeviceDType::f32, {2, 256}, {256, 1}, 2048), original[6]
    };
    const auto frequency_original = frequency;
    auto validate_frequency = [&](float timestep) {
        validate_image_frequency(0, frequency[0], frequency[1], timestep);
    };
    auto rejects_frequency = [&](float timestep = .625f) {
        bool rejected = false;
        try {
            validate_frequency(timestep);
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        check(rejected);
        frequency = frequency_original;
    };
    for (const float timestep : {0.f, -0.f, .625f, 1.f})
        validate_frequency(timestep);
    for (const float timestep :
         {-1.f,
          1.01f,
          std::numeric_limits<float>::infinity(),
          std::numeric_limits<float>::quiet_NaN()})
        rejects_frequency(timestep);
    for (std::size_t i = 0; i < frequency.size(); ++i) {
        frequency[i].device = 1;
        rejects_frequency();
        frequency[i].data = nullptr;
        rejects_frequency();
        frequency[i].bytes -= 1;
        rejects_frequency();
        frequency[i].strides.back() = 2;
        rejects_frequency();
        frequency[i].dimensions[0] += 1;
        rejects_frequency();
        frequency[i].dtype = DeviceDType::bf16;
        rejects_frequency();
        frequency[i].offset = SIZE_MAX;
        rejects_frequency();
        frequency[i].data = static_cast<std::byte*>(frequency[i].data) + 1;
        rejects_frequency();
    }
    frequency[1].data = frequency[0].data;
    rejects_frequency();
    frequency[1].data = static_cast<std::byte*>(frequency[0].data) + 2044;
    rejects_frequency();
    std::array text_norm{
        span(0x1000, DeviceDType::f32, {3, 8}, {8, 1}, 96),
        span(0x2000, DeviceDType::bf16, {8, 1}, {1, 1}, 16),
        span(0x3000, DeviceDType::f32, {3, 8}, {8, 1}, 96),
        original[6]
    };
    auto validate_text_norm = [&] {
        return validate_image_text_normalization(
            0, text_norm[0], text_norm[1], text_norm[2], text_norm[3]
        );
    };
    check(validate_text_norm().in == 8);
    text_norm[1].dtype = DeviceDType::f32;
    text_norm[1].bytes = 32;
    check(validate_text_norm().rows == 3);
    const auto text_norm_original = text_norm;
    auto rejects_text_norm = [&] {
        bool rejected = false;
        try {
            validate_text_norm();
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        check(rejected);
        text_norm = text_norm_original;
    };
    for (std::size_t i = 0; i < text_norm.size(); ++i) {
        text_norm[i].device = 1;
        rejects_text_norm();
        text_norm[i].data = nullptr;
        rejects_text_norm();
        text_norm[i].dtype = DeviceDType::f64;
        rejects_text_norm();
        text_norm[i].strides.back() = 2;
        rejects_text_norm();
        text_norm[i].bytes -= 1;
        rejects_text_norm();
        text_norm[i].offset = SIZE_MAX;
        rejects_text_norm();
        text_norm[i].data = storage.data() + 0x1001;
        rejects_text_norm();
        for (std::size_t j = 0; j < i; ++j) {
            text_norm[i].data = text_norm[j].data;
            rejects_text_norm();
        }
    }
    text_norm[1].dimensions = {1, 8};
    rejects_text_norm();
    text_norm[2].data = static_cast<std::byte*>(text_norm[0].data) + 4;
    rejects_text_norm();
    std::cout << "BF16 projection layout validation passed\n";
}
