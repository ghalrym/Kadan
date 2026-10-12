#pragma once
#include <array>
#include <climits>
#include <stdexcept>

#include "kadan/media_session.hpp"

namespace kadan::video {
inline constexpr Bytes bf16_projection_workspace_bytes = 8 * 1024 * 1024;

struct Bf16ProjectionShape {
    std::size_t rows, in, out;
};

// Host-only layout validation; no device reads or projection arithmetic.
inline Bf16ProjectionShape validate_bf16_projection(
    int device,
    const MediaDeviceSpan& input,
    const MediaDeviceSpan& weight,
    const MediaDeviceSpan& bias,
    const MediaDeviceSpan& rounded,
    const MediaDeviceSpan& output,
    const MediaDeviceSpan& workspace,
    const MediaDeviceSpan& status
) {
    auto require = [](bool valid) {
        if (!valid)
            throw std::invalid_argument("bf16_projection_layout");
    };
    require(input.dimensions.size() == 2 && weight.dimensions.size() == 2);
    const auto rows = input.dimensions[0], in = input.dimensions[1],
               out = weight.dimensions[0];
    require(rows && in && out && rows <= INT_MAX && in <= INT_MAX && out <= INT_MAX);
    auto matrix = [&](const MediaDeviceSpan& span,
                      DeviceDType dtype,
                      std::size_t r,
                      std::size_t c,
                      std::size_t width) {
        require(r <= SIZE_MAX / c / width);
        require(
            span.dtype == dtype && span.dimensions == std::vector<std::size_t>{r, c}
            && span.strides == std::vector<std::size_t>{c, 1}
            && span.bytes == r * c * width
        );
    };
    matrix(input, DeviceDType::f32, rows, in, 4);
    matrix(weight, DeviceDType::bf16, out, in, 2);
    matrix(rounded, DeviceDType::bf16, rows, in, 2);
    matrix(output, DeviceDType::f32, rows, out, 4);
    // One-dimensional kernels use CUDA's signed 31-bit grid-x limit.
    require(
        rows * in <= std::size_t(INT_MAX) * 256
        && rows * out <= std::size_t(INT_MAX) * 256
    );
    auto vector = [&](const MediaDeviceSpan& span,
                      DeviceDType dtype,
                      std::size_t count,
                      std::size_t width) {
        require(count <= SIZE_MAX / width);
        require(
            span.dtype == dtype && span.dimensions == std::vector<std::size_t>{count}
            && span.strides == std::vector<std::size_t>{1}
            && span.bytes == count * width
        );
    };
    if (bias.data)
        vector(bias, DeviceDType::f32, out, 4);
    else
        require(!bias.bytes && bias.dimensions.empty() && bias.strides.empty());
    require(workspace.bytes >= bf16_projection_workspace_bytes);
    vector(workspace, DeviceDType::u8, workspace.bytes, 1);
    require(reinterpret_cast<std::uintptr_t>(workspace.data) % 256 == 0);
    vector(status, DeviceDType::u32, 1, 4);
    const std::array spans{
        &input, &weight, &bias, &rounded, &output, &workspace, &status
    };
    for (std::size_t i = 0; i < spans.size(); ++i) {
        const auto& a = *spans[i];
        require(a.device == device && (a.data || !a.bytes));
        const auto x = reinterpret_cast<std::uintptr_t>(a.data);
        const auto alignment = a.dtype == DeviceDType::bf16 ? 2u : 4u;
        require(!a.bytes || x % alignment == 0);
        for (std::size_t j = 0; j < i; ++j) {
            const auto& b = *spans[j];
            const auto y = reinterpret_cast<std::uintptr_t>(b.data);
            require(
                !a.bytes || !b.bytes || (x <= y ? y - x >= a.bytes : x - y >= b.bytes)
            );
        }
    }
    return {rows, in, out};
}

// Bias-free image FF pair. Scratch is reused serially by the two projections;
// each producer has its own status so the second launch cannot erase the first.
inline Bf16ProjectionShape validate_image_silu_pair(
    int device,
    const MediaDeviceSpan& input,
    const MediaDeviceSpan& gate_weight,
    const MediaDeviceSpan& up_weight,
    const MediaDeviceSpan& rounded,
    const MediaDeviceSpan& gate,
    const MediaDeviceSpan& up,
    const MediaDeviceSpan& workspace,
    const MediaDeviceSpan& gate_status,
    const MediaDeviceSpan& up_status,
    const MediaDeviceSpan& status
) {
    const MediaDeviceSpan bias{nullptr, device, DeviceDType::f32, {}, {}, 0, 0, 0};
    const auto shape = validate_bf16_projection(
        device, input, gate_weight, bias, rounded, gate, workspace, gate_status
    );
    const auto up_shape = validate_bf16_projection(
        device, input, up_weight, bias, rounded, up, workspace, up_status
    );
    // Also validate the aggregate status with the existing scalar layout rules.
    validate_bf16_projection(
        device, input, gate_weight, bias, rounded, gate, workspace, status
    );
    if (shape.out != up_shape.out)
        throw std::invalid_argument("image_silu_pair_layout");
    const std::array spans{
        &input,
        &gate_weight,
        &up_weight,
        &rounded,
        &gate,
        &up,
        &workspace,
        &gate_status,
        &up_status,
        &status
    };
    // Per-projection validation does not see cross-producer overlap.
    for (std::size_t i = 0; i < spans.size(); ++i)
        for (std::size_t j = 0; j < i; ++j) {
            const auto x = reinterpret_cast<std::uintptr_t>(spans[i]->data),
                       y = reinterpret_cast<std::uintptr_t>(spans[j]->data);
            if (x <= y ? y - x < spans[i]->bytes : x - y < spans[j]->bytes)
                throw std::invalid_argument("image_silu_pair_layout");
        }
    return shape;
}

// A single session output owns this envelope; only completed halves are gathered.
inline std::array<MediaDeviceSpan, 2>
split_image_silu_pair(const MediaDeviceSpan& envelope, std::size_t rows) {
    const auto pointer = reinterpret_cast<std::uintptr_t>(envelope.data);
    if (!rows || rows > SIZE_MAX / 2 || envelope.dimensions.size() != 2
        || !envelope.dimensions[1] || envelope.dimensions[0] != 2 * rows
        || rows > SIZE_MAX / envelope.dimensions[1] / 8
        || envelope.dtype != DeviceDType::f32 || !pointer || pointer % 4
        || envelope.strides != std::vector<std::size_t>{envelope.dimensions[1], 1}
        || envelope.bytes != rows * envelope.dimensions[1] * 8
        || pointer > UINTPTR_MAX - envelope.bytes
        || envelope.offset > SIZE_MAX - envelope.bytes)
        throw std::invalid_argument("image_ff_pair_layout");
    auto gate = envelope;
    gate.dimensions[0] = rows;
    gate.bytes /= 2;
    auto up = gate;
    up.data = static_cast<std::byte*>(gate.data) + gate.bytes;
    up.offset += gate.bytes;
    return {gate, up};
}

// Layout only; hidden is both the certified input and the exclusive output.
inline Bf16ProjectionShape validate_image_residual(
    int device,
    const MediaDeviceSpan& hidden,
    const MediaDeviceSpan& projected,
    const MediaDeviceSpan& gates,
    const MediaDeviceSpan& status,
    std::size_t first,
    std::size_t /*text*/
) {
    auto require = [](bool valid) {
        if (!valid)
            throw std::invalid_argument("image_residual_layout");
    };
    require(hidden.dimensions.size() == 2);
    const auto rows = hidden.dimensions[0], state = hidden.dimensions[1];
    require(
        rows && state && rows <= SIZE_MAX / state / 4 && state <= SIZE_MAX / 8
        && rows * state <= std::size_t(INT_MAX) * 256 && first <= SIZE_MAX - rows
    );
    auto matrix = [&](const MediaDeviceSpan& span, std::size_t count) {
        require(
            span.dtype == DeviceDType::f32
            && span.dimensions == std::vector<std::size_t>{count, state}
            && span.strides == std::vector<std::size_t>{state, 1}
            && span.bytes == count * state * sizeof(float)
        );
    };
    matrix(hidden, rows);
    matrix(projected, rows);
    matrix(gates, 2);
    require(
        status.dtype == DeviceDType::u32
        && status.dimensions == std::vector<std::size_t>{1}
        && status.strides == std::vector<std::size_t>{1}
        && status.bytes == sizeof(std::uint32_t)
    );
    const std::array spans{&hidden, &projected, &gates, &status};
    for (std::size_t i = 0; i < spans.size(); ++i) {
        const auto& a = *spans[i];
        require(
            a.device == device && a.data
            && reinterpret_cast<std::uintptr_t>(a.data) % 4 == 0
        );
        for (std::size_t j = 0; j < i; ++j) {
            const auto x = reinterpret_cast<std::uintptr_t>(a.data),
                       y = reinterpret_cast<std::uintptr_t>(spans[j]->data);
            require(x <= y ? y - x >= a.bytes : x - y >= spans[j]->bytes);
        }
    }
    return {rows, state, state};
}

// FF stage of image.cpp norm_mod: no affine normalization checkpoint weight.
inline Bf16ProjectionShape validate_image_ff_preparation(
    int device,
    const MediaDeviceSpan& hidden,
    const MediaDeviceSpan& modulation,
    const MediaDeviceSpan& normalized,
    const MediaDeviceSpan& gates,
    const MediaDeviceSpan& status,
    std::size_t text
) {
    auto require = [](bool valid) {
        if (!valid)
            throw std::invalid_argument("image_ff_preparation_layout");
    };
    require(hidden.dimensions.size() == 2);
    const auto rows = hidden.dimensions[0], state = hidden.dimensions[1];
    require(
        rows && state && rows <= INT_MAX && rows <= SIZE_MAX / state / 4
        && state <= SIZE_MAX / 32 && state * 8 <= std::size_t(INT_MAX) * 256
        && text <= rows
    );
    auto matrix = [&](const MediaDeviceSpan& span, std::size_t r, std::size_t c) {
        require(
            span.dtype == DeviceDType::f32
            && span.dimensions == std::vector<std::size_t>{r, c}
            && span.strides == std::vector<std::size_t>{c, 1}
            && span.bytes == r * c * sizeof(float)
        );
    };
    matrix(hidden, rows, state);
    matrix(modulation, 2, 4 * state);
    matrix(normalized, rows, state);
    matrix(gates, 2, state);
    require(
        status.dtype == DeviceDType::u32
        && status.dimensions == std::vector<std::size_t>{1}
        && status.strides == std::vector<std::size_t>{1}
        && status.bytes == sizeof(std::uint32_t)
    );
    const std::array spans{&hidden, &modulation, &normalized, &gates, &status};
    for (std::size_t i = 0; i < spans.size(); ++i) {
        const auto& a = *spans[i];
        const auto x = reinterpret_cast<std::uintptr_t>(a.data);
        require(
            a.device == device && x && x % 4 == 0 && x <= UINTPTR_MAX - a.bytes
            && a.offset <= SIZE_MAX - a.bytes
        );
        for (std::size_t j = 0; j < i; ++j) {
            const auto y = reinterpret_cast<std::uintptr_t>(spans[j]->data);
            require(x <= y ? y - x >= a.bytes : x - y >= spans[j]->bytes);
        }
    }
    return {rows, state, state};
}

// The host scheduler uses this boundary without linking CUDA or doing arithmetic.
// Projection inputs are [input,gate-weight,up-weight] or [input,down-weight];
// outputs are [pair,rounded,workspace,gate-status,up-status,aggregate-status]
// or [down,rounded,workspace,aggregate-status]. Pair is compact [2*rows,local-out].
class ImageFeedForwardLauncher {
public:
    virtual ~ImageFeedForwardLauncher() = default;
    virtual int device() const = 0;
    // Inputs [hidden,modulation]; outputs [normalized,gates,aggregate-status].
    // Complete both outputs before the FF consumer can bind them.
    virtual void prepare(
        void*,
        std::span<const MediaDeviceSpan>,
        std::span<const MediaDeviceSpan>,
        std::size_t text
    ) = 0;
    virtual void
    pair(void*, std::span<const MediaDeviceSpan>, std::span<const MediaDeviceSpan>) = 0;
    virtual void
    down(void*, std::span<const MediaDeviceSpan>, std::span<const MediaDeviceSpan>) = 0;
    virtual void residual(
        void*,
        const MediaDeviceSpan& hidden,
        const MediaDeviceSpan& projected,
        const MediaDeviceSpan& gates,
        const MediaDeviceSpan& status,
        std::size_t first,
        std::size_t text
    ) = 0;
};

// One request-owner thread. Construct before admission's fresh memory probe so
// library handle costs are measured and retained in the same worker ledger;
// declare the workspace above in the consumed manifest. Use WorkerCompute's same
// activity guard. Close after draining, before the session destroys its stream;
// failed cleanup retains the worker busy charge through process shutdown.
class CudaBf16Projection final : public ImageFeedForwardLauncher {
public:
    CudaBf16Projection(
        std::shared_ptr<DenseCompute> context_owner,
        std::shared_ptr<MediaSessionActivity> activity,
        std::shared_ptr<Resources> resources,
        Workload workload,
        int device
    );
    ~CudaBf16Projection() override;
    void close();

    int device() const override {
        return device_;
    }

    void prepare(
        void*,
        std::span<const MediaDeviceSpan>,
        std::span<const MediaDeviceSpan>,
        std::size_t
    ) override;
    void pair(
        void*, std::span<const MediaDeviceSpan>, std::span<const MediaDeviceSpan>
    ) override;
    void down(
        void*, std::span<const MediaDeviceSpan>, std::span<const MediaDeviceSpan>
    ) override;
    void residual(
        void*,
        const MediaDeviceSpan&,
        const MediaDeviceSpan&,
        const MediaDeviceSpan&,
        const MediaDeviceSpan&,
        std::size_t,
        std::size_t
    ) override;
    CudaBf16Projection(const CudaBf16Projection&) = delete;
    CudaBf16Projection& operator=(const CudaBf16Projection&) = delete;
    // Pass the stream supplied by MediaSession::launch. Rounded input, output,
    // workspace and status are exclusive outputs of that same launch event.
    void launch(
        void* stream,
        const MediaDeviceSpan& input,
        const MediaDeviceSpan& weight,
        const MediaDeviceSpan& bias,
        const MediaDeviceSpan& rounded,
        const MediaDeviceSpan& output,
        const MediaDeviceSpan& workspace,
        const MediaDeviceSpan& status
    );
    // Gate is overwritten with SiLU(gate) * up using image's double-intermediate
    // formula and F32 cast before multiplication. All validation precedes enqueue.
    // Pass certified input/weights and exclusive rounded/gate/up/workspace/three
    // status outputs from ONE MediaSession::launch event. Complete gate and any
    // other consumed outputs against aggregate status; projection statuses are
    // internal producers, not separate session completion certificates.
    void launch_image_silu_pair(
        void* stream,
        const MediaDeviceSpan& input,
        const MediaDeviceSpan& gate_weight,
        const MediaDeviceSpan& up_weight,
        const MediaDeviceSpan& rounded,
        const MediaDeviceSpan& gate,
        const MediaDeviceSpan& up,
        const MediaDeviceSpan& workspace,
        const MediaDeviceSpan& gate_status,
        const MediaDeviceSpan& up_status,
        const MediaDeviceSpan& status
    );

private:
    std::shared_ptr<DenseCompute> context_owner_;
    std::shared_ptr<MediaSessionActivity> activity_;
    std::shared_ptr<Resources> resources_;
    Handle library_charge_ = 0;
    int device_;
    void* stream_ = nullptr;
    void* blas_ = nullptr;
    bool quarantined_ = false;
};
}  // namespace kadan::video
