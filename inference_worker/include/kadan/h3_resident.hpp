#pragma once
#include <algorithm>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <stdexcept>
#include <vector>

#include "kadan/dense_compute.hpp"
#include "kadan/media_session.hpp"

namespace kadan::video {
// cuBLAS workspace bound for one resident H3 launcher call.
inline constexpr Bytes h3_resident_blas_bytes = 8ULL * 1024 * 1024;
inline constexpr std::size_t h3_lora_rank_limit = 384;

namespace detail {
inline void h3_need(bool value, const char* why) {
    if (!value)
        throw std::invalid_argument(why);
}

inline Bytes h3_aligned(Bytes bytes) {
    return (bytes + 255) / 256 * 256;
}

inline bool h3_matrix(
    const MediaDeviceSpan& s,
    int device,
    DeviceDType dtype,
    std::size_t rows,
    std::size_t columns
) {
    return s.data && s.device == device && s.dtype == dtype
           && s.dimensions == std::vector<std::size_t>{rows, columns}
           && s.strides == std::vector<std::size_t>{columns, 1};
}

inline bool h3_buffer(const MediaDeviceSpan& s, int device, DeviceDType dtype) {
    return s.data && s.device == device && s.dtype == dtype && s.dimensions.size() == 1
           && s.dimensions[0] && s.strides == std::vector<std::size_t>{1};
}

inline bool h3_disjoint(const MediaDeviceSpan& a, const MediaDeviceSpan& b) {
    const auto* x = static_cast<const std::byte*>(a.data);
    const auto* y = static_cast<const std::byte*>(b.data);
    return x + a.bytes <= y || y + b.bytes <= x;
}
}  // namespace detail

// Convrot scratch for `rows` inputs of width `in`: rotated F32 input, INT8 codes
// and per-row scales, then one INT32 (or LoRA F64) column tile in the remainder.
// Returns the widest tile in multiples of four columns, or zero if none fits.
inline std::size_t
h3_convrot_tile(Bytes scratch, std::size_t rows, std::size_t in, bool lora = false) {
    using detail::h3_aligned;
    const auto fixed = h3_aligned(Bytes(rows) * in * 4) + h3_aligned(Bytes(rows) * in)
                       + h3_aligned(Bytes(rows) * 4);
    if (!rows || scratch <= fixed)
        return 0;
    return std::size_t((scratch - fixed) / (Bytes(rows) * (lora ? 8 : 4)) / 4 * 4);
}

struct H3ConvrotShape {
    std::size_t rows = 0, in = 0, out = 0, rank = 0, tile = 0;
};

// Inputs [input [rows,in] F32, weight [out,in] I8, scale [out,1] F32], then either
// [bias [out,1] BF16] or, with Turbo LoRA, [lora-a [rank,in] BF16, lora-b [out,rank]
// BF16]. Outputs
// [output [rows,out] F32, conversion, blas, status] or, with LoRA,
// [output, conversion, lora-low [rows,rank] F64, lora-weights/input F64, blas, status].
inline H3ConvrotShape validate_h3_convrot(
    int device,
    std::span<const MediaDeviceSpan> read,
    std::span<const MediaDeviceSpan> write,
    std::size_t group
) {
    using namespace detail;
    const bool lora = read.size() == 5, bias = read.size() == 4;
    h3_need(
        ((read.size() == 3 || bias) && write.size() == 4)
            || (lora && write.size() == 6),
        "h3_convrot_arity"
    );
    const auto& input = read[0];
    const auto& output = write[0];
    h3_need(
        input.dimensions.size() == 2 && read[1].dimensions.size() == 2,
        "h3_convrot_layout"
    );
    H3ConvrotShape shape{
        input.dimensions[0], input.dimensions[1], read[1].dimensions[0], 0, 0
    };
    h3_need(
        shape.rows && shape.rows <= std::size_t(INT_MAX) && shape.in
            && shape.in <= std::size_t(INT_MAX) && shape.in % 4 == 0 && shape.out
            && shape.out <= std::size_t(INT_MAX) && shape.out % 4 == 0
            && (group == 64 || group == 256) && shape.in % group == 0,
        "h3_convrot_shape"
    );
    h3_need(
        h3_matrix(input, device, DeviceDType::f32, shape.rows, shape.in)
            && h3_matrix(read[1], device, DeviceDType::i8, shape.out, shape.in)
            && h3_matrix(read[2], device, DeviceDType::f32, shape.out, 1)
            && h3_matrix(output, device, DeviceDType::f32, shape.rows, shape.out)
            && (!bias || h3_matrix(read[3], device, DeviceDType::bf16, shape.out, 1)),
        "h3_convrot_layout"
    );
    const auto& conversion = write[1];
    const auto& blas = write[write.size() - 2];
    const auto& status = write.back();
    h3_need(
        h3_buffer(conversion, device, DeviceDType::u8)
            && h3_buffer(blas, device, DeviceDType::u8)
            && blas.bytes >= h3_resident_blas_bytes
            && h3_buffer(status, device, DeviceDType::u32) && status.bytes == 4,
        "h3_convrot_workspace"
    );
    shape.tile = std::min(
        shape.out, h3_convrot_tile(conversion.bytes, shape.rows, shape.in, lora)
    );
    h3_need(shape.tile >= 4, "h3_convrot_workspace");
    if (lora) {
        h3_need(read[3].dimensions.size() == 2, "h3_convrot_lora");
        shape.rank = read[3].dimensions[0];
        h3_need(
            shape.rank && shape.rank <= h3_lora_rank_limit
                && h3_matrix(read[3], device, DeviceDType::bf16, shape.rank, shape.in)
                && h3_matrix(read[4], device, DeviceDType::bf16, shape.out, shape.rank)
                && h3_matrix(write[2], device, DeviceDType::f64, shape.rows, shape.rank)
                && h3_buffer(write[3], device, DeviceDType::f64)
                && write[3].dimensions[0] >= shape.rank * shape.in
                                                 + shape.out * shape.rank
                                                 + shape.rows * shape.in,
            "h3_convrot_lora"
        );
    }
    for (const auto& w : write)
        for (const auto& r : read)
            h3_need(h3_disjoint(w, r), "h3_convrot_alias");
    return shape;
}

// Inputs [gate, value]; outputs [same gate, status]. gate = SiLU(gate) * value.
inline std::size_t validate_h3_silu_gate(
    int device,
    std::span<const MediaDeviceSpan> read,
    std::span<const MediaDeviceSpan> write
) {
    using namespace detail;
    h3_need(read.size() == 2 && write.size() == 2, "h3_silu_arity");
    const auto& gate = read[0];
    h3_need(gate.dimensions.size() == 2, "h3_silu_layout");
    const auto rows = gate.dimensions[0], columns = gate.dimensions[1];
    h3_need(
        h3_matrix(gate, device, DeviceDType::f32, rows, columns)
            && h3_matrix(read[1], device, DeviceDType::f32, rows, columns)
            && write[0].data == gate.data && write[0].dimensions == gate.dimensions
            && write[0].strides == gate.strides && write[0].dtype == gate.dtype
            && write[0].device == device && h3_disjoint(gate, read[1])
            && h3_buffer(write[1], device, DeviceDType::u32) && write[1].bytes == 4,
        "h3_silu_layout"
    );
    return rows * columns;
}

// H3 AdaLN rows: each token group (timestep, modality tag) owns six [hidden]
// vectors: shift/scale/gate before attention, then shift/scale/gate before FF.
// The table is the AdaLN output [timesteps, 3 tags * 6 * hidden], so group
// g = timestep * 3 + tag starts at row-major offset g * 6 * hidden.
inline constexpr std::size_t h3_modulation_rows = 6, h3_modulation_tags = 3;

// Modulation [timesteps, 18 * hidden] F32 and token-groups [tokens] U32 shared by
// the normalization and residual checks. Returns the group count.
inline std::size_t h3_modulation(
    int device,
    const MediaDeviceSpan& modulation,
    const MediaDeviceSpan& groups,
    std::size_t width,
    std::size_t first,
    std::size_t rows
) {
    using namespace detail;
    h3_need(modulation.dimensions.size() == 2, "h3_modulation_layout");
    const auto timesteps = modulation.dimensions[0];
    h3_need(
        timesteps && timesteps <= 3
            && h3_matrix(
                modulation,
                device,
                DeviceDType::f32,
                timesteps,
                h3_modulation_tags * h3_modulation_rows * width
            )
            && h3_buffer(groups, device, DeviceDType::u32)
            && first <= groups.dimensions[0] && rows <= groups.dimensions[0] - first,
        "h3_modulation_layout"
    );
    return timesteps * h3_modulation_tags;
}

// Inputs [hidden rows, projected, modulation, token-groups]; outputs [same hidden
// rows, status]. hidden += projected * modulation[group * 6 + index].
inline std::size_t validate_h3_gated_residual(
    int device,
    std::span<const MediaDeviceSpan> read,
    std::span<const MediaDeviceSpan> write,
    std::size_t first,
    std::size_t index
) {
    using namespace detail;
    h3_need(read.size() == 4 && write.size() == 2, "h3_residual_arity");
    const auto& hidden = read[0];
    h3_need(
        hidden.dimensions.size() == 2 && index < h3_modulation_rows,
        "h3_residual_layout"
    );
    const auto rows = hidden.dimensions[0], width = hidden.dimensions[1];
    h3_modulation(device, read[2], read[3], width, first, rows);
    h3_need(
        h3_matrix(hidden, device, DeviceDType::f32, rows, width)
            && h3_matrix(read[1], device, DeviceDType::f32, rows, width)
            && write[0].data == hidden.data && write[0].dimensions == hidden.dimensions
            && write[0].strides == hidden.strides && write[0].dtype == hidden.dtype
            && write[0].device == device
            && h3_buffer(write[1], device, DeviceDType::u32) && write[1].bytes == 4
            && h3_disjoint(hidden, read[1]) && h3_disjoint(hidden, read[2])
            && h3_disjoint(hidden, read[3]),
        "h3_residual_layout"
    );
    return rows * width;
}

// Inputs [hidden rows, norm-weight [hidden,1] BF16, modulation, token-groups];
// outputs [normalized [rows,hidden], status]. RMS norm (epsilon 1e-5), then
// normalized * (1 + modulation[group*6 + shift + 1]) + modulation[group*6 + shift].
inline std::size_t validate_h3_normalize(
    int device,
    std::span<const MediaDeviceSpan> read,
    std::span<const MediaDeviceSpan> write,
    std::size_t first,
    std::size_t shift
) {
    using namespace detail;
    h3_need(read.size() == 4 && write.size() == 2, "h3_normalize_arity");
    const auto& hidden = read[0];
    h3_need(
        hidden.dimensions.size() == 2 && (shift == 0 || shift == 3),
        "h3_normalize_layout"
    );
    const auto rows = hidden.dimensions[0], width = hidden.dimensions[1];
    h3_modulation(device, read[2], read[3], width, first, rows);
    h3_need(
        h3_matrix(hidden, device, DeviceDType::f32, rows, width)
            && h3_matrix(read[1], device, DeviceDType::bf16, width, 1)
            && h3_matrix(write[0], device, DeviceDType::f32, rows, width)
            && h3_buffer(write[1], device, DeviceDType::u32) && write[1].bytes == 4
            && h3_disjoint(write[0], hidden) && h3_disjoint(write[0], read[1])
            && h3_disjoint(write[0], read[2]) && h3_disjoint(write[0], read[3]),
        "h3_normalize_layout"
    );
    return rows;
}

// Inputs [Q-or-K rows [rows,heads*128], norm-weight [128,1] BF16, rotary
// [tokens,96] F64 cos/sin pairs]; outputs [same rows, status]. Per-head RMS norm
// (epsilon 1e-5), then rotation of channels c and c+48 for c < 48.
inline std::size_t validate_h3_qk_rotary(
    int device,
    std::span<const MediaDeviceSpan> read,
    std::span<const MediaDeviceSpan> write,
    std::size_t first
) {
    using namespace detail;
    h3_need(read.size() == 3 && write.size() == 2, "h3_rotary_arity");
    const auto& x = read[0];
    h3_need(
        x.dimensions.size() == 2 && read[2].dimensions.size() == 2, "h3_rotary_layout"
    );
    const auto rows = x.dimensions[0], width = x.dimensions[1],
               tokens = read[2].dimensions[0];
    h3_need(
        width && width % 128 == 0 && h3_matrix(x, device, DeviceDType::f32, rows, width)
            && h3_matrix(read[1], device, DeviceDType::bf16, 128, 1)
            && h3_matrix(read[2], device, DeviceDType::f64, tokens, 96)
            && first <= tokens && rows <= tokens - first && write[0].data == x.data
            && write[0].dimensions == x.dimensions && write[0].strides == x.strides
            && write[0].dtype == x.dtype && write[0].device == device
            && h3_buffer(write[1], device, DeviceDType::u32) && write[1].bytes == 4
            && h3_disjoint(x, read[1]) && h3_disjoint(x, read[2]),
        "h3_rotary_layout"
    );
    return rows * (width / 128);
}

struct H3AttentionShape {
    std::size_t queries = 0, keys = 0, heads = 0, key_tile = 0;
};

// Scores for every local head of one query tile against one key tile, plus the
// running maximum and total for each (head, query).
inline Bytes h3_attention_scores_bytes(
    std::size_t heads, std::size_t queries, std::size_t key_tile
) {
    return Bytes(heads) * queries * key_tile * sizeof(double);
}

// Widened query, accumulated output, and one widened key/value tile.
inline Bytes
h3_attention_wide_bytes(std::size_t heads, std::size_t queries, std::size_t key_tile) {
    return Bytes(heads) * 128 * (2 * queries + 2 * key_tile) * sizeof(double);
}

// Inputs [query [queries,heads*128], key [tokens,heads*128], value]; outputs
// [attended [queries,heads*128], scores F64, state F64 [2*heads*queries], wide F64,
// blas, status]. Non-causal softmax(QK^T / sqrt(128)) V with key tiles of key_tile.
inline H3AttentionShape validate_h3_attention(
    int device,
    std::span<const MediaDeviceSpan> read,
    std::span<const MediaDeviceSpan> write,
    std::size_t key_tile
) {
    using namespace detail;
    h3_need(read.size() == 3 && write.size() == 6, "h3_attention_arity");
    h3_need(
        read[0].dimensions.size() == 2 && read[1].dimensions.size() == 2,
        "h3_attention_layout"
    );
    H3AttentionShape shape{
        read[0].dimensions[0],
        read[1].dimensions[0],
        read[0].dimensions[1] / 128,
        key_tile
    };
    const auto width = read[0].dimensions[1];
    h3_need(
        shape.queries && shape.queries <= std::size_t(INT_MAX) && shape.keys
            && shape.keys <= std::size_t(INT_MAX) && width % 128 == 0 && shape.heads
            && key_tile && key_tile <= shape.keys
            && h3_matrix(read[0], device, DeviceDType::f32, shape.queries, width)
            && h3_matrix(read[1], device, DeviceDType::f32, shape.keys, width)
            && h3_matrix(read[2], device, DeviceDType::f32, shape.keys, width)
            && h3_matrix(write[0], device, DeviceDType::f32, shape.queries, width)
            && h3_buffer(write[1], device, DeviceDType::f64)
            && write[1].bytes
                   >= h3_attention_scores_bytes(shape.heads, shape.queries, key_tile)
            && h3_buffer(write[2], device, DeviceDType::f64)
            && write[2].dimensions[0] >= 2 * shape.heads * shape.queries
            && h3_buffer(write[3], device, DeviceDType::f64)
            && write[3].bytes
                   >= h3_attention_wide_bytes(shape.heads, shape.queries, key_tile)
            && h3_buffer(write[4], device, DeviceDType::u8)
            && write[4].bytes >= h3_resident_blas_bytes
            && h3_buffer(write[5], device, DeviceDType::u32) && write[5].bytes == 4,
        "h3_attention_layout"
    );
    for (const auto& w : write)
        for (const auto& r : read)
            h3_need(h3_disjoint(w, r), "h3_attention_alias");
    return shape;
}

// Device-resident H3 operations on MediaSession views. Each call enqueues on the
// session's stream and writes a zero/nonzero status word; it never synchronizes.
class H3ResidentLauncher {
public:
    virtual ~H3ResidentLauncher() = default;
    virtual int device() const = 0;
    // Waits for and forgets the bound session stream. Call before that session
    // ends so the next session can bind its own stream.
    virtual void release_stream() = 0;
    // Spans follow validate_h3_convrot. Output = convrot(input) [+ LoRA * scale].
    virtual void convrot(
        void* stream,
        std::span<const MediaDeviceSpan> read,
        std::span<const MediaDeviceSpan> write,
        std::size_t group,
        float lora_scale
    ) = 0;
    virtual void silu_gate(
        void* stream,
        std::span<const MediaDeviceSpan> read,
        std::span<const MediaDeviceSpan> write
    ) = 0;
    virtual void gated_residual(
        void* stream,
        std::span<const MediaDeviceSpan> read,
        std::span<const MediaDeviceSpan> write,
        std::size_t first,
        std::size_t index
    ) = 0;
    virtual void normalize(
        void* stream,
        std::span<const MediaDeviceSpan> read,
        std::span<const MediaDeviceSpan> write,
        std::size_t first,
        std::size_t shift
    ) = 0;
    virtual void qk_rotary(
        void* stream,
        std::span<const MediaDeviceSpan> read,
        std::span<const MediaDeviceSpan> write,
        std::size_t first
    ) = 0;
    virtual void attention(
        void* stream,
        std::span<const MediaDeviceSpan> read,
        std::span<const MediaDeviceSpan> write,
        std::size_t key_tile
    ) = 0;
};

// One request-owner thread, constructed before admission's memory probe so its
// cuBLAS handle is measured and charged. Close before the session finishes.
class CudaH3Launcher final : public H3ResidentLauncher {
public:
    CudaH3Launcher(
        std::shared_ptr<DenseCompute> context_owner,
        std::shared_ptr<MediaSessionActivity> activity,
        std::shared_ptr<Resources> resources,
        int device
    );
    ~CudaH3Launcher() override;
    CudaH3Launcher(const CudaH3Launcher&) = delete;
    CudaH3Launcher& operator=(const CudaH3Launcher&) = delete;
    void close();

    int device() const override {
        return device_;
    }

    void release_stream() override;
    void convrot(
        void*,
        std::span<const MediaDeviceSpan>,
        std::span<const MediaDeviceSpan>,
        std::size_t,
        float
    ) override;
    void silu_gate(
        void*, std::span<const MediaDeviceSpan>, std::span<const MediaDeviceSpan>
    ) override;
    void gated_residual(
        void*,
        std::span<const MediaDeviceSpan>,
        std::span<const MediaDeviceSpan>,
        std::size_t,
        std::size_t
    ) override;
    void normalize(
        void*,
        std::span<const MediaDeviceSpan>,
        std::span<const MediaDeviceSpan>,
        std::size_t,
        std::size_t
    ) override;
    void qk_rotary(
        void*,
        std::span<const MediaDeviceSpan>,
        std::span<const MediaDeviceSpan>,
        std::size_t
    ) override;
    void attention(
        void*,
        std::span<const MediaDeviceSpan>,
        std::span<const MediaDeviceSpan>,
        std::size_t
    ) override;

private:
    std::shared_ptr<DenseCompute> context_owner_;
    std::shared_ptr<MediaSessionActivity> activity_;
    std::shared_ptr<Resources> resources_;
    Handle library_charge_ = 0;
    int device_;
    void* stream_ = nullptr;
    void* blas_ = nullptr;
    bool quarantined_ = false;
    void own(void* stream);
    void bind(void* stream, const MediaDeviceSpan& workspace);
};
}  // namespace kadan::video
