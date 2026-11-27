#pragma once
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <optional>
#include <span>
#include <vector>

#include "kadan/gguf_quantization.hpp"
#include "kadan/text_state.hpp"
#include "kadan/weight_identity.hpp"

namespace kadan {
// Compile-time geometry of Inkling 2 Mini (inkling2_loop), shared between the
// host reference model, the GPU-resident CUDA executor, and unit tests.
struct InklingGeometry {
    static constexpr std::size_t n_phys = 24;
    static constexpr std::size_t n_embd = 2304;
    static constexpr std::size_t n_heads = 24;
    static constexpr std::size_t n_kv_heads = 8;
    static constexpr std::size_t head_dim = 128;
    static constexpr std::size_t n_embd_k_gqa = n_kv_heads * head_dim;  // 1024
    static constexpr std::size_t n_embd_q = n_heads * head_dim;         // 3072
    static constexpr std::size_t n_rel = 16;
    static constexpr std::size_t n_rel_extent = 1024;
    static constexpr std::size_t n_swa_window = 1024;
    static constexpr std::size_t n_block_size = 6;
    static constexpr std::size_t n_conv = 4;
    static constexpr std::size_t n_conv_hist = n_conv - 1;  // 3
    static constexpr std::size_t n_conv_channels = 2 * n_embd_k_gqa + 2 * n_embd;  // 6656
    static constexpr std::size_t n_vocab_padded = 201024;
    static constexpr std::size_t n_vocab_unpadded = 200058;
    static constexpr std::size_t n_experts = 128;
    static constexpr std::size_t n_experts_used = 8;
    static constexpr std::size_t n_ff_exp = 1152;
    static constexpr std::size_t loop_start = 6;
    static constexpr std::size_t loop_end = 18;
    static constexpr std::size_t loop_core_len = loop_end - loop_start;  // 12
    static constexpr std::size_t loop_passes = 3;
    static constexpr std::size_t loop_gate_rank = 64;
    static constexpr std::size_t loop_ctrl_count = 4;
    static constexpr std::size_t n_slots = n_phys + (loop_passes - 1) * loop_core_len;  // 48
    static constexpr std::size_t n_swa_slots = 40;
    static constexpr std::size_t n_gpu_boundary_slots = 2;

    static constexpr bool is_swa_phys(std::size_t ip) {
        return (ip % n_block_size) != (n_block_size - 1);
    }

    static constexpr std::size_t slot_phys(std::size_t slot) {
        return slot < n_phys ? slot : loop_start + (slot - n_phys) % loop_core_len;
    }

    static constexpr std::size_t slot_kv_capacity(std::size_t slot, std::size_t context) {
        return is_swa_phys(slot_phys(slot)) ? std::min(context, n_swa_window) : context;
    }

    static constexpr std::size_t slot_ring_capacity(std::size_t slot, std::size_t context) {
        return is_swa_phys(slot_phys(slot)) ? 2 * slot_kv_capacity(slot, context) : context;
    }
};

// Descriptor for one immutable packed (or F32) matrix loaded into the device
// weight bank. `encoding` empty means raw F32 rows.
struct InklingMatrixDesc {
    WeightIdentity identity{};
    std::optional<gguf::Encoding> encoding;
    std::size_t in = 0;
    std::size_t out = 0;
    std::size_t row_bytes = 0;
    std::function<void(std::size_t, std::span<std::byte>)> source;
};

struct InklingPhysLayerDesc {
    bool is_swa = false;
    bool has_attn_res = false;
    bool has_exp_bias = false;

    std::span<const float> attn_norm;
    std::span<const float> attn_q_inp_norm;
    std::span<const float> attn_k_inp_norm;
    std::span<const float> attn_v_inp_norm;
    std::span<const float> attn_q_clip;
    std::span<const float> attn_rel_bias;
    std::span<const float> attn_rel_clip;
    std::array<std::span<const float>, InklingGeometry::n_conv> attn_k_conv;
    std::array<std::span<const float>, InklingGeometry::n_conv> attn_v_conv;
    std::array<std::span<const float>, InklingGeometry::n_conv> attn_post_conv;
    std::array<std::span<const float>, InklingGeometry::n_conv> ffn_post_conv;
    std::span<const float> attn_res_query;
    std::span<const float> ffn_res_query;
    std::span<const float> ffn_norm;
    std::span<const float> ffn_exp_probs_b;

    InklingMatrixDesc wq, wk, wv, wo, wgate, wrel, wsink;
    InklingMatrixDesc wk_conv_proj, wv_conv_proj;
    InklingMatrixDesc wffn_router;
    std::array<InklingMatrixDesc, InklingGeometry::n_experts> wffn_gate_up;
    std::array<InklingMatrixDesc, InklingGeometry::n_experts> wffn_down;
};

struct InklingLoopLayerDesc {
    std::span<const float> msg_query;
    std::array<InklingMatrixDesc, InklingGeometry::loop_ctrl_count> ctrl_down;
    std::array<InklingMatrixDesc, InklingGeometry::loop_ctrl_count> ctrl_gain;
    std::array<InklingMatrixDesc, InklingGeometry::loop_ctrl_count> ctrl_forget;
};

struct InklingGpuModelDesc {
    std::size_t context = 0;
    std::size_t total_matrix_bytes = 0;
    std::uint32_t initial_variant_id = 0;
    std::span<const float> out_norm;
    std::span<const float> out_res_query;
    InklingMatrixDesc output_weight;
    std::vector<InklingPhysLayerDesc> layers;
    std::vector<InklingLoopLayerDesc> loop_layers;
};

// Descriptor for one differing matrix (or routed expert slice) hot-swapped in
// place into the existing GPU weight buffer without changing device addresses.
struct InklingVariantTensor {
    enum class Role : std::uint8_t {
        attn_output = 0,
        ffn_down_expert = 1
    };

    std::size_t layer = 0;
    Role role = Role::attn_output;
    std::size_t expert = 0;
    InklingMatrixDesc desc;
};

// Option A in-place variant swap tier chosen from spare memory headroom:
//  - gpu_dual_buffer: both variants' differing SoA tensors stay in VRAM (~5 ms D2D).
//  - pinned_host_ram: both variants' differing SoA tensors stay in pinned RAM (~200-400
//  ms H2D).
//  - stream_repack: streams differing tensors from disk and repacks in-place (0 extra
//  cache).
enum class InklingSwapTier : std::uint8_t {
    gpu_dual_buffer = 0,
    pinned_host_ram = 1,
    stream_repack = 2
};

inline InklingSwapTier select_inkling_swap_tier(
    std::size_t delta_bytes,
    std::size_t spare_vram_bytes,
    std::size_t spare_host_ram_bytes
) {
    constexpr std::size_t vram_headroom = 512ULL << 20;
    constexpr std::size_t host_headroom = 2ULL << 30;
    if (delta_bytes > 0
        && delta_bytes <= (std::numeric_limits<std::size_t>::max() - vram_headroom) / 2
        && spare_vram_bytes >= 2 * delta_bytes + vram_headroom) {
        return InklingSwapTier::gpu_dual_buffer;
    }
    if (delta_bytes > 0
        && delta_bytes <= (std::numeric_limits<std::size_t>::max() - host_headroom) / 2
        && spare_host_ram_bytes >= 2 * delta_bytes + host_headroom) {
        return InklingSwapTier::pinned_host_ram;
    }
    return InklingSwapTier::stream_repack;
}

// Pure-C++ scratch sizing helper (unit-testable without CUDA).
// Computes the required scratch arena bytes on the single state device for
// `max_rows` (up to 512) of GPU-resident Inkling 2 Mini execution.
inline std::size_t inkling_gpu_scratch_bytes(std::size_t max_rows) {
    constexpr std::size_t align = 256;
    auto up = [](std::size_t n) { return (n + align - 1) / align * align; };
    using G = InklingGeometry;
    const std::size_t rows = std::max<std::size_t>(max_rows, 1);
    const std::size_t expert_rows = rows * G::n_experts_used;
    // Residual stream pool (9 buffers of [rows, n_embd]), attention buffers
    // (q, k, v, r, sinks, gate, attn_out, 3 inputs, 4 messages, control
    // concat/gate/gain/forget), MoE buffers (router, gather/down f32+fp16,
    // gate_up f32, act f32+fp16), FP16 weight decode tile (3072*2304*2 =
    // 14.16 MiB), Q8 quantized input buffer, routing metadata, output logits,
    // attention partials (32 MiB), and cuBLAS workspace (8 MiB).
    std::size_t total = 0;
    total += 9 * up(rows * G::n_embd * sizeof(float));             // cur_a/b, prelude, bank0/out, blocks[0..3]
    total += 7 * up(rows * G::n_embd * sizeof(float));             // inputs[3] + messages[4]
    total += 3 * up(rows * G::n_embd_q * sizeof(float));           // q, gate, attn_out
    total += 4 * up(rows * G::n_embd_k_gqa * sizeof(float));       // k_raw, v_raw, k, v
    total += up(rows * G::n_heads * G::n_rel * sizeof(float));     // r_cur
    total += up(rows * G::n_heads * sizeof(float));                // sinks
    total += 2 * up(rows * G::n_conv * G::n_kv_heads * sizeof(float));  // dyn conv taps
    total += up(rows * 2 * G::n_embd * sizeof(float));             // ctrl concat
    total += up(rows * G::loop_gate_rank * sizeof(float));         // ctrl gate
    total += 2 * up(rows * G::n_embd * sizeof(float));             // ctrl gain/forget
    total += up(rows * G::n_experts * sizeof(float));              // router logits
    total += up(expert_rows * G::n_embd * sizeof(float));          // moe gather / down_out
    total += up(expert_rows * G::n_embd * sizeof(std::uint16_t));  // moe gather fp16
    total += up(expert_rows * 2 * G::n_ff_exp * sizeof(float));    // gate_up_out
    total += up(expert_rows * G::n_ff_exp * sizeof(float));        // act_out f32
    total += up(expert_rows * G::n_ff_exp * sizeof(std::uint16_t));  // act_out fp16
    total += up(rows * 2 * G::n_embd * sizeof(std::uint16_t));     // dense / ctrl concat fp16 input
    total += up(G::n_embd_q * G::n_embd * sizeof(std::uint16_t));  // decoded fp16 weight tile
    total += up(expert_rows * (G::n_embd / 32) * 36);              // Q8 quantized input buffer
    total += up(G::n_vocab_unpadded * sizeof(float));              // final logits
    total += up(32ULL * 1024 * 1024);                              // attention partials
    total += up(8ULL * 1024 * 1024);                               // cuBLAS workspace
    total += up(2ULL * 1024 * 1024);                               // routing/indices/staging
    return total;
}

// Exact bytes of one GPU-resident boundary snapshot (40 active FP16 SWA windows
// + 48 F32 short-conv states). Full-attention KV mirrors are append-only and
// rewind in place with zero bytes copied.
inline std::size_t inkling_gpu_boundary_bytes(std::size_t context) {
    using G = InklingGeometry;
    const std::size_t swa_win = std::min(context, G::n_swa_window);
    const std::size_t swa_bytes =
        G::n_swa_slots * 2 * swa_win * G::n_embd_k_gqa * sizeof(std::uint16_t);
    const std::size_t conv_bytes =
        G::n_slots * G::n_conv_hist * G::n_conv_channels * sizeof(float);
    return swa_bytes + conv_bytes;
}

// GPU-resident Inkling 2 Mini forward executor owned by a single-GPU
// DenseCompute when the model's packed weights + FP16 KV state fit in VRAM.
class InklingGpuExecutor {
public:
    virtual ~InklingGpuExecutor() = default;

    // Resets GPU-resident KV/conv validity (invalidates any captured decode graph).
    virtual void reset() = 0;

    // Copies the 40 active FP16 SWA windows and 48 F32 short-conv states on GPU
    // (D2D) into boundary slot `slot_idx` (< n_gpu_boundary_slots) at `valid_tokens`,
    // tagged with `variant_id` so boundaries never leak across weight variants.
    virtual void save_boundary(
        unsigned /*slot_idx*/,
        std::size_t /*valid_tokens*/,
        std::uint32_t /*variant_id*/,
        const std::atomic_bool& /*cancel*/
    ) {
    }

    // Restores GPU state to `boundary_tokens` from boundary slot `slot_idx` D2D,
    // rewinding the 8 full-attention KV mirrors in place (0 bytes copied).
    // Returns false if the GPU snapshot or live full-attention mirrors are absent
    // or belong to a different weight variant.
    virtual bool restore_boundary(
        unsigned /*slot_idx*/,
        std::size_t /*boundary_tokens*/,
        std::uint32_t /*variant_id*/,
        const std::atomic_bool& /*cancel*/
    ) {
        return false;
    }

    // Hot-swaps the differing variant tensors in place into the existing GPU
    // buffers (`wo`, `d_down_ptrs`), preserving all device addresses and keeping
    // any captured single-token decode CUDA Graph valid.
    virtual bool swap_variant(
        std::uint32_t /*variant_id*/,
        std::span<const InklingVariantTensor> /*target_tensors*/,
        const std::atomic_bool& /*cancel*/
    ) {
        return false;
    }

    // Copies GPU-resident FP16 KV + F32 conv state back into host `slots`
    // for prefix checkpointing (`CheckpointPool`). Only copies [0..valid_tokens)
    // for full-attention slots and the active ring for SWA slots.
    virtual void sync_to_host(
        std::span<text::InklingLayerState> slots,
        std::size_t valid_tokens,
        const std::atomic_bool& cancel
    ) = 0;

    // Uploads host `slots` (after checkpoint restore) into GPU FP16 KV + F32 conv.
    virtual void sync_from_host(
        std::span<const text::InklingLayerState> slots,
        std::size_t valid_tokens,
        const std::atomic_bool& cancel
    ) = 0;

    // Runs all 48 Inkling 2 Mini logical slots 100% on GPU from pre-embedded
    // `input_embeddings` ([rows * n_embd]). Returns `logits` ([n_vocab_unpadded])
    // when `want_logits` is true and `argmax_out` is null; when `argmax_out` is
    // non-null, computes exact argmax + finiteness on GPU and writes only the
    // winning token ID (avoiding an 800 KB logits D2H copy). Returns false if
    // the device cannot admit the full model or KV state (clean Strata fallback).
    virtual bool forward(
        std::span<const float> input_embeddings,
        std::size_t rows,
        std::size_t position,
        bool want_logits,
        std::vector<float>& logits_out,
        const std::function<void(const char*, std::size_t)>& progress,
        const std::atomic_bool& cancel,
        unsigned* argmax_out = nullptr
    ) = 0;
};
}  // namespace kadan
