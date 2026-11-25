#pragma once

// GPU-resident Inkling 2 Mini forward executor (Stages 2 & 3):
// - Keeps all activations across all 48 logical slots in VRAM on the single
//   state device, with FP16 KV caches in arena.mirrors and F32 short-conv
//   state + small layer params in arena.recurrent / arena.rel_biases so
//   WorkerCompute::idle() exact-byte accounting and cleanup stay intact.
// - Prefill (m >= 16): FP16 Tensor Core GEMM via cublasGemmEx (FP32 accum).
// - Decode & small-row experts (m < 16): Q8 quantized input + __dp4a packed
//   IQ4_NL / Q8_0 matvec (zero F32 weight expansion).
// - Fused Inkling kernels: mix_states+rms_norm, ctrl_blend+rms_norm,
//   qkvr_clip+tau+fp16_kv_store, attn_fp16+sigmoid_gate, post_conv+residual,
//   indirect-GPU MoE gate_up+SiTU and deterministic down+reduce, output+softcap.
// - CUDA Graph capture & replay for the complete 48-slot single-token decode pass.

namespace inkling_gpu_detail {

using G = kadan::InklingGeometry;

constexpr std::uint64_t kInklingConvKey = 0x494E4B0000000001ULL;
constexpr std::uint64_t kInklingParamsKey = 0x494E4B0000000002ULL;
constexpr std::uint64_t kInklingStepKey = 0x494E4B0000000003ULL;

constexpr float kRmsEps = 1e-6f;
constexpr float kQkClip = 22.627416997969522f;  // 16 * sqrt(2)
constexpr float kRelClip = 8.0f;
constexpr float kLogScaleFloor = 8192.0f;
constexpr float kLogScaleAlpha = 0.1f;
constexpr float kExpertScale = 8.0f;
constexpr float kExpertNormEps = 1e-20f;
constexpr float kSituBeta = 4.0f;
constexpr float kSituLinearBeta = 25.0f;
constexpr float kFinalSoftcap = 30.0f;

constexpr std::size_t kAttnPartialBytes = 32ULL * 1024 * 1024;
constexpr std::size_t kBlasWorkspaceBytes = 8ULL * 1024 * 1024;
constexpr std::size_t kRoutingMetaBytes = 2ULL * 1024 * 1024;
constexpr unsigned kMaxOutputChunks = 64;

enum WeightKind : int {
    kWKindF32 = 0,
    kWKindQ8_0 = 1,
    kWKindIQ4_NL = 2,
};

struct alignas(4) Q8Block {
    float d;
    std::int32_t qs[8];  // 32 signed int8 packed as 8 int32 for __dp4a
};
static_assert(sizeof(Q8Block) == 36);

struct StepParams {
    unsigned position;
    unsigned pad[3];
};

struct ArgmaxResult {
    unsigned token;
    unsigned nonfinite;
};

struct OutputChunkDesc {
    const unsigned char* qs;
    const __half* scales;
    unsigned first_row;
    unsigned row_count;
};

struct MatrixRef {
    const unsigned char* qs = nullptr;
    const __half* scales = nullptr;
    std::size_t qs_row_bytes = 0;
    unsigned in = 0;
    unsigned out = 0;
    WeightKind kind = kWKindF32;
};

inline std::size_t soa_qs_row_bytes(std::size_t in, WeightKind kind) {
    if (kind == kWKindF32)
        return in * sizeof(float);
    return (in / 32u) * (kind == kWKindIQ4_NL ? 16u : 32u);
}

inline std::size_t soa_scales_offset(std::size_t rows, std::size_t in, WeightKind kind) {
    if (kind == kWKindF32)
        return 0;
    return rows * soa_qs_row_bytes(in, kind);
}

struct PhysLayerGpu {
    bool is_swa = false;
    bool has_attn_res = false;
    bool has_exp_bias = false;

    // Device pointers into immutable layer_params / rel_biases allocations
    const float* attn_norm = nullptr;
    const float* attn_q_inp_norm = nullptr;
    const float* attn_k_inp_norm = nullptr;
    const float* attn_v_inp_norm = nullptr;
    const float* attn_q_clip = nullptr;
    const float* attn_rel_bias = nullptr;
    const float* attn_rel_clip = nullptr;
    const float* attn_k_conv[G::n_conv]{};
    const float* attn_v_conv[G::n_conv]{};
    const float* attn_post_conv[G::n_conv]{};
    const float* ffn_post_conv[G::n_conv]{};
    const float* attn_res_query = nullptr;
    const float* ffn_res_query = nullptr;
    const float* ffn_norm = nullptr;
    const float* ffn_exp_probs_b = nullptr;

    MatrixRef wq, wk, wv, wo, wgate, wrel, wsink;
    MatrixRef wk_conv_proj, wv_conv_proj;
    MatrixRef wffn_router;

    WeightKind gate_up_kind = kWKindIQ4_NL;
    WeightKind down_kind = kWKindIQ4_NL;
    std::size_t gate_up_qs_row_bytes = 0;
    std::size_t gate_up_scales_offset = 0;
    std::size_t down_qs_row_bytes = 0;
    std::size_t down_scales_offset = 0;
    std::array<const unsigned char*, G::n_experts> gate_up_ptrs{};
    std::array<const unsigned char*, G::n_experts> down_ptrs{};
    const unsigned char* const* d_gate_up_ptrs = nullptr;
    const unsigned char* const* d_down_ptrs = nullptr;
};

struct LoopLayerGpu {
    const float* msg_query = nullptr;
    std::array<MatrixRef, G::loop_ctrl_count> ctrl_down{};
    std::array<MatrixRef, G::loop_ctrl_count> ctrl_gain{};
    std::array<MatrixRef, G::loop_ctrl_count> ctrl_forget{};
};

// Branchless register-resident IQ4_NL LUT (zero constant-cache serialization!).
// 16 signed int8 values packed into 4 uint32 immediates:
//   0..3:   -127, -104, -83, -65  -> 0xbfad9881u
//   4..7:    -49,  -35, -22, -10  -> 0xf6eaddcfu
//   8..11:     1,   13,  25,  38  -> 0x26190d01u
//   12..15:   53,   69,  89, 113  -> 0x71594535u
__device__ inline unsigned iq4nl_u8(unsigned idx) {
    const unsigned w0 = (idx & 4u) ? 0xf6eaddcfu : 0xbfad9881u;
    const unsigned w1 = (idx & 4u) ? 0x71594535u : 0x26190d01u;
    const unsigned w = (idx & 8u) ? w1 : w0;
    return (w >> ((idx & 3u) << 3)) & 0xffu;
}

__device__ inline float block_sum_256(float v, float* shared) {
#pragma unroll
    for (unsigned off = 16; off; off >>= 1)
        v += __shfl_down_sync(0xffffffffu, v, off);
    if ((threadIdx.x & 31u) == 0u)
        shared[threadIdx.x >> 5] = v;
    __syncthreads();
    if (threadIdx.x < 32u) {
        float acc = threadIdx.x < 8u ? shared[threadIdx.x] : 0.0f;
#pragma unroll
        for (unsigned off = 4; off; off >>= 1)
            acc += __shfl_down_sync(0xffffffffu, acc, off);
        if (threadIdx.x == 0u)
            shared[0] = acc;
    }
    __syncthreads();
    return shared[0];
}

__device__ inline float block_sum_128(float v, float* shared) {
#pragma unroll
    for (unsigned off = 16; off; off >>= 1)
        v += __shfl_down_sync(0xffffffffu, v, off);
    if ((threadIdx.x & 31u) == 0u)
        shared[threadIdx.x >> 5] = v;
    __syncthreads();
    if (threadIdx.x < 32u) {
        float acc = threadIdx.x < 4u ? shared[threadIdx.x] : 0.0f;
#pragma unroll
        for (unsigned off = 2; off; off >>= 1)
            acc += __shfl_down_sync(0xffffffffu, acc, off);
        if (threadIdx.x == 0u)
            shared[0] = acc;
    }
    __syncthreads();
    return shared[0];
}

// ============================================================================
// Weight SoA repacking, validation & FP16 dequantization kernels
// ============================================================================

// Validates weight finiteness and (for IQ4_NL / Q8_0) repacks AoS blocks into
// 16B/32B-aligned SoA layout: [all contiguous qs] ++ [all contiguous __half scales].
__global__ void inkling_repack_soa_validate_kernel(
    const unsigned char* __restrict__ src,
    unsigned char* __restrict__ dst_qs,
    __half* __restrict__ dst_scales,
    std::size_t count,
    int kind,
    unsigned* __restrict__ status
) {
    const auto i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count)
        return;
    if (kind == kWKindF32) {
        const float v = reinterpret_cast<const float*>(src)[i];
        if (!isfinite(v))
            atomicOr(status, 1U);
    } else if (kind == kWKindIQ4_NL) {
        const unsigned char* blk = src + i * 18;
        const unsigned short bits = *reinterpret_cast<const unsigned short*>(blk);
        const __half h = __ushort_as_half(bits);
        if (!isfinite(__half2float(h)))
            atomicOr(status, 1U);
        dst_scales[i] = h;
        const auto* p = reinterpret_cast<const unsigned short*>(blk + 2);
        reinterpret_cast<uint4*>(dst_qs)[i] = make_uint4(
            unsigned(p[0]) | (unsigned(p[1]) << 16),
            unsigned(p[2]) | (unsigned(p[3]) << 16),
            unsigned(p[4]) | (unsigned(p[5]) << 16),
            unsigned(p[6]) | (unsigned(p[7]) << 16)
        );
    } else {
        // Q8_0: 2B half scale + 32B signed int8 qs -> 32B-aligned qs + contiguous half scales
        const unsigned char* blk = src + i * 34;
        const unsigned short bits = *reinterpret_cast<const unsigned short*>(blk);
        const __half h = __ushort_as_half(bits);
        if (!isfinite(__half2float(h)))
            atomicOr(status, 1U);
        dst_scales[i] = h;
        const auto* p = reinterpret_cast<const unsigned short*>(blk + 2);
        auto* out4 = reinterpret_cast<uint4*>(dst_qs) + i * 2;
        out4[0] = make_uint4(
            unsigned(p[0]) | (unsigned(p[1]) << 16),
            unsigned(p[2]) | (unsigned(p[3]) << 16),
            unsigned(p[4]) | (unsigned(p[5]) << 16),
            unsigned(p[6]) | (unsigned(p[7]) << 16)
        );
        out4[1] = make_uint4(
            unsigned(p[8]) | (unsigned(p[9]) << 16),
            unsigned(p[10]) | (unsigned(p[11]) << 16),
            unsigned(p[12]) | (unsigned(p[13]) << 16),
            unsigned(p[14]) | (unsigned(p[15]) << 16)
        );
    }
}

__global__ void inkling_f32_to_fp16_kernel(
    const float* __restrict__ src, __half* __restrict__ dst, std::size_t count
) {
    const auto i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count)
        dst[i] = __float2half_rn(src[i]);
}

__global__ void inkling_fp16_to_f32_kernel(
    const __half* __restrict__ src, float* __restrict__ dst, std::size_t count
) {
    const auto i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count)
        dst[i] = __half2float(src[i]);
}

// Dequantizes SoA packed weights [out, in] into __half [out, in] for FP16 Tensor Core GEMM.
// One warp (32 lanes) per 32-element quant block -> 8 blocks per 256-thread CTA.
template <int Kind>
__global__ void inkling_dequant_fp16_kernel(
    const unsigned char* __restrict__ qs,
    const __half* __restrict__ scales,
    __half* __restrict__ out,
    std::size_t total_blocks
) {
    const std::size_t blk_idx =
        (std::size_t(blockIdx.x) * (blockDim.x / 32u)) + (threadIdx.x >> 5);
    if (blk_idx >= total_blocks)
        return;
    const unsigned lane = threadIdx.x & 31u;
    if constexpr (Kind == kWKindF32) {
        const auto* f = reinterpret_cast<const float*>(qs) + blk_idx * 32 + lane;
        out[blk_idx * 32 + lane] = __float2half_rn(*f);
    } else if constexpr (Kind == kWKindQ8_0) {
        const float d = __half2float(scales[blk_idx]);
        const signed char q = static_cast<signed char>(qs[blk_idx * 32 + lane]);
        out[blk_idx * 32 + lane] = __float2half_rn(d * float(q));
    } else {
        // IQ4_NL SoA: 16B nibbles (low = 0..15, high = 16..31) + contiguous half scales
        const float d = __half2float(scales[blk_idx]);
        const unsigned byte = qs[blk_idx * 16 + (lane & 15u)];
        const unsigned idx = (lane < 16u) ? (byte & 15u) : (byte >> 4);
        out[blk_idx * 32 + lane] =
            __float2half_rn(d * float(static_cast<signed char>(iq4nl_u8(idx))));
    }
}

// ============================================================================
// Q8 input quantization + __dp4a GEMV & fused decode projection kernels
// ============================================================================

// One warp per 32-element block; 8 blocks per 256-thread CTA.
__global__ void inkling_quantize_q8_kernel(
    const float* __restrict__ src, Q8Block* __restrict__ dst, std::size_t total_blocks
) {
    const std::size_t blk_idx =
        (std::size_t(blockIdx.x) * (blockDim.x / 32u)) + (threadIdx.x >> 5);
    if (blk_idx >= total_blocks)
        return;
    const unsigned lane = threadIdx.x & 31u;
    const float v = src[blk_idx * 32 + lane];
    float amax = fabsf(v);
#pragma unroll
    for (unsigned off = 16; off; off >>= 1)
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, off));
    const float d = amax * (1.0f / 127.0f);
    const float inv_d = amax > 0.0f ? (127.0f / amax) : 0.0f;
    const int qi = max(-127, min(127, __float2int_rn(v * inv_d)));
    const unsigned byte_val = static_cast<unsigned>(qi) & 0xffu;
    // Pack 4 consecutive lanes into one int32 in lane (4*k)
    const unsigned b1 = __shfl_down_sync(0xffffffffu, byte_val, 1);
    const unsigned b2 = __shfl_down_sync(0xffffffffu, byte_val, 2);
    const unsigned b3 = __shfl_down_sync(0xffffffffu, byte_val, 3);
    if (lane == 0u)
        dst[blk_idx].d = d;
    if ((lane & 3u) == 0u) {
        const std::int32_t packed = static_cast<std::int32_t>(
            byte_val | (b1 << 8) | (b2 << 16) | (b3 << 24)
        );
        dst[blk_idx].qs[lane >> 2] = packed;
    }
}

// 4-lane-per-block coalesced SoA IQ4_NL quarter-block loader:
// Single 32-bit aligned LDG.32 per lane -> 16 contiguous bytes across 4 lanes!
__device__ inline float load_iq4nl_soa_quarter(
    const unsigned char* qs_row,
    const __half* sc_row,
    unsigned b,
    unsigned sub,
    int32_t& w_lo,
    int32_t& w_hi
) {
    const float d = __half2float(sc_row[b]);
    const unsigned p = reinterpret_cast<const unsigned*>(qs_row)[b * 4u + sub];
    const unsigned b0 = p & 0xffu;
    const unsigned b1 = (p >> 8) & 0xffu;
    const unsigned b2 = (p >> 16) & 0xffu;
    const unsigned b3 = p >> 24;
    w_lo = static_cast<int32_t>(
        iq4nl_u8(b0 & 15u) | (iq4nl_u8(b1 & 15u) << 8)
        | (iq4nl_u8(b2 & 15u) << 16) | (iq4nl_u8(b3 & 15u) << 24)
    );
    w_hi = static_cast<int32_t>(
        iq4nl_u8(b0 >> 4) | (iq4nl_u8(b1 >> 4) << 8)
        | (iq4nl_u8(b2 >> 4) << 16) | (iq4nl_u8(b3 >> 4) << 24)
    );
    return d;
}

// 4-lane-per-block coalesced SoA Q8_0 quarter-block loader:
// Single 64-bit aligned LDG.64 per lane -> 32 contiguous bytes across 4 lanes!
__device__ inline float load_q8_0_soa_quarter(
    const unsigned char* qs_row,
    const __half* sc_row,
    unsigned b,
    unsigned sub,
    int32_t& w0,
    int32_t& w1
) {
    const float d = __half2float(sc_row[b]);
    const uint2 p = reinterpret_cast<const uint2*>(qs_row)[b * 4u + sub];
    w0 = static_cast<int32_t>(p.x);
    w1 = static_cast<int32_t>(p.y);
    return d;
}

// Cooperatively loads `n_blocks` Q8Blocks into shared memory (9 int32/block).
__device__ inline void load_q8_shared(
    const Q8Block* __restrict__ src, Q8Block* dst, unsigned n_blocks
) {
    const auto* s = reinterpret_cast<const std::int32_t*>(src);
    auto* d = reinterpret_cast<std::int32_t*>(dst);
    const unsigned total_words = n_blocks * (sizeof(Q8Block) / sizeof(std::int32_t));
    for (unsigned i = threadIdx.x; i < total_words; i += blockDim.x)
        d[i] = s[i];
    __syncthreads();
}

// Single-row or small-multi-row (1..15) matvec on aligned SoA weights:
// 1 warp per output column, 4 warps/CTA (128 threads -> 2x SM occupancy).
template <int Kind, unsigned MaxRows>
__global__ void inkling_matvec_kernel(
    const unsigned char* __restrict__ qs,
    const __half* __restrict__ scales,
    const Q8Block* __restrict__ x_q8,
    const float* __restrict__ x_f32,
    float* __restrict__ out,
    std::size_t qs_row_bytes,
    unsigned in_dim,
    unsigned out_dim,
    unsigned rows
) {
    __shared__ Q8Block s_x[MaxRows == 1 && Kind != kWKindF32 ? 144 : 1];
    const unsigned n_blocks = in_dim >> 5;
    if constexpr (MaxRows == 1 && Kind != kWKindF32) {
        if (n_blocks <= 144u)
            load_q8_shared(x_q8, s_x, n_blocks);
    }

    const unsigned col = blockIdx.x * 4u + (threadIdx.x >> 5);
    if (col >= out_dim)
        return;
    const unsigned lane = threadIdx.x & 31u;
    const unsigned char* wrow = qs + std::size_t(col) * qs_row_bytes;
    float sums[MaxRows];
#pragma unroll
    for (unsigned r = 0; r < MaxRows; ++r)
        sums[r] = 0.0f;

    if constexpr (Kind == kWKindF32) {
        const auto* wf = reinterpret_cast<const float4*>(wrow);
        const unsigned n_vec = in_dim >> 2;
        for (unsigned v = lane; v < n_vec; v += 32u) {
            const float4 w = wf[v];
#pragma unroll
            for (unsigned r = 0; r < MaxRows; ++r) {
                if (MaxRows == 1 || r < rows) {
                    const float4 x = reinterpret_cast<const float4*>(
                        x_f32 + std::size_t(r) * in_dim
                    )[v];
                    float s = sums[r];
                    s = fmaf(w.x, x.x, s);
                    s = fmaf(w.y, x.y, s);
                    s = fmaf(w.z, x.z, s);
                    s = fmaf(w.w, x.w, s);
                    sums[r] = s;
                }
            }
        }
    } else {
        const unsigned grp = lane >> 2;  // 0..7
        const unsigned sub = lane & 3u;  // 0..3
        const __half* sc_row = scales + std::size_t(col) * n_blocks;
        const Q8Block* x_src =
            (MaxRows == 1 && n_blocks <= 144u) ? static_cast<const Q8Block*>(s_x) : x_q8;
        for (unsigned b = grp; b < n_blocks; b += 8u) {
            int32_t w0, w1;
            const float wd = (Kind == kWKindIQ4_NL)
                                 ? load_iq4nl_soa_quarter(wrow, sc_row, b, sub, w0, w1)
                                 : load_q8_0_soa_quarter(wrow, sc_row, b, sub, w0, w1);
#pragma unroll
            for (unsigned r = 0; r < MaxRows; ++r) {
                if (MaxRows == 1 || r < rows) {
                    const Q8Block& xb = x_src[std::size_t(r) * n_blocks + b];
                    int32_t acc;
                    if constexpr (Kind == kWKindIQ4_NL) {
                        acc = __dp4a(w0, xb.qs[sub], 0);
                        acc = __dp4a(w1, xb.qs[sub + 4u], acc);
                    } else {
                        acc = __dp4a(w0, xb.qs[sub * 2u], 0);
                        acc = __dp4a(w1, xb.qs[sub * 2u + 1u], acc);
                    }
                    sums[r] = fmaf(wd * xb.d, float(acc), sums[r]);
                }
            }
        }
    }

#pragma unroll
    for (unsigned r = 0; r < MaxRows; ++r) {
        if (MaxRows > 1 && r >= rows)
            break;
        float s = sums[r];
#pragma unroll
        for (unsigned off = 16; off; off >>= 1)
            s += __shfl_down_sync(0xffffffffu, s, off);
        if (lane == 0u)
            out[std::size_t(r) * out_dim + col] = s;
    }
}

struct DecodeProjEntry {
    const unsigned char* qs;
    const __half* scales;
    float* out;
    std::size_t qs_row_bytes;
    unsigned out_dim;
    unsigned col_offset;
    int kind;
};

struct FusedDecodeProjArgs {
    DecodeProjEntry entries[6];
    const Q8Block* x_q8;
    const float* x_f32;
    unsigned count;
    unsigned in_dim;
    unsigned total_cols;
};

// Fused multi-matrix single-row decode projection megakernel:
// Projects up to 6 matrices sharing the same input vector in one kernel launch,
// loading x_q8 once into shared memory with zero warp divergence (1 warp = 1 column).
__global__ void inkling_fused_decode_proj_kernel(FusedDecodeProjArgs args) {
    __shared__ Q8Block s_x[144];
    const unsigned n_blocks = args.in_dim >> 5;
    if (args.x_q8 && n_blocks <= 144u)
        load_q8_shared(args.x_q8, s_x, n_blocks);

    const unsigned global_col = blockIdx.x * 4u + (threadIdx.x >> 5);
    if (global_col >= args.total_cols)
        return;

    unsigned idx = 0u;
    for (unsigned i = 1u; i < args.count; ++i) {
        if (global_col >= args.entries[i].col_offset)
            idx = i;
    }
    const DecodeProjEntry& e = args.entries[idx];
    const unsigned col = global_col - e.col_offset;
    const unsigned lane = threadIdx.x & 31u;
    const unsigned char* wrow = e.qs + std::size_t(col) * e.qs_row_bytes;
    float sum = 0.0f;

    if (e.kind == kWKindF32) {
        const auto* wf = reinterpret_cast<const float4*>(wrow);
        const auto* xf = reinterpret_cast<const float4*>(args.x_f32);
        const unsigned n_vec = args.in_dim >> 2;
        for (unsigned v = lane; v < n_vec; v += 32u) {
            const float4 w = wf[v], x = xf[v];
            sum = fmaf(w.x, x.x, sum);
            sum = fmaf(w.y, x.y, sum);
            sum = fmaf(w.z, x.z, sum);
            sum = fmaf(w.w, x.w, sum);
        }
    } else if (e.kind == kWKindIQ4_NL) {
        const unsigned grp = lane >> 2;
        const unsigned sub = lane & 3u;
        const __half* sc_row = e.scales + std::size_t(col) * n_blocks;
        const Q8Block* x_src = (n_blocks <= 144u) ? static_cast<const Q8Block*>(s_x) : args.x_q8;
        for (unsigned b = grp; b < n_blocks; b += 8u) {
            int32_t w0, w1;
            const float wd = load_iq4nl_soa_quarter(wrow, sc_row, b, sub, w0, w1);
            const Q8Block& xb = x_src[b];
            int32_t acc = __dp4a(w0, xb.qs[sub], 0);
            acc = __dp4a(w1, xb.qs[sub + 4u], acc);
            sum = fmaf(wd * xb.d, float(acc), sum);
        }
    } else {
        const unsigned grp = lane >> 2;
        const unsigned sub = lane & 3u;
        const __half* sc_row = e.scales + std::size_t(col) * n_blocks;
        const Q8Block* x_src = (n_blocks <= 144u) ? static_cast<const Q8Block*>(s_x) : args.x_q8;
        for (unsigned b = grp; b < n_blocks; b += 8u) {
            int32_t w0, w1;
            const float wd = load_q8_0_soa_quarter(wrow, sc_row, b, sub, w0, w1);
            const Q8Block& xb = x_src[b];
            int32_t acc = __dp4a(w0, xb.qs[sub * 2u], 0);
            acc = __dp4a(w1, xb.qs[sub * 2u + 1u], acc);
            sum = fmaf(wd * xb.d, float(acc), sum);
        }
    }

#pragma unroll
    for (unsigned off = 16; off; off >>= 1)
        sum += __shfl_down_sync(0xffffffffu, sum, off);
    if (lane == 0u)
        e.out[col] = sum;
}

// Fused final output head (200058 x 2304, chunked across <=16 MiB SoA slices)
// + softcap 30 * tanh(x / 30), 4 warps/CTA (128 threads), 4 lanes/block + shared x_q8!
template <int Kind>
__global__ void inkling_output_softcap_kernel(
    const OutputChunkDesc* __restrict__ chunks,
    unsigned n_chunks,
    const Q8Block* __restrict__ x_q8,
    const float* __restrict__ x_f32,
    float* __restrict__ logits,
    std::size_t qs_row_bytes,
    unsigned out_dim
) {
    constexpr unsigned n_blocks = G::n_embd >> 5;  // 72
    __shared__ Q8Block s_x[Kind != kWKindF32 ? n_blocks : 1];
    if constexpr (Kind != kWKindF32)
        load_q8_shared(x_q8, s_x, n_blocks);

    const unsigned col = blockIdx.x * 4u + (threadIdx.x >> 5);
    if (col >= out_dim)
        return;
    const unsigned lane = threadIdx.x & 31u;
    unsigned lo = 0, hi = n_chunks;
    while (hi - lo > 1u) {
        const unsigned mid = (lo + hi) >> 1;
        if (chunks[mid].first_row <= col)
            lo = mid;
        else
            hi = mid;
    }
    const OutputChunkDesc ch = chunks[lo];
    const unsigned local_row = col - ch.first_row;
    const unsigned char* wrow = ch.qs + std::size_t(local_row) * qs_row_bytes;
    float sum = 0.0f;
    if constexpr (Kind == kWKindF32) {
        const auto* wf = reinterpret_cast<const float4*>(wrow);
        const auto* xf = reinterpret_cast<const float4*>(x_f32);
        for (unsigned v = lane; v < (G::n_embd >> 2); v += 32u) {
            const float4 w = wf[v], x = xf[v];
            sum = fmaf(w.x, x.x, sum);
            sum = fmaf(w.y, x.y, sum);
            sum = fmaf(w.z, x.z, sum);
            sum = fmaf(w.w, x.w, sum);
        }
    } else {
        const unsigned grp = lane >> 2;
        const unsigned sub = lane & 3u;
        const __half* sc_row = ch.scales + std::size_t(local_row) * n_blocks;
        for (unsigned b = grp; b < n_blocks; b += 8u) {
            int32_t w0, w1;
            const float wd = (Kind == kWKindIQ4_NL)
                                 ? load_iq4nl_soa_quarter(wrow, sc_row, b, sub, w0, w1)
                                 : load_q8_0_soa_quarter(wrow, sc_row, b, sub, w0, w1);
            const Q8Block& xb = s_x[b];
            int32_t acc;
            if constexpr (Kind == kWKindIQ4_NL) {
                acc = __dp4a(w0, xb.qs[sub], 0);
                acc = __dp4a(w1, xb.qs[sub + 4u], acc);
            } else {
                acc = __dp4a(w0, xb.qs[sub * 2u], 0);
                acc = __dp4a(w1, xb.qs[sub * 2u + 1u], acc);
            }
            sum = fmaf(wd * xb.d, float(acc), sum);
        }
    }
#pragma unroll
    for (unsigned off = 16; off; off >>= 1)
        sum += __shfl_down_sync(0xffffffffu, sum, off);
    if (lane == 0u)
        logits[col] = kFinalSoftcap * tanhf(sum * (1.0f / kFinalSoftcap));
}

// 2-stage GPU argmax + finiteness check over logits[0..n-1] (lowest-index tie break).
// Eliminates 800 KB D2H copy + 2 host scans per token on greedy decode.
__device__ inline void argmax_merge(
    float& bv, unsigned& bi, unsigned& bad, float ov, unsigned oi, unsigned ob
) {
    bad |= ob;
    if (oi != 0xffffffffu && (bi == 0xffffffffu || ov > bv || (ov == bv && oi < bi))) {
        bv = ov;
        bi = oi;
    }
}

__global__ void inkling_argmax_stage1_kernel(
    const float* __restrict__ logits,
    unsigned n,
    float* __restrict__ p_val,
    unsigned* __restrict__ p_idx,
    unsigned* __restrict__ p_bad
) {
    __shared__ float sv[8];
    __shared__ unsigned si[8];
    __shared__ unsigned sb[8];
    float bv = -INFINITY;
    unsigned bi = 0xffffffffu;
    unsigned bad = 0u;
    const unsigned stride = gridDim.x * blockDim.x;
    for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        const float v = logits[i];
        if (!isfinite(v)) {
            bad = 1u;
        } else if (bi == 0xffffffffu || v > bv || (v == bv && i < bi)) {
            bv = v;
            bi = i;
        }
    }
#pragma unroll
    for (unsigned off = 16; off; off >>= 1) {
        const float ov = __shfl_down_sync(0xffffffffu, bv, off);
        const unsigned oi = __shfl_down_sync(0xffffffffu, bi, off);
        const unsigned ob = __shfl_down_sync(0xffffffffu, bad, off);
        argmax_merge(bv, bi, bad, ov, oi, ob);
    }
    if ((threadIdx.x & 31u) == 0u) {
        const unsigned w = threadIdx.x >> 5;
        sv[w] = bv;
        si[w] = bi;
        sb[w] = bad;
    }
    __syncthreads();
    if (threadIdx.x < 32u) {
        bv = threadIdx.x < 8u ? sv[threadIdx.x] : -INFINITY;
        bi = threadIdx.x < 8u ? si[threadIdx.x] : 0xffffffffu;
        bad = threadIdx.x < 8u ? sb[threadIdx.x] : 0u;
#pragma unroll
        for (unsigned off = 4; off; off >>= 1) {
            const float ov = __shfl_down_sync(0xffffffffu, bv, off);
            const unsigned oi = __shfl_down_sync(0xffffffffu, bi, off);
            const unsigned ob = __shfl_down_sync(0xffffffffu, bad, off);
            argmax_merge(bv, bi, bad, ov, oi, ob);
        }
        if (threadIdx.x == 0u) {
            p_val[blockIdx.x] = bv;
            p_idx[blockIdx.x] = bi;
            p_bad[blockIdx.x] = bad;
        }
    }
}

__global__ void inkling_argmax_stage2_kernel(
    const float* __restrict__ p_val,
    const unsigned* __restrict__ p_idx,
    const unsigned* __restrict__ p_bad,
    ArgmaxResult* __restrict__ out
) {
    __shared__ float sv[8];
    __shared__ unsigned si[8];
    __shared__ unsigned sb[8];
    float bv = p_val[threadIdx.x];
    unsigned bi = p_idx[threadIdx.x];
    unsigned bad = p_bad[threadIdx.x];
#pragma unroll
    for (unsigned off = 16; off; off >>= 1) {
        const float ov = __shfl_down_sync(0xffffffffu, bv, off);
        const unsigned oi = __shfl_down_sync(0xffffffffu, bi, off);
        const unsigned ob = __shfl_down_sync(0xffffffffu, bad, off);
        argmax_merge(bv, bi, bad, ov, oi, ob);
    }
    if ((threadIdx.x & 31u) == 0u) {
        const unsigned w = threadIdx.x >> 5;
        sv[w] = bv;
        si[w] = bi;
        sb[w] = bad;
    }
    __syncthreads();
    if (threadIdx.x < 32u) {
        bv = threadIdx.x < 8u ? sv[threadIdx.x] : -INFINITY;
        bi = threadIdx.x < 8u ? si[threadIdx.x] : 0xffffffffu;
        bad = threadIdx.x < 8u ? sb[threadIdx.x] : 0u;
#pragma unroll
        for (unsigned off = 4; off; off >>= 1) {
            const float ov = __shfl_down_sync(0xffffffffu, bv, off);
            const unsigned oi = __shfl_down_sync(0xffffffffu, bi, off);
            const unsigned ob = __shfl_down_sync(0xffffffffu, bad, off);
            argmax_merge(bv, bi, bad, ov, oi, ob);
        }
        if (threadIdx.x == 0u) {
            out->token = (bi == 0xffffffffu) ? 0u : bi;
            out->nonfinite = bad;
        }
    }
}

// ============================================================================
// Fused Inkling elementwise, residual-mixing, control, conv, and attention kernels
// ============================================================================

__global__ void inkling_rms_norm_2304_kernel(
    const float* __restrict__ src,
    const float* __restrict__ weight,
    float* __restrict__ dst
) {
    __shared__ float s_red[8];
    const unsigned row = blockIdx.x;
    const unsigned t = threadIdx.x;
    const float* x = src + std::size_t(row) * G::n_embd;
    float* y = dst + std::size_t(row) * G::n_embd;
    float reg[9];
    float sq = 0.0f;
#pragma unroll
    for (unsigned j = 0; j < 9; ++j) {
        const float v = x[t + j * 256u];
        reg[j] = v;
        sq = fmaf(v, v, sq);
    }
    const float inv = rsqrtf(block_sum_256(sq, s_red) * (1.0f / 2304.0f) + kRmsEps);
#pragma unroll
    for (unsigned j = 0; j < 9; ++j) {
        const unsigned idx = t + j * 256u;
        y[idx] = reg[j] * inv * weight[idx];
    }
}

struct MixStatesArgs {
    const float* states[5];
    float* outputs[4];
    const float* norm_w[4];
    const float* query;
    unsigned n_states;
    unsigned n_queries;
    unsigned row_offset;  // non-zero for final output head (last row only)
};

// Fused block-attention residual mixing + optional per-query RMSNorm.
// Grid: rows blocks, Block: 256 threads (9 elements/thread = 2304 exact).
__global__ void inkling_mix_states_norm_kernel(MixStatesArgs args) {
    __shared__ float s_red[8];
    const unsigned row = blockIdx.x;
    const unsigned src_row = args.row_offset + row;
    const unsigned t = threadIdx.x;

    float ns[5][9];
    for (unsigned s = 0; s < args.n_states; ++s) {
        const float* src = args.states[s] + std::size_t(src_row) * G::n_embd;
        float sq = 0.0f;
#pragma unroll
        for (unsigned j = 0; j < 9; ++j) {
            const float v = src[t + j * 256u];
            ns[s][j] = v;
            sq = fmaf(v, v, sq);
        }
        const float inv = rsqrtf(block_sum_256(sq, s_red) * (1.0f / 2304.0f) + kRmsEps);
#pragma unroll
        for (unsigned j = 0; j < 9; ++j)
            ns[s][j] *= inv;
    }

    for (unsigned q = 0; q < args.n_queries; ++q) {
        const float* qv = args.query + std::size_t(q) * G::n_embd;
        float qr[9];
#pragma unroll
        for (unsigned j = 0; j < 9; ++j)
            qr[j] = qv[t + j * 256u];

        float sc[5];
        float max_sc = -INFINITY;
        for (unsigned s = 0; s < args.n_states; ++s) {
            float dot = 0.0f;
#pragma unroll
            for (unsigned j = 0; j < 9; ++j)
                dot = fmaf(ns[s][j], qr[j], dot);
            // 1 / sqrt(2304) == 1 / 48 exact
            sc[s] = block_sum_256(dot, s_red) * (1.0f / 48.0f);
            max_sc = fmaxf(max_sc, sc[s]);
        }
        float denom = 0.0f;
        for (unsigned s = 0; s < args.n_states; ++s) {
            sc[s] = expf(sc[s] - max_sc);
            denom += sc[s];
        }
        const float inv_d = 1.0f / denom;
        float outv[9];
        const float w0 = sc[0] * inv_d;
#pragma unroll
        for (unsigned j = 0; j < 9; ++j)
            outv[j] = w0 * ns[0][j];
        for (unsigned s = 1; s < args.n_states; ++s) {
            const float w = sc[s] * inv_d;
#pragma unroll
            for (unsigned j = 0; j < 9; ++j)
                outv[j] = fmaf(w, ns[s][j], outv[j]);
        }
        const float* nw = args.norm_w[q];
        if (nw) {
            float osq = 0.0f;
#pragma unroll
            for (unsigned j = 0; j < 9; ++j)
                osq = fmaf(outv[j], outv[j], osq);
            const float oinv =
                rsqrtf(block_sum_256(osq, s_red) * (1.0f / 2304.0f) + kRmsEps);
#pragma unroll
            for (unsigned j = 0; j < 9; ++j) {
                const unsigned idx = t + j * 256u;
                outv[j] = outv[j] * oinv * nw[idx];
            }
        }
        float* dst = args.outputs[q] + std::size_t(row) * G::n_embd;
#pragma unroll
        for (unsigned j = 0; j < 9; ++j)
            dst[t + j * 256u] = outv[j];
    }
}

// Packs unweighted RMSNorm(target) || unweighted RMSNorm(message) into [rows, 4608].
__global__ void inkling_ctrl_pack_kernel(
    const float* __restrict__ target,
    const float* __restrict__ msg,
    float* __restrict__ cat
) {
    __shared__ float s_red[8];
    const unsigned row = blockIdx.x;
    const unsigned t = threadIdx.x;
    const float* a = target + std::size_t(row) * G::n_embd;
    const float* b = msg + std::size_t(row) * G::n_embd;
    float* dst_a = cat + std::size_t(row) * (2 * G::n_embd);
    float* dst_b = dst_a + G::n_embd;

    float ra[9], rb[9];
    float sqa = 0.0f, sqb = 0.0f;
#pragma unroll
    for (unsigned j = 0; j < 9; ++j) {
        const unsigned idx = t + j * 256u;
        ra[j] = a[idx];
        rb[j] = b[idx];
        sqa = fmaf(ra[j], ra[j], sqa);
        sqb = fmaf(rb[j], rb[j], sqb);
    }
    const float inv_a = rsqrtf(block_sum_256(sqa, s_red) * (1.0f / 2304.0f) + kRmsEps);
    const float inv_b = rsqrtf(block_sum_256(sqb, s_red) * (1.0f / 2304.0f) + kRmsEps);
#pragma unroll
    for (unsigned j = 0; j < 9; ++j) {
        const unsigned idx = t + j * 256u;
        dst_a[idx] = ra[j] * inv_a;
        dst_b[idx] = rb[j] * inv_b;
    }
}

// Fused control blend + RMSNorm:
//   blended = (1 - tanh(f)) * src + tanh(g) * msg
//   dst = rms_norm(blended, norm_w)
__global__ void inkling_ctrl_blend_norm_kernel(
    const float* __restrict__ src,
    const float* __restrict__ msg,
    const float* __restrict__ g,
    const float* __restrict__ f,
    const float* __restrict__ norm_w,
    float* __restrict__ dst
) {
    __shared__ float s_red[8];
    const unsigned row = blockIdx.x;
    const unsigned t = threadIdx.x;
    const std::size_t base = std::size_t(row) * G::n_embd;
    float v[9];
    float sq = 0.0f;
#pragma unroll
    for (unsigned j = 0; j < 9; ++j) {
        const unsigned idx = t + j * 256u;
        const float gain = tanhf(g[base + idx]);
        const float forget = tanhf(f[base + idx]);
        const float blended = (1.0f - forget) * src[base + idx] + gain * msg[base + idx];
        v[j] = blended;
        sq = fmaf(blended, blended, sq);
    }
    const float inv = rsqrtf(block_sum_256(sq, s_red) * (1.0f / 2304.0f) + kRmsEps);
#pragma unroll
    for (unsigned j = 0; j < 9; ++j) {
        const unsigned idx = t + j * 256u;
        dst[base + idx] = v[j] * inv * norm_w[idx];
    }
}

// Single-row causal short convolution (+ optional residual add) + in-place history update.
__global__ void inkling_shortconv_single_kernel(
    const float* __restrict__ cur,
    const float* __restrict__ w0,
    const float* __restrict__ w1,
    const float* __restrict__ w2,
    const float* __restrict__ w3,
    const float* __restrict__ dynamic_taps,  // null for static; [8, 4] for full-attn K/V
    const float* __restrict__ residual,      // optional residual add on store
    float* __restrict__ conv_state,          // [channels, 3]
    float* __restrict__ out,
    unsigned channels,
    unsigned head_width,
    float scale
) {
    const unsigned c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= channels)
        return;
    float* hist = conv_state + std::size_t(c) * 3;
    const float h0 = hist[0], h1 = hist[1], h2 = hist[2];
    const float x = cur[c];
    float t0 = 0.0f, t1 = 0.0f, t2 = 0.0f, t3 = 0.0f;
    if (dynamic_taps) {
        const unsigned dh = c / head_width;
        const float* tp = dynamic_taps + dh * 4u;
        t0 = tp[0];
        t1 = tp[1];
        t2 = tp[2];
        t3 = tp[3];
    }
    const float acc =
        h0 * (w0[c] + t3) + h1 * (w1[c] + t2) + h2 * (w2[c] + t1) + x * (w3[c] + t0);
    const float y = (x + acc) * scale;
    out[c] = residual ? (residual[c] + y) : y;
    hist[0] = h1;
    hist[1] = h2;
    hist[2] = x;
}

// Multi-row causal short convolution (read-only conv_state) + optional residual add.
__global__ void inkling_shortconv_multi_kernel(
    const float* __restrict__ cur,
    const float* __restrict__ w0,
    const float* __restrict__ w1,
    const float* __restrict__ w2,
    const float* __restrict__ w3,
    const float* __restrict__ dynamic_taps,
    const float* __restrict__ residual,
    const float* __restrict__ conv_state,
    float* __restrict__ out,
    unsigned rows,
    unsigned channels,
    unsigned dyn_heads,
    unsigned head_width,
    float scale
) {
    const unsigned c = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned r = blockIdx.y;
    if (c >= channels || r >= rows)
        return;
    const float* hist = conv_state + std::size_t(c) * 3;
    const float x0 = r >= 3u ? cur[std::size_t(r - 3u) * channels + c] : hist[r];
    const float x1 = r >= 2u ? cur[std::size_t(r - 2u) * channels + c] : hist[r + 1u];
    const float x2 = r >= 1u ? cur[std::size_t(r - 1u) * channels + c] : hist[r + 2u];
    const float x3 = cur[std::size_t(r) * channels + c];
    float t0 = 0.0f, t1 = 0.0f, t2 = 0.0f, t3 = 0.0f;
    if (dynamic_taps) {
        const unsigned dh = c / head_width;
        const float* tp = dynamic_taps + (std::size_t(r) * dyn_heads + dh) * 4u;
        t0 = tp[0];
        t1 = tp[1];
        t2 = tp[2];
        t3 = tp[3];
    }
    const float acc =
        x0 * (w0[c] + t3) + x1 * (w1[c] + t2) + x2 * (w2[c] + t1) + x3 * (w3[c] + t0);
    const float y = (x3 + acc) * scale;
    const std::size_t off = std::size_t(r) * channels + c;
    out[off] = residual ? (residual[off] + y) : y;
}

__global__ void inkling_shortconv_commit_kernel(
    const float* __restrict__ cur,
    float* __restrict__ conv_state,
    unsigned rows,
    unsigned channels
) {
    const unsigned c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= channels)
        return;
    float* hist = conv_state + std::size_t(c) * 3;
    const float h1 = hist[1], h2 = hist[2];
    const float n0 =
        rows >= 3u ? cur[std::size_t(rows - 3u) * channels + c] : (rows == 2u ? h2 : h1);
    const float n1 =
        rows >= 2u ? cur[std::size_t(rows - 2u) * channels + c] : h2;
    const float n2 = cur[std::size_t(rows - 1u) * channels + c];
    hist[0] = n0;
    hist[1] = n1;
    hist[2] = n2;
}

// Fused Q/R/K clip + V head RMSNorm + full-attn tau + FP16 KV mirror store.
// Grid: dim3(rows, 64), Block: 128 threads.
//   blockIdx.y in [0..23]:  Q head clip (+ *tau if !is_swa)
//   blockIdx.y in [24..47]: R head clip (+ *tau if !is_swa)
//   blockIdx.y in [48..55]: K head clip + store FP16 to kv_keys
//   blockIdx.y in [56..63]: V head RMSNorm + store FP16 to kv_vals
__global__ void inkling_qkvr_prep_kv_store_kernel(
    float* __restrict__ q,
    float* __restrict__ r_cur,
    const float* __restrict__ k,
    const float* __restrict__ v,
    const float* __restrict__ q_clip,
    const float* __restrict__ rel_clip,
    __half* __restrict__ kv_keys,
    __half* __restrict__ kv_vals,
    const StepParams* __restrict__ step_params,
    unsigned base_position,
    unsigned ring_cap,
    unsigned ring_mask,
    bool is_swa
) {
    __shared__ float s_red[4];
    const unsigned row = blockIdx.x;
    const unsigned task = blockIdx.y;
    const unsigned d = threadIdx.x;  // 0..127
    const unsigned pos = (step_params ? step_params->position : base_position) + row;
    const float tau =
        is_swa ? 1.0f
               : (1.0f + kLogScaleAlpha * logf(fmaxf(1.0f, float(pos + 1u) * (1.0f / kLogScaleFloor))));

    if (task < 24u) {
        const unsigned h = task;
        float* ptr = q + (std::size_t(row) * G::n_heads + h) * G::head_dim;
        float val = ptr[d] * q_clip[h * G::head_dim + d];
        const float sum_sq = block_sum_128(val * val, s_red);
        const float norm = sqrtf(sum_sq);
        const float factor = (norm > kQkClip ? (kQkClip / norm) : 1.0f) * tau;
        ptr[d] = val * factor;
    } else if (task < 48u) {
        if (d < G::n_rel) {
            const unsigned h = task - 24u;
            float* ptr = r_cur + (std::size_t(row) * G::n_heads + h) * G::n_rel;
            float val = ptr[d] * rel_clip[h * G::n_rel + d];
            float sq = val * val;
#pragma unroll
            for (unsigned off = 8; off; off >>= 1)
                sq += __shfl_xor_sync(0x0000ffffu, sq, off);
            const float norm = sqrtf(sq);
            const float factor = (norm > kRelClip ? (kRelClip / norm) : 1.0f) * tau;
            ptr[d] = val * factor;
        }
    } else if (task < 56u) {
        const unsigned kh = task - 48u;
        const float* src = k + (std::size_t(row) * G::n_kv_heads + kh) * G::head_dim;
        const float val = src[d];
        const float sum_sq = block_sum_128(val * val, s_red);
        const float norm = sqrtf(sum_sq);
        const float factor = norm > kQkClip ? (kQkClip / norm) : 1.0f;
        const unsigned slot_k =
            ring_mask ? (pos & ring_mask) : (ring_cap ? (pos % ring_cap) : pos);
        kv_keys[(std::size_t(slot_k) * G::n_kv_heads + kh) * G::head_dim + d] =
            __float2half_rn(val * factor);
    } else {
        const unsigned kh = task - 56u;
        const float* src = v + (std::size_t(row) * G::n_kv_heads + kh) * G::head_dim;
        const float val = src[d];
        const float sum_sq = block_sum_128(val * val, s_red);
        const float inv = rsqrtf(sum_sq * (1.0f / float(G::head_dim)) + kRmsEps);
        const unsigned slot_k =
            ring_mask ? (pos & ring_mask) : (ring_cap ? (pos % ring_cap) : pos);
        kv_vals[(std::size_t(slot_k) * G::n_kv_heads + kh) * G::head_dim + d] =
            __float2half_rn(val * inv);
    }
}

// FP16 KV Inkling attention stage 1 (supports dynamic device `step_params` for CUDA Graph).
__global__ void inkling_res_attention_kernel(
    const float* __restrict__ q,
    const float* __restrict__ rel_q,
    const float* __restrict__ rel_bias,
    const __half* __restrict__ keys,
    const __half* __restrict__ values,
    float* __restrict__ partial,
    const StepParams* __restrict__ step_params,
    unsigned base_position,
    unsigned window,
    unsigned ring_cap,
    unsigned ring_mask,
    float scale
) {
    constexpr unsigned threads = 256u;
    constexpr unsigned warps = threads / 32u;
    constexpr unsigned seg_len = 256u;
    constexpr unsigned dim = G::head_dim;
    __shared__ float query[dim], rel_query[G::n_rel], scores[warps], s_acc[dim];
    const unsigned row = blockIdx.x, head = blockIdx.y, segment = blockIdx.z;
    const unsigned position = step_params ? step_params->position : base_position;
    const unsigned pos_q = position + row;
    const unsigned pos_k_min =
        (window && pos_q + 1u > window) ? (pos_q + 1u - window) : 0u;
    const unsigned visible = pos_q - pos_k_min + 1u;
    const unsigned begin = segment * seg_len;
    float* out =
        partial + ((std::size_t(row) * G::n_heads + head) * gridDim.z + segment) * (dim + 2u);
    if (begin >= visible) {
        if (threadIdx.x == 0u) {
            out[0] = -INFINITY;
            out[1] = 0.0f;
        }
        return;
    }
    const unsigned end = min(visible, begin + seg_len);
    const unsigned lane = threadIdx.x & 31u, warp = threadIdx.x >> 5;
    const unsigned val_half = threadIdx.x >> 7;  // 0 (threads 0..127) or 1 (128..255)
    const unsigned val_d = threadIdx.x & 127u;
    const unsigned kv = head / (G::n_heads / G::n_kv_heads);
    const std::size_t width = G::n_embd_k_gqa;
    if (threadIdx.x < dim)
        query[threadIdx.x] = q[(std::size_t(row) * G::n_heads + head) * dim + threadIdx.x];
    if (threadIdx.x < G::n_rel)
        rel_query[threadIdx.x] =
            rel_q[(std::size_t(row) * G::n_heads + head) * G::n_rel + threadIdx.x];
    __syncthreads();

    float maximum = -INFINITY, total = 0.0f, acc = 0.0f;
    for (unsigned first = begin; first < end; first += warps) {
        const unsigned t = first + warp;
        if (t < end) {
            const unsigned pos_k = pos_k_min + t;
            const unsigned slot_k =
                ring_mask ? (pos_k & ring_mask) : (ring_cap ? (pos_k % ring_cap) : pos_k);
            const __half* key =
                keys + std::size_t(slot_k) * width + std::size_t(kv) * dim;
            float dot = 0.0f;
#pragma unroll
            for (unsigned d = lane; d < dim; d += 32u)
                dot = fmaf(query[d], __half2float(key[d]), dot);
#pragma unroll
            for (unsigned off = 16; off; off >>= 1)
                dot += __shfl_down_sync(0xffffffffu, dot, off);
            const unsigned dist = pos_q - pos_k;
            float rel_dot = 0.0f;
            if (dist < G::n_rel_extent) {
                rel_dot = lane < G::n_rel
                              ? rel_query[lane]
                                    * rel_bias[std::size_t(dist) * G::n_rel + lane]
                              : 0.0f;
#pragma unroll
                for (unsigned off = 8; off; off >>= 1)
                    rel_dot += __shfl_down_sync(0xffffffffu, rel_dot, off);
            }
            if (lane == 0u)
                scores[warp] = fmaf(dot, scale, rel_dot);
        } else if (lane == 0u) {
            scores[warp] = -INFINITY;
        }
        __syncthreads();
        float tile = maximum;
#pragma unroll
        for (unsigned w = 0; w < warps; ++w)
            tile = fmaxf(tile, scores[w]);
        const float corr = isfinite(maximum) ? expf(maximum - tile) : 0.0f;
        total *= corr;
        acc *= corr;
        if (threadIdx.x == 0u) {
            for (unsigned w = 0; w < warps && first + w < end; ++w)
                total += expf(scores[w] - tile);
        }
        // Both 128-thread halves accumulate even/odd keys in parallel (2x value bandwidth!).
        for (unsigned w = val_half; w < warps && first + w < end; w += 2u) {
            const float p = expf(scores[w] - tile);
            const unsigned pos_k = pos_k_min + first + w;
            const unsigned slot_k =
                ring_mask ? (pos_k & ring_mask) : (ring_cap ? (pos_k % ring_cap) : pos_k);
            const __half* value =
                values + std::size_t(slot_k) * width + std::size_t(kv) * dim;
            acc = fmaf(p, __half2float(value[val_d]), acc);
        }
        maximum = tile;
        __syncthreads();
    }
    if (val_half == 1u)
        s_acc[val_d] = acc;
    __syncthreads();
    if (threadIdx.x == 0u) {
        out[0] = maximum;
        out[1] = total;
    }
    if (val_half == 0u)
        out[2u + val_d] = acc + s_acc[val_d];
}

// Stage 2 combine + virtual sink + fused sigmoid output gate!
__global__ void inkling_res_attention_combine_gated_kernel(
    const float* __restrict__ partial,
    const float* __restrict__ sinks,
    const float* __restrict__ gate,
    float* __restrict__ out,
    const StepParams* __restrict__ step_params,
    unsigned base_position,
    unsigned window,
    unsigned max_segments
) {
    constexpr unsigned dim = G::head_dim;
    const unsigned row = blockIdx.x, head = blockIdx.y;
    const unsigned d = threadIdx.x;  // 0..127
    const unsigned position = step_params ? step_params->position : base_position;
    const unsigned pos_q = position + row;
    const unsigned visible =
        (window && pos_q + 1u > window) ? window : (pos_q + 1u);
    const unsigned active_segs = min(max_segments, (visible + 255u) / 256u);

    const float* in =
        partial + (std::size_t(row) * G::n_heads + head) * max_segments * (dim + 2u);
    const float sink_logit = sinks[std::size_t(row) * G::n_heads + head];
    float maximum = sink_logit;
    for (unsigned z = 0; z < active_segs; ++z)
        if (in[z * (dim + 2u) + 1u] > 0.0f)
            maximum = fmaxf(maximum, in[z * (dim + 2u)]);
    float total = expf(sink_logit - maximum);
    float sum = 0.0f;
    for (unsigned z = 0; z < active_segs; ++z) {
        const float seg_total = in[z * (dim + 2u) + 1u];
        if (seg_total > 0.0f) {
            const float scale = expf(in[z * (dim + 2u)] - maximum);
            total += seg_total * scale;
            sum = fmaf(in[z * (dim + 2u) + 2u + d], scale, sum);
        }
    }
    const std::size_t idx = (std::size_t(row) * G::n_heads + head) * dim + d;
    const float g = gate[idx];
    const float sig = 1.0f / (1.0f + expf(-g));
    out[idx] = (sum / total) * sig;
}

// ============================================================================
// MoE routing, fused decode gate_up+SiTU & down+reduce, and prefill gather/scatter
// ============================================================================

// Top-8 MoE routing on GPU (works for both rows==1 decode & rows>1 prefill).
// Grid: rows blocks, Block: 128 threads (1 thread per expert).
__global__ void inkling_moe_route_kernel(
    const float* __restrict__ router_logits,
    const float* __restrict__ exp_bias,  // nullable
    unsigned* __restrict__ out_experts,  // [rows, 8]
    float* __restrict__ out_weights      // [rows, 8]
) {
    __shared__ float s_probs[G::n_experts];
    __shared__ float s_scores[G::n_experts];
    __shared__ unsigned s_top_e[G::n_experts_used];
    const unsigned row = blockIdx.x;
    const unsigned e = threadIdx.x;
    const float logit = router_logits[std::size_t(row) * G::n_experts + e];
    const float prob = 1.0f / (1.0f + expf(-logit));
    s_probs[e] = prob;
    s_scores[e] = prob + (exp_bias ? exp_bias[e] : 0.0f);
    __syncthreads();

    if (e == 0u) {
        double sum = 0.0;
        for (unsigned k = 0; k < G::n_experts_used; ++k) {
            float best_sc = -INFINITY;
            unsigned best_e = 0;
            for (unsigned j = 0; j < G::n_experts; ++j) {
                const float sc = s_scores[j];
                if (sc > best_sc) {
                    best_sc = sc;
                    best_e = j;
                }
            }
            s_top_e[k] = best_e;
            s_scores[best_e] = -INFINITY;
            sum += double(s_probs[best_e]);
        }
        const float inv = kExpertScale / float(sum + double(kExpertNormEps));
        for (unsigned k = 0; k < G::n_experts_used; ++k) {
            const unsigned ex = s_top_e[k];
            out_experts[std::size_t(row) * G::n_experts_used + k] = ex;
            out_weights[std::size_t(row) * G::n_experts_used + k] = s_probs[ex] * inv;
        }
    }
}

__device__ inline float situ_act(float g, float u) {
    const float g_act =
        kSituBeta * tanhf(g * (1.0f / kSituBeta)) * (1.0f / (1.0f + expf(-g)));
    const float u_act = kSituLinearBeta * tanhf(u * (1.0f / kSituLinearBeta));
    return g_act * u_act;
}

// Fused decode (rows == 1) indirect-GPU MoE gate_up + SiTU on aligned SoA weights:
// Grid: dim3(ceil(1152/4)=288, 8), Block: 128 threads (4 warps = 4 cols),
// 4 lanes/block coalesced LDG.32/LDG.64 + shared x_q8 (2.59 KB).
template <int Kind>
__global__ void inkling_moe_decode_gate_up_situ_kernel(
    const unsigned* __restrict__ experts,  // [8]
    const unsigned char* const* __restrict__ gate_up_ptrs,
    const Q8Block* __restrict__ x_q8,
    const float* __restrict__ x_f32,
    float* __restrict__ act_out,  // [8, 1152]
    std::size_t qs_row_bytes,
    std::size_t scales_offset
) {
    constexpr unsigned n_blocks = G::n_embd >> 5;  // 72
    __shared__ Q8Block s_x[Kind != kWKindF32 ? n_blocks : 1];
    if constexpr (Kind != kWKindF32)
        load_q8_shared(x_q8, s_x, n_blocks);

    const unsigned col = blockIdx.x * 4u + (threadIdx.x >> 5);
    if (col >= G::n_ff_exp)
        return;
    const unsigned k = blockIdx.y;
    const unsigned lane = threadIdx.x & 31u;
    const unsigned e = experts[k];
    const unsigned char* w_base = gate_up_ptrs[e];
    const unsigned char* w_gate = w_base + std::size_t(col) * qs_row_bytes;
    const unsigned char* w_up = w_base + std::size_t(col + G::n_ff_exp) * qs_row_bytes;

    float g_sum = 0.0f, u_sum = 0.0f;
    if constexpr (Kind == kWKindF32) {
        const auto* wg = reinterpret_cast<const float4*>(w_gate);
        const auto* wu = reinterpret_cast<const float4*>(w_up);
        const auto* xf = reinterpret_cast<const float4*>(x_f32);
        for (unsigned v = lane; v < (G::n_embd >> 2); v += 32u) {
            const float4 x = xf[v], g = wg[v], u = wu[v];
            g_sum = fmaf(g.x, x.x, g_sum);
            g_sum = fmaf(g.y, x.y, g_sum);
            g_sum = fmaf(g.z, x.z, g_sum);
            g_sum = fmaf(g.w, x.w, g_sum);
            u_sum = fmaf(u.x, x.x, u_sum);
            u_sum = fmaf(u.y, x.y, u_sum);
            u_sum = fmaf(u.z, x.z, u_sum);
            u_sum = fmaf(u.w, x.w, u_sum);
        }
    } else {
        const unsigned grp = lane >> 2;
        const unsigned sub = lane & 3u;
        const auto* sc_base = reinterpret_cast<const __half*>(w_base + scales_offset);
        const __half* sc_gate = sc_base + std::size_t(col) * n_blocks;
        const __half* sc_up = sc_base + std::size_t(col + G::n_ff_exp) * n_blocks;
        for (unsigned b = grp; b < n_blocks; b += 8u) {
            int32_t wg0, wg1, wu0, wu1;
            const float dg = (Kind == kWKindIQ4_NL)
                                 ? load_iq4nl_soa_quarter(w_gate, sc_gate, b, sub, wg0, wg1)
                                 : load_q8_0_soa_quarter(w_gate, sc_gate, b, sub, wg0, wg1);
            const float du = (Kind == kWKindIQ4_NL)
                                 ? load_iq4nl_soa_quarter(w_up, sc_up, b, sub, wu0, wu1)
                                 : load_q8_0_soa_quarter(w_up, sc_up, b, sub, wu0, wu1);
            const Q8Block& xb = s_x[b];
            int32_t ag, au;
            if constexpr (Kind == kWKindIQ4_NL) {
                const int32_t x0 = xb.qs[sub], x1 = xb.qs[sub + 4u];
                ag = __dp4a(wg1, x1, __dp4a(wg0, x0, 0));
                au = __dp4a(wu1, x1, __dp4a(wu0, x0, 0));
            } else {
                const int32_t x0 = xb.qs[sub * 2u], x1 = xb.qs[sub * 2u + 1u];
                ag = __dp4a(wg1, x1, __dp4a(wg0, x0, 0));
                au = __dp4a(wu1, x1, __dp4a(wu0, x0, 0));
            }
            g_sum = fmaf(dg * xb.d, float(ag), g_sum);
            u_sum = fmaf(du * xb.d, float(au), u_sum);
        }
    }
#pragma unroll
    for (unsigned off = 16; off; off >>= 1) {
        g_sum += __shfl_down_sync(0xffffffffu, g_sum, off);
        u_sum += __shfl_down_sync(0xffffffffu, u_sum, off);
    }
    if (lane == 0u)
        act_out[std::size_t(k) * G::n_ff_exp + col] = situ_act(g_sum, u_sum);
}

// Fused decode (rows == 1) deterministic MoE down + weighted reduce across all 8 experts:
// Grid: ceil(2304/4)=576 blocks, Block: 128 threads (4 warps = 4 cols),
// 4 lanes/block coalesced LDG.32/LDG.64 + shared act_q8 (8 * 36 blocks = 10.37 KB).
template <int Kind>
__global__ void inkling_moe_decode_down_reduce_kernel(
    const unsigned* __restrict__ experts,    // [8]
    const float* __restrict__ exp_weights,   // [8]
    const unsigned char* const* __restrict__ down_ptrs,
    const Q8Block* __restrict__ act_q8,      // [8, 1152/32=36]
    const float* __restrict__ act_f32,       // [8, 1152]
    float* __restrict__ moe_out,             // [2304]
    std::size_t qs_row_bytes,
    std::size_t scales_offset
) {
    constexpr unsigned n_blocks = G::n_ff_exp >> 5;  // 36
    __shared__ Q8Block s_act[Kind != kWKindF32 ? G::n_experts_used * n_blocks : 1];
    if constexpr (Kind != kWKindF32)
        load_q8_shared(act_q8, s_act, unsigned(G::n_experts_used * n_blocks));

    const unsigned col = blockIdx.x * 4u + (threadIdx.x >> 5);
    if (col >= G::n_embd)
        return;
    const unsigned lane = threadIdx.x & 31u;
    const unsigned grp = lane >> 2;
    const unsigned sub = lane & 3u;
    float total = 0.0f;
    for (unsigned k = 0; k < G::n_experts_used; ++k) {
        const unsigned e = experts[k];
        const float wk = exp_weights[k];
        const unsigned char* w_base = down_ptrs[e];
        const unsigned char* wrow = w_base + std::size_t(col) * qs_row_bytes;
        float sum = 0.0f;
        if constexpr (Kind == kWKindF32) {
            const auto* wf = reinterpret_cast<const float4*>(wrow);
            const auto* xf =
                reinterpret_cast<const float4*>(act_f32 + std::size_t(k) * G::n_ff_exp);
            for (unsigned v = lane; v < (G::n_ff_exp >> 2); v += 32u) {
                const float4 w = wf[v], x = xf[v];
                sum = fmaf(w.x, x.x, sum);
                sum = fmaf(w.y, x.y, sum);
                sum = fmaf(w.z, x.z, sum);
                sum = fmaf(w.w, x.w, sum);
            }
        } else {
            const __half* sc_row =
                reinterpret_cast<const __half*>(w_base + scales_offset) + std::size_t(col) * n_blocks;
            const Q8Block* xk = s_act + std::size_t(k) * n_blocks;
            for (unsigned b = grp; b < n_blocks; b += 8u) {
                int32_t w0, w1;
                const float wd = (Kind == kWKindIQ4_NL)
                                     ? load_iq4nl_soa_quarter(wrow, sc_row, b, sub, w0, w1)
                                     : load_q8_0_soa_quarter(wrow, sc_row, b, sub, w0, w1);
                const Q8Block& xb = xk[b];
                int32_t acc;
                if constexpr (Kind == kWKindIQ4_NL) {
                    acc = __dp4a(w0, xb.qs[sub], 0);
                    acc = __dp4a(w1, xb.qs[sub + 4u], acc);
                } else {
                    acc = __dp4a(w0, xb.qs[sub * 2u], 0);
                    acc = __dp4a(w1, xb.qs[sub * 2u + 1u], acc);
                }
                sum = fmaf(wd * xb.d, float(acc), sum);
            }
        }
#pragma unroll
        for (unsigned off = 16; off; off >>= 1)
            sum += __shfl_down_sync(0xffffffffu, sum, off);
        if (lane == 0u)
            total = fmaf(wk, sum, total);
    }
    if (lane == 0u)
        moe_out[col] = total;
}

// GPU-side CSR expert bucketing (zero host syncs during prefill!).
// 1 CTA (256 threads) counts 128 expert buckets, exclusive-scans offsets, and
// builds exp_token_idx[slot] and exp_slot_for_rk[row * 8 + k].
__global__ void inkling_moe_bucket_gpu_kernel(
    const unsigned* __restrict__ row_experts,  // [rows * 8]
    unsigned total_slots,
    unsigned* __restrict__ exp_offsets,        // [129]
    unsigned* __restrict__ exp_token_idx,      // [rows * 8]
    unsigned* __restrict__ exp_slot_for_rk     // [rows * 8]
) {
    static_assert(G::n_experts == 128 && G::n_experts_used == 8);
    __shared__ unsigned s_count[G::n_experts];
    __shared__ unsigned s_cursor[G::n_experts];
    const unsigned t = threadIdx.x;
    if (t < G::n_experts)
        s_count[t] = 0u;
    __syncthreads();

    for (unsigned rk = t; rk < total_slots; rk += blockDim.x)
        atomicAdd(&s_count[row_experts[rk]], 1u);
    __syncthreads();

    if (t == 0u) {
        unsigned acc = 0u;
        for (unsigned e = 0u; e < G::n_experts; ++e) {
            s_cursor[e] = acc;
            exp_offsets[e] = acc;
            acc += s_count[e];
        }
        exp_offsets[G::n_experts] = acc;
    }
    __syncthreads();

    for (unsigned rk = t; rk < total_slots; rk += blockDim.x) {
        const unsigned e = row_experts[rk];
        const unsigned pos = atomicAdd(&s_cursor[e], 1u);
        exp_token_idx[pos] = rk >> 3;
        exp_slot_for_rk[rk] = pos;
    }
}

// Grouped tiled quantized MMQ prefill kernel: gate_up + SiTU across ALL 128 experts
// in ONE launch! Reads SoA weights directly (zero W_fp16 HBM round-trip) with 16-row
// tiling (16x weight reuse, 4.5 KB shared memory) and gathers ffn_q8 on the fly.
// Grid: dim3(1152/4 = 288, 128), Block: 128 threads (4 warps = 4 cols).
template <int Kind>
__global__ void inkling_moe_prefill_gate_up_situ_kernel(
    const unsigned* __restrict__ exp_offsets,    // [129]
    const unsigned* __restrict__ exp_token_idx,  // [total_slots]
    const unsigned char* const* __restrict__ gate_up_ptrs,
    const Q8Block* __restrict__ ffn_q8,          // [rows, 72]
    const float* __restrict__ ffn_f32,           // [rows, 2304]
    float* __restrict__ act_out,                 // [total_slots, 1152]
    std::size_t qs_row_bytes,
    std::size_t scales_offset
) {
    static_assert(G::n_ff_exp % 4 == 0 && (G::n_embd >> 5) % 8 == 0);
    constexpr unsigned n_blocks = G::n_embd >> 5;  // 72
    constexpr unsigned kTileM = 16u;

    const unsigned e = blockIdx.y;
    const unsigned off = exp_offsets[e];
    const unsigned m = exp_offsets[e + 1u] - off;
    if (m == 0u)
        return;

    __shared__ unsigned s_tok[kTileM];
    __shared__ Q8Block s_x[Kind != kWKindF32 ? kTileM * 8u : 1];

    const unsigned col = blockIdx.x * 4u + (threadIdx.x >> 5);
    const unsigned lane = threadIdx.x & 31u;
    const unsigned grp = lane >> 2;
    const unsigned sub = lane & 3u;

    const unsigned char* w_base = gate_up_ptrs[e];
    const unsigned char* w_gate = w_base + std::size_t(col) * qs_row_bytes;
    const unsigned char* w_up = w_base + std::size_t(col + G::n_ff_exp) * qs_row_bytes;
    const __half* sc_gate =
        (Kind == kWKindF32)
            ? nullptr
            : (reinterpret_cast<const __half*>(w_base + scales_offset) + std::size_t(col) * n_blocks);
    const __half* sc_up =
        (Kind == kWKindF32)
            ? nullptr
            : (reinterpret_cast<const __half*>(w_base + scales_offset)
               + std::size_t(col + G::n_ff_exp) * n_blocks);

    for (unsigned r0 = 0u; r0 < m; r0 += kTileM) {
        const unsigned tile_rows = min(kTileM, m - r0);
        if (threadIdx.x < tile_rows)
            s_tok[threadIdx.x] = exp_token_idx[off + r0 + threadIdx.x];
        __syncthreads();

        float g_sum[kTileM];
        float u_sum[kTileM];
#pragma unroll
        for (unsigned r = 0u; r < kTileM; ++r) {
            g_sum[r] = 0.0f;
            u_sum[r] = 0.0f;
        }

        if constexpr (Kind == kWKindF32) {
            const auto* wg = reinterpret_cast<const float4*>(w_gate);
            const auto* wu = reinterpret_cast<const float4*>(w_up);
            for (unsigned v = lane; v < (G::n_embd >> 2); v += 32u) {
                const float4 g = wg[v], u = wu[v];
                for (unsigned r = 0u; r < tile_rows; ++r) {
                    const float4 x = reinterpret_cast<const float4*>(
                        ffn_f32 + std::size_t(s_tok[r]) * G::n_embd
                    )[v];
                    float gs = g_sum[r], us = u_sum[r];
                    gs = fmaf(g.x, x.x, gs);
                    gs = fmaf(g.y, x.y, gs);
                    gs = fmaf(g.z, x.z, gs);
                    gs = fmaf(g.w, x.w, gs);
                    us = fmaf(u.x, x.x, us);
                    us = fmaf(u.y, x.y, us);
                    us = fmaf(u.z, x.z, us);
                    us = fmaf(u.w, x.w, us);
                    g_sum[r] = gs;
                    u_sum[r] = us;
                }
            }
        } else {
            const unsigned lr = threadIdx.x >> 3;   // 0..15
            const unsigned lkb = threadIdx.x & 7u;  // 0..7
            for (unsigned b_base = 0u; b_base < n_blocks; b_base += 8u) {
                if (lr < tile_rows)
                    s_x[lr * 8u + lkb] =
                        ffn_q8[std::size_t(s_tok[lr]) * n_blocks + b_base + lkb];
                __syncthreads();

                const unsigned b = b_base + grp;
                int32_t wg0, wg1, wu0, wu1;
                const float dg = (Kind == kWKindIQ4_NL)
                                     ? load_iq4nl_soa_quarter(w_gate, sc_gate, b, sub, wg0, wg1)
                                     : load_q8_0_soa_quarter(w_gate, sc_gate, b, sub, wg0, wg1);
                const float du = (Kind == kWKindIQ4_NL)
                                     ? load_iq4nl_soa_quarter(w_up, sc_up, b, sub, wu0, wu1)
                                     : load_q8_0_soa_quarter(w_up, sc_up, b, sub, wu0, wu1);

                if (tile_rows == kTileM) {
#pragma unroll
                    for (unsigned r = 0u; r < kTileM; ++r) {
                        const Q8Block& xb = s_x[r * 8u + grp];
                        int32_t x0, x1;
                        if constexpr (Kind == kWKindIQ4_NL) {
                            x0 = xb.qs[sub];
                            x1 = xb.qs[sub + 4u];
                        } else {
                            x0 = xb.qs[sub * 2u];
                            x1 = xb.qs[sub * 2u + 1u];
                        }
                        const int32_t ag = __dp4a(wg1, x1, __dp4a(wg0, x0, 0));
                        const int32_t au = __dp4a(wu1, x1, __dp4a(wu0, x0, 0));
                        g_sum[r] = fmaf(dg * xb.d, float(ag), g_sum[r]);
                        u_sum[r] = fmaf(du * xb.d, float(au), u_sum[r]);
                    }
                } else {
                    for (unsigned r = 0u; r < tile_rows; ++r) {
                        const Q8Block& xb = s_x[r * 8u + grp];
                        int32_t x0, x1;
                        if constexpr (Kind == kWKindIQ4_NL) {
                            x0 = xb.qs[sub];
                            x1 = xb.qs[sub + 4u];
                        } else {
                            x0 = xb.qs[sub * 2u];
                            x1 = xb.qs[sub * 2u + 1u];
                        }
                        const int32_t ag = __dp4a(wg1, x1, __dp4a(wg0, x0, 0));
                        const int32_t au = __dp4a(wu1, x1, __dp4a(wu0, x0, 0));
                        g_sum[r] = fmaf(dg * xb.d, float(ag), g_sum[r]);
                        u_sum[r] = fmaf(du * xb.d, float(au), u_sum[r]);
                    }
                }
                __syncthreads();
            }
        }

        for (unsigned r = 0u; r < tile_rows; ++r) {
            float gs = g_sum[r], us = u_sum[r];
#pragma unroll
            for (unsigned sh = 16u; sh; sh >>= 1) {
                gs += __shfl_down_sync(0xffffffffu, gs, sh);
                us += __shfl_down_sync(0xffffffffu, us, sh);
            }
            if (lane == 0u)
                act_out[std::size_t(off + r0 + r) * G::n_ff_exp + col] = situ_act(gs, us);
        }
        __syncthreads();
    }
}

// Grouped tiled quantized MMQ prefill kernel: down projection across ALL 128 experts
// in ONE launch! Reads SoA weights directly with 32-row tiling (32x weight reuse,
// 9 KB shared memory) — single weight pass per expert on typical 512-token chunks.
// Grid: dim3(2304/4 = 576, 128), Block: 128 threads (4 warps = 4 cols).
template <int Kind>
__global__ void inkling_moe_prefill_down_kernel(
    const unsigned* __restrict__ exp_offsets,  // [129]
    const unsigned char* const* __restrict__ down_ptrs,
    const Q8Block* __restrict__ act_q8,        // [total_slots, 36]
    const float* __restrict__ act_f32,         // [total_slots, 1152]
    float* __restrict__ down_out,              // [total_slots, 2304]
    std::size_t qs_row_bytes,
    std::size_t scales_offset
) {
    static_assert(G::n_embd % 4 == 0);
    constexpr unsigned n_blocks = G::n_ff_exp >> 5;  // 36
    constexpr unsigned kTileM = 32u;

    const unsigned e = blockIdx.y;
    const unsigned off = exp_offsets[e];
    const unsigned m = exp_offsets[e + 1u] - off;
    if (m == 0u)
        return;

    __shared__ Q8Block s_x[Kind != kWKindF32 ? kTileM * 8u : 1];

    const unsigned col = blockIdx.x * 4u + (threadIdx.x >> 5);
    const unsigned lane = threadIdx.x & 31u;
    const unsigned grp = lane >> 2;
    const unsigned sub = lane & 3u;

    const unsigned char* w_base = down_ptrs[e];
    const unsigned char* wrow = w_base + std::size_t(col) * qs_row_bytes;
    const __half* sc_row =
        (Kind == kWKindF32)
            ? nullptr
            : (reinterpret_cast<const __half*>(w_base + scales_offset) + std::size_t(col) * n_blocks);

    for (unsigned r0 = 0u; r0 < m; r0 += kTileM) {
        const unsigned tile_rows = min(kTileM, m - r0);
        float sums[kTileM];
#pragma unroll
        for (unsigned r = 0u; r < kTileM; ++r)
            sums[r] = 0.0f;

        if constexpr (Kind == kWKindF32) {
            const auto* wf = reinterpret_cast<const float4*>(wrow);
            for (unsigned v = lane; v < (G::n_ff_exp >> 2); v += 32u) {
                const float4 w = wf[v];
                for (unsigned r = 0u; r < tile_rows; ++r) {
                    const float4 x = reinterpret_cast<const float4*>(
                        act_f32 + std::size_t(off + r0 + r) * G::n_ff_exp
                    )[v];
                    float s = sums[r];
                    s = fmaf(w.x, x.x, s);
                    s = fmaf(w.y, x.y, s);
                    s = fmaf(w.z, x.z, s);
                    s = fmaf(w.w, x.w, s);
                    sums[r] = s;
                }
            }
        } else {
            for (unsigned b_base = 0u; b_base < n_blocks; b_base += 8u) {
                for (unsigned idx = threadIdx.x; idx < tile_rows * 8u; idx += 128u) {
                    const unsigned lr = idx >> 3;
                    const unsigned lkb = idx & 7u;
                    if (b_base + lkb < n_blocks)
                        s_x[idx] =
                            act_q8[std::size_t(off + r0 + lr) * n_blocks + b_base + lkb];
                }
                __syncthreads();

                const unsigned b = b_base + grp;
                if (b < n_blocks) {
                    int32_t w0, w1;
                    const float wd = (Kind == kWKindIQ4_NL)
                                         ? load_iq4nl_soa_quarter(wrow, sc_row, b, sub, w0, w1)
                                         : load_q8_0_soa_quarter(wrow, sc_row, b, sub, w0, w1);
                    if (tile_rows == kTileM) {
#pragma unroll
                        for (unsigned r = 0u; r < kTileM; ++r) {
                            const Q8Block& xb = s_x[r * 8u + grp];
                            int32_t acc;
                            if constexpr (Kind == kWKindIQ4_NL) {
                                acc = __dp4a(w0, xb.qs[sub], 0);
                                acc = __dp4a(w1, xb.qs[sub + 4u], acc);
                            } else {
                                acc = __dp4a(w0, xb.qs[sub * 2u], 0);
                                acc = __dp4a(w1, xb.qs[sub * 2u + 1u], acc);
                            }
                            sums[r] = fmaf(wd * xb.d, float(acc), sums[r]);
                        }
                    } else {
                        for (unsigned r = 0u; r < tile_rows; ++r) {
                            const Q8Block& xb = s_x[r * 8u + grp];
                            int32_t acc;
                            if constexpr (Kind == kWKindIQ4_NL) {
                                acc = __dp4a(w0, xb.qs[sub], 0);
                                acc = __dp4a(w1, xb.qs[sub + 4u], acc);
                            } else {
                                acc = __dp4a(w0, xb.qs[sub * 2u], 0);
                                acc = __dp4a(w1, xb.qs[sub * 2u + 1u], acc);
                            }
                            sums[r] = fmaf(wd * xb.d, float(acc), sums[r]);
                        }
                    }
                }
                __syncthreads();
            }
        }

        for (unsigned r = 0u; r < tile_rows; ++r) {
            float s = sums[r];
#pragma unroll
            for (unsigned sh = 16u; sh; sh >>= 1)
                s += __shfl_down_sync(0xffffffffu, s, sh);
            if (lane == 0u)
                down_out[std::size_t(off + r0 + r) * G::n_embd + col] = s;
        }
    }
}

// Prefill MoE scatter: weighted sum of each token's 8 expert outputs into moe_out[rows, 2304].
__global__ void inkling_moe_scatter_kernel(
    const float* __restrict__ down_out,       // [rows*8, 2304]
    const unsigned* __restrict__ slot_for_rk, // [rows, 8]
    const float* __restrict__ row_weights,    // [rows, 8]
    float* __restrict__ moe_out               // [rows, 2304]
) {
    const unsigned row = blockIdx.x;
    const unsigned t = threadIdx.x;
    float acc[9];
#pragma unroll
    for (unsigned j = 0; j < 9; ++j)
        acc[j] = 0.0f;
    for (unsigned k = 0; k < G::n_experts_used; ++k) {
        const unsigned slot = slot_for_rk[std::size_t(row) * G::n_experts_used + k];
        const float w = row_weights[std::size_t(row) * G::n_experts_used + k];
        const float* src = down_out + std::size_t(slot) * G::n_embd;
#pragma unroll
        for (unsigned j = 0; j < 9; ++j)
            acc[j] = fmaf(w, src[t + j * 256u], acc[j]);
    }
    float* dst = moe_out + std::size_t(row) * G::n_embd;
#pragma unroll
    for (unsigned j = 0; j < 9; ++j)
        dst[t + j * 256u] = acc[j];
}

// ============================================================================
// Helper dispatchers
// ============================================================================

inline std::optional<WeightKind> to_weight_kind(std::optional<gguf::Encoding> enc) {
    if (!enc)
        return kWKindF32;
    if (*enc == gguf::Encoding::q8_0)
        return kWKindQ8_0;
    if (*enc == gguf::Encoding::iq4_nl)
        return kWKindIQ4_NL;
    return std::nullopt;
}

inline void launch_dequant_fp16(
    const MatrixRef& m, __half* dst, cudaStream_t stream
) {
    const std::size_t total_blocks = (std::size_t(m.out) * m.in) / 32u;
    const unsigned grid = unsigned((total_blocks + 7u) / 8u);
    if (m.kind == kWKindF32)
        inkling_dequant_fp16_kernel<kWKindF32><<<grid, 256, 0, stream>>>(
            m.qs, m.scales, dst, total_blocks
        );
    else if (m.kind == kWKindQ8_0)
        inkling_dequant_fp16_kernel<kWKindQ8_0><<<grid, 256, 0, stream>>>(
            m.qs, m.scales, dst, total_blocks
        );
    else
        inkling_dequant_fp16_kernel<kWKindIQ4_NL><<<grid, 256, 0, stream>>>(
            m.qs, m.scales, dst, total_blocks
        );
    checked(cudaGetLastError());
}

inline void launch_quantize_q8(
    const float* src, Q8Block* dst, std::size_t floats, cudaStream_t stream
) {
    const std::size_t total_blocks = floats / 32u;
    const unsigned grid = unsigned((total_blocks + 7u) / 8u);
    inkling_quantize_q8_kernel<<<grid, 256, 0, stream>>>(src, dst, total_blocks);
    checked(cudaGetLastError());
}

inline void launch_matvec_dp4a(
    const MatrixRef& m,
    const Q8Block* x_q8,
    const float* x_f32,
    float* out,
    unsigned rows,
    cudaStream_t stream
) {
    const unsigned grid = (m.out + 3u) / 4u;
    auto dispatch = [&](auto kind_tag) {
        constexpr int K = decltype(kind_tag)::value;
        if (rows == 1u)
            inkling_matvec_kernel<K, 1><<<grid, 128, 0, stream>>>(
                m.qs, m.scales, x_q8, x_f32, out, m.qs_row_bytes, m.in, m.out, 1u
            );
        else if (rows <= 4u)
            inkling_matvec_kernel<K, 4><<<grid, 128, 0, stream>>>(
                m.qs, m.scales, x_q8, x_f32, out, m.qs_row_bytes, m.in, m.out, rows
            );
        else if (rows <= 8u)
            inkling_matvec_kernel<K, 8><<<grid, 128, 0, stream>>>(
                m.qs, m.scales, x_q8, x_f32, out, m.qs_row_bytes, m.in, m.out, rows
            );
        else
            inkling_matvec_kernel<K, 15><<<grid, 128, 0, stream>>>(
                m.qs, m.scales, x_q8, x_f32, out, m.qs_row_bytes, m.in, m.out, rows
            );
    };
    if (m.kind == kWKindF32)
        dispatch(std::integral_constant<int, kWKindF32>{});
    else if (m.kind == kWKindQ8_0)
        dispatch(std::integral_constant<int, kWKindQ8_0>{});
    else
        dispatch(std::integral_constant<int, kWKindIQ4_NL>{});
    checked(cudaGetLastError());
}

inline void launch_fused_decode_proj(
    std::initializer_list<std::pair<const MatrixRef*, float*>> pairs,
    const Q8Block* x_q8,
    const float* x_f32,
    cudaStream_t stream
) {
    FusedDecodeProjArgs args{};
    args.x_q8 = nullptr;
    args.x_f32 = x_f32;
    unsigned off = 0u;
    for (const auto& [m, out] : pairs) {
        if (!m)
            continue;
        if (args.count == 0u)
            args.in_dim = m->in;
        if (m->kind != kWKindF32)
            args.x_q8 = x_q8;
        args.entries[args.count++] = {
            m->qs, m->scales, out, m->qs_row_bytes, m->out, off, int(m->kind)
        };
        off += m->out;
    }
    args.total_cols = off;
    const unsigned grid = (args.total_cols + 3u) / 4u;
    inkling_fused_decode_proj_kernel<<<grid, 128, 0, stream>>>(args);
    checked(cudaGetLastError());
}

// ============================================================================
// GPU-resident Inkling 2 Mini Executor
// ============================================================================

struct ScratchViews {
    float* residual[9]{};
    float* inputs[3]{};
    float* messages[4]{};
    float* q = nullptr;
    float* gate = nullptr;
    float* attn_out = nullptr;
    float* k_raw = nullptr;
    float* v_raw = nullptr;
    float* k = nullptr;
    float* v = nullptr;
    float* r_cur = nullptr;
    float* sinks = nullptr;
    float* k_dyn = nullptr;
    float* v_dyn = nullptr;
    float* ctrl_cat = nullptr;
    float* ctrl_gate = nullptr;
    float* ctrl_g = nullptr;
    float* ctrl_f = nullptr;
    float* router_logits = nullptr;
    float* moe_in = nullptr;       // also reused as down_out
    __half* moe_in_fp16 = nullptr;
    float* gate_up_out = nullptr;
    float* act_out = nullptr;
    __half* act_out_fp16 = nullptr;
    __half* in_fp16 = nullptr;
    __half* w_fp16 = nullptr;
    Q8Block* x_q8 = nullptr;
    float* logits = nullptr;
    float* attn_partial = nullptr;
    void* blas_ws = nullptr;
    unsigned* row_experts = nullptr;      // [rows, 8]
    float* row_weights = nullptr;         // [rows, 8]
    unsigned* exp_offsets = nullptr;      // [129]
    unsigned* exp_token_idx = nullptr;    // [rows * 8]
    unsigned* exp_slot_for_rk = nullptr;  // [rows * 8]
    float* argmax_p_val = nullptr;        // [256]
    unsigned* argmax_p_idx = nullptr;     // [256]
    unsigned* argmax_p_bad = nullptr;     // [256]
    ArgmaxResult* argmax_out = nullptr;   // [1]
    std::size_t total_bytes = 0;

    static ScratchViews plan(std::size_t rows, CudaDeviceOperations& arena, bool bind_ptrs) {
        ScratchViews v{};
        Layout l;
        const std::size_t expert_rows = rows * G::n_experts_used;
        std::size_t off_res[9], off_inp[3], off_msg[4];
        for (int i = 0; i < 9; ++i)
            off_res[i] = l.add(rows * G::n_embd * sizeof(float));
        for (int i = 0; i < 3; ++i)
            off_inp[i] = l.add(rows * G::n_embd * sizeof(float));
        for (int i = 0; i < 4; ++i)
            off_msg[i] = l.add(rows * G::n_embd * sizeof(float));
        const auto off_q = l.add(rows * G::n_embd_q * sizeof(float));
        const auto off_gate = l.add(rows * G::n_embd_q * sizeof(float));
        const auto off_attn_out = l.add(rows * G::n_embd_q * sizeof(float));
        const auto off_k_raw = l.add(rows * G::n_embd_k_gqa * sizeof(float));
        const auto off_v_raw = l.add(rows * G::n_embd_k_gqa * sizeof(float));
        const auto off_k = l.add(rows * G::n_embd_k_gqa * sizeof(float));
        const auto off_v = l.add(rows * G::n_embd_k_gqa * sizeof(float));
        const auto off_r_cur = l.add(rows * G::n_heads * G::n_rel * sizeof(float));
        const auto off_sinks = l.add(rows * G::n_heads * sizeof(float));
        const auto off_k_dyn = l.add(rows * G::n_conv * G::n_kv_heads * sizeof(float));
        const auto off_v_dyn = l.add(rows * G::n_conv * G::n_kv_heads * sizeof(float));
        const auto off_ctrl_cat = l.add(rows * 2 * G::n_embd * sizeof(float));
        const auto off_ctrl_gate = l.add(rows * G::loop_gate_rank * sizeof(float));
        const auto off_ctrl_g = l.add(rows * G::n_embd * sizeof(float));
        const auto off_ctrl_f = l.add(rows * G::n_embd * sizeof(float));
        const auto off_router = l.add(rows * G::n_experts * sizeof(float));
        const auto off_moe_in = l.add(expert_rows * G::n_embd * sizeof(float));
        const auto off_moe_in_fp16 = l.add(expert_rows * G::n_embd * sizeof(__half));
        const auto off_gate_up = l.add(expert_rows * 2 * G::n_ff_exp * sizeof(float));
        const auto off_act = l.add(expert_rows * G::n_ff_exp * sizeof(float));
        const auto off_act_fp16 = l.add(expert_rows * G::n_ff_exp * sizeof(__half));
        const auto off_in_fp16 = l.add(rows * 2 * G::n_embd * sizeof(__half));
        const auto off_w_fp16 = l.add(G::n_embd_q * G::n_embd * sizeof(__half));
        const auto off_x_q8 = l.add(expert_rows * (G::n_embd / 32) * sizeof(Q8Block));
        const auto off_logits = l.add(G::n_vocab_unpadded * sizeof(float));
        const auto off_partial = l.add(kAttnPartialBytes);
        const auto off_blas = l.add(kBlasWorkspaceBytes);
        const auto off_meta = l.add(kRoutingMetaBytes);
        v.total_bytes = l.bytes;
        if (!bind_ptrs)
            return v;
        for (int i = 0; i < 9; ++i)
            v.residual[i] = arena.at<float>(off_res[i]);
        for (int i = 0; i < 3; ++i)
            v.inputs[i] = arena.at<float>(off_inp[i]);
        for (int i = 0; i < 4; ++i)
            v.messages[i] = arena.at<float>(off_msg[i]);
        v.q = arena.at<float>(off_q);
        v.gate = arena.at<float>(off_gate);
        v.attn_out = arena.at<float>(off_attn_out);
        v.k_raw = arena.at<float>(off_k_raw);
        v.v_raw = arena.at<float>(off_v_raw);
        v.k = arena.at<float>(off_k);
        v.v = arena.at<float>(off_v);
        v.r_cur = arena.at<float>(off_r_cur);
        v.sinks = arena.at<float>(off_sinks);
        v.k_dyn = arena.at<float>(off_k_dyn);
        v.v_dyn = arena.at<float>(off_v_dyn);
        v.ctrl_cat = arena.at<float>(off_ctrl_cat);
        v.ctrl_gate = arena.at<float>(off_ctrl_gate);
        v.ctrl_g = arena.at<float>(off_ctrl_g);
        v.ctrl_f = arena.at<float>(off_ctrl_f);
        v.router_logits = arena.at<float>(off_router);
        v.moe_in = arena.at<float>(off_moe_in);
        v.moe_in_fp16 = arena.at<__half>(off_moe_in_fp16);
        v.gate_up_out = arena.at<float>(off_gate_up);
        v.act_out = arena.at<float>(off_act);
        v.act_out_fp16 = arena.at<__half>(off_act_fp16);
        v.in_fp16 = arena.at<__half>(off_in_fp16);
        v.w_fp16 = arena.at<__half>(off_w_fp16);
        v.x_q8 = arena.at<Q8Block>(off_x_q8);
        v.logits = arena.at<float>(off_logits);
        v.attn_partial = arena.at<float>(off_partial);
        v.blas_ws = arena.at<void>(off_blas);
        // Sub-slice routing & argmax metadata inside the 2 MiB block
        Layout ml;
        const auto off_re = ml.add(expert_rows * sizeof(unsigned));
        const auto off_rw = ml.add(expert_rows * sizeof(float));
        const auto off_eo = ml.add((G::n_experts + 1) * sizeof(unsigned));
        const auto off_ti = ml.add(expert_rows * sizeof(unsigned));
        const auto off_sr = ml.add(expert_rows * sizeof(unsigned));
        const auto off_apv = ml.add(256 * sizeof(float));
        const auto off_api = ml.add(256 * sizeof(unsigned));
        const auto off_apb = ml.add(256 * sizeof(unsigned));
        const auto off_ao = ml.add(sizeof(ArgmaxResult));
        require(ml.bytes <= kRoutingMetaBytes, "inkling_gpu_meta_scratch");
        v.row_experts = arena.at<unsigned>(off_meta + off_re);
        v.row_weights = arena.at<float>(off_meta + off_rw);
        v.exp_offsets = arena.at<unsigned>(off_meta + off_eo);
        v.exp_token_idx = arena.at<unsigned>(off_meta + off_ti);
        v.exp_slot_for_rk = arena.at<unsigned>(off_meta + off_sr);
        v.argmax_p_val = arena.at<float>(off_meta + off_apv);
        v.argmax_p_idx = arena.at<unsigned>(off_meta + off_api);
        v.argmax_p_bad = arena.at<unsigned>(off_meta + off_apb);
        v.argmax_out = arena.at<ArgmaxResult>(off_meta + off_ao);
        return v;
    }
};

class InklingGpuRunner final : public kadan::InklingGpuExecutor {
public:
    using StateRunner = std::function<void(
        const std::atomic_bool&,
        std::function<void(DeviceWorkspace&, const std::atomic_bool&)>
    )>;

    InklingGpuRunner(
        StateRunner run_on_state,
        kadan::InklingGpuModelDesc desc
    )
        : run_on_state_(std::move(run_on_state)), desc_(std::move(desc)) {
        // Copy small host-side F32 parameter spans so desc_ remains self-contained
        // if re-uploaded after release_scratch(false) / release_devices().
        out_norm_.assign(desc_.out_norm.begin(), desc_.out_norm.end());
        out_res_query_.assign(desc_.out_res_query.begin(), desc_.out_res_query.end());
        host_layers_.resize(G::n_phys);
        for (std::size_t i = 0; i < G::n_phys; ++i) {
            const auto& src = desc_.layers[i];
            auto& dst = host_layers_[i];
            dst.is_swa = src.is_swa;
            dst.has_attn_res = src.has_attn_res;
            dst.has_exp_bias = src.has_exp_bias;
            dst.attn_norm.assign(src.attn_norm.begin(), src.attn_norm.end());
            dst.attn_q_inp_norm.assign(src.attn_q_inp_norm.begin(), src.attn_q_inp_norm.end());
            dst.attn_k_inp_norm.assign(src.attn_k_inp_norm.begin(), src.attn_k_inp_norm.end());
            dst.attn_v_inp_norm.assign(src.attn_v_inp_norm.begin(), src.attn_v_inp_norm.end());
            dst.attn_q_clip.assign(src.attn_q_clip.begin(), src.attn_q_clip.end());
            dst.attn_rel_bias.assign(src.attn_rel_bias.begin(), src.attn_rel_bias.end());
            dst.attn_rel_clip.assign(src.attn_rel_clip.begin(), src.attn_rel_clip.end());
            for (std::size_t k = 0; k < G::n_conv; ++k) {
                dst.attn_k_conv[k].assign(src.attn_k_conv[k].begin(), src.attn_k_conv[k].end());
                dst.attn_v_conv[k].assign(src.attn_v_conv[k].begin(), src.attn_v_conv[k].end());
                dst.attn_post_conv[k].assign(src.attn_post_conv[k].begin(), src.attn_post_conv[k].end());
                dst.ffn_post_conv[k].assign(src.ffn_post_conv[k].begin(), src.ffn_post_conv[k].end());
            }
            dst.attn_res_query.assign(src.attn_res_query.begin(), src.attn_res_query.end());
            dst.ffn_res_query.assign(src.ffn_res_query.begin(), src.ffn_res_query.end());
            dst.ffn_norm.assign(src.ffn_norm.begin(), src.ffn_norm.end());
            dst.ffn_exp_probs_b.assign(src.ffn_exp_probs_b.begin(), src.ffn_exp_probs_b.end());
        }
        host_loops_.resize(G::loop_core_len);
        for (std::size_t i = 0; i < G::loop_core_len; ++i) {
            host_loops_[i].msg_query.assign(
                desc_.loop_layers[i].msg_query.begin(),
                desc_.loop_layers[i].msg_query.end()
            );
        }
    }

    // Validates formats and eagerly admits all weights + small params on GPU.
    // Returns false if the GPU cannot hold the full model (clean Strata fallback).
    bool initialize(const std::atomic_bool& cancel) {
        if (desc_.layers.size() != G::n_phys || desc_.loop_layers.size() != G::loop_core_len)
            return false;
        if (!to_weight_kind(desc_.output_weight.encoding))
            return false;
        for (std::size_t i = 0; i < G::n_phys; ++i) {
            const auto& l = desc_.layers[i];
            for (const auto* m :
                 {&l.wq, &l.wk, &l.wv, &l.wo, &l.wgate, &l.wrel, &l.wsink, &l.wffn_router})
                if (!to_weight_kind(m->encoding))
                    return false;
            if (!l.is_swa) {
                if (!to_weight_kind(l.wk_conv_proj.encoding)
                    || !to_weight_kind(l.wv_conv_proj.encoding))
                    return false;
            }
            const auto gk = to_weight_kind(l.wffn_gate_up[0].encoding);
            const auto dk = to_weight_kind(l.wffn_down[0].encoding);
            if (!gk || !dk)
                return false;
            for (std::size_t e = 0; e < G::n_experts; ++e) {
                if (to_weight_kind(l.wffn_gate_up[e].encoding) != gk
                    || to_weight_kind(l.wffn_down[e].encoding) != dk)
                    return false;
            }
        }
        for (std::size_t i = 0; i < G::loop_core_len; ++i) {
            const auto& ll = desc_.loop_layers[i];
            for (std::size_t c = 0; c < G::loop_ctrl_count; ++c) {
                if (!to_weight_kind(ll.ctrl_down[c].encoding)
                    || !to_weight_kind(ll.ctrl_gain[c].encoding)
                    || !to_weight_kind(ll.ctrl_forget[c].encoding))
                    return false;
            }
        }
        bool ok = false;
        run_on_state_(
            cancel,
            [&](DeviceWorkspace& ws, const std::atomic_bool& failed) {
                ok = ensure_weights_and_params(ws, cancel, failed);
            }
        );
        return ok;
    }

    void reset() override {
        std::atomic_bool no_cancel{false};
        run_on_state_(
            no_cancel,
            [&](DeviceWorkspace& ws, const std::atomic_bool&) {
                auto& arena =
                    static_cast<CudaDeviceOperations&>(ws.operations());
                arena.invalidate_inkling_graph();
                if (auto it = arena.recurrent.find(kInklingConvKey);
                    it != arena.recurrent.end() && it->second.data) {
                    checked(cudaMemset(
                        it->second.data, 0, it->second.floats * sizeof(float)
                    ));
                }
                for (std::size_t s = 0; s < G::n_slots; ++s) {
                    if (auto it = arena.mirrors.find(s); it != arena.mirrors.end())
                        it->second.valid = 0;
                }
            }
        );
    }

    void sync_to_host(
        std::span<text::InklingLayerState> slots,
        std::size_t valid_tokens,
        const std::atomic_bool& cancel
    ) override {
        require(slots.size() == G::n_slots, "inkling_gpu_sync_slots");
        run_on_state_(
            cancel,
            [&](DeviceWorkspace& ws, const std::atomic_bool& failed) {
                auto& arena =
                    static_cast<CudaDeviceOperations&>(ws.operations());
                cudaStream_t stream = arena.ensure_inkling_stream();
                const auto conv_it = arena.recurrent.find(kInklingConvKey);
                if (conv_it == arena.recurrent.end() || !conv_it->second.data)
                    return;

                // Double-buffered pinned staging (2 x 8 MiB) overlaps FP16->F32
                // conversion, PCIe DMA, and host unpack across all 48 slots.
                constexpr std::size_t staging_bytes = gguf_projection_staging_bytes;
                constexpr std::size_t half_bytes = staging_bytes / 2;
                constexpr std::size_t half_floats = half_bytes / sizeof(float);
                static_assert(half_floats >= G::n_swa_window * G::n_embd_k_gqa);

                auto pinned = ws.host_staging(staging_bytes);
                ws.ensure(staging_bytes);
                float* pinned_half[2] = {
                    reinterpret_cast<float*>(pinned.data()),
                    reinterpret_cast<float*>(pinned.data() + half_bytes)
                };
                float* dev_half[2] = {
                    arena.at<float>(0),
                    arena.at<float>(half_bytes)
                };

                // 1. Download all 48 slots' conv state (~3.83 MiB) in one pinned DMA.
                const std::size_t conv_slot_floats = G::n_conv_hist * G::n_conv_channels;
                const std::size_t total_conv_bytes =
                    G::n_slots * conv_slot_floats * sizeof(float);
                require(total_conv_bytes <= half_bytes, "inkling_gpu_conv_staging");
                checked(cudaMemcpyAsync(
                    pinned_half[0],
                    conv_it->second.data,
                    total_conv_bytes,
                    cudaMemcpyDeviceToHost,
                    stream
                ));
                checked(cudaStreamSynchronize(stream));
                for (std::size_t s = 0; s < G::n_slots; ++s) {
                    std::memcpy(
                        slots[s].conv.data(),
                        pinned_half[0] + s * conv_slot_floats,
                        conv_slot_floats * sizeof(float)
                    );
                }
                if (!valid_tokens)
                    return;

                struct Events {
                    cudaEvent_t ev[2]{};
                    ~Events() {
                        for (auto e : ev)
                            if (e)
                                cudaEventDestroy(e);
                    }
                } events;
                for (auto& e : events.ev)
                    checked(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));

                struct Pending {
                    bool active = false;
                    float* dst = nullptr;
                    std::size_t rows = 0;
                    bool is_swa = false;
                    std::size_t swa_win = 0;
                    std::size_t start_pos = 0;
                };
                Pending pending[2]{};

                auto drain_half = [&](unsigned p) {
                    if (!pending[p].active)
                        return;
                    checked(cudaEventSynchronize(events.ev[p]));
                    const auto& job = pending[p];
                    const float* src = pinned_half[p];
                    if (job.is_swa) {
                        for (std::size_t i = 0; i < job.rows; ++i) {
                            const std::size_t dst_slot = (job.start_pos + i) % job.swa_win;
                            std::memcpy(
                                job.dst + dst_slot * G::n_embd_k_gqa,
                                src + i * G::n_embd_k_gqa,
                                G::n_embd_k_gqa * sizeof(float)
                            );
                        }
                    } else {
                        std::memcpy(
                            job.dst,
                            src,
                            job.rows * G::n_embd_k_gqa * sizeof(float)
                        );
                    }
                    pending[p].active = false;
                };

                unsigned step = 0;
                auto enqueue_download = [&](const __half* d_src,
                                            float* h_dst,
                                            std::size_t n_rows,
                                            bool is_swa,
                                            std::size_t swa_win,
                                            std::size_t start_pos) {
                    constexpr std::size_t max_rows_per_half = half_floats / G::n_embd_k_gqa;
                    for (std::size_t off = 0; off < n_rows; off += max_rows_per_half) {
                        stop(cancel, failed);
                        const std::size_t chunk_rows =
                            std::min(max_rows_per_half, n_rows - off);
                        const std::size_t n_floats = chunk_rows * G::n_embd_k_gqa;
                        const unsigned p = step++ & 1u;
                        drain_half(p);
                        inkling_fp16_to_f32_kernel<<<
                            unsigned((n_floats + 255) / 256), 256, 0, stream>>>(
                            d_src + off * G::n_embd_k_gqa, dev_half[p], n_floats
                        );
                        checked(cudaGetLastError());
                        checked(cudaMemcpyAsync(
                            pinned_half[p],
                            dev_half[p],
                            n_floats * sizeof(float),
                            cudaMemcpyDeviceToHost,
                            stream
                        ));
                        checked(cudaEventRecord(events.ev[p], stream));
                        pending[p] = {
                            true,
                            is_swa ? h_dst : (h_dst + off * G::n_embd_k_gqa),
                            chunk_rows,
                            is_swa,
                            swa_win,
                            start_pos + off
                        };
                    }
                };

                for (std::size_t s = 0; s < G::n_slots; ++s) {
                    const auto mit = arena.mirrors.find(s);
                    if (mit == arena.mirrors.end() || !mit->second.data)
                        continue;
                    const auto& m = mit->second;
                    const auto* m_keys = reinterpret_cast<const __half*>(m.data);
                    const auto* m_vals = m_keys + m.capacity * G::n_embd_k_gqa;
                    const bool is_swa = G::is_swa_phys(G::slot_phys(s));
                    if (is_swa) {
                        const std::size_t swa_win = G::slot_kv_capacity(s, desc_.context);
                        const std::size_t ring_cap = 2 * swa_win;
                        const std::size_t win_start =
                            valid_tokens > swa_win ? valid_tokens - swa_win : 0;
                        const std::size_t count = valid_tokens - win_start;
                        const std::size_t src_start = win_start % ring_cap;
                        const std::size_t part1 = std::min(count, ring_cap - src_start);
                        enqueue_download(
                            m_keys + src_start * G::n_embd_k_gqa,
                            slots[s].attention.keys.data(),
                            part1,
                            true,
                            swa_win,
                            win_start
                        );
                        enqueue_download(
                            m_vals + src_start * G::n_embd_k_gqa,
                            slots[s].attention.values.data(),
                            part1,
                            true,
                            swa_win,
                            win_start
                        );
                        if (part1 < count) {
                            const std::size_t part2 = count - part1;
                            enqueue_download(
                                m_keys,
                                slots[s].attention.keys.data(),
                                part2,
                                true,
                                swa_win,
                                win_start + part1
                            );
                            enqueue_download(
                                m_vals,
                                slots[s].attention.values.data(),
                                part2,
                                true,
                                swa_win,
                                win_start + part1
                            );
                        }
                    } else {
                        enqueue_download(
                            m_keys,
                            slots[s].attention.keys.data(),
                            valid_tokens,
                            false,
                            0,
                            0
                        );
                        enqueue_download(
                            m_vals,
                            slots[s].attention.values.data(),
                            valid_tokens,
                            false,
                            0,
                            0
                        );
                    }
                }
                drain_half(0);
                drain_half(1);
            }
        );
    }

    void sync_from_host(
        std::span<const text::InklingLayerState> slots,
        std::size_t valid_tokens,
        const std::atomic_bool& cancel
    ) override {
        require(slots.size() == G::n_slots, "inkling_gpu_sync_slots");
        run_on_state_(
            cancel,
            [&](DeviceWorkspace& ws, const std::atomic_bool& failed) {
                auto& arena =
                    static_cast<CudaDeviceOperations&>(ws.operations());
                arena.sync_inkling_stream();
                if (!ensure_weights_and_params(ws, cancel, failed))
                    return;
                auto* conv_rec = ensure_conv_state(arena);
                if (!conv_rec)
                    return;

                // Ensure all 48 KV mirrors exist before launching async stream transfers.
                for (std::size_t s = 0; s < G::n_slots; ++s) {
                    const bool is_swa = G::is_swa_phys(G::slot_phys(s));
                    const std::size_t swa_win = G::slot_kv_capacity(s, desc_.context);
                    const std::size_t need_pos =
                        is_swa ? 2 * swa_win : std::max<std::size_t>(valid_tokens, 256);
                    const std::size_t old_cap =
                        arena.mirrors.count(s) ? arena.mirrors[s].capacity : 0;
                    auto* m = arena.mirror(s, need_pos, G::n_embd_k_gqa, true);
                    if (!m)
                        return;
                    if (m->capacity != old_cap)
                        arena.invalidate_inkling_graph();
                    m->valid = valid_tokens;
                }
                checked(cuda_trace::synchronize(cudaStreamLegacy));

                constexpr std::size_t staging_bytes = gguf_projection_staging_bytes;
                constexpr std::size_t half_bytes = staging_bytes / 2;
                constexpr std::size_t half_floats = half_bytes / sizeof(float);
                static_assert(half_floats >= G::n_swa_window * G::n_embd_k_gqa);

                auto pinned = ws.host_staging(staging_bytes);
                ws.ensure(staging_bytes);
                cudaStream_t stream = arena.ensure_inkling_stream();
                float* pinned_half[2] = {
                    reinterpret_cast<float*>(pinned.data()),
                    reinterpret_cast<float*>(pinned.data() + half_bytes)
                };
                float* dev_half[2] = {
                    arena.at<float>(0),
                    arena.at<float>(half_bytes)
                };

                struct Events {
                    cudaEvent_t ev[2]{};
                    ~Events() {
                        for (auto e : ev)
                            if (e)
                                cudaEventDestroy(e);
                    }
                } events;
                for (auto& e : events.ev)
                    checked(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
                bool ev_busy[2] = {false, false};

                // 1. Pack all 48 slots' conv state into pinned half 0 and DMA H2D once.
                const std::size_t conv_slot_floats = G::n_conv_hist * G::n_conv_channels;
                const std::size_t total_conv_bytes =
                    G::n_slots * conv_slot_floats * sizeof(float);
                require(total_conv_bytes <= half_bytes, "inkling_gpu_conv_staging");
                for (std::size_t s = 0; s < G::n_slots; ++s) {
                    std::memcpy(
                        pinned_half[0] + s * conv_slot_floats,
                        slots[s].conv.data(),
                        conv_slot_floats * sizeof(float)
                    );
                }
                checked(cudaMemcpyAsync(
                    conv_rec->data,
                    pinned_half[0],
                    total_conv_bytes,
                    cudaMemcpyHostToDevice,
                    stream
                ));
                checked(cudaEventRecord(events.ev[0], stream));
                ev_busy[0] = true;

                if (!valid_tokens) {
                    checked(cudaStreamSynchronize(stream));
                    return;
                }

                // 2. Double-buffered pinned H2D + FP16 conversion for all 48 KV mirrors.
                unsigned step = 1;  // start on half 1 while half 0 DMAs conv_state!
                constexpr std::size_t max_rows_per_half = half_floats / G::n_embd_k_gqa;
                auto wait_half = [&](unsigned p) {
                    if (ev_busy[p]) {
                        checked(cudaEventSynchronize(events.ev[p]));
                        ev_busy[p] = false;
                    }
                };
                auto launch_upload = [&](unsigned p, __half* d_dst, std::size_t n_floats) {
                    checked(cudaMemcpyAsync(
                        dev_half[p],
                        pinned_half[p],
                        n_floats * sizeof(float),
                        cudaMemcpyHostToDevice,
                        stream
                    ));
                    inkling_f32_to_fp16_kernel<<<
                        unsigned((n_floats + 255) / 256), 256, 0, stream>>>(
                        dev_half[p], d_dst, n_floats
                    );
                    checked(cudaGetLastError());
                    checked(cudaEventRecord(events.ev[p], stream));
                    ev_busy[p] = true;
                };

                for (std::size_t s = 0; s < G::n_slots; ++s) {
                    stop(cancel, failed);
                    auto& m = arena.mirrors.at(s);
                    auto* m_keys = reinterpret_cast<__half*>(m.data);
                    auto* m_vals = m_keys + m.capacity * G::n_embd_k_gqa;
                    const bool is_swa = G::is_swa_phys(G::slot_phys(s));
                    if (is_swa) {
                        const std::size_t swa_win = G::slot_kv_capacity(s, desc_.context);
                        const std::size_t ring_cap = 2 * swa_win;
                        const std::size_t win_start =
                            valid_tokens > swa_win ? valid_tokens - swa_win : 0;
                        const std::size_t count = valid_tokens - win_start;
                        const std::size_t dst_start = win_start % ring_cap;
                        const std::size_t part1 = std::min(count, ring_cap - dst_start);
                        const std::size_t part2 = count - part1;

                        auto upload_swa_vec = [&](const float* h_src, __half* d_ring) {
                            const unsigned p = step++ & 1u;
                            wait_half(p);
                            float* dst_p = pinned_half[p];
                            for (std::size_t i = 0; i < count; ++i) {
                                const std::size_t src_slot = (win_start + i) % swa_win;
                                std::memcpy(
                                    dst_p + i * G::n_embd_k_gqa,
                                    h_src + src_slot * G::n_embd_k_gqa,
                                    G::n_embd_k_gqa * sizeof(float)
                                );
                            }
                            const std::size_t n_floats = count * G::n_embd_k_gqa;
                            checked(cudaMemcpyAsync(
                                dev_half[p],
                                dst_p,
                                n_floats * sizeof(float),
                                cudaMemcpyHostToDevice,
                                stream
                            ));
                            const std::size_t n1 = part1 * G::n_embd_k_gqa;
                            inkling_f32_to_fp16_kernel<<<
                                unsigned((n1 + 255) / 256), 256, 0, stream>>>(
                                dev_half[p], d_ring + dst_start * G::n_embd_k_gqa, n1
                            );
                            checked(cudaGetLastError());
                            if (part2 > 0) {
                                const std::size_t n2 = part2 * G::n_embd_k_gqa;
                                inkling_f32_to_fp16_kernel<<<
                                    unsigned((n2 + 255) / 256), 256, 0, stream>>>(
                                    dev_half[p] + n1, d_ring, n2
                                );
                                checked(cudaGetLastError());
                            }
                            checked(cudaEventRecord(events.ev[p], stream));
                            ev_busy[p] = true;
                        };
                        upload_swa_vec(slots[s].attention.keys.data(), m_keys);
                        upload_swa_vec(slots[s].attention.values.data(), m_vals);
                    } else {
                        auto upload_full_vec = [&](const float* h_src, __half* d_dst) {
                            for (std::size_t off = 0; off < valid_tokens;
                                 off += max_rows_per_half) {
                                stop(cancel, failed);
                                const std::size_t chunk_rows =
                                    std::min(max_rows_per_half, valid_tokens - off);
                                const std::size_t n_floats = chunk_rows * G::n_embd_k_gqa;
                                const unsigned p = step++ & 1u;
                                wait_half(p);
                                std::memcpy(
                                    pinned_half[p],
                                    h_src + off * G::n_embd_k_gqa,
                                    n_floats * sizeof(float)
                                );
                                launch_upload(p, d_dst + off * G::n_embd_k_gqa, n_floats);
                            }
                        };
                        upload_full_vec(slots[s].attention.keys.data(), m_keys);
                        upload_full_vec(slots[s].attention.values.data(), m_vals);
                    }
                }
                checked(cudaStreamSynchronize(stream));
            }
        );
    }

    bool forward(
        std::span<const float> input_embeddings,
        std::size_t rows,
        std::size_t position,
        bool want_logits,
        std::vector<float>& logits_out,
        const std::function<void(const char*, std::size_t)>& progress,
        const std::atomic_bool& cancel,
        unsigned* argmax_out = nullptr
    ) override {
        require(!cancel.load(), "h3_cuda_cancelled");
        require(
            rows > 0 && rows <= 512
                && input_embeddings.size() == rows * G::n_embd
                && position + rows <= desc_.context,
            "inkling_gpu_forward_shape"
        );
        for (float v : input_embeddings)
            require(std::isfinite(v), "h3_cuda_nonfinite_input");

        const bool want_argmax = (argmax_out != nullptr);
        const bool need_head = want_logits || want_argmax;

        if (want_logits)
            logits_out.resize(G::n_vocab_unpadded);
        else
            logits_out.clear();

        bool completed = false;
        ArgmaxResult host_argmax{0u, 0u};
        run_on_state_(
            cancel,
            [&](DeviceWorkspace& ws, const std::atomic_bool& failed) {
                auto& arena =
                    static_cast<CudaDeviceOperations&>(ws.operations());
                if (!ensure_weights_and_params(ws, cancel, failed))
                    return;
                auto* conv_rec = ensure_conv_state(arena);
                auto* step_rec = ensure_step_params(arena);
                if (!conv_rec || !step_rec)
                    return;

                // Ensure all 48 FP16 KV mirrors have capacity for position + rows.
                std::size_t full_cap = 0;
                for (std::size_t s = 0; s < G::n_slots; ++s) {
                    const bool is_swa = G::is_swa_phys(G::slot_phys(s));
                    const std::size_t swa_win = G::slot_kv_capacity(s, desc_.context);
                    const std::size_t need_pos =
                        is_swa ? 2 * swa_win : std::min(desc_.context, position + rows);
                    const std::size_t old_cap =
                        arena.mirrors.count(s) ? arena.mirrors[s].capacity : 0;
                    if (old_cap < need_pos)
                        arena.sync_inkling_stream();
                    auto* m = arena.mirror(s, need_pos, G::n_embd_k_gqa, true);
                    if (!m)
                        return;
                    if (m->capacity != old_cap)
                        arena.invalidate_inkling_graph();
                    if (!is_swa)
                        full_cap = std::max(full_cap, m->capacity);
                }

                const auto dry = ScratchViews::plan(rows, arena, false);
                if (dry.total_bytes > arena.scratch_limit)
                    return;
                ws.ensure(dry.total_bytes);
                const auto sv = ScratchViews::plan(rows, arena, true);

                cudaStream_t stream = arena.ensure_inkling_stream();
                // Synchronize legacy stream once so any mirror/weight copy is visible.
                checked(cuda_trace::synchronize(cudaStreamLegacy));

                if (rows == 1 && need_head) {
                    // Stage 3: CUDA Graph capture & replay for single-token decode!
                    const unsigned full_segs =
                        unsigned((std::max<std::size_t>(full_cap, 256) + 255) / 256);
                    const std::size_t partial_need =
                        std::size_t(G::n_heads) * full_segs * (G::head_dim + 2) * sizeof(float);
                    require(partial_need <= kAttnPartialBytes, "inkling_gpu_partial_limit");

                    const CudaDeviceOperations::InklingGraphKey key{
                        arena.data, arena.inkling_kv_gen, full_cap, want_argmax
                    };
                    if (!arena.inkling_graph || arena.inkling_graph_key != key) {
                        arena.invalidate_inkling_graph();
                        cudaGraph_t graph = nullptr;
                        checked(cudaStreamBeginCapture(
                            stream, cudaStreamCaptureModeThreadLocal
                        ));
                        try {
                            run_graph_body_decode(
                                arena,
                                sv,
                                conv_rec->data,
                                reinterpret_cast<const StepParams*>(step_rec->data),
                                full_segs,
                                want_argmax,
                                stream
                            );
                            checked(cudaStreamEndCapture(stream, &graph));
                            checked(cudaGraphInstantiate(
                                &arena.inkling_graph, graph, nullptr, nullptr, 0
                            ));
                            cudaGraphDestroy(graph);
                            arena.inkling_graph_key = key;
                        } catch (...) {
                            cudaGraph_t discard = nullptr;
                            cudaStreamEndCapture(stream, &discard);
                            if (discard)
                                cudaGraphDestroy(discard);
                            if (graph)
                                cudaGraphDestroy(graph);
                            arena.invalidate_inkling_graph();
                            throw;
                        }
                    }
                    stop(cancel, failed);
                    const StepParams hp{unsigned(position), {0, 0, 0}};
                    checked(cudaMemcpyAsync(
                        sv.residual[0],
                        input_embeddings.data(),
                        G::n_embd * sizeof(float),
                        cudaMemcpyHostToDevice,
                        stream
                    ));
                    checked(cudaMemcpyAsync(
                        step_rec->data,
                        &hp,
                        sizeof(hp),
                        cudaMemcpyHostToDevice,
                        stream
                    ));
                    checked(cudaGraphLaunch(arena.inkling_graph, stream));
                    if (want_argmax) {
                        checked(cudaMemcpyAsync(
                            &host_argmax,
                            sv.argmax_out,
                            sizeof(host_argmax),
                            cudaMemcpyDeviceToHost,
                            stream
                        ));
                    }
                    if (want_logits) {
                        checked(cudaMemcpyAsync(
                            logits_out.data(),
                            sv.logits,
                            G::n_vocab_unpadded * sizeof(float),
                            cudaMemcpyDeviceToHost,
                            stream
                        ));
                    }
                    checked(cudaStreamSynchronize(stream));
                    for (std::size_t s = 0; s < G::n_slots; ++s)
                        arena.mirrors[s].valid = position + 1;
                    if (progress) {
                        for (std::size_t v = 1; v <= G::n_slots; ++v)
                            progress("layer", v);
                    }
                    completed = true;
                    return;
                }

                // Multi-row prefill (or 1-row without head): eager GPU-resident path
                // using FP16 Tensor Core GEMM (m >= 16) + dp4a (m < 16).
                struct BlasScope {
                    cublasHandle_t h;
                    cublasMath_t prev_mode;
                    BlasScope(cublasHandle_t handle, cudaStream_t s, void* ws_ptr)
                        : h(handle), prev_mode(CUBLAS_PEDANTIC_MATH) {
                        checked(cublasGetMathMode(h, &prev_mode));
                        checked(cublasSetStream(h, s));
                        checked(cublasSetWorkspace(h, ws_ptr, kBlasWorkspaceBytes));
                        checked(cublasSetMathMode(h, CUBLAS_DEFAULT_MATH));
                    }
                    ~BlasScope() {
                        cublasSetMathMode(h, prev_mode);
                        cublasSetStream(h, cudaStreamLegacy);
                    }
                } blas_scope(arena.blas, stream, sv.blas_ws);

                checked(cudaMemcpyAsync(
                    sv.residual[0],
                    input_embeddings.data(),
                    rows * G::n_embd * sizeof(float),
                    cudaMemcpyHostToDevice,
                    stream
                ));

                run_eager_forward(
                    arena,
                    sv,
                    conv_rec->data,
                    rows,
                    position,
                    need_head,
                    want_argmax,
                    progress,
                    cancel,
                    failed,
                    stream
                );

                if (want_argmax) {
                    checked(cudaMemcpyAsync(
                        &host_argmax,
                        sv.argmax_out,
                        sizeof(host_argmax),
                        cudaMemcpyDeviceToHost,
                        stream
                    ));
                }
                if (want_logits) {
                    checked(cudaMemcpyAsync(
                        logits_out.data(),
                        sv.logits,
                        G::n_vocab_unpadded * sizeof(float),
                        cudaMemcpyDeviceToHost,
                        stream
                    ));
                }
                checked(cudaStreamSynchronize(stream));
                for (std::size_t s = 0; s < G::n_slots; ++s)
                    arena.mirrors[s].valid = position + rows;
                completed = true;
            }
        );

        if (completed) {
            if (want_argmax) {
                require(!host_argmax.nonfinite, "h3_cuda_nonfinite_output");
                *argmax_out = host_argmax.token;
            }
            if (want_logits && !want_argmax) {
                for (float v : logits_out)
                    require(std::isfinite(v), "h3_cuda_nonfinite_output");
            }
        }
        return completed;
    }

private:
    struct HostPhysParams {
        bool is_swa = false;
        bool has_attn_res = false;
        bool has_exp_bias = false;
        std::vector<float> attn_norm, attn_q_inp_norm, attn_k_inp_norm, attn_v_inp_norm;
        std::vector<float> attn_q_clip, attn_rel_bias, attn_rel_clip;
        std::array<std::vector<float>, G::n_conv> attn_k_conv, attn_v_conv, attn_post_conv,
            ffn_post_conv;
        std::vector<float> attn_res_query, ffn_res_query, ffn_norm, ffn_exp_probs_b;
    };

    struct HostLoopParams {
        std::vector<float> msg_query;
    };

    StateRunner run_on_state_;
    kadan::InklingGpuModelDesc desc_;
    std::vector<float> out_norm_, out_res_query_;
    std::vector<HostPhysParams> host_layers_;
    std::vector<HostLoopParams> host_loops_;

    // GPU-resolved handles (valid while kInklingParamsKey is present in arena.rel_biases)
    const float* d_out_norm_ = nullptr;
    const float* d_out_res_query_ = nullptr;
    std::array<PhysLayerGpu, G::n_phys> gpu_layers_{};
    std::array<LoopLayerGpu, G::loop_core_len> gpu_loops_{};
    WeightKind output_kind_ = kWKindQ8_0;
    std::size_t output_qs_row_bytes_ = 0;
    unsigned output_chunk_count_ = 0;
    const OutputChunkDesc* d_output_chunks_ = nullptr;

    // Distinct key width for SoA-repacked quantized chunks (9 for IQ4_NL, 17 for Q8_0)
    // so retained SoA chunks never collide with AoS Strata keys while preserving
    // key.count * key.width == exact retained byte count.
    static DeviceWeights::Key matrix_key(
        const kadan::InklingMatrixDesc& m,
        std::size_t first_row,
        std::size_t row_count
    ) {
        if (!m.encoding) {
            return DeviceWeights::Key{
                m.identity, first_row * m.in, row_count * m.in, sizeof(float)
            };
        }
        const std::size_t values = gguf::block_values(*m.encoding);
        const std::size_t block = gguf::block_bytes(*m.encoding);
        // block is 18 (IQ4_NL) or 34 (Q8_0), both even -> half width (9 / 17) with 2x count.
        return DeviceWeights::Key{
            m.identity,
            first_row * (m.in / values) * 2u,
            (row_count * m.in / values) * 2u,
            block / 2u
        };
    }

    // Admits a single matrix chunk (<= 16 MiB) into arena.weights with GPU finiteness
    // validation and (for IQ4_NL / Q8_0) on-GPU repacking into 16B/32B-aligned SoA layout.
    const unsigned char* admit_chunk(
        DeviceWorkspace& ws,
        CudaDeviceOperations& arena,
        const kadan::InklingMatrixDesc& m,
        std::size_t first_row,
        std::size_t row_count,
        WeightKind kind,
        const std::atomic_bool& cancel,
        const std::atomic_bool& failed
    ) {
        stop(cancel, failed);
        const auto key = matrix_key(m, first_row, row_count);
        if (auto* hit = arena.weights.find(key, true))
            return static_cast<const unsigned char*>(hit);

        const std::size_t bytes = row_count * m.row_bytes;
        require(bytes <= gguf_projection_staging_bytes, "inkling_gpu_chunk_staging");
        auto host_buf = ws.host_staging(bytes);
        if (row_count >= 4 && bytes >= PackedReadTeam::parallel_threshold)
            ws.packed_read_team().read(m.source, first_row, m.row_bytes, host_buf);
        else
            m.source(first_row, host_buf);
        stop(cancel, failed);

        Layout l;
        const auto off_status = l.add(sizeof(unsigned));
        const auto off_staged = l.add(bytes);
        const auto off_repacked = (kind == kWKindF32) ? off_staged : l.add(bytes);
        ws.ensure(l.bytes);
        checked(cudaMemsetAsync(
            arena.at<void>(off_status), 0, sizeof(unsigned), cudaStreamLegacy
        ));
        checked(cuda_trace::copy(
            arena.at<void>(off_staged), host_buf.data(), bytes, cudaMemcpyHostToDevice
        ));
        const std::size_t elem_count =
            (kind == kWKindF32) ? (row_count * m.in) : ((row_count * m.in) / 32u);
        unsigned char* dst_qs =
            (kind == kWKindF32) ? nullptr : arena.at<unsigned char>(off_repacked);
        __half* dst_scales =
            (kind == kWKindF32)
                ? nullptr
                : reinterpret_cast<__half*>(
                      dst_qs + soa_scales_offset(row_count, m.in, kind)
                  );
        inkling_repack_soa_validate_kernel<<<unsigned((elem_count + 255) / 256), 256>>>(
            arena.at<unsigned char>(off_staged),
            dst_qs,
            dst_scales,
            elem_count,
            int(kind),
            arena.at<unsigned>(off_status)
        );
        checked(cudaGetLastError());
        unsigned invalid = 0;
        checked(cuda_trace::copy(
            &invalid, arena.at<void>(off_status), sizeof(invalid), cudaMemcpyDeviceToHost
        ));
        require(!invalid, "h3_cuda_nonfinite_weight");
        stop(cancel, failed);

        auto* retained = arena.weights.retain_device(
            key, arena.at<void>(off_repacked), bytes, cancel
        );
        return static_cast<const unsigned char*>(retained);
    }

    bool admit_matrix(
        DeviceWorkspace& ws,
        CudaDeviceOperations& arena,
        const kadan::InklingMatrixDesc& m,
        MatrixRef& dst,
        const std::atomic_bool& cancel,
        const std::atomic_bool& failed
    ) {
        const auto k = to_weight_kind(m.encoding);
        if (!k)
            return false;
        const std::size_t total_bytes = m.out * m.row_bytes;
        if (total_bytes > gguf_projection_staging_bytes)
            return false;
        const auto* ptr = admit_chunk(ws, arena, m, 0, m.out, *k, cancel, failed);
        if (!ptr)
            return false;
        const std::size_t sc_off = soa_scales_offset(m.out, m.in, *k);
        const __half* scales =
            (*k == kWKindF32) ? nullptr : reinterpret_cast<const __half*>(ptr + sc_off);
        dst = {
            ptr,
            scales,
            soa_qs_row_bytes(m.in, *k),
            unsigned(m.in),
            unsigned(m.out),
            *k
        };
        return true;
    }

    bool ensure_weights_and_params(
        DeviceWorkspace& ws,
        const std::atomic_bool& cancel,
        const std::atomic_bool& failed
    ) {
        auto& arena = static_cast<CudaDeviceOperations&>(ws.operations());
        // Fast path: if immutable params AND first weight are still resident, pointers are valid!
        if (arena.rel_biases.count(kInklingParamsKey) && d_out_norm_ && output_chunk_count_) {
            const auto first_key = matrix_key(desc_.layers[0].wq, 0, desc_.layers[0].wq.out);
            if (arena.weights.find(first_key, true) == gpu_layers_[0].wq.qs)
                return true;
        }

        // Check bank capacity up front before admitting weights.
        ws.ensure(256);
        if (desc_.total_matrix_bytes + (32ULL << 20) > arena.weights.capacity())
            return false;

        // 1. Admit every dense matrix, loop controller matrix, and routed expert.
        for (std::size_t i = 0; i < G::n_phys; ++i) {
            const auto& src = desc_.layers[i];
            auto& dst = gpu_layers_[i];
            dst.is_swa = src.is_swa;
            dst.has_attn_res = src.has_attn_res;
            dst.has_exp_bias = src.has_exp_bias;
            if (!admit_matrix(ws, arena, src.wq, dst.wq, cancel, failed)
                || !admit_matrix(ws, arena, src.wk, dst.wk, cancel, failed)
                || !admit_matrix(ws, arena, src.wv, dst.wv, cancel, failed)
                || !admit_matrix(ws, arena, src.wo, dst.wo, cancel, failed)
                || !admit_matrix(ws, arena, src.wgate, dst.wgate, cancel, failed)
                || !admit_matrix(ws, arena, src.wrel, dst.wrel, cancel, failed)
                || !admit_matrix(ws, arena, src.wsink, dst.wsink, cancel, failed)
                || !admit_matrix(ws, arena, src.wffn_router, dst.wffn_router, cancel, failed))
                return false;
            if (!src.is_swa) {
                if (!admit_matrix(ws, arena, src.wk_conv_proj, dst.wk_conv_proj, cancel, failed)
                    || !admit_matrix(ws, arena, src.wv_conv_proj, dst.wv_conv_proj, cancel, failed))
                    return false;
            }
            dst.gate_up_kind = *to_weight_kind(src.wffn_gate_up[0].encoding);
            dst.down_kind = *to_weight_kind(src.wffn_down[0].encoding);
            dst.gate_up_qs_row_bytes = soa_qs_row_bytes(G::n_embd, dst.gate_up_kind);
            dst.gate_up_scales_offset =
                soa_scales_offset(2 * G::n_ff_exp, G::n_embd, dst.gate_up_kind);
            dst.down_qs_row_bytes = soa_qs_row_bytes(G::n_ff_exp, dst.down_kind);
            dst.down_scales_offset =
                soa_scales_offset(G::n_embd, G::n_ff_exp, dst.down_kind);
            for (std::size_t e = 0; e < G::n_experts; ++e) {
                MatrixRef gu_ref, dn_ref;
                if (!admit_matrix(ws, arena, src.wffn_gate_up[e], gu_ref, cancel, failed)
                    || !admit_matrix(ws, arena, src.wffn_down[e], dn_ref, cancel, failed))
                    return false;
                dst.gate_up_ptrs[e] = gu_ref.qs;
                dst.down_ptrs[e] = dn_ref.qs;
            }
        }
        for (std::size_t i = 0; i < G::loop_core_len; ++i) {
            const auto& src = desc_.loop_layers[i];
            auto& dst = gpu_loops_[i];
            for (std::size_t c = 0; c < G::loop_ctrl_count; ++c) {
                if (!admit_matrix(ws, arena, src.ctrl_down[c], dst.ctrl_down[c], cancel, failed)
                    || !admit_matrix(ws, arena, src.ctrl_gain[c], dst.ctrl_gain[c], cancel, failed)
                    || !admit_matrix(ws, arena, src.ctrl_forget[c], dst.ctrl_forget[c], cancel, failed))
                    return false;
            }
        }

        // 2. Admit output.weight (only unpadded rows 0..200057!) in <= 16 MiB SoA chunks.
        const auto& ow = desc_.output_weight;
        output_kind_ = *to_weight_kind(ow.encoding);
        output_qs_row_bytes_ = soa_qs_row_bytes(ow.in, output_kind_);
        const std::size_t max_chunk_rows =
            std::max<std::size_t>(1, gguf_projection_staging_bytes / ow.row_bytes);
        std::vector<OutputChunkDesc> host_chunks;
        for (std::size_t first = 0; first < G::n_vocab_unpadded; first += max_chunk_rows) {
            const std::size_t count = std::min(max_chunk_rows, G::n_vocab_unpadded - first);
            const auto* ptr =
                admit_chunk(ws, arena, ow, first, count, output_kind_, cancel, failed);
            if (!ptr)
                return false;
            const __half* scales =
                (output_kind_ == kWKindF32)
                    ? nullptr
                    : reinterpret_cast<const __half*>(
                          ptr + soa_scales_offset(count, ow.in, output_kind_)
                      );
            host_chunks.push_back({ptr, scales, unsigned(first), unsigned(count)});
        }
        require(host_chunks.size() <= kMaxOutputChunks, "inkling_gpu_output_chunks");
        output_chunk_count_ = unsigned(host_chunks.size());

        // 3. Upload 24 per-layer relative-position bias tables (shared key space 0..23).
        for (std::size_t i = 0; i < G::n_phys; ++i) {
            const float* rb = arena.rel_bias(i, host_layers_[i].attn_rel_bias);
            if (!rb)
                return false;
            gpu_layers_[i].attn_rel_bias = rb;
        }

        // 4. Pack all small F32 layer params + pointer tables + output chunk descriptors
        //    into one flat float-aligned blob stored in arena.rel_biases[kInklingParamsKey].
        std::vector<float> blob;
        blob.reserve(600000);
        auto append_floats = [&](const std::vector<float>& vec) -> std::size_t {
            if (vec.empty())
                return ~std::size_t(0);
            while (blob.size() % 4 != 0)
                blob.push_back(0.0f);
            const std::size_t off = blob.size();
            blob.insert(blob.end(), vec.begin(), vec.end());
            return off;
        };
        auto append_raw = [&](const void* src_ptr, std::size_t bytes) -> std::size_t {
            while (blob.size() % 4 != 0)
                blob.push_back(0.0f);
            const std::size_t off = blob.size();
            const std::size_t n_floats = (bytes + sizeof(float) - 1) / sizeof(float);
            blob.resize(off + n_floats, 0.0f);
            std::memcpy(blob.data() + off, src_ptr, bytes);
            return off;
        };

        const auto off_out_norm = append_floats(out_norm_);
        const auto off_out_res_q = append_floats(out_res_query_);
        struct LayerOffsets {
            std::size_t attn_norm, q_norm, k_norm, v_norm, q_clip, rel_clip;
            std::size_t k_conv[G::n_conv], v_conv[G::n_conv], ap_conv[G::n_conv], fp_conv[G::n_conv];
            std::size_t attn_res_q, ffn_res_q, ffn_norm, exp_bias;
            std::size_t gu_ptrs, dn_ptrs;
        };
        std::array<LayerOffsets, G::n_phys> loffs{};
        for (std::size_t i = 0; i < G::n_phys; ++i) {
            const auto& hl = host_layers_[i];
            auto& o = loffs[i];
            o.attn_norm = append_floats(hl.attn_norm);
            o.q_norm = append_floats(hl.attn_q_inp_norm);
            o.k_norm = append_floats(hl.attn_k_inp_norm);
            o.v_norm = append_floats(hl.attn_v_inp_norm);
            o.q_clip = append_floats(hl.attn_q_clip);
            o.rel_clip = append_floats(hl.attn_rel_clip);
            for (std::size_t k = 0; k < G::n_conv; ++k) {
                o.k_conv[k] = append_floats(hl.attn_k_conv[k]);
                o.v_conv[k] = append_floats(hl.attn_v_conv[k]);
                o.ap_conv[k] = append_floats(hl.attn_post_conv[k]);
                o.fp_conv[k] = append_floats(hl.ffn_post_conv[k]);
            }
            o.attn_res_q = append_floats(hl.attn_res_query);
            o.ffn_res_q = append_floats(hl.ffn_res_query);
            o.ffn_norm = append_floats(hl.ffn_norm);
            o.exp_bias = append_floats(hl.ffn_exp_probs_b);
            o.gu_ptrs = append_raw(
                gpu_layers_[i].gate_up_ptrs.data(),
                gpu_layers_[i].gate_up_ptrs.size() * sizeof(const unsigned char*)
            );
            o.dn_ptrs = append_raw(
                gpu_layers_[i].down_ptrs.data(),
                gpu_layers_[i].down_ptrs.size() * sizeof(const unsigned char*)
            );
        }
        std::array<std::size_t, G::loop_core_len> loop_msg_offs{};
        for (std::size_t i = 0; i < G::loop_core_len; ++i)
            loop_msg_offs[i] = append_floats(host_loops_[i].msg_query);
        const auto off_chunks = append_raw(
            host_chunks.data(), host_chunks.size() * sizeof(OutputChunkDesc)
        );

        const float* base = arena.rel_bias(kInklingParamsKey, blob);
        if (!base)
            return false;
        auto ptr_at = [&](std::size_t off) -> const float* {
            return off == ~std::size_t(0) ? nullptr : (base + off);
        };
        d_out_norm_ = ptr_at(off_out_norm);
        d_out_res_query_ = ptr_at(off_out_res_q);
        for (std::size_t i = 0; i < G::n_phys; ++i) {
            auto& gl = gpu_layers_[i];
            const auto& o = loffs[i];
            gl.attn_norm = ptr_at(o.attn_norm);
            gl.attn_q_inp_norm = ptr_at(o.q_norm);
            gl.attn_k_inp_norm = ptr_at(o.k_norm);
            gl.attn_v_inp_norm = ptr_at(o.v_norm);
            gl.attn_q_clip = ptr_at(o.q_clip);
            gl.attn_rel_clip = ptr_at(o.rel_clip);
            for (std::size_t k = 0; k < G::n_conv; ++k) {
                gl.attn_k_conv[k] = ptr_at(o.k_conv[k]);
                gl.attn_v_conv[k] = ptr_at(o.v_conv[k]);
                gl.attn_post_conv[k] = ptr_at(o.ap_conv[k]);
                gl.ffn_post_conv[k] = ptr_at(o.fp_conv[k]);
            }
            gl.attn_res_query = ptr_at(o.attn_res_q);
            gl.ffn_res_query = ptr_at(o.ffn_res_q);
            gl.ffn_norm = ptr_at(o.ffn_norm);
            gl.ffn_exp_probs_b = ptr_at(o.exp_bias);
            gl.d_gate_up_ptrs =
                reinterpret_cast<const unsigned char* const*>(base + o.gu_ptrs);
            gl.d_down_ptrs =
                reinterpret_cast<const unsigned char* const*>(base + o.dn_ptrs);
        }
        for (std::size_t i = 0; i < G::loop_core_len; ++i)
            gpu_loops_[i].msg_query = ptr_at(loop_msg_offs[i]);
        d_output_chunks_ = reinterpret_cast<const OutputChunkDesc*>(base + off_chunks);
        arena.invalidate_inkling_graph();
        return true;
    }

    CudaDeviceOperations::Recurrent* ensure_conv_state(CudaDeviceOperations& arena) {
        const std::size_t floats = G::n_slots * G::n_conv_hist * G::n_conv_channels;
        if (auto* existing = arena.recurrent_state(kInklingConvKey, floats, false))
            return existing;
        arena.invalidate_inkling_graph();
        return arena.recurrent_state(kInklingConvKey, floats, true);
    }

    CudaDeviceOperations::Recurrent* ensure_step_params(CudaDeviceOperations& arena) {
        constexpr std::size_t floats = 16;  // 64 bytes
        if (auto* existing = arena.recurrent_state(kInklingStepKey, floats, false))
            return existing;
        arena.invalidate_inkling_graph();
        return arena.recurrent_state(kInklingStepKey, floats, true);
    }

    // Projects `x_f32` ([rows, m.in]) -> `out` ([rows, m.out]).
    // When rows >= 16: uses FP16 Tensor Core cublasGemmEx (expects `x_fp16` already converted).
    // When rows < 16: uses Q8 + __dp4a matvec (expects `x_q8` already quantized if !F32).
    void project_matrix(
        cublasHandle_t blas,
        const MatrixRef& m,
        const float* x_f32,
        const __half* x_fp16,
        const Q8Block* x_q8,
        __half* w_fp16,
        float* out,
        unsigned rows,
        cudaStream_t stream
    ) const {
        if (rows >= 16u) {
            launch_dequant_fp16(m, w_fp16, stream);
            const float one = 1.0f, zero = 0.0f;
            checked(cublasGemmEx(
                blas,
                CUBLAS_OP_T,
                CUBLAS_OP_N,
                int(m.out),
                int(rows),
                int(m.in),
                &one,
                w_fp16,
                CUDA_R_16F,
                int(m.in),
                x_fp16,
                CUDA_R_16F,
                int(m.in),
                &zero,
                out,
                CUDA_R_32F,
                int(m.out),
                CUBLAS_COMPUTE_32F,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP
            ));
        } else {
            launch_matvec_dp4a(m, x_q8, x_f32, out, rows, stream);
        }
    }

    // Applies cross-pass loop controls + fused RMSNorm for `count` targets (3 for attn, 1 for FFN).
    void apply_controls_gpu(
        cublasHandle_t blas,
        const ScratchViews& sv,
        const float* const* src_targets,  // [count] read buffers
        float* const* dst_targets,        // [count] write buffers (may alias src_targets)
        const float* const* messages,     // [count]
        const float* const* norm_w,       // [count]
        const LoopLayerGpu& loop,
        std::size_t ctrl_base,
        std::size_t count,
        unsigned rows,
        cudaStream_t stream
    ) const {
        for (std::size_t k = 0; k < count; ++k) {
            inkling_ctrl_pack_kernel<<<rows, 256, 0, stream>>>(
                src_targets[k], messages[k], sv.ctrl_cat
            );
            checked(cudaGetLastError());

            const auto& m_down = loop.ctrl_down[ctrl_base + k];
            if (rows >= 16u) {
                const std::size_t n = std::size_t(rows) * 2 * G::n_embd;
                inkling_f32_to_fp16_kernel<<<unsigned((n + 255) / 256), 256, 0, stream>>>(
                    sv.ctrl_cat, sv.in_fp16, n
                );
                checked(cudaGetLastError());
            } else if (m_down.kind != kWKindF32) {
                launch_quantize_q8(
                    sv.ctrl_cat, sv.x_q8, std::size_t(rows) * 2 * G::n_embd, stream
                );
            }
            project_matrix(
                blas,
                m_down,
                sv.ctrl_cat,
                sv.in_fp16,
                sv.x_q8,
                sv.w_fp16,
                sv.ctrl_gate,
                rows,
                stream
            );

            const auto& m_gain = loop.ctrl_gain[ctrl_base + k];
            const auto& m_forget = loop.ctrl_forget[ctrl_base + k];
            if (rows >= 16u) {
                const std::size_t n = std::size_t(rows) * G::loop_gate_rank;
                inkling_f32_to_fp16_kernel<<<unsigned((n + 255) / 256), 256, 0, stream>>>(
                    sv.ctrl_gate, sv.in_fp16, n
                );
                checked(cudaGetLastError());
            } else if (m_gain.kind != kWKindF32 || m_forget.kind != kWKindF32) {
                launch_quantize_q8(
                    sv.ctrl_gate, sv.x_q8, std::size_t(rows) * G::loop_gate_rank, stream
                );
            }
            if (rows == 1u) {
                launch_fused_decode_proj(
                    {{&m_gain, sv.ctrl_g}, {&m_forget, sv.ctrl_f}},
                    sv.x_q8,
                    sv.ctrl_gate,
                    stream
                );
            } else {
                project_matrix(
                    blas,
                    m_gain,
                    sv.ctrl_gate,
                    sv.in_fp16,
                    sv.x_q8,
                    sv.w_fp16,
                    sv.ctrl_g,
                    rows,
                    stream
                );
                project_matrix(
                    blas,
                    m_forget,
                    sv.ctrl_gate,
                    sv.in_fp16,
                    sv.x_q8,
                    sv.w_fp16,
                    sv.ctrl_f,
                    rows,
                    stream
                );
            }

            inkling_ctrl_blend_norm_kernel<<<rows, 256, 0, stream>>>(
                src_targets[k], messages[k], sv.ctrl_g, sv.ctrl_f, norm_w[k], dst_targets[k]
            );
            checked(cudaGetLastError());
        }
    }

    // Launches one logical slot (either graph-captured decode or eager prefill).
    // Returns updated `cur` pointer (alternating between residual[0] and residual[1]).
    float* run_layer_gpu(
        CudaDeviceOperations& arena,
        const ScratchViews& sv,
        float* conv_base,
        std::size_t slot,
        std::size_t ip,
        float* cur_in,
        float* cur_alt,
        float* blocks[4],
        unsigned& n_blocks,
        bool has_messages,
        const LoopLayerGpu* loop,
        const StepParams* d_step_params,  // non-null in decode graph mode
        unsigned base_position,
        unsigned rows,
        unsigned graph_full_segs,
        cudaStream_t stream
    ) const {
        const auto& l = gpu_layers_[ip];
        float* slot_conv = conv_base + slot * (G::n_conv_hist * G::n_conv_channels);
        const bool shared_swa = l.is_swa && !has_messages;

        // 1. Attention input mixing + optional loop controls + RMSNorm
        const float* in_q = nullptr;
        const float* in_k = nullptr;
        const float* in_v = nullptr;
        if (!l.has_attn_res) {
            // ip == 0: SWA, no attn_res, no messages -> direct rms_norm(cur_in, attn_norm)
            inkling_rms_norm_2304_kernel<<<rows, 256, 0, stream>>>(
                cur_in, l.attn_norm, sv.inputs[0]
            );
            checked(cudaGetLastError());
            in_q = in_k = in_v = sv.inputs[0];
        } else if (!has_messages) {
            MixStatesArgs margs{};
            for (unsigned b = 0; b < n_blocks; ++b)
                margs.states[b] = blocks[b];
            margs.states[n_blocks] = cur_in;
            margs.n_states = n_blocks + 1u;
            margs.query = l.attn_res_query;
            if (shared_swa) {
                margs.n_queries = 1u;
                margs.outputs[0] = sv.inputs[0];
                margs.norm_w[0] = l.attn_norm;
                in_q = in_k = in_v = sv.inputs[0];
            } else {
                margs.n_queries = 3u;
                margs.outputs[0] = sv.inputs[0];
                margs.outputs[1] = sv.inputs[1];
                margs.outputs[2] = sv.inputs[2];
                margs.norm_w[0] = l.attn_q_inp_norm;
                margs.norm_w[1] = l.attn_k_inp_norm;
                margs.norm_w[2] = l.attn_v_inp_norm;
                in_q = sv.inputs[0];
                in_k = sv.inputs[1];
                in_v = sv.inputs[2];
            }
            inkling_mix_states_norm_kernel<<<rows, 256, 0, stream>>>(margs);
            checked(cudaGetLastError());
        } else {
            // Cross-pass loop controls active: mix without norm, then ctrl_blend + fused norm.
            MixStatesArgs margs{};
            for (unsigned b = 0; b < n_blocks; ++b)
                margs.states[b] = blocks[b];
            margs.states[n_blocks] = cur_in;
            margs.n_states = n_blocks + 1u;
            margs.query = l.attn_res_query;
            if (l.is_swa) {
                margs.n_queries = 1u;
                margs.outputs[0] = sv.inputs[0];
                inkling_mix_states_norm_kernel<<<rows, 256, 0, stream>>>(margs);
                checked(cudaGetLastError());
                // Broadcast inputs[0] into inputs[1] and inputs[2] by passing inputs[0] as src
                // for k=2, k=1, k=0 in reverse order so inputs[0] isn't clobbered before k=1,2 read it!
                const float* srcs[3] = {sv.inputs[0], sv.inputs[0], sv.inputs[0]};
                float* dsts[3] = {sv.inputs[2], sv.inputs[1], sv.inputs[0]};
                const float* msgs[3] = {sv.messages[2], sv.messages[1], sv.messages[0]};
                const float* nws[3] = {l.attn_norm, l.attn_norm, l.attn_norm};
                // Apply ctrl 2, 1, 0 so sv.inputs[0] is overwritten last!
                for (int idx = 0; idx < 3; ++idx) {
                    const std::size_t ctrl_idx = std::size_t(2 - idx);
                    const float* s1[1] = {srcs[idx]};
                    float* d1[1] = {dsts[idx]};
                    const float* m1[1] = {msgs[idx]};
                    const float* n1[1] = {nws[idx]};
                    apply_controls_gpu(
                        arena.blas, sv, s1, d1, m1, n1, *loop, ctrl_idx, 1, rows, stream
                    );
                }
            } else {
                margs.n_queries = 3u;
                margs.outputs[0] = sv.inputs[0];
                margs.outputs[1] = sv.inputs[1];
                margs.outputs[2] = sv.inputs[2];
                inkling_mix_states_norm_kernel<<<rows, 256, 0, stream>>>(margs);
                checked(cudaGetLastError());
                const float* srcs[3] = {sv.inputs[0], sv.inputs[1], sv.inputs[2]};
                float* dsts[3] = {sv.inputs[0], sv.inputs[1], sv.inputs[2]};
                const float* msgs[3] = {sv.messages[0], sv.messages[1], sv.messages[2]};
                const float* nws[3] = {
                    l.attn_q_inp_norm, l.attn_k_inp_norm, l.attn_v_inp_norm
                };
                apply_controls_gpu(
                    arena.blas, sv, srcs, dsts, msgs, nws, *loop, 0, 3, rows, stream
                );
            }
            in_q = sv.inputs[0];
            in_k = sv.inputs[1];
            in_v = sv.inputs[2];
        }

        // 2. Q, K, V, R, sink, gate (+ dynamic conv taps on full-attn layers)
        auto prep_input = [&](const float* x_f32) {
            if (rows >= 16u) {
                const std::size_t n = std::size_t(rows) * G::n_embd;
                inkling_f32_to_fp16_kernel<<<unsigned((n + 255) / 256), 256, 0, stream>>>(
                    x_f32, sv.in_fp16, n
                );
                checked(cudaGetLastError());
            } else {
                launch_quantize_q8(x_f32, sv.x_q8, std::size_t(rows) * G::n_embd, stream);
            }
        };

        prep_input(in_q);
        if (rows == 1u) {
            if (shared_swa) {
                launch_fused_decode_proj(
                    {{&l.wq, sv.q},
                     {&l.wrel, sv.r_cur},
                     {&l.wsink, sv.sinks},
                     {&l.wgate, sv.gate},
                     {&l.wk, sv.k_raw},
                     {&l.wv, sv.v_raw}},
                    sv.x_q8,
                    in_q,
                    stream
                );
            } else {
                launch_fused_decode_proj(
                    {{&l.wq, sv.q},
                     {&l.wrel, sv.r_cur},
                     {&l.wsink, sv.sinks},
                     {&l.wgate, sv.gate}},
                    sv.x_q8,
                    in_q,
                    stream
                );
                prep_input(in_k);
                if (!l.is_swa) {
                    launch_fused_decode_proj(
                        {{&l.wk, sv.k_raw}, {&l.wk_conv_proj, sv.k_dyn}},
                        sv.x_q8,
                        in_k,
                        stream
                    );
                } else {
                    project_matrix(
                        arena.blas, l.wk, in_k, sv.in_fp16, sv.x_q8, sv.w_fp16, sv.k_raw, rows, stream
                    );
                }
                prep_input(in_v);
                if (!l.is_swa) {
                    launch_fused_decode_proj(
                        {{&l.wv, sv.v_raw}, {&l.wv_conv_proj, sv.v_dyn}},
                        sv.x_q8,
                        in_v,
                        stream
                    );
                } else {
                    project_matrix(
                        arena.blas, l.wv, in_v, sv.in_fp16, sv.x_q8, sv.w_fp16, sv.v_raw, rows, stream
                    );
                }
            }
        } else {
            project_matrix(arena.blas, l.wq, in_q, sv.in_fp16, sv.x_q8, sv.w_fp16, sv.q, rows, stream);
            project_matrix(arena.blas, l.wrel, in_q, sv.in_fp16, sv.x_q8, sv.w_fp16, sv.r_cur, rows, stream);
            project_matrix(arena.blas, l.wsink, in_q, sv.in_fp16, sv.x_q8, sv.w_fp16, sv.sinks, rows, stream);
            project_matrix(arena.blas, l.wgate, in_q, sv.in_fp16, sv.x_q8, sv.w_fp16, sv.gate, rows, stream);
            if (shared_swa) {
                project_matrix(arena.blas, l.wk, in_k, sv.in_fp16, sv.x_q8, sv.w_fp16, sv.k_raw, rows, stream);
                project_matrix(arena.blas, l.wv, in_v, sv.in_fp16, sv.x_q8, sv.w_fp16, sv.v_raw, rows, stream);
            } else {
                prep_input(in_k);
                project_matrix(arena.blas, l.wk, in_k, sv.in_fp16, sv.x_q8, sv.w_fp16, sv.k_raw, rows, stream);
                if (!l.is_swa)
                    project_matrix(
                        arena.blas, l.wk_conv_proj, in_k, sv.in_fp16, sv.x_q8, sv.w_fp16, sv.k_dyn, rows, stream
                    );
                prep_input(in_v);
                project_matrix(arena.blas, l.wv, in_v, sv.in_fp16, sv.x_q8, sv.w_fp16, sv.v_raw, rows, stream);
                if (!l.is_swa)
                    project_matrix(
                        arena.blas, l.wv_conv_proj, in_v, sv.in_fp16, sv.x_q8, sv.w_fp16, sv.v_dyn, rows, stream
                    );
            }
        }

        // 3. K and V causal short convolutions
        const float kv_conv_scale =
            l.is_swa ? (1.0f / std::sqrt(5.0f)) : (1.0f / 3.0f);
        auto run_conv = [&](const float* src,
                            const float* const w[G::n_conv],
                            const float* dyn,
                            const float* residual,
                            std::size_t ch_offset,
                            unsigned channels,
                            float* dst,
                            float scale) {
            float* cstate = slot_conv + ch_offset * G::n_conv_hist;
            const unsigned grid_x = (channels + 255u) / 256u;
            const unsigned hw = dyn ? (channels / unsigned(G::n_kv_heads)) : 0u;
            if (rows == 1u) {
                inkling_shortconv_single_kernel<<<grid_x, 256, 0, stream>>>(
                    src, w[0], w[1], w[2], w[3], dyn, residual, cstate, dst, channels, hw, scale
                );
                checked(cudaGetLastError());
            } else {
                inkling_shortconv_multi_kernel<<<dim3(grid_x, rows), 256, 0, stream>>>(
                    src,
                    w[0],
                    w[1],
                    w[2],
                    w[3],
                    dyn,
                    residual,
                    cstate,
                    dst,
                    rows,
                    channels,
                    dyn ? unsigned(G::n_kv_heads) : 0u,
                    hw,
                    scale
                );
                checked(cudaGetLastError());
                inkling_shortconv_commit_kernel<<<grid_x, 256, 0, stream>>>(
                    src, cstate, rows, channels
                );
                checked(cudaGetLastError());
            }
        };
        run_conv(
            sv.k_raw,
            l.attn_k_conv,
            l.is_swa ? nullptr : sv.k_dyn,
            nullptr,
            0,
            unsigned(G::n_embd_k_gqa),
            sv.k,
            kv_conv_scale
        );
        run_conv(
            sv.v_raw,
            l.attn_v_conv,
            l.is_swa ? nullptr : sv.v_dyn,
            nullptr,
            G::n_embd_k_gqa,
            unsigned(G::n_embd_k_gqa),
            sv.v,
            kv_conv_scale
        );

        // 4. Fused Q/K/R clip + V head RMSNorm + tau + FP16 KV mirror store
        auto& mirror = arena.mirrors.at(slot);
        auto* m_keys = reinterpret_cast<__half*>(mirror.data);
        auto* m_vals = m_keys + mirror.capacity * G::n_embd_k_gqa;
        const std::size_t swa_win = l.is_swa ? G::slot_kv_capacity(slot, desc_.context) : 0;
        const unsigned ring_cap = l.is_swa ? unsigned(2 * swa_win) : 0u;
        const unsigned ring_mask =
            (ring_cap && (ring_cap & (ring_cap - 1u)) == 0u) ? (ring_cap - 1u) : 0u;
        inkling_qkvr_prep_kv_store_kernel<<<dim3(rows, 64), 128, 0, stream>>>(
            sv.q,
            sv.r_cur,
            sv.k,
            sv.v,
            l.attn_q_clip,
            l.attn_rel_clip,
            m_keys,
            m_vals,
            d_step_params,
            base_position,
            ring_cap,
            ring_mask,
            l.is_swa
        );
        checked(cudaGetLastError());

        // 5. Fused Inkling FP16 attention + virtual sink + sigmoid output gate
        const float qk_scale = 1.0f / std::sqrt(float(G::head_dim));
        if (d_step_params) {
            const unsigned segs = l.is_swa ? unsigned((swa_win + 255) / 256) : graph_full_segs;
            inkling_res_attention_kernel<<<dim3(1, G::n_heads, segs), 256, 0, stream>>>(
                sv.q,
                sv.r_cur,
                l.attn_rel_bias,
                m_keys,
                m_vals,
                sv.attn_partial,
                d_step_params,
                0u,
                unsigned(swa_win),
                ring_cap,
                ring_mask,
                qk_scale
            );
            checked(cudaGetLastError());
            inkling_res_attention_combine_gated_kernel<<<dim3(1, G::n_heads), 128, 0, stream>>>(
                sv.attn_partial,
                sv.sinks,
                sv.gate,
                sv.attn_out,
                d_step_params,
                0u,
                unsigned(swa_win),
                segs
            );
            checked(cudaGetLastError());
        } else {
            const std::size_t positions = std::size_t(base_position) + rows;
            const std::size_t longest = l.is_swa ? std::min(positions, swa_win) : positions;
            const unsigned segs = unsigned((longest + 255) / 256);
            const std::size_t bytes_per_row =
                std::size_t(G::n_heads) * segs * (G::head_dim + 2) * sizeof(float);
            const unsigned group = unsigned(std::min<std::size_t>(
                rows, std::max<std::size_t>(1, kAttnPartialBytes / bytes_per_row)
            ));
            for (unsigned first = 0; first < rows; first += group) {
                const unsigned count = std::min(group, rows - first);
                inkling_res_attention_kernel<<<dim3(count, G::n_heads, segs), 256, 0, stream>>>(
                    sv.q + std::size_t(first) * G::n_embd_q,
                    sv.r_cur + std::size_t(first) * G::n_heads * G::n_rel,
                    l.attn_rel_bias,
                    m_keys,
                    m_vals,
                    sv.attn_partial,
                    nullptr,
                    base_position + first,
                    unsigned(swa_win),
                    ring_cap,
                    ring_mask,
                    qk_scale
                );
                checked(cudaGetLastError());
                inkling_res_attention_combine_gated_kernel<<<dim3(count, G::n_heads), 128, 0, stream>>>(
                    sv.attn_partial,
                    sv.sinks + std::size_t(first) * G::n_heads,
                    sv.gate + std::size_t(first) * G::n_embd_q,
                    sv.attn_out + std::size_t(first) * G::n_embd_q,
                    nullptr,
                    base_position + first,
                    unsigned(swa_win),
                    segs
                );
                checked(cudaGetLastError());
            }
        }

        // 6. attn_output projection -> attn_post_conv (+ fused residual add when ip % 6 != 0)
        if (rows >= 16u) {
            const std::size_t n = std::size_t(rows) * G::n_embd_q;
            inkling_f32_to_fp16_kernel<<<unsigned((n + 255) / 256), 256, 0, stream>>>(
                sv.attn_out, sv.in_fp16, n
            );
            checked(cudaGetLastError());
        } else {
            launch_quantize_q8(
                sv.attn_out, sv.x_q8, std::size_t(rows) * G::n_embd_q, stream
            );
        }
        // Reuse sv.inputs[0] for wo_out ([rows, 2304])
        project_matrix(
            arena.blas, l.wo, sv.attn_out, sv.in_fp16, sv.x_q8, sv.w_fp16, sv.inputs[0], rows, stream
        );

        const float post_conv_scale = 1.0f / std::sqrt(5.0f);
        float* ffn_inp = cur_alt;
        if (ip % G::n_block_size == 0) {
            // First layer of a 6-layer block pushes cur_in into blocks[n_blocks++]
            // and starts a fresh stream (no residual add).
            checked(cudaMemcpyAsync(
                blocks[n_blocks],
                cur_in,
                std::size_t(rows) * G::n_embd * sizeof(float),
                cudaMemcpyDeviceToDevice,
                stream
            ));
            ++n_blocks;
            run_conv(
                sv.inputs[0],
                l.attn_post_conv,
                nullptr,
                nullptr,
                2 * G::n_embd_k_gqa,
                unsigned(G::n_embd),
                ffn_inp,
                post_conv_scale
            );
        } else {
            run_conv(
                sv.inputs[0],
                l.attn_post_conv,
                nullptr,
                cur_in,
                2 * G::n_embd_k_gqa,
                unsigned(G::n_embd),
                ffn_inp,
                post_conv_scale
            );
        }

        // 7. FFN input: mix(blocks, ffn_inp, ffn_res_query) -> optional loop ctrl -> ffn_norm
        float* ffn_h = sv.inputs[0];
        {
            MixStatesArgs margs{};
            for (unsigned b = 0; b < n_blocks; ++b)
                margs.states[b] = blocks[b];
            margs.states[n_blocks] = ffn_inp;
            margs.n_states = n_blocks + 1u;
            margs.n_queries = 1u;
            margs.query = l.ffn_res_query;
            margs.outputs[0] = ffn_h;
            margs.norm_w[0] = has_messages ? nullptr : l.ffn_norm;
            inkling_mix_states_norm_kernel<<<rows, 256, 0, stream>>>(margs);
            checked(cudaGetLastError());
            if (has_messages) {
                const float* s1[1] = {ffn_h};
                float* d1[1] = {ffn_h};
                const float* m1[1] = {sv.messages[3]};
                const float* n1[1] = {l.ffn_norm};
                apply_controls_gpu(arena.blas, sv, s1, d1, m1, n1, *loop, 3, 1, rows, stream);
            }
        }

        // 8. MoE Router + Expert execution
        prep_input(ffn_h);
        project_matrix(
            arena.blas,
            l.wffn_router,
            ffn_h,
            sv.in_fp16,
            sv.x_q8,
            sv.w_fp16,
            sv.router_logits,
            rows,
            stream
        );
        inkling_moe_route_kernel<<<rows, G::n_experts, 0, stream>>>(
            sv.router_logits,
            l.has_exp_bias ? l.ffn_exp_probs_b : nullptr,
            sv.row_experts,
            sv.row_weights
        );
        checked(cudaGetLastError());

        float* moe_out = sv.inputs[1];
        if (rows == 1u) {
            // Fused indirect-GPU decode path (100% CUDA Graph compatible, zero D2H!)
            auto launch_gu = [&](auto kind_tag) {
                constexpr int K = decltype(kind_tag)::value;
                inkling_moe_decode_gate_up_situ_kernel<K>
                    <<<dim3((G::n_ff_exp + 3u) / 4u, G::n_experts_used), 128, 0, stream>>>(
                        sv.row_experts,
                        l.d_gate_up_ptrs,
                        sv.x_q8,
                        ffn_h,
                        sv.act_out,
                        l.gate_up_qs_row_bytes,
                        l.gate_up_scales_offset
                    );
                checked(cudaGetLastError());
            };
            if (l.gate_up_kind == kWKindF32)
                launch_gu(std::integral_constant<int, kWKindF32>{});
            else if (l.gate_up_kind == kWKindQ8_0)
                launch_gu(std::integral_constant<int, kWKindQ8_0>{});
            else
                launch_gu(std::integral_constant<int, kWKindIQ4_NL>{});

            if (l.down_kind != kWKindF32)
                launch_quantize_q8(
                    sv.act_out, sv.x_q8, G::n_experts_used * G::n_ff_exp, stream
                );
            auto launch_dn = [&](auto kind_tag) {
                constexpr int K = decltype(kind_tag)::value;
                inkling_moe_decode_down_reduce_kernel<K>
                    <<<(G::n_embd + 3u) / 4u, 128, 0, stream>>>(
                        sv.row_experts,
                        sv.row_weights,
                        l.d_down_ptrs,
                        sv.x_q8,
                        sv.act_out,
                        moe_out,
                        l.down_qs_row_bytes,
                        l.down_scales_offset
                    );
                checked(cudaGetLastError());
            };
            if (l.down_kind == kWKindF32)
                launch_dn(std::integral_constant<int, kWKindF32>{});
            else if (l.down_kind == kWKindQ8_0)
                launch_dn(std::integral_constant<int, kWKindQ8_0>{});
            else
                launch_dn(std::integral_constant<int, kWKindIQ4_NL>{});
        } else {
            // Prefill MoE (rows > 1): 100% async GPU-side CSR bucketing (zero host syncs!)
            // + grouped tiled quantized MMQ kernels reading SoA weights directly with zero
            // FP16 dequantization HBM traffic (2 launches per layer instead of ~512!).
            const std::size_t total_slots = std::size_t(rows) * G::n_experts_used;
            inkling_moe_bucket_gpu_kernel<<<1, 256, 0, stream>>>(
                sv.row_experts,
                unsigned(total_slots),
                sv.exp_offsets,
                sv.exp_token_idx,
                sv.exp_slot_for_rk
            );
            checked(cudaGetLastError());

            // Quantize the `rows` unique FFN inputs once (when rows < 16, prep_input already did).
            if (rows >= 16u && l.gate_up_kind != kWKindF32) {
                launch_quantize_q8(ffn_h, sv.x_q8, std::size_t(rows) * G::n_embd, stream);
            }

            // Grouped gate_up + fused SiTU across all 128 experts in one launch.
            auto launch_prefill_gu = [&](auto kind_tag) {
                constexpr int K = decltype(kind_tag)::value;
                inkling_moe_prefill_gate_up_situ_kernel<K>
                    <<<dim3(G::n_ff_exp / 4u, G::n_experts), 128, 0, stream>>>(
                        sv.exp_offsets,
                        sv.exp_token_idx,
                        l.d_gate_up_ptrs,
                        sv.x_q8,
                        ffn_h,
                        sv.act_out,
                        l.gate_up_qs_row_bytes,
                        l.gate_up_scales_offset
                    );
                checked(cudaGetLastError());
            };
            if (l.gate_up_kind == kWKindF32)
                launch_prefill_gu(std::integral_constant<int, kWKindF32>{});
            else if (l.gate_up_kind == kWKindQ8_0)
                launch_prefill_gu(std::integral_constant<int, kWKindQ8_0>{});
            else
                launch_prefill_gu(std::integral_constant<int, kWKindIQ4_NL>{});

            if (l.down_kind != kWKindF32) {
                launch_quantize_q8(
                    sv.act_out, sv.x_q8, total_slots * G::n_ff_exp, stream
                );
            }

            // Grouped down projection across all 128 experts in one launch (reuses sv.moe_in).
            auto launch_prefill_dn = [&](auto kind_tag) {
                constexpr int K = decltype(kind_tag)::value;
                inkling_moe_prefill_down_kernel<K>
                    <<<dim3(G::n_embd / 4u, G::n_experts), 128, 0, stream>>>(
                        sv.exp_offsets,
                        l.d_down_ptrs,
                        sv.x_q8,
                        sv.act_out,
                        sv.moe_in,
                        l.down_qs_row_bytes,
                        l.down_scales_offset
                    );
                checked(cudaGetLastError());
            };
            if (l.down_kind == kWKindF32)
                launch_prefill_dn(std::integral_constant<int, kWKindF32>{});
            else if (l.down_kind == kWKindQ8_0)
                launch_prefill_dn(std::integral_constant<int, kWKindQ8_0>{});
            else
                launch_prefill_dn(std::integral_constant<int, kWKindIQ4_NL>{});

            inkling_moe_scatter_kernel<<<rows, 256, 0, stream>>>(
                sv.moe_in, sv.exp_slot_for_rk, sv.row_weights, moe_out
            );
            checked(cudaGetLastError());
        }

        // 9. ffn_post_conv + fused residual add with ffn_inp -> cur_in
        run_conv(
            moe_out,
            l.ffn_post_conv,
            nullptr,
            ffn_inp,
            2 * G::n_embd_k_gqa + G::n_embd,
            unsigned(G::n_embd),
            cur_in,
            post_conv_scale
        );
        return cur_in;
    }

    void run_output_head_gpu(
        const ScratchViews& sv,
        float* const blocks[4],
        unsigned n_blocks,
        const float* cur,
        unsigned rows,
        bool run_argmax,
        cudaStream_t stream
    ) const {
        // Last row only: mix(blocks, cur, out_res_query) + fused rms_norm(out_norm)
        MixStatesArgs margs{};
        for (unsigned b = 0; b < n_blocks; ++b)
            margs.states[b] = blocks[b];
        margs.states[n_blocks] = cur;
        margs.n_states = n_blocks + 1u;
        margs.n_queries = 1u;
        margs.query = d_out_res_query_;
        margs.outputs[0] = sv.inputs[0];
        margs.norm_w[0] = d_out_norm_;
        margs.row_offset = rows - 1u;
        inkling_mix_states_norm_kernel<<<1, 256, 0, stream>>>(margs);
        checked(cudaGetLastError());

        if (output_kind_ != kWKindF32)
            launch_quantize_q8(sv.inputs[0], sv.x_q8, G::n_embd, stream);

        const unsigned grid = (unsigned(G::n_vocab_unpadded) + 3u) / 4u;
        if (output_kind_ == kWKindF32)
            inkling_output_softcap_kernel<kWKindF32><<<grid, 128, 0, stream>>>(
                d_output_chunks_,
                output_chunk_count_,
                sv.x_q8,
                sv.inputs[0],
                sv.logits,
                output_qs_row_bytes_,
                unsigned(G::n_vocab_unpadded)
            );
        else if (output_kind_ == kWKindQ8_0)
            inkling_output_softcap_kernel<kWKindQ8_0><<<grid, 128, 0, stream>>>(
                d_output_chunks_,
                output_chunk_count_,
                sv.x_q8,
                sv.inputs[0],
                sv.logits,
                output_qs_row_bytes_,
                unsigned(G::n_vocab_unpadded)
            );
        else
            inkling_output_softcap_kernel<kWKindIQ4_NL><<<grid, 128, 0, stream>>>(
                d_output_chunks_,
                output_chunk_count_,
                sv.x_q8,
                sv.inputs[0],
                sv.logits,
                output_qs_row_bytes_,
                unsigned(G::n_vocab_unpadded)
            );
        checked(cudaGetLastError());

        if (run_argmax) {
            inkling_argmax_stage1_kernel<<<256, 256, 0, stream>>>(
                sv.logits,
                unsigned(G::n_vocab_unpadded),
                sv.argmax_p_val,
                sv.argmax_p_idx,
                sv.argmax_p_bad
            );
            checked(cudaGetLastError());
            inkling_argmax_stage2_kernel<<<1, 256, 0, stream>>>(
                sv.argmax_p_val, sv.argmax_p_idx, sv.argmax_p_bad, sv.argmax_out
            );
            checked(cudaGetLastError());
        }
    }

    // Complete 48-slot decode + output head captured inside a single CUDA Graph.
    void run_graph_body_decode(
        CudaDeviceOperations& arena,
        const ScratchViews& sv,
        float* conv_base,
        const StepParams* d_step_params,
        unsigned full_segs,
        bool run_argmax,
        cudaStream_t stream
    ) const {
        float* cur = sv.residual[0];
        float* cur_alt = sv.residual[1];
        float* prelude = sv.residual[2];
        float* bank0 = sv.residual[3];
        float* bank_out = sv.residual[4];
        float* blocks[4] = {
            sv.residual[5], sv.residual[6], sv.residual[7], sv.residual[8]
        };
        unsigned n_blocks = 0;

        // Prelude: ip 0..5 (slots 0..5)
        for (std::size_t ip = 0; ip < G::loop_start; ++ip) {
            cur = run_layer_gpu(
                arena, sv, conv_base, ip, ip, cur, cur_alt, blocks, n_blocks,
                false, nullptr, d_step_params, 0u, 1u, full_segs, stream
            );
        }

        // Save prelude (blocks[0] is prelude_blocks[0] and never overwritten)
        checked(cudaMemcpyAsync(
            prelude, cur, G::n_embd * sizeof(float), cudaMemcpyDeviceToDevice, stream
        ));

        // Looped core: passes 0, 1, 2 over ip 6..17
        for (std::size_t pass = 0; pass < G::loop_passes; ++pass) {
            if (pass > 0) {
                checked(cudaMemcpyAsync(
                    bank0, blocks[2], G::n_embd * sizeof(float), cudaMemcpyDeviceToDevice, stream
                ));
                checked(cudaMemcpyAsync(
                    bank_out, cur, G::n_embd * sizeof(float), cudaMemcpyDeviceToDevice, stream
                ));
                checked(cudaMemcpyAsync(
                    cur, prelude, G::n_embd * sizeof(float), cudaMemcpyDeviceToDevice, stream
                ));
                n_blocks = 1u;  // restore prelude_blocks = [blocks[0]]
            }
            for (std::size_t ip = G::loop_start; ip < G::loop_end; ++ip) {
                const std::size_t slot =
                    pass == 0 ? ip
                              : G::n_phys + (pass - 1) * G::loop_core_len + (ip - G::loop_start);
                const LoopLayerGpu* loop = nullptr;
                if (pass > 0) {
                    loop = &gpu_loops_[ip - G::loop_start];
                    MixStatesArgs margs{};
                    margs.states[0] = bank0;
                    margs.states[1] = bank_out;
                    margs.n_states = 2u;
                    margs.n_queries = G::loop_ctrl_count;
                    margs.query = loop->msg_query;
                    for (unsigned q = 0; q < G::loop_ctrl_count; ++q)
                        margs.outputs[q] = sv.messages[q];
                    inkling_mix_states_norm_kernel<<<1, 256, 0, stream>>>(margs);
                    checked(cudaGetLastError());
                }
                cur = run_layer_gpu(
                    arena, sv, conv_base, slot, ip, cur, cur_alt, blocks, n_blocks,
                    pass > 0, loop, d_step_params, 0u, 1u, full_segs, stream
                );
            }
        }

        // Coda: ip 18..23 (slots 18..23)
        for (std::size_t ip = G::loop_end; ip < G::n_phys; ++ip) {
            cur = run_layer_gpu(
                arena, sv, conv_base, ip, ip, cur, cur_alt, blocks, n_blocks,
                false, nullptr, d_step_params, 0u, 1u, full_segs, stream
            );
        }

        run_output_head_gpu(sv, blocks, n_blocks, cur, 1u, run_argmax, stream);
    }

    // Complete 48-slot eager prefill (or 1-row without logits).
    void run_eager_forward(
        CudaDeviceOperations& arena,
        const ScratchViews& sv,
        float* conv_base,
        std::size_t rows,
        std::size_t position,
        bool need_head,
        bool run_argmax,
        const std::function<void(const char*, std::size_t)>& progress,
        const std::atomic_bool& cancel,
        const std::atomic_bool& failed,
        cudaStream_t stream
    ) const {
        float* cur = sv.residual[0];
        float* cur_alt = sv.residual[1];
        float* prelude = sv.residual[2];
        float* bank0 = sv.residual[3];
        float* bank_out = sv.residual[4];
        float* blocks[4] = {
            sv.residual[5], sv.residual[6], sv.residual[7], sv.residual[8]
        };
        unsigned n_blocks = 0;
        std::size_t visit = 0;
        const std::size_t row_bytes = rows * G::n_embd * sizeof(float);

        // Prelude: ip 0..5
        for (std::size_t ip = 0; ip < G::loop_start; ++ip) {
            stop(cancel, failed);
            cur = run_layer_gpu(
                arena, sv, conv_base, ip, ip, cur, cur_alt, blocks, n_blocks,
                false, nullptr, nullptr, unsigned(position), unsigned(rows), 0u, stream
            );
            if (progress)
                progress("layer", ++visit);
        }

        checked(cudaMemcpyAsync(prelude, cur, row_bytes, cudaMemcpyDeviceToDevice, stream));

        // Looped core: passes 0, 1, 2
        for (std::size_t pass = 0; pass < G::loop_passes; ++pass) {
            if (pass > 0) {
                checked(cudaMemcpyAsync(bank0, blocks[2], row_bytes, cudaMemcpyDeviceToDevice, stream));
                checked(cudaMemcpyAsync(bank_out, cur, row_bytes, cudaMemcpyDeviceToDevice, stream));
                checked(cudaMemcpyAsync(cur, prelude, row_bytes, cudaMemcpyDeviceToDevice, stream));
                n_blocks = 1u;
            }
            for (std::size_t ip = G::loop_start; ip < G::loop_end; ++ip) {
                stop(cancel, failed);
                const std::size_t slot =
                    pass == 0 ? ip
                              : G::n_phys + (pass - 1) * G::loop_core_len + (ip - G::loop_start);
                const LoopLayerGpu* loop = nullptr;
                if (pass > 0) {
                    loop = &gpu_loops_[ip - G::loop_start];
                    MixStatesArgs margs{};
                    margs.states[0] = bank0;
                    margs.states[1] = bank_out;
                    margs.n_states = 2u;
                    margs.n_queries = G::loop_ctrl_count;
                    margs.query = loop->msg_query;
                    for (unsigned q = 0; q < G::loop_ctrl_count; ++q)
                        margs.outputs[q] = sv.messages[q];
                    inkling_mix_states_norm_kernel<<<unsigned(rows), 256, 0, stream>>>(margs);
                    checked(cudaGetLastError());
                }
                cur = run_layer_gpu(
                    arena, sv, conv_base, slot, ip, cur, cur_alt, blocks, n_blocks,
                    pass > 0, loop, nullptr, unsigned(position), unsigned(rows), 0u, stream
                );
                if (progress)
                    progress("layer", ++visit);
            }
        }

        // Coda: ip 18..23
        for (std::size_t ip = G::loop_end; ip < G::n_phys; ++ip) {
            stop(cancel, failed);
            cur = run_layer_gpu(
                arena, sv, conv_base, ip, ip, cur, cur_alt, blocks, n_blocks,
                false, nullptr, nullptr, unsigned(position), unsigned(rows), 0u, stream
            );
            if (progress)
                progress("layer", ++visit);
        }

        if (need_head)
            run_output_head_gpu(sv, blocks, n_blocks, cur, unsigned(rows), run_argmax, stream);
    }
};

}  // namespace inkling_gpu_detail
