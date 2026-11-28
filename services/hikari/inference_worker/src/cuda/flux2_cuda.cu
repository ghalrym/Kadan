#include "flux2_cuda.hpp"

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>

namespace kadan::image::gpu {
namespace {

constexpr int kMaxTrackedDevices = 16;

void cuda_check(cudaError_t err, const char* context) {
    if (err != cudaSuccess)
        throw std::runtime_error(
            std::string(context) + ": " + cudaGetErrorString(err)
        );
}

void blas_check(cublasStatus_t status, const char* context) {
    if (status != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error(
            std::string(context) + ": cublas_status_" + std::to_string(int(status))
        );
}

void need(bool cond, const char* msg) {
    if (!cond)
        throw std::runtime_error(msg);
}

struct PerDeviceState {
    std::mutex mutex;
    Bytes allocated = 0;
    Bytes peak = 0;
    cublasHandle_t blas = nullptr;
    void* blas_workspace = nullptr;
    std::size_t blas_workspace_bytes = 0;
};

PerDeviceState& device_state(int device) {
    need(device >= 0 && device < kMaxTrackedDevices, "flux2_cuda_device_index");
    static std::array<PerDeviceState, kMaxTrackedDevices> states;
    return states[std::size_t(device)];
}

cublasHandle_t get_blas(int device) {
    auto& st = device_state(device);
    std::lock_guard lock(st.mutex);
    if (!st.blas) {
        DeviceGuard guard(device);
        blas_check(cublasCreate(&st.blas), "cublasCreate");
        blas_check(
            cublasSetMathMode(st.blas, CUBLAS_PEDANTIC_MATH), "cublasSetMathMode"
        );
        blas_check(cublasSetStream(st.blas, nullptr), "cublasSetStream");
    }
    return st.blas;
}

void* tracked_alloc(int device, Bytes bytes) {
    if (bytes == 0)
        return nullptr;
    DeviceGuard guard(device);
    void* ptr = nullptr;
    const cudaError_t err = cudaMalloc(&ptr, bytes);
    if (err != cudaSuccess)
        throw ResourceExhausted(
            std::string("flux2_cuda_oom: ") + cudaGetErrorString(err)
        );
    auto& st = device_state(device);
    std::lock_guard lock(st.mutex);
    st.allocated += bytes;
    if (st.allocated > st.peak)
        st.peak = st.allocated;
    return ptr;
}

void tracked_free(int device, void* ptr, Bytes bytes) noexcept {
    if (!ptr)
        return;
    int prev = -1;
    (void)cudaGetDevice(&prev);
    if (prev != device)
        (void)cudaSetDevice(device);
    (void)cudaFree(ptr);
    if (prev >= 0 && prev != device)
        (void)cudaSetDevice(prev);
    auto& st = device_state(device);
    std::lock_guard lock(st.mutex);
    st.allocated = st.allocated >= bytes ? st.allocated - bytes : 0;
}

std::size_t shape_numel(const std::vector<std::int64_t>& shape) {
    std::size_t total = 1;
    for (const auto d : shape) {
        need(d >= 0, "flux2_negative_dim");
        if (d == 0)
            return 0;
        const auto ud = std::size_t(d);
        need(total <= std::numeric_limits<std::size_t>::max() / ud, "flux2_shape_overflow");
        total *= ud;
    }
    return total;
}

inline int div_ceil(std::size_t n, int block) {
    return int((n + std::size_t(block) - 1) / std::size_t(block));
}

// ---------- Basic elementwise & conversion kernels ----------

__global__ void cast_bf16_to_fp32_kernel(
    const __nv_bfloat16* src, float* dst, std::size_t n
) {
    const std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n)
        dst[i] = __bfloat162float(src[i]);
}

__global__ void cast_fp32_to_bf16_kernel(
    const float* src, __nv_bfloat16* dst, std::size_t n
) {
    const std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n)
        dst[i] = __float2bfloat16(src[i]);
}

__global__ void add_bias_fp32_kernel(float* y, const float* bias, std::size_t rows, std::size_t cols) {
    const std::size_t idx = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < rows * cols)
        y[idx] += bias[idx % cols];
}

__global__ void add_bias_bf16_kernel(
    __nv_bfloat16* y, const __nv_bfloat16* bias, std::size_t rows, std::size_t cols
) {
    const std::size_t idx = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < rows * cols)
        y[idx] = __float2bfloat16(
            __bfloat162float(y[idx]) + __bfloat162float(bias[idx % cols])
        );
}

__global__ void add_fp32_kernel(const float* a, const float* b, float* out, std::size_t n) {
    const std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n)
        out[i] = a[i] + b[i];
}

__global__ void add_bf16_kernel(
    const __nv_bfloat16* a, const __nv_bfloat16* b, __nv_bfloat16* out, std::size_t n
) {
    const std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n)
        out[i] = __float2bfloat16(__bfloat162float(a[i]) + __bfloat162float(b[i]));
}

__global__ void add_inplace_fp32_kernel(float* dst, const float* src, std::size_t n) {
    const std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n)
        dst[i] += src[i];
}

__global__ void embedding_bf16_to_fp32_kernel(
    const __nv_bfloat16* weight,
    std::int64_t vocab,
    std::int64_t dim,
    const std::int64_t* ids,
    std::int64_t seq,
    float* out
) {
    const std::size_t idx = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t total = std::size_t(seq) * std::size_t(dim);
    if (idx >= total)
        return;
    const std::int64_t s = std::int64_t(idx / std::size_t(dim));
    const std::int64_t d = std::int64_t(idx % std::size_t(dim));
    const std::int64_t tok = ids[s];
    if (tok >= 0 && tok < vocab)
        out[idx] = __bfloat162float(weight[std::size_t(tok) * std::size_t(dim) + std::size_t(d)]);
    else
        out[idx] = 0.0f;
}

__global__ void rms_norm_fp32_kernel(
    const float* x,
    const float* gamma_f32,
    const __nv_bfloat16* gamma_bf16,
    float* out,
    std::size_t rows,
    std::size_t dim,
    float eps
) {
    __shared__ float smem[256];
    const std::size_t row = blockIdx.x;
    if (row >= rows)
        return;
    const float* xr = x + row * dim;
    float* yr = out + row * dim;
    float sum = 0.0f;
    for (std::size_t i = threadIdx.x; i < dim; i += blockDim.x) {
        const float v = xr[i];
        sum += v * v;
    }
    smem[threadIdx.x] = sum;
    __syncthreads();
    for (unsigned stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            smem[threadIdx.x] += smem[threadIdx.x + stride];
        __syncthreads();
    }
    const float inv = rsqrtf(smem[0] / float(dim) + eps);
    for (std::size_t i = threadIdx.x; i < dim; i += blockDim.x) {
        const float g = gamma_f32 ? gamma_f32[i] : __bfloat162float(gamma_bf16[i]);
        yr[i] = xr[i] * inv * g;
    }
}

__global__ void rms_norm_bf16_kernel(
    const __nv_bfloat16* x,
    const __nv_bfloat16* gamma,
    __nv_bfloat16* out,
    std::size_t rows,
    std::size_t dim,
    float eps
) {
    __shared__ float smem[256];
    const std::size_t row = blockIdx.x;
    if (row >= rows)
        return;
    const __nv_bfloat16* xr = x + row * dim;
    __nv_bfloat16* yr = out + row * dim;
    float sum = 0.0f;
    for (std::size_t i = threadIdx.x; i < dim; i += blockDim.x) {
        const float v = __bfloat162float(xr[i]);
        sum += v * v;
    }
    smem[threadIdx.x] = sum;
    __syncthreads();
    for (unsigned stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            smem[threadIdx.x] += smem[threadIdx.x + stride];
        __syncthreads();
    }
    const float inv = rsqrtf(smem[0] / float(dim) + eps);
    for (std::size_t i = threadIdx.x; i < dim; i += blockDim.x) {
        const float v = __bfloat162float(xr[i]);
        const float g = __bfloat162float(gamma[i]);
        yr[i] = __float2bfloat16(v * inv * g);
    }
}

template <typename T>
__global__ void transpose_to_heads_kernel(
    const T* src, T* dst, std::size_t B, std::size_t S, std::size_t H, std::size_t D
) {
    const std::size_t idx = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t total = B * S * H * D;
    if (idx >= total)
        return;
    const std::size_t d = idx % D;
    const std::size_t t1 = idx / D;
    const std::size_t h = t1 % H;
    const std::size_t t2 = t1 / H;
    const std::size_t s = t2 % S;
    const std::size_t b = t2 / S;
    // src: [B, S, H, D], dst: [B, H, S, D]
    dst[((b * H + h) * S + s) * D + d] = src[idx];
}

template <typename T>
__global__ void unheads_kernel(
    const T* src, T* dst, std::size_t B, std::size_t H, std::size_t S, std::size_t D
) {
    const std::size_t idx = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t total = B * H * S * D;
    if (idx >= total)
        return;
    const std::size_t d = idx % D;
    const std::size_t t1 = idx / D;
    const std::size_t s = t1 % S;
    const std::size_t t2 = t1 / S;
    const std::size_t h = t2 % H;
    const std::size_t b = t2 / H;
    // src: [B, H, S, D], dst: [B, S, H, D]
    dst[((b * S + s) * H + h) * D + d] = src[idx];
}

__global__ void text_rope_fp32_kernel(
    const float* x,
    const float* cos_t,
    const float* sin_t,
    float* out,
    std::size_t H,
    std::size_t S
) {
    constexpr std::size_t half = 64;
    const std::size_t idx = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t total = H * S * half;
    if (idx >= total)
        return;
    const std::size_t d = idx % half;
    const std::size_t t1 = idx / half;
    const std::size_t s = t1 % S;
    const std::size_t h = t1 / S;
    const std::size_t base = (h * S + s) * 128;
    const float a = x[base + d];
    const float b = x[base + d + half];
    const float c = cos_t[s * half + d];
    const float sn = sin_t[s * half + d];
    out[base + d] = a * c - b * sn;
    out[base + d + half] = b * c + a * sn;
}

__global__ void silu_mul_fp32_kernel(
    const float* gate, const float* up, float* out, std::size_t n
) {
    const std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        const float g = gate[i];
        out[i] = (g / (1.0f + expf(-g))) * up[i];
    }
}

__global__ void concat_last_3_fp32_kernel(
    const float* a,
    const float* b,
    const float* c,
    float* out,
    std::size_t rows,
    std::size_t d
) {
    const std::size_t idx = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t total = rows * d * 3;
    if (idx >= total)
        return;
    const std::size_t col = idx % (3 * d);
    const std::size_t row = idx / (3 * d);
    if (col < d)
        out[idx] = a[row * d + col];
    else if (col < 2 * d)
        out[idx] = b[row * d + (col - d)];
    else
        out[idx] = c[row * d + (col - 2 * d)];
}

// ---------- Qwen3 GQA Causal+Active SDPA (FP32, D=128) ----------
// 4 warps per block = 128 threads; each warp computes 1 query row.
// Shared memory holds BK=16 K and V rows (16 * 128 * 4 * 2 = 16 KiB).
__global__ void sdpa_qwen3_fp32_kernel(
    const float* Q,
    const float* K,
    const float* V,
    float* O,
    std::size_t H_q,
    std::size_t H_kv,
    std::size_t S,
    std::int64_t active_count
) {
    constexpr int kWarps = 4;
    constexpr int kBK = 16;
    constexpr int kD = 128;
    const float scale = 0.08838834764831844f;  // 1 / sqrt(128)

    __shared__ float k_smem[kBK][kD];
    __shared__ float v_smem[kBK][kD];

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const std::size_t h_q = blockIdx.y;
    const std::size_t h_kv = h_q / (H_q / H_kv);
    const std::size_t q_row = std::size_t(blockIdx.x) * kWarps + std::size_t(warp);

    float q_reg[4] = {0.f, 0.f, 0.f, 0.f};
    if (q_row < S) {
        const float* q_ptr = Q + (h_q * S + q_row) * kD + lane * 4;
        q_reg[0] = q_ptr[0];
        q_reg[1] = q_ptr[1];
        q_reg[2] = q_ptr[2];
        q_reg[3] = q_ptr[3];
    }

    float o_reg[4] = {0.f, 0.f, 0.f, 0.f};
    float m_val = -INFINITY;
    float l_val = 0.0f;

    const std::size_t kv_base = h_kv * S * kD;
    const std::size_t k_limit =
        active_count <= 0 ? 0 : std::min<std::size_t>(S, std::size_t(active_count));

    for (std::size_t k_start = 0; k_start < k_limit; k_start += kBK) {
        // Cooperatively load kBK rows of K and V (kBK * 128 floats = 2048 floats; 128 threads -> 16 each).
        for (int idx = tid; idx < kBK * kD; idx += blockDim.x) {
            const int r = idx / kD;
            const int c = idx % kD;
            const std::size_t k_row = k_start + std::size_t(r);
            if (k_row < k_limit) {
                k_smem[r][c] = K[kv_base + k_row * kD + std::size_t(c)];
                v_smem[r][c] = V[kv_base + k_row * kD + std::size_t(c)];
            } else {
                k_smem[r][c] = 0.0f;
                v_smem[r][c] = 0.0f;
            }
        }
        __syncthreads();

        if (q_row < S) {
            const int tile_end = int(std::min<std::size_t>(kBK, k_limit - k_start));
            for (int j = 0; j < tile_end; ++j) {
                const std::size_t k_row = k_start + std::size_t(j);
                if (k_row > q_row)
                    break;
                float dot = q_reg[0] * k_smem[j][lane * 4 + 0]
                            + q_reg[1] * k_smem[j][lane * 4 + 1]
                            + q_reg[2] * k_smem[j][lane * 4 + 2]
                            + q_reg[3] * k_smem[j][lane * 4 + 3];
                for (int offset = 16; offset > 0; offset >>= 1)
                    dot += __shfl_xor_sync(0xffffffffu, dot, offset);
                const float score = dot * scale;
                const float m_new = fmaxf(m_val, score);
                const float alpha = m_val == -INFINITY ? 0.0f : expf(m_val - m_new);
                const float p = expf(score - m_new);
                l_val = l_val * alpha + p;
                m_val = m_new;
                o_reg[0] = o_reg[0] * alpha + p * v_smem[j][lane * 4 + 0];
                o_reg[1] = o_reg[1] * alpha + p * v_smem[j][lane * 4 + 1];
                o_reg[2] = o_reg[2] * alpha + p * v_smem[j][lane * 4 + 2];
                o_reg[3] = o_reg[3] * alpha + p * v_smem[j][lane * 4 + 3];
            }
        }
        __syncthreads();
    }

    if (q_row < S) {
        const float inv_l = l_val > 0.0f ? (1.0f / l_val) : 0.0f;
        float* o_ptr = O + (h_q * S + q_row) * kD + lane * 4;
        o_ptr[0] = o_reg[0] * inv_l;
        o_ptr[1] = o_reg[1] * inv_l;
        o_ptr[2] = o_reg[2] * inv_l;
        o_ptr[3] = o_reg[3] * inv_l;
    }
}

// ---------- FLUX.2 Denoiser kernels ----------

__global__ void silu_bf16_kernel(const __nv_bfloat16* x, __nv_bfloat16* out, std::size_t n) {
    const std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        const float v = __bfloat162float(x[i]);
        out[i] = __float2bfloat16(v / (1.0f + expf(-v)));
    }
}

__global__ void split_mod_scale_shift_gate_kernel(
    const __nv_bfloat16* src,
    float* shift,
    float* scale,
    __nv_bfloat16* gate,
    std::size_t d,
    std::size_t offset
) {
    const std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < d) {
        shift[i] = __bfloat162float(src[offset + i]);
        scale[i] = __bfloat162float(src[offset + d + i]) + 1.0f;
        gate[i] = src[offset + 2 * d + i];
    }
}

__global__ void split_out_mod_kernel(
    const __nv_bfloat16* src, float* scale, float* shift, std::size_t d
) {
    const std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < d) {
        scale[i] = __bfloat162float(src[i]) + 1.0f;
        shift[i] = __bfloat162float(src[d + i]);
    }
}

__global__ void flux2_modulated_norm_bf16_kernel(
    const __nv_bfloat16* x,
    const float* shift,
    const float* scale,
    __nv_bfloat16* out,
    std::size_t rows,
    std::size_t d,
    float eps
) {
    __shared__ float smem[256];
    const std::size_t row = blockIdx.x;
    if (row >= rows)
        return;
    const __nv_bfloat16* xr = x + row * d;
    __nv_bfloat16* yr = out + row * d;

    float sum = 0.0f;
    for (std::size_t i = threadIdx.x; i < d; i += blockDim.x)
        sum += __bfloat162float(xr[i]);
    smem[threadIdx.x] = sum;
    __syncthreads();
    for (unsigned stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            smem[threadIdx.x] += smem[threadIdx.x + stride];
        __syncthreads();
    }
    const float mean = smem[0] / float(d);

    float var_sum = 0.0f;
    for (std::size_t i = threadIdx.x; i < d; i += blockDim.x) {
        const float diff = __bfloat162float(xr[i]) - mean;
        var_sum += diff * diff;
    }
    smem[threadIdx.x] = var_sum;
    __syncthreads();
    for (unsigned stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            smem[threadIdx.x] += smem[threadIdx.x + stride];
        __syncthreads();
    }
    const float inv_std = rsqrtf(smem[0] / float(d) + eps);
    for (std::size_t i = threadIdx.x; i < d; i += blockDim.x) {
        const float normed = (__bfloat162float(xr[i]) - mean) * inv_std;
        yr[i] = __float2bfloat16(normed * scale[i] + shift[i]);
    }
}

__global__ void flux2_gated_residual_bf16_kernel(
    __nv_bfloat16* x,
    const __nv_bfloat16* delta,
    const __nv_bfloat16* gate,
    std::size_t rows,
    std::size_t d
) {
    const std::size_t idx = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < rows * d) {
        const float xv = __bfloat162float(x[idx]);
        const float dv = __bfloat162float(delta[idx]);
        const float gv = __bfloat162float(gate[idx % d]);
        x[idx] = __float2bfloat16(xv + dv * gv);
    }
}

__global__ void flux2_rope_bf16_kernel(
    const __nv_bfloat16* x,
    const float* cos_t,
    const float* sin_t,
    __nv_bfloat16* out,
    std::size_t BH,
    std::size_t S
) {
    constexpr std::size_t half = 64;
    const std::size_t idx = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t total = BH * S * half;
    if (idx >= total)
        return;
    const std::size_t k = idx % half;
    const std::size_t t1 = idx / half;
    const std::size_t s = t1 % S;
    const std::size_t bh = t1 / S;
    const std::size_t base = (bh * S + s) * 128 + 2 * k;
    const float a = __bfloat162float(x[base]);
    const float b = __bfloat162float(x[base + 1]);
    const float c = cos_t[s * half + k];
    const float sn = sin_t[s * half + k];
    out[base] = __float2bfloat16(a * c - b * sn);
    out[base + 1] = __float2bfloat16(b * c + a * sn);
}

// ---------- FLUX.2 FlashAttention-style SDPA (BF16, D=128, non-causal) ----------
// 4 warps = 128 threads; each warp computes 1 query row.
// Shared memory holds BK=32 BF16 K and V rows (32 * 128 * 2 * 2 = 16 KiB).
__global__ void sdpa_flux2_bf16_kernel(
    const __nv_bfloat16* Q,
    const __nv_bfloat16* K,
    const __nv_bfloat16* V,
    __nv_bfloat16* O,
    std::size_t BH,
    std::size_t S
) {
    constexpr int kWarps = 4;
    constexpr int kBK = 32;
    constexpr int kD = 128;
    const float scale = 0.08838834764831844f;  // 1 / sqrt(128)

    __shared__ __nv_bfloat16 k_smem[kBK][kD];
    __shared__ __nv_bfloat16 v_smem[kBK][kD];

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const std::size_t bh = blockIdx.y;
    if (bh >= BH)
        return;
    const std::size_t q_row = std::size_t(blockIdx.x) * kWarps + std::size_t(warp);

    float q_reg[4] = {0.f, 0.f, 0.f, 0.f};
    const std::size_t head_base = bh * S * kD;
    if (q_row < S) {
        const __nv_bfloat16* q_ptr = Q + head_base + q_row * kD + lane * 4;
        q_reg[0] = __bfloat162float(q_ptr[0]);
        q_reg[1] = __bfloat162float(q_ptr[1]);
        q_reg[2] = __bfloat162float(q_ptr[2]);
        q_reg[3] = __bfloat162float(q_ptr[3]);
    }

    float o_reg[4] = {0.f, 0.f, 0.f, 0.f};
    float m_val = -INFINITY;
    float l_val = 0.0f;

    for (std::size_t k_start = 0; k_start < S; k_start += kBK) {
        // 32 * 128 = 4096 bf16s = 1024 uint64 chunks? Keep clean bf16 loads (32 per thread).
        for (int idx = tid; idx < kBK * kD; idx += blockDim.x) {
            const int r = idx / kD;
            const int c = idx % kD;
            const std::size_t k_row = k_start + std::size_t(r);
            if (k_row < S) {
                k_smem[r][c] = K[head_base + k_row * kD + std::size_t(c)];
                v_smem[r][c] = V[head_base + k_row * kD + std::size_t(c)];
            } else {
                k_smem[r][c] = __float2bfloat16(0.0f);
                v_smem[r][c] = __float2bfloat16(0.0f);
            }
        }
        __syncthreads();

        if (q_row < S) {
            const int tile_end = int(std::min<std::size_t>(kBK, S - k_start));
            for (int j = 0; j < tile_end; ++j) {
                float dot = q_reg[0] * __bfloat162float(k_smem[j][lane * 4 + 0])
                            + q_reg[1] * __bfloat162float(k_smem[j][lane * 4 + 1])
                            + q_reg[2] * __bfloat162float(k_smem[j][lane * 4 + 2])
                            + q_reg[3] * __bfloat162float(k_smem[j][lane * 4 + 3]);
                for (int offset = 16; offset > 0; offset >>= 1)
                    dot += __shfl_xor_sync(0xffffffffu, dot, offset);
                const float score = dot * scale;
                const float m_new = fmaxf(m_val, score);
                const float alpha = m_val == -INFINITY ? 0.0f : expf(m_val - m_new);
                const float p = expf(score - m_new);
                l_val = l_val * alpha + p;
                m_val = m_new;
                o_reg[0] = o_reg[0] * alpha + p * __bfloat162float(v_smem[j][lane * 4 + 0]);
                o_reg[1] = o_reg[1] * alpha + p * __bfloat162float(v_smem[j][lane * 4 + 1]);
                o_reg[2] = o_reg[2] * alpha + p * __bfloat162float(v_smem[j][lane * 4 + 2]);
                o_reg[3] = o_reg[3] * alpha + p * __bfloat162float(v_smem[j][lane * 4 + 3]);
            }
        }
        __syncthreads();
    }

    if (q_row < S) {
        const float inv_l = l_val > 0.0f ? (1.0f / l_val) : 0.0f;
        __nv_bfloat16* o_ptr = O + head_base + q_row * kD + lane * 4;
        o_ptr[0] = __float2bfloat16(o_reg[0] * inv_l);
        o_ptr[1] = __float2bfloat16(o_reg[1] * inv_l);
        o_ptr[2] = __float2bfloat16(o_reg[2] * inv_l);
        o_ptr[3] = __float2bfloat16(o_reg[3] * inv_l);
    }
}

__global__ void flux2_swiglu_bf16_kernel(
    const __nv_bfloat16* x, __nv_bfloat16* out, std::size_t rows, std::size_t half
) {
    const std::size_t idx = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= rows * half)
        return;
    const std::size_t r = idx / half;
    const std::size_t c = idx % half;
    const float g = __bfloat162float(x[r * (2 * half) + c]);
    const float u = __bfloat162float(x[r * (2 * half) + half + c]);
    out[idx] = __float2bfloat16((g / (1.0f + expf(-g))) * u);
}

__global__ void concat_axis1_bf16_kernel(
    const __nv_bfloat16* a,
    const __nv_bfloat16* b,
    __nv_bfloat16* out,
    std::size_t outer,
    std::size_t la,
    std::size_t lb,
    std::size_t inner
) {
    const std::size_t idx = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t lt = la + lb;
    const std::size_t total = outer * lt * inner;
    if (idx >= total)
        return;
    const std::size_t in = idx % inner;
    const std::size_t t = idx / inner;
    const std::size_t mid = t % lt;
    const std::size_t out_i = t / lt;
    if (mid < la)
        out[idx] = a[(out_i * la + mid) * inner + in];
    else
        out[idx] = b[(out_i * lb + (mid - la)) * inner + in];
}

__global__ void slice_axis1_bf16_kernel(
    const __nv_bfloat16* src,
    __nv_bfloat16* dst,
    std::size_t outer,
    std::size_t src_mid,
    std::size_t start,
    std::size_t len,
    std::size_t inner
) {
    const std::size_t idx = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t total = outer * len * inner;
    if (idx >= total)
        return;
    const std::size_t in = idx % inner;
    const std::size_t t = idx / inner;
    const std::size_t mid = t % len;
    const std::size_t out_i = t / len;
    dst[idx] = src[(out_i * src_mid + (start + mid)) * inner + in];
}

__global__ void broadcast_batch_bf16_kernel(
    const __nv_bfloat16* src, __nv_bfloat16* dst, std::size_t batch, std::size_t elems
) {
    const std::size_t idx = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < batch * elems)
        dst[idx] = src[idx % elems];
}

__global__ void euler_step_fp32_kernel(
    float* latent, const float* velocity, float dt, std::size_t n
) {
    const std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n)
        latent[i] += velocity[i] * dt;
}

__global__ void mix_source_noise_fp32_kernel(
    float* latent, const float* source, const float* noise, float sigma, std::size_t n
) {
    const std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n)
        latent[i] = source[i] * (1.0f - sigma) + noise[i] * sigma;
}

__global__ void apply_latent_mask_fp32_kernel(
    float* latent,
    const float* source,
    const float* noise,
    const float* mask,
    float next_sigma,
    std::size_t points,
    std::size_t channels
) {
    const std::size_t idx = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= points * channels)
        return;
    const float m = mask[idx / channels];
    const float bg = source[idx] * (1.0f - next_sigma) + noise[idx] * next_sigma;
    latent[idx] = m * latent[idx] + (1.0f - m) * bg;
}

// ---------- VAE kernels (FP32) ----------

__global__ void group_norm_32_fp32_kernel(
    const float* x,
    const float* weight,
    const float* bias,
    float* out,
    std::size_t C,
    std::size_t HW,
    bool apply_silu,
    float eps
) {
    __shared__ double smem[256];
    const std::size_t g = blockIdx.x;  // 0..31
    const std::size_t c_per_group = C / 32;
    const std::size_t group_elems = c_per_group * HW;
    const std::size_t c_start = g * c_per_group;

    double sum = 0.0;
    for (std::size_t i = threadIdx.x; i < group_elems; i += blockDim.x) {
        const std::size_t c = c_start + i / HW;
        const std::size_t p = i % HW;
        sum += double(x[c * HW + p]);
    }
    smem[threadIdx.x] = sum;
    __syncthreads();
    for (unsigned stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            smem[threadIdx.x] += smem[threadIdx.x + stride];
        __syncthreads();
    }
    const double mean = smem[0] / double(group_elems);

    double var_sum = 0.0;
    for (std::size_t i = threadIdx.x; i < group_elems; i += blockDim.x) {
        const std::size_t c = c_start + i / HW;
        const std::size_t p = i % HW;
        const double diff = double(x[c * HW + p]) - mean;
        var_sum += diff * diff;
    }
    smem[threadIdx.x] = var_sum;
    __syncthreads();
    for (unsigned stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            smem[threadIdx.x] += smem[threadIdx.x + stride];
        __syncthreads();
    }
    const float inv_std = float(1.0 / sqrt(smem[0] / double(group_elems) + double(eps)));
    const float mean_f = float(mean);

    for (std::size_t i = threadIdx.x; i < group_elems; i += blockDim.x) {
        const std::size_t c = c_start + i / HW;
        const std::size_t p = i % HW;
        float v = (x[c * HW + p] - mean_f) * inv_std * weight[c] + bias[c];
        if (apply_silu)
            v = v / (1.0f + expf(-v));
        out[c * HW + p] = v;
    }
}

__global__ void im2col_3x3_chunk_kernel(
    const float* x,
    float* col,
    std::size_t C_in,
    std::size_t H,
    std::size_t W,
    std::size_t H_out,
    std::size_t W_out,
    std::size_t p_start,
    std::size_t chunk_p,
    int stride,
    int pad_top,
    int pad_left
) {
    const std::size_t K = C_in * 9;
    const std::size_t idx = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= K * chunk_p)
        return;
    const std::size_t p_local = idx % chunk_p;
    const std::size_t k = idx / chunk_p;
    const std::size_t p = p_start + p_local;
    const std::size_t oy = p / W_out;
    const std::size_t ox = p % W_out;
    const std::size_t c = k / 9;
    const std::size_t rem = k % 9;
    const int ky = int(rem / 3);
    const int kx = int(rem % 3);
    const std::int64_t iy = std::int64_t(oy) * stride + ky - pad_top;
    const std::int64_t ix = std::int64_t(ox) * stride + kx - pad_left;
    if (iy >= 0 && iy < std::int64_t(H) && ix >= 0 && ix < std::int64_t(W))
        col[idx] = x[(c * H + std::size_t(iy)) * W + std::size_t(ix)];
    else
        col[idx] = 0.0f;
}

__global__ void add_channel_bias_nchw_kernel(
    float* y, const float* bias, std::size_t C_out, std::size_t HW
) {
    const std::size_t idx = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < C_out * HW)
        y[idx] += bias[idx / HW];
}

__global__ void upsample_nearest2d_kernel(
    const float* x, float* out, std::size_t C, std::size_t H, std::size_t W
) {
    const std::size_t H2 = H * 2;
    const std::size_t W2 = W * 2;
    const std::size_t idx = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= C * H2 * W2)
        return;
    const std::size_t ox = idx % W2;
    const std::size_t t = idx / W2;
    const std::size_t oy = t % H2;
    const std::size_t c = t / H2;
    out[idx] = x[(c * H + (oy >> 1)) * W + (ox >> 1)];
}

// VAE single-head mid-block SDPA (FP32, D=512).
// 4 warps = 128 threads; each lane holds 16 elements (512/32).
// Shared memory holds BK=8 rows of K and V (8 * 512 * 4 * 2 = 32 KiB).
__global__ void vae_sdpa_fp32_kernel(
    const float* Q,
    const float* K,
    const float* V,
    float* O,
    std::size_t points,
    std::size_t q_offset,
    std::size_t q_count
) {
    constexpr int kWarps = 4;
    constexpr int kBK = 8;
    constexpr int kD = 512;
    constexpr int kPerLane = 16;
    const float scale = 0.04419417382415922f;  // 1 / sqrt(512)

    __shared__ float k_smem[kBK][kD];
    __shared__ float v_smem[kBK][kD];

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const std::size_t local_q = std::size_t(blockIdx.x) * kWarps + std::size_t(warp);
    const bool valid_q = local_q < q_count;
    const std::size_t q_row = q_offset + local_q;

    float q_reg[kPerLane];
    float o_reg[kPerLane];
    for (int i = 0; i < kPerLane; ++i) {
        q_reg[i] = valid_q ? Q[q_row * kD + std::size_t(lane * kPerLane + i)] : 0.0f;
        o_reg[i] = 0.0f;
    }

    float m_val = -INFINITY;
    float l_val = 0.0f;

    for (std::size_t k_start = 0; k_start < points; k_start += kBK) {
        for (int idx = tid; idx < kBK * kD; idx += blockDim.x) {
            const int r = idx / kD;
            const int c = idx % kD;
            const std::size_t k_row = k_start + std::size_t(r);
            if (k_row < points) {
                k_smem[r][c] = K[k_row * kD + std::size_t(c)];
                v_smem[r][c] = V[k_row * kD + std::size_t(c)];
            } else {
                k_smem[r][c] = 0.0f;
                v_smem[r][c] = 0.0f;
            }
        }
        __syncthreads();

        if (valid_q) {
            const int tile_end = int(std::min<std::size_t>(kBK, points - k_start));
            for (int j = 0; j < tile_end; ++j) {
                float dot = 0.0f;
                const float* kr = &k_smem[j][lane * kPerLane];
                for (int i = 0; i < kPerLane; ++i)
                    dot += q_reg[i] * kr[i];
                for (int offset = 16; offset > 0; offset >>= 1)
                    dot += __shfl_xor_sync(0xffffffffu, dot, offset);
                const float score = dot * scale;
                const float m_new = fmaxf(m_val, score);
                const float alpha = m_val == -INFINITY ? 0.0f : expf(m_val - m_new);
                const float p = expf(score - m_new);
                l_val = l_val * alpha + p;
                m_val = m_new;
                const float* vr = &v_smem[j][lane * kPerLane];
                for (int i = 0; i < kPerLane; ++i)
                    o_reg[i] = o_reg[i] * alpha + p * vr[i];
            }
        }
        __syncthreads();
    }

    if (valid_q) {
        const float inv_l = l_val > 0.0f ? (1.0f / l_val) : 0.0f;
        float* op = O + q_row * kD + lane * kPerLane;
        for (int i = 0; i < kPerLane; ++i)
            op[i] = o_reg[i] * inv_l;
    }
}

__global__ void nchw_to_seq_kernel(
    const float* x, float* out, std::size_t C, std::size_t HW
) {
    const std::size_t idx = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= C * HW)
        return;
    const std::size_t c = idx % C;
    const std::size_t p = idx / C;
    out[p * C + c] = x[c * HW + p];
}

__global__ void seq_to_nchw_kernel(
    const float* x, float* out, std::size_t C, std::size_t HW
) {
    const std::size_t idx = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= C * HW)
        return;
    const std::size_t p = idx % HW;
    const std::size_t c = idx / HW;
    out[c * HW + p] = x[p * C + c];
}

// 2x2 patchify + BN normalize: x[32, h*2, w*2] -> out[h*w, 128]
// Matches x.reshape({1,32,h,2,w,2}).permute({0,1,3,5,2,4}).reshape({1,128,h,w})
// then (x - mean)/std -> flatten(2).transpose(0,1).
__global__ void vae_patchify_bn_norm_kernel(
    const float* x,
    const float* bn_mean,
    const float* bn_var,
    float bn_eps,
    float* out,
    std::size_t h,
    std::size_t w
) {
    const std::size_t idx = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t total = h * w * 128;
    if (idx >= total)
        return;
    const std::size_t ch128 = idx % 128;
    const std::size_t pos = idx / 128;
    const std::size_t y = pos / w;
    const std::size_t xw = pos % w;
    const std::size_t c32 = ch128 >> 2;
    const std::size_t py = (ch128 >> 1) & 1;
    const std::size_t px = ch128 & 1;
    const float v = x[(c32 * (h * 2) + (y * 2 + py)) * (w * 2) + (xw * 2 + px)];
    const float std_v = sqrtf(bn_var[ch128] + bn_eps);
    out[idx] = (v - bn_mean[ch128]) / std_v;
}

// BN denormalize + 2x2 unpatchify: latent[h*w, 128] -> out[32, h*2, w*2]
// Matches reshape({1,128,h,w}) * bn_std + bn_mean -> reshape({1,32,2,2,h,w}).permute({0,1,4,2,5,3}).reshape({1,32,h*2,w*2}).
__global__ void vae_bn_denorm_unpatchify_kernel(
    const float* latent,
    const float* bn_mean,
    const float* bn_var,
    float bn_eps,
    float* out,
    std::size_t h,
    std::size_t w
) {
    const std::size_t idx = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t total = 32 * (h * 2) * (w * 2);
    if (idx >= total)
        return;
    const std::size_t x2 = idx % (w * 2);
    const std::size_t t = idx / (w * 2);
    const std::size_t y2 = t % (h * 2);
    const std::size_t c32 = t / (h * 2);
    const std::size_t y = y2 >> 1;
    const std::size_t py = y2 & 1;
    const std::size_t xw = x2 >> 1;
    const std::size_t px = x2 & 1;
    const std::size_t ch128 = (c32 << 2) | (py << 1) | px;
    const float v = latent[(y * w + xw) * 128 + ch128];
    out[idx] = v * sqrtf(bn_var[ch128] + bn_eps) + bn_mean[ch128];
}

__global__ void vae_clamp_rgba_kernel(
    const float* rgb, float* rgba, std::size_t HW
) {
    const std::size_t idx = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= 4 * HW)
        return;
    const std::size_t c = idx / HW;
    if (c < 3)
        rgba[idx] = fminf(1.0f, fmaxf(-1.0f, rgb[idx]));
    else
        rgba[idx] = 1.0f;
}

__global__ void crop_2d_kernel(
    const float* src,
    float* dst,
    std::size_t C,
    std::size_t src_H,
    std::size_t src_W,
    std::size_t y0,
    std::size_t y_len,
    std::size_t x0,
    std::size_t x_len
) {
    const std::size_t idx = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= C * y_len * x_len)
        return;
    const std::size_t dx = idx % x_len;
    const std::size_t t = idx / x_len;
    const std::size_t dy = t % y_len;
    const std::size_t c = t / y_len;
    dst[idx] = src[(c * src_H + (y0 + dy)) * src_W + (x0 + dx)];
}

__global__ void paste_2d_kernel(
    float* dst,
    const float* src,
    std::size_t C,
    std::size_t dst_H,
    std::size_t dst_W,
    std::size_t src_H,
    std::size_t src_W,
    std::size_t dst_y,
    std::size_t dst_x,
    std::size_t src_y,
    std::size_t src_x,
    std::size_t h_len,
    std::size_t w_len
) {
    const std::size_t idx = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= C * h_len * w_len)
        return;
    const std::size_t dx = idx % w_len;
    const std::size_t t = idx / w_len;
    const std::size_t dy = t % h_len;
    const std::size_t c = t / h_len;
    dst[(c * dst_H + (dst_y + dy)) * dst_W + (dst_x + dx)] =
        src[(c * src_H + (src_y + dy)) * src_W + (src_x + dx)];
}

}  // namespace

// ---------- Device management ----------

DeviceGuard::DeviceGuard(int device) {
    cuda_check(cudaGetDevice(&previous_), "cudaGetDevice");
    if (previous_ != device) {
        cuda_check(cudaSetDevice(device), "cudaSetDevice");
        active_ = true;
    }
}

DeviceGuard::~DeviceGuard() {
    if (active_)
        (void)cudaSetDevice(previous_);
}

void init_device(int device) {
    DeviceGuard guard(device);
    cuda_check(cudaFree(nullptr), "cuda_init");
    (void)get_blas(device);
}

DeviceStats device_stats(int device) {
    auto& st = device_state(device);
    std::lock_guard lock(st.mutex);
    return {st.allocated, st.peak};
}

void reset_peak_stats(int device) {
    auto& st = device_state(device);
    std::lock_guard lock(st.mutex);
    st.peak = st.allocated;
}

void synchronize_device(int device) {
    DeviceGuard guard(device);
    cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
}

void clear_device_workspaces(int device) {
    auto& st = device_state(device);
    std::lock_guard lock(st.mutex);
    if (st.blas) {
        DeviceGuard guard(device);
        (void)cublasSetWorkspace(st.blas, nullptr, 0);
        if (st.blas_workspace) {
            (void)cudaFree(st.blas_workspace);
            st.blas_workspace = nullptr;
            st.blas_workspace_bytes = 0;
        }
    }
}

void destroy_device_handles(int device) noexcept {
    if (device < 0 || device >= kMaxTrackedDevices)
        return;
    auto& st = device_state(device);
    std::lock_guard lock(st.mutex);
    int prev = -1;
    (void)cudaGetDevice(&prev);
    if (prev != device)
        (void)cudaSetDevice(device);
    if (st.blas) {
        (void)cublasDestroy(st.blas);
        st.blas = nullptr;
    }
    if (st.blas_workspace) {
        (void)cudaFree(st.blas_workspace);
        st.blas_workspace = nullptr;
        st.blas_workspace_bytes = 0;
    }
    if (prev >= 0 && prev != device)
        (void)cudaSetDevice(prev);
}

void query_device_memory(int device, Bytes& free_bytes, Bytes& total_bytes) {
    DeviceGuard guard(device);
    std::size_t f = 0, t = 0;
    cuda_check(cudaMemGetInfo(&f, &t), "cudaMemGetInfo");
    free_bytes = Bytes(f);
    total_bytes = Bytes(t);
}

// ---------- Tensor ----------

Tensor::Tensor(int device, Dtype dtype, std::vector<std::int64_t> shape)
    : device_(device), dtype_(dtype), shape_(std::move(shape)),
      numel_(shape_numel(shape_)), bytes_(Bytes(numel_) * dtype_bytes(dtype_)) {
    data_ = tracked_alloc(device_, bytes_);
}

Tensor::~Tensor() {
    reset();
}

Tensor::Tensor(Tensor&& other) noexcept
    : data_(other.data_), device_(other.device_), dtype_(other.dtype_),
      shape_(std::move(other.shape_)), numel_(other.numel_), bytes_(other.bytes_) {
    other.data_ = nullptr;
    other.device_ = -1;
    other.numel_ = 0;
    other.bytes_ = 0;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this != &other) {
        reset();
        data_ = other.data_;
        device_ = other.device_;
        dtype_ = other.dtype_;
        shape_ = std::move(other.shape_);
        numel_ = other.numel_;
        bytes_ = other.bytes_;
        other.data_ = nullptr;
        other.device_ = -1;
        other.numel_ = 0;
        other.bytes_ = 0;
    }
    return *this;
}

void Tensor::reset() noexcept {
    if (data_) {
        tracked_free(device_, data_, bytes_);
        data_ = nullptr;
        bytes_ = 0;
        numel_ = 0;
        shape_.clear();
    }
}

Tensor& Tensor::reshape_inplace(std::vector<std::int64_t> new_shape) {
    need(shape_numel(new_shape) == numel_, "flux2_reshape_numel");
    shape_ = std::move(new_shape);
    return *this;
}

Tensor Tensor::to_dtype(Dtype target_dtype) const {
    if (target_dtype == dtype_)
        return clone();
    DeviceGuard guard(device_);
    Tensor out(device_, target_dtype, shape_);
    if (numel_ == 0)
        return out;
    if (dtype_ == Dtype::bf16 && target_dtype == Dtype::fp32) {
        cast_bf16_to_fp32_kernel<<<div_ceil(numel_, 256), 256>>>(
            static_cast<const __nv_bfloat16*>(data_),
            static_cast<float*>(out.data_),
            numel_
        );
    } else {
        cast_fp32_to_bf16_kernel<<<div_ceil(numel_, 256), 256>>>(
            static_cast<const float*>(data_),
            static_cast<__nv_bfloat16*>(out.data_),
            numel_
        );
    }
    cuda_check(cudaGetLastError(), "cast_dtype");
    return out;
}

Tensor Tensor::clone() const {
    DeviceGuard guard(device_);
    Tensor out(device_, dtype_, shape_);
    if (bytes_ > 0)
        cuda_check(
            cudaMemcpy(out.data_, data_, bytes_, cudaMemcpyDeviceToDevice),
            "tensor_clone"
        );
    return out;
}

Tensor Tensor::to(int target_device, Dtype target_dtype) const {
    if (target_device == device_)
        return to_dtype(target_dtype);
    Tensor dst(target_device, dtype_, shape_);
    if (bytes_ > 0) {
        const cudaError_t err =
            cudaMemcpyPeer(dst.data_, target_device, data_, device_, bytes_);
        if (err != cudaSuccess) {
            std::vector<std::uint8_t> bounce(bytes_);
            {
                DeviceGuard g1(device_);
                cuda_check(
                    cudaMemcpy(bounce.data(), data_, bytes_, cudaMemcpyDeviceToHost),
                    "tensor_peer_d2h"
                );
            }
            {
                DeviceGuard g2(target_device);
                cuda_check(
                    cudaMemcpy(dst.data_, bounce.data(), bytes_, cudaMemcpyHostToDevice),
                    "tensor_peer_h2d"
                );
            }
        }
    }
    return target_dtype == dtype_ ? std::move(dst) : dst.to_dtype(target_dtype);
}

Tensor Tensor::upload_raw(
    int device,
    Dtype stored_dtype,
    Dtype target_dtype,
    std::vector<std::int64_t> shape,
    const void* host_bytes,
    Bytes host_nbytes
) {
    const std::size_t n = shape_numel(shape);
    need(Bytes(n) * dtype_bytes(stored_dtype) == host_nbytes, "flux2_upload_bytes");
    DeviceGuard guard(device);
    Tensor raw(device, stored_dtype, shape);
    if (host_nbytes > 0)
        cuda_check(
            cudaMemcpy(raw.data_, host_bytes, host_nbytes, cudaMemcpyHostToDevice),
            "upload_raw"
        );
    if (stored_dtype == target_dtype)
        return raw;
    return raw.to_dtype(target_dtype);
}

Tensor Tensor::from_cpu_float(
    int device,
    std::vector<std::int64_t> shape,
    std::span<const float> values,
    Dtype target_dtype
) {
    return upload_raw(
        device,
        Dtype::fp32,
        target_dtype,
        std::move(shape),
        values.data(),
        Bytes(values.size()) * sizeof(float)
    );
}

Tensor Tensor::from_cpu_int64(
    int device,
    std::vector<std::int64_t> shape,
    std::span<const std::int64_t> values
) {
    // Stored as raw 8-byte elements inside an fp32-typed buffer of 2x elements
    // (only used transiently for Qwen3 token indices).
    const std::size_t n = shape_numel(shape);
    need(values.size() == n, "flux2_int64_size");
    DeviceGuard guard(device);
    Tensor out(device, Dtype::fp32, {std::int64_t(n * 2)});
    if (n > 0)
        cuda_check(
            cudaMemcpy(
                out.data_,
                values.data(),
                n * sizeof(std::int64_t),
                cudaMemcpyHostToDevice
            ),
            "from_cpu_int64"
        );
    return out;
}

std::vector<float> Tensor::to_cpu_float() const {
    if (dtype_ != Dtype::fp32)
        return to_dtype(Dtype::fp32).to_cpu_float();
    DeviceGuard guard(device_);
    std::vector<float> out(numel_);
    if (bytes_ > 0)
        cuda_check(
            cudaMemcpy(out.data(), data_, bytes_, cudaMemcpyDeviceToHost),
            "to_cpu_float"
        );
    return out;
}

HostFloatTensor cpu_float(const Tensor& x) {
    HostFloatTensor result{x.shape(), x.to_cpu_float()};
    for (float v : result.data)
        need(std::isfinite(v), "image_aten_nonfinite");
    return result;
}

// ---------- Linear (cuBLAS GEMM) ----------

Tensor linear(const Tensor& x, const Tensor& weight, const Tensor* bias) {
    need(weight.dim() == 2 && x.dim() >= 2, "flux2_linear_rank");
    const std::int64_t N = weight.size(0);
    const std::int64_t K = weight.size(1);
    need(x.size(-1) == K, "flux2_linear_k");
    const std::size_t M = x.numel() / std::size_t(K);
    std::vector<std::int64_t> out_shape = x.shape();
    out_shape.back() = N;

    const int dev = x.device();
    DeviceGuard guard(dev);
    const Dtype dtype = x.dtype();

    Tensor w_cast;
    const Tensor* w_ptr = &weight;
    if (weight.dtype() != dtype) {
        w_cast = weight.to_dtype(dtype);
        w_ptr = &w_cast;
    }

    Tensor out(dev, dtype, std::move(out_shape));
    const float alpha = 1.0f;
    const float beta = 0.0f;
    const cudaDataType_t cuda_type =
        dtype == Dtype::fp32 ? CUDA_R_32F : CUDA_R_16BF;

    cublasHandle_t blas = get_blas(dev);
    blas_check(
        cublasGemmEx(
            blas,
            CUBLAS_OP_T,
            CUBLAS_OP_N,
            int(N),
            int(M),
            int(K),
            &alpha,
            w_ptr->data(),
            cuda_type,
            int(K),
            x.data(),
            cuda_type,
            int(K),
            &beta,
            out.data(),
            cuda_type,
            int(N),
            CUBLAS_COMPUTE_32F_PEDANTIC,
            CUBLAS_GEMM_DEFAULT
        ),
        "cublasGemmEx_linear"
    );

    if (bias && bias->defined()) {
        Tensor b_cast;
        const Tensor* b_ptr = bias;
        if (bias->dtype() != dtype) {
            b_cast = bias->to_dtype(dtype);
            b_ptr = &b_cast;
        }
        const std::size_t total = M * std::size_t(N);
        if (dtype == Dtype::fp32) {
            add_bias_fp32_kernel<<<div_ceil(total, 256), 256>>>(
                static_cast<float*>(out.data()),
                static_cast<const float*>(b_ptr->data()),
                M,
                std::size_t(N)
            );
        } else {
            add_bias_bf16_kernel<<<div_ceil(total, 256), 256>>>(
                static_cast<__nv_bfloat16*>(out.data()),
                static_cast<const __nv_bfloat16*>(b_ptr->data()),
                M,
                std::size_t(N)
            );
        }
        cuda_check(cudaGetLastError(), "linear_bias");
    }
    return out;
}

// ---------- Qwen3 ops ----------

Tensor embedding_bf16_to_fp32(const Tensor& weight, const Tensor& indices_i64) {
    need(weight.dim() == 2 && weight.dtype() == Dtype::bf16, "flux2_embed_weight");
    const std::int64_t vocab = weight.size(0);
    const std::int64_t dim = weight.size(1);
    const std::int64_t seq = std::int64_t(indices_i64.nbytes() / sizeof(std::int64_t));
    const int dev = weight.device();
    DeviceGuard guard(dev);
    Tensor out(dev, Dtype::fp32, {seq, dim});
    const std::size_t total = std::size_t(seq) * std::size_t(dim);
    embedding_bf16_to_fp32_kernel<<<div_ceil(total, 256), 256>>>(
        static_cast<const __nv_bfloat16*>(weight.data()),
        vocab,
        dim,
        static_cast<const std::int64_t*>(indices_i64.data()),
        seq,
        static_cast<float*>(out.data())
    );
    cuda_check(cudaGetLastError(), "embedding_bf16_to_fp32");
    return out;
}

Tensor rms_norm(const Tensor& x, const Tensor& gamma, float eps) {
    const std::size_t dim = std::size_t(x.size(-1));
    const std::size_t rows = x.numel() / dim;
    const int dev = x.device();
    DeviceGuard guard(dev);
    Tensor out(dev, x.dtype(), x.shape());
    if (x.dtype() == Dtype::fp32) {
        const float* gf =
            gamma.dtype() == Dtype::fp32 ? static_cast<const float*>(gamma.data()) : nullptr;
        const __nv_bfloat16* gb = gamma.dtype() == Dtype::bf16
                                      ? static_cast<const __nv_bfloat16*>(gamma.data())
                                      : nullptr;
        rms_norm_fp32_kernel<<<int(rows), 256>>>(
            static_cast<const float*>(x.data()),
            gf,
            gb,
            static_cast<float*>(out.data()),
            rows,
            dim,
            eps
        );
    } else {
        Tensor g_bf16;
        const Tensor* gp = &gamma;
        if (gamma.dtype() != Dtype::bf16) {
            g_bf16 = gamma.to_dtype(Dtype::bf16);
            gp = &g_bf16;
        }
        rms_norm_bf16_kernel<<<int(rows), 256>>>(
            static_cast<const __nv_bfloat16*>(x.data()),
            static_cast<const __nv_bfloat16*>(gp->data()),
            static_cast<__nv_bfloat16*>(out.data()),
            rows,
            dim,
            eps
        );
    }
    cuda_check(cudaGetLastError(), "rms_norm");
    return out;
}

Tensor transpose_to_heads(const Tensor& x, std::int64_t heads, std::int64_t head_dim) {
    const std::int64_t B = x.dim() == 3 ? x.size(0) : 1;
    const std::int64_t S = x.size(-2);
    need(x.size(-1) == heads * head_dim, "flux2_heads_dim");
    const int dev = x.device();
    DeviceGuard guard(dev);
    Tensor out(dev, x.dtype(), {B, heads, S, head_dim});
    const std::size_t total = out.numel();
    if (x.dtype() == Dtype::fp32) {
        transpose_to_heads_kernel<float><<<div_ceil(total, 256), 256>>>(
            static_cast<const float*>(x.data()),
            static_cast<float*>(out.data()),
            std::size_t(B),
            std::size_t(S),
            std::size_t(heads),
            std::size_t(head_dim)
        );
    } else {
        transpose_to_heads_kernel<__nv_bfloat16><<<div_ceil(total, 256), 256>>>(
            static_cast<const __nv_bfloat16*>(x.data()),
            static_cast<__nv_bfloat16*>(out.data()),
            std::size_t(B),
            std::size_t(S),
            std::size_t(heads),
            std::size_t(head_dim)
        );
    }
    cuda_check(cudaGetLastError(), "transpose_to_heads");
    return out;
}

Tensor unheads(const Tensor& x) {
    need(x.dim() == 4, "flux2_unheads_rank");
    const std::int64_t B = x.size(0);
    const std::int64_t H = x.size(1);
    const std::int64_t S = x.size(2);
    const std::int64_t D = x.size(3);
    const int dev = x.device();
    DeviceGuard guard(dev);
    Tensor out(dev, x.dtype(), {B, S, H * D});
    const std::size_t total = out.numel();
    if (x.dtype() == Dtype::fp32) {
        unheads_kernel<float><<<div_ceil(total, 256), 256>>>(
            static_cast<const float*>(x.data()),
            static_cast<float*>(out.data()),
            std::size_t(B),
            std::size_t(H),
            std::size_t(S),
            std::size_t(D)
        );
    } else {
        unheads_kernel<__nv_bfloat16><<<div_ceil(total, 256), 256>>>(
            static_cast<const __nv_bfloat16*>(x.data()),
            static_cast<__nv_bfloat16*>(out.data()),
            std::size_t(B),
            std::size_t(H),
            std::size_t(S),
            std::size_t(D)
        );
    }
    cuda_check(cudaGetLastError(), "unheads");
    return out;
}

Tensor text_rope_fp32(
    const Tensor& x, const Tensor& cos_table, const Tensor& sin_table
) {
    need(x.dim() == 4 && x.size(0) == 1 && x.size(3) == 128, "flux2_text_rope_shape");
    const std::size_t H = std::size_t(x.size(1));
    const std::size_t S = std::size_t(x.size(2));
    const int dev = x.device();
    DeviceGuard guard(dev);
    Tensor out(dev, Dtype::fp32, x.shape());
    const std::size_t total = H * S * 64;
    text_rope_fp32_kernel<<<div_ceil(total, 256), 256>>>(
        static_cast<const float*>(x.data()),
        static_cast<const float*>(cos_table.data()),
        static_cast<const float*>(sin_table.data()),
        static_cast<float*>(out.data()),
        H,
        S
    );
    cuda_check(cudaGetLastError(), "text_rope_fp32");
    return out;
}

Tensor sdpa_qwen3_fp32(
    const Tensor& q, const Tensor& k, const Tensor& v, std::int64_t active_count
) {
    need(q.dim() == 4 && q.size(0) == 1 && q.size(3) == 128, "flux2_qwen_sdpa_shape");
    const std::size_t H_q = std::size_t(q.size(1));
    const std::size_t H_kv = std::size_t(k.size(1));
    const std::size_t S = std::size_t(q.size(2));
    const int dev = q.device();
    DeviceGuard guard(dev);
    Tensor out(dev, Dtype::fp32, q.shape());
    dim3 grid(div_ceil(S, 4), unsigned(H_q), 1);
    sdpa_qwen3_fp32_kernel<<<grid, 128>>>(
        static_cast<const float*>(q.data()),
        static_cast<const float*>(k.data()),
        static_cast<const float*>(v.data()),
        static_cast<float*>(out.data()),
        H_q,
        H_kv,
        S,
        active_count
    );
    cuda_check(cudaGetLastError(), "sdpa_qwen3_fp32");
    return out;
}

Tensor silu_mul_fp32(const Tensor& gate, const Tensor& up) {
    need(gate.numel() == up.numel(), "flux2_silu_mul_shape");
    const int dev = gate.device();
    DeviceGuard guard(dev);
    Tensor out(dev, Dtype::fp32, gate.shape());
    silu_mul_fp32_kernel<<<div_ceil(gate.numel(), 256), 256>>>(
        static_cast<const float*>(gate.data()),
        static_cast<const float*>(up.data()),
        static_cast<float*>(out.data()),
        gate.numel()
    );
    cuda_check(cudaGetLastError(), "silu_mul_fp32");
    return out;
}

Tensor add_tensors(const Tensor& a, const Tensor& b) {
    need(a.numel() == b.numel() && a.dtype() == b.dtype(), "flux2_add_shape");
    const int dev = a.device();
    DeviceGuard guard(dev);
    Tensor out(dev, a.dtype(), a.shape());
    if (a.dtype() == Dtype::fp32) {
        add_fp32_kernel<<<div_ceil(a.numel(), 256), 256>>>(
            static_cast<const float*>(a.data()),
            static_cast<const float*>(b.data()),
            static_cast<float*>(out.data()),
            a.numel()
        );
    } else {
        add_bf16_kernel<<<div_ceil(a.numel(), 256), 256>>>(
            static_cast<const __nv_bfloat16*>(a.data()),
            static_cast<const __nv_bfloat16*>(b.data()),
            static_cast<__nv_bfloat16*>(out.data()),
            a.numel()
        );
    }
    cuda_check(cudaGetLastError(), "add_tensors");
    return out;
}

void add_inplace_fp32(Tensor& dst, const Tensor& src) {
    need(dst.numel() == src.numel() && dst.dtype() == Dtype::fp32 && src.dtype() == Dtype::fp32,
         "flux2_add_inplace_shape");
    DeviceGuard guard(dst.device());
    add_inplace_fp32_kernel<<<div_ceil(dst.numel(), 256), 256>>>(
        static_cast<float*>(dst.data()),
        static_cast<const float*>(src.data()),
        dst.numel()
    );
    cuda_check(cudaGetLastError(), "add_inplace_fp32");
}

Tensor concat_last_3_fp32(const Tensor& a, const Tensor& b, const Tensor& c) {
    need(a.dim() == 2 && b.shape() == a.shape() && c.shape() == a.shape(),
         "flux2_concat3_shape");
    const std::size_t rows = std::size_t(a.size(0));
    const std::size_t d = std::size_t(a.size(1));
    const int dev = a.device();
    DeviceGuard guard(dev);
    Tensor out(dev, Dtype::fp32, {std::int64_t(rows), std::int64_t(3 * d)});
    concat_last_3_fp32_kernel<<<div_ceil(rows * 3 * d, 256), 256>>>(
        static_cast<const float*>(a.data()),
        static_cast<const float*>(b.data()),
        static_cast<const float*>(c.data()),
        static_cast<float*>(out.data()),
        rows,
        d
    );
    cuda_check(cudaGetLastError(), "concat_last_3_fp32");
    return out;
}

// ---------- FLUX.2 Denoiser ops ----------

Tensor flux2_timestep_proj_bf16(int device, float sigma) {
    std::vector<float> proj(256);
    const double t = double(sigma) * 1000.0;
    const double scale = -std::log(10000.0) / 128.0;
    for (int i = 0; i < 128; ++i) {
        const double angle = t * std::exp(double(i) * scale);
        proj[std::size_t(i)] = float(std::cos(angle));
        proj[std::size_t(128 + i)] = float(std::sin(angle));
    }
    return Tensor::from_cpu_float(device, {1, 256}, proj, Dtype::bf16);
}

Tensor silu_bf16(const Tensor& x) {
    const int dev = x.device();
    DeviceGuard guard(dev);
    Tensor out(dev, Dtype::bf16, x.shape());
    silu_bf16_kernel<<<div_ceil(x.numel(), 256), 256>>>(
        static_cast<const __nv_bfloat16*>(x.data()),
        static_cast<__nv_bfloat16*>(out.data()),
        x.numel()
    );
    cuda_check(cudaGetLastError(), "silu_bf16");
    return out;
}

DoubleModulation split_double_modulation(const Tensor& projected_bf16) {
    need(projected_bf16.numel() % 6 == 0, "flux2_double_mod_shape");
    const std::size_t d = projected_bf16.numel() / 6;
    const int dev = projected_bf16.device();
    DeviceGuard guard(dev);
    DoubleModulation res{
        {Tensor(dev, Dtype::fp32, {std::int64_t(d)}),
         Tensor(dev, Dtype::fp32, {std::int64_t(d)}),
         Tensor(dev, Dtype::bf16, {std::int64_t(d)})},
        {Tensor(dev, Dtype::fp32, {std::int64_t(d)}),
         Tensor(dev, Dtype::fp32, {std::int64_t(d)}),
         Tensor(dev, Dtype::bf16, {std::int64_t(d)})}
    };
    const auto* src = static_cast<const __nv_bfloat16*>(projected_bf16.data());
    split_mod_scale_shift_gate_kernel<<<div_ceil(d, 256), 256>>>(
        src,
        static_cast<float*>(res.attention.shift.data()),
        static_cast<float*>(res.attention.scale.data()),
        static_cast<__nv_bfloat16*>(res.attention.gate.data()),
        d,
        0
    );
    split_mod_scale_shift_gate_kernel<<<div_ceil(d, 256), 256>>>(
        src,
        static_cast<float*>(res.mlp.shift.data()),
        static_cast<float*>(res.mlp.scale.data()),
        static_cast<__nv_bfloat16*>(res.mlp.gate.data()),
        d,
        3 * d
    );
    cuda_check(cudaGetLastError(), "split_double_modulation");
    return res;
}

ModulationTriple split_single_modulation(const Tensor& projected_bf16) {
    need(projected_bf16.numel() % 3 == 0, "flux2_single_mod_shape");
    const std::size_t d = projected_bf16.numel() / 3;
    const int dev = projected_bf16.device();
    DeviceGuard guard(dev);
    ModulationTriple res{
        Tensor(dev, Dtype::fp32, {std::int64_t(d)}),
        Tensor(dev, Dtype::fp32, {std::int64_t(d)}),
        Tensor(dev, Dtype::bf16, {std::int64_t(d)})
    };
    split_mod_scale_shift_gate_kernel<<<div_ceil(d, 256), 256>>>(
        static_cast<const __nv_bfloat16*>(projected_bf16.data()),
        static_cast<float*>(res.shift.data()),
        static_cast<float*>(res.scale.data()),
        static_cast<__nv_bfloat16*>(res.gate.data()),
        d,
        0
    );
    cuda_check(cudaGetLastError(), "split_single_modulation");
    return res;
}

std::pair<Tensor, Tensor> split_out_modulation(const Tensor& projected_bf16) {
    need(projected_bf16.numel() % 2 == 0, "flux2_out_mod_shape");
    const std::size_t d = projected_bf16.numel() / 2;
    const int dev = projected_bf16.device();
    DeviceGuard guard(dev);
    Tensor scale(dev, Dtype::fp32, {std::int64_t(d)});
    Tensor shift(dev, Dtype::fp32, {std::int64_t(d)});
    split_out_mod_kernel<<<div_ceil(d, 256), 256>>>(
        static_cast<const __nv_bfloat16*>(projected_bf16.data()),
        static_cast<float*>(scale.data()),
        static_cast<float*>(shift.data()),
        d
    );
    cuda_check(cudaGetLastError(), "split_out_modulation");
    return {std::move(scale), std::move(shift)};
}

Tensor flux2_modulated_norm_bf16(
    const Tensor& x_bf16, const Tensor& shift_fp32, const Tensor& scale_fp32, float eps
) {
    const std::size_t d = std::size_t(x_bf16.size(-1));
    const std::size_t rows = x_bf16.numel() / d;
    const int dev = x_bf16.device();
    DeviceGuard guard(dev);
    Tensor out(dev, Dtype::bf16, x_bf16.shape());
    flux2_modulated_norm_bf16_kernel<<<int(rows), 256>>>(
        static_cast<const __nv_bfloat16*>(x_bf16.data()),
        static_cast<const float*>(shift_fp32.data()),
        static_cast<const float*>(scale_fp32.data()),
        static_cast<__nv_bfloat16*>(out.data()),
        rows,
        d,
        eps
    );
    cuda_check(cudaGetLastError(), "flux2_modulated_norm_bf16");
    return out;
}

void flux2_gated_residual_bf16(
    Tensor& x_bf16, const Tensor& delta_bf16, const Tensor& gate_bf16
) {
    const std::size_t d = std::size_t(x_bf16.size(-1));
    const std::size_t rows = x_bf16.numel() / d;
    DeviceGuard guard(x_bf16.device());
    flux2_gated_residual_bf16_kernel<<<div_ceil(rows * d, 256), 256>>>(
        static_cast<__nv_bfloat16*>(x_bf16.data()),
        static_cast<const __nv_bfloat16*>(delta_bf16.data()),
        static_cast<const __nv_bfloat16*>(gate_bf16.data()),
        rows,
        d
    );
    cuda_check(cudaGetLastError(), "flux2_gated_residual_bf16");
}

Tensor flux2_rope_bf16(
    const Tensor& x_bf16, const Tensor& cos_fp32, const Tensor& sin_fp32
) {
    need(x_bf16.dim() == 4 && x_bf16.size(3) == 128, "flux2_rope_shape");
    const std::size_t BH = std::size_t(x_bf16.size(0)) * std::size_t(x_bf16.size(1));
    const std::size_t S = std::size_t(x_bf16.size(2));
    const int dev = x_bf16.device();
    DeviceGuard guard(dev);
    Tensor out(dev, Dtype::bf16, x_bf16.shape());
    const std::size_t total = BH * S * 64;
    flux2_rope_bf16_kernel<<<div_ceil(total, 256), 256>>>(
        static_cast<const __nv_bfloat16*>(x_bf16.data()),
        static_cast<const float*>(cos_fp32.data()),
        static_cast<const float*>(sin_fp32.data()),
        static_cast<__nv_bfloat16*>(out.data()),
        BH,
        S
    );
    cuda_check(cudaGetLastError(), "flux2_rope_bf16");
    return out;
}

Tensor sdpa_flux2_bf16(const Tensor& q, const Tensor& k, const Tensor& v) {
    need(q.dim() == 4 && q.size(3) == 128, "flux2_sdpa_shape");
    const std::size_t BH = std::size_t(q.size(0)) * std::size_t(q.size(1));
    const std::size_t S = std::size_t(q.size(2));
    const int dev = q.device();
    DeviceGuard guard(dev);
    Tensor out(dev, Dtype::bf16, q.shape());
    dim3 grid(div_ceil(S, 4), unsigned(BH), 1);
    sdpa_flux2_bf16_kernel<<<grid, 128>>>(
        static_cast<const __nv_bfloat16*>(q.data()),
        static_cast<const __nv_bfloat16*>(k.data()),
        static_cast<const __nv_bfloat16*>(v.data()),
        static_cast<__nv_bfloat16*>(out.data()),
        BH,
        S
    );
    cuda_check(cudaGetLastError(), "sdpa_flux2_bf16");
    return out;
}

Tensor flux2_swiglu_bf16(const Tensor& x_bf16) {
    need(x_bf16.size(-1) % 2 == 0, "flux2_swiglu_dim");
    const std::size_t half = std::size_t(x_bf16.size(-1) / 2);
    const std::size_t rows = x_bf16.numel() / (2 * half);
    auto out_shape = x_bf16.shape();
    out_shape.back() = std::int64_t(half);
    const int dev = x_bf16.device();
    DeviceGuard guard(dev);
    Tensor out(dev, Dtype::bf16, std::move(out_shape));
    flux2_swiglu_bf16_kernel<<<div_ceil(rows * half, 256), 256>>>(
        static_cast<const __nv_bfloat16*>(x_bf16.data()),
        static_cast<__nv_bfloat16*>(out.data()),
        rows,
        half
    );
    cuda_check(cudaGetLastError(), "flux2_swiglu_bf16");
    return out;
}

Tensor concat_axis1_bf16(const Tensor& a, const Tensor& b) {
    need(a.dim() == 3 && b.dim() == 3 && a.size(0) == b.size(0) && a.size(2) == b.size(2),
         "flux2_cat1_shape");
    const std::size_t outer = std::size_t(a.size(0));
    const std::size_t la = std::size_t(a.size(1));
    const std::size_t lb = std::size_t(b.size(1));
    const std::size_t inner = std::size_t(a.size(2));
    const int dev = a.device();
    DeviceGuard guard(dev);
    Tensor out(dev, Dtype::bf16, {a.size(0), a.size(1) + b.size(1), a.size(2)});
    concat_axis1_bf16_kernel<<<div_ceil(out.numel(), 256), 256>>>(
        static_cast<const __nv_bfloat16*>(a.data()),
        static_cast<const __nv_bfloat16*>(b.data()),
        static_cast<__nv_bfloat16*>(out.data()),
        outer,
        la,
        lb,
        inner
    );
    cuda_check(cudaGetLastError(), "concat_axis1_bf16");
    return out;
}

Tensor concat_axis2_bf16(const Tensor& a, const Tensor& b) {
    need(a.dim() == 4 && b.dim() == 4 && a.size(0) == b.size(0) && a.size(1) == b.size(1)
             && a.size(3) == b.size(3),
         "flux2_cat2_shape");
    const std::size_t outer = std::size_t(a.size(0)) * std::size_t(a.size(1));
    const std::size_t la = std::size_t(a.size(2));
    const std::size_t lb = std::size_t(b.size(2));
    const std::size_t inner = std::size_t(a.size(3));
    const int dev = a.device();
    DeviceGuard guard(dev);
    Tensor out(dev, Dtype::bf16, {a.size(0), a.size(1), a.size(2) + b.size(2), a.size(3)});
    concat_axis1_bf16_kernel<<<div_ceil(out.numel(), 256), 256>>>(
        static_cast<const __nv_bfloat16*>(a.data()),
        static_cast<const __nv_bfloat16*>(b.data()),
        static_cast<__nv_bfloat16*>(out.data()),
        outer,
        la,
        lb,
        inner
    );
    cuda_check(cudaGetLastError(), "concat_axis2_bf16");
    return out;
}

Tensor slice_axis1_bf16(const Tensor& x, std::int64_t start, std::int64_t len) {
    need(x.dim() == 3 && start >= 0 && len >= 0 && start + len <= x.size(1),
         "flux2_slice1_shape");
    const int dev = x.device();
    DeviceGuard guard(dev);
    Tensor out(dev, Dtype::bf16, {x.size(0), len, x.size(2)});
    slice_axis1_bf16_kernel<<<div_ceil(out.numel(), 256), 256>>>(
        static_cast<const __nv_bfloat16*>(x.data()),
        static_cast<__nv_bfloat16*>(out.data()),
        std::size_t(x.size(0)),
        std::size_t(x.size(1)),
        std::size_t(start),
        std::size_t(len),
        std::size_t(x.size(2))
    );
    cuda_check(cudaGetLastError(), "slice_axis1_bf16");
    return out;
}

Tensor slice_last_bf16(const Tensor& x, std::int64_t start, std::int64_t len) {
    need(x.dim() >= 2 && start >= 0 && len >= 0 && start + len <= x.size(-1),
         "flux2_slicelast_shape");
    const std::size_t outer = x.numel() / std::size_t(x.size(-1));
    auto out_shape = x.shape();
    out_shape.back() = len;
    const int dev = x.device();
    DeviceGuard guard(dev);
    Tensor out(dev, Dtype::bf16, std::move(out_shape));
    slice_axis1_bf16_kernel<<<div_ceil(out.numel(), 256), 256>>>(
        static_cast<const __nv_bfloat16*>(x.data()),
        static_cast<__nv_bfloat16*>(out.data()),
        outer,
        std::size_t(x.size(-1)),
        std::size_t(start),
        std::size_t(len),
        1
    );
    cuda_check(cudaGetLastError(), "slice_last_bf16");
    return out;
}

Tensor concat_last_bf16(const Tensor& a, const Tensor& b) {
    need(a.dim() == 3 && b.dim() == 3 && a.size(0) == b.size(0) && a.size(1) == b.size(1),
         "flux2_catlast_shape");
    const std::size_t outer = std::size_t(a.size(0)) * std::size_t(a.size(1));
    const std::size_t la = std::size_t(a.size(2));
    const std::size_t lb = std::size_t(b.size(2));
    const int dev = a.device();
    DeviceGuard guard(dev);
    Tensor out(dev, Dtype::bf16, {a.size(0), a.size(1), a.size(2) + b.size(2)});
    concat_axis1_bf16_kernel<<<div_ceil(out.numel(), 256), 256>>>(
        static_cast<const __nv_bfloat16*>(a.data()),
        static_cast<const __nv_bfloat16*>(b.data()),
        static_cast<__nv_bfloat16*>(out.data()),
        outer,
        la,
        lb,
        1
    );
    cuda_check(cudaGetLastError(), "concat_last_bf16");
    return out;
}

Tensor stack_batch_bf16(const std::vector<Tensor>& items) {
    need(!items.empty() && items.front().dim() == 2, "flux2_stack_shape");
    const int dev = items.front().device();
    const std::int64_t S = items.front().size(0);
    const std::int64_t D = items.front().size(1);
    const Bytes item_bytes = items.front().nbytes();
    DeviceGuard guard(dev);
    Tensor out(dev, Dtype::bf16, {std::int64_t(items.size()), S, D});
    auto* dst = static_cast<std::uint8_t*>(out.data());
    for (std::size_t i = 0; i < items.size(); ++i) {
        need(items[i].nbytes() == item_bytes, "flux2_stack_mismatch");
        cuda_check(
            cudaMemcpy(
                dst + i * item_bytes,
                items[i].data(),
                item_bytes,
                cudaMemcpyDeviceToDevice
            ),
            "stack_batch_bf16"
        );
    }
    return out;
}

Tensor broadcast_batch_bf16(const Tensor& single_2d, std::int64_t batch) {
    need(single_2d.dim() == 2 && batch >= 1, "flux2_broadcast_shape");
    const int dev = single_2d.device();
    DeviceGuard guard(dev);
    Tensor out(dev, Dtype::bf16, {batch, single_2d.size(0), single_2d.size(1)});
    broadcast_batch_bf16_kernel<<<div_ceil(out.numel(), 256), 256>>>(
        static_cast<const __nv_bfloat16*>(single_2d.data()),
        static_cast<__nv_bfloat16*>(out.data()),
        std::size_t(batch),
        single_2d.numel()
    );
    cuda_check(cudaGetLastError(), "broadcast_batch_bf16");
    return out;
}

// ---------- Euler & edit mask ----------

void euler_step_fp32(Tensor& latent, const Tensor& velocity, float dt) {
    need(latent.numel() == velocity.numel(), "flux2_euler_shape");
    DeviceGuard guard(latent.device());
    euler_step_fp32_kernel<<<div_ceil(latent.numel(), 256), 256>>>(
        static_cast<float*>(latent.data()),
        static_cast<const float*>(velocity.data()),
        dt,
        latent.numel()
    );
    cuda_check(cudaGetLastError(), "euler_step_fp32");
}

void mix_source_noise_fp32(
    Tensor& latent, const Tensor& source, const Tensor& noise, float sigma
) {
    need(latent.numel() == source.numel() && latent.numel() == noise.numel(),
         "flux2_mix_shape");
    DeviceGuard guard(latent.device());
    mix_source_noise_fp32_kernel<<<div_ceil(latent.numel(), 256), 256>>>(
        static_cast<float*>(latent.data()),
        static_cast<const float*>(source.data()),
        static_cast<const float*>(noise.data()),
        sigma,
        latent.numel()
    );
    cuda_check(cudaGetLastError(), "mix_source_noise_fp32");
}

void apply_latent_mask_fp32(
    Tensor& latent,
    const Tensor& source,
    const Tensor& noise,
    const Tensor& mask,
    float next_sigma
) {
    need(latent.dim() == 3 && latent.size(0) == 1, "flux2_mask_shape");
    const std::size_t points = std::size_t(latent.size(1));
    const std::size_t channels = std::size_t(latent.size(2));
    DeviceGuard guard(latent.device());
    apply_latent_mask_fp32_kernel<<<div_ceil(points * channels, 256), 256>>>(
        static_cast<float*>(latent.data()),
        static_cast<const float*>(source.data()),
        static_cast<const float*>(noise.data()),
        static_cast<const float*>(mask.data()),
        next_sigma,
        points,
        channels
    );
    cuda_check(cudaGetLastError(), "apply_latent_mask_fp32");
}

// ---------- VAE ops (FP32) ----------

Tensor group_norm_32_fp32(
    const Tensor& x,
    const Tensor& weight,
    const Tensor& bias,
    bool apply_silu,
    float eps
) {
    need(x.dim() == 4 && x.size(0) == 1 && x.size(1) % 32 == 0, "flux2_gn_shape");
    const std::size_t C = std::size_t(x.size(1));
    const std::size_t HW = std::size_t(x.size(2)) * std::size_t(x.size(3));
    const int dev = x.device();
    DeviceGuard guard(dev);
    Tensor out(dev, Dtype::fp32, x.shape());
    group_norm_32_fp32_kernel<<<32, 256>>>(
        static_cast<const float*>(x.data()),
        static_cast<const float*>(weight.data()),
        static_cast<const float*>(bias.data()),
        static_cast<float*>(out.data()),
        C,
        HW,
        apply_silu,
        eps
    );
    cuda_check(cudaGetLastError(), "group_norm_32_fp32");
    return out;
}

Tensor conv2d_fp32(
    const Tensor& x, const Tensor& weight, const Tensor& bias, bool downsample
) {
    need(x.dim() == 4 && x.size(0) == 1 && weight.dim() == 4, "flux2_conv_rank");
    const std::size_t C_in = std::size_t(x.size(1));
    const std::size_t H = std::size_t(x.size(2));
    const std::size_t W = std::size_t(x.size(3));
    const std::size_t C_out = std::size_t(weight.size(0));
    const std::size_t kH = std::size_t(weight.size(2));
    const std::size_t kW = std::size_t(weight.size(3));
    need(std::size_t(weight.size(1)) == C_in, "flux2_conv_cin");

    const int dev = x.device();
    DeviceGuard guard(dev);
    cublasHandle_t blas = get_blas(dev);
    const float alpha = 1.0f;
    const float beta = 0.0f;

    if (kH == 1 && kW == 1 && !downsample) {
        const std::size_t HW = H * W;
        Tensor out(dev, Dtype::fp32, {1, std::int64_t(C_out), std::int64_t(H), std::int64_t(W)});
        blas_check(
            cublasGemmEx(
                blas,
                CUBLAS_OP_N,
                CUBLAS_OP_N,
                int(HW),
                int(C_out),
                int(C_in),
                &alpha,
                x.data(),
                CUDA_R_32F,
                int(HW),
                weight.data(),
                CUDA_R_32F,
                int(C_in),
                &beta,
                out.data(),
                CUDA_R_32F,
                int(HW),
                CUBLAS_COMPUTE_32F_PEDANTIC,
                CUBLAS_GEMM_DEFAULT
            ),
            "cublasGemmEx_conv1x1"
        );
        add_channel_bias_nchw_kernel<<<div_ceil(C_out * HW, 256), 256>>>(
            static_cast<float*>(out.data()),
            static_cast<const float*>(bias.data()),
            C_out,
            HW
        );
        cuda_check(cudaGetLastError(), "conv1x1_bias");
        return out;
    }

    need(kH == 3 && kW == 3, "flux2_conv_kernel_size");
    const std::size_t H_out = downsample ? H / 2 : H;
    const std::size_t W_out = downsample ? W / 2 : W;
    const std::size_t HW_out = H_out * W_out;
    const std::size_t K = C_in * 9;
    const int stride = downsample ? 2 : 1;
    const int pad_top = downsample ? 0 : 1;
    const int pad_left = downsample ? 0 : 1;

    Tensor out(
        dev, Dtype::fp32, {1, std::int64_t(C_out), std::int64_t(H_out), std::int64_t(W_out)}
    );

    // Cap im2col scratch at 128 MiB so full-resolution 1024x1024 layers stay tight.
    constexpr Bytes kMaxColBytes = 128ULL * 1024 * 1024;
    const std::size_t max_chunk_p =
        std::max<std::size_t>(1, std::size_t(kMaxColBytes / (K * sizeof(float))));
    const std::size_t col_p = std::min(HW_out, max_chunk_p);
    Tensor col(dev, Dtype::fp32, {std::int64_t(K), std::int64_t(col_p)});

    for (std::size_t p_start = 0; p_start < HW_out; p_start += col_p) {
        const std::size_t chunk_p = std::min(col_p, HW_out - p_start);
        im2col_3x3_chunk_kernel<<<div_ceil(K * chunk_p, 256), 256>>>(
            static_cast<const float*>(x.data()),
            static_cast<float*>(col.data()),
            C_in,
            H,
            W,
            H_out,
            W_out,
            p_start,
            chunk_p,
            stride,
            pad_top,
            pad_left
        );
        cuda_check(cudaGetLastError(), "im2col_3x3");
        blas_check(
            cublasGemmEx(
                blas,
                CUBLAS_OP_N,
                CUBLAS_OP_N,
                int(chunk_p),
                int(C_out),
                int(K),
                &alpha,
                col.data(),
                CUDA_R_32F,
                int(chunk_p),
                weight.data(),
                CUDA_R_32F,
                int(K),
                &beta,
                static_cast<float*>(out.data()) + p_start,
                CUDA_R_32F,
                int(HW_out),
                CUBLAS_COMPUTE_32F_PEDANTIC,
                CUBLAS_GEMM_DEFAULT
            ),
            "cublasGemmEx_conv3x3"
        );
    }

    add_channel_bias_nchw_kernel<<<div_ceil(C_out * HW_out, 256), 256>>>(
        static_cast<float*>(out.data()),
        static_cast<const float*>(bias.data()),
        C_out,
        HW_out
    );
    cuda_check(cudaGetLastError(), "conv3x3_bias");
    return out;
}

Tensor upsample_nearest2d_fp32(const Tensor& x) {
    need(x.dim() == 4 && x.size(0) == 1, "flux2_upsample_shape");
    const std::size_t C = std::size_t(x.size(1));
    const std::size_t H = std::size_t(x.size(2));
    const std::size_t W = std::size_t(x.size(3));
    const int dev = x.device();
    DeviceGuard guard(dev);
    Tensor out(dev, Dtype::fp32, {1, x.size(1), x.size(2) * 2, x.size(3) * 2});
    upsample_nearest2d_kernel<<<div_ceil(out.numel(), 256), 256>>>(
        static_cast<const float*>(x.data()),
        static_cast<float*>(out.data()),
        C,
        H,
        W
    );
    cuda_check(cudaGetLastError(), "upsample_nearest2d");
    return out;
}

Tensor vae_sdpa_fp32(
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    const std::atomic_bool& cancel
) {
    need(q.dim() == 2 && q.size(1) == 512, "flux2_vae_sdpa_shape");
    const std::size_t points = std::size_t(q.size(0));
    const int dev = q.device();
    DeviceGuard guard(dev);
    Tensor out(dev, Dtype::fp32, q.shape());
    constexpr std::size_t kChunk = 1024;
    for (std::size_t row = 0; row < points; row += kChunk) {
        need(!cancel.load(), "image_cancelled");
        const std::size_t count = std::min(kChunk, points - row);
        vae_sdpa_fp32_kernel<<<div_ceil(count, 4), 128>>>(
            static_cast<const float*>(q.data()),
            static_cast<const float*>(k.data()),
            static_cast<const float*>(v.data()),
            static_cast<float*>(out.data()),
            points,
            row,
            count
        );
        cuda_check(cudaGetLastError(), "vae_sdpa_fp32");
    }
    return out;
}

Tensor nchw_to_seq_fp32(const Tensor& x) {
    need(x.dim() == 4 && x.size(0) == 1, "flux2_nchw_to_seq");
    const std::size_t C = std::size_t(x.size(1));
    const std::size_t HW = std::size_t(x.size(2)) * std::size_t(x.size(3));
    const int dev = x.device();
    DeviceGuard guard(dev);
    Tensor out(dev, Dtype::fp32, {std::int64_t(HW), std::int64_t(C)});
    nchw_to_seq_kernel<<<div_ceil(C * HW, 256), 256>>>(
        static_cast<const float*>(x.data()), static_cast<float*>(out.data()), C, HW
    );
    cuda_check(cudaGetLastError(), "nchw_to_seq_fp32");
    return out;
}

Tensor seq_to_nchw_fp32(
    const Tensor& x, std::int64_t c, std::int64_t h, std::int64_t w
) {
    const std::size_t C = std::size_t(c);
    const std::size_t HW = std::size_t(h) * std::size_t(w);
    need(x.numel() == C * HW, "flux2_seq_to_nchw");
    const int dev = x.device();
    DeviceGuard guard(dev);
    Tensor out(dev, Dtype::fp32, {1, c, h, w});
    seq_to_nchw_kernel<<<div_ceil(C * HW, 256), 256>>>(
        static_cast<const float*>(x.data()), static_cast<float*>(out.data()), C, HW
    );
    cuda_check(cudaGetLastError(), "seq_to_nchw_fp32");
    return out;
}

Tensor vae_patchify_bn_norm_fp32(
    const Tensor& x_32, const Tensor& bn_mean, const Tensor& bn_var, float bn_eps
) {
    need(x_32.dim() == 4 && x_32.size(1) == 32, "flux2_patchify_shape");
    const std::size_t h = std::size_t(x_32.size(2) / 2);
    const std::size_t w = std::size_t(x_32.size(3) / 2);
    const int dev = x_32.device();
    DeviceGuard guard(dev);
    Tensor out(dev, Dtype::fp32, {std::int64_t(h * w), 128});
    vae_patchify_bn_norm_kernel<<<div_ceil(h * w * 128, 256), 256>>>(
        static_cast<const float*>(x_32.data()),
        static_cast<const float*>(bn_mean.data()),
        static_cast<const float*>(bn_var.data()),
        bn_eps,
        static_cast<float*>(out.data()),
        h,
        w
    );
    cuda_check(cudaGetLastError(), "vae_patchify_bn_norm_fp32");
    return out;
}

Tensor vae_bn_denorm_unpatchify_fp32(
    const Tensor& latent_hw_128,
    std::int64_t h,
    std::int64_t w,
    const Tensor& bn_mean,
    const Tensor& bn_var,
    float bn_eps
) {
    need(latent_hw_128.numel() == std::size_t(h * w * 128), "flux2_unpatchify_shape");
    const int dev = latent_hw_128.device();
    DeviceGuard guard(dev);
    Tensor out(dev, Dtype::fp32, {1, 32, h * 2, w * 2});
    vae_bn_denorm_unpatchify_kernel<<<div_ceil(out.numel(), 256), 256>>>(
        static_cast<const float*>(latent_hw_128.data()),
        static_cast<const float*>(bn_mean.data()),
        static_cast<const float*>(bn_var.data()),
        bn_eps,
        static_cast<float*>(out.data()),
        std::size_t(h),
        std::size_t(w)
    );
    cuda_check(cudaGetLastError(), "vae_bn_denorm_unpatchify_fp32");
    return out;
}

Tensor vae_clamp_rgba_fp32(const Tensor& rgb) {
    need(rgb.dim() == 4 && rgb.size(0) == 1 && rgb.size(1) == 3, "flux2_rgba_shape");
    const std::size_t HW = std::size_t(rgb.size(2)) * std::size_t(rgb.size(3));
    const int dev = rgb.device();
    DeviceGuard guard(dev);
    Tensor out(dev, Dtype::fp32, {1, 4, rgb.size(2), rgb.size(3)});
    vae_clamp_rgba_kernel<<<div_ceil(4 * HW, 256), 256>>>(
        static_cast<const float*>(rgb.data()), static_cast<float*>(out.data()), HW
    );
    cuda_check(cudaGetLastError(), "vae_clamp_rgba_fp32");
    return out;
}

Tensor slice_channel_fp32(
    const Tensor& x_nchw, std::int64_t c_start, std::int64_t c_len
) {
    need(
        x_nchw.dim() == 4 && x_nchw.size(0) == 1 && c_start >= 0 && c_len > 0
            && c_start + c_len <= x_nchw.size(1),
        "flux2_slice_channel"
    );
    const std::size_t HW = std::size_t(x_nchw.size(2)) * std::size_t(x_nchw.size(3));
    const int dev = x_nchw.device();
    DeviceGuard guard(dev);
    Tensor out(dev, Dtype::fp32, {1, c_len, x_nchw.size(2), x_nchw.size(3)});
    cuda_check(
        cudaMemcpy(
            out.data(),
            static_cast<const float*>(x_nchw.data()) + std::size_t(c_start) * HW,
            out.nbytes(),
            cudaMemcpyDeviceToDevice
        ),
        "slice_channel_fp32"
    );
    return out;
}

Tensor crop_2d_fp32(
    const Tensor& x_nchw,
    std::int64_t y_start,
    std::int64_t y_len,
    std::int64_t x_start,
    std::int64_t x_len
) {
    need(x_nchw.dim() == 4 && x_nchw.size(0) == 1, "flux2_crop_shape");
    const int dev = x_nchw.device();
    DeviceGuard guard(dev);
    Tensor out(dev, Dtype::fp32, {1, x_nchw.size(1), y_len, x_len});
    crop_2d_kernel<<<div_ceil(out.numel(), 256), 256>>>(
        static_cast<const float*>(x_nchw.data()),
        static_cast<float*>(out.data()),
        std::size_t(x_nchw.size(1)),
        std::size_t(x_nchw.size(2)),
        std::size_t(x_nchw.size(3)),
        std::size_t(y_start),
        std::size_t(y_len),
        std::size_t(x_start),
        std::size_t(x_len)
    );
    cuda_check(cudaGetLastError(), "crop_2d_fp32");
    return out;
}

void paste_2d_fp32(
    Tensor& dst_nchw,
    const Tensor& src_nchw,
    std::int64_t dst_y,
    std::int64_t dst_x,
    std::int64_t src_y,
    std::int64_t src_x,
    std::int64_t h_len,
    std::int64_t w_len
) {
    need(
        dst_nchw.dim() == 4 && src_nchw.dim() == 4 && dst_nchw.size(1) == src_nchw.size(1),
        "flux2_paste_shape"
    );
    DeviceGuard guard(dst_nchw.device());
    const std::size_t total =
        std::size_t(dst_nchw.size(1)) * std::size_t(h_len) * std::size_t(w_len);
    paste_2d_kernel<<<div_ceil(total, 256), 256>>>(
        static_cast<float*>(dst_nchw.data()),
        static_cast<const float*>(src_nchw.data()),
        std::size_t(dst_nchw.size(1)),
        std::size_t(dst_nchw.size(2)),
        std::size_t(dst_nchw.size(3)),
        std::size_t(src_nchw.size(2)),
        std::size_t(src_nchw.size(3)),
        std::size_t(dst_y),
        std::size_t(dst_x),
        std::size_t(src_y),
        std::size_t(src_x),
        std::size_t(h_len),
        std::size_t(w_len)
    );
    cuda_check(cudaGetLastError(), "paste_2d_fp32");
}

}  // namespace kadan::image::gpu
