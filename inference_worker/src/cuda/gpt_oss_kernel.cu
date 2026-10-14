// GPT-OSS equations: Transformers models/gpt_oss/modeling_gpt_oss.py and the
// OpenAI gpt_oss reference. FP32 activations and accumulation over BF16 and
// MXFP4 checkpoint weights; K/V are retained in BF16 like the reference cache.
#include <cuda_runtime.h>
#include <math_constants.h>

#include "gpt_oss_kernel.cuh"

namespace kadan::cuda::detail::gpt_oss {
namespace {
constexpr unsigned hidden = 2880, heads = 64, kv_heads = 8, head_dim = 64,
                   group = heads / kv_heads, top_k = 4, intermediate = 2880,
                   gate_up_rows = 2 * intermediate, groups = hidden / 32,
                   block_bytes = groups * 16, chunk = 256, partial = head_dim + 2;
constexpr float limit = 7.F, alpha = 1.702F;

__device__ float widen(std::uint16_t value) {
    return __uint_as_float(unsigned(value) << 16);
}

__device__ std::uint16_t narrow(float value) {
    unsigned bits = __float_as_uint(value);
    bits += 0x7fffu + ((bits >> 16) & 1u);
    return std::uint16_t(bits >> 16);
}

__device__ float warp_sum(float value) {
    for (unsigned offset = 16; offset; offset >>= 1)
        value += __shfl_xor_sync(0xffffffffu, value, offset);
    return value;
}

__device__ float warp_max(float value) {
    for (unsigned offset = 16; offset; offset >>= 1)
        value = fmaxf(value, __shfl_xor_sync(0xffffffffu, value, offset));
    return value;
}

// Every thread receives the block result; blockDim is a multiple of 32.
template <bool maximum> __device__ float block_reduce(float value, float* shared) {
    const unsigned lane = threadIdx.x % 32, warp = threadIdx.x / 32,
                   warps = blockDim.x / 32;
    value = maximum ? warp_max(value) : warp_sum(value);
    __syncthreads();
    if (!lane)
        shared[warp] = value;
    __syncthreads();
    value = lane < warps ? shared[lane] : (maximum ? -CUDART_INF_F : 0.F);
    return maximum ? warp_max(value) : warp_sum(value);
}

__global__ void embed_kernel(const std::uint16_t* table, unsigned token, float* x) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < hidden)
        x[i] = widen(table[std::size_t(token) * hidden + i]);
}

__global__ void norm_kernel(const float* x, const std::uint16_t* weight, float* y) {
    __shared__ float shared[32];
    float sum = 0;
    for (unsigned i = threadIdx.x; i < hidden; i += blockDim.x)
        sum += x[i] * x[i];
    const float scale = rsqrtf(block_reduce<false>(sum, shared) / hidden + 1e-5F);
    for (unsigned i = threadIdx.x; i < hidden; i += blockDim.x)
        y[i] = widen(weight[i]) * (x[i] * scale);
}

// One warp per output row; the input vector is staged in shared memory.
__global__ void linear_kernel(
    const std::uint16_t* weight,
    const std::uint16_t* bias,
    const float* x,
    float* y,
    unsigned rows,
    unsigned cols,
    bool accumulate
) {
    extern __shared__ float input[];
    for (unsigned i = threadIdx.x; i < cols; i += blockDim.x)
        input[i] = x[i];
    __syncthreads();
    const unsigned lane = threadIdx.x % 32,
                   row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    if (row >= rows)
        return;
    const auto* packed =
        reinterpret_cast<const uint4*>(weight + std::size_t(row) * cols);
    float sum = 0;
    for (unsigned c = lane; c < cols / 8; c += 32) {
        const uint4 p = packed[c];
        const float* v = input + 8 * c;
        const unsigned words[4] = {p.x, p.y, p.z, p.w};
        for (unsigned k = 0; k < 4; ++k) {
            sum += __uint_as_float(words[k] << 16) * v[2 * k];
            sum += __uint_as_float(words[k] & 0xffff0000u) * v[2 * k + 1];
        }
    }
    sum = warp_sum(sum);
    if (!lane) {
        if (bias)
            sum += widen(bias[row]);
        y[row] = accumulate ? y[row] + sum : sum;
    }
}

__global__ void rotary_kernel(
    float* qkv, const float* inverse_frequency, float scaling, float position
) {
    const unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= (heads + kv_heads) * head_dim / 2)
        return;
    const unsigned head = index / (head_dim / 2), i = index % (head_dim / 2);
    float sine, cosine;
    sincosf(position * inverse_frequency[i], &sine, &cosine);
    sine *= scaling;
    cosine *= scaling;
    float* h = qkv + head * head_dim;
    const float first = h[i], second = h[i + head_dim / 2];
    h[i] = first * cosine - second * sine;
    h[i + head_dim / 2] = second * cosine + first * sine;
}

__global__ void store_kernel(
    const float* qkv, std::uint16_t* keys, std::uint16_t* values, std::size_t slot
) {
    const unsigned i = threadIdx.x;
    const float* k = qkv + heads * head_dim;
    keys[slot * kv_heads * head_dim + i] = narrow(k[i]);
    values[slot * kv_heads * head_dim + i] = narrow(k[kv_heads * head_dim + i]);
}

// One block per (chunk, head): local max, exponent sum and weighted values.
__global__ void attention_partial(
    const float* qkv,
    const std::uint16_t* keys,
    const std::uint16_t* values,
    unsigned count,
    float* out
) {
    __shared__ float query[head_dim], probability[chunk], reduce[32],
        sums[chunk / head_dim][head_dim];
    const unsigned head = blockIdx.y, kv = head / group, tid = threadIdx.x,
                   first = blockIdx.x * chunk, key = first + tid;
    if (tid < head_dim)
        query[tid] = qkv[head * head_dim + tid] * 0.125F;
    __syncthreads();
    float score = -CUDART_INF_F;
    if (key < count) {
        const auto* k = reinterpret_cast<const uint4*>(
            keys + (std::size_t(key) * kv_heads + kv) * head_dim
        );
        score = 0;
        for (unsigned c = 0; c < head_dim / 8; ++c) {
            const uint4 p = k[c];
            const unsigned words[4] = {p.x, p.y, p.z, p.w};
            for (unsigned j = 0; j < 4; ++j) {
                score += __uint_as_float(words[j] << 16) * query[8 * c + 2 * j];
                score +=
                    __uint_as_float(words[j] & 0xffff0000u) * query[8 * c + 2 * j + 1];
            }
        }
    }
    const float maximum = block_reduce<true>(score, reduce);
    const float e = key < count ? __expf(score - maximum) : 0.F;
    probability[tid] = e;
    const float total = block_reduce<false>(e, reduce);
    const unsigned d = tid % head_dim, lane_group = tid / head_dim;
    float value = 0;
    for (unsigned j = lane_group; j < chunk && first + j < count; j += chunk / head_dim)
        value +=
            probability[j]
            * widen(values[(std::size_t(first + j) * kv_heads + kv) * head_dim + d]);
    sums[lane_group][d] = value;
    __syncthreads();
    if (tid < head_dim) {
        float* result = out + (std::size_t(head) * gridDim.x + blockIdx.x) * partial;
        float o = 0;
        for (unsigned g = 0; g < chunk / head_dim; ++g)
            o += sums[g][tid];
        result[2 + tid] = o;
        if (!tid) {
            result[0] = maximum;
            result[1] = total;
        }
    }
}

// The sink joins each head's softmax denominator but contributes no value.
__global__ void attention_combine(
    const float* in, unsigned chunks, const std::uint16_t* sinks, float* output
) {
    const unsigned head = blockIdx.x, d = threadIdx.x;
    const float* p = in + std::size_t(head) * chunks * partial;
    const float sink = widen(sinks[head]);
    float maximum = sink;
    for (unsigned c = 0; c < chunks; ++c)
        maximum = fmaxf(maximum, p[c * partial]);
    float total = __expf(sink - maximum), value = 0;
    for (unsigned c = 0; c < chunks; ++c) {
        const float scale = __expf(p[c * partial] - maximum);
        total += p[c * partial + 1] * scale;
        value += p[c * partial + 2 + d] * scale;
    }
    output[head * head_dim + d] = value / total;
}

__global__ void route_kernel(
    const float* logits,
    unsigned experts,
    unsigned* indices,
    float* weights,
    unsigned* status
) {
    if (threadIdx.x)
        return;
    unsigned chosen[top_k];
    float values[top_k];
    for (unsigned rank = 0; rank < top_k; ++rank) {
        unsigned best = experts;
        for (unsigned e = 0; e < experts; ++e) {
            bool taken = false;
            for (unsigned r = 0; r < rank; ++r)
                taken = taken || chosen[r] == e;
            if (!isfinite(logits[e]))
                atomicOr(status, 1u);
            if (!taken && (best == experts || logits[e] > logits[best]))
                best = e;
        }
        chosen[rank] = best;
        values[rank] = logits[best];
    }
    float total = 0;
    for (unsigned rank = 0; rank < top_k; ++rank) {
        values[rank] = expf(values[rank] - values[0]);
        total += values[rank];
    }
    for (unsigned rank = 0; rank < top_k; ++rank) {
        indices[rank] = chosen[rank];
        weights[rank] = values[rank] / total;
    }
}

__device__ const float fp4[16] = {
    0.F,
    0.5F,
    1.F,
    1.5F,
    2.F,
    3.F,
    4.F,
    6.F,
    -0.F,
    -0.5F,
    -1.F,
    -1.5F,
    -2.F,
    -3.F,
    -4.F,
    -6.F
};

// One MXFP4 row against a shared FP32 vector; the caller reduces the warp.
// Low nibbles precede high nibbles; each 32-column block has an E8M0 scale.
__device__ float mxfp4_row(
    const std::uint8_t* blocks,
    const std::uint8_t* scales,
    const float* input,
    const float* table,
    unsigned lane
) {
    float sum = 0;
    for (unsigned g = lane; g < groups; g += 32) {
        const uint4 p = *reinterpret_cast<const uint4*>(blocks + g * 16);
        const unsigned words[4] = {p.x, p.y, p.z, p.w};
        const float* v = input + g * 32;
        float block = 0;
        for (unsigned w = 0; w < 4; ++w)
            for (unsigned b = 0; b < 4; ++b) {
                const unsigned byte = (words[w] >> (8 * b)) & 0xffu;
                block += table[byte & 15u] * v[8 * w + 2 * b];
                block += table[byte >> 4] * v[8 * w + 2 * b + 1];
            }
        sum += ldexpf(block, int(scales[g]) - 127);
    }
    return sum;
}

// One warp per interleaved (gate, up) pair, for each selected expert.
__global__ void gate_up_kernel(
    const std::uint8_t* const* experts,
    const std::uint16_t* bias,
    const unsigned* indices,
    const float* x,
    float* activation
) {
    __shared__ float input[hidden], table[16];
    for (unsigned i = threadIdx.x; i < hidden; i += blockDim.x)
        input[i] = x[i];
    if (threadIdx.x < 16)
        table[threadIdx.x] = fp4[threadIdx.x];
    __syncthreads();
    const unsigned lane = threadIdx.x % 32,
                   pair = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    if (pair >= intermediate)
        return;
    for (unsigned slot = 0; slot < top_k; ++slot) {
        const unsigned expert = indices[slot];
        const std::uint8_t* blocks = experts[expert];
        const std::uint8_t* scales = blocks + std::size_t(gate_up_rows) * block_bytes;
        float gate = mxfp4_row(
            blocks + std::size_t(2 * pair) * block_bytes,
            scales + std::size_t(2 * pair) * groups,
            input,
            table,
            lane
        );
        float up = mxfp4_row(
            blocks + std::size_t(2 * pair + 1) * block_bytes,
            scales + std::size_t(2 * pair + 1) * groups,
            input,
            table,
            lane
        );
        gate = warp_sum(gate);
        up = warp_sum(up);
        if (!lane) {
            const auto* b = bias + std::size_t(expert) * gate_up_rows + 2 * pair;
            gate = fminf(gate + widen(b[0]), limit);
            up = fminf(fmaxf(up + widen(b[1]), -limit), limit);
            const float glu = gate / (1.F + __expf(-alpha * gate));
            activation[slot * intermediate + pair] = (up + 1.F) * glu;
        }
    }
}

// One warp per output channel: all selected experts, weighted, into residual.
__global__ void down_kernel(
    const std::uint8_t* const* experts,
    const std::uint16_t* bias,
    const unsigned* indices,
    const float* weights,
    const float* activation,
    float* residual
) {
    __shared__ float input[top_k * intermediate], table[16];
    for (unsigned i = threadIdx.x; i < top_k * intermediate; i += blockDim.x)
        input[i] = activation[i];
    if (threadIdx.x < 16)
        table[threadIdx.x] = fp4[threadIdx.x];
    __syncthreads();
    const unsigned lane = threadIdx.x % 32,
                   row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    if (row >= hidden)
        return;
    float total = 0;
    for (unsigned slot = 0; slot < top_k; ++slot) {
        const unsigned expert = indices[slot];
        const std::uint8_t* blocks =
            experts[expert] + std::size_t(gate_up_rows) * (block_bytes + groups);
        const std::uint8_t* scales = blocks + std::size_t(hidden) * block_bytes;
        const float value = warp_sum(mxfp4_row(
            blocks + std::size_t(row) * block_bytes,
            scales + std::size_t(row) * groups,
            input + slot * intermediate,
            table,
            lane
        ));
        total +=
            weights[slot] * (value + widen(bias[std::size_t(expert) * hidden + row]));
    }
    if (!lane)
        residual[row] += total;
}
}  // namespace

cudaError_t embed(const std::uint16_t* table, unsigned token, float* x) {
    embed_kernel<<<(hidden + 255) / 256, 256, 0, cudaStreamLegacy>>>(table, token, x);
    return cudaGetLastError();
}

cudaError_t norm(const float* x, const std::uint16_t* weight, float* y) {
    norm_kernel<<<1, 1024, 0, cudaStreamLegacy>>>(x, weight, y);
    return cudaGetLastError();
}

cudaError_t linear(
    const std::uint16_t* weight,
    const std::uint16_t* bias,
    const float* x,
    float* y,
    std::size_t rows,
    std::size_t cols,
    bool accumulate
) {
    if (!rows || !cols || cols % 8 || cols > 4096 || rows > 0x7fffffffu)
        return cudaErrorInvalidValue;
    constexpr unsigned rows_per_block = 8;
    linear_kernel<<<
        unsigned((rows + rows_per_block - 1) / rows_per_block),
        rows_per_block * 32,
        cols * sizeof(float),
        cudaStreamLegacy>>>(
        weight, bias, x, y, unsigned(rows), unsigned(cols), accumulate
    );
    return cudaGetLastError();
}

cudaError_t rotary(
    float* qkv, const float* inverse_frequency, float scaling, std::size_t position
) {
    constexpr unsigned pairs = (heads + kv_heads) * head_dim / 2;
    rotary_kernel<<<(pairs + 255) / 256, 256, 0, cudaStreamLegacy>>>(
        qkv, inverse_frequency, scaling, float(position)
    );
    return cudaGetLastError();
}

cudaError_t
store(const float* qkv, std::uint16_t* keys, std::uint16_t* values, std::size_t slot) {
    store_kernel<<<1, kv_heads * head_dim, 0, cudaStreamLegacy>>>(
        qkv, keys, values, slot
    );
    return cudaGetLastError();
}

cudaError_t attention(
    const float* qkv,
    const std::uint16_t* keys,
    const std::uint16_t* values,
    const std::uint16_t* sinks,
    std::size_t count,
    float* partial_out,
    float* output
) {
    if (!count || count > 0x7fffffffu)
        return cudaErrorInvalidValue;
    const unsigned chunks = unsigned((count + chunk - 1) / chunk);
    attention_partial<<<dim3(chunks, heads), chunk, 0, cudaStreamLegacy>>>(
        qkv, keys, values, unsigned(count), partial_out
    );
    if (const auto error = cudaGetLastError(); error != cudaSuccess)
        return error;
    attention_combine<<<heads, head_dim, 0, cudaStreamLegacy>>>(
        partial_out, chunks, sinks, output
    );
    return cudaGetLastError();
}

cudaError_t route(
    const float* logits,
    std::size_t experts,
    unsigned* indices,
    float* weights,
    unsigned* status
) {
    if (experts < top_k || experts > 128)
        return cudaErrorInvalidValue;
    route_kernel<<<1, 32, 0, cudaStreamLegacy>>>(
        logits, unsigned(experts), indices, weights, status
    );
    return cudaGetLastError();
}

cudaError_t experts(
    const std::uint8_t* const* table,
    const std::uint16_t* gate_up_bias,
    const std::uint16_t* down_bias,
    const unsigned* indices,
    const float* weights,
    const float* x,
    float* activation,
    float* residual
) {
    constexpr unsigned warps = 8;
    gate_up_kernel<<<
        (intermediate + warps - 1) / warps,
        warps * 32,
        0,
        cudaStreamLegacy>>>(table, gate_up_bias, indices, x, activation);
    if (const auto error = cudaGetLastError(); error != cudaSuccess)
        return error;
    down_kernel<<<(hidden + warps - 1) / warps, warps * 32, 0, cudaStreamLegacy>>>(
        table, down_bias, indices, weights, activation, residual
    );
    return cudaGetLastError();
}
}  // namespace kadan::cuda::detail::gpt_oss
