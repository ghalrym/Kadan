#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cuda_bf16.h>
#include <future>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "../models/image/image_activation.hpp"
#include "kadan/bf16_projection.hpp"
#include "kadan/device_placement.hpp"
#include "kadan/device_weights.hpp"
#include "kadan/device_workspace.hpp"
#include "kadan/gguf_quantization.hpp"
#include "kadan/h3_compute.hpp"
#include "kadan/h3_resident.hpp"
#include "kadan/image_vae.hpp"
#include "kadan/quantization.hpp"

namespace kadan::video {
namespace {
void require(bool b, const char* message) {
    if (!b)
        throw std::runtime_error(message);
}

void checked(cudaError_t code) {
    if (code != cudaSuccess)
        throw std::runtime_error(std::string("h3_cuda: ") + cudaGetErrorString(code));
}

void checked(cublasStatus_t code) {
    if (code != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error("h3_cublas_" + std::to_string(int(code)));
}

void stop(const std::atomic_bool& cancel, const std::atomic_bool& failed) {
    require(!cancel.load() && !failed.load(), "h3_cuda_cancelled");
}

std::size_t product(std::size_t a, std::size_t b) {
    require(
        !b || a <= std::numeric_limits<std::size_t>::max() / b, "h3_cuda_size_overflow"
    );
    return a * b;
}

bool overlap(const void* a, std::size_t an, const void* b, std::size_t bn) {
    if (!an || !bn)
        return false;
    const auto x = reinterpret_cast<std::uintptr_t>(a),
               y = reinterpret_cast<std::uintptr_t>(b);
    return x <= y ? y - x < an : x - y < bn;
}

__global__ void
bf16_input(const float* source, __nv_bfloat16* destination, std::size_t count) {
    const auto i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count)
        destination[i] = __float2bfloat16_rn(source[i]);
}

// Packed rows remain in the weight bank; only this bounded tile becomes F32.
__global__ void gguf_decode_kernel(
    const unsigned char* packed,
    float* decoded,
    unsigned* status,
    gguf::Encoding encoding,
    std::size_t count
) {
    const auto i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count)
        return;
    const auto block_values = gguf::block_values(encoding);
    const float value = gguf::decode(
        encoding,
        packed + i / block_values * gguf::block_bytes(encoding),
        i % block_values
    );
    decoded[i] = value;
    if (!isfinite(value))
        atomicOr(status, 1U);
}

struct Layout {
    std::size_t bytes = 0;

    std::size_t add(std::size_t n) {
        require(
            n <= std::numeric_limits<std::size_t>::max() - bytes - 255,
            "h3_cuda_size_overflow"
        );
        const auto at = bytes;
        bytes += (n + 255) / 256 * 256;
        return at;
    }
};

__global__ void image_vae_gather_kernel(
    const float* raw,
    float* patches,
    unsigned* status,
    image::VaeConvolution shape,
    image::VaePatchRegion region,
    std::size_t first,
    std::size_t rows
) {
    const auto index = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= rows * shape.columns())
        return;
    const auto column = index % shape.columns();
    const auto source =
        image::vae_patch_index(shape, first + index / shape.columns(), column);
    float value = 0;
    if (source != image::vae_patch_padding) {
        const auto channel = column / (shape.kernel * shape.kernel);
        const auto point = source - channel * shape.points();
        value =
            raw[channel * region.points()
                + (point / shape.width - region.top) * region.width
                + point % shape.width - region.left];
    }
    patches[index] = value;
    if (!isfinite(value))
        atomicOr(status, 1U);
}

__global__ void image_vae_scatter_kernel(
    const float* matrix,
    float* planar,
    std::size_t rows,
    std::size_t columns,
    std::size_t first,
    std::size_t stride
) {
    const auto index = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < rows * columns)
        planar[image::vae_planar_group_index(index, columns, first, stride)] =
            matrix[index];
}

__global__ void image_vae_norm_kernel(
    const float* input,
    const float* gamma,
    float* output,
    unsigned* status,
    std::size_t channels,
    std::size_t stride,
    std::size_t count,
    bool activate
) {
    const auto point = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (point >= count)
        return;
    double sum = 0;
    bool invalid = false;
    // Adjacent threads read adjacent spatial points. Each point retains the
    // CPU channel order and separate double multiply/add rounding.
    for (std::size_t c = 0; c < channels; ++c) {
        const float value = input[c * stride + point];
        sum = __dadd_rn(sum, __dmul_rn(double(value), double(value)));
        invalid |= !isfinite(value);
    }
    const auto scale = image::detail::vae_norm_scale(sum, channels);
    for (std::size_t c = 0; c < channels; ++c) {
        const float value = image::detail::vae_norm_value(
            input[c * stride + point], scale, gamma[c], activate
        );
        output[c * stride + point] = value;
        invalid |= !isfinite(gamma[c]) || !isfinite(value);
    }
    if (invalid)
        atomicOr(status, 1U);
}

__global__ void image_silu_pair_kernel(
    float* gate, const float* up, std::size_t count, unsigned* status
) {
    const auto i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count)
        return;
    // Match the existing bias-free projection's post-GEMM +0, including -0.
    const float g = __fadd_rn(gate[i], 0.f), u = __fadd_rn(up[i], 0.f);
    if (!isfinite(g) || !isfinite(u)) {
        atomicOr(status, 1U);
        return;
    }
    const float fused = image::detail::silu_product(g, u);
    if (!isfinite(fused))
        atomicOr(status, 2U);
    gate[i] = fused;
}

__global__ void
image_frequency_kernel(float* frequency, float timestep, unsigned* status) {
    const auto i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= 512)
        return;
    const float value =
        image::detail::timestep_frequency(i < 256 ? timestep : 0.f, i % 256);
    if (!isfinite(value))
        atomicOr(status, 1U);
    frequency[i] = value;
}

__global__ void
image_timestep_silu_kernel(float* output, std::size_t count, unsigned* status) {
    const auto i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count)
        return;
    const float projected = output[i];
    if (!isfinite(projected)) {
        atomicOr(status, 1U);
        return;
    }
    const float activated = image::detail::silu(projected);
    if (!isfinite(activated))
        atomicOr(status, 2U);
    output[i] = activated;
}

__global__ void image_scheduler_kernel(
    float* latent, const float* velocity, std::size_t count, float dt, unsigned* status
) {
    const auto i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count)
        return;
    const float previous = latent[i], change = velocity[i];
    if (!isfinite(previous) || !isfinite(change)) {
        atomicOr(status, 1U);
        return;
    }
    // Match image_generation's F32 multiply then add; CUDA must not fuse them.
    const float updated = __fadd_rn(previous, __fmul_rn(dt, change));
    if (!isfinite(updated))
        atomicOr(status, 2U);
    latent[i] = updated;
}

__global__ void image_text_norm_kernel(
    const float* input,
    const void* weight,
    bool bf16,
    float* output,
    std::size_t context,
    unsigned* status
) {
    __shared__ float scale;
    const auto base = std::size_t(blockIdx.x) * context;
    if (threadIdx.x == 0) {
        double sum = 0;
        for (std::size_t c = 0; c < context; ++c) {
            const double value = double(input[base + c]);
            sum = __dadd_rn(sum, __dmul_rn(value, value));
        }
        // Match the host's ordered F64 sum and F32 reciprocal RMS cast.
        scale = __double2float_rn(
            __ddiv_rn(1.0, sqrt(__dadd_rn(__ddiv_rn(sum, double(context)), 1e-6)))
        );
    }
    __syncthreads();
    for (std::size_t c = threadIdx.x; c < context; c += blockDim.x) {
        const float w =
            bf16 ? __bfloat162float(static_cast<const __nv_bfloat16*>(weight)[c])
                 : static_cast<const float*>(weight)[c];
        const float value =
            __fmul_rn(__fmul_rn(input[base + c], scale), __fadd_rn(1.f, w));
        if (!isfinite(input[base + c]) || !isfinite(w) || !isfinite(value))
            atomicOr(status, 1U);
        output[base + c] = value;
    }
}

__global__ void image_rotary_kernel(
    float* values,
    const void* weight,
    bool bf16,
    std::size_t heads,
    std::size_t head_dim,
    std::size_t axis0,
    std::size_t axis1,
    std::size_t axis2,
    std::size_t text,
    std::size_t height,
    std::size_t width,
    unsigned* status
) {
    __shared__ float scale;
    const auto token = std::size_t(blockIdx.x);
    const auto base = (token * heads + blockIdx.y) * head_dim;
    if (threadIdx.x == 0) {
        double sum = 0;
        for (std::size_t c = 0; c < head_dim; ++c) {
            const double value = double(values[base + c]);
            sum = __dadd_rn(sum, __dmul_rn(value, value));
        }
        scale = __double2float_rn(
            __ddiv_rn(1.0, sqrt(__dadd_rn(__ddiv_rn(sum, double(head_dim)), 1e-6)))
        );
    }
    __syncthreads();
    const auto c = std::size_t(threadIdx.x) * 2;
    if (c >= head_dim)
        return;
    const std::size_t axes[]{axis0, axis1, axis2};
    long positions[]{long(token), long(token), long(token)};
    if (token >= text) {
        positions[0] = long(text);
        positions[1] = long((token - text) / width) - long(height - height / 2);
        positions[2] = long((token - text) % width) - long(width - width / 2);
    }
    std::size_t axis = 0, offset = 0;
    while (c >= offset + axes[axis])
        offset += axes[axis++];
    const float exponent = __fdiv_rn(-float(c - offset), float(axes[axis]));
    const float angle = __fmul_rn(float(positions[axis]), powf(10000.f, exponent));
    const float co = cosf(angle), si = sinf(angle);
    const float norm_a =
        bf16 ? __bfloat162float(static_cast<const __nv_bfloat16*>(weight)[c])
             : static_cast<const float*>(weight)[c];
    const float norm_b =
        bf16 ? __bfloat162float(static_cast<const __nv_bfloat16*>(weight)[c + 1])
             : static_cast<const float*>(weight)[c + 1];
    const float original_a = values[base + c], original_b = values[base + c + 1];
    const float a = __fmul_rn(original_a, __fmul_rn(scale, norm_a));
    const float b = __fmul_rn(original_b, __fmul_rn(scale, norm_b));
    const float rotated_a = __fsub_rn(__fmul_rn(a, co), __fmul_rn(b, si));
    const float rotated_b = __fadd_rn(__fmul_rn(b, co), __fmul_rn(a, si));
    if (!isfinite(original_a) || !isfinite(original_b) || !isfinite(norm_a)
        || !isfinite(norm_b) || !isfinite(rotated_a) || !isfinite(rotated_b))
        atomicOr(status, 1U);
    values[base + c] = rotated_a;
    values[base + c + 1] = rotated_b;
}

__global__ void
image_text_gelu_kernel(float* output, std::size_t count, unsigned* status) {
    const auto i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count)
        return;
    const float v = output[i];
    // Preserve host F32 expression order; this target disables contraction/FTZ.
    const float value =
        .5f * v * (1 + tanhf(.7978845608028654f * (v + .044715f * v * v * v)));
    if (!isfinite(v) || !isfinite(value))
        atomicOr(status, 1U);
    output[i] = value;
}

__global__ void
image_projection_status(const unsigned* gate, const unsigned* up, unsigned* status) {
    *status = *gate | *up;
}

__global__ void image_residual_kernel(
    float* hidden,
    const float* projected,
    const float* gates,
    std::size_t count,
    std::size_t state,
    std::size_t first,
    std::size_t text,
    unsigned* status
) {
    const auto i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count)
        return;
    const auto gate = (first + i / state < text ? state : 0) + i % state;
    // Match image.cpp's F32 product, then F32 addition with contraction disabled.
    const float result = __fadd_rn(hidden[i], __fmul_rn(gates[gate], projected[i]));
    if (!isfinite(result))
        atomicOr(status, 1U);
    hidden[i] = result;
}

__global__ void image_gates_kernel(
    const float* modulation,
    float* gates,
    std::size_t state,
    std::size_t stage,
    unsigned* status
) {
    const auto i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    // Preserve the host boundary's finite check over the whole modulation tensor.
    if (i < 8 * state && !isfinite(modulation[i]))
        atomicOr(status, 1U);
    if (i < 2 * state) {
        const auto group = i / state, channel = i % state;
        const auto gate = (group * 4 + stage * 2 + 1) * state + channel;
        const float value = tanhf(modulation[gate]);
        if (!isfinite(value))
            atomicOr(status, 1U);
        gates[i] = value;
    }
}

__global__ void image_norm_kernel(
    const float* hidden,
    const float* modulation,
    float* normalized,
    std::size_t state,
    std::size_t text,
    bool final,
    std::size_t stage,
    unsigned* status
) {
    __shared__ double mean;
    __shared__ float scale;
    const auto row = std::size_t(blockIdx.x);
    const auto base = row * state;
    if (threadIdx.x == 0) {
        double sum = 0;
        for (std::size_t c = 0; c < state; ++c) {
            const float value = hidden[base + c];
            if (!isfinite(value))
                atomicOr(status, 1U);
            sum = __dadd_rn(sum, double(value));
        }
        mean = __ddiv_rn(sum, double(state));
        double variance = 0;
        for (std::size_t c = 0; c < state; ++c) {
            const double v = __dsub_rn(double(hidden[base + c]), mean);
            variance = __dadd_rn(variance, __dmul_rn(v, v));
        }
        // Ordered double reductions/casts match norm_mod and final image norm.
        scale = __double2float_rn(
            __ddiv_rn(1.0, sqrt(__dadd_rn(__ddiv_rn(variance, double(state)), 1e-6)))
        );
    }
    __syncthreads();
    const auto offset = final ? 0 : (row < text ? 4 * state : 0) + stage * 2 * state;
    for (std::size_t c = threadIdx.x; c < state; c += blockDim.x) {
        const float centered =
            __double2float_rn(__dsub_rn(double(hidden[base + c]), mean));
        const float value = __fmul_rn(
            __fmul_rn(centered, scale), __fadd_rn(1.f, modulation[offset + c])
        );
        if ((final && !isfinite(modulation[c])) || !isfinite(value))
            atomicOr(status, 1U);
        normalized[base + c] = value;
    }
}

// CUDA calls stay on the persistent thread that owns this selected device.
class CudaDeviceOperations final : public DeviceOperations,
                                   public WeightDeviceOperations {
public:
    void* data = nullptr;
    cublasHandle_t blas = nullptr;
    int compute_major = 0;
    const std::atomic_bool* cancellation = nullptr;
    const std::atomic_bool* failed = nullptr;
    Bytes scratch_bytes = 0;
    const Bytes scratch_limit;
    DeviceWeights weights;

    CudaDeviceOperations(
        Resources& resources, int device, Workload workload, Bytes limit, Bytes scratch
    )
        : scratch_limit(scratch), weights(resources, device, workload, limit, *this) {
    }

    Bytes available_weight_bytes() override {
        std::size_t free = 0, total = 0;
        checked(cudaMemGetInfo(&free, &total));
        const auto remaining =
            scratch_limit > scratch_bytes ? scratch_limit - scratch_bytes : 0;
        return free > remaining ? free - remaining : 0;
    }

    void* allocate_weight(Bytes bytes) override {
        void* pointer = nullptr;
        const auto status = cudaMalloc(&pointer, bytes);
        if (status == cudaErrorMemoryAllocation) {
            cudaGetLastError();
            return nullptr;
        }
        checked(status);
        return pointer;
    }

    void copy_weight(
        void* pointer, std::size_t offset, std::span<const std::byte> source
    ) override {
        checked(cudaMemcpy(
            static_cast<char*>(pointer) + offset,
            source.data(),
            source.size(),
            cudaMemcpyHostToDevice
        ));
    }

    void free_weight(void* pointer) override {
        checked(cudaFree(pointer));
    }

    void* allocate_host(Bytes bytes) override {
        void* pointer = nullptr;
        checked(cudaHostAlloc(&pointer, bytes, cudaHostAllocPortable));
        return pointer;
    }

    void free_host(void* pointer) override {
        checked(cudaFreeHost(pointer));
    }

    void synchronize_weights() override {
        checked(cudaStreamSynchronize(cudaStreamLegacy));
    }

    void release_weights() override {
        weights.release();
    }

    void select(int device) override {
        checked(cudaSetDevice(device));
        checked(cudaDeviceGetAttribute(
            &compute_major, cudaDevAttrComputeCapabilityMajor, device
        ));
    }

    void allocate(Bytes bytes) override {
        // External users can race the queue's probe. Keep ownership while waiting
        // for scratch; cancellation is checked between bounded allocation attempts.
        wait_for_device_allocation(
            [&] {
                return (cancellation && cancellation->load())
                       || (failed && failed->load());
            },
            [&] {
                std::size_t free = 0, total = 0;
                checked(cudaMemGetInfo(&free, &total));
                if (free < bytes)
                    return false;
                const auto status = cudaMalloc(&data, bytes);
                if (status == cudaSuccess) {
                    scratch_bytes = bytes;
                    return true;
                }
                if (status != cudaErrorMemoryAllocation)
                    checked(status);
                cudaGetLastError();
                return false;
            },
            [] { std::this_thread::sleep_for(std::chrono::milliseconds(50)); }
        );
    }

    void free() override {
        checked(cudaFree(data));
        data = nullptr;
        scratch_bytes = 0;
    }

    void create_handle() override {
        checked(cublasCreate(&blas));
    }

    void configure_handle() override {
        checked(cublasSetMathMode(blas, CUBLAS_PEDANTIC_MATH));
        checked(cublasSetStream(blas, cudaStreamLegacy));
    }

    void destroy_handle() override {
        checked(cublasDestroy(blas));
        blas = nullptr;
    }

    void synchronize() override {
        checked(cudaStreamSynchronize(cudaStreamLegacy));
    }

    void reset() override {
        checked(cudaDeviceSynchronize());
#ifndef KADAN_ATEN_RUNTIME
        checked(cudaDeviceReset());
#endif
    }

    template <class T> T* at(std::size_t offset) {
        return reinterpret_cast<T*>(static_cast<char*>(data) + offset);
    }
};

__global__ void
rotate_kernel(const float* x, float* rotated, std::size_t in, unsigned group) {
    __shared__ double values[256];
    const unsigned c = threadIdx.x;
    const auto at = std::size_t(blockIdx.x) * in + std::size_t(blockIdx.y) * group + c;
    values[c] = x[at];
    __syncthreads();
    for (unsigned stride = 1; stride < group; stride *= 4) {
        const unsigned base = c / (stride * 4) * (stride * 4) + c % stride,
                       k = c / stride % 4;
        const double a = values[base], b = values[base + stride],
                     v = values[base + 2 * stride], d = values[base + 3 * stride];
        double y = k == 0   ? ((a + b) + (v - d)) * .5
                   : k == 1 ? ((a + b) + (d - v)) * .5
                   : k == 2 ? ((a + v) + (d - b)) * .5
                            : ((b + v) + (d - a)) * .5;
        __syncthreads();
        values[c] = y;
        __syncthreads();
    }
    rotated[at] = float(values[c]);
}

__global__ void
quantize_kernel(const float* x, std::int8_t* codes, float* scales, std::size_t in) {
    __shared__ float maxima[256];
    float maximum = 1e-10f;
    const auto start = std::size_t(blockIdx.x) * in;
    for (std::size_t c = threadIdx.x; c < in; c += 256)
        maximum = fmaxf(maximum, fabsf(x[start + c]));
    maxima[threadIdx.x] = maximum;
    __syncthreads();
    for (unsigned s = 128; s; s /= 2) {
        if (threadIdx.x < s)
            maxima[threadIdx.x] = fmaxf(maxima[threadIdx.x], maxima[threadIdx.x + s]);
        __syncthreads();
    }
    const float scale = maxima[0] / 127.0f;
    if (threadIdx.x == 0)
        scales[blockIdx.x] = scale;
    for (std::size_t c = threadIdx.x; c < in; c += 256)
        codes[start + c] =
            std::int8_t(max(-127, min(127, __float2int_rn(x[start + c] / scale))));
}

__global__ void finish_int8(
    const std::int32_t* sums,
    const float* row_scale,
    const float* weight_scale,
    const float* bias,
    float* y,
    std::size_t count,
    std::size_t out
) {
    const auto i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count)
        y[i] = (float(sums[i]) * row_scale[i / out]) * weight_scale[i % out]
               + bias[i % out];
}

__global__ void add_bias(
    float* y, const float* bias, std::size_t count, std::size_t out, unsigned* nonfinite
) {
    const auto i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) {
        const float value = y[i] + (bias ? bias[i % out] : 0.f);
        y[i] = value;
        if (nonfinite && !projection_output_finite(__float_as_uint(value)))
            atomicExch(nonfinite, 1U);
    }
}

// Shared with the synchronous host boundary; status clearing belongs to the
// caller so that the host path continues accumulating failures across its tiles.
void launch_bf16_projection(
    cudaStream_t stream,
    cublasHandle_t blas,
    const float* input,
    const void* weight,
    const float* bias,
    __nv_bfloat16* rounded,
    float* output,
    unsigned* status,
    std::size_t rows,
    std::size_t in,
    std::size_t out
) {
    bf16_input<<<unsigned((rows * in + 255) / 256), 256, 0, stream>>>(
        input, rounded, rows * in
    );
    checked(cudaGetLastError());
    const float one = 1, zero = 0;
    with_projection_math(
        true,
        CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION,
        [&] {
            cublasMath_t mode;
            checked(cublasGetMathMode(blas, &mode));
            return mode;
        },
        [&](cublasMath_t mode) { checked(cublasSetMathMode(blas, mode)); },
        [&] {
            checked(cublasGemmEx(
                blas,
                CUBLAS_OP_T,
                CUBLAS_OP_N,
                int(out),
                int(rows),
                int(in),
                &one,
                weight,
                CUDA_R_16BF,
                int(in),
                rounded,
                CUDA_R_16BF,
                int(in),
                &zero,
                output,
                CUDA_R_32F,
                int(out),
                CUBLAS_COMPUTE_32F,
                CUBLAS_GEMM_DEFAULT
            ));
        }
    );
    add_bias<<<unsigned((rows * out + 255) / 256), 256, 0, stream>>>(
        output, bias, rows * out, out, status
    );
    checked(cudaGetLastError());
}

template <class Real>
__device__ void online_softmax_row(
    Real* scores,
    Real* output,
    Real* maximum,
    Real* total_sum,
    unsigned keys,
    unsigned dim,
    Real* reduction
) {
    const unsigned t = threadIdx.x;
    Real m = -INFINITY;
    for (unsigned k = t; k < keys; k += 256) {
        Real v = scores[k];
        m = fmax(m, v);
    }
    reduction[t] = m;
    __syncthreads();
    for (unsigned stride = 128; stride; stride /= 2) {
        if (t < stride)
            reduction[t] = fmax(reduction[t], reduction[t + stride]);
        __syncthreads();
    }
    const Real next = fmax(*maximum, reduction[0]);
    const Real alpha = isfinite(*maximum) ? exp(*maximum - next) : 0;
    Real total = 0;
    // All warps must consume the maximum before shared storage becomes sums.
    __syncthreads();
    for (unsigned k = t; k < keys; k += 256) {
        const Real v = scores[k];
        const Real p = isfinite(v) ? exp(v - next) : 0;
        scores[k] = p;
        total += p;
    }
    reduction[t] = total;
    __syncthreads();
    for (unsigned stride = 128; stride; stride /= 2) {
        if (t < stride)
            reduction[t] += reduction[t + stride];
        __syncthreads();
    }
    for (unsigned c = t; c < dim; c += 256)
        output[c] *= alpha;
    if (t == 0) {
        *maximum = next;
        *total_sum = *total_sum * alpha + reduction[0];
    }
}

template <class Real>
__global__ void online_softmax(
    Real* scores,
    Real* output,
    Real* maxima,
    Real* totals,
    unsigned keys,
    unsigned dim,
    unsigned first_query,
    unsigned first_key,
    unsigned causal_queries,
    unsigned window,
    unsigned offset
) {
    __shared__ Real reduction[256];
    const unsigned q = blockIdx.x, t = threadIdx.x;
    for (unsigned k = t; k < keys; k += 256)
        if (!attention_visible(
                first_query + q, first_key + k, causal_queries, window, offset
            ))
            scores[q * keys + k] = -INFINITY;
    online_softmax_row(
        scores + q * keys,
        output + q * dim,
        maxima + q,
        totals + q,
        keys,
        dim,
        reduction
    );
}

template <class Real>
__global__ void initialize_attention(Real* maximum, Real* total, unsigned rows) {
    const auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < rows) {
        maximum[i] = -INFINITY;
        total[i] = 0;
    }
}

template <class Real>
__global__ void
divide_attention(Real* output, const Real* total, unsigned rows, unsigned dim) {
    const auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < rows * dim)
        output[i] /= total[i / dim];
}

// Pack/unpack only on the owned stream; attention arithmetic is shared above.
__global__ void pack_image_attention(
    const float* input,
    float* packed,
    unsigned rows,
    unsigned state,
    unsigned dim,
    unsigned first,
    unsigned head,
    unsigned* status
) {
    const auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < rows * dim) {
        const float value = input[(first + i / dim) * state + head * dim + i % dim];
        if (!isfinite(value))
            atomicOr(status, 2U);
        packed[i] = value;
    }
}

// Check all resident K/V, including masked keys, before direct strided GEMMs.
__global__ void check_image_key_value(
    const float* key, const float* value, unsigned count, unsigned* status
) {
    const auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count && (!isfinite(key[i]) || !isfinite(value[i])))
        atomicOr(status, 2U);
}

__global__ void scatter_image_attention(
    const float* packed,
    const float* total,
    float* output,
    unsigned rows,
    unsigned state,
    unsigned dim,
    unsigned head,
    unsigned* status
) {
    const auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < rows * dim) {
        const float value = packed[i];
        if (!isfinite(value) || !(total[i / dim] > 0) || !isfinite(total[i / dim]))
            atomicOr(status, 1U);
        output[(i / dim) * state + head * dim + i % dim] = value;
    }
}

struct HostAdmission {
    Resources& resources;
    Handle handle;

    HostAdmission(Resources& r, std::size_t bytes, Workload workload) : resources(r) {
        auto f = r.snapshot().capacity;
        std::fill(f.begin(), f.end(), 0);
        f[0] = bytes;
        handle = r.reserve(workload, std::move(f));
    }

    ~HostAdmission() {
        resources.released(handle);
    }
};

template <class Real>
cublasStatus_t gemm(
    cublasHandle_t h,
    cublasOperation_t a,
    cublasOperation_t b,
    int m,
    int n,
    int k,
    Real alpha,
    const Real* x,
    int ldx,
    const Real* y,
    int ldy,
    Real beta,
    Real* z,
    int ldz
) {
    if constexpr (std::is_same_v<Real, double>)
        return cublasDgemm(h, a, b, m, n, k, &alpha, x, ldx, y, ldy, &beta, z, ldz);
    else
        return cublasSgemm(h, a, b, m, n, k, &alpha, x, ldx, y, ldy, &beta, z, ldz);
}

__global__ void widen(const float* source, double* destination, std::size_t n) {
    const auto i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n)
        destination[i] = source[i];
}

__global__ void narrow_bias(
    const double* source,
    const float* bias,
    float* destination,
    std::size_t n,
    std::size_t out
) {
    const auto i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n)
        destination[i] = float(source[i]) + bias[i % out];
}

class Compute final : public H3Compute {
    std::shared_ptr<Resources> resources_;
    std::vector<int> devices_;
    bool native_bf16_ = false;
    Workload workload_;
    bool column_split_;
    std::atomic_bool quarantined_ = false;
    std::vector<std::unique_ptr<DeviceThread>> threads_;
    mutable std::mutex operation_mutex_;
    Footprint weight_reservations_;
    std::vector<Bytes> weight_capacities_;
    std::vector<Bytes> scratch_limits_;
    Bytes inventory_bytes_ = 0;
    bool replan_ = false;

    template <class F> void all_devices(F function) {
        std::vector<std::future<void>> tasks;
        tasks.reserve(threads_.size());
        std::exception_ptr error;
        for (auto& thread : threads_)
            try {
                tasks.push_back(thread->submit(function));
            } catch (...) {
                if (!error)
                    error = std::current_exception();
            }
        join_device_tasks(tasks, error);
        if (error) {
            quarantined_ = true;
            std::rethrow_exception(error);
        }
    }

    void release_scratch_locked() {
        all_devices([](DeviceWorkspace& workspace) { workspace.release(); });
        weight_reservations_[0] = 0;
        for (std::size_t i = 0; i < threads_.size(); ++i)
            threads_[i]
                ->submit([&, i](DeviceWorkspace& workspace) {
                    const auto bytes =
                        static_cast<CudaDeviceOperations&>(workspace.operations())
                            .weights.reserved();
                    weight_reservations_[devices_[i] + 1] = bytes;
                    if (bytes)
                        weight_reservations_[0] += DeviceWeights::metadata_bytes;
                })
                .get();
    }

    template <class F>
    void split(
        std::size_t rows,
        const std::atomic_bool& cancel,
        F function,
        bool weighted = false
    ) {
        std::lock_guard lock(operation_mutex_);
        require(!quarantined_.load(), "h3_cuda_cleanup_unconfirmed");
        if (cancel.load())
            rethrow_after_cleanup(
                std::make_exception_ptr(std::runtime_error("h3_cuda_cancelled")),
                [&] { release_scratch_locked(); }
            );
        if (replan_)
            plan_weights_locked();
        std::atomic_bool failed = false;
        std::vector<std::future<void>> tasks;
        tasks.reserve(threads_.size());
        std::exception_ptr error;
        try {
            for (std::size_t d = 0; d < threads_.size(); ++d) {
                const auto begin = weighted
                                       ? weight_boundary(rows, weight_capacities_, d)
                                       : rows * d / threads_.size(),
                           end = weighted
                                     ? weight_boundary(rows, weight_capacities_, d + 1)
                                     : rows * (d + 1) / threads_.size();
                if (begin == end)
                    continue;
                tasks.push_back(
                    threads_[d]->submit_work(
                        [&, begin, end](DeviceWorkspace& workspace) {
                            auto& ops = static_cast<CudaDeviceOperations&>(
                                workspace.operations()
                            );
                            struct CancelScope {
                                CudaDeviceOperations& ops;
                                ~CancelScope() {
                                    ops.cancellation = nullptr;
                                    ops.failed = nullptr;
                                }
                            } scope{ops};
                            ops.cancellation = &cancel;
                            ops.failed = &failed;
                            function(workspace, begin, end, failed);
                        },
                        &failed
                    )
                );
            }
        } catch (...) {
            failed = true;
            error = std::current_exception();
        }
        join_device_tasks(tasks, error);
        if (error || cancel.load()) {
            if (!error)
                error =
                    std::make_exception_ptr(std::runtime_error("h3_cuda_cancelled"));
            rethrow_after_cleanup(error, [&] { release_scratch_locked(); });
        }
    }

    static void validate(
        std::span<const float> x,
        std::span<const float> bias,
        std::size_t in,
        std::size_t out,
        std::span<float> y
    ) {
        require(
            in && out && in <= std::size_t(INT_MAX) && out <= std::size_t(INT_MAX)
                && !x.empty() && x.size() % in == 0
                && y.size() == product(x.size() / in, out)
                && (bias.empty() || bias.size() == out),
            "h3_cuda_projection_shape"
        );
        for (float v : x)
            require(std::isfinite(v), "h3_cuda_nonfinite_input");
        for (float v : bias)
            require(std::isfinite(v), "h3_cuda_nonfinite_bias");
    }

    template <class Weight>
    void project(
        std::span<const Weight> weights,
        std::span<const float> scales,
        std::span<const float> bias,
        std::span<const float> x,
        std::size_t in,
        std::size_t out,
        std::size_t group,
        std::span<float> y,
        const std::atomic_bool& cancel,
        bool precise = false,
        const WeightIdentity* identity = nullptr,
        const std::function<void(std::size_t, std::span<Weight>)>* source = nullptr,
        ProjectionPolicy policy = ProjectionPolicy::standard,
        const image::VaeConvolutionTile* convolution = nullptr,
        const GgufWeightSource* packed_source = nullptr,
        gguf::Encoding encoding = gguf::Encoding::q3_k
    ) {
        require(!cancel.load(), "h3_cuda_cancelled");
        if (convolution) {
            image::validate_vae_convolution(convolution->shape, x, bias, y);
            require(
                convolution->count > 0
                    && convolution->first < convolution->shape.points()
                    && convolution->count
                           <= convolution->shape.points() - convolution->first
                    && convolution->count <= convolution->capacity,
                "image_vae_conv_group"
            );
            require(
                in == convolution->shape.columns() && out == convolution->shape.outputs
                    && !precise && std::is_same_v<Weight, float>,
                "image_vae_conv_projection"
            );
            for (float value : bias)
                require(std::isfinite(value), "h3_cuda_nonfinite_bias");
        } else
            validate(x, bias, in, out, y);
        require(
            !overlap(y.data(), y.size_bytes(), x.data(), x.size_bytes())
                && !overlap(
                    y.data(), y.size_bytes(), weights.data(), weights.size_bytes()
                )
                && !overlap(y.data(), y.size_bytes(), bias.data(), bias.size_bytes())
                && !overlap(
                    y.data(), y.size_bytes(), scales.data(), scales.size_bytes()
                ),
            "h3_cuda_alias"
        );
        require(
            ((source || packed_source) && identity)
                || weights.size() == product(in, out),
            "h3_cuda_weight_shape"
        );
        require(
            !packed_source
                || (std::is_same_v<Weight, float> && !source && !convolution
                    && (encoding == gguf::Encoding::q3_k
                        || encoding == gguf::Encoding::q4_k
                        || encoding == gguf::Encoding::q8_0)
                    && in % gguf::block_values(encoding) == 0),
            "h3_cuda_gguf_shape"
        );
        const auto packed_row_bytes =
            packed_source ? gguf_projection_row_bytes(in, encoding) : 0;
        constexpr bool quant = sizeof(Weight) == 1;
        constexpr bool bf16 = std::is_same_v<Weight, std::uint16_t>;
        require(!bf16 || (!precise && native_bf16_), "h3_cuda_bf16_policy");
        if constexpr (quant) {
            require(
                (group == 64 || group == 256) && in % group == 0
                    && in <= std::size_t(INT_MAX / (128 * 127)) && out % 4 == 0
                    && scales.size() == out,
                "h3_cuda_quant_shape"
            );
            for (float v : scales)
                require(std::isfinite(v) && v >= 0, "h3_cuda_scale");
        }

        const bool validate_output = workload_ == Workload::image && !quant && !precise;
        std::atomic_bool nonfinite_output = false;
        const auto total_out = out;
        const auto total_rows = convolution ? convolution->count : x.size() / in;
        const bool columns =
            identity
            || (column_split_ && !quant
                && projection_split_columns(total_rows, devices_.size()));
        const auto all_weights = weights;
        const auto all_bias = bias;
        const auto all_scales = scales;
        std::mutex source_mutex;
        split(
            columns ? (quant ? out / 4 : out) : total_rows,
            cancel,
            [&](DeviceWorkspace& device_workspace,
                std::size_t begin,
                std::size_t end,
                const std::atomic_bool& failed) {
                if constexpr (quant) {
                    if (columns) {
                        begin *= 4;
                        end *= 4;
                    }
                }
                const auto first_output = columns ? begin : 0,
                           last_output = columns ? end : total_out;
                if (columns) {
                    begin = 0;
                    end = total_rows;
                }
                auto& arena =
                    static_cast<CudaDeviceOperations&>(device_workspace.operations());
                // At most 16MiB of source staging per active device, independent of
                // free VRAM. Plan for that actual chunk, leaving room for inputs.
                constexpr std::size_t source_bytes = 16 * 1024 * 1024;
                const auto alignment = quant ? 4u : 1u;
                const auto host_columns =
                    (source_bytes
                     / (packed_source ? packed_row_bytes : in * sizeof(Weight)))
                    / alignment * alignment;
                require(
                    !(source || packed_source) || host_columns > 0,
                    "h3_source_staging_floor"
                );
                const auto convolution_plan = convolution
                                                  ? image_vae_convolution_plan(
                                                        convolution->shape,
                                                        last_output - first_output,
                                                        arena.scratch_limit,
                                                        convolution->capacity
                                                    )
                                                  : std::optional<VaeConvolutionPlan>{};
                require(
                    !convolution
                        || (convolution_plan.has_value()
                            && convolution_plan->points == convolution->capacity),
                    "image_vae_conv_scratch"
                );
                const auto patch_bytes =
                    convolution ? convolution_plan->patch_bytes : 0;
                const auto plan =
                    convolution ? convolution_plan->projection
                    : packed_source
                        ? gguf_projection_plan(
                              in,
                              last_output - first_output,
                              encoding,
                              precise,
                              arena.scratch_limit
                          )
                        : projection_plan(
                              in,
                              source
                                  ? std::min(last_output - first_output, host_columns)
                                  : last_output - first_output,
                              sizeof(Weight),
                              precise,
                              alignment,
                              arena.scratch_limit
                          );
                const auto chunk_columns = plan.columns;
                const bool grouped = convolution && convolution->capacity > 256;
                const auto input_rows =
                    grouped
                        ? convolution->capacity
                        : projection_input_rows(
                              end - begin, in, plan, arena.scratch_limit - patch_bytes
                          );
                const bool keep_input = input_rows >= end - begin;
                // Reserve the largest layout before uploading. Later chunks must
                // neither move the input at its fixed offset nor grow the workspace.
                device_workspace.ensure(
                    grouped ? convolution_plan->bytes
                            : plan.bytes + patch_bytes
                                  + (input_rows - plan.rows) * in * sizeof(float)
                );
                // The aligned status prefix stays fixed across every column chunk.
                auto* nonfinite = (validate_output || packed_source)
                                      ? arena.at<unsigned>(0)
                                      : nullptr;
                if (nonfinite)
                    checked(cudaMemset(nonfinite, 0, sizeof(unsigned)));
                Layout input_layout;
                input_layout.add(sizeof(unsigned));
                const auto input = input_layout.add(input_rows * in * sizeof(float));
                const auto raw_input = input_layout.add(patch_bytes);
                auto upload_patch = [&](std::size_t first, std::size_t rows) {
                    const auto& shape = convolution->shape;
                    const auto region = image::vae_patch_region(shape, first, rows);
                    require(
                        shape.channels * region.points() * 4 <= patch_bytes,
                        "image_vae_conv_patch_scratch"
                    );
                    cudaMemcpy3DParms copy{};
                    copy.srcPtr = make_cudaPitchedPtr(
                        const_cast<float*>(x.data()),
                        shape.width * 4,
                        shape.width * 4,
                        shape.height
                    );
                    copy.srcPos = make_cudaPos(region.left * 4, region.top, 0);
                    copy.dstPtr = make_cudaPitchedPtr(
                        arena.at<void>(raw_input),
                        region.width * 4,
                        region.width * 4,
                        region.height
                    );
                    copy.extent = make_cudaExtent(
                        region.width * 4, region.height, shape.channels
                    );
                    copy.kind = cudaMemcpyHostToDevice;
                    checked(cudaMemcpy3D(&copy));
                    return region;
                };
                std::optional<image::VaePatchRegion> group_region;
                if (grouped) {
                    stop(cancel, failed);
                    group_region =
                        upload_patch(convolution->first + begin, end - begin);
                }
                auto upload_input = [&](float* target,
                                        std::size_t start,
                                        std::size_t rows) {
                    if (!convolution) {
                        checked(cudaMemcpy(
                            target,
                            x.data() + start * in,
                            rows * in * sizeof(float),
                            cudaMemcpyHostToDevice
                        ));
                        return;
                    }
                    const auto& shape = convolution->shape;
                    const auto first = convolution->first + start;
                    const auto region =
                        group_region ? *group_region : upload_patch(first, rows);
                    image_vae_gather_kernel<<<unsigned((rows * in + 255) / 256), 256>>>(
                        arena.at<float>(raw_input),
                        target,
                        nonfinite,
                        shape,
                        region,
                        first,
                        rows
                    );
                    checked(cudaGetLastError());
                };
                if (keep_input)
                    for (auto start = begin; start < end; start += plan.rows) {
                        stop(cancel, failed);
                        const auto rows = std::min(plan.rows, end - start);
                        upload_input(
                            arena.at<float>(input) + (start - begin) * in, start, rows
                        );
                    }
                for (auto first_column = first_output; first_column < last_output;
                     first_column += chunk_columns) {
                    const auto out =
                        std::min(chunk_columns, last_output - first_column);
                    auto weights =
                        (source || packed_source)
                            ? std::span<const Weight>{}
                            : all_weights.subspan(first_column * in, out * in);
                    const auto bias = all_bias.empty()
                                          ? all_bias
                                          : all_bias.subspan(first_column, out);
                    const auto scales = all_scales.empty()
                                            ? all_scales
                                            : all_scales.subspan(first_column, out);
                    stop(cancel, failed);
                    const auto tile = std::min(plan.rows, end - begin);
                    // Establish device/thread ownership before any retained allocation.
                    device_workspace.ensure(256);
                    const auto weight_count = product(in, out);
                    const DeviceWeights::Key key{
                        identity ? *identity : WeightIdentity{},
                        packed_source
                            ? first_column * (in / gguf::block_values(encoding))
                            : first_column * in,
                        packed_source ? weight_count / gguf::block_values(encoding)
                                      : weight_count,
                        packed_source ? gguf::block_bytes(encoding) : sizeof(Weight)
                    };
                    void* retained = identity ? arena.weights.find(key) : nullptr;
                    std::unique_ptr<HostAdmission> host_admission;
                    std::unique_ptr<Weight[]> host_weights;
                    std::span<std::byte> host_packed;
                    const auto packed_bytes = product(out, packed_row_bytes);
                    if (!retained && packed_source) {
                        static_assert(
                            DeviceWorkspace::host_staging_capacity
                            == gguf_projection_staging_bytes
                        );
                        host_packed = device_workspace.host_staging(packed_bytes);
                        stop(cancel, failed);
                        (*packed_source)(first_column, host_packed);
                        stop(cancel, failed);
                    }
                    if (!retained && !packed_source) {
                        if (source) {
                            host_admission = std::make_unique<HostAdmission>(
                                *resources_,
                                weight_count * sizeof(Weight) + 4096,
                                workload_
                            );
                            host_weights =
                                std::make_unique_for_overwrite<Weight[]>(weight_count);
                            {
                                std::lock_guard source_lock(source_mutex);
                                stop(cancel, failed);
                                (*source)(
                                    first_column* in, {host_weights.get(), weight_count}
                                );
                            }
                            weights = {host_weights.get(), weight_count};
                        }
                        if constexpr (bf16) {
                            for (auto value : weights)
                                require(
                                    (value & 0x7f80) != 0x7f80,
                                    "h3_cuda_nonfinite_weight"
                                );
                        } else if constexpr (!quant)
                            require(
                                quantization::all_finite(weights),
                                "h3_cuda_nonfinite_weight"
                            );
                        if (identity)
                            retained = arena.weights.retain(
                                key, std::as_bytes(weights), cancel
                            );
                    }
                    Layout l = input_layout;
                    const auto packed_status =
                                   l.add(packed_source ? sizeof(unsigned) : 0),
                               packed = l.add(
                                   packed_source ? gguf_projection_staging_bytes : 0
                               ),
                               w = l.add(
                                   packed_source ? weight_count * sizeof(float)
                                                 : (retained ? 0 : weights.size_bytes())
                               ),
                               b = l.add(out * 4), s = l.add(out * 4),
                               rotated = l.add(tile * in * 4), codes = l.add(tile * in),
                               row_scales = l.add(tile * 4),
                               sums = l.add((grouped ? input_rows : tile) * out * 4),
                               output = l.add(tile * out * 4),
                               workspace = l.add(8 * 1024 * 1024),
                               wide_weights = l.add(precise ? weight_count * 8 : 0),
                               wide_input = l.add(precise ? tile * in * 8 : 0),
                               wide_output = l.add(precise ? tile * out * 8 : 0);
                    require(
                        l.bytes <= device_workspace.retained_bytes(),
                        "h3_cuda_projection_scratch"
                    );
                    checked(cublasSetWorkspace(
                        arena.blas, arena.at<void>(workspace), 8 * 1024 * 1024
                    ));
                    auto* weight_pointer =
                        packed_source ? arena.at<void>(w)
                                      : (retained ? retained : arena.at<void>(w));
                    if (packed_source) {
                        auto* packed_pointer =
                            retained ? retained : arena.at<void>(packed);
                        if (!retained)
                            checked(cudaMemcpy(
                                packed_pointer,
                                host_packed.data(),
                                packed_bytes,
                                cudaMemcpyHostToDevice
                            ));
                        auto* decode_status = arena.at<unsigned>(packed_status);
                        checked(cudaMemset(decode_status, 0, sizeof(unsigned)));
                        gguf_decode_kernel<<<
                            unsigned((weight_count + 255) / 256),
                            256>>>(
                            static_cast<const unsigned char*>(packed_pointer),
                            static_cast<float*>(weight_pointer),
                            decode_status,
                            encoding,
                            weight_count
                        );
                        checked(cudaGetLastError());
                        checked(cudaStreamSynchronize(cudaStreamLegacy));
                        unsigned invalid = 0;
                        checked(cudaMemcpy(
                            &invalid,
                            decode_status,
                            sizeof(invalid),
                            cudaMemcpyDeviceToHost
                        ));
                        require(!invalid, "h3_cuda_nonfinite_weight");
                        stop(cancel, failed);
                        // Admission follows successful decoding, so invalid source
                        // bytes never become a reusable bank entry. Existing bank
                        // ownership and cleanup cover the retained packed allocation.
                        if (!retained)
                            arena.weights.retain(key, host_packed, cancel);
                    } else if (!retained)
                        checked(cudaMemcpy(
                            weight_pointer,
                            weights.data(),
                            weights.size_bytes(),
                            cudaMemcpyHostToDevice
                        ));
                    if (bias.empty())
                        checked(cudaMemset(arena.at<void>(b), 0, out * 4));
                    else
                        checked(cudaMemcpy(
                            arena.at<void>(b),
                            bias.data(),
                            out * 4,
                            cudaMemcpyHostToDevice
                        ));
                    if constexpr (quant)
                        checked(cudaMemcpy(
                            arena.at<void>(s),
                            scales.data(),
                            out * 4,
                            cudaMemcpyHostToDevice
                        ));
                    else if (precise) {
                        widen<<<unsigned((weight_count + 255) / 256), 256>>>(
                            static_cast<float*>(weight_pointer),
                            arena.at<double>(wide_weights),
                            weight_count
                        );
                        checked(cudaGetLastError());
                    }
                    for (std::size_t start = begin; start < end; start += tile) {
                        stop(cancel, failed);
                        const auto rows = std::min(tile, end - start);
                        auto* input_pointer = arena.at<float>(input)
                                              + (keep_input ? (start - begin) * in : 0);
                        if (!keep_input)
                            upload_input(input_pointer, start, rows);
                        if constexpr (quant) {
                            rotate_kernel<<<
                                dim3(unsigned(rows), unsigned(in / group)),
                                unsigned(group)>>>(
                                input_pointer,
                                arena.at<float>(rotated),
                                in,
                                unsigned(group)
                            );
                            checked(cudaGetLastError());
                            quantize_kernel<<<unsigned(rows), 256>>>(
                                arena.at<float>(rotated),
                                arena.at<std::int8_t>(codes),
                                arena.at<float>(row_scales),
                                in
                            );
                            checked(cudaGetLastError());
                            const std::int32_t one = 1, zero = 0;
                            checked(cublasGemmEx(
                                arena.blas,
                                CUBLAS_OP_T,
                                CUBLAS_OP_N,
                                int(out),
                                int(rows),
                                int(in),
                                &one,
                                weight_pointer,
                                CUDA_R_8I,
                                int(in),
                                arena.at<void>(codes),
                                CUDA_R_8I,
                                int(in),
                                &zero,
                                arena.at<void>(sums),
                                CUDA_R_32I,
                                int(out),
                                CUBLAS_COMPUTE_32I,
                                CUBLAS_GEMM_DEFAULT
                            ));
                            finish_int8<<<unsigned((rows * out + 255) / 256), 256>>>(
                                arena.at<std::int32_t>(sums),
                                arena.at<float>(row_scales),
                                arena.at<float>(s),
                                arena.at<float>(b),
                                arena.at<float>(output),
                                rows * out,
                                out
                            );
                        } else if constexpr (bf16) {
                            // Mutually exclusive with INT8 rotation; reuse its larger
                            // scratch slice. The host boundary owns synchronization.
                            launch_bf16_projection(
                                cudaStreamLegacy,
                                arena.blas,
                                input_pointer,
                                weight_pointer,
                                arena.at<float>(b),
                                arena.at<__nv_bfloat16>(rotated),
                                arena.at<float>(output),
                                nonfinite,
                                rows,
                                in,
                                out
                            );
                        } else if (precise) {
                            widen<<<unsigned((rows * in + 255) / 256), 256>>>(
                                input_pointer, arena.at<double>(wide_input), rows * in
                            );
                            checked(cudaGetLastError());
                            checked(
                                gemm<double>(
                                    arena.blas,
                                    CUBLAS_OP_T,
                                    CUBLAS_OP_N,
                                    int(out),
                                    int(rows),
                                    int(in),
                                    1.0,
                                    arena.at<double>(wide_weights),
                                    int(in),
                                    arena.at<double>(wide_input),
                                    int(in),
                                    0.0,
                                    arena.at<double>(wide_output),
                                    int(out)
                                )
                            );
                            narrow_bias<<<unsigned((rows * out + 255) / 256), 256>>>(
                                arena.at<double>(wide_output),
                                arena.at<float>(b),
                                arena.at<float>(output),
                                rows * out,
                                out
                            );
                        } else {
                            const float one = 1, zero = 0;
                            with_projection_math(
                                projection_uses_tf32(
                                    policy, workload_, arena.compute_major, precise
                                ),
                                CUBLAS_TF32_TENSOR_OP_MATH,
                                [&] {
                                    cublasMath_t mode;
                                    checked(cublasGetMathMode(arena.blas, &mode));
                                    return mode;
                                },
                                [&](cublasMath_t mode) {
                                    checked(cublasSetMathMode(arena.blas, mode));
                                },
                                [&] {
                                    checked(cublasSgemm(
                                        arena.blas,
                                        CUBLAS_OP_T,
                                        CUBLAS_OP_N,
                                        int(out),
                                        int(rows),
                                        int(in),
                                        &one,
                                        static_cast<float*>(weight_pointer),
                                        int(in),
                                        input_pointer,
                                        int(in),
                                        &zero,
                                        arena.at<float>(output),
                                        int(out)
                                    ));
                                }
                            );
                            add_bias<<<unsigned((rows * out + 255) / 256), 256>>>(
                                arena.at<float>(output),
                                arena.at<float>(b),
                                rows * out,
                                out,
                                nonfinite
                            );
                        }
                        checked(cudaGetLastError());
                        if (convolution) {
                            // The F32 path does not use INT8 sums: reuse that
                            // admitted slice for the bit-preserving planar tile.
                            image_vae_scatter_kernel<<<
                                unsigned((rows * out + 255) / 256),
                                256>>>(
                                arena.at<float>(output),
                                arena.at<float>(sums),
                                rows,
                                out,
                                grouped ? start - begin : 0,
                                grouped ? input_rows : rows
                            );
                            checked(cudaGetLastError());
                        }
                        checked(cudaStreamSynchronize(cudaStreamLegacy));
                        stop(cancel, failed);
                        if (convolution && !grouped)
                            checked(cudaMemcpy2D(
                                y.data() + first_column * convolution->shape.points()
                                    + convolution->first + start,
                                convolution->shape.points() * 4,
                                arena.at<void>(sums),
                                rows * 4,
                                rows * 4,
                                out,
                                cudaMemcpyDeviceToHost
                            ));
                        else if (!convolution)
                            checked(cudaMemcpy2D(
                                y.data() + start * total_out + first_column,
                                total_out * 4,
                                arena.at<void>(output),
                                out * 4,
                                out * 4,
                                rows,
                                cudaMemcpyDeviceToHost
                            ));
                    }
                    if (grouped) {
                        stop(cancel, failed);
                        checked(cudaMemcpy2D(
                            y.data() + first_column * convolution->shape.points()
                                + convolution->first + begin,
                            convolution->shape.points() * 4,
                            arena.at<void>(sums),
                            input_rows * 4,
                            (end - begin) * 4,
                            out,
                            cudaMemcpyDeviceToHost
                        ));
                    }
                }
                if (nonfinite) {
                    unsigned invalid = 0;
                    checked(cudaMemcpy(
                        &invalid, nonfinite, sizeof(invalid), cudaMemcpyDeviceToHost
                    ));
                    if (invalid)
                        nonfinite_output = true;
                }
            },
            identity != nullptr
        );
        require(!nonfinite_output.load(), "h3_cuda_nonfinite_output");
        if (!validate_output)
            for (float v : y)
                require(std::isfinite(v), "h3_cuda_nonfinite_output");
    }

public:
    Compute(
        std::shared_ptr<Resources> r,
        std::vector<int> devices,
        Workload workload = Workload::video,
        bool columns = false
    )
        : resources_(std::move(r)), devices_(std::move(devices)), workload_(workload),
          column_split_(columns) {
        require(bool(resources_) && !devices_.empty(), "h3_cuda_devices");
        const auto capacity = resources_->snapshot().capacity;
        for (std::size_t i = 0; i < devices_.size(); ++i) {
            const int d = devices_[i];
            require(
                d >= 0 && std::size_t(d) + 1 < capacity.size() && capacity[d + 1] > 0,
                "h3_cuda_unbudgeted_device"
            );
            require(
                std::find(devices_.begin(), devices_.begin() + i, d)
                    == devices_.begin() + i,
                "h3_cuda_duplicate_device"
            );
        }
        weight_reservations_.resize(capacity.size());
        weight_capacities_.resize(devices_.size());
        const auto used = resources_->snapshot().used;
        native_bf16_ = workload_ == Workload::image;
        for (int device : devices_) {
            int major = 0;
            checked(cudaDeviceGetAttribute(
                &major, cudaDevAttrComputeCapabilityMajor, device
            ));
            native_bf16_ = native_bf16_ && major >= 8;
            const auto available = capacity[device + 1] - used[device + 1];
            const Bytes scratch = std::min<Bytes>(2ULL * 1024 * 1024 * 1024, available);
            scratch_limits_.push_back(scratch);
            const Bytes limit = available > scratch ? available - scratch : 0;
            threads_.push_back(
                std::make_unique<DeviceThread>(
                    *resources_,
                    device,
                    workload_,
                    std::make_unique<CudaDeviceOperations>(
                        *resources_, device, workload_, limit, scratch
                    )
                )
            );
        }
    }

    template <class Real>
    void attention_impl(
        std::span<const float> q,
        std::span<const float> k,
        std::span<const float> v,
        std::size_t n,
        std::size_t heads,
        std::size_t kv_heads,
        std::size_t dim,
        bool causal,
        std::span<float> y,
        const std::atomic_bool& cancel,
        std::size_t key_tokens = 0,
        std::size_t causal_queries = 0,
        std::size_t window = 0,
        std::size_t offset = 0
    ) {
        require(!cancel.load(), "h3_cuda_cancelled");
        if (!key_tokens)
            key_tokens = n;
        if (causal)
            causal_queries = n;
        require(
            n > 0 && n <= std::size_t(INT_MAX) && heads > 0 && kv_heads > 0
                && heads % kv_heads == 0 && dim > 0 && dim <= 2048,
            "h3_cuda_attention_shape"
        );
        require(
            key_tokens > 0 && key_tokens <= std::size_t(INT_MAX) && causal_queries <= n
                && offset <= key_tokens && window <= key_tokens
                && (!causal_queries || causal_queries <= key_tokens - offset),
            "h3_cuda_attention_shape"
        );
        const auto head_values = product(n, dim), key_values = product(key_tokens, dim);
        require(
            q.size() == product(head_values, heads)
                && k.size() == product(key_values, kv_heads) && v.size() == k.size()
                && y.size() == q.size(),
            "h3_cuda_attention_shape"
        );
        for (auto input : {q, k, v}) {
            require(
                !overlap(y.data(), y.size_bytes(), input.data(), input.size_bytes()),
                "h3_cuda_alias"
            );
            for (float x : input)
                require(std::isfinite(x), "h3_cuda_nonfinite_input");
        }
        split(
            heads,
            cancel,
            [&](DeviceWorkspace& device_workspace,
                std::size_t first,
                std::size_t last,
                const std::atomic_bool& failed) {
                stop(cancel, failed);
                auto& arena =
                    static_cast<CudaDeviceOperations&>(device_workspace.operations());
                const auto plan = attention_plan(
                    n, key_tokens, dim, sizeof(Real), arena.scratch_limit
                );
                const auto query_tile = plan.queries, key_tile = plan.keys;
                const auto query_storage = (plan.full_heads ? n : query_tile) * dim,
                           key_storage =
                               (plan.full_heads ? key_tokens : key_tile) * dim;
                Layout l;
                const auto dq = l.add(query_storage * sizeof(Real)),
                           dk = l.add(key_storage * sizeof(Real)),
                           dv = l.add(key_storage * sizeof(Real)),
                           dy = l.add(query_tile * dim * sizeof(Real)),
                           scores = l.add(query_tile * key_tile * sizeof(Real)),
                           maximum = l.add(query_tile * sizeof(Real)),
                           total = l.add(query_tile * sizeof(Real)),
                           workspace = l.add(8 * 1024 * 1024);
                HostAdmission admission(
                    *resources_,
                    (query_storage + key_storage * 2 + query_tile * dim) * sizeof(Real),
                    workload_
                );
                auto memory = std::make_unique_for_overwrite<Real[]>(
                    query_storage + key_storage * 2 + query_tile * dim
                );
                auto hq = memory.get(), hk = hq + query_storage, hv = hk + key_storage,
                     hy = hv + key_storage;
                device_workspace.ensure(l.bytes);
                checked(cublasSetWorkspace(
                    arena.blas, arena.at<void>(workspace), 8 * 1024 * 1024
                ));
                for (std::size_t h = first; h < last; ++h) {
                    stop(cancel, failed);
                    const auto kh = h / (heads / kv_heads);
                    auto pack_query = [&](std::size_t start, std::size_t rows) {
                        for (std::size_t t = 0; t < rows; ++t) {
                            if (t % 64 == 0)
                                stop(cancel, failed);
                            for (std::size_t c = 0; c < dim; ++c)
                                hq[t * dim + c] =
                                    q[((start + t) * heads + h) * dim + c];
                        }
                        checked(cudaMemcpy(
                            arena.at<void>(dq),
                            hq,
                            rows * dim * sizeof(Real),
                            cudaMemcpyHostToDevice
                        ));
                    };
                    auto pack_keys = [&](std::size_t start, std::size_t rows) {
                        for (std::size_t t = 0; t < rows; ++t) {
                            if (t % 64 == 0)
                                stop(cancel, failed);
                            for (std::size_t c = 0; c < dim; ++c) {
                                hk[t * dim + c] =
                                    k[((start + t) * kv_heads + kh) * dim + c];
                                hv[t * dim + c] =
                                    v[((start + t) * kv_heads + kh) * dim + c];
                            }
                        }
                        checked(cudaMemcpy(
                            arena.at<void>(dk),
                            hk,
                            rows * dim * sizeof(Real),
                            cudaMemcpyHostToDevice
                        ));
                        checked(cudaMemcpy(
                            arena.at<void>(dv),
                            hv,
                            rows * dim * sizeof(Real),
                            cudaMemcpyHostToDevice
                        ));
                    };
                    if (plan.full_heads) {
                        pack_query(0, n);
                        pack_keys(0, key_tokens);
                    }
                    for (std::size_t start = 0; start < n; start += query_tile) {
                        stop(cancel, failed);
                        const auto rows = std::min(query_tile, n - start);
                        if (!plan.full_heads)
                            pack_query(start, rows);
                        checked(
                            cudaMemset(arena.at<void>(dy), 0, rows * dim * sizeof(Real))
                        );
                        initialize_attention<<<1, 128>>>(
                            arena.at<Real>(maximum),
                            arena.at<Real>(total),
                            unsigned(rows)
                        );
                        checked(cudaGetLastError());
                        for (std::size_t key = 0; key < key_tokens; key += key_tile) {
                            stop(cancel, failed);
                            const auto keys = std::min(key_tile, key_tokens - key);
                            if (!plan.full_heads)
                                pack_keys(key, keys);
                            const Real scale = 1 / std::sqrt(Real(dim)), zero = 0,
                                       one = 1;
                            checked(
                                gemm<Real>(
                                    arena.blas,
                                    CUBLAS_OP_T,
                                    CUBLAS_OP_N,
                                    int(keys),
                                    int(rows),
                                    int(dim),
                                    scale,
                                    arena.at<Real>(dk)
                                        + (plan.full_heads ? key * dim : 0),
                                    int(dim),
                                    arena.at<Real>(dq)
                                        + (plan.full_heads ? start * dim : 0),
                                    int(dim),
                                    zero,
                                    arena.at<Real>(scores),
                                    int(keys)
                                )
                            );
                            online_softmax<<<unsigned(rows), 256>>>(
                                arena.at<Real>(scores),
                                arena.at<Real>(dy),
                                arena.at<Real>(maximum),
                                arena.at<Real>(total),
                                unsigned(keys),
                                unsigned(dim),
                                unsigned(start),
                                unsigned(key),
                                unsigned(causal_queries),
                                unsigned(window),
                                unsigned(offset)
                            );
                            checked(cudaGetLastError());
                            checked(
                                gemm<Real>(
                                    arena.blas,
                                    CUBLAS_OP_N,
                                    CUBLAS_OP_N,
                                    int(dim),
                                    int(rows),
                                    int(keys),
                                    one,
                                    arena.at<Real>(dv)
                                        + (plan.full_heads ? key * dim : 0),
                                    int(dim),
                                    arena.at<Real>(scores),
                                    int(keys),
                                    one,
                                    arena.at<Real>(dy),
                                    int(dim)
                                )
                            );
                        }
                        divide_attention<<<unsigned((rows * dim + 255) / 256), 256>>>(
                            arena.at<Real>(dy),
                            arena.at<Real>(total),
                            unsigned(rows),
                            unsigned(dim)
                        );
                        checked(cudaGetLastError());
                        // The legacy stream orders all key tiles; this blocking copy
                        // waits for the complete query tile before host consumption.
                        checked(cudaMemcpy(
                            hy,
                            arena.at<void>(dy),
                            rows * dim * sizeof(Real),
                            cudaMemcpyDeviceToHost
                        ));
                        stop(cancel, failed);
                        for (std::size_t t = 0; t < rows; ++t)
                            for (std::size_t c = 0; c < dim; ++c)
                                y[((start + t) * heads + h) * dim + c] =
                                    hy[t * dim + c];
                    }
                }
            }
        );
        for (float x : y)
            require(std::isfinite(x), "h3_cuda_nonfinite_output");
    }

    void attention(
        std::span<const float> q,
        std::span<const float> k,
        std::span<const float> v,
        std::size_t n,
        std::size_t heads,
        std::size_t kv,
        std::size_t dim,
        bool causal,
        std::span<float> y,
        const std::atomic_bool& cancel,
        bool precise
    ) override {
        if (precise)
            attention_impl<double>(q, k, v, n, heads, kv, dim, causal, y, cancel);
        else
            attention_impl<float>(q, k, v, n, heads, kv, dim, causal, y, cancel);
    }

    bool attend(
        std::span<const float> q,
        std::span<const float> k,
        std::span<const float> v,
        std::size_t queries,
        std::size_t keys,
        std::size_t heads,
        std::size_t kv,
        std::size_t dim,
        std::size_t causal_queries,
        std::size_t window,
        std::size_t offset,
        std::span<float> y,
        const std::atomic_bool& cancel
    ) override {
        attention_impl<float>(
            q,
            k,
            v,
            queries,
            heads,
            kv,
            dim,
            false,
            y,
            cancel,
            keys,
            causal_queries,
            window,
            offset
        );
        return true;
    }

    void release_scratch() override {
        std::lock_guard lock(operation_mutex_);
        release_scratch_locked();
    }

    void release_devices() override {
        std::lock_guard lock(operation_mutex_);
        require(!quarantined_.load(), "h3_cuda_cleanup_unconfirmed");
        all_devices([](DeviceWorkspace& workspace) { workspace.reset(); });
        std::fill(weight_reservations_.begin(), weight_reservations_.end(), 0);
        replan_ = inventory_bytes_ != 0;
    }

    Bytes release_weights(std::span<const WeightIdentity> sources) override {
        std::lock_guard lock(operation_mutex_);
        require(!quarantined_.load(), "h3_cuda_cleanup_unconfirmed");
        std::atomic<Bytes> released = 0;
        all_devices([&](DeviceWorkspace& workspace) {
            released.fetch_add(
                static_cast<CudaDeviceOperations&>(workspace.operations())
                    .weights.release_sources(sources)
            );
        });
        return released.load();
    }

    Footprint retained_weights() const override {
        std::lock_guard lock(operation_mutex_);
        return weight_reservations_;
    }

    std::vector<DeviceWeightPlacement> weight_placement() const override {
        std::lock_guard lock(operation_mutex_);
        std::vector<DeviceWeightPlacement> result;
        result.reserve(threads_.size());
        for (std::size_t i = 0; i < threads_.size(); ++i)
            threads_[i]
                ->submit([&, i](DeviceWorkspace& workspace) {
                    const auto& bank =
                        static_cast<CudaDeviceOperations&>(workspace.operations())
                            .weights;
                    result.push_back({devices_[i], bank.capacity(), bank.allocated()});
                })
                .get();
        return result;
    }

    bool prepare_weights(Bytes bytes) override {
        std::lock_guard lock(operation_mutex_);
        require(!quarantined_.load(), "h3_cuda_cleanup_unconfirmed");
        inventory_bytes_ = bytes;
        return plan_weights_locked();
    }

private:
    bool plan_weights_locked() {
        const auto bytes = inventory_bytes_;
        // Plan from free physical memory on each owned device thread. Freeze the
        // partition until park/replan so cached identities keep stable slices.
        Bytes total = 0;
        for (std::size_t i = 0; i < threads_.size(); ++i)
            threads_[i]
                ->submit([&, i](DeviceWorkspace& workspace) {
                    workspace.activate();
                    auto& ops =
                        static_cast<CudaDeviceOperations&>(workspace.operations());
                    weight_capacities_[i] = std::min(
                        ops.weights.maximum_capacity(), ops.available_weight_bytes()
                    );
                    total += weight_capacities_[i];
                })
                .get();
        for (std::size_t i = 0; i < threads_.size(); ++i)
            threads_[i]
                ->submit([&, i](DeviceWorkspace& workspace) {
                    auto& bank =
                        static_cast<CudaDeviceOperations&>(workspace.operations())
                            .weights;
                    // A ceiling per bank leaves room for tensor channel rounding.
                    const auto share = weight_boundary(bytes, weight_capacities_, i + 1)
                                       - weight_boundary(bytes, weight_capacities_, i);
                    bank.bound(
                        std::min(
                            weight_capacities_[i],
                            share
                                + std::min<Bytes>(
                                    16 * 1024 * 1024,
                                    std::numeric_limits<Bytes>::max() - share
                                )
                        )
                    );
                })
                .get();
        replan_ = false;
        return total >= bytes;
    }

public:
    bool image_vae_convolution_source(
        const WeightIdentity& identity,
        const FloatWeightSource& source,
        std::span<const float> bias,
        std::span<const float> input,
        image::VaeConvolution shape,
        std::span<float> output,
        const std::atomic_bool& cancel
    ) override {
        require(!cancel.load(), "h3_cuda_cancelled");
        image::validate_vae_convolution(shape, input, bias, output);
        if (workload_ != Workload::image)
            return false;
        // Every selected device must support the operation before any callback
        // or submission. Project owns the existing weight/staging/cleanup path.
        std::size_t points = 4096;
        for (auto limit : scratch_limits_) {
            const auto plan = image_vae_convolution_plan(shape, shape.outputs, limit);
            if (!plan)
                return false;
            points = std::min(points, plan->points);
        }
        for (std::size_t first = 0; first < shape.points(); first += points) {
            require(!cancel.load(), "h3_cuda_cancelled");
            const image::VaeConvolutionTile tile{
                shape, first, std::min(points, shape.points() - first), points
            };
            project<float>(
                {},
                {},
                bias,
                input,
                shape.columns(),
                shape.outputs,
                0,
                output,
                cancel,
                false,
                &identity,
                &source,
                ProjectionPolicy::standard,
                &tile
            );
        }
        require(!cancel.load(), "h3_cuda_cancelled");
        return true;
    }

    bool image_vae_normalize(
        std::span<const float> input,
        std::span<const float> gamma,
        std::size_t points,
        std::span<float> output,
        bool activate,
        const std::atomic_bool& cancel
    ) override {
        require(!cancel.load(), "h3_cuda_cancelled");
        image::validate_vae_normalization(input, gamma, points, output);
        // Unsupported must precede submission or output writes on every device.
        for (auto limit : scratch_limits_)
            if (!image_vae_normalization_plan(gamma.size(), points, limit))
                return false;
        std::atomic_bool nonfinite = false;
        split(
            points,
            cancel,
            [&](DeviceWorkspace& workspace,
                std::size_t begin,
                std::size_t end,
                const std::atomic_bool& failed) {
                stop(cancel, failed);
                auto& arena =
                    static_cast<CudaDeviceOperations&>(workspace.operations());
                const auto plan = *image_vae_normalization_plan(
                    gamma.size(), end - begin, arena.scratch_limit
                );
                Layout layout;
                const auto g = layout.add(gamma.size_bytes()),
                           x = layout.add(gamma.size() * plan.points * 4),
                           y = layout.add(gamma.size() * plan.points * 4),
                           status = layout.add(sizeof(unsigned));
                require(layout.bytes <= plan.bytes, "image_vae_norm_scratch");
                workspace.ensure(layout.bytes);
                checked(cudaMemcpy(
                    arena.at<void>(g),
                    gamma.data(),
                    gamma.size_bytes(),
                    cudaMemcpyHostToDevice
                ));
                checked(cudaMemset(arena.at<void>(status), 0, sizeof(unsigned)));
                for (std::size_t first = begin; first < end; first += plan.points) {
                    stop(cancel, failed);
                    const auto count = std::min(plan.points, end - first);
                    checked(cudaMemcpy2D(
                        arena.at<void>(x),
                        plan.points * 4,
                        input.data() + first,
                        points * 4,
                        count * 4,
                        gamma.size(),
                        cudaMemcpyHostToDevice
                    ));
                    image_vae_norm_kernel<<<unsigned((count + 255) / 256), 256>>>(
                        arena.at<float>(x),
                        arena.at<float>(g),
                        arena.at<float>(y),
                        arena.at<unsigned>(status),
                        gamma.size(),
                        plan.points,
                        count,
                        activate
                    );
                    checked(cudaGetLastError());
                    checked(cudaStreamSynchronize(cudaStreamLegacy));
                    unsigned invalid = 0;
                    checked(cudaMemcpy(
                        &invalid,
                        arena.at<void>(status),
                        sizeof(invalid),
                        cudaMemcpyDeviceToHost
                    ));
                    if (invalid) {
                        nonfinite = true;
                        return;
                    }
                    stop(cancel, failed);
                    checked(cudaMemcpy2D(
                        output.data() + first,
                        points * 4,
                        arena.at<void>(y),
                        plan.points * 4,
                        count * 4,
                        gamma.size(),
                        cudaMemcpyDeviceToHost
                    ));
                }
            }
        );
        require(!nonfinite.load(), "image_vae_nonfinite");
        return true;
    }

    bool image_silu_pair_bf16_source(
        const WeightIdentity& gate_identity,
        const Bf16WeightSource& gate_source,
        const WeightIdentity& up_identity,
        const Bf16WeightSource& up_source,
        std::span<const float> x,
        std::size_t in,
        std::size_t intermediate,
        std::span<float> y,
        const std::atomic_bool& cancel
    ) override {
        if (!native_bf16_)
            return false;
        validate(x, {}, in, intermediate, y);
        require(
            !overlap(x.data(), x.size_bytes(), y.data(), y.size_bytes()),
            "h3_cuda_alias"
        );
        require(bool(gate_source) && bool(up_source), "h3_cuda_weight_source");
        // Preflight every selected device before any source read or output write.
        // A partition can only be smaller than this conservative full width.
        for (auto limit : scratch_limits_) {
            try {
                image_silu_pair_plan(in, intermediate, limit);
            } catch (const std::invalid_argument& error) {
                if (std::string_view(error.what()) == "image_pair_scratch_limit")
                    return false;
                throw;
            }
        }
        std::mutex source_mutex;
        split(
            intermediate,
            cancel,
            [&](DeviceWorkspace& workspace,
                std::size_t begin,
                std::size_t end,
                const std::atomic_bool& failed) {
                auto& arena =
                    static_cast<CudaDeviceOperations&>(workspace.operations());
                const auto rows = x.size() / in;
                const auto plan =
                    image_silu_pair_plan(in, end - begin, arena.scratch_limit);
                const auto input_rows =
                    image_silu_pair_input_rows(rows, in, plan, arena.scratch_limit);
                const bool keep_input = input_rows >= rows;
                Layout layout;
                const auto input = layout.add(input_rows * in * 4),
                           bf16 = layout.add(input_rows * in * 2),
                           gate_weight = layout.add(plan.columns * in * 2),
                           up_weight = layout.add(plan.columns * in * 2),
                           gate = layout.add(plan.rows * plan.columns * 4),
                           up = layout.add(plan.rows * plan.columns * 4),
                           blas = layout.add(8 * 1024 * 1024),
                           status = layout.add(sizeof(int));
                require(
                    layout.bytes <= arena.scratch_limit, "image_pair_scratch_limit"
                );
                // Reserve once; no later ensure can invalidate live input or tiles.
                workspace.ensure(layout.bytes);
                checked(cublasSetWorkspace(
                    arena.blas, arena.at<void>(blas), 8 * 1024 * 1024
                ));
                auto upload_input =
                    [&](std::size_t first, std::size_t count, std::size_t offset) {
                        stop(cancel, failed);
                        checked(cudaMemcpy(
                            arena.at<float>(input) + offset * in,
                            x.data() + first * in,
                            count * in * 4,
                            cudaMemcpyHostToDevice
                        ));
                        stop(cancel, failed);
                        bf16_input<<<unsigned((count * in + 255) / 256), 256>>>(
                            arena.at<float>(input) + offset * in,
                            arena.at<__nv_bfloat16>(bf16) + offset * in,
                            count * in
                        );
                        checked(cudaGetLastError());
                        checked(cudaStreamSynchronize(cudaStreamLegacy));
                        stop(cancel, failed);
                    };
                if (keep_input)
                    for (std::size_t first = 0; first < rows; first += plan.rows)
                        upload_input(first, std::min(plan.rows, rows - first), first);
                for (auto first_column = begin; first_column < end;
                     first_column += plan.columns) {
                    const auto columns = std::min(plan.columns, end - first_column);
                    const auto count = columns * in;
                    auto weight = [&](const WeightIdentity& identity,
                                      const Bf16WeightSource& source,
                                      std::size_t destination) -> void* {
                        stop(cancel, failed);
                        const DeviceWeights::Key key{
                            identity, first_column * in, count, sizeof(std::uint16_t)
                        };
                        if (auto retained = arena.weights.find(key))
                            return retained;
                        HostAdmission admission(
                            *resources_, count * 2 + 4096, workload_
                        );
                        auto staging =
                            std::make_unique_for_overwrite<std::uint16_t[]>(count);
                        {
                            // Shard callbacks may share parser state across
                            // devices/tensors.
                            std::lock_guard lock(source_mutex);
                            stop(cancel, failed);
                            source(first_column * in, {staging.get(), count});
                        }
                        stop(cancel, failed);
                        const std::span<const std::uint16_t> values{
                            staging.get(), count
                        };
                        for (auto value : values)
                            require(
                                (value & 0x7f80) != 0x7f80, "h3_cuda_nonfinite_weight"
                            );
                        if (auto retained = arena.weights.retain(
                                key, std::as_bytes(values), cancel
                            ))
                            return retained;
                        checked(cudaMemcpy(
                            arena.at<void>(destination),
                            staging.get(),
                            count * 2,
                            cudaMemcpyHostToDevice
                        ));
                        stop(cancel, failed);
                        return arena.at<void>(destination);
                    };
                    // Release each host staging allocation before reading the other
                    // tile.
                    auto* gate_pointer =
                        weight(gate_identity, gate_source, gate_weight);
                    auto* up_pointer = weight(up_identity, up_source, up_weight);
                    for (std::size_t first = 0; first < rows; first += plan.rows) {
                        const auto tile_rows = std::min(plan.rows, rows - first);
                        if (!keep_input)
                            upload_input(first, tile_rows, 0);
                        const auto* input_pointer = arena.at<__nv_bfloat16>(bf16)
                                                    + (keep_input ? first * in : 0);
                        auto gemm = [&](void* weights, std::size_t result) {
                            stop(cancel, failed);
                            const float one = 1, zero = 0;
                            checked(cublasGemmEx(
                                arena.blas,
                                CUBLAS_OP_T,
                                CUBLAS_OP_N,
                                int(columns),
                                int(tile_rows),
                                int(in),
                                &one,
                                weights,
                                CUDA_R_16BF,
                                int(in),
                                input_pointer,
                                CUDA_R_16BF,
                                int(in),
                                &zero,
                                arena.at<void>(result),
                                CUDA_R_32F,
                                int(columns),
                                CUBLAS_COMPUTE_32F,
                                CUBLAS_GEMM_DEFAULT
                            ));
                            checked(cudaStreamSynchronize(cudaStreamLegacy));
                            stop(cancel, failed);
                        };
                        with_projection_math(
                            true,
                            CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION,
                            [&] {
                                cublasMath_t mode;
                                checked(cublasGetMathMode(arena.blas, &mode));
                                return mode;
                            },
                            [&](cublasMath_t mode) {
                                checked(cublasSetMathMode(arena.blas, mode));
                            },
                            [&] {
                                gemm(gate_pointer, gate);
                                gemm(up_pointer, up);
                            }
                        );
                        stop(cancel, failed);
                        checked(cudaMemset(arena.at<void>(status), 0, sizeof(int)));
                        image_silu_pair_kernel<<<
                            unsigned((tile_rows * columns + 255) / 256),
                            256>>>(
                            arena.at<float>(gate),
                            arena.at<float>(up),
                            tile_rows * columns,
                            arena.at<unsigned>(status)
                        );
                        checked(cudaGetLastError());
                        int result = 0;
                        checked(cudaMemcpy(
                            &result,
                            arena.at<void>(status),
                            sizeof(result),
                            cudaMemcpyDeviceToHost
                        ));
                        stop(cancel, failed);
                        require(!(result & 1), "h3_cuda_nonfinite_output");
                        require(!(result & 2), "h3_cuda_nonfinite_input");
                        checked(cudaMemcpy2D(
                            y.data() + first * intermediate + first_column,
                            intermediate * 4,
                            arena.at<void>(gate),
                            columns * 4,
                            columns * 4,
                            tile_rows,
                            cudaMemcpyDeviceToHost
                        ));
                        stop(cancel, failed);
                    }
                }
            },
            true
        );
        return true;
    }

    bool supports_weight_sources() const override {
        return true;
    }

    bool supports_bf16_weights() const override {
        return native_bf16_;
    }

    ProjectionResult dense_bf16_source(
        const WeightIdentity& identity,
        const Bf16WeightSource& source,
        std::span<const float> b,
        std::span<const float> x,
        std::size_t in,
        std::size_t out,
        std::span<float> y,
        const std::atomic_bool& c
    ) override {
        if (!native_bf16_)
            return ProjectionResult::unsupported;
        project<std::uint16_t>(
            {}, {}, b, x, in, out, 0, y, c, false, &identity, &source
        );
        return ProjectionResult::finite;
    }

    ProjectionResult dense_source(
        const WeightIdentity& identity,
        const FloatWeightSource& source,
        std::span<const float> b,
        std::span<const float> x,
        std::size_t in,
        std::size_t out,
        std::span<float> y,
        const std::atomic_bool& c,
        bool precise,
        ProjectionPolicy policy
    ) override {
        project<float>(
            {}, {}, b, x, in, out, 0, y, c, precise, &identity, &source, policy
        );
        return ProjectionResult::finite;
    }

    ProjectionResult dense_gguf_source(
        const WeightIdentity& identity,
        gguf::Encoding encoding,
        const GgufWeightSource& source,
        std::span<const float> b,
        std::span<const float> x,
        std::size_t in,
        std::size_t out,
        std::span<float> y,
        const std::atomic_bool& c,
        bool precise
    ) override {
        project<float>(
            {},
            {},
            b,
            x,
            in,
            out,
            0,
            y,
            c,
            precise,
            &identity,
            nullptr,
            ProjectionPolicy::standard,
            nullptr,
            &source,
            encoding
        );
        return ProjectionResult::finite;
    }

    bool convrot_source(
        const WeightIdentity& identity,
        const ByteWeightSource& source,
        std::span<const float> s,
        std::span<const float> b,
        std::span<const float> x,
        std::size_t in,
        std::size_t out,
        std::size_t g,
        std::span<float> y,
        const std::atomic_bool& c
    ) override {
        project<std::uint8_t>({}, s, b, x, in, out, g, y, c, false, &identity, &source);
        return true;
    }

    void dense_weight(
        const WeightIdentity& identity,
        std::span<const float> w,
        std::span<const float> b,
        std::span<const float> x,
        std::size_t in,
        std::size_t out,
        std::span<float> y,
        const std::atomic_bool& c,
        bool precise
    ) override {
        project(w, {}, b, x, in, out, 0, y, c, precise, &identity);
    }

    void dense(
        std::span<const float> w,
        std::span<const float> b,
        std::span<const float> x,
        std::size_t in,
        std::size_t out,
        std::span<float> y,
        const std::atomic_bool& c,
        bool precise
    ) override {
        project(w, {}, b, x, in, out, 0, y, c, precise);
    }

    void convrot(
        std::span<const std::uint8_t> w,
        std::span<const float> s,
        std::span<const float> b,
        std::span<const float> x,
        std::size_t in,
        std::size_t out,
        std::size_t g,
        std::span<float> y,
        const std::atomic_bool& c
    ) override {
        project(w, s, b, x, in, out, g, y, c);
    }

    void convrot_weight(
        const WeightIdentity& identity,
        std::span<const std::uint8_t> w,
        std::span<const float> s,
        std::span<const float> b,
        std::span<const float> x,
        std::size_t in,
        std::size_t out,
        std::size_t g,
        std::span<float> y,
        const std::atomic_bool& c
    ) override {
        project(w, s, b, x, in, out, g, y, c, false, &identity);
    }
};
}  // namespace

CudaBf16Projection::CudaBf16Projection(
    std::shared_ptr<DenseCompute> context_owner,
    std::shared_ptr<MediaSessionActivity> activity,
    std::shared_ptr<Resources> resources,
    Workload workload,
    int device,
    image::AttentionPrecision attention_precision
)
    : context_owner_(std::move(context_owner)), activity_(std::move(activity)),
      resources_(std::move(resources)), device_(device),
      attention_precision_(attention_precision) {
    image::validate_attention_precision(attention_precision_, workload);
    require(
        context_owner_ && activity_ && resources_ && device >= 0
            && std::size_t(device) + 1 < resources_->snapshot().capacity.size(),
        "bf16_projection_context_owner"
    );
    // Initialization may touch context resources even when CUDA reports failure.
    ++activity_->owners;
    checked(cudaSetDevice(device_));
    int major = 0;
    checked(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device_));
    require(major >= 8, "h3_cuda_bf16_policy");
    std::size_t before = 0, after = 0, total = 0;
    checked(cudaMemGetInfo(&before, &total));
    cublasHandle_t handle = nullptr;
    checked(cublasCreate(&handle));
    blas_ = handle;
    try {
        checked(cudaMemGetInfo(&after, &total));
        Footprint charge(resources_->snapshot().capacity.size());
        charge[device_ + 1] = before > after ? before - after : 0;
        library_charge_ = resources_->reserve(workload, charge);
    } catch (...) {
        close();
        throw;
    }
}

CudaBf16Projection::~CudaBf16Projection() {
    try {
        close();
    } catch (...) {
        // Unconfirmed cleanup must not permit WorkerCompute to park/reset.
    }
}

void CudaBf16Projection::close() {
    if (!blas_)
        return;
    try {
        checked(cudaSetDevice(device_));
        if (stream_)
            checked(cudaStreamSynchronize(static_cast<cudaStream_t>(stream_)));
        checked(cublasDestroy(static_cast<cublasHandle_t>(blas_)));
    } catch (...) {
        quarantined_ = true;
        throw;
    }
    blas_ = nullptr;
    if (library_charge_) {
        resources_->released(library_charge_);
        library_charge_ = 0;
    }
    --activity_->owners;
}

void CudaBf16Projection::launch(
    void* stream,
    const MediaDeviceSpan& input,
    const MediaDeviceSpan& weight,
    const MediaDeviceSpan& bias,
    const MediaDeviceSpan& rounded,
    const MediaDeviceSpan& output,
    const MediaDeviceSpan& workspace,
    const MediaDeviceSpan& status
) {
    const auto shape = validate_bf16_projection(
        device_, input, weight, bias, rounded, output, workspace, status
    );
    require(
        blas_ && !quarantined_ && stream && (!stream_ || stream_ == stream),
        "bf16_projection_owned_stream"
    );
    stream_ = stream;
    checked(cudaSetDevice(device_));
    const auto owned_stream = static_cast<cudaStream_t>(stream);
    const auto handle = static_cast<cublasHandle_t>(blas_);
    // Stream binding resets cuBLAS workspace, so bind before installing the
    // session allocation. Never borrow the host wrapper's growable arena.
    checked(cublasSetStream(handle, owned_stream));
    checked(cublasSetWorkspace(handle, workspace.data, workspace.bytes));
    checked(cudaMemsetAsync(status.data, 0, sizeof(unsigned), owned_stream));
    launch_bf16_projection(
        owned_stream,
        handle,
        static_cast<const float*>(input.data),
        weight.data,
        static_cast<const float*>(bias.data),
        static_cast<__nv_bfloat16*>(rounded.data),
        static_cast<float*>(output.data),
        static_cast<unsigned*>(status.data),
        shape.rows,
        shape.in,
        shape.out
    );
}

void CudaBf16Projection::launch_image_silu_pair(
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
) {
    const auto shape = validate_image_silu_pair(
        device_,
        input,
        gate_weight,
        up_weight,
        rounded,
        gate,
        up,
        workspace,
        gate_status,
        up_status,
        status
    );
    const MediaDeviceSpan bias{nullptr, device_, DeviceDType::f32, {}, {}, 0, 0, 0};
    launch(stream, input, gate_weight, bias, rounded, gate, workspace, gate_status);
    launch(stream, input, up_weight, bias, rounded, up, workspace, up_status);
    const auto owned_stream = static_cast<cudaStream_t>(stream);
    image_projection_status<<<1, 1, 0, owned_stream>>>(
        static_cast<const unsigned*>(gate_status.data),
        static_cast<const unsigned*>(up_status.data),
        static_cast<unsigned*>(status.data)
    );
    checked(cudaGetLastError());
    // F2 already performs the bias-free +0. Repeating it in the shared image
    // kernel is idempotent, including -0; keep its existing arithmetic unchanged.
    image_silu_pair_kernel<<<
        unsigned((shape.rows * shape.out + 255) / 256),
        256,
        0,
        owned_stream>>>(
        static_cast<float*>(gate.data),
        static_cast<const float*>(up.data),
        shape.rows * shape.out,
        static_cast<unsigned*>(status.data)
    );
    checked(cudaGetLastError());
}

void CudaBf16Projection::standard(
    void* stream,
    std::span<const MediaDeviceSpan> inputs,
    std::span<const MediaDeviceSpan> outputs
) {
    require(inputs.size() == 2 && outputs.size() == 4, "image_attention_projection");
    const MediaDeviceSpan bias{nullptr, device_, DeviceDType::f32, {}, {}, 0, 0, 0};
    // This is also project_source's STANDARD native-BF16 branch: identical input
    // rounding, FP32 reduction, post-GEMM +0 and stored-output finite check.
    launch(
        stream,
        inputs[0],
        inputs[1],
        bias,
        outputs[1],
        outputs[0],
        outputs[2],
        outputs[3]
    );
}

void CudaBf16Projection::text_normalization(
    void* stream,
    std::span<const MediaDeviceSpan> inputs,
    std::span<const MediaDeviceSpan> outputs
) {
    require(
        inputs.size() == 2 && outputs.size() == 2, "image_text_normalization_layout"
    );
    const auto shape = validate_image_text_normalization(
        device_, inputs[0], inputs[1], outputs[0], outputs[1]
    );
    require(
        blas_ && !quarantined_ && stream && (!stream_ || stream_ == stream),
        "bf16_projection_owned_stream"
    );
    stream_ = stream;
    checked(cudaSetDevice(device_));
    const auto owned_stream = static_cast<cudaStream_t>(stream);
    auto* status = static_cast<unsigned*>(outputs[1].data);
    checked(cudaMemsetAsync(status, 0, sizeof(unsigned), owned_stream));
    image_text_norm_kernel<<<unsigned(shape.rows), 256, 0, owned_stream>>>(
        static_cast<const float*>(inputs[0].data),
        inputs[1].data,
        inputs[1].dtype == DeviceDType::bf16,
        static_cast<float*>(outputs[0].data),
        shape.in,
        status
    );
    checked(cudaGetLastError());
}

void CudaBf16Projection::rotary(
    void* stream,
    std::span<const MediaDeviceSpan> inputs,
    std::span<const MediaDeviceSpan> outputs,
    std::size_t heads,
    std::size_t head_dim,
    const std::array<std::size_t, 3>& axes,
    std::size_t text,
    std::size_t height,
    std::size_t width
) {
    require(inputs.size() == 2 && outputs.size() == 2, "image_rotary_layout");
    const auto shape = validate_image_rotary(
        device_,
        inputs[0],
        inputs[1],
        outputs[0],
        outputs[1],
        heads,
        head_dim,
        axes,
        text,
        height,
        width
    );
    require(
        blas_ && !quarantined_ && stream && (!stream_ || stream_ == stream),
        "bf16_projection_owned_stream"
    );
    stream_ = stream;
    checked(cudaSetDevice(device_));
    const auto owned_stream = static_cast<cudaStream_t>(stream);
    auto* status = static_cast<unsigned*>(outputs[1].data);
    checked(cudaMemsetAsync(status, 0, sizeof(unsigned), owned_stream));
    image_rotary_kernel<<<
        dim3(unsigned(shape.rows), unsigned(heads)),
        64,
        0,
        owned_stream>>>(
        static_cast<float*>(outputs[0].data),
        inputs[1].data,
        inputs[1].dtype == DeviceDType::bf16,
        heads,
        head_dim,
        axes[0],
        axes[1],
        axes[2],
        text,
        height,
        width,
        status
    );
    checked(cudaGetLastError());
}

void CudaBf16Projection::masked_attention(
    void* stream,
    std::span<const MediaDeviceSpan> inputs,
    std::span<const MediaDeviceSpan> outputs,
    std::size_t heads,
    std::size_t head_dim,
    std::size_t text,
    std::size_t first,
    const std::atomic_bool& cancel
) {
    require(inputs.size() == 3 && outputs.size() == 4, "image_attention_layout");
    const auto shape = validate_image_masked_attention(
        device_,
        inputs[0],
        inputs[1],
        inputs[2],
        outputs[0],
        outputs[1],
        outputs[2],
        outputs[3],
        heads,
        head_dim,
        text,
        first
    );
    require(
        blas_ && !quarantined_ && stream && (!stream_ || stream_ == stream),
        "bf16_projection_owned_stream"
    );
    require(!cancel.load(), "h3_cuda_cancelled");
    stream_ = stream;
    checked(cudaSetDevice(device_));
    const auto owned_stream = static_cast<cudaStream_t>(stream);
    const auto handle = static_cast<cublasHandle_t>(blas_);
    checked(cublasSetStream(handle, owned_stream));
    checked(cublasSetWorkspace(handle, outputs[2].data, outputs[2].bytes));
    auto* status = static_cast<unsigned*>(outputs[3].data);
    checked(cudaMemsetAsync(status, 0, sizeof(unsigned), owned_stream));
    const auto tokens = inputs[0].dimensions[0];
    const ImageAttentionScratch layout(tokens, head_dim);
    const auto* key_source = static_cast<const float*>(inputs[1].data);
    const auto* value_source = static_cast<const float*>(inputs[2].data);
    check_image_key_value<<<
        unsigned((tokens * shape.in + 255) / 256),
        256,
        0,
        owned_stream>>>(key_source, value_source, unsigned(tokens * shape.in), status);
    checked(cudaGetLastError());
    auto at = [&](Bytes offset) {
        return reinterpret_cast<float*>(
            static_cast<std::byte*>(outputs[1].data) + offset
        );
    };
    const float scale = 1 / std::sqrt(float(head_dim)), zero = 0, one = 1;
    // Keep QK and the default PV in attend<float>'s PEDANTIC F32 mode.
    with_projection_math(
        true,
        CUBLAS_PEDANTIC_MATH,
        [&] {
            cublasMath_t mode;
            checked(cublasGetMathMode(handle, &mode));
            return mode;
        },
        [&](cublasMath_t mode) { checked(cublasSetMathMode(handle, mode)); },
        [&] {
            for (std::size_t head = 0; head < heads; ++head) {
                require(!cancel.load(), "h3_cuda_cancelled");
                pack_image_attention<<<
                    unsigned((shape.rows * head_dim + 255) / 256),
                    256,
                    0,
                    owned_stream>>>(
                    static_cast<const float*>(inputs[0].data),
                    at(layout.q),
                    unsigned(shape.rows),
                    unsigned(shape.in),
                    unsigned(head_dim),
                    unsigned(first),
                    unsigned(head),
                    status
                );
                checked(cudaGetLastError());
                checked(cudaMemsetAsync(
                    at(layout.output),
                    0,
                    shape.rows * head_dim * sizeof(float),
                    owned_stream
                ));
                initialize_attention<<<1, 128, 0, owned_stream>>>(
                    at(layout.maximum), at(layout.total), unsigned(shape.rows)
                );
                checked(cudaGetLastError());
                for (std::size_t key = 0; key < tokens; key += layout.keys) {
                    require(!cancel.load(), "h3_cuda_cancelled");
                    const auto keys = std::min(layout.keys, tokens - key);
                    const auto source = key * shape.in + head * head_dim;
                    checked(
                        gemm<float>(
                            handle,
                            CUBLAS_OP_T,
                            CUBLAS_OP_N,
                            int(keys),
                            int(shape.rows),
                            int(head_dim),
                            scale,
                            key_source + source,
                            int(shape.in),
                            at(layout.q),
                            int(head_dim),
                            zero,
                            at(layout.scores),
                            int(keys)
                        )
                    );
                    online_softmax<<<unsigned(shape.rows), 256, 0, owned_stream>>>(
                        at(layout.scores),
                        at(layout.output),
                        at(layout.maximum),
                        at(layout.total),
                        unsigned(keys),
                        unsigned(head_dim),
                        unsigned(first),
                        unsigned(key),
                        unsigned(text),
                        0,
                        0
                    );
                    checked(cudaGetLastError());
                    // Request TF32 only for PV products, restoring PEDANTIC before
                    // the next QK. Keep F32 accumulation and beta=1 for the
                    // rescaled numerator; cuBLAS still selects the actual kernel.
                    with_projection_math(
                        attention_precision_
                            == image::AttentionPrecision::experimental_pv_tf32,
                        CUBLAS_TF32_TENSOR_OP_MATH,
                        [&] {
                            cublasMath_t mode;
                            checked(cublasGetMathMode(handle, &mode));
                            return mode;
                        },
                        [&](cublasMath_t mode) {
                            checked(cublasSetMathMode(handle, mode));
                        },
                        [&] {
                            checked(
                                gemm<float>(
                                    handle,
                                    CUBLAS_OP_N,
                                    CUBLAS_OP_N,
                                    int(head_dim),
                                    int(shape.rows),
                                    int(keys),
                                    one,
                                    value_source + source,
                                    int(shape.in),
                                    at(layout.scores),
                                    int(keys),
                                    one,
                                    at(layout.output),
                                    int(head_dim)
                                )
                            );
                        }
                    );
                }
                divide_attention<<<
                    unsigned((shape.rows * head_dim + 255) / 256),
                    256,
                    0,
                    owned_stream>>>(
                    at(layout.output),
                    at(layout.total),
                    unsigned(shape.rows),
                    unsigned(head_dim)
                );
                checked(cudaGetLastError());
                scatter_image_attention<<<
                    unsigned((shape.rows * head_dim + 255) / 256),
                    256,
                    0,
                    owned_stream>>>(
                    at(layout.output),
                    at(layout.total),
                    static_cast<float*>(outputs[0].data),
                    unsigned(shape.rows),
                    unsigned(shape.in),
                    unsigned(head_dim),
                    unsigned(head),
                    status
                );
                checked(cudaGetLastError());
            }
        }
    );
}

void CudaBf16Projection::text_projection(
    void* stream,
    std::span<const MediaDeviceSpan> inputs,
    std::span<const MediaDeviceSpan> outputs
) {
    standard(stream, inputs, outputs);
    const auto count = outputs[0].bytes / sizeof(float);
    // Keep the projection status so activation cannot conceal a bad producer.
    image_text_gelu_kernel<<<
        unsigned((count + 255) / 256),
        256,
        0,
        static_cast<cudaStream_t>(stream)>>>(
        static_cast<float*>(outputs[0].data),
        count,
        static_cast<unsigned*>(outputs[3].data)
    );
    checked(cudaGetLastError());
}

void CudaBf16Projection::frequency(
    void* stream, std::span<const MediaDeviceSpan> outputs, float timestep
) {
    require(outputs.size() == 2, "image_frequency_layout");
    validate_image_frequency(device_, outputs[0], outputs[1], timestep);
    require(
        blas_ && !quarantined_ && stream && (!stream_ || stream_ == stream),
        "bf16_projection_owned_stream"
    );
    stream_ = stream;
    checked(cudaSetDevice(device_));
    const auto owned_stream = static_cast<cudaStream_t>(stream);
    auto* status = static_cast<unsigned*>(outputs[1].data);
    checked(cudaMemsetAsync(status, 0, sizeof(unsigned), owned_stream));
    image_frequency_kernel<<<2, 256, 0, owned_stream>>>(
        static_cast<float*>(outputs[0].data), timestep, status
    );
    checked(cudaGetLastError());
}

void CudaBf16Projection::timestep(
    void* stream,
    std::span<const MediaDeviceSpan> inputs,
    std::span<const MediaDeviceSpan> outputs
) {
    // Standard validates every span and stream before enqueue. Keep its status:
    // SiLU must not conceal a nonfinite projection behind a finite negative tail.
    standard(stream, inputs, outputs);
    const auto count = outputs[0].bytes / sizeof(float);
    image_timestep_silu_kernel<<<
        unsigned((count + 255) / 256),
        256,
        0,
        static_cast<cudaStream_t>(stream)>>>(
        static_cast<float*>(outputs[0].data),
        count,
        static_cast<unsigned*>(outputs[3].data)
    );
    checked(cudaGetLastError());
}

void CudaBf16Projection::attention_residual(
    void* stream,
    std::span<const MediaDeviceSpan> inputs,
    std::span<const MediaDeviceSpan> outputs,
    std::size_t text
) {
    require(
        inputs.size() == 3 && outputs.size() == 3, "image_attention_residual_layout"
    );
    const auto shape = validate_image_attention_residual(
        device_,
        inputs[0],
        inputs[1],
        inputs[2],
        outputs[0],
        outputs[1],
        outputs[2],
        text
    );
    require(
        blas_ && !quarantined_ && stream && (!stream_ || stream_ == stream),
        "bf16_projection_owned_stream"
    );
    stream_ = stream;
    checked(cudaSetDevice(device_));
    const auto owned_stream = static_cast<cudaStream_t>(stream);
    auto* status = static_cast<unsigned*>(outputs[2].data);
    checked(cudaMemsetAsync(status, 0, sizeof(unsigned), owned_stream));
    image_gates_kernel<<<unsigned((8 * shape.in + 255) / 256), 256, 0, owned_stream>>>(
        static_cast<const float*>(inputs[2].data),
        static_cast<float*>(outputs[1].data),
        shape.in,
        0,
        status
    );
    checked(cudaGetLastError());
    // Keep the gate producer's status; residual() would reset that certificate.
    image_residual_kernel<<<
        unsigned((shape.rows * shape.in + 255) / 256),
        256,
        0,
        owned_stream>>>(
        static_cast<float*>(outputs[0].data),
        static_cast<const float*>(inputs[1].data),
        static_cast<const float*>(outputs[1].data),
        shape.rows * shape.in,
        shape.in,
        0,
        text,
        status
    );
    checked(cudaGetLastError());
}

void CudaBf16Projection::prepare(
    void* stream,
    std::span<const MediaDeviceSpan> inputs,
    std::span<const MediaDeviceSpan> outputs,
    std::size_t text,
    ImageModulationStage stage
) {
    require(inputs.size() == 2 && outputs.size() == 3, "image_ff_preparation_layout");
    require(
        stage == ImageModulationStage::qkv
            || stage == ImageModulationStage::feed_forward,
        "image_modulation_stage"
    );
    const auto shape = validate_image_modulation(
        device_, inputs[0], inputs[1], outputs[0], outputs[1], outputs[2], text
    );
    require(
        blas_ && !quarantined_ && stream && (!stream_ || stream_ == stream),
        "bf16_projection_owned_stream"
    );
    stream_ = stream;
    checked(cudaSetDevice(device_));
    const auto owned_stream = static_cast<cudaStream_t>(stream);
    auto* status = static_cast<unsigned*>(outputs[2].data);
    checked(cudaMemsetAsync(status, 0, sizeof(unsigned), owned_stream));
    image_gates_kernel<<<unsigned((8 * shape.in + 255) / 256), 256, 0, owned_stream>>>(
        static_cast<const float*>(inputs[1].data),
        static_cast<float*>(outputs[1].data),
        shape.in,
        std::size_t(stage),
        status
    );
    checked(cudaGetLastError());
    image_norm_kernel<<<unsigned(shape.rows), 256, 0, owned_stream>>>(
        static_cast<const float*>(inputs[0].data),
        static_cast<const float*>(inputs[1].data),
        static_cast<float*>(outputs[0].data),
        shape.in,
        text,
        false,
        std::size_t(stage),
        status
    );
    checked(cudaGetLastError());
}

void CudaBf16Projection::final_normalization(
    void* stream,
    std::span<const MediaDeviceSpan> inputs,
    std::span<const MediaDeviceSpan> outputs
) {
    require(
        inputs.size() == 2 && outputs.size() == 2, "image_final_normalization_layout"
    );
    const auto shape = validate_image_final_normalization(
        device_, inputs[0], inputs[1], outputs[0], outputs[1]
    );
    require(
        blas_ && !quarantined_ && stream && (!stream_ || stream_ == stream),
        "bf16_projection_owned_stream"
    );
    stream_ = stream;
    checked(cudaSetDevice(device_));
    const auto owned_stream = static_cast<cudaStream_t>(stream);
    auto* status = static_cast<unsigned*>(outputs[1].data);
    checked(cudaMemsetAsync(status, 0, sizeof(unsigned), owned_stream));
    image_norm_kernel<<<unsigned(shape.rows), 256, 0, owned_stream>>>(
        static_cast<const float*>(inputs[0].data),
        static_cast<const float*>(inputs[1].data),
        static_cast<float*>(outputs[0].data),
        shape.in,
        0,
        true,
        0,
        status
    );
    checked(cudaGetLastError());
}

void CudaBf16Projection::scheduler(
    void* stream,
    std::span<const MediaDeviceSpan> inputs,
    std::span<const MediaDeviceSpan> outputs,
    float current,
    float next
) {
    require(inputs.size() == 2 && outputs.size() == 2, "image_scheduler_layout");
    const auto count = validate_image_scheduler(
        device_, inputs[0], inputs[1], outputs[0], outputs[1], current, next
    );
    require(
        blas_ && !quarantined_ && stream && (!stream_ || stream_ == stream),
        "bf16_projection_owned_stream"
    );
    stream_ = stream;
    checked(cudaSetDevice(device_));
    const auto owned_stream = static_cast<cudaStream_t>(stream);
    auto* status = static_cast<unsigned*>(outputs[1].data);
    checked(cudaMemsetAsync(status, 0, sizeof(unsigned), owned_stream));
    const float dt = next - current;
    image_scheduler_kernel<<<unsigned((count + 255) / 256), 256, 0, owned_stream>>>(
        static_cast<float*>(outputs[0].data),
        static_cast<const float*>(inputs[1].data),
        count,
        dt,
        status
    );
    checked(cudaGetLastError());
}

void CudaBf16Projection::pair(
    void* stream,
    std::span<const MediaDeviceSpan> inputs,
    std::span<const MediaDeviceSpan> outputs
) {
    require(inputs.size() == 3 && outputs.size() == 6, "image_ff_pair_layout");
    require(inputs[0].dimensions.size() == 2, "image_ff_pair_layout");
    const auto [gate, up] = split_image_silu_pair(outputs[0], inputs[0].dimensions[0]);
    launch_image_silu_pair(
        stream,
        inputs[0],
        inputs[1],
        inputs[2],
        outputs[1],
        gate,
        up,
        outputs[2],
        outputs[3],
        outputs[4],
        outputs[5]
    );
}

void CudaBf16Projection::down(
    void* stream,
    std::span<const MediaDeviceSpan> inputs,
    std::span<const MediaDeviceSpan> outputs
) {
    require(inputs.size() == 2 && outputs.size() == 4, "image_ff_down_layout");
    const MediaDeviceSpan bias{nullptr, device_, DeviceDType::f32, {}, {}, 0, 0, 0};
    launch(
        stream,
        inputs[0],
        inputs[1],
        bias,
        outputs[1],
        outputs[0],
        outputs[2],
        outputs[3]
    );
}

void CudaBf16Projection::residual(
    void* stream,
    const MediaDeviceSpan& hidden,
    const MediaDeviceSpan& projected,
    const MediaDeviceSpan& gates,
    const MediaDeviceSpan& status,
    std::size_t first,
    std::size_t text
) {
    const auto shape =
        validate_image_residual(device_, hidden, projected, gates, status, first, text);
    require(
        blas_ && !quarantined_ && stream && (!stream_ || stream_ == stream),
        "bf16_projection_owned_stream"
    );
    stream_ = stream;
    checked(cudaSetDevice(device_));
    const auto owned_stream = static_cast<cudaStream_t>(stream);
    checked(cudaMemsetAsync(status.data, 0, sizeof(unsigned), owned_stream));
    image_residual_kernel<<<
        unsigned((shape.rows * shape.in + 255) / 256),
        256,
        0,
        owned_stream>>>(
        static_cast<float*>(hidden.data),
        static_cast<const float*>(projected.data),
        static_cast<const float*>(gates.data),
        shape.rows * shape.in,
        shape.in,
        first,
        text,
        static_cast<unsigned*>(status.data)
    );
    checked(cudaGetLastError());
}

namespace {
unsigned blocks(std::size_t count) {
    return unsigned((count + 255) / 256);
}

__global__ void h3_finite_kernel(const float* x, std::size_t count, unsigned* status) {
    const auto i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count && !isfinite(x[i]))
        atomicOr(status, 1U);
}

__global__ void h3_widen_kernel(
    const std::uint16_t* source,
    double* destination,
    std::size_t count,
    unsigned* status
) {
    const auto i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count)
        return;
    const auto value = __uint_as_float(unsigned(source[i]) << 16);
    if (!isfinite(value))
        atomicOr(status, 1U);
    destination[i] = value;
}

// Preserve the host projection's F32 rounding and bias-free +0 between GEMMs.
__global__ void h3_zero_bias_kernel(double* x, std::size_t count) {
    const auto i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count)
        x[i] = __fadd_rn(float(x[i]), 0.f);
}

// Same arithmetic as finish_int8 with a zero bias, into a column tile of y.
__global__ void h3_finish_int8_kernel(
    const std::int32_t* sums,
    const float* row_scale,
    const float* weight_scale,
    float* y,
    std::size_t rows,
    std::size_t tile,
    std::size_t out,
    std::size_t first_column,
    const std::uint16_t* bias,
    unsigned* status
) {
    const auto i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= rows * tile)
        return;
    const auto row = i / tile, column = i % tile;
    // finish_int8 adds the exact BF16 bias, or zero for a bias-free projection.
    const float value = __fadd_rn(
        __fmul_rn(
            __fmul_rn(float(sums[i]), row_scale[row]),
            weight_scale[first_column + column]
        ),
        bias ? __uint_as_float(unsigned(bias[first_column + column]) << 16) : 0.f
    );
    if (!isfinite(value))
        atomicOr(status, 2U);
    y[row * out + first_column + column] = value;
}

// Host Turbo order: delta gets the projection's +0, then y += delta * scale.
__global__ void h3_lora_add_kernel(
    const double* delta,
    float* y,
    std::size_t rows,
    std::size_t tile,
    std::size_t out,
    std::size_t first_column,
    float scale,
    unsigned* status
) {
    const auto i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= rows * tile)
        return;
    const auto at = (i / tile) * out + first_column + i % tile;
    const float value =
        __fadd_rn(y[at], __fmul_rn(__fadd_rn(float(delta[i]), 0.f), scale));
    if (!isfinite(value))
        atomicOr(status, 2U);
    y[at] = value;
}

// h3_denoiser.cpp feed(): double SiLU, F32 cast, then F32 product.
__global__ void h3_silu_gate_kernel(
    float* gate, const float* value, std::size_t count, unsigned* status
) {
    const auto i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count)
        return;
    const double g = gate[i];
    const float result = __fmul_rn(float(g / (1.0 + exp(-g))), value[i]);
    if (!isfinite(result))
        atomicOr(status, 1U);
    gate[i] = result;
}

// h3_denoiser.cpp residual(): F32 product, then F32 addition. The gate is row
// group * 6 + index of the block's modulation table.
__global__ void h3_gated_residual_kernel(
    float* hidden,
    const float* projected,
    const float* modulation,
    const std::uint32_t* token_groups,
    std::size_t count,
    std::size_t width,
    std::size_t groups,
    std::size_t first,
    std::size_t index,
    unsigned* status
) {
    const auto i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count)
        return;
    const auto group = token_groups[first + i / width];
    if (group >= groups) {
        atomicOr(status, 4U);
        return;
    }
    const float gate =
        modulation[(group * h3_modulation_rows + index) * width + i % width];
    const float result = __fadd_rn(hidden[i], __fmul_rn(projected[i], gate));
    if (!isfinite(result))
        atomicOr(status, 1U);
    hidden[i] = result;
}

__device__ float h3_bf16(std::uint16_t bits) {
    return __uint_as_float(unsigned(bits) << 16);
}

// h3_denoiser.cpp norm() then modulate(): double RMS (epsilon 1e-5), F32 cast,
// then F32 x * (1 + scale) + shift. One block of 256 threads per row.
__global__ void h3_normalize_kernel(
    const float* hidden,
    const std::uint16_t* weight,
    const float* modulation,
    const std::uint32_t* token_groups,
    float* normalized,
    std::size_t width,
    std::size_t groups,
    std::size_t first,
    std::size_t shift,
    std::size_t stride_rows,
    unsigned* status
) {
    __shared__ double partial[256];
    const auto row = std::size_t(blockIdx.x);
    const unsigned t = threadIdx.x;
    const float* x = hidden + row * width;
    double sum = 0;
    for (std::size_t c = t; c < width; c += 256) {
        const double v = x[c];
        sum = __dadd_rn(sum, __dmul_rn(v, v));
    }
    partial[t] = sum;
    __syncthreads();
    for (unsigned stride = 128; stride; stride /= 2) {
        if (t < stride)
            partial[t] = __dadd_rn(partial[t], partial[t + stride]);
        __syncthreads();
    }
    const double scale =
        __drcp_rn(sqrt(__dadd_rn(__ddiv_rn(partial[0], double(width)), 1e-5)));
    const float* base = nullptr;
    if (modulation) {
        const auto group = token_groups[first + row];
        if (group >= groups) {
            if (t == 0)
                atomicOr(status, 4U);
            return;
        }
        base = modulation + group * stride_rows * width;
    }
    // A plain norm may run in place: each thread reads and writes only its own
    // channels, after the reduction completes.
    float* y = normalized + row * width;
    for (std::size_t c = t; c < width; c += 256) {
        const float n = float(
            __dmul_rn(__dmul_rn(double(x[c]), scale), double(h3_bf16(weight[c])))
        );
        const float result =
            base ? __fadd_rn(
                       __fmul_rn(n, __fadd_rn(1.f, base[(shift + 1) * width + c])),
                       base[shift * width + c]
                   )
                 : n;
        if (!isfinite(result))
            atomicOr(status, 1U);
        y[c] = result;
    }
}

// h3_denoiser.cpp q/k norm() then rope(): per-head double RMS (epsilon 1e-5),
// F32 cast, then double rotation of channels c and c + 48 with the shared
// [token, channel] cos/sin table. One block of 128 threads per (row, head).
__global__ void h3_qk_rotary_kernel(
    float* x,
    const std::uint16_t* weight,
    const double* rotary,
    std::size_t width,
    std::size_t first,
    unsigned* status
) {
    __shared__ double partial[128];
    __shared__ float values[128];
    const auto row = std::size_t(blockIdx.x);
    const unsigned t = threadIdx.x;
    float* p = x + row * width + std::size_t(blockIdx.y) * 128;
    const float v = p[t];
    partial[t] = __dmul_rn(double(v), double(v));
    __syncthreads();
    for (unsigned stride = 64; stride; stride /= 2) {
        if (t < stride)
            partial[t] = __dadd_rn(partial[t], partial[t + stride]);
        __syncthreads();
    }
    const double scale = __drcp_rn(sqrt(__dadd_rn(__ddiv_rn(partial[0], 128.0), 1e-5)));
    values[t] =
        float(__dmul_rn(__dmul_rn(double(v), scale), double(h3_bf16(weight[t]))));
    __syncthreads();
    float out[2] = {values[t], 0.f};
    // Without a rotary table (refiner text attention) only the norm applies.
    if (!rotary) {
        if (!isfinite(out[0]))
            atomicOr(status, 1U);
        p[t] = out[0];
        return;
    }
    if (t < 48) {
        const double* angle = rotary + (first + row) * 96 + std::size_t(t) * 2;
        const double a = values[t], b = values[t + 48];
        out[0] = float(__dsub_rn(__dmul_rn(a, angle[0]), __dmul_rn(b, angle[1])));
        out[1] = float(__dadd_rn(__dmul_rn(b, angle[0]), __dmul_rn(a, angle[1])));
        if (!isfinite(out[1]))
            atomicOr(status, 1U);
        p[t + 48] = out[1];
    }
    if (t < 48 || t >= 96) {
        if (!isfinite(out[0]))
            atomicOr(status, 1U);
        p[t] = out[0];
    }
}

// h3_denoiser.cpp refiner: current[i] += projected[i] in F32.
__global__ void h3_add_kernel(
    float* hidden, const float* projected, std::size_t count, unsigned* status
) {
    const auto i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count)
        return;
    const float result = __fadd_rn(hidden[i], projected[i]);
    if (!isfinite(result))
        atomicOr(status, 1U);
    hidden[i] = result;
}

// Exact BF16 or F32 weights widened to F64, one column tile at a time.
__global__ void h3_widen_real_kernel(
    const void* source,
    bool bf16,
    double* destination,
    std::size_t count,
    unsigned* status
) {
    const auto i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count)
        return;
    const float value =
        bf16 ? __uint_as_float(
                   unsigned(static_cast<const std::uint16_t*>(source)[i]) << 16
               )
             : static_cast<const float*>(source)[i];
    if (!isfinite(value))
        atomicOr(status, 1U);
    destination[i] = value;
}

// narrow_bias into a column tile of y: F32(F64 sum) + bias, the host's precise
// projection, with a zero bias for bias-free layers.
__global__ void h3_narrow_kernel(
    const double* sums,
    const void* bias,
    bool bf16_bias,
    float* y,
    std::size_t rows,
    std::size_t tile,
    std::size_t out,
    std::size_t first_column,
    unsigned* status
) {
    const auto i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= rows * tile)
        return;
    const auto row = i / tile, column = first_column + i % tile;
    const float b =
        !bias ? 0.f
        : bf16_bias
            ? __uint_as_float(
                  unsigned(static_cast<const std::uint16_t*>(bias)[column]) << 16
              )
            : static_cast<const float*>(bias)[column];
    const float value = __fadd_rn(float(sums[i]), b);
    if (!isfinite(value))
        atomicOr(status, 2U);
    y[row * out + column] = value;
}

__global__ void h3_attention_init_kernel(
    const float* query,
    double* output,
    std::size_t ld,
    double* maxima,
    double* totals,
    unsigned queries,
    unsigned* status
) {
    const unsigned q = blockIdx.x, h = blockIdx.y, c = threadIdx.x;
    const auto at = std::size_t(q) * ld + std::size_t(h) * 128 + c;
    if (!isfinite(query[at]))
        atomicOr(status, 1U);
    output[at] = 0;
    if (c == 0) {
        maxima[h * queries + q] = -INFINITY;
        totals[h * queries + q] = 0;
    }
}

// The row arithmetic is shared; resident heads keep their explicit strided layout.
__global__ void h3_online_softmax_kernel(
    double* scores,
    double* output,
    std::size_t ld,
    double* maxima,
    double* totals,
    unsigned keys,
    unsigned queries
) {
    __shared__ double reduction[256];
    const unsigned q = blockIdx.x, h = blockIdx.y;
    const auto state = std::size_t(h) * queries + q;
    online_softmax_row(
        scores + state * keys,
        output + std::size_t(q) * ld + std::size_t(h) * 128,
        maxima + state,
        totals + state,
        keys,
        128,
        reduction
    );
}

__global__ void h3_attention_divide_kernel(
    const double* accumulated,
    float* output,
    std::size_t ld,
    const double* totals,
    unsigned queries,
    unsigned heads,
    unsigned* status
) {
    const auto i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= std::size_t(queries) * heads * 128)
        return;
    const auto q = i / (std::size_t(heads) * 128), h = i / 128 % heads, c = i % 128;
    const auto at = q * ld + h * 128 + c;
    const float value = float(accumulated[at] / totals[h * queries + q]);
    output[at] = value;
    if (!isfinite(value))
        atomicOr(status, 2U);
}
}  // namespace

CudaH3Launcher::CudaH3Launcher(
    std::shared_ptr<DenseCompute> context_owner,
    std::shared_ptr<MediaSessionActivity> activity,
    std::shared_ptr<Resources> resources,
    int device
)
    : context_owner_(std::move(context_owner)), activity_(std::move(activity)),
      resources_(std::move(resources)), device_(device) {
    require(
        context_owner_ && activity_ && resources_ && device >= 0
            && std::size_t(device) + 1 < resources_->snapshot().capacity.size(),
        "h3_resident_context_owner"
    );
    // Initialization may touch context resources even when CUDA reports failure.
    ++activity_->owners;
    checked(cudaSetDevice(device_));
    std::size_t before = 0, after = 0, total = 0;
    checked(cudaMemGetInfo(&before, &total));
    cublasHandle_t handle = nullptr;
    checked(cublasCreate(&handle));
    blas_ = handle;
    try {
        checked(cudaMemGetInfo(&after, &total));
        Footprint charge(resources_->snapshot().capacity.size());
        charge[device_ + 1] = before > after ? before - after : 0;
        library_charge_ = resources_->reserve(Workload::video, charge);
    } catch (...) {
        close();
        throw;
    }
}

CudaH3Launcher::~CudaH3Launcher() {
    try {
        close();
    } catch (...) {
        // Unconfirmed cleanup keeps the activity guard and its busy charge.
    }
}

void CudaH3Launcher::close() {
    if (!blas_)
        return;
    try {
        checked(cudaSetDevice(device_));
        if (stream_)
            checked(cudaStreamSynchronize(static_cast<cudaStream_t>(stream_)));
        checked(cublasDestroy(static_cast<cublasHandle_t>(blas_)));
    } catch (...) {
        quarantined_ = true;
        throw;
    }
    blas_ = nullptr;
    if (library_charge_) {
        resources_->released(library_charge_);
        library_charge_ = 0;
    }
    --activity_->owners;
}

void CudaH3Launcher::release_stream() {
    if (!stream_)
        return;
    try {
        checked(cudaSetDevice(device_));
        checked(cudaStreamSynchronize(static_cast<cudaStream_t>(stream_)));
    } catch (...) {
        // Never touch a stream its session may already have destroyed.
        stream_ = nullptr;
        quarantined_ = true;
        throw;
    }
    stream_ = nullptr;
}

void CudaH3Launcher::own(void* stream) {
    require(
        blas_ && !quarantined_ && stream && (!stream_ || stream_ == stream),
        "h3_resident_owned_stream"
    );
    stream_ = stream;
    checked(cudaSetDevice(device_));
}

void CudaH3Launcher::bind(void* stream, const MediaDeviceSpan& workspace) {
    own(stream);
    const auto handle = static_cast<cublasHandle_t>(blas_);
    // Stream binding resets the cuBLAS workspace; install the session buffer after.
    checked(cublasSetStream(handle, static_cast<cudaStream_t>(stream)));
    checked(cublasSetWorkspace(handle, workspace.data, workspace.bytes));
}

void CudaH3Launcher::convrot(
    void* stream,
    std::span<const MediaDeviceSpan> read,
    std::span<const MediaDeviceSpan> write,
    std::size_t group,
    float lora_scale
) {
    const auto shape = validate_h3_convrot(device_, read, write, group);
    const bool lora = shape.rank != 0;
    bind(stream, write[write.size() - 2]);
    const auto s = static_cast<cudaStream_t>(stream);
    const auto handle = static_cast<cublasHandle_t>(blas_);
    auto* status = static_cast<unsigned*>(write.back().data);
    const auto rows = shape.rows, in = shape.in, out = shape.out;
    const auto* input = static_cast<const float*>(read[0].data);
    auto* output = static_cast<float*>(write[0].data);
    auto* scratch = static_cast<std::byte*>(write[1].data);
    auto* rotated = reinterpret_cast<float*>(scratch);
    auto* codes = reinterpret_cast<std::int8_t*>(
        scratch + detail::h3_aligned(Bytes(rows) * in * 4)
    );
    auto* row_scales = reinterpret_cast<float*>(
        reinterpret_cast<std::byte*>(codes) + detail::h3_aligned(Bytes(rows) * in)
    );
    auto* sums =
        reinterpret_cast<std::byte*>(row_scales) + detail::h3_aligned(Bytes(rows) * 4);
    checked(cudaMemsetAsync(status, 0, sizeof(unsigned), s));
    h3_finite_kernel<<<blocks(rows * in), 256, 0, s>>>(input, rows * in, status);
    checked(cudaGetLastError());
    rotate_kernel<<<
        dim3(unsigned(rows), unsigned(in / group)),
        unsigned(group),
        0,
        s>>>(input, rotated, in, unsigned(group));
    checked(cudaGetLastError());
    quantize_kernel<<<unsigned(rows), 256, 0, s>>>(rotated, codes, row_scales, in);
    checked(cudaGetLastError());
    double* low = nullptr;
    const double* lora_b = nullptr;
    if (lora) {
        this->lora(
            stream,
            read[0],
            read[3],
            read[4],
            write[2],
            write[3],
            rows,
            in,
            out,
            shape.rank,
            status
        );
        low = static_cast<double*>(write[2].data);
        lora_b = static_cast<const double*>(write[3].data) + shape.rank * in;
    }
    for (std::size_t first = 0; first < out; first += shape.tile) {
        const auto tile = std::min(shape.tile, out - first);
        const std::int32_t one = 1, zero = 0;
        checked(cublasGemmEx(
            handle,
            CUBLAS_OP_T,
            CUBLAS_OP_N,
            int(tile),
            int(rows),
            int(in),
            &one,
            static_cast<const std::int8_t*>(read[1].data) + first * in,
            CUDA_R_8I,
            int(in),
            codes,
            CUDA_R_8I,
            int(in),
            &zero,
            sums,
            CUDA_R_32I,
            int(tile),
            CUBLAS_COMPUTE_32I,
            CUBLAS_GEMM_DEFAULT
        ));
        h3_finish_int8_kernel<<<blocks(rows * tile), 256, 0, s>>>(
            reinterpret_cast<const std::int32_t*>(sums),
            row_scales,
            static_cast<const float*>(read[2].data),
            output,
            rows,
            tile,
            out,
            first,
            read.size() == 4 ? static_cast<const std::uint16_t*>(read[3].data)
                             : nullptr,
            status
        );
        checked(cudaGetLastError());
        if (!lora)
            continue;
        // The INT32 tile is consumed; reuse it for this tile's F64 LoRA delta.
        const double unit = 1, none = 0;
        checked(cublasDgemm(
            handle,
            CUBLAS_OP_T,
            CUBLAS_OP_N,
            int(tile),
            int(rows),
            int(shape.rank),
            &unit,
            lora_b + first * shape.rank,
            int(shape.rank),
            low,
            int(shape.rank),
            &none,
            reinterpret_cast<double*>(sums),
            int(tile)
        ));
        h3_lora_add_kernel<<<blocks(rows * tile), 256, 0, s>>>(
            reinterpret_cast<const double*>(sums),
            output,
            rows,
            tile,
            out,
            first,
            lora_scale,
            status
        );
        checked(cudaGetLastError());
    }
}

void CudaH3Launcher::lora(
    void* stream,
    const MediaDeviceSpan& input,
    const MediaDeviceSpan& a,
    const MediaDeviceSpan& b,
    const MediaDeviceSpan& low,
    const MediaDeviceSpan& weights,
    std::size_t rows,
    std::size_t in,
    std::size_t out,
    std::size_t rank,
    unsigned* status
) {
    const auto s = static_cast<cudaStream_t>(stream);
    auto* widened = static_cast<double*>(weights.data);
    h3_widen_kernel<<<blocks(rank * in), 256, 0, s>>>(
        static_cast<const std::uint16_t*>(a.data), widened, rank * in, status
    );
    checked(cudaGetLastError());
    h3_widen_kernel<<<blocks(out * rank), 256, 0, s>>>(
        static_cast<const std::uint16_t*>(b.data),
        widened + rank * in,
        out * rank,
        status
    );
    checked(cudaGetLastError());
    auto* wide_input = widened + rank * in + out * rank;
    widen<<<blocks(rows * in), 256, 0, s>>>(
        static_cast<const float*>(input.data), wide_input, rows * in
    );
    checked(cudaGetLastError());
    auto* result = static_cast<double*>(low.data);
    const double one = 1, zero = 0;
    checked(cublasDgemm(
        static_cast<cublasHandle_t>(blas_),
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        int(rank),
        int(rows),
        int(in),
        &one,
        widened,
        int(in),
        wide_input,
        int(in),
        &zero,
        result,
        int(rank)
    ));
    h3_zero_bias_kernel<<<blocks(rows * rank), 256, 0, s>>>(result, rows * rank);
    checked(cudaGetLastError());
}

// The host's precise dense projection over session buffers: F64 GEMM of the
// widened input and one widened weight column tile at a time, narrowed to F32
// with its bias, then the Turbo delta in the same order as convrot.
void CudaH3Launcher::dense(
    void* stream,
    std::span<const MediaDeviceSpan> read,
    std::span<const MediaDeviceSpan> write,
    float lora_scale
) {
    const auto shape = validate_h3_dense(device_, read, write);
    const bool lora = shape.rank != 0;
    bind(stream, write[write.size() - 2]);
    const auto s = static_cast<cudaStream_t>(stream);
    const auto handle = static_cast<cublasHandle_t>(blas_);
    auto* status = static_cast<unsigned*>(write.back().data);
    const auto rows = shape.rows, in = shape.in, out = shape.out;
    const auto* input = static_cast<const float*>(read[0].data);
    auto* output = static_cast<float*>(write[0].data);
    auto* wide_input = static_cast<double*>(write[1].data);
    auto* wide_weight = reinterpret_cast<double*>(
        static_cast<std::byte*>(write[1].data)
        + detail::h3_aligned(Bytes(rows) * in * 8)
    );
    auto* wide_output = wide_weight + shape.tile * in;
    const bool bf16 = read[1].dtype == DeviceDType::bf16;
    const auto* weight = static_cast<const std::byte*>(read[1].data);
    checked(cudaMemsetAsync(status, 0, sizeof(unsigned), s));
    h3_finite_kernel<<<blocks(rows * in), 256, 0, s>>>(input, rows * in, status);
    checked(cudaGetLastError());
    widen<<<blocks(rows * in), 256, 0, s>>>(input, wide_input, rows * in);
    checked(cudaGetLastError());
    const double* low = nullptr;
    const double* lora_b = nullptr;
    if (lora) {
        this->lora(
            stream,
            read[0],
            read[2],
            read[3],
            write[2],
            write[3],
            rows,
            in,
            out,
            shape.rank,
            status
        );
        low = static_cast<const double*>(write[2].data);
        lora_b = static_cast<const double*>(write[3].data) + shape.rank * in;
    }
    for (std::size_t first = 0; first < out; first += shape.tile) {
        const auto tile = std::min(shape.tile, out - first);
        h3_widen_real_kernel<<<blocks(tile * in), 256, 0, s>>>(
            weight + first * in * (bf16 ? 2 : 4), bf16, wide_weight, tile * in, status
        );
        checked(cudaGetLastError());
        const double one = 1, zero = 0;
        checked(cublasDgemm(
            handle,
            CUBLAS_OP_T,
            CUBLAS_OP_N,
            int(tile),
            int(rows),
            int(in),
            &one,
            wide_weight,
            int(in),
            wide_input,
            int(in),
            &zero,
            wide_output,
            int(tile)
        ));
        h3_narrow_kernel<<<blocks(rows * tile), 256, 0, s>>>(
            wide_output,
            shape.bias ? read[2].data : nullptr,
            shape.bias && read[2].dtype == DeviceDType::bf16,
            output,
            rows,
            tile,
            out,
            first,
            status
        );
        checked(cudaGetLastError());
        if (!lora)
            continue;
        // The narrowed tile is consumed; reuse its F64 buffer for the delta.
        checked(cublasDgemm(
            handle,
            CUBLAS_OP_T,
            CUBLAS_OP_N,
            int(tile),
            int(rows),
            int(shape.rank),
            &one,
            lora_b + first * shape.rank,
            int(shape.rank),
            low,
            int(shape.rank),
            &zero,
            wide_output,
            int(tile)
        ));
        h3_lora_add_kernel<<<blocks(rows * tile), 256, 0, s>>>(
            wide_output, output, rows, tile, out, first, lora_scale, status
        );
        checked(cudaGetLastError());
    }
}

void CudaH3Launcher::silu_gate(
    void* stream,
    std::span<const MediaDeviceSpan> read,
    std::span<const MediaDeviceSpan> write
) {
    const auto count = validate_h3_silu_gate(device_, read, write);
    own(stream);
    const auto s = static_cast<cudaStream_t>(stream);
    auto* status = static_cast<unsigned*>(write[1].data);
    checked(cudaMemsetAsync(status, 0, sizeof(unsigned), s));
    h3_silu_gate_kernel<<<blocks(count), 256, 0, s>>>(
        static_cast<float*>(write[0].data),
        static_cast<const float*>(read[1].data),
        count,
        status
    );
    checked(cudaGetLastError());
}

void CudaH3Launcher::gated_residual(
    void* stream,
    std::span<const MediaDeviceSpan> read,
    std::span<const MediaDeviceSpan> write,
    std::size_t first,
    std::size_t index
) {
    const auto count = validate_h3_gated_residual(device_, read, write, first, index);
    own(stream);
    const auto s = static_cast<cudaStream_t>(stream);
    auto* status = static_cast<unsigned*>(write[1].data);
    checked(cudaMemsetAsync(status, 0, sizeof(unsigned), s));
    h3_gated_residual_kernel<<<blocks(count), 256, 0, s>>>(
        static_cast<float*>(write[0].data),
        static_cast<const float*>(read[1].data),
        static_cast<const float*>(read[2].data),
        static_cast<const std::uint32_t*>(read[3].data),
        count,
        read[0].dimensions[1],
        read[2].dimensions[0] * h3_modulation_tags,
        first,
        index,
        status
    );
    checked(cudaGetLastError());
}

void CudaH3Launcher::normalize(
    void* stream,
    std::span<const MediaDeviceSpan> read,
    std::span<const MediaDeviceSpan> write,
    std::size_t first,
    std::size_t shift,
    std::size_t stride
) {
    const auto rows = validate_h3_normalize(device_, read, write, first, shift, stride);
    own(stream);
    const auto s = static_cast<cudaStream_t>(stream);
    auto* status = static_cast<unsigned*>(write[1].data);
    checked(cudaMemsetAsync(status, 0, sizeof(unsigned), s));
    const bool modulated = read.size() == 4;
    const auto width = read[0].dimensions[1];
    h3_normalize_kernel<<<unsigned(rows), 256, 0, s>>>(
        static_cast<const float*>(read[0].data),
        static_cast<const std::uint16_t*>(read[1].data),
        modulated ? static_cast<const float*>(read[2].data) : nullptr,
        modulated ? static_cast<const std::uint32_t*>(read[3].data) : nullptr,
        static_cast<float*>(write[0].data),
        width,
        modulated ? h3_modulation(device_, read[2], read[3], width, first, rows, stride)
                  : 0,
        first,
        shift,
        stride,
        status
    );
    checked(cudaGetLastError());
}

void CudaH3Launcher::add(
    void* stream,
    std::span<const MediaDeviceSpan> read,
    std::span<const MediaDeviceSpan> write
) {
    const auto count = validate_h3_add(device_, read, write);
    own(stream);
    const auto s = static_cast<cudaStream_t>(stream);
    auto* status = static_cast<unsigned*>(write[1].data);
    checked(cudaMemsetAsync(status, 0, sizeof(unsigned), s));
    h3_add_kernel<<<blocks(count), 256, 0, s>>>(
        static_cast<float*>(write[0].data),
        static_cast<const float*>(read[1].data),
        count,
        status
    );
    checked(cudaGetLastError());
}

void CudaH3Launcher::qk_rotary(
    void* stream,
    std::span<const MediaDeviceSpan> read,
    std::span<const MediaDeviceSpan> write,
    std::size_t first
) {
    validate_h3_qk_rotary(device_, read, write, first);
    own(stream);
    const auto s = static_cast<cudaStream_t>(stream);
    auto* status = static_cast<unsigned*>(write[1].data);
    checked(cudaMemsetAsync(status, 0, sizeof(unsigned), s));
    const auto rows = read[0].dimensions[0], width = read[0].dimensions[1];
    h3_qk_rotary_kernel<<<dim3(unsigned(rows), unsigned(width / 128)), 128, 0, s>>>(
        static_cast<float*>(write[0].data),
        static_cast<const std::uint16_t*>(read[1].data),
        read.size() == 3 ? static_cast<const double*>(read[2].data) : nullptr,
        width,
        first,
        status
    );
    checked(cudaGetLastError());
}

// attention_impl<double>'s tile loop over resident Q/K/V: per key tile, scaled
// QK^T into scores, online softmax rescaling the output, then accumulate PV.
// Heads are batched with stride 128 inside the [tokens, heads * 128] rows.
void CudaH3Launcher::attention(
    void* stream,
    std::span<const MediaDeviceSpan> read,
    std::span<const MediaDeviceSpan> write,
    std::size_t key_tile
) {
    const auto shape = validate_h3_attention(device_, read, write, key_tile);
    bind(stream, write[4]);
    const auto s = static_cast<cudaStream_t>(stream);
    const auto handle = static_cast<cublasHandle_t>(blas_);
    auto* status = static_cast<unsigned*>(write[5].data);
    const auto queries = shape.queries, heads = shape.heads, ld = heads * 128;
    const auto* query = static_cast<const float*>(read[0].data);
    const auto* key = static_cast<const float*>(read[1].data);
    const auto* value = static_cast<const float*>(read[2].data);
    auto* destination = static_cast<float*>(write[0].data);
    auto* scores = static_cast<double*>(write[1].data);
    auto* maxima = static_cast<double*>(write[2].data);
    auto* totals = maxima + heads * queries;
    auto* wide_query = static_cast<double*>(write[3].data);
    auto* output = wide_query + queries * ld;
    auto* wide_key = output + queries * ld;
    auto* wide_value = wide_key + key_tile * ld;
    checked(cudaMemsetAsync(status, 0, sizeof(unsigned), s));
    widen<<<blocks(queries * ld), 256, 0, s>>>(query, wide_query, queries * ld);
    checked(cudaGetLastError());
    h3_attention_init_kernel<<<dim3(unsigned(queries), unsigned(heads)), 128, 0, s>>>(
        query, output, ld, maxima, totals, unsigned(queries), status
    );
    checked(cudaGetLastError());
    const double scale = 1 / std::sqrt(128.0), zero = 0, one = 1;
    for (std::size_t first = 0; first < shape.keys; first += key_tile) {
        const auto keys = std::min(key_tile, shape.keys - first);
        widen<<<blocks(keys * ld), 256, 0, s>>>(key + first * ld, wide_key, keys * ld);
        checked(cudaGetLastError());
        widen<<<blocks(keys * ld), 256, 0, s>>>(
            value + first * ld, wide_value, keys * ld
        );
        checked(cudaGetLastError());
        checked(cublasDgemmStridedBatched(
            handle,
            CUBLAS_OP_T,
            CUBLAS_OP_N,
            int(keys),
            int(queries),
            128,
            &scale,
            wide_key,
            int(ld),
            128,
            wide_query,
            int(ld),
            128,
            &zero,
            scores,
            int(keys),
            (long long)(queries * keys),
            int(heads)
        ));
        h3_online_softmax_kernel<<<
            dim3(unsigned(queries), unsigned(heads)),
            256,
            0,
            s>>>(scores, output, ld, maxima, totals, unsigned(keys), unsigned(queries));
        checked(cudaGetLastError());
        checked(cublasDgemmStridedBatched(
            handle,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            128,
            int(queries),
            int(keys),
            &one,
            wide_value,
            int(ld),
            128,
            scores,
            int(keys),
            (long long)(queries * keys),
            &one,
            output,
            int(ld),
            128,
            int(heads)
        ));
    }
    h3_attention_divide_kernel<<<blocks(queries * heads * 128), 256, 0, s>>>(
        output, destination, ld, totals, unsigned(queries), unsigned(heads), status
    );
    checked(cudaGetLastError());
}

std::shared_ptr<H3Compute>
h3_cuda_compute(std::shared_ptr<Resources> r, std::vector<int> d) {
    return std::make_shared<Compute>(std::move(r), std::move(d));
}
}  // namespace kadan::video

namespace kadan {
std::shared_ptr<DenseCompute> cuda_dense_compute(
    std::shared_ptr<Resources> r, std::vector<int> devices, Workload workload
) {
    return std::make_shared<video::Compute>(
        std::move(r), std::move(devices), workload, true
    );
}
}  // namespace kadan
