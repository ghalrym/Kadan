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

#include "image_activation.hpp"
#include "kadan/bf16_projection.hpp"
#include "kadan/device_placement.hpp"
#include "kadan/device_weights.hpp"
#include "kadan/device_workspace.hpp"
#include "kadan/h3_compute.hpp"

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
        checked(cudaDeviceReset());
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
    Real m = -INFINITY;
    for (unsigned k = t; k < keys; k += 256) {
        Real v = scores[q * keys + k];
        if (!attention_visible(
                first_query + q, first_key + k, causal_queries, window, offset
            ))
            v = -INFINITY;
        scores[q * keys + k] = v;
        m = fmax(m, v);
    }
    reduction[t] = m;
    __syncthreads();
    for (unsigned stride = 128; stride; stride /= 2) {
        if (t < stride)
            reduction[t] = fmax(reduction[t], reduction[t + stride]);
        __syncthreads();
    }
    const Real next = fmax(maxima[q], reduction[0]);
    const Real alpha = isfinite(maxima[q]) ? exp(maxima[q] - next) : 0;
    Real total = 0;
    // All warps must consume the maximum before shared storage becomes sums.
    __syncthreads();
    for (unsigned k = t; k < keys; k += 256) {
        const Real v = scores[q * keys + k];
        const Real p = isfinite(v) ? exp(v - next) : 0;
        scores[q * keys + k] = p;
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
        output[q * dim + c] *= alpha;
    if (t == 0) {
        maxima[q] = next;
        totals[q] = totals[q] * alpha + reduction[0];
    }
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
        ProjectionPolicy policy = ProjectionPolicy::standard
    ) {
        require(!cancel.load(), "h3_cuda_cancelled");
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
            (source && identity) || weights.size() == product(in, out),
            "h3_cuda_weight_shape"
        );
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
        const bool columns =
            identity
            || (column_split_ && !quant
                && projection_split_columns(x.size() / in, devices_.size()));
        const auto all_weights = weights;
        const auto all_bias = bias;
        const auto all_scales = scales;
        std::mutex source_mutex;
        split(
            columns ? (quant ? out / 4 : out) : x.size() / in,
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
                    end = x.size() / in;
                }
                auto& arena =
                    static_cast<CudaDeviceOperations&>(device_workspace.operations());
                // At most 16MiB of source staging per active device, independent of
                // free VRAM. Plan for that actual chunk, leaving room for inputs.
                constexpr std::size_t source_bytes = 16 * 1024 * 1024;
                const auto alignment = quant ? 4u : 1u;
                const auto host_columns =
                    (source_bytes / (in * sizeof(Weight))) / alignment * alignment;
                require(!source || host_columns > 0, "h3_source_staging_floor");
                const auto plan = projection_plan(
                    in,
                    source ? std::min(last_output - first_output, host_columns)
                           : last_output - first_output,
                    sizeof(Weight),
                    precise,
                    alignment,
                    arena.scratch_limit
                );
                const auto chunk_columns = plan.columns;
                const auto input_rows =
                    projection_input_rows(end - begin, in, plan, arena.scratch_limit);
                const bool keep_input = input_rows >= end - begin;
                // Reserve the largest layout before uploading. Later chunks must
                // neither move the input at its fixed offset nor grow the workspace.
                device_workspace.ensure(
                    plan.bytes + (input_rows - plan.rows) * in * sizeof(float)
                );
                // The aligned status prefix stays fixed across every column chunk.
                auto* nonfinite = validate_output ? arena.at<unsigned>(0) : nullptr;
                if (nonfinite)
                    checked(cudaMemset(nonfinite, 0, sizeof(unsigned)));
                if (keep_input)
                    for (auto start = begin; start < end; start += plan.rows) {
                        stop(cancel, failed);
                        const auto rows = std::min(plan.rows, end - start);
                        checked(cudaMemcpy(
                            arena.at<float>(projection_status_bytes)
                                + (start - begin) * in,
                            x.data() + start * in,
                            rows * in * sizeof(float),
                            cudaMemcpyHostToDevice
                        ));
                    }
                for (auto first_column = first_output; first_column < last_output;
                     first_column += chunk_columns) {
                    const auto out =
                        std::min(chunk_columns, last_output - first_column);
                    auto weights =
                        source ? std::span<const Weight>{}
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
                        first_column * in,
                        weight_count,
                        sizeof(Weight)
                    };
                    void* retained = identity ? arena.weights.find(key) : nullptr;
                    std::unique_ptr<HostAdmission> host_admission;
                    std::unique_ptr<Weight[]> host_weights;
                    if (!retained) {
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
                            for (float value : weights)
                                require(
                                    std::isfinite(value), "h3_cuda_nonfinite_weight"
                                );
                        if (identity)
                            retained = arena.weights.retain(
                                key, std::as_bytes(weights), cancel
                            );
                    }
                    Layout l;
                    l.add(sizeof(unsigned));
                    const auto input = l.add(input_rows * in * sizeof(float)),
                               w = l.add(retained ? 0 : weights.size_bytes()),
                               b = l.add(out * 4), s = l.add(out * 4),
                               rotated = l.add(tile * in * 4), codes = l.add(tile * in),
                               row_scales = l.add(tile * 4),
                               sums = l.add(tile * out * 4),
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
                    auto* weight_pointer = retained ? retained : arena.at<void>(w);
                    if (!retained)
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
                            checked(cudaMemcpy(
                                input_pointer,
                                x.data() + start * in,
                                rows * in * sizeof(float),
                                cudaMemcpyHostToDevice
                            ));
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
                        checked(cudaStreamSynchronize(cudaStreamLegacy));
                        stop(cancel, failed);
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
    int device
)
    : context_owner_(std::move(context_owner)), activity_(std::move(activity)),
      resources_(std::move(resources)), device_(device) {
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
