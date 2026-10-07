#ifdef CUDA_API_PER_THREAD_DEFAULT_STREAM
#error "The first Kadan projection path requires legacy default-stream semantics"
#endif
#include "kadan/cuda_projection.hpp"
#include "nvfp4_kernel.cuh"
#include "fp8_kernel.cuh"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <thread>

namespace kadan::cuda {
namespace {
void check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}
struct Nvfp4Operations {
    using Plan = Nvfp4Plan;
    static Plan plan(const quantization::Matrix& host, std::size_t budget) { return plan_nvfp4(host, budget); }
    static void upload_scales(void* storage, const Plan& layout, const quantization::Matrix& host) {
        auto* bytes = static_cast<std::uint8_t*>(storage);
        check(cudaMemcpy(bytes + layout.block_offset, host.block_scales.data(), layout.block_bytes,
                         cudaMemcpyHostToDevice), "upload_scales");
    }
    static cudaError_t launch(void* storage, const Plan& layout, float global,
                              const float* input, float* output, unsigned* status) {
        auto* bytes = static_cast<std::uint8_t*>(storage);
        return kadan_launch_nvfp4(bytes, bytes + layout.block_offset, global,
                                  input, output, status, layout.rows, layout.columns);
    }
};
struct Fp8Operations {
    using Plan = Fp8Plan;
    static Plan plan(const quantization::Matrix& host, std::size_t budget) { return plan_fp8(host, budget); }
    static void upload_scales(void* storage, const Plan& layout, const quantization::Matrix& host) {
        auto* bytes = static_cast<std::uint8_t*>(storage);
        check(cudaMemcpy(bytes + layout.scale_offset, host.multipliers.data(), layout.scale_bytes,
                         cudaMemcpyHostToDevice), "upload_scales");
    }
    static cudaError_t launch(void* storage, const Plan& layout, float,
                              const float* input, float* output, unsigned* status) {
        auto* bytes = static_cast<std::uint8_t*>(storage);
        return kadan_launch_fp8(bytes, reinterpret_cast<const float*>(bytes + layout.scale_offset),
                               layout.scale_count != 1, input, output, status, layout.rows, layout.columns);
    }
};
// One resource/runtime owner for both encodings. Operations only choose layout,
// immutable scale upload and kernel; pinning/error/quarantine semantics are shared.
template<class Operations>
struct ProjectionImpl {
    std::shared_ptr<Resources> resources;
    typename Operations::Plan layout{};
    std::thread::id owner = std::this_thread::get_id();
    int device;
    Handle handle = 0;
    void* storage = nullptr;
    float global = 0;
    bool loaded = false, pinned = false, poisoned = false, cleanup_failed = false;

    ProjectionImpl(const quantization::Matrix& host, int ordinal, std::shared_ptr<Resources> manager)
        : resources(std::move(manager)), device(ordinal) {
        if (!resources || device < 0) throw std::invalid_argument("invalid_cuda_resource_owner");
        auto request = resources->snapshot().capacity;
        if (static_cast<std::size_t>(device) + 1 >= request.size()) throw std::invalid_argument("unbudgeted_cuda_device");
        layout = Operations::plan(host, request[device + 1]);
        global = host.multipliers[0];
        std::fill(request.begin(), request.end(), 0); request[device + 1] = layout.device_bytes;
        handle = resources->reserve(Workload::llm, std::move(request));
        try {
            current_device();
            int major = 0, minor = 0;
            check(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device), "cudaDeviceGetAttribute");
            check(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device), "cudaDeviceGetAttribute");
            if (major != 8 || minor != 6) throw std::invalid_argument("requires_sm86");
            void* allocation = nullptr;
            check(cudaMalloc(&allocation, layout.device_bytes), "cudaMalloc");
            storage = allocation;
            check(cudaMemcpy(storage, host.weights.data(), layout.weight_bytes, cudaMemcpyHostToDevice), "upload_weights");
            Operations::upload_scales(storage, layout, host);
            check(cudaStreamSynchronize(cudaStreamLegacy), "upload_synchronize");
            resources->loaded(handle); loaded = true;
        } catch (...) { cleanup(); throw; }
    }
    std::uint8_t* bytes(std::size_t offset) const { return static_cast<std::uint8_t*>(storage) + offset; }
    float* floats(std::size_t offset) const { return reinterpret_cast<float*>(bytes(offset)); }
    unsigned* status() const { return reinterpret_cast<unsigned*>(bytes(layout.status_offset)); }
    void current_device() const {
        if (std::this_thread::get_id() != owner) throw std::logic_error("cuda_projection_thread_changed");
        int current = -1; check(cudaGetDevice(&current), "cudaGetDevice");
        if (current != device) throw std::logic_error("cuda_projection_device_changed");
    }
    // Conservative quarantine: any failed synchronization/free leaves the ledger
    // charged and prohibits another cleanup attempt on an uncertain pointer.
    bool cleanup() noexcept {
        if (handle == 0) return true;
        if (cleanup_failed) return false;
        try {
            if (storage) {
                current_device();
                check(cudaStreamSynchronize(cudaStreamLegacy), "close_synchronize");
            }
            if (pinned) { resources->unpin(handle); pinned = false; }
            if (loaded) { resources->begin_eviction(handle); loaded = false; }
            if (storage) { check(cudaFree(storage), "cudaFree"); storage = nullptr; }
            resources->released(handle); handle = 0;
            return true;
        } catch (...) { cleanup_failed = true; poisoned = true; return false; }
    }
    void matvec(std::span<const float> input, std::span<float> output) {
        if (handle == 0 || poisoned) throw std::logic_error("cuda_projection_unavailable");
        if (input.size() != layout.columns || output.size() != layout.rows) throw std::invalid_argument("cuda_input_output_shape");
        for (float value : input) if (!std::isfinite(value)) throw std::invalid_argument("nonfinite_input");
        current_device();
        resources->pin(handle); pinned = true;
        unsigned flags = 0;
        try {
            // Copies and the fused launch use the same explicit legacy stream.
            // No caller host buffers are referenced by the kernel itself.
            check(cudaMemcpy(floats(layout.input_offset), input.data(), input.size_bytes(), cudaMemcpyHostToDevice), "upload_input");
            check(cudaMemsetAsync(status(), 0, sizeof(unsigned), cudaStreamLegacy), "clear_status");
            check(Operations::launch(storage, layout, global,
                floats(layout.input_offset), floats(layout.output_offset), status()), "launch_projection");
            check(cudaStreamSynchronize(cudaStreamLegacy), "matvec_synchronize");
            check(cudaMemcpy(&flags, status(), sizeof(flags), cudaMemcpyDeviceToHost), "read_status");
            if (flags == 0)
                check(cudaMemcpy(output.data(), floats(layout.output_offset), output.size_bytes(), cudaMemcpyDeviceToHost), "download_output");
            resources->unpin(handle); pinned = false;
        } catch (...) {
            // Best-effort quiescence before returning from a failed runtime call.
            // Keep the pin/reservation until explicit cleanup establishes success.
            cudaStreamSynchronize(cudaStreamLegacy);
            poisoned = true;
            throw;
        }
        if (flags != 0) throw std::overflow_error("nonfinite_cuda_projection");
    }
};
} // namespace
struct Nvfp4Projection::Impl : ProjectionImpl<Nvfp4Operations> {
    using ProjectionImpl::ProjectionImpl;
};
struct Fp8Projection::Impl : ProjectionImpl<Fp8Operations> {
    using ProjectionImpl::ProjectionImpl;
};
Nvfp4Projection::Nvfp4Projection(const quantization::Matrix& host, int device, std::shared_ptr<Resources> resources)
    : impl_(std::make_unique<Impl>(host, device, std::move(resources))) {}
Nvfp4Projection::~Nvfp4Projection() { impl_->cleanup(); }
const Nvfp4Plan& Nvfp4Projection::plan() const { return impl_->layout; }
void Nvfp4Projection::matvec(std::span<const float> input, std::span<float> output) { impl_->matvec(input, output); }
void Nvfp4Projection::close() {
    if (!impl_->cleanup()) throw std::runtime_error("cuda_cleanup_failed_reservation_retained");
}
Fp8Projection::Fp8Projection(const quantization::Matrix& host, int device, std::shared_ptr<Resources> resources)
    : impl_(std::make_unique<Impl>(host, device, std::move(resources))) {}
Fp8Projection::~Fp8Projection() { impl_->cleanup(); }
const Fp8Plan& Fp8Projection::plan() const { return impl_->layout; }
void Fp8Projection::matvec(std::span<const float> input, std::span<float> output) { impl_->matvec(input, output); }
void Fp8Projection::close() {
    if (!impl_->cleanup()) throw std::runtime_error("cuda_cleanup_failed_reservation_retained");
}
} // namespace kadan::cuda
