#pragma once
#include <array>
#include <chrono>
#include <cmath>
#include <cuda.h>

#include <cuda_runtime_api.h>

#include "kadan/cuda_phase_metrics.hpp"

namespace kadan::cuda_trace {
using Clock = std::chrono::steady_clock;

inline double seconds(Clock::time_point start) noexcept {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

// All event operations use the driver API on the existing current context.
// They neither consume nor overwrite CUDART's per-thread last-error slot.
struct Pending {
    CUevent start = nullptr, end = nullptr;
    std::size_t kernel = 0;
};

struct Events {
    static constexpr std::size_t capacity = 128;
    std::array<Pending, capacity> pending{};
    std::size_t size = 0;
    std::shared_ptr<serving::CudaPhase> owner;
    bool disabled = false;
};

inline thread_local Events events;

template <class F>
inline void
update(const std::shared_ptr<serving::CudaPhase>& phase, F function) noexcept {
    if (!phase)
        return;
    try {
        std::lock_guard lock(phase->mutex);
        phase->metrics.observed = true;
        function(phase->metrics);
    } catch (...) {
        // Export and diagnostic bookkeeping must not fail inference.
    }
}

inline void destroy(Pending pair) noexcept {
    for (auto event : {pair.start, pair.end})
        if (event && cuEventDestroy(event) != CUDA_SUCCESS) {
            // Do not accumulate unbounded resources after an unconfirmed cleanup.
            events.disabled = true;
            update(events.owner, [](auto& m) { ++m.unavailable; });
        }
}

inline void collect(bool terminal) noexcept {
    std::size_t retained = 0;
    for (std::size_t i = 0; i < events.size; ++i) {
        const auto pair = events.pending[i];
        const auto status = cuEventQuery(pair.end);
        if (status == CUDA_ERROR_NOT_READY && !terminal) {
            events.pending[retained++] = pair;
            continue;
        }
        if (status == CUDA_SUCCESS) {
            float elapsed = 0;
            if (cuEventElapsedTime(&elapsed, pair.start, pair.end) == CUDA_SUCCESS
                && std::isfinite(elapsed) && elapsed >= 0)
                update(events.owner, [&](auto& m) {
                    ++m.kernels[pair.kernel].measured;
                    m.kernels[pair.kernel].gpu_seconds += double(elapsed) / 1000;
                });
            else
                update(events.owner, [](auto& m) { ++m.unavailable; });
        } else if (status == CUDA_ERROR_NOT_READY)
            update(events.owner, [](auto& m) { ++m.dropped; });
        else
            update(events.owner, [](auto& m) { ++m.unavailable; });
        destroy(pair);
    }
    events.size = retained;
}

inline void finalize() noexcept {
    collect(true);
    events.owner.reset();
}

inline bool prepare() noexcept {
    if (events.owner != serving::cuda_phase) {
        finalize();
        events.owner = serving::cuda_phase;
    }
    serving::cuda_finalize = finalize;
    return bool(events.owner);
}

template <class F>
inline void kernel(serving::CudaKernel kernel, cudaStream_t stream, F launch) {
    if (!prepare()) {
        launch();
        return;
    }
    const auto index = std::size_t(kernel);
    Pending pair{nullptr, nullptr, index};
    bool timed = false;
    if (events.size == events.capacity)
        collect(false);
    if (events.size == events.capacity)
        update(events.owner, [](auto& m) { ++m.dropped; });
    else if (events.disabled)
        update(events.owner, [](auto& m) { ++m.unavailable; });
    else {
        CUcontext context = nullptr;
        timed = cuCtxGetCurrent(&context) == CUDA_SUCCESS && context
                && cuEventCreate(&pair.start, CU_EVENT_DEFAULT) == CUDA_SUCCESS
                && cuEventCreate(&pair.end, CU_EVENT_DEFAULT) == CUDA_SUCCESS
                && cuEventRecord(pair.start, reinterpret_cast<CUstream>(stream))
                       == CUDA_SUCCESS;
        if (!timed) {
            update(events.owner, [](auto& m) { ++m.unavailable; });
            destroy(pair);
            pair = {};
        }
    }
    const auto start = Clock::now();
    launch();
    const auto host = seconds(start);
    // End the GPU interval before taking a shared bookkeeping lock. Otherwise
    // contention between devices would be charged as kernel execution time.
    const auto recorded =
        timed ? cuEventRecord(pair.end, reinterpret_cast<CUstream>(stream))
              : CUDA_SUCCESS;
    update(events.owner, [&](auto& m) {
        ++m.kernels[index].calls;
        m.kernels[index].host_seconds += host;
    });
    if (timed) {
        if (recorded == CUDA_SUCCESS)
            events.pending[events.size++] = pair;
        else {
            update(events.owner, [](auto& m) { ++m.unavailable; });
            destroy(pair);
        }
    }
}

inline cudaError_t
copy(void* destination, const void* source, std::size_t bytes, cudaMemcpyKind kind) {
    if (!prepare())
        return cudaMemcpy(destination, source, bytes, kind);
    const auto start = Clock::now();
    const auto result = cudaMemcpy(destination, source, bytes, kind);
    const auto host = seconds(start);
    update(events.owner, [&](auto& m) {
        const auto index = static_cast<unsigned>(kind);
        if (index < m.copies.size()) {
            ++m.copies[index].calls;
            m.copies[index].host_seconds += host;
        }
    });
    // Query only. A copy's return is not proof that unrelated streams completed.
    collect(false);
    return result;
}

inline cudaError_t synchronize(cudaStream_t stream) {
    if (!prepare())
        return cudaStreamSynchronize(stream);
    const auto start = Clock::now();
    const auto result = cudaStreamSynchronize(stream);
    const auto host = seconds(start);
    update(events.owner, [&](auto& m) {
        ++m.sync.calls;
        m.sync.host_seconds += host;
    });
    collect(false);
    return result;
}

inline cudaError_t set_device(int device) {
    // Drain or discard handles on their original context before an existing
    // device switch. Never switch contexts or add a wait just for telemetry.
    if (events.size) {
        CUdevice current = -1;
        if (cuCtxGetDevice(&current) != CUDA_SUCCESS || current != device)
            finalize();
    }
    return cudaSetDevice(device);
}
}  // namespace kadan::cuda_trace
