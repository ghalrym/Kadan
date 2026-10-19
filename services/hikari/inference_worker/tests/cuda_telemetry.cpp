// CUDA boundaries are fakes. No driver, device, tensors, or inference runs.
#include "../src/cuda/cuda_telemetry.cuh"

#include <cassert>
#include <future>
#include <map>
#include <thread>
using namespace kadan;

namespace {
std::uintptr_t next_event = 0;
std::map<CUevent, bool> ready;
unsigned creates = 0, destroys = 0, copies = 0, waits = 0, launches = 0;
bool event_failure = false, destroy_failure = false;
cudaError_t call_result = cudaSuccess;
std::promise<void>* end_recorded = nullptr;
}  // namespace

extern "C" CUresult CUDAAPI cuCtxGetCurrent(CUcontext* context) {
    *context = reinterpret_cast<CUcontext>(1);
    return CUDA_SUCCESS;
}

extern "C" CUresult CUDAAPI cuCtxGetDevice(CUdevice* device) {
    *device = 0;
    return CUDA_SUCCESS;
}

extern "C" CUresult CUDAAPI cuEventCreate(CUevent* event, unsigned) {
    if (event_failure)
        return CUDA_ERROR_OUT_OF_MEMORY;
    *event = reinterpret_cast<CUevent>(++next_event);
    ready[*event] = false;
    ++creates;
    return CUDA_SUCCESS;
}

extern "C" CUresult CUDAAPI cuEventRecord(CUevent event, CUstream) {
    if (end_recorded && event == reinterpret_cast<CUevent>(next_event))
        end_recorded->set_value();
    return CUDA_SUCCESS;
}

extern "C" CUresult CUDAAPI cuEventQuery(CUevent event) {
    return ready.at(event) ? CUDA_SUCCESS : CUDA_ERROR_NOT_READY;
}

extern "C" CUresult CUDAAPI cuEventElapsedTime(float* value, CUevent, CUevent) {
    *value = 2;
    return CUDA_SUCCESS;
}

extern "C" CUresult CUDAAPI cuEventDestroy(CUevent event) {
    ++destroys;
    ready.erase(event);
    return destroy_failure ? CUDA_ERROR_INVALID_HANDLE : CUDA_SUCCESS;
}

extern "C" cudaError_t CUDARTAPI
cudaMemcpy(void*, const void*, std::size_t, cudaMemcpyKind) {
    ++copies;
    return call_result;
}

extern "C" cudaError_t CUDARTAPI cudaStreamSynchronize(cudaStream_t) {
    ++waits;
    if (call_result == cudaSuccess)
        for (auto& [event, completed] : ready)
            completed = true;
    return call_result;
}

extern "C" cudaError_t CUDARTAPI cudaSetDevice(int) {
    return call_result;
}

int main() {
    using serving::CudaKernel;
    auto launch = [] {
        cuda_trace::kernel(CudaKernel::gguf_matvec_kernel, cudaStreamLegacy, [] {
            ++launches;
        });
    };
    {
        serving::CudaPhaseScope disabled(false);
        launch();
        assert(creates == 0 && waits == 0 && launches == 1);
    }
    {
        serving::CudaPhaseScope phase(true);
        launch();
        assert(
            phase.snapshot()
                .kernels[std::size_t(CudaKernel::gguf_matvec_kernel)]
                .measured
            == 0
        );
        // A snapshot discards incomplete events, never waits for them.
        assert(phase.snapshot().dropped == 1 && waits == 0 && ready.empty());
    }
    {
        serving::CudaPhaseScope phase(true);
        for (std::size_t i = 0; i < cuda_trace::Events::capacity + 1; ++i)
            launch();
        assert(ready.size() == 2 * cuda_trace::Events::capacity && waits == 0);
        assert(cuda_trace::synchronize(cudaStreamLegacy) == cudaSuccess);
        const auto result = phase.snapshot();
        const auto kernel = result.kernels[std::size_t(CudaKernel::gguf_matvec_kernel)];
        assert(kernel.calls == 129 && kernel.measured == 128);
        assert(std::abs(kernel.gpu_seconds - .256) < 1e-9);
        assert(result.dropped == 1 && result.sync.calls == 1 && waits == 1);
        assert(ready.empty());
    }
    {
        serving::CudaPhaseScope phase(true);
        call_result = cudaErrorInvalidValue;
        assert(cuda_trace::copy(nullptr, nullptr, 0, cudaMemcpyDefault) == call_result);
        assert(cuda_trace::synchronize(cudaStreamLegacy) == call_result);
        const auto result = phase.snapshot();
        assert(result.copies[4].calls == 1 && result.sync.calls == 1);
        call_result = cudaSuccess;
        event_failure = true;
        launch();
        const auto failed = phase.snapshot();
        assert(failed.unavailable == 1);
        assert(failed.kernels[std::size_t(CudaKernel::gguf_matvec_kernel)].calls == 1);
        event_failure = false;
    }
    {
        serving::CudaPhaseScope phase(true);
        launch();
        assert(cuda_trace::set_device(1) == cudaSuccess);
        assert(phase.snapshot().dropped == 1 && ready.empty());
    }
    {
        // The GPU interval must end even while another device holds the
        // aggregate lock. Bookkeeping contention is not kernel duration.
        serving::CudaPhaseScope phase(true);
        std::promise<void> recorded;
        auto finished = recorded.get_future();
        end_recorded = &recorded;
        std::unique_lock lock(phase.context()->mutex);
        std::thread device([context = phase.context(), &launch] {
            serving::CudaPhaseScope tracing(context);
            launch();
        });
        const auto status = finished.wait_for(std::chrono::seconds(1));
        lock.unlock();
        device.join();
        end_recorded = nullptr;
        assert(status == std::future_status::ready);
    }
    {
        serving::CudaPhaseScope phase(true);
        launch();
        destroy_failure = true;
        phase.snapshot();
        const auto allocated = creates;
        launch();
        assert(creates == allocated && cuda_trace::events.disabled);
        assert(phase.snapshot().unavailable >= 1);
    }
    assert(creates == destroys && copies == 1 && waits == 2);
}
