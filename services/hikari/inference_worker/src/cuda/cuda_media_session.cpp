#include <map>

#include <cuda_runtime_api.h>

#include "kadan/media_session.hpp"

namespace kadan {
namespace {
void checked(cudaError_t error) {
    if (error != cudaSuccess)
        throw std::runtime_error(
            std::string("media_cuda: ") + cudaGetErrorString(error)
        );
}

class CudaMediaOperations final : public MediaDeviceOperations {
public:
    CudaMediaOperations(
        std::shared_ptr<DenseCompute> context_owner, std::span<const int> devices
    )
        : context_owner_(std::move(context_owner)) {
        if (!context_owner_ || devices.empty())
            throw std::invalid_argument("media_cuda_context_owner");
        try {
            for (int device : devices) {
                auto [it, inserted] = streams_.emplace(device, nullptr);
                if (!inserted)
                    throw std::invalid_argument("media_cuda_duplicate_device");
                checked(cudaSetDevice(device));
                checked(cudaStreamCreateWithFlags(&it->second, cudaStreamNonBlocking));
            }
        } catch (...) {
            destroy_streams();
            throw;
        }
    }

    ~CudaMediaOperations() override {
        destroy_streams();
    }

    std::size_t allocation_alignment() const override {
        return 256;
    }

    void* stream(int device) override {
        return selected(device);
    }

    Bytes free_memory(int device) override {
        selected(device);
        std::size_t free = 0, total = 0;
        checked(cudaMemGetInfo(&free, &total));
        return free;
    }

    bool direct_peer(int from, int to) override {
        if (!streams_.contains(from) || !streams_.contains(to))
            throw std::invalid_argument("media_cuda_peer_device");
        int direct = 0;
        checked(cudaDeviceCanAccessPeer(&direct, from, to));
        return direct != 0;
    }

    void* allocate(int device, Bytes bytes, std::size_t alignment) override {
        selected(device);
        if (alignment > allocation_alignment()
            || bytes > std::numeric_limits<std::size_t>::max())
            throw std::invalid_argument("media_cuda_allocation_layout");
        void* pointer = nullptr;
        checked(cudaMalloc(&pointer, bytes));
        return pointer;
    }

    void free(int device, void* pointer) override {
        selected(device);
        checked(cudaFree(pointer));
    }

    void wait(int device, MediaEvent event) override {
        const auto it = events_.find(event.id);
        if (it == events_.end() || it->second.device != event.device)
            throw std::invalid_argument("media_cuda_stale_event");
        checked(cudaStreamWaitEvent(selected(device), it->second.event, 0));
    }

    MediaEvent record(int device) override {
        const auto owned_stream = selected(device);
        if (next_ == std::numeric_limits<std::uint64_t>::max())
            throw std::overflow_error("media_cuda_events");
        // Own metadata before creating/recording the event: failed recording
        // leaves a handle in this table for confirmed cleanup.
        const auto id = next_++;
        auto& entry = events_.try_emplace(id, Event{device, nullptr}).first->second;
        checked(cudaEventCreateWithFlags(&entry.event, cudaEventDisableTiming));
        checked(cudaEventRecord(entry.event, owned_stream));
        return {device, id};
    }

    void synchronize(int device) override {
        checked(cudaStreamSynchronize(selected(device)));
    }

    void release_events(int device) override {
        selected(device);
        for (auto it = events_.begin(); it != events_.end();) {
            if (it->second.device != device) {
                ++it;
                continue;
            }
            if (it->second.event)
                checked(cudaEventDestroy(it->second.event));
            it = events_.erase(it);
        }
    }

    void
    upload(int device, void* destination, std::span<const std::byte> source) override {
        checked(cudaMemcpyAsync(
            destination,
            source.data(),
            source.size(),
            cudaMemcpyHostToDevice,
            selected(device)
        ));
    }

    void download(
        int device, std::span<std::byte> destination, const void* source
    ) override {
        checked(cudaMemcpyAsync(
            destination.data(),
            source,
            destination.size(),
            cudaMemcpyDeviceToHost,
            selected(device)
        ));
    }

    void peer(
        int from, const void* source, int to, void* destination, Bytes bytes
    ) override {
        if (from == to) {
            checked(cudaMemcpyAsync(
                destination, source, bytes, cudaMemcpyDeviceToDevice, selected(to)
            ));
            return;
        }
        // Enable only a previously probed direct route. Never allow CUDA's
        // implicit host-staging route to count as resident peer traffic.
        if (!direct_peer(from, to) || !direct_peer(to, from))
            throw std::runtime_error("media_cuda_direct_peer_required");
        enable_peer(from, to);
        enable_peer(to, from);
        checked(
            cudaMemcpyPeerAsync(destination, to, source, from, bytes, selected(to))
        );
    }

    std::uint32_t read_status(int device, const void* source) override {
        std::uint32_t status = 0;
        checked(cudaMemcpyAsync(
            &status, source, sizeof(status), cudaMemcpyDeviceToHost, selected(device)
        ));
        synchronize(device);
        return status;
    }

private:
    struct Event {
        int device;
        cudaEvent_t event;
    };

    std::shared_ptr<DenseCompute> context_owner_;
    std::map<int, cudaStream_t> streams_;
    std::map<std::uint64_t, Event> events_;
    std::uint64_t next_ = 1;

    cudaStream_t selected(int device) {
        const auto it = streams_.find(device);
        if (it == streams_.end())
            throw std::invalid_argument("media_cuda_device");
        checked(cudaSetDevice(device));
        return it->second;
    }

    void enable_peer(int from, int to) {
        selected(from);
        const auto error = cudaDeviceEnablePeerAccess(to, 0);
        if (error == cudaErrorPeerAccessAlreadyEnabled) {
            cudaGetLastError();
            return;
        }
        checked(error);
    }

    void destroy_streams() noexcept {
        // Physical allocations are session-owned. Failed cleanup does not
        // acknowledge their release or reset a context shared by another request.
        for (const auto& [device, stream] : streams_)
            if (stream && cudaSetDevice(device) == cudaSuccess)
                cudaStreamDestroy(stream);
    }
};
}  // namespace

std::unique_ptr<MediaDeviceOperations> cuda_media_device_operations(
    std::shared_ptr<DenseCompute> context_owner, std::span<const int> devices
) {
    return std::make_unique<CudaMediaOperations>(std::move(context_owner), devices);
}
}  // namespace kadan
