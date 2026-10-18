#pragma once

#include <ATen/cuda/CUDAContextLight.h>
#include <ATen/cuda/MemPool.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/csrc/cuda/CUDAPluggableAllocator.h>

#include "kadan/image_pool_memory.hpp"
#include "kadan/resources.hpp"

namespace kadan::image::detail {
// Streams are model/device owned, so an idle weight bank and every stage using it
// share ordering without a default-stream dependency on another inference engine.
class ImageCudaStream {
public:
    explicit ImageCudaStream(int device) : device_(device) {
        c10::cuda::CUDAGuard guard(device_);
        C10_CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
    }

    ImageCudaStream(const ImageCudaStream&) = delete;
    ImageCudaStream& operator=(const ImageCudaStream&) = delete;

    ~ImageCudaStream() noexcept {
        try {
            close();
        } catch (...) { /* Explicit cleanup retains admission on failure. */
        }
    }

    void close() {
        if (!stream_)
            return;
        synchronize();
        c10::cuda::CUDAGuard guard(device_);
        at::cuda::clearCublasWorkspacesForStream(stream_);
        C10_CUDA_CHECK(cudaStreamDestroy(stream_));
        stream_ = nullptr;
    }

    c10::cuda::CUDAStream stream() const {
        return c10::cuda::getStreamFromExternal(stream_, device_);
    }

    void synchronize() const {
        stream().synchronize();
    }

private:
    int device_;
    cudaStream_t stream_ = nullptr;
};

// Weight banks and execution scratch have separate pools. A live private pool
// cannot return its cached segments, so scratch pools end with each stage while
// weight pools remain resident. No global allocator limit or cache is changed.
class ImageCudaPool {
public:
    ImageCudaPool(c10::cuda::CUDAStream stream, Bytes limit)
        : stream_(stream), memory_(std::make_shared<ImagePoolMemory>(limit)),
          id_(at::cuda::MemPool::graph_pool_handle()) {
        auto allocator = torch::cuda::CUDAPluggableAllocator::createCustomAllocator(
            [memory = memory_](std::size_t bytes, int device, cudaStream_t) -> void* {
                c10::cuda::CUDAGuard guard(device);
                if (!memory->acquire(bytes))
                    throw ResourceExhausted("image_aten_pool_capacity");
                void* pointer = nullptr;
                const auto error = cudaMalloc(&pointer, bytes);
                if (error != cudaSuccess) {
                    memory->release(bytes);
                    C10_CUDA_CHECK(error);
                }
                return pointer;
            },
            [memory =
                 memory_](void* pointer, std::size_t bytes, int device, cudaStream_t) {
                c10::cuda::CUDAGuard guard(device);
                C10_CUDA_CHECK(cudaFree(pointer));
                memory->release(bytes);
            }
        );
        c10::cuda::CUDACachingAllocator::createOrIncrefPool(
            stream_.device_index(), id_, allocator
        );
    }

    ImageCudaPool(const ImageCudaPool&) = delete;
    ImageCudaPool& operator=(const ImageCudaPool&) = delete;

    ~ImageCudaPool() noexcept {
        try {
            close();
        } catch (...) { /* Caller retains the resource reservation. */
        }
    }

    void begin() {
        c10::cuda::CUDACachingAllocator::beginAllocateToPool(
            stream_.device_index(),
            id_,
            [stream = stream_.stream()](cudaStream_t candidate) {
                return candidate == stream;
            }
        );
        routing_ = true;
    }

    void end() {
        if (!routing_)
            return;
        c10::cuda::CUDACachingAllocator::endAllocateToPool(stream_.device_index(), id_);
        routing_ = false;
        c10::cuda::CUDACachingAllocator::releasePool(stream_.device_index(), id_);
    }

    Bytes allocated() const {
        Bytes bytes = 0;
        for (const auto& segment :
             c10::cuda::CUDACachingAllocator::snapshot(id_, false).segments)
            if (segment.owner_private_pool_id == id_)
                bytes += segment.allocated_size;
        return bytes;
    }

    Bytes reserved() const {
        return memory_->current();
    }

    Bytes peak() const {
        return memory_->peak();
    }

    void close() {
        if (closed_)
            return;
        end();
        if (allocated())
            throw std::runtime_error("image_aten_pool_live_allocations");
        if (!released_) {
            c10::cuda::CUDACachingAllocator::releasePool(stream_.device_index(), id_);
            released_ = true;
        }
        c10::cuda::CUDACachingAllocator::emptyCache(id_);
        if (reserved())
            throw std::runtime_error("image_aten_pool_cleanup_unconfirmed");
        closed_ = true;
    }

private:
    c10::cuda::CUDAStream stream_;
    std::shared_ptr<ImagePoolMemory> memory_;
    c10::MempoolId_t id_;
    bool routing_ = false, released_ = false, closed_ = false;
};
}  // namespace kadan::image::detail
