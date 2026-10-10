#pragma once
#include "kadan/dense_compute.hpp"
#include <atomic>
#include <memory>
#include <span>
#include <vector>
namespace kadan::video {
// Synchronous host-span boundary. Caller admits all host tensors through return.
// Implementations admit their own staging/device allocations before allocation.
// Each method finishes all devices before returning (including cancellation).
class H3Compute : public DenseCompute {
public:
    virtual ~H3Compute() = default;
    virtual void attention(std::span<const float> query,std::span<const float> key,
        std::span<const float> value,std::size_t tokens,std::size_t heads,std::size_t kv_heads,
        std::size_t dimension,bool causal,std::span<float> output,const std::atomic_bool& cancel,bool precise=false)=0;
    virtual void dense(std::span<const float> weights,std::span<const float> bias,
        std::span<const float> input,std::size_t in,std::size_t out,
        std::span<float> output,const std::atomic_bool& cancel,bool precise=false)=0;
    virtual void convrot(std::span<const std::uint8_t> weights,std::span<const float> scales,
        std::span<const float> bias,std::span<const float> input,std::size_t in,std::size_t out,
        std::size_t group,std::span<float> output,const std::atomic_bool& cancel)=0;
    virtual void convrot_weight(const WeightIdentity&,std::span<const std::uint8_t> weights,std::span<const float> scales,
        std::span<const float> bias,std::span<const float> input,std::size_t in,std::size_t out,
        std::size_t group,std::span<float> output,const std::atomic_bool& cancel){convrot(weights,scales,bias,input,in,out,group,output,cancel);}
};
// Optional CUDA implementation, built only with KADAN_ENABLE_CUDA. The caller
// owns CUDA context residency in its process/queue budget. Only explicit
// release_devices resets its dedicated worker contexts after scratch cleanup;
// callers must not share those contexts with another executor. Empty, duplicate
// and unbudgeted devices are rejected. Work is split by activation rows; the checkpoint is unchanged.
std::shared_ptr<H3Compute> h3_cuda_compute(std::shared_ptr<Resources>,std::vector<int> devices);
}
