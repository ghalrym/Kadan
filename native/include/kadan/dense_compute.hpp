#pragma once
#include "kadan/resources.hpp"
#include <atomic>
#include <memory>
#include <span>
#include <vector>
namespace kadan {
#ifdef __CUDACC__
__host__ __device__
#endif
inline bool attention_visible(std::size_t query,std::size_t key,std::size_t causal_queries,std::size_t window,std::size_t offset){
 return (query>=causal_queries||key<=query+offset)&&(!window||key+window>query+offset);
}
inline bool projection_split_columns(std::size_t rows,std::size_t devices){return rows<devices;}

// Synchronous projection boundary. Caller owns admitted host spans until return;
// implementations admit staging/device memory and synchronize before returning.
class DenseCompute {
public:
 virtual ~DenseCompute() = default;
 // Request boundary: release reusable scratch before reporting completion.
 virtual void release_scratch() {}
 // Dedicated worker only: synchronize and destroy its process-local contexts.
 virtual void release_devices() {}
 // Return false to retain the model's CPU attention. Q/K/V use [token,head,dim].
 // Only the first causal_queries query rows are causal. window=0 is unbounded;
 // query_offset addresses cached decoding where queries are a suffix of keys.
 virtual bool attend(std::span<const float>,std::span<const float>,std::span<const float>,
     std::size_t,std::size_t,std::size_t,std::size_t,std::size_t,std::size_t,std::size_t,std::size_t,
     std::span<float>,const std::atomic_bool&){return false;}

 virtual void dense(std::span<const float> weights,std::span<const float> bias,
     std::span<const float> input,std::size_t in,std::size_t out,
     std::span<float> output,const std::atomic_bool& cancel,bool precise=false)=0;
};
std::shared_ptr<DenseCompute> cuda_dense_compute(std::shared_ptr<Resources>,std::vector<int>,Workload);
}
