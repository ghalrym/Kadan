#pragma once
#include "kadan/resources.hpp"
#include "kadan/weight_identity.hpp"
#include <atomic>
#include <functional>
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

struct DeviceWeightPlacement {int device;Bytes planned_bytes,allocated_bytes;};

// Synchronous projection boundary. Caller owns admitted host spans until return;
// implementations admit staging/device memory and synchronize before returning.
class DenseCompute {
public:
 virtual ~DenseCompute() = default;
 // Request boundary: release reusable scratch before reporting completion.
 virtual void release_scratch() {}
 // Dedicated worker only: synchronize and destroy its process-local contexts.
 virtual void release_devices() {}
 virtual Footprint retained_weights() const {return {};}
 // Weight-bank limits and live allocation bytes, excluding context and scratch.
 virtual std::vector<DeviceWeightPlacement> weight_placement() const {return {};}
 // Header-derived execution bytes, before any weight placement. True means the
 // conservative full inventory fits; false selects bounded RAM-backed streaming.
 virtual bool prepare_weights(Bytes) {return false;}
 // Stable serialized identity opts immutable weights into bounded retention.
 virtual void dense_weight(const WeightIdentity&,std::span<const float> weights,std::span<const float> bias,
     std::span<const float> input,std::size_t in,std::size_t out,
     std::span<float> output,const std::atomic_bool& cancel,bool precise=false){dense(weights,bias,input,in,out,output,cancel,precise);}
 // The callback reads only a missing partition; it must preserve exact F32 values.
 virtual bool supports_weight_sources() const {return false;}
 using FloatWeightSource=std::function<void(std::size_t,std::span<float>)>;
 virtual bool dense_source(const WeightIdentity&,const FloatWeightSource&,std::span<const float>,
     std::span<const float>,std::size_t,std::size_t,std::span<float>,const std::atomic_bool&,bool=false){return false;}
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
