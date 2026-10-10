#pragma once
#include "kadan/dense_compute.hpp"
#include "kadan/device_placement.hpp"
#include <algorithm>
#include <charconv>
#include <cstdlib>
#include <string_view>
namespace kadan {
inline constexpr Bytes compute_context_bytes=384ULL*1024*1024;
inline Footprint host_footprint(Resources& r,Bytes bytes){auto f=r.snapshot().capacity;std::fill(f.begin(),f.end(),0);f[0]=bytes;return f;}
struct ComputePlan {Footprint capacity;std::vector<int> devices;};
inline ComputePlan compute_plan(Bytes host,std::string_view selection,std::string_view budget,bool available,std::string_view indexed={}){
 if(!host)throw std::invalid_argument("compute_host_budget");
 ComputePlan p{{host},{}};if(selection.empty())return p;
 if(!available)throw std::invalid_argument("compute_cuda_not_built");
 if(selection=="0")p.devices={0};else if(selection=="1")p.devices={1};else if(selection=="0,1")p.devices={0,1};else if(selection=="1,0")p.devices={1,0};else throw std::invalid_argument("compute_devices");
 auto budgets=device_budgets(p.devices,budget,indexed,compute_context_bytes+16*1024*1024);
 p.capacity.resize(3);for(int device:p.devices)p.capacity[device+1]=budgets[device+1];return p;
}
inline ComputePlan worker_compute_plan(Bytes host,const char* selection_name){
 const char* s=std::getenv(selection_name);const char* b=std::getenv("KADAN_NATIVE_GPU_BUDGET_BYTES");
#ifdef KADAN_WORKER_CUDA
 constexpr bool available=true;
#else
 constexpr bool available=false;
#endif
 const char* indexed=std::getenv("KADAN_NATIVE_GPU_BUDGETS");
 return compute_plan(host,s?s:"",b?b:"",available,indexed?indexed:"");
}
struct WorkerCompute {
 std::shared_ptr<Resources> resources;std::shared_ptr<DenseCompute> compute;Footprint context;Handle context_handle=0;Workload workload;
 WorkerCompute(const ComputePlan& plan,Workload kind):resources(std::make_shared<Resources>(plan.capacity)),context(plan.capacity.size()),workload(kind){
#ifdef KADAN_WORKER_CUDA
  if(!plan.devices.empty()){
   for(int d:plan.devices)context[d+1]=compute_context_bytes;
   // Keep the matching parent reservation until acknowledged park destroys
   // these contexts, or the child has been reaped during cancellation.
   context_handle=resources->reserve(workload,context);
   compute=cuda_dense_compute(resources,plan.devices,workload);
  }
#else
  (void)workload;if(!plan.devices.empty())throw std::invalid_argument("compute_cuda_not_built");
#endif
 }
 void idle() const {if(compute)compute->release_scratch();const auto retained=compute?compute->retained_weights():Footprint{};const auto used=resources->snapshot().used;for(std::size_t i=1;i<used.size();++i)if(used[i]!=(context_handle?context[i]:0)+(retained.empty()?0:retained.at(i)))throw std::runtime_error("compute_device_cleanup_unconfirmed");}
 void park(){idle();if(context_handle){compute->release_devices();resources->released(context_handle);context_handle=0;}}
 void resume(){if(compute&&!context_handle)context_handle=resources->reserve(workload,context);}
};
}
