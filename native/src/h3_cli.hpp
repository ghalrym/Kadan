#pragma once
#include "kadan/h3_compute.hpp"
#include "kadan/device_placement.hpp"
#include <cstdlib>
#include <charconv>
#include <algorithm>
#include <string_view>
namespace kadan::video {
inline Footprint h3_host(Resources& r,Bytes bytes){auto f=r.snapshot().capacity;std::fill(f.begin(),f.end(),0);f[0]=bytes;return f;}
struct H3Execution {
    std::shared_ptr<Resources> resources;
    std::shared_ptr<H3Compute> compute;
};
// Explicit development/worker device selection; absence retains the CPU oracle.
// The parent process reservation must additionally retain the CUDA context
// allowance until the worker exits. This factory never evicts another process.
inline H3Execution h3_execution(Bytes cpu_host_bytes,Bytes cache_bytes=0){
    if(cache_bytes>32ULL*1024*1024*1024)throw std::invalid_argument("h3_cache_budget");
    const char* value=std::getenv("KADAN_H3_DEVICES");
    if(!value||!*value)return {std::make_shared<Resources>(Footprint{cpu_host_bytes+cache_bytes}),{}};
#ifdef KADAN_H3_CUDA
    const std::string_view requested(value);std::vector<int> devices;
    if(requested=="0")devices={0};else if(requested=="1")devices={1};else if(requested=="0,1")devices={0,1};else if(requested=="1,0")devices={1,0};else throw std::invalid_argument("h3_cuda_devices");
    const char* scalar=std::getenv("KADAN_H3_GPU_BUDGET_BYTES");
    const char* indexed=std::getenv("KADAN_H3_GPU_BUDGETS");
    const auto budgets=device_budgets(devices,scalar?scalar:"3221225472",indexed?indexed:"",3ULL*1024*1024*1024);
    Bytes host_bytes=128ULL*1024*1024*1024;
    if(const char* value=std::getenv("KADAN_H3_HOST_BUDGET_BYTES")){const std::string_view text(value);auto parsed=std::from_chars(text.data(),text.data()+text.size(),host_bytes);if(parsed.ec!=std::errc{}||parsed.ptr!=text.data()+text.size()||host_bytes<1024ULL*1024*1024||host_bytes>128ULL*1024*1024*1024)throw std::invalid_argument("h3_host_budget");}
    Footprint capacity{host_bytes+cache_bytes,0,0};for(int device:devices)capacity[device+1]=budgets[device+1]-1024ULL*1024*1024;
    auto resources=std::make_shared<Resources>(std::move(capacity));
    return {resources,h3_cuda_compute(resources,std::move(devices))};
#else
    throw std::runtime_error("h3_cuda_not_built");
#endif
}
}
