#pragma once
#include "kadan/h3_compute.hpp"
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
    Bytes gpu_bytes=3ULL*1024*1024*1024;
    if(const char* raw=std::getenv("KADAN_H3_GPU_BUDGET_BYTES")){
        const std::string_view text(raw);auto parsed=std::from_chars(text.data(),text.data()+text.size(),gpu_bytes);
        if(parsed.ec!=std::errc{}||parsed.ptr!=text.data()+text.size()||gpu_bytes<3ULL*1024*1024*1024||gpu_bytes>24ULL*1024*1024*1024)throw std::invalid_argument("h3_gpu_budget");
    }
    Footprint capacity{128ULL*1024*1024*1024+cache_bytes,0,0};for(int device:devices)capacity[device+1]=gpu_bytes-1024ULL*1024*1024;
    auto resources=std::make_shared<Resources>(std::move(capacity));
    return {resources,h3_cuda_compute(resources,std::move(devices))};
#else
    throw std::runtime_error("h3_cuda_not_built");
#endif
}
}
