#include "full_fixture.hpp"
#include "kadan/cuda_full_attention.hpp"
#include "kadan/cuda_state.hpp"
#include "kadan/cuda_projection.hpp"
#include "full_kernel.cuh"
#include "fp8_kernel.cuh"
#include "nvfp4_kernel.cuh"
#include <array>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <source_location>
#include <stdexcept>
#include <unordered_map>
#include <algorithm>
namespace {
std::unordered_map<void*,std::size_t> allocations;
std::size_t used=0,peak=0,launches=0,syncs=0,copies=0,memsets=0,mallocs=0,frees=0;
enum class Operation { none,get_device,memset,copy,synchronize,launch };
Operation failure=Operation::none;int remaining=0,repeats=0;bool pending=false;
bool fail(Operation op){if(failure!=op)return false;if(--remaining)return false;if(--repeats==0)failure=Operation::none;else remaining=1;return true;}
void inject(Operation op,int call=1,int count=1){failure=op;remaining=call;repeats=count;}
void check(bool ok,std::source_location at=std::source_location::current()){if(!ok)throw std::runtime_error("line_"+std::to_string(at.line()));}
template<class F>void runtime_failure(F f){bool caught=false;try{f();}catch(const std::runtime_error&){caught=true;}check(caught);}
}
cudaError_t cudaGetDevice(int* d){if(fail(Operation::get_device))return cudaErrorUnknown;*d=0;return cudaSuccess;}
cudaError_t cudaDeviceGetAttribute(int* v,cudaDeviceAttr attr,int){*v=attr==cudaDevAttrComputeCapabilityMajor?8:6;return cudaSuccess;}
cudaError_t cudaMalloc(void** p,std::size_t n){*p=std::calloc(1,n);if(!*p)return cudaErrorUnknown;allocations.emplace(*p,n);used+=n;peak=std::max(peak,used);++mallocs;return cudaSuccess;}
cudaError_t cudaFree(void* p){check(!pending);used-=allocations.at(p);check(allocations.erase(p)==1);++frees;std::free(p);return cudaSuccess;}
cudaError_t cudaMemcpy(void* dst,const void* src,std::size_t n,cudaMemcpyKind){++copies;if(fail(Operation::copy))return cudaErrorUnknown;std::memcpy(dst,src,n);return cudaSuccess;}
cudaError_t cudaMemsetAsync(void* p,int value,std::size_t n,cudaStream_t){++memsets;if(fail(Operation::memset))return cudaErrorUnknown;std::memset(p,value,n);return cudaSuccess;}
cudaError_t cudaStreamSynchronize(cudaStream_t){++syncs;if(fail(Operation::synchronize))return cudaErrorUnknown;pending=false;return cudaSuccess;}
const char* cudaGetErrorString(cudaError_t){return "injected_cpu_cuda_failure";}
cudaError_t kadan_launch_fp8(const std::uint8_t*,const float*,bool,const float*,float*,unsigned*,std::size_t,std::size_t){++launches;pending=true;return fail(Operation::launch)?cudaErrorUnknown:cudaSuccess;}
cudaError_t kadan_launch_nvfp4(const std::uint8_t*,const std::uint8_t*,float,const float*,float*,unsigned*,std::size_t,std::size_t){return cudaSuccess;}
namespace kadan::cuda::detail {
cudaError_t full_normalize(full::Config,FullBuffers,const float*){++launches;pending=true;return cudaSuccess;}
cudaError_t full_core(full::Config,FullBuffers,std::size_t){launches+=3;pending=true;return cudaSuccess;}
cudaError_t full_residual(full::Config,FullBuffers,const float*,float*){++launches;pending=true;return cudaSuccess;}
}
int main(){try{
    // Real C++ owner/state/projection sources against a CPU-only runtime shim.
    // No CUDA execution, numerical emulation, or driver-overhead claim.
    for(int scenario=0;scenario<8;++scenario){
        FullFixture f;auto resources=std::make_shared<kadan::Resources>(kadan::Footprint{0,65536});
        kadan::cuda::FullAttention layer(f.config,f.weights(),0,resources);
        std::array<std::uint16_t,48> keys{},values{};std::array<float,3> x{1,2,-1},y{};
        std::array<float,16> prob{};std::array<float,24> core{},gated{};
        switch(scenario){
        case 0:inject(Operation::memset);runtime_failure([&]{layer.reset();});break;
        case 1:inject(Operation::synchronize);runtime_failure([&]{layer.reset();});break;
        case 2:inject(Operation::copy);runtime_failure([&]{layer.read_state(keys,values);});break;
        case 3:inject(Operation::copy,2);runtime_failure([&]{layer.read_state(keys,values);});break;
        case 4:inject(Operation::get_device,2);runtime_failure([&]{layer.step_device(x,y);});break;
        default:layer.step_device(x,y);inject(Operation::copy,scenario-4);runtime_failure([&]{layer.read_intermediates(prob,core,gated);});break;
        }
        check(!layer.valid()&&layer.tokens()==std::size_t(scenario>=5));
        bool rejected=false;try{layer.reset();}catch(const std::invalid_argument&){rejected=true;}check(rejected);
        layer.close();layer.close();check(!layer.valid()&&used==0&&allocations.empty()&&resources->snapshot().used[1]==0);
    }
    // Keep actual admitted borrowed storage pinned across uncertain sync failure.
    for(int repeats=1;repeats<=2;++repeats){
        FullFixture f;auto resources=std::make_shared<kadan::Resources>(kadan::Footprint{0,65536});
        auto io=resources->reserve(kadan::Workload::llm,{0,256});void* raw=nullptr;
        check(cudaMalloc(&raw,256)==cudaSuccess);resources->loaded(io);resources->pin(io);
        kadan::cuda::FullAttention layer(f.config,f.weights(),0,resources);auto* x=static_cast<float*>(raw);
        inject(Operation::synchronize,1,repeats);bool caught=false,quarantine=false;
        try{layer.step_device({x,3},{x+3,3});}catch(const kadan::cuda::DeviceBufferQuarantine&){caught=quarantine=true;}catch(const std::runtime_error&){caught=true;}
        check(caught&&quarantine==(repeats==2)&&pending==(repeats==2)&&!layer.valid());
        layer.close();check(!pending);resources->unpin(io);resources->begin_eviction(io);check(cudaFree(raw)==cudaSuccess);resources->released(io);
        check(used==0&&allocations.empty()&&resources->snapshot().used[1]==0);
    }
    // Audit stage 2's complete source-level allocation/operation bounds.
    {
        peak=launches=syncs=copies=memsets=mallocs=frees=0;
        FullFixture f;auto resources=std::make_shared<kadan::Resources>(kadan::Footprint{0,65536});
        auto io=resources->reserve(kadan::Workload::llm,{0,256});void* raw=nullptr;
        check(cudaMalloc(&raw,256)==cudaSuccess);resources->loaded(io);resources->pin(io);
        kadan::cuda::FullAttention layer(f.config,f.weights(),0,resources);auto* x=static_cast<float*>(raw);
        std::array<float,3> result{};std::array<std::uint16_t,48> keys{},values{};
        std::array<float,16> probabilities{};std::array<float,24> core{},gated{};
        check(peak==7424&&resources->snapshot().used[1]==7424);
        for(int replay=0;replay<2;++replay){
            for(auto token:full_golden::inputs){
                check(cudaMemcpy(x,token.data(),12,cudaMemcpyHostToDevice)==cudaSuccess);layer.step_device({x,3},{x+3,3});
                check(cudaMemcpy(result.data(),x+3,12,cudaMemcpyDeviceToHost)==cudaSuccess);layer.read_state(keys,values);layer.read_intermediates(probabilities,core,gated);
            }
            const auto before=launches;bool rejected=false;try{layer.step_device({x,3},{x+3,3});}catch(const std::invalid_argument&){rejected=true;}
            check(rejected&&layer.valid()&&layer.tokens()==4&&launches==before);
            layer.reset();layer.read_state(keys,values);
        }
        layer.close();resources->unpin(io);resources->begin_eviction(io);check(cudaFree(raw)==cudaSuccess);resources->released(io);
        check(used==0&&allocations.empty()&&resources->snapshot().used[1]==0);
        check(mallocs==7&&frees==7&&launches==72&&syncs==78&&memsets==43&&copies==128);
    }
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
