#include "linear_fixture.hpp"
#include "kadan/cuda_linear_attention.hpp"
#include "kadan/cuda_state.hpp"
#include "kadan/cuda_projection.hpp"
#include "linear_kernel.cuh"
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
cudaError_t cudaMalloc(void** p,std::size_t n){*p=std::malloc(n);if(!*p)return cudaErrorUnknown;allocations.emplace(*p,n);used+=n;peak=std::max(peak,used);++mallocs;return cudaSuccess;}
cudaError_t cudaFree(void* p){check(!pending);used-=allocations.at(p);check(allocations.erase(p)==1);++frees;std::free(p);return cudaSuccess;}
cudaError_t cudaMemcpy(void* dst,const void* src,std::size_t n,cudaMemcpyKind){++copies;if(fail(Operation::copy))return cudaErrorUnknown;std::memcpy(dst,src,n);return cudaSuccess;}
cudaError_t cudaMemsetAsync(void* p,int value,std::size_t n,cudaStream_t){++memsets;if(fail(Operation::memset))return cudaErrorUnknown;std::memset(p,value,n);return cudaSuccess;}
cudaError_t cudaStreamSynchronize(cudaStream_t){++syncs;if(fail(Operation::synchronize))return cudaErrorUnknown;pending=false;return cudaSuccess;}
const char* cudaGetErrorString(cudaError_t){return "injected_cpu_cuda_failure";}
cudaError_t kadan_launch_fp8(const std::uint8_t*,const float*,bool,const float*,float*,unsigned*,std::size_t,std::size_t){++launches;pending=true;return fail(Operation::launch)?cudaErrorUnknown:cudaSuccess;}
cudaError_t kadan_launch_nvfp4(const std::uint8_t*,const std::uint8_t*,float,const float*,float*,unsigned*,std::size_t,std::size_t){return cudaSuccess;}
namespace kadan::cuda::detail {
cudaError_t linear_normalize(linear::Config,LinearBuffers,const float*){++launches;pending=true;return cudaSuccess;}
cudaError_t linear_core(linear::Config,LinearBuffers){launches+=5;return cudaSuccess;}
cudaError_t linear_residual(linear::Config,LinearBuffers,const float*,float*){++launches;return cudaSuccess;}
}
int main(){try{
    // Compile the real owner/projection/state C++ sources against this CPU shim;
    // no CUDA library, compiler or device is used. These are lifecycle tests,
    // not a simulation of numerical kernel results or real CUDA error semantics.
    for(int scenario=0;scenario<5;++scenario){
        LinearFixture f;auto resources=std::make_shared<kadan::Resources>(kadan::Footprint{0,65536});
        kadan::cuda::LinearAttention layer(f.config,f.weights(),0,resources);check(layer.valid());
        std::array<std::uint16_t,16> conv{};std::array<float,4> state{};std::array<float,2> x{1,1},y{};
        switch(scenario){
        case 0:inject(Operation::memset);runtime_failure([&]{layer.reset();});break;
        case 1:inject(Operation::synchronize);runtime_failure([&]{layer.reset();});break;
        case 2:inject(Operation::copy);runtime_failure([&]{layer.read_state(conv,state);});break;
        case 3:inject(Operation::copy,2);runtime_failure([&]{layer.read_state(conv,state);});break;
        case 4:inject(Operation::get_device,2);runtime_failure([&]{layer.step_device(x,y);});break;
        }
        check(!layer.valid() && layer.tokens()==0);
        bool rejected=false;try{layer.reset();}catch(const std::invalid_argument&){rejected=true;}check(rejected && !layer.valid());
        layer.close();check(!layer.valid());layer.close();
        check(resources->snapshot().used[1]==0 && resources->snapshot().residents==0 && allocations.empty());
    }
    // Borrowed work may outlive a failed launch/sync. Ordinary errors recover
    // quiescence; failed recovery must explicitly quarantine caller storage.
    for(int scenario=0;scenario<4;++scenario){
        LinearFixture f;auto resources=std::make_shared<kadan::Resources>(kadan::Footprint{0,65536});
        const auto io=resources->reserve(kadan::Workload::llm,{0,256});void* raw=nullptr;
        check(cudaMalloc(&raw,256)==cudaSuccess);resources->loaded(io);resources->pin(io);
        auto* x=static_cast<float*>(raw);auto* y=x+2;
        bool quarantined=false,caught=false;
        if(scenario<3){
            kadan::cuda::Fp8Projection projection(f.weights().qkv,0,resources);
            if(scenario==0)inject(Operation::launch);else inject(Operation::synchronize,1,scenario==2?2:1);
            try{projection.matvec_device({x,2},{y,8});}
            catch(const kadan::cuda::DeviceBufferQuarantine&){quarantined=true;caught=true;}
            catch(const std::runtime_error&){caught=true;}
            check(caught && quarantined==(scenario==2) && pending==(scenario==2));
            // Keep borrowed allocation admitted/pinned even after the throw.
            check(allocations.contains(raw));projection.close();check(!pending);
        }else{
            kadan::cuda::LinearAttention layer(f.config,f.weights(),0,resources);
            inject(Operation::synchronize,1,2); // normalize boundary + abort recovery
            try{layer.step_device({x,2},{y,2});}
            catch(const kadan::cuda::DeviceBufferQuarantine&){quarantined=true;}
            check(quarantined && pending && !layer.valid() && allocations.contains(raw));
            layer.close();check(!pending);
        }
        resources->unpin(io);resources->begin_eviction(io);check(cudaFree(raw)==cudaSuccess);resources->released(io);
        check(resources->snapshot().used[1]==0 && allocations.empty());
    }
    // Audit revised stage 2 against real owner calls with fake kernels. This
    // verifies requested allocations and explicit source-level operations only;
    // CUDA context/driver work and numerical correctness are not simulated.
    {
        check(used==0);peak=launches=syncs=copies=memsets=mallocs=frees=0;
        AsymmetricLinearFixture f;auto resources=std::make_shared<kadan::Resources>(kadan::Footprint{0,65536});
        auto io=resources->reserve(kadan::Workload::llm,{0,256});void* raw=nullptr;
        check(cudaMalloc(&raw,256)==cudaSuccess);resources->loaded(io);resources->pin(io);
        kadan::cuda::LinearAttention layer(f.config,f.weights(),0,resources);
        auto* x=static_cast<float*>(raw);auto* y=x+3;
        std::array<float,3> result{};std::array<float,24> state{};
        std::array<float,12> core{},gated{};std::array<std::uint16_t,60> conv{};
        check(resources->snapshot().used[1]==5632 && peak==5632);
        for(int replay=0;replay<2;++replay){
            for(auto token:linear_golden::inputs){
                check(cudaMemcpy(x,token.data(),12,cudaMemcpyHostToDevice)==cudaSuccess);
                layer.step_device({x,3},{y,3});
                check(cudaMemcpy(result.data(),y,12,cudaMemcpyDeviceToHost)==cudaSuccess);
                layer.read_state(conv,state);layer.read_intermediates(core,gated);
            }
            layer.reset();layer.read_state(conv,state);
        }
        layer.close();resources->unpin(io);resources->begin_eviction(io);
        check(cudaFree(raw)==cudaSuccess);resources->released(io);
        check(used==0 && allocations.empty() && resources->snapshot().used[1]==0);
        check(mallocs==6 && frees==6 && launches==80 && syncs==68 && memsets==35 && copies==113);
    }
    // Caller shape errors are not CUDA poison and must not destroy valid state.
    LinearFixture f;auto resources=std::make_shared<kadan::Resources>(kadan::Footprint{0,65536});
    kadan::cuda::LinearAttention layer(f.config,f.weights(),0,resources);
    std::array<std::uint16_t,1> short_conv{};std::array<float,4> state{};
    bool rejected=false;try{layer.read_state(short_conv,state);}catch(const std::invalid_argument&){rejected=true;}
    check(rejected && layer.valid());layer.reset();check(layer.valid());layer.close();check(allocations.empty());
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
