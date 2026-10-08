#include "moe_fixture.hpp"
#include "kadan/cuda_moe.hpp"
#include "moe_kernel.cuh"
#include "nvfp4_kernel.cuh"
#include <array>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <source_location>
#include <stdexcept>
#include <unordered_map>
#include <algorithm>
#include <new>
// Test-only allocation instrumentation. Track the actual allocation/deallocation
// of sizeof(Impl) while the regression is enabled; fixed pointer slots avoid any
// allocation recursion. Caller wrappers/fixtures/ledger storage are not owner bytes.
namespace owner_allocations {
bool enabled=false;std::size_t target=0,live=0,peak=0,count=0;
std::array<void*,64> pointers{};
void record(void* p,std::size_t size){
    if(!enabled||size!=target)return;
    for(auto& slot:pointers)if(!slot){slot=p;live+=size;peak=std::max(peak,live);++count;return;}
    std::abort();
}
void forget(void* p){if(!p)return;for(auto& slot:pointers)if(slot==p){slot=nullptr;live-=target;return;}}
}
void* operator new(std::size_t n){if(void* p=std::malloc(n?n:1)){owner_allocations::record(p,n);return p;}throw std::bad_alloc();}
void operator delete(void* p)noexcept{owner_allocations::forget(p);std::free(p);}
void operator delete(void* p,std::size_t)noexcept{::operator delete(p);}
void* operator new[](std::size_t n){return ::operator new(n);}
void operator delete[](void* p)noexcept{::operator delete(p);}
void operator delete[](void* p,std::size_t)noexcept{::operator delete(p);}
namespace {
std::unordered_map<void*,std::size_t> allocations;
std::size_t used=0,peak=0,launches=0,syncs=0,copies=0,memsets=0,mallocs=0,frees=0;
enum class Operation { none,get_device,memset,copy,synchronize,launch,free };
Operation failure=Operation::none;int remaining=0,repeats=0;bool pending=false;
bool fail(Operation op){if(failure!=op)return false;if(--remaining)return false;if(--repeats==0)failure=Operation::none;else remaining=1;return true;}
void inject(Operation op,int call=1,int count=1){failure=op;remaining=call;repeats=count;}
void check(bool ok,std::source_location at=std::source_location::current()){if(!ok)throw std::runtime_error("line_"+std::to_string(at.line()));}
template<class F>void runtime_failure(F f){bool caught=false;try{f();}catch(const std::runtime_error&){caught=true;}check(caught);}
}
cudaError_t cudaGetDevice(int* d){if(fail(Operation::get_device))return cudaErrorUnknown;*d=0;return cudaSuccess;}
cudaError_t cudaDeviceGetAttribute(int* v,cudaDeviceAttr attr,int){*v=attr==cudaDevAttrComputeCapabilityMajor?8:6;return cudaSuccess;}
cudaError_t cudaMalloc(void** p,std::size_t n){*p=std::calloc(1,n);if(!*p)return cudaErrorUnknown;allocations.emplace(*p,n);used+=n;peak=std::max(peak,used);++mallocs;return cudaSuccess;}
cudaError_t cudaFree(void* p){if(fail(Operation::free))return cudaErrorUnknown;check(!pending);used-=allocations.at(p);check(allocations.erase(p)==1);++frees;std::free(p);return cudaSuccess;}
cudaError_t cudaMemcpy(void* dst,const void* src,std::size_t n,cudaMemcpyKind){++copies;if(fail(Operation::copy))return cudaErrorUnknown;std::memcpy(dst,src,n);return cudaSuccess;}
cudaError_t cudaMemsetAsync(void* p,int value,std::size_t n,cudaStream_t){++memsets;if(fail(Operation::memset))return cudaErrorUnknown;std::memset(p,value,n);return cudaSuccess;}
cudaError_t cudaStreamSynchronize(cudaStream_t){++syncs;if(fail(Operation::synchronize))return cudaErrorUnknown;pending=false;return cudaSuccess;}
const char* cudaGetErrorString(cudaError_t){return "injected_cpu_cuda_failure";}
cudaError_t kadan_launch_nvfp4(const std::uint8_t*,const std::uint8_t*,float,const float*,float*,unsigned*,std::size_t,std::size_t){++launches;pending=true;return fail(Operation::launch)?cudaErrorUnknown:cudaSuccess;}
namespace kadan::cuda::detail {
cudaError_t moe_route(moe::Config c,MoeBuffers b,const float*){++launches;pending=true;for(std::size_t j=0;j<c.top_k;++j)b.selected[j]=unsigned(j);return cudaSuccess;}
cudaError_t moe_activate(std::size_t,MoeBuffers){++launches;pending=true;return cudaSuccess;}
cudaError_t moe_accumulate(moe::Config,MoeBuffers,unsigned){++launches;pending=true;return cudaSuccess;}
cudaError_t moe_finish(moe::Config,MoeBuffers,float*){++launches;pending=true;return cudaSuccess;}
}
int main(){try{
    // CPU shim runs real layer-arena ownership code. No CUDA execution or math emulation.
    for(int scenario=0;scenario<10;++scenario){
        MoeFixture f;auto resources=std::make_shared<kadan::Resources>(kadan::Footprint{131072,65536});
        kadan::cuda::Moe layer(f.config,f.weights(),0,resources);std::array<float,16> x{},y{},routed{},shared{},result{};
        std::array<unsigned,2> selected{};std::array<float,4> logits{},prob{};std::array<float,2> weights{};
        switch(scenario){
        case 0:inject(Operation::memset);runtime_failure([&]{layer.reset();});break;
        case 1:inject(Operation::synchronize);runtime_failure([&]{layer.reset();});break;
        case 2:inject(Operation::get_device);runtime_failure([&]{layer.forward_device(x,y);});break;
        default:
            layer.forward_device(x,y);inject(Operation::copy,scenario<=6?scenario-2:scenario-6);
            if(scenario<=6)runtime_failure([&]{layer.read_routes(selected,logits,prob,weights);});else runtime_failure([&]{layer.read_outputs(routed,shared,result);});
        }
        check(!layer.valid());bool rejected=false;try{layer.reset();}catch(const std::invalid_argument&){rejected=true;}check(rejected);
        layer.close();layer.close();check(!layer.valid()&&used==0&&allocations.empty()&&resources->snapshot().used[0]==0&&resources->snapshot().used[1]==0);
    }
    // Regression: retaining closed public wrappers must not retain uncharged
    // descriptor tables. A one-owner RAM cap must bound actual owner allocation.
    {
        MoeFixture f;const auto bytes=kadan::cuda::Moe::host_metadata_bytes();
        auto resources=std::make_shared<kadan::Resources>(kadan::Footprint{bytes,65536});
        std::vector<std::unique_ptr<kadan::cuda::Moe>> closed;closed.reserve(8);
        owner_allocations::target=bytes;owner_allocations::enabled=true;
        for(int iteration=0;iteration<8;++iteration){
            auto wrapper=std::make_unique<kadan::cuda::Moe>(f.config,f.weights(),0,resources);
            check(owner_allocations::live==bytes&&resources->snapshot().used[0]==bytes);
            wrapper->close();check(owner_allocations::live==0&&resources->snapshot().used[0]==0&&used==0);
            check(!wrapper->valid());wrapper->close();
            std::array<float,16> x{},y{};std::array<float,4> logits{},prob{};std::array<float,2> weights{};std::array<unsigned,2> selected{};
            auto rejected=[&](auto fn){bool caught=false;try{fn();}catch(const std::invalid_argument& e){caught=std::string_view(e.what())=="moe_closed";}check(caught);};
            rejected([&]{wrapper->reset();});rejected([&]{wrapper->forward_device(x,y);});
            rejected([&]{wrapper->read_routes(selected,logits,prob,weights);});rejected([&]{wrapper->read_outputs(x,y,x);});
            closed.push_back(std::move(wrapper));
        }
        check(closed.size()==8&&owner_allocations::count==8&&owner_allocations::peak==bytes&&owner_allocations::live==0);
        closed.clear();check(owner_allocations::live==0&&resources->snapshot().residents==0);owner_allocations::enabled=false;
    }
    // Admission is one layer handle, even when only one ledger entry remains.
    {
        MoeFixture f;auto resources=std::make_shared<kadan::Resources>(kadan::Footprint{131072,65536});std::vector<kadan::Handle> handles;
        for(std::size_t i=0;i<kadan::Resources::max_residents-1;++i)handles.push_back(resources->reserve(kadan::Workload::image,{0,0}));
        kadan::cuda::Moe layer(f.config,f.weights(),0,resources);check(resources->snapshot().residents==1024);check(resources->snapshot().used[0]==layer.host_metadata_bytes());
        layer.close();for(auto h:handles)resources->released(h);check(resources->snapshot().residents==0&&used==0);
    }
    // Reject insufficient host/device budgets without allocating device storage.
    for(int dimension=0;dimension<2;++dimension){
        MoeFixture f;kadan::Footprint cap{131072,65536};cap[dimension]=1;auto resources=std::make_shared<kadan::Resources>(cap);
        auto calls=mallocs;runtime_failure([&]{kadan::cuda::Moe layer(f.config,f.weights(),0,resources);});check(mallocs==calls&&resources->snapshot().residents==0);
    }
    // Constructor upload failure cleans the partially populated arena.
    {MoeFixture f;auto resources=std::make_shared<kadan::Resources>(kadan::Footprint{131072,65536});inject(Operation::copy,3);
        runtime_failure([&]{kadan::cuda::Moe layer(f.config,f.weights(),0,resources);});check(used==0&&resources->snapshot().residents==0);}
    for(int scenario=0;scenario<3;++scenario){
        MoeFixture f;auto resources=std::make_shared<kadan::Resources>(kadan::Footprint{131072,65536});
        auto io=resources->reserve(kadan::Workload::llm,{0,256});void* raw=nullptr;check(cudaMalloc(&raw,256)==cudaSuccess);resources->loaded(io);resources->pin(io);
        kadan::cuda::Moe layer(f.config,f.weights(),0,resources);auto* x=static_cast<float*>(raw);
        if(scenario==0)inject(Operation::launch);else inject(Operation::synchronize,1,scenario==2?2:1);
        bool caught=false,quarantine=false;try{layer.forward_device({x,16},{x+16,16});}catch(const kadan::cuda::DeviceBufferQuarantine&){caught=quarantine=true;}catch(const std::runtime_error&){caught=true;}
        check(caught&&quarantine==(scenario==2)&&pending==(scenario==2)&&!layer.valid());layer.close();check(!pending);
        resources->unpin(io);resources->begin_eviction(io);check(cudaFree(raw)==cudaSuccess);resources->released(io);check(used==0&&resources->snapshot().residents==0);
    }
    // Failed explicit close latches, retains admission and is never retried by destruction.
    for(auto op:{Operation::synchronize,Operation::free}){
        MoeFixture f;auto resources=std::make_shared<kadan::Resources>(kadan::Footprint{131072,65536});std::size_t calls=0;
        {kadan::cuda::Moe layer(f.config,f.weights(),0,resources);inject(op);runtime_failure([&]{layer.close();});calls=syncs;check(!layer.valid());runtime_failure([&]{layer.close();});check(syncs==calls);}
        check(syncs==calls&&resources->snapshot().residents==1&&resources->snapshot().used[1]==9472);
        // Test-only context teardown of fake allocations; NOT a retry or a claim
        // that production can release quarantined reservations without cleanup.
        for(auto [ptr,n]:allocations){std::free(ptr);used-=n;}allocations.clear();pending=false;check(used==0);
    }
    {
        peak=launches=syncs=copies=memsets=mallocs=frees=0;MoeFixture f;auto resources=std::make_shared<kadan::Resources>(kadan::Footprint{131072,65536});
        auto io=resources->reserve(kadan::Workload::llm,{0,256});void* raw=nullptr;check(cudaMalloc(&raw,256)==cudaSuccess);resources->loaded(io);resources->pin(io);
        kadan::cuda::Moe layer(f.config,f.weights(),0,resources);auto* x=static_cast<float*>(raw);check(peak==9728&&resources->snapshot().residents==2);
        std::array<float,16> result{},routed{},shared{},snapshot{};std::array<unsigned,2> selected{};std::array<float,4> logits{},prob{};std::array<float,2> weights{};
        for(int replay=0;replay<2;++replay){for(auto input:moe_golden::inputs){
            check(cudaMemcpy(x,input.data(),64,cudaMemcpyHostToDevice)==cudaSuccess);layer.forward_device({x,16},{x+16,16});check(cudaMemcpy(result.data(),x+16,64,cudaMemcpyDeviceToHost)==cudaSuccess);
            layer.read_routes(selected,logits,prob,weights);layer.read_outputs(routed,shared,snapshot);
        }layer.reset();}
        layer.close();resources->unpin(io);resources->begin_eviction(io);check(cudaFree(raw)==cudaSuccess);resources->released(io);
        check(used==0&&resources->snapshot().residents==0);
        check(mallocs==2&&frees==2&&launches==160&&syncs==74&&memsets==13&&copies==202);
        std::cout<<"Owner RAM bytes: "<<layer.host_metadata_bytes()<<"; stage2 device peak "<<peak<<" bytes.\n";
    }
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
