#include "decoder_fixture.hpp"
#include "kadan/cuda_decoder.hpp"
#include "linear_kernel.cuh"
#include "full_kernel.cuh"
#include "moe_kernel.cuh"
#include "decoder_kernel.cuh"
#include "fp8_kernel.cuh"
#include "nvfp4_kernel.cuh"
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <new>
#include <source_location>
#include <unordered_map>
// Test-only allocation instrumentation. Track the actual allocation/deallocation
// of sizeof(Impl) while the regression is enabled; fixed pointer slots avoid any
// allocation recursion. Caller wrappers/fixtures/ledger storage are not owner bytes.
namespace owner_allocations {
bool enabled=false,count_all=false;std::size_t all_count=0;std::size_t target=0,live=0,peak=0,count=0;
std::array<void*,64> pointers{};
void record(void* p,std::size_t size){
    if(count_all)++all_count;
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
void check(bool x,std::source_location at=std::source_location::current()){if(!x)throw std::runtime_error("line_"+std::to_string(at.line()));}
std::unordered_map<void*,std::size_t> allocations;std::size_t used=0,peak=0,launches=0,syncs=0,copies=0,memsets=0,mallocs=0,frees=0;
enum class Op{none,copy,sync,memset,launch,free,get};Op failure=Op::none;int remaining=0,repeats=0;
int numeric_stage=0;
bool pending=false,fail_final_copy=false,fail_final_sync=false,late_numeric=false;int final_sync_failures=1;std::atomic_bool* cancel_after_copy=nullptr;std::atomic_bool* cancellation=nullptr;
bool fail(Op op){if(op!=failure)return false;if(--remaining)return false;if(--repeats==0)failure=Op::none;else remaining=1;return true;}
void inject(Op op,int call=1,int count=1){failure=op;remaining=call;repeats=count;}
template<class F>bool rejected(F f){try{f();}catch(const kadan::cuda::DeviceBufferQuarantine&){return true;}catch(const std::exception&){return false;}throw std::runtime_error("expected_rejection");}
cudaError_t launch(std::size_t n=1){launches+=n;pending=true;return fail(Op::launch)?cudaErrorUnknown:cudaSuccess;}
}
cudaError_t cudaGetDevice(int* d){if(fail(Op::get))return cudaErrorUnknown;*d=0;return cudaSuccess;}
cudaError_t cudaDeviceGetAttribute(int* v,cudaDeviceAttr attr,int){*v=attr==cudaDevAttrComputeCapabilityMajor?8:6;return cudaSuccess;}
cudaError_t cudaMalloc(void**p,std::size_t n){*p=std::calloc(1,n);if(!*p)return cudaErrorUnknown;allocations.emplace(*p,n);used+=n;peak=std::max(peak,used);++mallocs;return cudaSuccess;}
cudaError_t cudaFree(void*p){if(fail(Op::free))return cudaErrorUnknown;check(!pending);used-=allocations.at(p);allocations.erase(p);std::free(p);++frees;return cudaSuccess;}
cudaError_t cudaMemcpy(void*d,const void*s,std::size_t n,cudaMemcpyKind kind){
    ++copies;if(fail(Op::copy))return cudaErrorUnknown;
    if(kind==cudaMemcpyDeviceToDevice){if(fail_final_copy){fail_final_copy=false;return cudaErrorUnknown;}if(fail_final_sync){fail_final_sync=false;inject(Op::sync,1,final_sync_failures);}if(cancel_after_copy)cancel_after_copy->store(true);}
    std::memcpy(d,s,n);return cudaSuccess;
}
cudaError_t cudaMemsetAsync(void*p,int v,std::size_t n,cudaStream_t){++memsets;if(fail(Op::memset))return cudaErrorUnknown;std::memset(p,v,n);return cudaSuccess;}
cudaError_t cudaStreamSynchronize(cudaStream_t){++syncs;if(fail(Op::sync))return cudaErrorUnknown;pending=false;return cudaSuccess;}
const char* cudaGetErrorString(cudaError_t){return "fake_cuda_failure";}
cudaError_t kadan_launch_fp8(const std::uint8_t*,const float*,bool,const float*,float*,unsigned* flags,std::size_t,std::size_t){if(numeric_stage==1)*flags=1;return launch();}
cudaError_t kadan_launch_nvfp4(const std::uint8_t*,const std::uint8_t*,float,const float*,float*,unsigned*,std::size_t,std::size_t){return launch();}
namespace kadan::cuda::detail {
cudaError_t linear_normalize(linear::Config,LinearBuffers,const float*){return launch();}
cudaError_t linear_core(linear::Config,LinearBuffers b){b.convolution[0]=123;b.recurrent[0]=456;return launch(5);}
cudaError_t linear_residual(linear::Config,LinearBuffers,const float*,float*){if(cancellation)cancellation->store(true);return launch();}
cudaError_t full_normalize(full::Config,FullBuffers,const float*){return launch();}
cudaError_t full_core(full::Config,FullBuffers b,std::size_t){b.keys[0]=123;b.values[0]=456;return launch(3);}
cudaError_t full_residual(full::Config,FullBuffers,const float*,float*){if(cancellation)cancellation->store(true);return launch();}
cudaError_t moe_route(moe::Config c,MoeBuffers b,const float*){for(std::size_t i=0;i<c.top_k;++i)b.selected[i]=numeric_stage==2?~0u:unsigned(i);return launch();}
cudaError_t moe_activate(std::size_t,MoeBuffers b){if(late_numeric)*b.status=1;return launch();}
cudaError_t moe_accumulate(moe::Config,MoeBuffers,unsigned){return launch();}
cudaError_t moe_finish(moe::Config,MoeBuffers,float*){return launch();}
cudaError_t decoder_norm(std::size_t,float,const std::uint16_t*,const float*,float*,unsigned*flags){if(numeric_stage==2)*flags=1;return launch();}
cudaError_t decoder_residual(std::size_t,const float*,const float*,float*,unsigned*){return launch();}
}
int main(){try{
    using kadan::cuda::Decoder;
    for(auto kind:{kadan::decoder::Attention::linear,kadan::decoder::Attention::full}){
        DecoderFixture f;auto c=f.config(kind);auto p=kadan::decoder::plan(c);const auto metadata=Decoder::host_metadata_bytes();
        auto manager=[&]{return std::make_shared<kadan::Resources>(kadan::Footprint{metadata,p.device_bytes});};
        // One host owner/one allocation even under a completely full ledger.
        {auto r=manager();std::vector<kadan::Handle> other;for(int i=0;i<1023;++i)other.push_back(r->reserve(kadan::Workload::image,{0,0}));Decoder d(c,f.weights(),0,r);check(r->snapshot().residents==1024&&used==p.device_bytes);d.close();for(auto h:other)r->released(h);check(used==0&&r->snapshot().residents==0);}
        // Actual metadata allocation vs reservation with closed wrappers retained.
        {auto r=manager();std::vector<std::unique_ptr<Decoder>> closed;closed.reserve(8);owner_allocations::target=metadata;owner_allocations::enabled=true;owner_allocations::count=owner_allocations::peak=0;
            for(int i=0;i<8;++i){auto d=std::make_unique<Decoder>(c,f.weights(),0,r);check(owner_allocations::live==metadata&&r->snapshot().used[0]==metadata);d->close();check(owner_allocations::live==0&&r->snapshot().used[0]==0&&!d->valid());d->close();rejected([&]{d->reset();});closed.push_back(std::move(d));}
            check(owner_allocations::count==8&&owner_allocations::peak==metadata);closed.clear();owner_allocations::enabled=false;}
        for(int dimension=0;dimension<2;++dimension){kadan::Footprint cap{metadata,p.device_bytes};--cap[dimension];auto r=std::make_shared<kadan::Resources>(cap);auto n=mallocs;rejected([&]{Decoder d(c,f.weights(),0,r);});check(mallocs==n&&r->snapshot().residents==0);}
        {auto r=manager();inject(Op::copy,3);rejected([&]{Decoder d(c,f.weights(),0,r);});check(used==0&&r->snapshot().residents==0);}
        for(int scenario=0;scenario<14;++scenario){
            auto r=manager();Decoder d(c,f.weights(),0,r);std::array<float,16>x{},y{},a{},u{},m{};std::atomic_bool stop=false;
            d.step_device(x,y);check(d.valid()&&d.tokens()==1);y.fill(42);
            switch(scenario){
            case 0:late_numeric=true;break;
            case 1:cancellation=&stop;break;
            case 2:fail_final_copy=true;break;
            case 3:fail_final_sync=true;break;
            case 4:inject(Op::sync,1,2);break;
            case 5:inject(Op::launch);break;
            case 6:inject(Op::copy);break;
            case 7:inject(Op::get);break;
            case 8:stop=true;break;
            case 9:cancel_after_copy=&stop;break;
            case 10:fail_final_sync=true;final_sync_failures=2;break;
            case 11:numeric_stage=1;break;
            case 12:numeric_stage=2;break;
            case 13:numeric_stage=1;cancellation=&stop;break;
            }
            auto before_launches=launches;bool numeric_error=false;bool quarantine=rejected([&]{try{d.step_device(x,y,&stop);}catch(const std::overflow_error&){numeric_error=true;throw;}});if(scenario>=11)check(numeric_error);check(quarantine==(scenario==4||scenario==10));check(!d.valid()&&d.tokens()==1);
            if(scenario!=3&&scenario!=9&&scenario!=10)for(float v:y)check(v==42);
            if(scenario==0){check(launches-before_launches==(kind==kadan::decoder::Attention::linear?23:22));auto* state=static_cast<std::uint8_t*>(allocations.begin()->first)+p.state_first;check(state[0]!=0);}
            late_numeric=false;numeric_stage=0;cancellation=nullptr;cancel_after_copy=nullptr;final_sync_failures=1;stop=false;auto n=launches;rejected([&]{d.step_device(x,y);});check(launches==n);rejected([&]{d.read_intermediates(a,u,m);});
            if(scenario==0||scenario==1||scenario==8||scenario==9||scenario>=11){d.reset();check(d.valid()&&d.tokens()==0);std::vector<std::uint8_t> first(p.state_first_bytes),second(p.state_second_bytes);d.read_state(first,second);for(auto v:first)check(v==0);for(auto v:second)check(v==0);d.step_device(x,y);check(d.tokens()==1);}
            else rejected([&]{d.reset();});
            d.close();check(!pending&&used==0&&r->snapshot().residents==0);
        }
        for(auto op:{Op::memset,Op::sync}){auto r=manager();Decoder d(c,f.weights(),0,r);inject(op);rejected([&]{d.reset();});check(!d.valid());d.close();check(used==0);}
        for(auto op:{Op::free,Op::sync}){auto r=manager();std::size_t count=0;{Decoder d(c,f.weights(),0,r);inject(op);rejected([&]{d.close();});check(r->snapshot().used[0]==metadata&&r->snapshot().used[1]==p.device_bytes);count=syncs;rejected([&]{d.close();});}check(count==syncs&&r->snapshot().residents==1);
            for(auto [ptr,n]:allocations){std::free(ptr);used-=n;}allocations.clear();pending=false;}
        // Count actual producer launches for the staged six-token replay proposal.
        {auto r=manager();Decoder d(c,f.weights(),0,r);std::array<float,16>x{},y{};auto n=launches;
            owner_allocations::all_count=0;owner_allocations::count_all=true;
            for(int replay=0;replay<2;++replay){for(int t=0;t<3;++t)d.step_device(x,y);d.reset();}
            owner_allocations::count_all=false;check(owner_allocations::all_count==0);
            check(launches-n==6*(kind==kadan::decoder::Attention::linear?23:22));d.close();check(r->snapshot().residents==0&&used==0);}
        std::cout<<(kind==kadan::decoder::Attention::linear?"linear":"full")<<" arena "<<p.device_bytes<<", metadata "<<metadata<<"; failure publication, quarantine and accounting passed\n";
    }
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}

cudaError_t kadan_launch_fp8_bf16(const std::uint8_t*w,const float*s,bool row,const float*x,float*y,unsigned*f,std::size_t r,std::size_t c){return kadan_launch_fp8(w,s,row,x,y,f,r,c);}

cudaError_t kadan_launch_nvfp4_bf16(const std::uint8_t*w,const std::uint8_t*s,float g,const float*x,float*y,unsigned*f,std::size_t r,std::size_t c){return kadan_launch_nvfp4(w,s,g,x,y,f,r,c);}

cudaError_t kadan_launch_nvfp4_pair(const std::uint8_t*,const std::uint8_t*,float,const float*,float*,unsigned*,std::size_t,std::size_t,bool,Nvfp4Pair){return launch();}
cudaError_t kadan_launch_nvfp4_accumulate(const std::uint8_t*,const std::uint8_t*,float,const float*,float*,unsigned*,std::size_t,std::size_t,bool,Nvfp4Accumulation){return launch();}
