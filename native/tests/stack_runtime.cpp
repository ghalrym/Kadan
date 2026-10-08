#include "stack_fixture.hpp"
#include "decoder_producer.hpp"
#include "stack_kernel.cuh"
#include "kadan/cuda_stack.hpp"
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
bool pending=false,fail_final_copy=false,fail_final_sync=false,late_numeric=false,head_numeric=false;int attention_index=-1,residuals=0,expected_layers=4;unsigned selection_override=0;void* selection_address=nullptr;int final_sync_failures=1;std::atomic_bool* cancel_after_copy=nullptr;std::atomic_bool* cancellation=nullptr;
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
    if(kind==cudaMemcpyDeviceToHost&&s==selection_address){if(fail_final_copy){fail_final_copy=false;return cudaErrorUnknown;}if(fail_final_sync){fail_final_sync=false;inject(Op::sync,1,final_sync_failures);}if(cancel_after_copy)cancel_after_copy->store(true);}
    std::memcpy(d,s,n);return cudaSuccess;
}
cudaError_t cudaMemsetAsync(void*p,int v,std::size_t n,cudaStream_t){++memsets;if(fail(Op::memset))return cudaErrorUnknown;std::memset(p,v,n);return cudaSuccess;}
cudaError_t cudaStreamSynchronize(cudaStream_t){++syncs;if(fail(Op::sync))return cudaErrorUnknown;pending=false;return cudaSuccess;}
const char* cudaGetErrorString(cudaError_t){return "fake_cuda_failure";}
cudaError_t kadan_launch_fp8(const std::uint8_t*,const float*,bool,const float*,float*,unsigned*,std::size_t,std::size_t){return launch();}
cudaError_t kadan_launch_nvfp4(const std::uint8_t*,const std::uint8_t*,float,const float*,float*,unsigned*flags,std::size_t,std::size_t){if(head_numeric&&residuals==expected_layers)*flags=1;return launch();}
namespace kadan::cuda::detail {
cudaError_t stack_embedding(std::size_t,const std::uint16_t*,unsigned,float*){attention_index=-1;residuals=0;selection_address=nullptr;return launch();}
cudaError_t stack_select(std::size_t,float*,unsigned*result,unsigned*){*result=selection_override;selection_address=result;return launch();}

cudaError_t linear_normalize(linear::Config,LinearBuffers,const float*){++attention_index;return launch();}
cudaError_t linear_core(linear::Config,LinearBuffers b){b.convolution[0]=123;b.recurrent[0]=456;return launch(5);}
cudaError_t linear_residual(linear::Config,LinearBuffers,const float*,float*){return launch();}
cudaError_t full_normalize(full::Config,FullBuffers,const float*){++attention_index;return launch();}
cudaError_t full_core(full::Config,FullBuffers b,std::size_t){b.keys[0]=123;b.values[0]=456;return launch(3);}
cudaError_t full_residual(full::Config,FullBuffers,const float*,float*){return launch();}
cudaError_t moe_route(moe::Config c,MoeBuffers b,const float*){for(std::size_t i=0;i<c.top_k;++i)b.selected[i]=unsigned(i);return launch();}
cudaError_t moe_activate(std::size_t,MoeBuffers b){if(late_numeric&&attention_index==3)*b.status=1;return launch();}
cudaError_t moe_accumulate(moe::Config,MoeBuffers,unsigned){return launch();}
cudaError_t moe_finish(moe::Config,MoeBuffers,float*){return launch();}
cudaError_t decoder_norm(std::size_t,float,const std::uint16_t*,const float*,float*,unsigned*){return launch();}
cudaError_t decoder_residual(std::size_t,const float*,const float*,float*,unsigned*){++residuals;if(cancellation&&residuals==3)cancellation->store(true);return launch();}
}
#ifndef KADAN_MODEL_RUNTIME
int main(){try{
    using kadan::cuda::Stack;StackFixture f;auto c=f.config();auto p=kadan::stack::plan(c);auto metadata=Stack::host_metadata_bytes();
    auto manager=[&]{return std::make_shared<kadan::Resources>(kadan::Footprint{metadata,p.device_bytes});};
    {auto r=manager();std::vector<kadan::Handle> other;for(int i=0;i<1023;++i)other.push_back(r->reserve(kadan::Workload::video,{0,0}));Stack m(c,f.weights(),0,r);check(r->snapshot().residents==1024&&used==p.device_bytes);m.close();for(auto h:other)r->released(h);check(used==0);}
    {auto r=manager();std::vector<std::unique_ptr<Stack>> closed;closed.reserve(8);owner_allocations::target=metadata;owner_allocations::enabled=true;
        for(int i=0;i<8;++i){auto m=std::make_unique<Stack>(c,f.weights(),0,r);check(owner_allocations::live==metadata&&r->snapshot().used[0]==metadata);m->close();check(owner_allocations::live==0&&r->snapshot().used[0]==0&&!m->valid());m->close();rejected([&]{m->step(2);});rejected([&]{m->reset();});closed.push_back(std::move(m));}
        check(owner_allocations::count==8&&owner_allocations::peak==metadata);closed.clear();owner_allocations::enabled=false;}
    for(int dimension=0;dimension<2;++dimension){kadan::Footprint cap{metadata,p.device_bytes};--cap[dimension];auto r=std::make_shared<kadan::Resources>(cap);auto n=mallocs;rejected([&]{Stack m(c,f.weights(),0,r);});check(mallocs==n&&r->snapshot().residents==0);}
    for(int at:{3,100}){auto r=manager();inject(Op::copy,at);rejected([&]{Stack m(c,f.weights(),0,r);});check(used==0&&r->snapshot().residents==0);}
    for(int scenario=0;scenario<13;++scenario){
        auto r=manager();Stack model(c,f.weights(),0,r);model.step(2);check(model.valid()&&model.tokens()==1);std::atomic_bool stop=false;
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
        case 11:head_numeric=true;break;
        case 12:selection_override=99;break;
        }
        auto before=launches;auto result=kadan::stack::Selection{77,false};bool quarantine=rejected([&]{result=model.step(7,true,&stop);});check(quarantine==(scenario==4||scenario==10));check(!model.valid()&&model.tokens()==1&&result.token==77);
        if(scenario==0){check(launches-before==91);for(std::size_t i=0;i<4;++i){auto* state=static_cast<std::uint8_t*>(allocations.begin()->first)+p.offset[i]+p.layer[i].state_first;check(state[0]!=0);}}
        if(scenario==11)check(launches-before==94);
        late_numeric=head_numeric=false;cancellation=cancel_after_copy=nullptr;final_sync_failures=1;selection_override=0;stop=false;before=launches;rejected([&]{model.step(2);});check(launches==before);
        std::array<float,16>a{},b{};rejected([&]{model.read_output(a,b);});
        if(scenario==0||scenario==1||scenario==8||scenario==9||scenario==11||scenario==12){model.reset();check(model.valid()&&model.tokens()==0);for(std::size_t i=0;i<4;++i){std::vector<std::uint8_t> x(p.layer[i].state_first_bytes),y(p.layer[i].state_second_bytes);model.read_state(i,x,y);for(auto v:x)check(v==0);for(auto v:y)check(v==0);}model.step(2);check(model.tokens()==1);}
        else rejected([&]{model.reset();});model.close();check(used==0&&!pending&&r->snapshot().residents==0);
    }
    for(auto op:{Op::memset,Op::sync}){auto r=manager();Stack m(c,f.weights(),0,r);inject(op);rejected([&]{m.reset();});check(!m.valid());m.close();check(used==0);}
    for(auto op:{Op::free,Op::sync}){auto r=manager();std::size_t count=0;{Stack m(c,f.weights(),0,r);inject(op);rejected([&]{m.close();});check(r->snapshot().used[0]==metadata&&r->snapshot().used[1]==p.device_bytes);count=syncs;rejected([&]{m.close();});}check(count==syncs&&r->snapshot().residents==1);for(auto[ptr,n]:allocations){std::free(ptr);used-=n;}allocations.clear();pending=false;}
    // Foreign/stale capabilities are rejected by the real borrowed producer
    // BEFORE any memory operation, even with no device arena bound.
    {kadan::StateCursor a(4,8),b(4,8);kadan::cuda::detail::DecoderProducer producer(c.layer[0],p.layer[0],a);auto own=a.begin(),foreign=b.begin();auto before=launches;rejected([&]{producer.run(foreign,nullptr,nullptr);});a.abort(own);a.reset();rejected([&]{producer.run(own,nullptr,nullptr);});check(launches==before);}
    {auto r=manager();Stack model(c,f.weights(),0,r);auto before=launches;owner_allocations::all_count=0;owner_allocations::count_all=true;
        for(int replay=0;replay<2;++replay){for(int t=0;t<5;++t)model.step(2,false);model.reset();}
        owner_allocations::count_all=false;check(owner_allocations::all_count==0&&launches-before==950);for(int i=0;i<8;++i)model.step(2,false);before=launches;rejected([&]{model.step(2);});check(launches==before&&model.tokens()==8&&model.valid());model.close();check(used==0);}
    {auto eos=c;eos.eos=0;auto r=manager();Stack model(eos,f.weights(),0,r);check(model.step(2,false).eos&&!model.finished());check(model.step(7).eos&&model.finished());auto before=launches;rejected([&]{model.step(0);});check(launches==before&&model.tokens()==2);model.reset();check(!model.finished());model.close();}
    std::cout<<"Stack arena "<<p.device_bytes<<", metadata "<<metadata<<"; whole-token failure/quarantine, allocation and operation-count tests passed.\n";
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}

// Shared fake runtime is also used by checkpoint-owner tests.
#endif
std::size_t bf16_launches=0;
cudaError_t kadan_launch_fp8_bf16(const std::uint8_t*w,const float*s,bool row,const float*x,float*y,unsigned*f,std::size_t r,std::size_t c){++bf16_launches;return kadan_launch_fp8(w,s,row,x,y,f,r,c);}

cudaError_t kadan_launch_nvfp4_bf16(const std::uint8_t*w,const std::uint8_t*s,float g,const float*x,float*y,unsigned*f,std::size_t r,std::size_t c){++bf16_launches;check(g>0);return kadan_launch_nvfp4(w,s,g,x,y,f,r,c);}

cudaError_t kadan_launch_nvfp4_pair(const std::uint8_t*,const std::uint8_t*,float,const float*,float*,unsigned*,std::size_t,std::size_t,bool,Nvfp4Pair){return launch();}
cudaError_t kadan_launch_nvfp4_accumulate(const std::uint8_t*,const std::uint8_t*,float,const float*,float*,unsigned*,std::size_t,std::size_t,bool,Nvfp4Accumulation){return launch();}
