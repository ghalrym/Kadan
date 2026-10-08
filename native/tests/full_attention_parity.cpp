#include "full_fixture.hpp"
#include "kadan/cuda_full_attention.hpp"
#include "kadan/linear_attention.hpp"
#include <cuda_runtime_api.h>
#include <algorithm>
#include <charconv>
#include <iostream>
#include <stdexcept>
#include <string_view>
namespace {
void check(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}
void cuda(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
int integer(const char* s){std::string_view v(s);int n=-1;auto [end,error]=std::from_chars(v.data(),v.data()+v.size(),n);check(error==std::errc{}&&end==v.data()+v.size(),"invalid_argument");return n;}
template<class A,class B>void exact(const A& a,const B& b,const char* why){check(a.size()==b.size()&&std::equal(a.begin(),a.end(),b.begin()),why);}
}
int main(int argc,char** argv){
    if(argc!=6||std::string_view(argv[1])!="--allow-gpu-validation"||std::string_view(argv[2])!="--device"||std::string_view(argv[4])!="--stage")return 2;
    try{
        const int device=integer(argv[3]),stage=integer(argv[5]);check(device>=0&&device<64&&stage>=1&&stage<=3,"invalid_device_or_stage");
        cuda(cudaSetDevice(device));kadan::Footprint cap(device+2,0);cap[device+1]=65536;
        auto resources=std::make_shared<kadan::Resources>(cap);auto io=cap;io[device+1]=256;
        const auto handle=resources->reserve(kadan::Workload::llm,io);void* storage=nullptr;cuda(cudaMalloc(&storage,256));resources->loaded(handle);resources->pin(handle);
        auto* input=static_cast<float*>(storage);auto* output=input+3;FullFixture fixture;
        if(stage==3)fixture.query_norm[0]=fixture.key_norm[0]=kadan::linear::bf16_round(1e20f);
        kadan::cuda::FullAttention gpu(fixture.config,fixture.weights(),device,resources);
        check(resources->snapshot().used[device+1]==7424,"full_peak_admission");
        kadan::full::Reference cpu(fixture.config,fixture.weights(),952);
        std::array<float,3> actual{},expected{};std::array<std::uint16_t,48> keys{},values{};
        std::array<float,16> probabilities{};std::array<float,24> core{},gated{};
        if(stage<=2){
            const int replays=stage==1?1:2,count=stage==1?1:4;
            for(int replay=0;replay<replays;++replay){
                for(int t=0;t<count;++t){
                    cuda(cudaMemcpy(input,full_golden::inputs[t].data(),12,cudaMemcpyHostToDevice));cpu.step(full_golden::inputs[t],expected);
                    gpu.step_device({input,3},{output,3});cuda(cudaMemcpy(actual.data(),output,12,cudaMemcpyDeviceToHost));
                    check(gpu.valid()&&gpu.tokens()==std::size_t(t+1),"full_progress");gpu.read_state(keys,values);gpu.read_intermediates(probabilities,core,gated);
                    exact(actual,expected,"full_cpu_output");exact(keys,cpu.keys(),"full_cpu_keys");exact(values,cpu.values(),"full_cpu_values");
                    exact(actual,full_golden::residual[t],"full_golden_residual");exact(keys,full_golden::keys[t],"full_golden_keys");exact(values,full_golden::values[t],"full_golden_values");
                    exact(probabilities,full_golden::probabilities[t],"full_golden_probabilities");exact(core,full_golden::core[t],"full_golden_core");exact(gated,full_golden::gated[t],"full_golden_gated");
                }
                if(stage==2){bool caught=false;try{gpu.step_device({input,3},{output,3});}catch(const std::invalid_argument&){caught=true;}check(caught&&gpu.valid()&&gpu.tokens()==4,"full_capacity_rejection");}
                gpu.reset();cpu.reset();gpu.read_state(keys,values);check(gpu.valid()&&gpu.tokens()==0,"full_reset");
                for(auto x:keys)check(x==0,"full_reset_keys");for(auto x:values)check(x==0,"full_reset_values");
            }
        }else{
            cuda(cudaMemcpy(input,full_golden::inputs[0].data(),12,cudaMemcpyHostToDevice));bool caught=false;
            try{gpu.step_device({input,3},{output,3});}catch(const std::overflow_error&){caught=true;}
            check(caught&&!gpu.valid()&&gpu.tokens()==0,"full_missing_numeric_invalidation");
            caught=false;try{gpu.step_device({input,3},{output,3});}catch(const std::invalid_argument&){caught=true;}check(caught,"full_invalid_retry");
            gpu.reset();gpu.read_state(keys,values);check(gpu.valid()&&gpu.tokens()==0,"full_reset_after_error");
            for(auto x:keys)check(x==0,"full_reset_keys");for(auto x:values)check(x==0,"full_reset_values");
        }
        gpu.close();gpu.close();check(resources->snapshot().used[device+1]==256,"full_owner_cleanup");
        resources->unpin(handle);resources->begin_eviction(handle);cuda(cudaFree(storage));resources->released(handle);
        check(resources->snapshot().used[device+1]==0&&resources->snapshot().residents==0,"full_final_cleanup");
        std::cout<<"Passed full attention stage "<<stage<<": "<<(stage==3?"expected numerical invalidation":"independent KV/probability/core/gated/residual goldens")<<", reset and zero-ledger cleanup; 7424 requested device bytes. No model or benchmark.\n";
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
    // Dedicated fail-stop process; no retries, GPU resets or automatic stage chaining.
}
