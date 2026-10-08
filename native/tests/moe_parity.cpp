#include "moe_fixture.hpp"
#include "kadan/cuda_moe.hpp"
#include <cuda_runtime_api.h>
#include <algorithm>
#include <charconv>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string_view>
namespace {
void check(bool x,const char* e){if(!x)throw std::runtime_error(e);}
void cuda(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
int integer(const char* s){std::string_view v(s);int n=-1;auto [end,error]=std::from_chars(v.data(),v.data()+v.size(),n);check(error==std::errc{}&&end==v.data()+v.size(),"invalid_argument");return n;}
template<class A,class B>void exact(const A& a,const B& b,const char* error){check(a.size()==b.size()&&std::equal(a.begin(),a.end(),b.begin()),error);}
}
int main(int argc,char** argv){
    if(argc!=6||std::string_view(argv[1])!="--allow-gpu-validation"||std::string_view(argv[2])!="--device"||std::string_view(argv[4])!="--stage")return 2;
    try{
        const int device=integer(argv[3]),stage=integer(argv[5]);check(device>=0&&device<64&&stage>=1&&stage<=3,"invalid_device_or_stage");cuda(cudaSetDevice(device));
        kadan::Footprint cap(device+2,0);cap[0]=131072;cap[device+1]=65536;auto resources=std::make_shared<kadan::Resources>(cap);
        auto io=cap;io[0]=0;io[device+1]=256;const auto handle=resources->reserve(kadan::Workload::llm,io);void* storage=nullptr;cuda(cudaMalloc(&storage,256));resources->loaded(handle);resources->pin(handle);
        auto* input=static_cast<float*>(storage);auto* output=input+16;MoeFixture fixture;
        if(stage==3)for(std::size_t e=0;e<4;++e){fixture.globals[3*e]=1e20f;fixture.globals[3*e+1]=1e20f;}
        kadan::cuda::Moe gpu(fixture.config,fixture.weights(),device,resources);kadan::moe::Reference cpu(fixture.config,fixture.weights(),692);
        check(resources->snapshot().used[device+1]==9728&&resources->snapshot().used[0]==gpu.host_metadata_bytes()&&resources->snapshot().residents==2,"moe_arena_admission");
        std::array<float,16> actual{},expected{},routed{},shared{},result{};std::array<unsigned,2> selected{};
        std::array<float,4> logits{},probs{};std::array<float,2> weights{};
        if(stage<=2){const int replays=stage==1?1:2,count=stage==1?1:5;
            for(int replay=0;replay<replays;++replay){for(int t=0;t<count;++t){
                cuda(cudaMemcpy(input,moe_golden::inputs[t].data(),64,cudaMemcpyHostToDevice));cpu.forward(moe_golden::inputs[t],expected);
                gpu.forward_device({input,16},{output,16});cuda(cudaMemcpy(actual.data(),output,64,cudaMemcpyDeviceToHost));check(gpu.valid(),"moe_health");
                gpu.read_routes(selected,logits,probs,weights);gpu.read_outputs(routed,shared,result);
                exact(actual,expected,"moe_cpu_output");exact(actual,moe_golden::result[t],"moe_golden_output");exact(result,actual,"moe_result");
                exact(selected,moe_golden::selected[t],"moe_selected");exact(logits,moe_golden::logits[t],"moe_logits");exact(weights,moe_golden::top_weights[t],"moe_weights");
                exact(routed,moe_golden::routed[t],"moe_routed");exact(shared,moe_golden::shared[t],"moe_shared");
                for(std::size_t j=0;j<4;++j)check(std::abs(probs[j]-moe_golden::probabilities[t][j])<=2e-6f,"moe_probabilities");
            }gpu.reset();cpu.reset();check(gpu.valid(),"moe_reset");bool rejected=false;try{gpu.read_outputs(routed,shared,result);}catch(const std::invalid_argument&){rejected=true;}check(rejected,"moe_stale_diagnostics");}
        }else{
            actual.fill(42);cuda(cudaMemcpy(output,actual.data(),64,cudaMemcpyHostToDevice));cuda(cudaMemcpy(input,moe_golden::inputs[0].data(),64,cudaMemcpyHostToDevice));
            bool caught=false;try{gpu.forward_device({input,16},{output,16});}catch(const std::overflow_error&){caught=true;}check(caught&&!gpu.valid(),"moe_missing_numeric_invalidation");
            cuda(cudaMemcpy(actual.data(),output,64,cudaMemcpyDeviceToHost));for(auto v:actual)check(v==42,"moe_failed_output_mutated");
            caught=false;try{gpu.forward_device({input,16},{output,16});}catch(const std::invalid_argument&){caught=true;}check(caught,"moe_invalid_retry");gpu.reset();check(gpu.valid(),"moe_reset_after_failure");
        }
        gpu.close();gpu.close();check(resources->snapshot().used[0]==0&&resources->snapshot().used[device+1]==256&&resources->snapshot().residents==1,"moe_owner_cleanup");
        resources->unpin(handle);resources->begin_eviction(handle);cuda(cudaFree(storage));resources->released(handle);
        check(resources->snapshot().used[device+1]==0&&resources->snapshot().residents==0,"moe_final_cleanup");
        std::cout<<"Passed resident MoE stage "<<stage<<": "<<(stage==3?"expected activation overflow/invalidation":"independent route/routed/shared/output goldens")<<", reset and zero-ledger cleanup; 9728 requested device bytes, "<<gpu.host_metadata_bytes()<<" charged owner RAM bytes. No model or benchmark.\n";
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
    // Fail-stop dedicated process; no retries, GPU resets or automatic stage chaining.
}
