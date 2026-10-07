#include "linear_fixture.hpp"
#include "kadan/cuda_linear_attention.hpp"
#include <cuda_runtime_api.h>
#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string_view>
namespace {
void check(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}
void cuda(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
int integer(const char* s){const std::string_view v(s);int n=-1;auto [end,error]=std::from_chars(v.data(),v.data()+v.size(),n);check(error==std::errc{}&&end==v.data()+v.size(),"invalid_argument");return n;}
}
int main(int argc,char** argv){
    if(argc!=6 || std::string_view(argv[1])!="--allow-gpu-validation" || std::string_view(argv[2])!="--device" || std::string_view(argv[4])!="--stage")return 2;
    try{
        const int device=integer(argv[3]),stage=integer(argv[5]);check(device>=0&&device<64&&stage>=1&&stage<=3,"invalid_device_or_stage");
        cuda(cudaSetDevice(device));kadan::Footprint cap(device+2,0);cap[device+1]=65536;
        auto resources=std::make_shared<kadan::Resources>(cap);auto io=cap;io[device+1]=256;
        const auto handle=resources->reserve(kadan::Workload::llm,io);void* storage=nullptr;cuda(cudaMalloc(&storage,256));resources->loaded(handle);resources->pin(handle);
        auto* input=static_cast<float*>(storage);auto* output=input+3;
        LinearFixture fixture;
        if(stage==2)fixture=AsymmetricLinearFixture{};
        if(stage==3)fixture.logs[0]=128;
        kadan::cuda::LinearAttention gpu(fixture.config,fixture.weights(),device,resources);
        const std::size_t requested=stage==2?5632:5376;
        check(resources->snapshot().used[device+1]==requested,"unexpected_tiny_admission");
        kadan::linear::Reference cpu(fixture.config,fixture.weights(),4096);
        const auto p=kadan::linear::plan(fixture.config);const auto hidden=fixture.config.hidden;
        std::vector<std::vector<float>> tokens{{1,1},{1,-1},{-1,1}};
        if(stage==2){tokens.clear();for(auto token:linear_golden::inputs)tokens.emplace_back(token.begin(),token.end());}
        std::vector<float> expected(hidden),actual(hidden),state(p.recurrent_elements),core(p.values),gated(p.values);
        std::vector<std::uint16_t> conv(p.conv_elements);
        const auto io_bytes=hidden*sizeof(float);
        if(stage<=2){
            const int replays=stage==1?1:2,count=stage==1?1:4;
            for(int replay=0;replay<replays;++replay){
                for(int t=0;t<count;++t){
                    cuda(cudaMemcpy(input,tokens[t].data(),io_bytes,cudaMemcpyHostToDevice));cpu.step(tokens[t],expected);
                    gpu.step_device({input,hidden},{output,hidden});cuda(cudaMemcpy(actual.data(),output,io_bytes,cudaMemcpyDeviceToHost));
                    check(actual==expected,"linear_output_parity");check(gpu.valid()&&gpu.tokens()==std::size_t(t+1),"linear_progress");
                    gpu.read_state(conv,state);check(std::equal(conv.begin(),conv.end(),cpu.convolution().begin()),"linear_convolution_parity");
                    for(std::size_t j=0;j<state.size();++j)check(std::abs(state[j]-cpu.recurrent()[j])<=2e-5f,"linear_recurrent_parity");
                    if(stage==2){
                        gpu.read_intermediates(core,gated);
                        check(std::equal(actual.begin(),actual.end(),linear_golden::residual[t].begin()),"golden_residual");
                        check(std::equal(conv.begin(),conv.end(),linear_golden::history[t].begin()),"golden_history");
                        check(std::equal(core.begin(),core.end(),linear_golden::core[t].begin()),"golden_core");
                        check(std::equal(gated.begin(),gated.end(),linear_golden::gated[t].begin()),"golden_gated");
                        for(std::size_t j=0;j<state.size();++j)check(std::abs(state[j]-linear_golden::state[t][j])<=2e-5f,"golden_state");
                    }
                }
                gpu.reset();cpu.reset();gpu.read_state(conv,state);check(gpu.tokens()==0,"linear_reset_progress");
                for(auto x:conv)check(x==0,"linear_reset_conv");for(auto x:state)check(x==0,"linear_reset_state");
            }
        }else{
            cuda(cudaMemcpy(input,tokens[0].data(),io_bytes,cudaMemcpyHostToDevice));bool caught=false;
            try{gpu.step_device({input,hidden},{output,hidden});}catch(const std::overflow_error&){caught=true;}
            check(caught&&!gpu.valid()&&gpu.tokens()==0,"linear_missing_invalidation");
            caught=false;try{gpu.step_device({input,hidden},{output,hidden});}catch(const std::invalid_argument&){caught=true;}check(caught,"linear_invalid_retry");
            gpu.reset();gpu.read_state(conv,state);for(auto x:conv)check(x==0,"linear_reset_conv");for(auto x:state)check(x==0,"linear_reset_state");
        }
        gpu.close();gpu.close();check(resources->snapshot().used[device+1]==256,"linear_owner_cleanup");
        resources->unpin(handle);resources->begin_eviction(handle);cuda(cudaFree(storage));resources->released(handle);
        check(resources->snapshot().used[device+1]==0&&resources->snapshot().residents==0,"linear_final_cleanup");
        std::cout<<"Passed linear sublayer stage "<<stage<<": hidden-to-residual path, state checks and zero-ledger cleanup; "<<requested<<" requested device bytes. No model or benchmark.\n";
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
    // Unexpected error exits this dedicated fixture process; no rerun/device reset.
}
