#include "decoder_fixture.hpp"
#include "kadan/cuda_decoder.hpp"
#include <cuda_runtime_api.h>
#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstring>
#include <iostream>
#include <string_view>
namespace {
void check(bool x,const char* e){if(!x)throw std::runtime_error(e);}
void cuda(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
int integer(const char*s){std::string_view v(s);int n=-1;auto[end,error]=std::from_chars(v.data(),v.data()+v.size(),n);check(error==std::errc{}&&end==v.data()+v.size(),"argument");return n;}
template<class A,class B>void exact(const A&a,const B&b,const char*e){check(a.size()==b.size()&&std::equal(a.begin(),a.end(),b.begin()),e);}
}
int main(int argc,char**argv){
    if(argc!=6||std::string_view(argv[1])!="--allow-gpu-validation"||std::string_view(argv[2])!="--device"||std::string_view(argv[4])!="--stage")return 2;
    try{
        const auto device=integer(argv[3]),stage=integer(argv[5]);check(device>=0&&device<64&&stage>=1&&stage<=3,"argument");cuda(cudaSetDevice(device));
        for(auto kind:{kadan::decoder::Attention::linear,kadan::decoder::Attention::full}){
            DecoderFixture fixture;const auto c=fixture.config(kind);const auto p=kadan::decoder::plan(c);const bool linear=kind==kadan::decoder::Attention::linear;
            if(stage==3)for(std::size_t e=0;e<4;++e){fixture.moe.globals[3*e]=1e20f;fixture.moe.globals[3*e+1]=1e20f;}
            kadan::Footprint cap(device+2,0);cap[0]=131072;cap[device+1]=65536;auto resources=std::make_shared<kadan::Resources>(cap);
            auto io=cap;io[0]=0;io[device+1]=256;auto handle=resources->reserve(kadan::Workload::llm,io);void* raw=nullptr;cuda(cudaMalloc(&raw,256));resources->loaded(handle);resources->pin(handle);auto*x=static_cast<float*>(raw);auto*y=x+16;
            kadan::cuda::Decoder decoder(c,fixture.weights(),device,resources);check(resources->snapshot().used[0]==decoder.host_metadata_bytes()&&resources->snapshot().used[device+1]==p.device_bytes+256&&resources->snapshot().residents==2,"admission");
            std::array<float,16> actual{},a{},u{},m{};std::vector<std::uint8_t> first(p.state_first_bytes),second(p.state_second_bytes);
            if(stage<=2){
                for(int replay=0;replay<(stage==1?1:2);++replay){for(int t=0;t<(stage==1?1:3);++t){
                    cuda(cudaMemcpy(x,decoder_golden::inputs[t].data(),64,cudaMemcpyHostToDevice));decoder.step_device({x,16},{y,16});cuda(cudaMemcpy(actual.data(),y,64,cudaMemcpyDeviceToHost));check(decoder.valid()&&decoder.tokens()==std::size_t(t+1),"publication");
                    decoder.read_intermediates(a,u,m);exact(a,linear?decoder_golden::linear_attention[t]:decoder_golden::full_attention[t],"attention_golden");exact(u,linear?decoder_golden::linear_normalized[t]:decoder_golden::full_normalized[t],"postnorm_golden");exact(m,linear?decoder_golden::linear_mixture[t]:decoder_golden::full_mixture[t],"mixture_golden");exact(actual,linear?decoder_golden::linear_output[t]:decoder_golden::full_output[t],"decoder_golden");
                    decoder.read_state(first,second);
                    if(linear){check(std::memcmp(first.data(),decoder_golden::linear_first[t].data(),first.size())==0,"convolution_golden");for(std::size_t j=0;j<second.size()/4;++j){float v;std::memcpy(&v,second.data()+4*j,4);check(std::abs(v-decoder_golden::linear_second[t][j])<=2e-5f,"recurrent_golden");}}
                    else{check(std::memcmp(first.data(),decoder_golden::full_first[t].data(),first.size())==0,"keys_golden");check(std::memcmp(second.data(),decoder_golden::full_second[t].data(),second.size())==0,"values_golden");}
                }decoder.reset();check(decoder.valid()&&decoder.tokens()==0,"reset");}
            }else{
                actual.fill(42);cuda(cudaMemcpy(y,actual.data(),64,cudaMemcpyHostToDevice));cuda(cudaMemcpy(x,decoder_golden::inputs[0].data(),64,cudaMemcpyHostToDevice));
                bool failed=false;try{decoder.step_device({x,16},{y,16});}catch(const std::overflow_error&){failed=true;}
                check(failed&&!decoder.valid()&&decoder.tokens()==0,"late_failure_publication");cuda(cudaMemcpy(actual.data(),y,64,cudaMemcpyDeviceToHost));for(float v:actual)check(v==42,"failed_output_changed");
                failed=false;try{decoder.step_device({x,16},{y,16});}catch(const std::invalid_argument&){failed=true;}check(failed,"failed_reuse");decoder.reset();check(decoder.valid()&&decoder.tokens()==0,"failure_reset");
                std::atomic_bool cancelled=true;failed=false;try{decoder.step_device({x,16},{y,16},&cancelled);}catch(const std::invalid_argument&){failed=true;}check(failed&&!decoder.valid()&&decoder.tokens()==0,"cancel_publication");decoder.reset();
            }
            decoder.read_state(first,second);for(auto v:first)check(v==0,"reset_first_zero");for(auto v:second)check(v==0,"reset_second_zero");
            bool stale=false;try{decoder.read_intermediates(a,u,m);}catch(const std::invalid_argument&){stale=true;}check(stale,"stale_diagnostic");
            decoder.close();decoder.close();check(!decoder.valid()&&resources->snapshot().used[0]==0&&resources->snapshot().used[device+1]==256,"owner_cleanup");resources->unpin(handle);resources->begin_eviction(handle);cuda(cudaFree(raw));resources->released(handle);check(resources->snapshot().residents==0&&resources->snapshot().used[device+1]==0,"cleanup");
            std::cout<<(linear?"linear":"full")<<" decoder stage "<<stage<<" passed; "<<p.device_bytes+256<<" peak requested device bytes, "<<decoder.host_metadata_bytes()<<" charged owner RAM bytes; zero ledger.\n";
        }
    }catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}
}
