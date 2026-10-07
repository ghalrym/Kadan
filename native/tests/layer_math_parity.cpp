// Opt-in tiny original-kernel parity only. Never registered with CTest.
#include "kadan/cuda_layer_math.hpp"
#include "kadan/resources.hpp"
#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string_view>
#include <vector>
namespace {
void check(bool ok,const char* reason){if(!ok)throw std::runtime_error(reason);}
void cuda(cudaError_t error){if(error!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(error));}
}
int main(int argc,char** argv) {
    if(argc!=4 || std::string_view(argv[1])!="--allow-gpu-validation" || std::string_view(argv[2])!="--device")return 2;
    try {
        int device=-1;const std::string_view arg(argv[3]);const auto [end,error]=std::from_chars(arg.data(),arg.data()+arg.size(),device);
        check(error==std::errc{} && end==arg.data()+arg.size() && device>=0 && device<64,"invalid_device");
        cuda(cudaSetDevice(device));int major=0,minor=0;
        cuda(cudaDeviceGetAttribute(&major,cudaDevAttrComputeCapabilityMajor,device));cuda(cudaDeviceGetAttribute(&minor,cudaDevAttrComputeCapabilityMinor,device));
        check(major==8 && minor==6,"requires_sm86");
        constexpr std::size_t count=2048,bytes=4*count*sizeof(float)+256;
        kadan::Footprint capacity(device+2,0);capacity[device+1]=65536;
        auto resources=std::make_shared<kadan::Resources>(capacity);auto request=capacity;request[device+1]=bytes;
        const auto handle=resources->reserve(kadan::Workload::llm,request);
        void* storage=nullptr;cuda(cudaMalloc(&storage,bytes));resources->loaded(handle);resources->pin(handle);
        auto* a=static_cast<float*>(storage);auto* b=a+count;auto* c=b+count;auto* out=c+count;
        auto* status=reinterpret_cast<unsigned*>(out+count);
        std::vector<float> x(count),w(count),z(count),expected(count),actual(count);
        auto upload=[](float* dst,std::span<const float> src){cuda(cudaMemcpy(dst,src.data(),src.size_bytes(),cudaMemcpyHostToDevice));};
        auto clear=[&]{cuda(cudaMemsetAsync(status,0,sizeof(unsigned),cudaStreamLegacy));};
        std::size_t launches=0;
        auto finish=[&](float* result,std::size_t n,bool invalid=false) {
            cuda(cudaStreamSynchronize(cudaStreamLegacy));unsigned flags=0;cuda(cudaMemcpy(&flags,status,sizeof(flags),cudaMemcpyDeviceToHost));
            if(invalid){check(flags!=0,"missing_math_error");return;}
            check(flags==0,"unexpected_math_error");cuda(cudaMemcpy(actual.data(),result,n*sizeof(float),cudaMemcpyDeviceToHost));
            for(std::size_t i=0;i<n;++i)check(std::isfinite(actual[i]) && std::abs(double(actual[i])-expected[i])<=3e-5*(1+std::abs(double(expected[i]))),"math_parity_failed");
        };
        auto host=[](auto& v,std::size_t n){return std::span(v).first(n);};
        for(std::size_t i=0;i<count;++i){x[i]=float(int(i%17)-8)/4;w[i]=float(int(i%7)-3)/8;z[i]=float(int(i%13)-6)/3;}
        upload(a,x);upload(b,w);upload(c,z);
        for(auto scale:{kadan::math::NormScale::one_plus,kadan::math::NormScale::direct}) {
            kadan::math::Norm shape{2,129,1e-6f,scale};kadan::math::rms_norm(shape,host(x,258),host(w,129),host(expected,258));
            clear();cuda(kadan::cuda::rms_norm(shape,{a,258},{b,129},{out,258},status));++launches;finish(out,258);
        }
        const kadan::math::Norm norm{1,count,1e-6f,kadan::math::NormScale::one_plus};
        kadan::math::rms_norm(norm,x,w,expected);clear();cuda(kadan::cuda::rms_norm(norm,{a,count},{b,count},{out,count},status));++launches;finish(out,count);
        for(auto kind:{kadan::math::Gate::sigmoid,kadan::math::Gate::silu}) {
            kadan::math::gate(kind,x,z,expected);clear();cuda(kadan::cuda::gate(kind,{a,count},{c,count},{out,count},status));++launches;finish(out,count);
        }
        // Two device-resident operations with no intermediate host copy.
        const kadan::math::Norm gated{1,128,1e-6f,kadan::math::NormScale::direct};
        kadan::math::rms_norm(gated,host(x,128),host(w,128),host(expected,128));
        kadan::math::gate(kadan::math::Gate::silu,host(expected,128),host(z,128),host(expected,128));
        clear();cuda(kadan::cuda::rms_norm(gated,{a,128},{b,128},{out,128},status));
        cuda(kadan::cuda::gate(kadan::math::Gate::silu,{out,128},{c,128},{out,128},status));launches+=2;finish(out,128);
        kadan::math::inverse_frequencies(10000000,host(w,32));upload(b,host(w,32));
        const kadan::math::Rotary rope{2,256,64,262144};
        for(std::size_t position:{0u,1u,262143u}) {
            kadan::math::rope(rope,position,host(w,32),host(x,512),host(expected,512));
            clear();cuda(kadan::cuda::rope(rope,position,{b,32},{a,512},{out,512},status));++launches;finish(out,512);
        }
        kadan::math::rope(rope,1,host(w,32),host(x,512),host(expected,512));
        clear();cuda(kadan::cuda::rope(rope,1,{b,32},{a,512},{a,512},status));++launches;finish(a,512);
        x[0]=std::numeric_limits<float>::quiet_NaN();upload(a,x);clear();
        cuda(kadan::cuda::rms_norm(gated,{a,128},{b,128},{out,128},status));++launches;finish(out,128,true);
        x[0]=std::numeric_limits<float>::max();z[0]=x[0];upload(a,x);upload(c,z);clear();
        cuda(kadan::cuda::gate(kadan::math::Gate::silu,{a,1},{c,1},{out,1},status));++launches;finish(out,1,true);
        w[0]=0;upload(b,host(w,32));clear();
        cuda(kadan::cuda::rope(rope,0,{b,32},{a,512},{out,512},status));++launches;finish(out,512,true);
        check(launches==14,"unexpected_launch_count");
        resources->unpin(handle);resources->begin_eviction(handle);cuda(cudaFree(storage));resources->released(handle);
        check(resources->snapshot().used[device+1]==0 && resources->snapshot().residents==0,"math_cleanup_failed");
        std::cout<<"Passed 14 bounded original layer-math launches, device chain, in-place RoPE, error flags and zero-ledger cleanup; 33024 requested device bytes. No model or benchmark.\n";
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
    // On unexpected failure the dedicated test process exits without retries;
    // its CUDA context tears down. This is not a production resource owner.
}
