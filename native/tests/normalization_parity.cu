#include "linear_normalization_reference.cuh"
#include "full_normalization_reference.cuh"
#include "decoder_normalization_reference.cuh"
#include <bit>
#include <charconv>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <vector>
namespace {void ck(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}}
int main(int argc,char**argv){
 if(argc!=4||std::string_view(argv[1])!="--allow-gpu-validation"||std::string_view(argv[2])!="--device")return 2;
 try{int device=-1;std::string_view arg(argv[3]);auto[end,error]=std::from_chars(arg.data(),arg.data()+arg.size(),device);if(error!=std::errc{}||end!=arg.data()+arg.size()||device<0||device>=64)return 2;ck(cudaSetDevice(device));unsigned cases=0;
 for(unsigned n:{1u,3u,2048u,4096u,4097u}){float*x,*a,*b;std::uint16_t*w;unsigned*flags;ck(cudaMalloc(&x,n*4));ck(cudaMalloc(&a,n*4));ck(cudaMalloc(&b,n*4));ck(cudaMalloc(&w,n*2));ck(cudaMalloc(&flags,8));std::vector<float> input(n),actual(n),expected(n);std::vector<std::uint16_t> weights(n);for(unsigned j=0;j<n;++j)weights[j]=std::uint16_t(std::bit_cast<unsigned>(float(int(j%31)-15)/1024.f)>>16);ck(cudaMemcpy(w,weights.data(),n*2,cudaMemcpyHostToDevice));
 for(int mode=0;mode<6;++mode){for(unsigned j=0;j<n;++j)input[j]=float(int(j%257)-128)/1024.f;
 if(mode==1)for(auto&v:input)v=0.f;if(mode==2)input[0]=.1f;if(mode==3)input[0]=INFINITY;if(mode==4)for(auto&v:input)v=std::bit_cast<float>(0x7f7f0000u);if(mode==5)for(unsigned j=0;j<n;++j)input[j]=j%2?0.f:-0.f;ck(cudaMemcpy(x,input.data(),n*4,cudaMemcpyHostToDevice));
 for(int family=0;family<3;++family){ck(cudaMemset(flags,0,8));
 if(family==0){kadan::linear::Config c{n,1,1,1,1,2,1,1e-6f};kadan::cuda::detail::LinearBuffers l{},r{};l.input_norm=r.input_norm=w;l.normalized=a;r.normalized=b;l.status=flags;r.status=flags+1;ck(baseline_linear::linear_normalize(c,l,x));ck(kadan::cuda::detail::linear_normalize(c,r,x));}
 if(family==1){kadan::full::Config c{n,1,1,2,2,1,1e-6f};kadan::cuda::detail::FullBuffers l{},r{};l.input_norm=r.input_norm=w;l.normalized=a;r.normalized=b;l.status=flags;r.status=flags+1;ck(baseline_full::full_normalize(c,l,x));ck(kadan::cuda::detail::full_normalize(c,r,x));}
 if(family==2){ck(baseline_decoder::decoder_norm(n,1e-6f,w,x,a,flags));ck(kadan::cuda::detail::decoder_norm(n,1e-6f,w,x,b,flags+1));}
 ck(cudaDeviceSynchronize());ck(cudaMemcpy(expected.data(),a,n*4,cudaMemcpyDeviceToHost));ck(cudaMemcpy(actual.data(),b,n*4,cudaMemcpyDeviceToHost));unsigned f[2];ck(cudaMemcpy(f,flags,8,cudaMemcpyDeviceToHost));if(std::memcmp(expected.data(),actual.data(),n*4)||f[0]!=f[1])throw std::runtime_error("normalization bitwise parity");const bool invalid=mode==3||mode==4||(mode==2&&family!=2);if(invalid!=(f[1]!=0))throw std::runtime_error("normalization expected status");++cases;}}
 ck(cudaFree(x));ck(cudaFree(a));ck(cudaFree(b));ck(cudaFree(w));ck(cudaFree(flags));}
 std::cout<<"PASS: "<<cases<<" bitwise normalization cases across linear/full/decoder, actual width, shared/fallback boundary, non-BF16/nonfinite/overflow/signed zero\n";
 }catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}
}
