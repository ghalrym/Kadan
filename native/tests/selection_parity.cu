#include "selection_reference.cuh"
#include <bit>
#include <charconv>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <vector>
namespace {void ck(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}}
int main(int argc,char**argv){
 if(argc!=4||std::string_view(argv[1])!="--allow-gpu-validation"||std::string_view(argv[2])!="--device")return 2;
 try{int device=-1;std::string_view arg(argv[3]);auto[end,error]=std::from_chars(arg.data(),arg.data()+arg.size(),device);if(error!=std::errc{}||end!=arg.data()+arg.size()||device<0||device>=64)return 2;ck(cudaSetDevice(device));unsigned cases=0;
 for(unsigned n:{1u,31u,255u,256u,257u,248320u}){float *x,*y;unsigned*state;ck(cudaMalloc(&x,n*4));ck(cudaMalloc(&y,n*4));ck(cudaMalloc(&state,16));std::vector<float> input(n),a(n),b(n);
 for(int mode=0;mode<6;++mode){for(unsigned i=0;i<n;++i)input[i]=float(int((i*137)%701)-350)/16.f;
 if(mode==1)for(auto&v:input)v=1.003f;
 if(mode==2){for(unsigned i=0;i<n;++i)input[i]=i%4==0?INFINITY:(i%4==1?-INFINITY:(i%4==2?std::numeric_limits<float>::max():std::numeric_limits<float>::quiet_NaN()));}
 if(mode==3)for(auto&v:input)v=std::numeric_limits<float>::quiet_NaN();
 if(mode==4)for(unsigned i=0;i<n;++i)input[i]=i%2?0.f:-0.f;
 if(mode==5){for(auto&v:input)v=0.f;input.back()=2.f;}
 ck(cudaMemcpy(x,input.data(),n*4,cudaMemcpyHostToDevice));ck(cudaMemcpy(y,input.data(),n*4,cudaMemcpyHostToDevice));ck(cudaMemset(state,0,16));ck(baseline::stack_select(n,x,state,state+2));ck(kadan::cuda::detail::stack_select(n,y,state+1,state+3));ck(cudaDeviceSynchronize());ck(cudaMemcpy(a.data(),x,n*4,cudaMemcpyDeviceToHost));ck(cudaMemcpy(b.data(),y,n*4,cudaMemcpyDeviceToHost));unsigned s[4];ck(cudaMemcpy(s,state,16,cudaMemcpyDeviceToHost));if(std::memcmp(a.data(),b.data(),n*4)||s[0]!=s[1]||s[2]!=s[3])throw std::runtime_error("selection bitwise parity");if((mode==2||mode==3)!=(s[3]!=0))throw std::runtime_error("selection status");if((mode==1||mode==2||mode==3||mode==4)&&s[1]!=0)throw std::runtime_error("smallest ID tie");if(mode==5&&s[1]!=n-1)throw std::runtime_error("last ID");++cases;}
 ck(cudaFree(x));ck(cudaFree(y));ck(cudaFree(state));}
 std::cout<<"PASS: "<<cases<<" bitwise selection cases: tails/full vocabulary, BF16 ties, signed zeros, nonfinite/overflow and final-ID maximum\n";
 }catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}
}
