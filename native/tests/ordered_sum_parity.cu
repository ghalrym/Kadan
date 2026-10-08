#include "ordered_sum.cuh"
#include <bit>
#include <charconv>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <vector>
__global__ void compare(const float* x,std::size_t n,float* y,unsigned* flags){
    if(threadIdx.x==0){float s=0;for(std::size_t j=0;j<n;++j){s=__fadd_rn(s,x[j]);if(!isfinite(s)){atomicOr(flags,1u);s=0;}}y[0]=s;}
    if(threadIdx.x==1)y[1]=kadan::cuda::detail::ordered_finite_sum(x,n,flags+1);
}
void ck(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
int main(int argc,char**argv){
 if(argc!=4||std::string_view(argv[1])!="--allow-gpu-validation"||std::string_view(argv[2])!="--device")return 2;
 try{int device=-1;std::string_view arg(argv[3]);auto[end,error]=std::from_chars(arg.data(),arg.data()+arg.size(),device);if(error!=std::errc{}||end!=arg.data()+arg.size()||device<0||device>=64)return 2;ck(cudaSetDevice(device));
 float *x,*y;unsigned*flags;ck(cudaMalloc(&x,4097*4));ck(cudaMalloc(&y,8));ck(cudaMalloc(&flags,8));unsigned cases=0,seed=123;
 for(unsigned n:{0u,1u,3u,2048u,4096u,4097u})for(unsigned mode=0;mode<9;++mode)for(unsigned initial:{0u,2u}){
 std::vector<float> input(n);for(auto&v:input){seed=1664525*seed+1013904223;v=float(int(seed>>16)-32768)/8192.f;}
 for(unsigned j=0;j<n;++j){if(mode==1)input[j]=j%2?0.f:-0.f;if(mode==2)input[j]=std::numeric_limits<float>::denorm_min();if(mode==3)input[j]=std::numeric_limits<float>::max();if(mode==4)input[j]=(j%4<2?1.f:-1.f)*std::numeric_limits<float>::max();}
 if(n&&mode>=5)input[n/2]=mode==5?INFINITY:mode==6?-INFINITY:NAN;
 if(n>1&&mode==8){input[0]=INFINITY;input[1]=-INFINITY;}
 unsigned f[2]={initial,initial};float result[2];ck(cudaMemcpy(x,input.data(),n*4,cudaMemcpyHostToDevice));ck(cudaMemcpy(flags,f,8,cudaMemcpyHostToDevice));compare<<<1,32>>>(x,n,y,flags);ck(cudaGetLastError());ck(cudaDeviceSynchronize());ck(cudaMemcpy(result,y,8,cudaMemcpyDeviceToHost));ck(cudaMemcpy(f,flags,8,cudaMemcpyDeviceToHost));if(std::memcmp(result,result+1,4)||f[0]!=f[1])throw std::runtime_error("ordered-sum bitwise output/status mismatch");++cases;}
 ck(cudaFree(x));ck(cudaFree(y));ck(cudaFree(flags));std::cout<<"PASS: "<<cases<<" ordered-sum cases: zero/odd/full lengths, random finite, signed zero, subnormal, intermediate overflow/cancellation, Inf/NaN, pre-existing flags\n";
 }catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
