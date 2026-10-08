#include "decoder_kernel.cuh"
#include <cuda_runtime.h>
namespace kadan::cuda::detail {
namespace {
__device__ float finite(float x,unsigned* s){if(!isfinite(x)){atomicOr(s,1u);return 0;}return x;}
__device__ float bf(float x,unsigned* s){unsigned u=__float_as_uint(finite(x,s));u+=0x7fffu+((u>>16)&1u);return finite(__uint_as_float(u&0xffff0000u),s);}
__global__ void norm(std::size_t n,float epsilon,const std::uint16_t* w,const float* x,float* y,unsigned* s){
    if(threadIdx.x)return;float sum=0;for(std::size_t j=0;j<n;++j)sum=finite(__fadd_rn(sum,__fmul_rn(x[j],x[j])),s);
    const float inv=1/sqrtf(__fadd_rn(sum/float(n),epsilon));
    for(std::size_t j=0;j<n;++j)y[j]=bf(__fmul_rn(__fmul_rn(x[j],inv),__fadd_rn(1,__uint_as_float(unsigned(w[j])<<16))),s);
}
__global__ void residual(std::size_t n,const float* a,const float* m,float* y,unsigned* s){const auto j=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(j<n)y[j]=bf(__fadd_rn(a[j],m[j]),s);}
}
cudaError_t decoder_norm(std::size_t n,float e,const std::uint16_t* w,const float* x,float* y,unsigned* s){norm<<<1,1,0,cudaStreamLegacy>>>(n,e,w,x,y,s);return cudaGetLastError();}
cudaError_t decoder_residual(std::size_t n,const float* a,const float* m,float* y,unsigned* s){residual<<<(n+127)/128,128,0,cudaStreamLegacy>>>(n,a,m,y,s);return cudaGetLastError();}
}
