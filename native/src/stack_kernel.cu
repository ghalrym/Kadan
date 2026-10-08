#include "stack_kernel.cuh"
#include <cuda_runtime.h>
namespace kadan::cuda::detail {
namespace {
__global__ void embedding(std::size_t n,const std::uint16_t* table,unsigned token,float* out){auto j=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(j<n)out[j]=__uint_as_float(unsigned(table[std::size_t(token)*n+j])<<16);}
__global__ void select(std::size_t n,float* logits,unsigned* result,unsigned* status){
    if(threadIdx.x)return;unsigned best=0;
    for(unsigned j=0;j<n;++j){float v=logits[j];if(!isfinite(v)){atomicOr(status,1u);v=0;}unsigned bits=__float_as_uint(v);bits+=0x7fffu+((bits>>16)&1u);v=__uint_as_float(bits&0xffff0000u);if(!isfinite(v)){atomicOr(status,1u);v=0;}logits[j]=v;if(j&&v>logits[best])best=j;}
    *result=best; // Ascending iteration and strict greater-than break ties by ID.
}
}
cudaError_t stack_embedding(std::size_t n,const std::uint16_t*w,unsigned t,float*y){embedding<<<(n+127)/128,128,0,cudaStreamLegacy>>>(n,w,t,y);return cudaGetLastError();}
cudaError_t stack_select(std::size_t n,float*x,unsigned*y,unsigned*s){select<<<1,1,0,cudaStreamLegacy>>>(n,x,y,s);return cudaGetLastError();}
}
