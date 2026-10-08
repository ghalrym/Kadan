#include "stack_kernel.cuh"
#include <cuda_runtime.h>
#include <math_constants.h>
namespace kadan::cuda::detail {
namespace {
__global__ void embedding(std::size_t n,const std::uint16_t* table,unsigned token,float* out){auto j=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(j<n)out[j]=__uint_as_float(unsigned(table[std::size_t(token)*n+j])<<16);}
// BF16 conversion is per element. Max with lowest-ID tie breaking is
// associative, so the reduction changes no accumulation/rounding order.
__global__ void select(std::size_t n,float* logits,unsigned* result,unsigned* status){
    const unsigned tid=threadIdx.x;float best=-CUDART_INF_F;unsigned id=~0u;
    for(unsigned j=tid;j<n;j+=blockDim.x){float v=logits[j];
        if(!isfinite(v)){atomicOr(status,1u);v=0;}
        unsigned bits=__float_as_uint(v);bits+=0x7fffu+((bits>>16)&1u);v=__uint_as_float(bits&0xffff0000u);
        if(!isfinite(v)){atomicOr(status,1u);v=0;}logits[j]=v;
        if(v>best||(v==best&&j<id)){best=v;id=j;}
    }
    __shared__ float values[256];__shared__ unsigned ids[256];values[tid]=best;ids[tid]=id;__syncthreads();
    for(unsigned stride=128;stride;stride>>=1){if(tid<stride){const float other=values[tid+stride];const unsigned other_id=ids[tid+stride];
        if(other>values[tid]||(other==values[tid]&&other_id<ids[tid])){values[tid]=other;ids[tid]=other_id;}}
        __syncthreads();}
    if(tid==0)*result=ids[0];

}
}
cudaError_t stack_embedding(std::size_t n,const std::uint16_t*w,unsigned t,float*y){embedding<<<(n+127)/128,128,0,cudaStreamLegacy>>>(n,w,t,y);return cudaGetLastError();}
cudaError_t stack_select(std::size_t n,float*x,unsigned*y,unsigned*s){select<<<1,256,0,cudaStreamLegacy>>>(n,x,y,s);return cudaGetLastError();}
}
