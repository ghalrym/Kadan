#include "moe_kernel.cuh"
#include <cuda_runtime.h>
#include <math_constants.h>
namespace kadan::cuda::detail {
namespace {
__device__ float expand(std::uint16_t x){return __uint_as_float(unsigned(x)<<16);}
__device__ float finite(float x,unsigned* s){if(!isfinite(x)){atomicOr(s,1u);return 0;}return x;}
__device__ float bf(float x,unsigned* s){unsigned v=__float_as_uint(finite(x,s));v+=0x7fffu+((v>>16)&1u);return finite(__uint_as_float(v&0xffff0000u),s);}
__device__ float add(float a,float b){return __fadd_rn(a,b);}
__device__ float mul(float a,float b){return __fmul_rn(a,b);}
__device__ float sigmoid(float x){const float e=expf(-fabsf(x));return x>=0?1/(1+e):e/(1+e);}
// Experts are independent. Keep the serial FP32 accumulation within each row
// (and its BF16 boundary) exactly as before, but schedule rows across SMs.
// A warp-sized block deliberately uses one lane: parallel reduction would change
// rounding and can change top-k routing. No additional arena storage is needed.
__global__ void route_logits(moe::Config c,MoeBuffers b,const float* x){
    if(threadIdx.x)return;const std::size_t e=blockIdx.x;float sum=0;
    for(std::size_t j=0;j<c.hidden;++j)sum=finite(add(sum,mul(expand(b.router[e*c.hidden+j]),x[j])),b.status);
    b.logits[e]=bf(sum,b.status);
}
__global__ void route(moe::Config c,MoeBuffers b,const float* x){
    if(threadIdx.x)return;float maximum=-CUDART_INF_F;
    for(std::size_t j=0;j<c.hidden;++j){if(bf(x[j],b.status)!=x[j])atomicOr(b.status,1u);b.accumulator[j]=0;}
    for(std::size_t e=0;e<c.experts;++e)maximum=fmaxf(maximum,b.logits[e]);
    float total=0;for(std::size_t e=0;e<c.experts;++e){b.probabilities[e]=expf(b.logits[e]-maximum);total=add(total,b.probabilities[e]);}
    for(std::size_t e=0;e<c.experts;++e)b.probabilities[e]/=total;
    float picked=0;for(std::size_t rank=0;rank<c.top_k;++rank){unsigned best=unsigned(c.experts);
        for(unsigned e=0;e<c.experts;++e){bool used=false;for(std::size_t j=0;j<rank;++j)used|=b.selected[j]==e;
            if(!used&&(best==c.experts||b.probabilities[e]>b.probabilities[best]))best=e;}
        b.selected[rank]=best;picked=add(picked,b.probabilities[best]);}
    for(std::size_t rank=0;rank<c.top_k;++rank)b.top_weights[rank]=bf(b.probabilities[b.selected[rank]]/picked,b.status);
    float shared=0;for(std::size_t j=0;j<c.hidden;++j)shared=finite(add(shared,mul(expand(b.shared_gate[j]),x[j])),b.status);
    *b.shared_factor=bf(sigmoid(bf(shared,b.status)),b.status);
}
__global__ void activate(std::size_t n,MoeBuffers b){
    const std::size_t j=blockIdx.x*blockDim.x+threadIdx.x;if(j<n){const float gate=bf(b.gate[j],b.status),up=bf(b.up[j],b.status);
        b.activation[j]=bf(mul(bf(mul(gate,sigmoid(gate)),b.status),up),b.status);}
}
__global__ void accumulate(moe::Config c,MoeBuffers b,unsigned expert){
    const std::size_t j=blockIdx.x*blockDim.x+threadIdx.x;if(j<c.hidden){float weight=0;
        for(std::size_t rank=0;rank<c.top_k;++rank)if(b.selected[rank]==expert)weight=b.top_weights[rank];
        b.accumulator[j]=bf(add(b.accumulator[j],bf(mul(bf(b.down[j],b.status),weight),b.status)),b.status);}
}
__global__ void finish(moe::Config c,MoeBuffers b,float* output){
    const std::size_t j=blockIdx.x*blockDim.x+threadIdx.x;if(j<c.hidden){b.shared[j]=bf(mul(bf(b.down[j],b.status),*b.shared_factor),b.status);
        b.result[j]=bf(add(b.accumulator[j],b.shared[j]),b.status);output[j]=b.result[j];}
}
}
cudaError_t moe_route(moe::Config c,MoeBuffers b,const float* x){
    route_logits<<<c.experts,32,0,cudaStreamLegacy>>>(c,b,x);auto e=cudaGetLastError();if(e!=cudaSuccess)return e;
    route<<<1,1,0,cudaStreamLegacy>>>(c,b,x);return cudaGetLastError();
}
cudaError_t moe_activate(std::size_t n,MoeBuffers b){activate<<<(n+127)/128,128,0,cudaStreamLegacy>>>(n,b);return cudaGetLastError();}
cudaError_t moe_accumulate(moe::Config c,MoeBuffers b,unsigned e){accumulate<<<(c.hidden+127)/128,128,0,cudaStreamLegacy>>>(c,b,e);return cudaGetLastError();}
cudaError_t moe_finish(moe::Config c,MoeBuffers b,float* y){finish<<<(c.hidden+127)/128,128,0,cudaStreamLegacy>>>(c,b,y);return cudaGetLastError();}
} // namespace kadan::cuda::detail
