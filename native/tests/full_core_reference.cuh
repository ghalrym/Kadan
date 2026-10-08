// Frozen serial full-attention GPU reference for bitwise regression.
#include "full_kernel.cuh"
#include <cuda_runtime.h>
#include <math_constants.h>
namespace baseline {
using kadan::cuda::detail::FullBuffers;
namespace full=kadan::full;
namespace {
__device__ float expand(std::uint16_t x){return __uint_as_float(unsigned(x)<<16);}
__device__ float finite(float x,unsigned* s){if(!isfinite(x)){atomicOr(s,1u);return 0;}return x;}
__device__ float bf(float x,unsigned* s){unsigned u=__float_as_uint(finite(x,s));u+=0x7fffu+((u>>16)&1u);return finite(__uint_as_float(u&0xffff0000u),s);}
__device__ float mul(float a,float b){return __fmul_rn(a,b);}
__device__ float add(float a,float b){return __fadd_rn(a,b);}
__device__ float sigmoid(float x){const float e=expf(-fabsf(x));return x>=0?1/(1+e):e/(1+e);}
__device__ void norm(const float* x,const std::uint16_t* w,float* y,std::size_t width,float epsilon,unsigned* s){
    float total=0;for(std::size_t j=0;j<width;++j){const float v=bf(x[j],s);total=finite(add(total,mul(v,v)),s);}
    const float inverse=1/sqrtf(add(total/float(width),epsilon));
    for(std::size_t j=0;j<width;++j)y[j]=bf(mul(mul(bf(x[j],s),inverse),add(1,expand(w[j]))),s);
}
__device__ void rotate(float* x,const float* freq,std::size_t half,std::size_t position,unsigned* s){
    for(std::size_t j=0;j<half;++j){const float angle=mul(float(position),freq[j]),c=bf(cosf(angle),s),sn=bf(sinf(angle),s),a=x[j],b=x[j+half];
        x[j]=bf(add(bf(mul(a,c),s),-bf(mul(b,sn),s)),s);x[j+half]=bf(add(bf(mul(a,sn),s),bf(mul(b,c),s)),s);}
}
__global__ void normalize(full::Config c,FullBuffers b,const float* input){
    if(threadIdx.x)return;
    for(std::size_t j=0;j<c.hidden;++j)if(bf(input[j],b.status)!=input[j])atomicOr(b.status,1u);
    norm(input,b.input_norm,b.normalized,c.hidden,c.epsilon,b.status);
}
__global__ void prepare(full::Config c,FullBuffers b,std::size_t position){
    if(threadIdx.x)return;const auto h=blockIdx.x;
    norm(b.qg+h*2*c.head_dim,b.query_norm,b.query+h*c.head_dim,c.head_dim,c.epsilon,b.status);
    for(std::size_t j=0;j<c.head_dim;++j)b.gate[h*c.head_dim+j]=bf(b.qg[(h*2+1)*c.head_dim+j],b.status);
    rotate(b.query+h*c.head_dim,b.frequencies,c.rotary_dim/2,position,b.status);
    if(h<c.kv_heads){norm(b.key+h*c.head_dim,b.key_norm,b.key+h*c.head_dim,c.head_dim,c.epsilon,b.status);rotate(b.key+h*c.head_dim,b.frequencies,c.rotary_dim/2,position,b.status);}
}
__global__ void append(full::Config c,FullBuffers b,std::size_t position){
    const std::size_t j=blockIdx.x*blockDim.x+threadIdx.x,n=c.kv_heads*c.head_dim;
    if(j<n){b.keys[position*n+j]=std::uint16_t(__float_as_uint(bf(b.key[j],b.status))>>16);b.values[position*n+j]=std::uint16_t(__float_as_uint(bf(b.value[j],b.status))>>16);}
}
__global__ void attention(full::Config c,FullBuffers b,std::size_t position){
    if(threadIdx.x)return;const std::size_t h=blockIdx.x,kh=h/(c.heads/c.kv_heads),kv=c.kv_heads*c.head_dim;
    auto* row=b.probabilities+h*c.capacity;for(std::size_t t=0;t<c.capacity;++t)row[t]=0;
    const float scale=1/sqrtf(float(c.head_dim));float maximum=-CUDART_INF_F;
    for(std::size_t t=0;t<=position;++t){float dot=0;for(std::size_t j=0;j<c.head_dim;++j)dot=finite(add(dot,mul(b.query[h*c.head_dim+j],expand(b.keys[t*kv+kh*c.head_dim+j]))),b.status);
        row[t]=bf(mul(bf(dot,b.status),scale),b.status);maximum=fmaxf(maximum,row[t]);}
    float denominator=0;for(std::size_t t=0;t<=position;++t){row[t]=expf(row[t]-maximum);denominator=finite(add(denominator,row[t]),b.status);}
    for(std::size_t t=0;t<=position;++t)row[t]=bf(row[t]/denominator,b.status);
    for(std::size_t j=0;j<c.head_dim;++j){float sum=0;for(std::size_t t=0;t<=position;++t)sum=finite(add(sum,mul(row[t],expand(b.values[t*kv+kh*c.head_dim+j]))),b.status);
        const auto at=h*c.head_dim+j;b.core[at]=bf(sum,b.status);b.gated[at]=bf(mul(b.core[at],bf(sigmoid(b.gate[at]),b.status)),b.status);}
}
__global__ void residual(full::Config c,FullBuffers b,const float* x,float* y){
    const std::size_t j=blockIdx.x*blockDim.x+threadIdx.x;if(j<c.hidden)y[j]=bf(add(x[j],bf(b.projected[j],b.status)),b.status);
}
}
cudaError_t full_normalize(full::Config c,FullBuffers b,const float* x){normalize<<<1,1,0,cudaStreamLegacy>>>(c,b,x);return cudaGetLastError();}
cudaError_t full_core(full::Config c,FullBuffers b,std::size_t position){
    prepare<<<c.heads,1,0,cudaStreamLegacy>>>(c,b,position);auto e=cudaGetLastError();if(e!=cudaSuccess)return e;
    append<<<(c.kv_heads*c.head_dim+127)/128,128,0,cudaStreamLegacy>>>(c,b,position);e=cudaGetLastError();if(e!=cudaSuccess)return e;
    attention<<<c.heads,1,0,cudaStreamLegacy>>>(c,b,position);return cudaGetLastError();
}
cudaError_t full_residual(full::Config c,FullBuffers b,const float* x,float* y){residual<<<(c.hidden+127)/128,128,0,cudaStreamLegacy>>>(c,b,x,y);return cudaGetLastError();}
} // namespace kadan::cuda::detail
