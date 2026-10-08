#include "linear_kernel.cuh"
#include <cuda_runtime.h>
namespace kadan::cuda::detail {
namespace {
__device__ float weight(std::uint16_t x){return __uint_as_float(unsigned(x)<<16);}
__device__ float finite(float x,unsigned* status){if(!isfinite(x)){atomicOr(status,1u);return 0;}return x;}
__device__ float round(float x,unsigned* status){x=finite(x,status);unsigned bits=__float_as_uint(x);bits+=0x7fffu+((bits>>16)&1u);return finite(__uint_as_float(bits&0xffff0000u),status);}
__device__ float sig(float x){const float e=expf(-fabsf(x));return x>=0?1/(1+e):e/(1+e);}
__device__ float product(float a,float b){return __fmul_rn(a,b);}
__device__ float plus(float a,float b){return __fadd_rn(a,b);}
__global__ void normalize(linear::Config c,LinearBuffers b,const float* input){
    if(threadIdx.x)return;float sum=0;
    for(std::size_t i=0;i<c.hidden;++i){const float x=finite(input[i],b.status);if(round(x,b.status)!=x)atomicOr(b.status,1u);sum=finite(plus(sum,product(x,x)),b.status);}
    const float inverse=1/sqrtf(sum/c.hidden+c.epsilon);
    for(std::size_t i=0;i<c.hidden;++i)b.normalized[i]=round(product(product(input[i],inverse),plus(1,weight(b.input_norm[i]))),b.status);
}
__global__ void normalize_shared(linear::Config c,LinearBuffers b,const float* input){
    extern __shared__ float x[];float* products=x+c.hidden;__shared__ float inverse;
    for(std::size_t j=threadIdx.x;j<c.hidden;j+=blockDim.x){x[j]=input[j];const float v=finite(x[j],b.status);if(round(v,b.status)!=v)atomicOr(b.status,1u);products[j]=product(v,v);}
    __syncthreads();
    if(threadIdx.x==0){float sum=0;for(std::size_t j=0;j<c.hidden;++j)sum=finite(plus(sum,products[j]),b.status);
        inverse=1/sqrtf(sum/c.hidden+c.epsilon);}
    __syncthreads();
    for(std::size_t j=threadIdx.x;j<c.hidden;j+=blockDim.x)b.normalized[j]=round(product(product(x[j],inverse),plus(1,weight(b.input_norm[j]))),b.status);
}
__global__ void auxiliary(linear::Config c,LinearBuffers b){
    const std::size_t i=blockIdx.x*blockDim.x+threadIdx.x;
    const auto values=c.value_heads*c.value_dim,channels=2*c.key_heads*c.key_dim+values;
    if(i<channels)b.qkv[i]=round(b.qkv[i],b.status);
    if(i<values)b.z[i]=round(b.z[i],b.status);
}
// Independent output heads retain their original ascending hidden-dimension sum.
__global__ void auxiliary_projection(linear::Config c,LinearBuffers b){
    if(threadIdx.x)return;const std::size_t i=blockIdx.x;
    float a=0,bb=0;for(std::size_t j=0;j<c.hidden;++j){
        a=finite(plus(a,product(weight(b.a_weight[i*c.hidden+j]),b.normalized[j])),b.status);
        bb=finite(plus(bb,product(weight(b.b_weight[i*c.hidden+j]),b.normalized[j])),b.status);}
    b.a[i]=round(a,b.status);b.b[i]=round(bb,b.status);
}

// Products are independent; retaining lane-zero ascending adds preserves the
// original rounding and per-add numeric error behavior, including overflow.
__global__ void auxiliary_projection_shared(linear::Config c,LinearBuffers b){
    extern __shared__ float products[];float* ap=products;float* bp=ap+c.hidden;
    const std::size_t i=blockIdx.x;
    for(std::size_t j=threadIdx.x;j<c.hidden;j+=blockDim.x){
        ap[j]=product(weight(b.a_weight[i*c.hidden+j]),b.normalized[j]);
        bp[j]=product(weight(b.b_weight[i*c.hidden+j]),b.normalized[j]);}
    __syncthreads();
    if(threadIdx.x==0){float a=0,bb=0;for(std::size_t j=0;j<c.hidden;++j){
        a=finite(plus(a,ap[j]),b.status);bb=finite(plus(bb,bp[j]),b.status);}
        b.a[i]=round(a,b.status);b.b[i]=round(bb,b.status);}
}

__global__ void convolution(linear::Config c,LinearBuffers b){
    const std::size_t channel=blockIdx.x*blockDim.x+threadIdx.x;
    const auto channels=2*c.key_heads*c.key_dim+c.value_heads*c.value_dim;if(channel>=channels)return;
    const auto start=channel*c.conv_kernel;
    for(std::size_t j=1;j<c.conv_kernel;++j)b.convolution[start+j-1]=b.convolution[start+j];
    b.convolution[start+c.conv_kernel-1]=std::uint16_t(__float_as_uint(b.qkv[channel])>>16);
    float sum=0;for(std::size_t j=0;j<c.conv_kernel;++j){const float x=__uint_as_float(unsigned(b.convolution[start+j])<<16);sum=finite(plus(sum,product(x,weight(b.conv_weight[start+j]))),b.status);}
    const float x=round(sum,b.status);b.qkv[channel]=round(product(x,sig(x)),b.status);
}
__global__ void normalize_qk(linear::Config c,LinearBuffers b){
    const auto head=blockIdx.x;if(threadIdx.x)return;const auto keys=c.key_heads*c.key_dim;
    float qsum=0,ksum=0;for(std::size_t j=0;j<c.key_dim;++j){const float q=b.qkv[head*c.key_dim+j],k=b.qkv[keys+head*c.key_dim+j];qsum=finite(plus(qsum,product(q,q)),b.status);ksum=finite(plus(ksum,product(k,k)),b.status);}
    const float qi=1/sqrtf(qsum+1e-6f),ki=1/sqrtf(ksum+1e-6f);
    for(std::size_t j=0;j<c.key_dim;++j){b.qkv[head*c.key_dim+j]=product(b.qkv[head*c.key_dim+j],qi)/sqrtf(float(c.key_dim));b.qkv[keys+head*c.key_dim+j]=product(b.qkv[keys+head*c.key_dim+j],ki);}
}
__global__ void recurrence(linear::Config c,LinearBuffers b){
    const auto head=blockIdx.x;const auto key_head=head/(c.value_heads/c.key_heads),keys=c.key_heads*c.key_dim;
    const float t=finite(plus(b.a[head],weight(b.dt_bias[head])),b.status);
    const float soft=plus(fmaxf(t,0),log1pf(expf(-fabsf(t))));
    const float g=finite(-product(expf(weight(b.a_log[head])),soft),b.status),decay=expf(g),beta=round(sig(b.b[head]),b.status);
    const auto base=head*c.key_dim*c.value_dim;
    for(std::size_t v=threadIdx.x;v<c.value_dim;v+=blockDim.x){
        float prediction=0;for(std::size_t k=0;k<c.key_dim;++k){auto& state=b.recurrent[base+k*c.value_dim+v];state=finite(product(state,decay),b.status);prediction=finite(plus(prediction,product(state,b.qkv[keys+key_head*c.key_dim+k])),b.status);}
        const float delta=finite(product(b.qkv[2*keys+head*c.value_dim+v]-prediction,beta),b.status);float result=0;
        for(std::size_t k=0;k<c.key_dim;++k){auto& state=b.recurrent[base+k*c.value_dim+v];state=finite(plus(state,product(b.qkv[keys+key_head*c.key_dim+k],delta)),b.status);result=finite(plus(result,product(state,b.qkv[key_head*c.key_dim+k])),b.status);}
        b.core[head*c.value_dim+v]=round(result,b.status);
    }
}
__global__ void gated_norm(linear::Config c,LinearBuffers b){
    const auto head=blockIdx.x;if(threadIdx.x)return;const auto base=head*c.value_dim;
    float sum=0;for(std::size_t j=0;j<c.value_dim;++j)sum=finite(plus(sum,product(b.core[base+j],b.core[base+j])),b.status);
    const float inverse=1/sqrtf(sum/c.value_dim+c.epsilon);
    for(std::size_t j=0;j<c.value_dim;++j){const float n=round(product(b.core[base+j],inverse),b.status);const float weighted=round(product(n,weight(b.output_norm[j])),b.status);b.gated[base+j]=round(product(weighted,product(b.z[base+j],sig(b.z[base+j]))),b.status);}
}
__global__ void residual(linear::Config c,LinearBuffers b,const float* input,float* out){
    const std::size_t i=blockIdx.x*blockDim.x+threadIdx.x;if(i<c.hidden)out[i]=round(plus(input[i],round(b.projected[i],b.status)),b.status);
}
}
cudaError_t linear_normalize(linear::Config c,LinearBuffers b,const float* input){if(c.hidden<=4096)normalize_shared<<<1,256,2*c.hidden*sizeof(float),cudaStreamLegacy>>>(c,b,input);
    else normalize<<<1,1,0,cudaStreamLegacy>>>(c,b,input);return cudaGetLastError();}
cudaError_t linear_auxiliary(linear::Config c,LinearBuffers b){
    const auto channels=2*c.key_heads*c.key_dim+c.value_heads*c.value_dim;
    auxiliary<<<(channels+127)/128,128,0,cudaStreamLegacy>>>(c,b);auto e=cudaGetLastError();if(e!=cudaSuccess)return e;
    if(c.hidden<=4096) auxiliary_projection_shared<<<c.value_heads,128,2*c.hidden*sizeof(float),cudaStreamLegacy>>>(c,b);
    else auxiliary_projection<<<c.value_heads,1,0,cudaStreamLegacy>>>(c,b);return cudaGetLastError();
}
cudaError_t linear_core(linear::Config c,LinearBuffers b){
    const auto channels=2*c.key_heads*c.key_dim+c.value_heads*c.value_dim;
    auto e=linear_auxiliary(c,b);if(e!=cudaSuccess)return e;
    convolution<<<(channels+127)/128,128,0,cudaStreamLegacy>>>(c,b);e=cudaGetLastError();if(e!=cudaSuccess)return e;
    normalize_qk<<<c.key_heads,1,0,cudaStreamLegacy>>>(c,b);e=cudaGetLastError();if(e!=cudaSuccess)return e;
    recurrence<<<c.value_heads,128,0,cudaStreamLegacy>>>(c,b);e=cudaGetLastError();if(e!=cudaSuccess)return e;
    gated_norm<<<c.value_heads,1,0,cudaStreamLegacy>>>(c,b);return cudaGetLastError();
}
cudaError_t linear_residual(linear::Config c,LinearBuffers b,const float* input,float* output){residual<<<(c.hidden+127)/128,128,0,cudaStreamLegacy>>>(c,b,input,output);return cudaGetLastError();}
} // namespace kadan::cuda::detail
