#ifdef CUDA_API_PER_THREAD_DEFAULT_STREAM
#error "Layer primitives require legacy default-stream semantics"
#endif
#include "kadan/cuda_layer_math.hpp"
#include "layer_math_validation.hpp"
#include <cuda_runtime.h>
#include <cmath>
namespace kadan::cuda {
namespace {
__device__ float checked(double x,unsigned* status) {
    if(!isfinite(x) || fabs(x)>3.40282346638528859812e38) {atomicOr(status,1u);return 0;}
    return static_cast<float>(x);
}
__device__ double sigmoid(double x) {const double e=exp(-fabs(x));return x>=0?1/(1+e):e/(1+e);}
__global__ void norm_kernel(math::Norm s,const float* in,const float* w,float* out,unsigned* status) {
    __shared__ double sums[128];
    const auto row=blockIdx.x;const auto lane=threadIdx.x;double local=0;
    for(std::size_t c=lane;c<s.width;c+=128) {const double x=in[row*s.width+c];if(!isfinite(x)) atomicOr(status,1u);local+=x*x;}
    sums[lane]=local;__syncthreads();
    for(unsigned distance=64;distance;distance/=2) {if(lane<distance) sums[lane]+=sums[lane+distance];__syncthreads();}
    const double denominator=sqrt(sums[0]/s.width+s.epsilon);
    for(std::size_t c=lane;c<s.width;c+=128) {
        const double scale=double(w[c])+(s.scale==math::NormScale::one_plus?1:0);
        out[row*s.width+c]=checked((double(in[row*s.width+c])/denominator)*scale,status);
    }
}
__global__ void gate_kernel(math::Gate g,std::size_t n,const float* in,const float* modulation,float* out,unsigned* status) {
    const std::size_t i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
    const double x=in[i],z=modulation[i];if(!isfinite(x) || !isfinite(z)) {atomicOr(status,1u);out[i]=0;return;}
    double factor=sigmoid(z);if(g==math::Gate::silu) factor*=z;out[i]=checked(x*factor,status);
}
__global__ void rope_kernel(math::Rotary s,std::size_t position,const float* freq,const float* in,float* out,unsigned* status) {
    const std::size_t i=blockIdx.x*blockDim.x+threadIdx.x;
    const auto half=s.rotary_dim/2,work=s.head_dim-half;if(i>=s.heads*work)return;
    const auto head=i/work,c=i%work,base=head*s.head_dim;
    if(c<half) {
        const float f=freq[c];if(!isfinite(f)||f<=0||f>1) {atomicOr(status,1u);return;}
        const float angle=static_cast<float>(position)*f;
        const double cosine=cosf(angle),sine=sinf(angle),a=in[base+c],b=in[base+half+c];
        out[base+c]=checked(a*cosine-b*sine,status);out[base+half+c]=checked(a*sine+b*cosine,status);
    } else {const auto index=base+c+half;out[index]=checked(in[index],status);}
}
void status_span(unsigned* status,std::span<const float> a,std::span<const float> b,std::span<float> out) {
    namespace d=math::detail;
    d::require(reinterpret_cast<std::uintptr_t>(status)%alignof(unsigned)==0,"math_pointer");
    for(auto s:{a,b,std::span<const float>(out)}) {
        d::require(reinterpret_cast<std::uintptr_t>(s.data())%alignof(float)==0,"math_pointer");
        d::require(!d::overlap(status,sizeof(unsigned),s.data(),s.size_bytes()),"math_status_overlap");
    }
}
}
cudaError_t rms_norm(math::Norm s,std::span<const float> in,std::span<const float> w,std::span<float> out,unsigned* status) {
    math::detail::norm(s,in.size(),w.size(),out.size());math::detail::output_alias(in,out,true);math::detail::output_alias(w,out,false);status_span(status,in,w,out);
    norm_kernel<<<s.rows,128,0,cudaStreamLegacy>>>(s,in.data(),w.data(),out.data(),status);return cudaGetLastError();
}
cudaError_t gate(math::Gate g,std::span<const float> in,std::span<const float> modulation,std::span<float> out,unsigned* status) {
    math::detail::gate(g,in.size(),modulation.size(),out.size());math::detail::output_alias(in,out,true);math::detail::output_alias(modulation,out,true);status_span(status,in,modulation,out);
    gate_kernel<<<(in.size()+127)/128,128,0,cudaStreamLegacy>>>(g,in.size(),in.data(),modulation.data(),out.data(),status);return cudaGetLastError();
}
cudaError_t rope(math::Rotary s,std::size_t position,std::span<const float> freq,std::span<const float> in,std::span<float> out,unsigned* status) {
    math::detail::rotary(s,position,freq.size(),in.size(),out.size());math::detail::output_alias(in,out,true);math::detail::output_alias(freq,out,false);status_span(status,in,freq,out);
    const auto work=s.heads*(s.head_dim-s.rotary_dim/2);
    rope_kernel<<<(work+127)/128,128,0,cudaStreamLegacy>>>(s,position,freq.data(),in.data(),out.data(),status);return cudaGetLastError();
}
} // namespace kadan::cuda
