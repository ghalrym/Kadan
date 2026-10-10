#include "kadan/h3_compute.hpp"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <algorithm>
#include <cmath>
#include <climits>
#include <future>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <type_traits>
namespace kadan::video {
namespace {
void require(bool b,const char* message){if(!b)throw std::runtime_error(message);}
void checked(cudaError_t code){if(code!=cudaSuccess)throw std::runtime_error(std::string("h3_cuda: ")+cudaGetErrorString(code));}
void checked(cublasStatus_t code){if(code!=CUBLAS_STATUS_SUCCESS)throw std::runtime_error("h3_cublas_"+std::to_string(int(code)));}
void stop(const std::atomic_bool& cancel,const std::atomic_bool& failed){require(!cancel.load()&&!failed.load(),"h3_cuda_cancelled");}
std::size_t product(std::size_t a,std::size_t b){require(!b||a<=std::numeric_limits<std::size_t>::max()/b,"h3_cuda_size_overflow");return a*b;}
bool overlap(const void* a,std::size_t an,const void* b,std::size_t bn){if(!an||!bn)return false;const auto x=reinterpret_cast<std::uintptr_t>(a),y=reinterpret_cast<std::uintptr_t>(b);return x<=y?y-x<an:x-y<bn;}
struct Layout {
    std::size_t bytes=0;
    std::size_t add(std::size_t n){require(n<=std::numeric_limits<std::size_t>::max()-bytes-255,"h3_cuda_size_overflow");const auto at=bytes;bytes+=(n+255)/256*256;return at;}
};
// Arenas run only on owned async threads. Do not restore their default device:
// cudaSetDevice(0) on exit would create an unselected device context on CUDA 12.
// Explicit close establishes the cleanup boundary. A failed synchronization,
// handle destruction or free deliberately retains the resource reservation.
struct Arena {
    Resources& resources;Handle handle=0;void* data=nullptr;cublasHandle_t blas=nullptr;bool uncertain=false;std::atomic_bool& quarantine;
    Arena(Resources& r,int device,std::size_t bytes,std::atomic_bool& q,Workload workload):resources(r),quarantine(q){
        auto f=r.snapshot().capacity;require(device>=0&&std::size_t(device)+1<f.size(),"h3_cuda_unbudgeted_device");std::fill(f.begin(),f.end(),0);f[device+1]=bytes;
        handle=r.reserve(workload,std::move(f));
        try{checked(cudaSetDevice(device));checked(cudaMalloc(&data,bytes));checked(cublasCreate(&blas));checked(cublasSetMathMode(blas,CUBLAS_PEDANTIC_MATH));checked(cublasSetStream(blas,cudaStreamLegacy));r.loaded(handle);r.pin(handle);}
        catch(...){close_construction();throw;}
    }
    void close_construction() noexcept {
        bool ok=true;if(blas){ok=cublasDestroy(blas)==CUBLAS_STATUS_SUCCESS;blas=nullptr;}
        if(data){ok=(cudaDeviceSynchronize()==cudaSuccess)&&ok;if(ok){ok=cudaFree(data)==cudaSuccess;if(ok)data=nullptr;}}
        if(ok&&handle){try{resources.released(handle);handle=0;}catch(...){}}
        uncertain=!ok;if(!ok)quarantine=true;
    }
    template<class T>T* at(std::size_t offset){return reinterpret_cast<T*>(static_cast<char*>(data)+offset);}
    void close(){
        if(!handle||uncertain)return;
        try{checked(cudaStreamSynchronize(cudaStreamLegacy));checked(cublasDestroy(blas));blas=nullptr;checked(cudaFree(data));data=nullptr;resources.unpin(handle);resources.begin_eviction(handle);resources.released(handle);handle=0;}
        catch(...){uncertain=true;quarantine=true;throw;}
    }
    ~Arena(){if(!uncertain)try{close();}catch(...){}}
};
__global__ void rotate_kernel(const float* x,float* rotated,std::size_t in,unsigned group){
    __shared__ double values[256];const unsigned c=threadIdx.x;const auto at=std::size_t(blockIdx.x)*in+std::size_t(blockIdx.y)*group+c;
    values[c]=x[at];__syncthreads();
    for(unsigned stride=1;stride<group;stride*=4){
        const unsigned base=c/(stride*4)*(stride*4)+c%stride,k=c/stride%4;
        const double a=values[base],b=values[base+stride],v=values[base+2*stride],d=values[base+3*stride];
        double y=k==0?((a+b)+(v-d))*.5:k==1?((a+b)+(d-v))*.5:k==2?((a+v)+(d-b))*.5:((b+v)+(d-a))*.5;
        __syncthreads();values[c]=y;__syncthreads();
    }
    rotated[at]=float(values[c]);
}
__global__ void quantize_kernel(const float* x,std::int8_t* codes,float* scales,std::size_t in){
    __shared__ float maxima[256];float maximum=1e-10f;const auto start=std::size_t(blockIdx.x)*in;
    for(std::size_t c=threadIdx.x;c<in;c+=256)maximum=fmaxf(maximum,fabsf(x[start+c]));maxima[threadIdx.x]=maximum;__syncthreads();
    for(unsigned s=128;s;s/=2){if(threadIdx.x<s)maxima[threadIdx.x]=fmaxf(maxima[threadIdx.x],maxima[threadIdx.x+s]);__syncthreads();}
    const float scale=maxima[0]/127.0f;if(threadIdx.x==0)scales[blockIdx.x]=scale;
    for(std::size_t c=threadIdx.x;c<in;c+=256)codes[start+c]=std::int8_t(max(-127,min(127,__float2int_rn(x[start+c]/scale))));
}
__global__ void finish_int8(const std::int32_t* sums,const float* row_scale,const float* weight_scale,const float* bias,float* y,std::size_t count,std::size_t out){
    const auto i=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i<count)y[i]=(float(sums[i])*row_scale[i/out])*weight_scale[i%out]+bias[i%out];
}
__global__ void add_bias(float* y,const float* bias,std::size_t count,std::size_t out){const auto i=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i<count)y[i]+=bias[i%out];}
template<class Real> __global__ void online_softmax(Real* scores,Real* output,Real* maxima,Real* totals,unsigned keys,unsigned dim,unsigned first_query,unsigned first_key,unsigned causal_queries,unsigned window,unsigned offset){
    __shared__ Real reduction[256];const unsigned q=blockIdx.x,t=threadIdx.x;Real m=-INFINITY;
    for(unsigned k=t;k<keys;k+=256){Real v=scores[q*keys+k];if(!attention_visible(first_query+q,first_key+k,causal_queries,window,offset))v=-INFINITY;scores[q*keys+k]=v;m=fmax(m,v);}reduction[t]=m;__syncthreads();
    for(unsigned stride=128;stride;stride/=2){if(t<stride)reduction[t]=fmax(reduction[t],reduction[t+stride]);__syncthreads();}
    const Real next=fmax(maxima[q],reduction[0]);const Real alpha=isfinite(maxima[q])?exp(maxima[q]-next):0;Real total=0;
    // All warps must consume the maximum before shared storage becomes sums.
    __syncthreads();
    for(unsigned k=t;k<keys;k+=256){const Real v=scores[q*keys+k];const Real p=isfinite(v)?exp(v-next):0;scores[q*keys+k]=p;total+=p;}reduction[t]=total;__syncthreads();
    for(unsigned stride=128;stride;stride/=2){if(t<stride)reduction[t]+=reduction[t+stride];__syncthreads();}
    for(unsigned c=t;c<dim;c+=256)output[q*dim+c]*=alpha;
    if(t==0){maxima[q]=next;totals[q]=totals[q]*alpha+reduction[0];}
}
template<class Real> __global__ void initialize_attention(Real* maximum,Real* total,unsigned rows){const auto i=blockIdx.x*blockDim.x+threadIdx.x;if(i<rows){maximum[i]=-INFINITY;total[i]=0;}}
template<class Real> __global__ void divide_attention(Real* output,const Real* total,unsigned rows,unsigned dim){const auto i=blockIdx.x*blockDim.x+threadIdx.x;if(i<rows*dim)output[i]/=total[i/dim];}
struct HostAdmission {
    Resources& resources;Handle handle;
    HostAdmission(Resources& r,std::size_t bytes,Workload workload):resources(r){auto f=r.snapshot().capacity;std::fill(f.begin(),f.end(),0);f[0]=bytes;handle=r.reserve(workload,std::move(f));}
    ~HostAdmission(){resources.released(handle);}
};
template<class Real> cublasStatus_t gemm(cublasHandle_t h,cublasOperation_t a,cublasOperation_t b,int m,int n,int k,Real alpha,const Real* x,int ldx,const Real* y,int ldy,Real beta,Real* z,int ldz){
    if constexpr(std::is_same_v<Real,double>)return cublasDgemm(h,a,b,m,n,k,&alpha,x,ldx,y,ldy,&beta,z,ldz);
    else return cublasSgemm(h,a,b,m,n,k,&alpha,x,ldx,y,ldy,&beta,z,ldz);
}
__global__ void widen(const float* source,double* destination,std::size_t n){const auto i=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i<n)destination[i]=source[i];}
__global__ void narrow_bias(const double* source,const float* bias,float* destination,std::size_t n,std::size_t out){const auto i=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i<n)destination[i]=float(source[i])+bias[i%out];}
class Compute final:public H3Compute {
    std::shared_ptr<Resources> resources_;std::vector<int> devices_;Workload workload_;bool column_split_;std::atomic_bool quarantined_=false;
    template<class F>void split(std::size_t rows,const std::atomic_bool& cancel,F function){
        require(!quarantined_.load(),"h3_cuda_cleanup_unconfirmed");require(!cancel.load(),"h3_cuda_cancelled");std::atomic_bool failed=false;std::vector<std::future<void>> tasks;std::exception_ptr error;
        try{for(std::size_t d=0;d<devices_.size();++d){const auto begin=rows*d/devices_.size(),end=rows*(d+1)/devices_.size();if(begin==end)continue;tasks.push_back(std::async(std::launch::async,[&,d,begin,end]{try{function(devices_[d],begin,end,failed);}catch(...){failed=true;throw;}}));}}
        catch(...){failed=true;error=std::current_exception();}
        for(auto& t:tasks)try{t.get();}catch(...){if(!error)error=std::current_exception();}
        if(error)std::rethrow_exception(error);require(!cancel.load(),"h3_cuda_cancelled");
    }
    static void validate(std::span<const float> x,std::span<const float> bias,std::size_t in,std::size_t out,std::span<float> y){
        require(in&&out&&in<=std::size_t(INT_MAX)&&out<=std::size_t(INT_MAX)&&!x.empty()&&x.size()%in==0&&y.size()==product(x.size()/in,out)&&(bias.empty()||bias.size()==out),"h3_cuda_projection_shape");
        for(float v:x)require(std::isfinite(v),"h3_cuda_nonfinite_input");for(float v:bias)require(std::isfinite(v),"h3_cuda_nonfinite_bias");
    }
    template<class Weight>void project(std::span<const Weight> weights,std::span<const float> scales,std::span<const float> bias,std::span<const float> x,std::size_t in,std::size_t out,std::size_t group,std::span<float> y,const std::atomic_bool& cancel,bool precise=false){
        require(!cancel.load(),"h3_cuda_cancelled");validate(x,bias,in,out,y);require(!overlap(y.data(),y.size_bytes(),x.data(),x.size_bytes())&&!overlap(y.data(),y.size_bytes(),weights.data(),weights.size_bytes())&&!overlap(y.data(),y.size_bytes(),bias.data(),bias.size_bytes())&&!overlap(y.data(),y.size_bytes(),scales.data(),scales.size_bytes()),"h3_cuda_alias");require(weights.size()==product(in,out),"h3_cuda_weight_shape");constexpr bool quant=sizeof(Weight)==1;
        if constexpr(quant){require((group==64||group==256)&&in%group==0&&in<=std::size_t(INT_MAX/(128*127))&&out%4==0&&scales.size()==out,"h3_cuda_quant_shape");for(float v:scales)require(std::isfinite(v)&&v>=0,"h3_cuda_scale");}
        else for(float v:weights)require(std::isfinite(v),"h3_cuda_nonfinite_weight");
        const auto total_out=out;const bool columns=column_split_&&!quant&&projection_split_columns(x.size()/in,devices_.size());
        const auto all_weights=weights;const auto all_bias=bias;
        split(columns?out:x.size()/in,cancel,[&](int device,std::size_t begin,std::size_t end,const std::atomic_bool& failed){
            const auto first_column=columns?begin:0;
            const auto out=columns?end-begin:total_out;
            const auto weights=columns?all_weights.subspan(begin*in,out*in):all_weights;
            const auto bias=columns&&!all_bias.empty()?all_bias.subspan(begin,out):all_bias;
            if(columns){begin=0;end=x.size()/in;}
            stop(cancel,failed);const auto tile=std::min<std::size_t>(256,end-begin);Layout l;
            const auto w=l.add(weights.size_bytes()),b=l.add(out*4),s=l.add(out*4),input=l.add(tile*in*4),rotated=l.add(tile*in*4),codes=l.add(tile*in),row_scales=l.add(tile*4),sums=l.add(tile*out*4),output=l.add(tile*out*4),workspace=l.add(8*1024*1024),wide_weights=l.add(precise?weights.size()*8:0),wide_input=l.add(precise?tile*in*8:0),wide_output=l.add(precise?tile*out*8:0);
            Arena arena(*resources_,device,l.bytes,quarantined_,workload_);checked(cublasSetWorkspace(arena.blas,arena.at<void>(workspace),8*1024*1024));
            checked(cudaMemcpy(arena.at<void>(w),weights.data(),weights.size_bytes(),cudaMemcpyHostToDevice));
            if(bias.empty())checked(cudaMemset(arena.at<void>(b),0,out*4));else checked(cudaMemcpy(arena.at<void>(b),bias.data(),out*4,cudaMemcpyHostToDevice));
            if constexpr(quant)checked(cudaMemcpy(arena.at<void>(s),scales.data(),out*4,cudaMemcpyHostToDevice));
            else if(precise){widen<<<unsigned((weights.size()+255)/256),256>>>(arena.at<float>(w),arena.at<double>(wide_weights),weights.size());checked(cudaGetLastError());}
            for(std::size_t start=begin;start<end;start+=tile){stop(cancel,failed);const auto rows=std::min(tile,end-start);checked(cudaMemcpy(arena.at<void>(input),x.data()+start*in,rows*in*4,cudaMemcpyHostToDevice));
                if constexpr(quant){
                    rotate_kernel<<<dim3(unsigned(rows),unsigned(in/group)),unsigned(group)>>>(arena.at<float>(input),arena.at<float>(rotated),in,unsigned(group));checked(cudaGetLastError());
                    quantize_kernel<<<unsigned(rows),256>>>(arena.at<float>(rotated),arena.at<std::int8_t>(codes),arena.at<float>(row_scales),in);checked(cudaGetLastError());
                    const std::int32_t one=1,zero=0;checked(cublasGemmEx(arena.blas,CUBLAS_OP_T,CUBLAS_OP_N,int(out),int(rows),int(in),&one,arena.at<void>(w),CUDA_R_8I,int(in),arena.at<void>(codes),CUDA_R_8I,int(in),&zero,arena.at<void>(sums),CUDA_R_32I,int(out),CUBLAS_COMPUTE_32I,CUBLAS_GEMM_DEFAULT));
                    finish_int8<<<unsigned((rows*out+255)/256),256>>>(arena.at<std::int32_t>(sums),arena.at<float>(row_scales),arena.at<float>(s),arena.at<float>(b),arena.at<float>(output),rows*out,out);
                }else if(precise){
                    widen<<<unsigned((rows*in+255)/256),256>>>(arena.at<float>(input),arena.at<double>(wide_input),rows*in);checked(cudaGetLastError());
                    checked(gemm<double>(arena.blas,CUBLAS_OP_T,CUBLAS_OP_N,int(out),int(rows),int(in),1.0,arena.at<double>(wide_weights),int(in),arena.at<double>(wide_input),int(in),0.0,arena.at<double>(wide_output),int(out)));
                    narrow_bias<<<unsigned((rows*out+255)/256),256>>>(arena.at<double>(wide_output),arena.at<float>(b),arena.at<float>(output),rows*out,out);
                }else{
                    const float one=1,zero=0;checked(cublasSgemm(arena.blas,CUBLAS_OP_T,CUBLAS_OP_N,int(out),int(rows),int(in),&one,arena.at<float>(w),int(in),arena.at<float>(input),int(in),&zero,arena.at<float>(output),int(out)));
                    add_bias<<<unsigned((rows*out+255)/256),256>>>(arena.at<float>(output),arena.at<float>(b),rows*out,out);
                }
                checked(cudaGetLastError());checked(cudaStreamSynchronize(cudaStreamLegacy));stop(cancel,failed);checked(cudaMemcpy2D(y.data()+start*total_out+first_column,total_out*4,arena.at<void>(output),out*4,out*4,rows,cudaMemcpyDeviceToHost));
            }
            arena.close();
        });
        for(float v:y)require(std::isfinite(v),"h3_cuda_nonfinite_output");
    }
public:
    Compute(std::shared_ptr<Resources> r,std::vector<int> devices,Workload workload=Workload::video,bool columns=false):resources_(std::move(r)),devices_(std::move(devices)),workload_(workload),column_split_(columns){
        require(bool(resources_)&&!devices_.empty(),"h3_cuda_devices");const auto capacity=resources_->snapshot().capacity;
        for(std::size_t i=0;i<devices_.size();++i){const int d=devices_[i];require(d>=0&&std::size_t(d)+1<capacity.size()&&capacity[d+1]>0,"h3_cuda_unbudgeted_device");require(std::find(devices_.begin(),devices_.begin()+i,d)==devices_.begin()+i,"h3_cuda_duplicate_device");}
    }
    template<class Real> void attention_impl(std::span<const float> q,std::span<const float> k,std::span<const float> v,std::size_t n,std::size_t heads,std::size_t kv_heads,std::size_t dim,bool causal,std::span<float> y,const std::atomic_bool& cancel,std::size_t key_tokens=0,std::size_t causal_queries=0,std::size_t window=0,std::size_t offset=0){
        require(!cancel.load(),"h3_cuda_cancelled");if(!key_tokens)key_tokens=n;if(causal)causal_queries=n;
        require(n>0&&n<=std::size_t(INT_MAX)&&heads>0&&kv_heads>0&&heads%kv_heads==0&&dim>0&&dim<=2048,"h3_cuda_attention_shape");
        require(key_tokens>0&&key_tokens<=std::size_t(INT_MAX)&&causal_queries<=n&&offset<=key_tokens&&window<=key_tokens&&(!causal_queries||causal_queries<=key_tokens-offset),"h3_cuda_attention_shape");
        const auto head_values=product(n,dim),key_values=product(key_tokens,dim);require(q.size()==product(head_values,heads)&&k.size()==product(key_values,kv_heads)&&v.size()==k.size()&&y.size()==q.size(),"h3_cuda_attention_shape");
        for(auto input:{q,k,v}){require(!overlap(y.data(),y.size_bytes(),input.data(),input.size_bytes()),"h3_cuda_alias");for(float x:input)require(std::isfinite(x),"h3_cuda_nonfinite_input");}
        split(heads,cancel,[&](int device,std::size_t first,std::size_t last,const std::atomic_bool& failed){
            stop(cancel,failed);const auto query_tile=std::min<std::size_t>(128,n),key_tile=std::min<std::size_t>(4096,key_tokens);Layout l;
            const auto dq=l.add(head_values*sizeof(Real)),dk=l.add(key_values*sizeof(Real)),dv=l.add(key_values*sizeof(Real)),dy=l.add(query_tile*dim*sizeof(Real)),scores=l.add(query_tile*key_tile*sizeof(Real)),maximum=l.add(query_tile*sizeof(Real)),total=l.add(query_tile*sizeof(Real)),workspace=l.add(8*1024*1024);
            HostAdmission admission(*resources_,(head_values+key_values*2+query_tile*dim)*sizeof(Real),workload_);
            auto memory=std::make_unique<Real[]>(head_values+key_values*2+query_tile*dim);auto hq=memory.get(),hk=hq+head_values,hv=hk+key_values,hy=hv+key_values;
            Arena arena(*resources_,device,l.bytes,quarantined_,workload_);checked(cublasSetWorkspace(arena.blas,arena.at<void>(workspace),8*1024*1024));
            for(std::size_t h=first;h<last;++h){stop(cancel,failed);const auto kh=h/(heads/kv_heads);
                for(std::size_t t=0;t<n;++t)for(std::size_t c=0;c<dim;++c)hq[t*dim+c]=q[(t*heads+h)*dim+c];
                for(std::size_t t=0;t<key_tokens;++t)for(std::size_t c=0;c<dim;++c){hk[t*dim+c]=k[(t*kv_heads+kh)*dim+c];hv[t*dim+c]=v[(t*kv_heads+kh)*dim+c];}
                checked(cudaMemcpy(arena.at<void>(dq),hq,head_values*sizeof(Real),cudaMemcpyHostToDevice));checked(cudaMemcpy(arena.at<void>(dk),hk,key_values*sizeof(Real),cudaMemcpyHostToDevice));checked(cudaMemcpy(arena.at<void>(dv),hv,key_values*sizeof(Real),cudaMemcpyHostToDevice));
                for(std::size_t start=0;start<n;start+=query_tile){stop(cancel,failed);const auto rows=std::min(query_tile,n-start);checked(cudaMemset(arena.at<void>(dy),0,rows*dim*sizeof(Real)));initialize_attention<<<1,128>>>(arena.at<Real>(maximum),arena.at<Real>(total),unsigned(rows));checked(cudaGetLastError());
                    for(std::size_t key=0;key<key_tokens;key+=key_tile){stop(cancel,failed);const auto keys=std::min(key_tile,key_tokens-key);const Real scale=1/std::sqrt(Real(dim)),zero=0,one=1;
                        checked(gemm<Real>(arena.blas,CUBLAS_OP_T,CUBLAS_OP_N,int(keys),int(rows),int(dim),scale,arena.at<Real>(dk)+key*dim,int(dim),arena.at<Real>(dq)+start*dim,int(dim),zero,arena.at<Real>(scores),int(keys)));
                        online_softmax<<<unsigned(rows),256>>>(arena.at<Real>(scores),arena.at<Real>(dy),arena.at<Real>(maximum),arena.at<Real>(total),unsigned(keys),unsigned(dim),unsigned(start),unsigned(key),unsigned(causal_queries),unsigned(window),unsigned(offset));checked(cudaGetLastError());
                        checked(gemm<Real>(arena.blas,CUBLAS_OP_N,CUBLAS_OP_N,int(dim),int(rows),int(keys),one,arena.at<Real>(dv)+key*dim,int(dim),arena.at<Real>(scores),int(keys),one,arena.at<Real>(dy),int(dim)));checked(cudaStreamSynchronize(cudaStreamLegacy));
                    }
                    divide_attention<<<unsigned((rows*dim+255)/256),256>>>(arena.at<Real>(dy),arena.at<Real>(total),unsigned(rows),unsigned(dim));checked(cudaGetLastError());checked(cudaMemcpy(hy,arena.at<void>(dy),rows*dim*sizeof(Real),cudaMemcpyDeviceToHost));stop(cancel,failed);
                    for(std::size_t t=0;t<rows;++t)for(std::size_t c=0;c<dim;++c)y[((start+t)*heads+h)*dim+c]=hy[t*dim+c];
                }
            }
            arena.close();
        });
        for(float x:y)require(std::isfinite(x),"h3_cuda_nonfinite_output");
    }
    void attention(std::span<const float> q,std::span<const float> k,std::span<const float> v,std::size_t n,std::size_t heads,std::size_t kv,std::size_t dim,bool causal,std::span<float> y,const std::atomic_bool& cancel,bool precise)override{
        if(precise)attention_impl<double>(q,k,v,n,heads,kv,dim,causal,y,cancel);else attention_impl<float>(q,k,v,n,heads,kv,dim,causal,y,cancel);
    }
    bool attend(std::span<const float> q,std::span<const float> k,std::span<const float> v,std::size_t queries,std::size_t keys,std::size_t heads,std::size_t kv,std::size_t dim,std::size_t causal_queries,std::size_t window,std::size_t offset,std::span<float> y,const std::atomic_bool& cancel) override {
        attention_impl<float>(q,k,v,queries,heads,kv,dim,false,y,cancel,keys,causal_queries,window,offset);return true;
    }
    void release_devices() override {
        require(!quarantined_.load(),"h3_cuda_cleanup_unconfirmed");
        try {for(int device:devices_){checked(cudaSetDevice(device));checked(cudaDeviceSynchronize());checked(cudaDeviceReset());}}
        catch(...){quarantined_=true;throw;}
    }
    void dense(std::span<const float> w,std::span<const float> b,std::span<const float> x,std::size_t in,std::size_t out,std::span<float> y,const std::atomic_bool& c,bool precise)override{project(w,{},b,x,in,out,0,y,c,precise);}
    void convrot(std::span<const std::uint8_t> w,std::span<const float> s,std::span<const float> b,std::span<const float> x,std::size_t in,std::size_t out,std::size_t g,std::span<float> y,const std::atomic_bool& c)override{project(w,s,b,x,in,out,g,y,c);}
};
}
std::shared_ptr<H3Compute> h3_cuda_compute(std::shared_ptr<Resources> r,std::vector<int> d){return std::make_shared<Compute>(std::move(r),std::move(d));}
}

namespace kadan {
std::shared_ptr<DenseCompute> cuda_dense_compute(std::shared_ptr<Resources> r,std::vector<int> devices,Workload workload){
 return std::make_shared<video::Compute>(std::move(r),std::move(devices),workload,true);
}
}
