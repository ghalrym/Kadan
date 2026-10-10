#include "kadan/h3_compute.hpp"
#include "kadan/device_workspace.hpp"
#include "kadan/device_weights.hpp"
#include "kadan/device_placement.hpp"
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
#include <chrono>
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
// CUDA calls stay on the persistent thread that owns this selected device.
class CudaDeviceOperations final:public DeviceOperations,public WeightDeviceOperations {
public:
    void* data=nullptr;cublasHandle_t blas=nullptr;const std::atomic_bool* cancellation=nullptr;const std::atomic_bool* failed=nullptr;Bytes scratch_bytes=0;const Bytes scratch_limit;
    DeviceWeights weights;
    CudaDeviceOperations(Resources& resources,int device,Workload workload,Bytes limit,Bytes scratch):scratch_limit(scratch),weights(resources,device,workload,limit,*this){}
    Bytes available_weight_bytes() override {std::size_t free=0,total=0;checked(cudaMemGetInfo(&free,&total));const auto remaining=scratch_limit>scratch_bytes?scratch_limit-scratch_bytes:0;return free>remaining?free-remaining:0;}
    void* allocate_weight(Bytes bytes) override {void* pointer=nullptr;const auto status=cudaMalloc(&pointer,bytes);if(status==cudaErrorMemoryAllocation){cudaGetLastError();return nullptr;}checked(status);return pointer;}
    void copy_weight(void* pointer,std::size_t offset,std::span<const std::byte> source) override {checked(cudaMemcpy(static_cast<char*>(pointer)+offset,source.data(),source.size(),cudaMemcpyHostToDevice));}
    void free_weight(void* pointer) override {checked(cudaFree(pointer));}
    void synchronize_weights() override {checked(cudaStreamSynchronize(cudaStreamLegacy));}
    void release_weights() override {weights.release();}
    void select(int device) override {checked(cudaSetDevice(device));}
    void allocate(Bytes bytes) override {
        // External users can race the queue's probe. Keep ownership while waiting
        // for scratch; cancellation is checked between bounded allocation attempts.
        wait_for_device_allocation([&]{return (cancellation&&cancellation->load())||(failed&&failed->load());},[&]{
            std::size_t free=0,total=0;checked(cudaMemGetInfo(&free,&total));
            if(free<bytes)return false;
            const auto status=cudaMalloc(&data,bytes);
            if(status==cudaSuccess){scratch_bytes=bytes;return true;}
            if(status!=cudaErrorMemoryAllocation)checked(status);
            cudaGetLastError();return false;
        },[]{std::this_thread::sleep_for(std::chrono::milliseconds(50));});
    }
    void free() override {checked(cudaFree(data));data=nullptr;scratch_bytes=0;}
    void create_handle() override {checked(cublasCreate(&blas));}
    void configure_handle() override {checked(cublasSetMathMode(blas,CUBLAS_PEDANTIC_MATH));checked(cublasSetStream(blas,cudaStreamLegacy));}
    void destroy_handle() override {checked(cublasDestroy(blas));blas=nullptr;}
    void synchronize() override {checked(cudaStreamSynchronize(cudaStreamLegacy));}
    void reset() override {checked(cudaDeviceSynchronize());checked(cudaDeviceReset());}
    template<class T>T* at(std::size_t offset){return reinterpret_cast<T*>(static_cast<char*>(data)+offset);}
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
    std::vector<std::unique_ptr<DeviceThread>> threads_;
    mutable std::mutex operation_mutex_;
    Footprint weight_reservations_;std::vector<Bytes> weight_capacities_;Bytes inventory_bytes_=0;bool replan_=false;
    template<class F>void all_devices(F function){
        std::vector<std::future<void>> tasks;tasks.reserve(threads_.size());std::exception_ptr error;
        for(auto& thread:threads_)try{tasks.push_back(thread->submit(function));}catch(...){if(!error)error=std::current_exception();}
        for(auto& task:tasks)try{task.get();}catch(...){if(!error)error=std::current_exception();}
        if(error){quarantined_=true;std::rethrow_exception(error);}
    }
    void release_scratch_locked(){
        all_devices([](DeviceWorkspace& workspace){workspace.release();});
        weight_reservations_[0]=0;
        for(std::size_t i=0;i<threads_.size();++i)threads_[i]->submit([&,i](DeviceWorkspace& workspace){
            const auto bytes=static_cast<CudaDeviceOperations&>(workspace.operations()).weights.reserved();
            weight_reservations_[devices_[i]+1]=bytes;if(bytes)weight_reservations_[0]+=DeviceWeights::metadata_bytes;
        }).get();
    }
    template<class F>void split(std::size_t rows,const std::atomic_bool& cancel,F function,bool weighted=false){
        std::lock_guard lock(operation_mutex_);
        require(!quarantined_.load(),"h3_cuda_cleanup_unconfirmed");
        if(cancel.load())rethrow_after_cleanup(std::make_exception_ptr(std::runtime_error("h3_cuda_cancelled")),[&]{release_scratch_locked();});
        if(replan_)plan_weights_locked();
        std::atomic_bool failed=false;std::vector<std::future<void>> tasks;tasks.reserve(threads_.size());std::exception_ptr error;
        try{for(std::size_t d=0;d<threads_.size();++d){const auto begin=weighted?weight_boundary(rows,weight_capacities_,d):rows*d/threads_.size(),end=weighted?weight_boundary(rows,weight_capacities_,d+1):rows*(d+1)/threads_.size();if(begin==end)continue;
            tasks.push_back(threads_[d]->submit_work([&,begin,end](DeviceWorkspace& workspace){
                auto& ops=static_cast<CudaDeviceOperations&>(workspace.operations());
                struct CancelScope {CudaDeviceOperations& ops;~CancelScope(){ops.cancellation=nullptr;ops.failed=nullptr;}} scope{ops};
                ops.cancellation=&cancel;ops.failed=&failed;function(workspace,begin,end,failed);
            },&failed));
        }}catch(...){failed=true;error=std::current_exception();}
        for(auto& t:tasks)try{t.get();}catch(...){if(!error)error=std::current_exception();}
        if(error||cancel.load()){
            if(!error)error=std::make_exception_ptr(std::runtime_error("h3_cuda_cancelled"));
            rethrow_after_cleanup(error,[&]{release_scratch_locked();});
        }
    }
    static void validate(std::span<const float> x,std::span<const float> bias,std::size_t in,std::size_t out,std::span<float> y){
        require(in&&out&&in<=std::size_t(INT_MAX)&&out<=std::size_t(INT_MAX)&&!x.empty()&&x.size()%in==0&&y.size()==product(x.size()/in,out)&&(bias.empty()||bias.size()==out),"h3_cuda_projection_shape");
        for(float v:x)require(std::isfinite(v),"h3_cuda_nonfinite_input");for(float v:bias)require(std::isfinite(v),"h3_cuda_nonfinite_bias");
    }
    template<class Weight>void project(std::span<const Weight> weights,std::span<const float> scales,std::span<const float> bias,std::span<const float> x,std::size_t in,std::size_t out,std::size_t group,std::span<float> y,const std::atomic_bool& cancel,bool precise=false,const WeightIdentity* identity=nullptr,const std::function<void(std::size_t,std::span<Weight>)>* source=nullptr){
        require(!cancel.load(),"h3_cuda_cancelled");validate(x,bias,in,out,y);require(!overlap(y.data(),y.size_bytes(),x.data(),x.size_bytes())&&!overlap(y.data(),y.size_bytes(),weights.data(),weights.size_bytes())&&!overlap(y.data(),y.size_bytes(),bias.data(),bias.size_bytes())&&!overlap(y.data(),y.size_bytes(),scales.data(),scales.size_bytes()),"h3_cuda_alias");require((source&&identity)||weights.size()==product(in,out),"h3_cuda_weight_shape");constexpr bool quant=sizeof(Weight)==1;
        if constexpr(quant){require((group==64||group==256)&&in%group==0&&in<=std::size_t(INT_MAX/(128*127))&&out%4==0&&scales.size()==out,"h3_cuda_quant_shape");for(float v:scales)require(std::isfinite(v)&&v>=0,"h3_cuda_scale");}

        const auto total_out=out;const bool columns=identity||(column_split_&&!quant&&projection_split_columns(x.size()/in,devices_.size()));
        const auto all_weights=weights;const auto all_bias=bias;const auto all_scales=scales;std::mutex source_mutex;
        split(columns?(quant?out/4:out):x.size()/in,cancel,[&](DeviceWorkspace& device_workspace,std::size_t begin,std::size_t end,const std::atomic_bool& failed){
            if constexpr(quant){if(columns){begin*=4;end*=4;}}
            const auto first_output=columns?begin:0,last_output=columns?end:total_out;
            if(columns){begin=0;end=x.size()/in;}
            auto& arena=static_cast<CudaDeviceOperations&>(device_workspace.operations());
            const auto plan=projection_plan(in,last_output-first_output,sizeof(Weight),precise,quant?4:1,arena.scratch_limit);
            const auto chunk_columns=plan.columns;
            for(auto first_column=first_output;first_column<last_output;first_column+=chunk_columns){
            const auto out=std::min(chunk_columns,last_output-first_column);
            auto weights=source?std::span<const Weight>{}:all_weights.subspan(first_column*in,out*in);
            const auto bias=all_bias.empty()?all_bias:all_bias.subspan(first_column,out);
            const auto scales=all_scales.empty()?all_scales:all_scales.subspan(first_column,out);
            stop(cancel,failed);const auto tile=std::min(plan.rows,end-begin);
            // Establish device/thread ownership before any retained allocation.
            device_workspace.ensure(256);
            const auto weight_count=product(in,out);
            const DeviceWeights::Key key{identity?*identity:WeightIdentity{},first_column*in,weight_count,sizeof(Weight)};
            void* retained=identity?arena.weights.find(key):nullptr;
            std::unique_ptr<HostAdmission> host_admission;std::unique_ptr<Weight[]> host_weights;
            if(!retained){
                if(source){
                    host_admission=std::make_unique<HostAdmission>(*resources_,weight_count*sizeof(Weight)+4096,workload_);
                    host_weights=std::make_unique_for_overwrite<Weight[]>(weight_count);
                    {std::lock_guard source_lock(source_mutex);stop(cancel,failed);(*source)(first_column*in,{host_weights.get(),weight_count});}
                    weights={host_weights.get(),weight_count};
                }
                if constexpr(!quant)for(float value:weights)require(std::isfinite(value),"h3_cuda_nonfinite_weight");
                if(identity)retained=arena.weights.retain(key,std::as_bytes(weights),cancel);
            }
            Layout l;
            const auto w=l.add(retained?0:weights.size_bytes()),b=l.add(out*4),s=l.add(out*4),input=l.add(tile*in*4),rotated=l.add(tile*in*4),codes=l.add(tile*in),row_scales=l.add(tile*4),sums=l.add(tile*out*4),output=l.add(tile*out*4),workspace=l.add(8*1024*1024),wide_weights=l.add(precise?weight_count*8:0),wide_input=l.add(precise?tile*in*8:0),wide_output=l.add(precise?tile*out*8:0);
            device_workspace.ensure(l.bytes);checked(cublasSetWorkspace(arena.blas,arena.at<void>(workspace),8*1024*1024));
            auto* weight_pointer=retained?retained:arena.at<void>(w);
            if(!retained)checked(cudaMemcpy(weight_pointer,weights.data(),weights.size_bytes(),cudaMemcpyHostToDevice));
            if(bias.empty())checked(cudaMemset(arena.at<void>(b),0,out*4));else checked(cudaMemcpy(arena.at<void>(b),bias.data(),out*4,cudaMemcpyHostToDevice));
            if constexpr(quant)checked(cudaMemcpy(arena.at<void>(s),scales.data(),out*4,cudaMemcpyHostToDevice));
            else if(precise){widen<<<unsigned((weight_count+255)/256),256>>>(static_cast<float*>(weight_pointer),arena.at<double>(wide_weights),weight_count);checked(cudaGetLastError());}
            for(std::size_t start=begin;start<end;start+=tile){stop(cancel,failed);const auto rows=std::min(tile,end-start);checked(cudaMemcpy(arena.at<void>(input),x.data()+start*in,rows*in*4,cudaMemcpyHostToDevice));
                if constexpr(quant){
                    rotate_kernel<<<dim3(unsigned(rows),unsigned(in/group)),unsigned(group)>>>(arena.at<float>(input),arena.at<float>(rotated),in,unsigned(group));checked(cudaGetLastError());
                    quantize_kernel<<<unsigned(rows),256>>>(arena.at<float>(rotated),arena.at<std::int8_t>(codes),arena.at<float>(row_scales),in);checked(cudaGetLastError());
                    const std::int32_t one=1,zero=0;checked(cublasGemmEx(arena.blas,CUBLAS_OP_T,CUBLAS_OP_N,int(out),int(rows),int(in),&one,weight_pointer,CUDA_R_8I,int(in),arena.at<void>(codes),CUDA_R_8I,int(in),&zero,arena.at<void>(sums),CUDA_R_32I,int(out),CUBLAS_COMPUTE_32I,CUBLAS_GEMM_DEFAULT));
                    finish_int8<<<unsigned((rows*out+255)/256),256>>>(arena.at<std::int32_t>(sums),arena.at<float>(row_scales),arena.at<float>(s),arena.at<float>(b),arena.at<float>(output),rows*out,out);
                }else if(precise){
                    widen<<<unsigned((rows*in+255)/256),256>>>(arena.at<float>(input),arena.at<double>(wide_input),rows*in);checked(cudaGetLastError());
                    checked(gemm<double>(arena.blas,CUBLAS_OP_T,CUBLAS_OP_N,int(out),int(rows),int(in),1.0,arena.at<double>(wide_weights),int(in),arena.at<double>(wide_input),int(in),0.0,arena.at<double>(wide_output),int(out)));
                    narrow_bias<<<unsigned((rows*out+255)/256),256>>>(arena.at<double>(wide_output),arena.at<float>(b),arena.at<float>(output),rows*out,out);
                }else{
                    const float one=1,zero=0;checked(cublasSgemm(arena.blas,CUBLAS_OP_T,CUBLAS_OP_N,int(out),int(rows),int(in),&one,static_cast<float*>(weight_pointer),int(in),arena.at<float>(input),int(in),&zero,arena.at<float>(output),int(out)));
                    add_bias<<<unsigned((rows*out+255)/256),256>>>(arena.at<float>(output),arena.at<float>(b),rows*out,out);
                }
                checked(cudaGetLastError());checked(cudaStreamSynchronize(cudaStreamLegacy));stop(cancel,failed);checked(cudaMemcpy2D(y.data()+start*total_out+first_column,total_out*4,arena.at<void>(output),out*4,out*4,rows,cudaMemcpyDeviceToHost));
            }
            }
        },identity!=nullptr);
        for(float v:y)require(std::isfinite(v),"h3_cuda_nonfinite_output");
    }
public:
    Compute(std::shared_ptr<Resources> r,std::vector<int> devices,Workload workload=Workload::video,bool columns=false):resources_(std::move(r)),devices_(std::move(devices)),workload_(workload),column_split_(columns){
        require(bool(resources_)&&!devices_.empty(),"h3_cuda_devices");const auto capacity=resources_->snapshot().capacity;
        for(std::size_t i=0;i<devices_.size();++i){const int d=devices_[i];require(d>=0&&std::size_t(d)+1<capacity.size()&&capacity[d+1]>0,"h3_cuda_unbudgeted_device");require(std::find(devices_.begin(),devices_.begin()+i,d)==devices_.begin()+i,"h3_cuda_duplicate_device");}
        weight_reservations_.resize(capacity.size());weight_capacities_.resize(devices_.size());
        const auto used=resources_->snapshot().used;
        for(int device:devices_){
            const auto available=capacity[device+1]-used[device+1];
            const Bytes scratch=std::min<Bytes>(2ULL*1024*1024*1024,available);
            const Bytes limit=available>scratch?available-scratch:0;
            threads_.push_back(std::make_unique<DeviceThread>(*resources_,device,workload_,std::make_unique<CudaDeviceOperations>(*resources_,device,workload_,limit,scratch)));
        }
    }
    template<class Real> void attention_impl(std::span<const float> q,std::span<const float> k,std::span<const float> v,std::size_t n,std::size_t heads,std::size_t kv_heads,std::size_t dim,bool causal,std::span<float> y,const std::atomic_bool& cancel,std::size_t key_tokens=0,std::size_t causal_queries=0,std::size_t window=0,std::size_t offset=0){
        require(!cancel.load(),"h3_cuda_cancelled");if(!key_tokens)key_tokens=n;if(causal)causal_queries=n;
        require(n>0&&n<=std::size_t(INT_MAX)&&heads>0&&kv_heads>0&&heads%kv_heads==0&&dim>0&&dim<=2048,"h3_cuda_attention_shape");
        require(key_tokens>0&&key_tokens<=std::size_t(INT_MAX)&&causal_queries<=n&&offset<=key_tokens&&window<=key_tokens&&(!causal_queries||causal_queries<=key_tokens-offset),"h3_cuda_attention_shape");
        const auto head_values=product(n,dim),key_values=product(key_tokens,dim);require(q.size()==product(head_values,heads)&&k.size()==product(key_values,kv_heads)&&v.size()==k.size()&&y.size()==q.size(),"h3_cuda_attention_shape");
        for(auto input:{q,k,v}){require(!overlap(y.data(),y.size_bytes(),input.data(),input.size_bytes()),"h3_cuda_alias");for(float x:input)require(std::isfinite(x),"h3_cuda_nonfinite_input");}
        split(heads,cancel,[&](DeviceWorkspace& device_workspace,std::size_t first,std::size_t last,const std::atomic_bool& failed){
            stop(cancel,failed);auto& arena=static_cast<CudaDeviceOperations&>(device_workspace.operations());
            const auto plan=attention_plan(n,key_tokens,dim,sizeof(Real),arena.scratch_limit);
            const auto query_tile=plan.queries,key_tile=plan.keys;
            const auto query_storage=(plan.full_heads?n:query_tile)*dim,key_storage=(plan.full_heads?key_tokens:key_tile)*dim;Layout l;
            const auto dq=l.add(query_storage*sizeof(Real)),dk=l.add(key_storage*sizeof(Real)),dv=l.add(key_storage*sizeof(Real)),dy=l.add(query_tile*dim*sizeof(Real)),scores=l.add(query_tile*key_tile*sizeof(Real)),maximum=l.add(query_tile*sizeof(Real)),total=l.add(query_tile*sizeof(Real)),workspace=l.add(8*1024*1024);
            HostAdmission admission(*resources_,(query_storage+key_storage*2+query_tile*dim)*sizeof(Real),workload_);
            auto memory=std::make_unique_for_overwrite<Real[]>(query_storage+key_storage*2+query_tile*dim);auto hq=memory.get(),hk=hq+query_storage,hv=hk+key_storage,hy=hv+key_storage;
            device_workspace.ensure(l.bytes);checked(cublasSetWorkspace(arena.blas,arena.at<void>(workspace),8*1024*1024));
            for(std::size_t h=first;h<last;++h){stop(cancel,failed);const auto kh=h/(heads/kv_heads);
                auto pack_query=[&](std::size_t start,std::size_t rows){
                    for(std::size_t t=0;t<rows;++t){if(t%64==0)stop(cancel,failed);for(std::size_t c=0;c<dim;++c)hq[t*dim+c]=q[((start+t)*heads+h)*dim+c];}
                    checked(cudaMemcpy(arena.at<void>(dq),hq,rows*dim*sizeof(Real),cudaMemcpyHostToDevice));
                };
                auto pack_keys=[&](std::size_t start,std::size_t rows){
                    for(std::size_t t=0;t<rows;++t){if(t%64==0)stop(cancel,failed);for(std::size_t c=0;c<dim;++c){hk[t*dim+c]=k[((start+t)*kv_heads+kh)*dim+c];hv[t*dim+c]=v[((start+t)*kv_heads+kh)*dim+c];}}
                    checked(cudaMemcpy(arena.at<void>(dk),hk,rows*dim*sizeof(Real),cudaMemcpyHostToDevice));checked(cudaMemcpy(arena.at<void>(dv),hv,rows*dim*sizeof(Real),cudaMemcpyHostToDevice));
                };
                if(plan.full_heads){pack_query(0,n);pack_keys(0,key_tokens);}
                for(std::size_t start=0;start<n;start+=query_tile){stop(cancel,failed);const auto rows=std::min(query_tile,n-start);if(!plan.full_heads)pack_query(start,rows);checked(cudaMemset(arena.at<void>(dy),0,rows*dim*sizeof(Real)));initialize_attention<<<1,128>>>(arena.at<Real>(maximum),arena.at<Real>(total),unsigned(rows));checked(cudaGetLastError());
                    for(std::size_t key=0;key<key_tokens;key+=key_tile){stop(cancel,failed);const auto keys=std::min(key_tile,key_tokens-key);if(!plan.full_heads)pack_keys(key,keys);const Real scale=1/std::sqrt(Real(dim)),zero=0,one=1;
                        checked(gemm<Real>(arena.blas,CUBLAS_OP_T,CUBLAS_OP_N,int(keys),int(rows),int(dim),scale,arena.at<Real>(dk)+(plan.full_heads?key*dim:0),int(dim),arena.at<Real>(dq)+(plan.full_heads?start*dim:0),int(dim),zero,arena.at<Real>(scores),int(keys)));
                        online_softmax<<<unsigned(rows),256>>>(arena.at<Real>(scores),arena.at<Real>(dy),arena.at<Real>(maximum),arena.at<Real>(total),unsigned(keys),unsigned(dim),unsigned(start),unsigned(key),unsigned(causal_queries),unsigned(window),unsigned(offset));checked(cudaGetLastError());
                        checked(gemm<Real>(arena.blas,CUBLAS_OP_N,CUBLAS_OP_N,int(dim),int(rows),int(keys),one,arena.at<Real>(dv)+(plan.full_heads?key*dim:0),int(dim),arena.at<Real>(scores),int(keys),one,arena.at<Real>(dy),int(dim)));checked(cudaStreamSynchronize(cudaStreamLegacy));
                    }
                    divide_attention<<<unsigned((rows*dim+255)/256),256>>>(arena.at<Real>(dy),arena.at<Real>(total),unsigned(rows),unsigned(dim));checked(cudaGetLastError());checked(cudaMemcpy(hy,arena.at<void>(dy),rows*dim*sizeof(Real),cudaMemcpyDeviceToHost));stop(cancel,failed);
                    for(std::size_t t=0;t<rows;++t)for(std::size_t c=0;c<dim;++c)y[((start+t)*heads+h)*dim+c]=hy[t*dim+c];
                }
            }
        });
        for(float x:y)require(std::isfinite(x),"h3_cuda_nonfinite_output");
    }
    void attention(std::span<const float> q,std::span<const float> k,std::span<const float> v,std::size_t n,std::size_t heads,std::size_t kv,std::size_t dim,bool causal,std::span<float> y,const std::atomic_bool& cancel,bool precise)override{
        if(precise)attention_impl<double>(q,k,v,n,heads,kv,dim,causal,y,cancel);else attention_impl<float>(q,k,v,n,heads,kv,dim,causal,y,cancel);
    }
    bool attend(std::span<const float> q,std::span<const float> k,std::span<const float> v,std::size_t queries,std::size_t keys,std::size_t heads,std::size_t kv,std::size_t dim,std::size_t causal_queries,std::size_t window,std::size_t offset,std::span<float> y,const std::atomic_bool& cancel) override {
        attention_impl<float>(q,k,v,queries,heads,kv,dim,false,y,cancel,keys,causal_queries,window,offset);return true;
    }
    void release_scratch() override {
        std::lock_guard lock(operation_mutex_);release_scratch_locked();
    }
    void release_devices() override {
        std::lock_guard lock(operation_mutex_);
        require(!quarantined_.load(),"h3_cuda_cleanup_unconfirmed");
        all_devices([](DeviceWorkspace& workspace){workspace.reset();});
        std::fill(weight_reservations_.begin(),weight_reservations_.end(),0);replan_=inventory_bytes_!=0;
    }
    Footprint retained_weights() const override {std::lock_guard lock(operation_mutex_);return weight_reservations_;}
    std::vector<DeviceWeightPlacement> weight_placement() const override {
        std::lock_guard lock(operation_mutex_);std::vector<DeviceWeightPlacement> result;result.reserve(threads_.size());
        for(std::size_t i=0;i<threads_.size();++i)threads_[i]->submit([&,i](DeviceWorkspace& workspace){
            const auto& bank=static_cast<CudaDeviceOperations&>(workspace.operations()).weights;
            result.push_back({devices_[i],bank.capacity(),bank.allocated()});
        }).get();
        return result;
    }
    bool prepare_weights(Bytes bytes) override {
        std::lock_guard lock(operation_mutex_);require(!quarantined_.load(),"h3_cuda_cleanup_unconfirmed");
        inventory_bytes_=bytes;return plan_weights_locked();
    }
private:
    bool plan_weights_locked(){
        const auto bytes=inventory_bytes_;
        // Plan from free physical memory on each owned device thread. Freeze the
        // partition until park/replan so cached identities keep stable slices.
        Bytes total=0;
        for(std::size_t i=0;i<threads_.size();++i)threads_[i]->submit([&,i](DeviceWorkspace& workspace){
            workspace.activate();
            auto& ops=static_cast<CudaDeviceOperations&>(workspace.operations());
            weight_capacities_[i]=std::min(ops.weights.maximum_capacity(),ops.available_weight_bytes());
            total+=weight_capacities_[i];
        }).get();
        for(std::size_t i=0;i<threads_.size();++i)threads_[i]->submit([&,i](DeviceWorkspace& workspace){
            auto& bank=static_cast<CudaDeviceOperations&>(workspace.operations()).weights;
            // A ceiling per bank leaves room for tensor channel rounding.
            const auto share=weight_boundary(bytes,weight_capacities_,i+1)-weight_boundary(bytes,weight_capacities_,i);
            bank.bound(std::min(weight_capacities_[i],share+std::min<Bytes>(16*1024*1024,std::numeric_limits<Bytes>::max()-share)));
        }).get();
        replan_=false;return total>=bytes;
    }
public:
    bool supports_weight_sources() const override {return true;}
    bool dense_source(const WeightIdentity& identity,const FloatWeightSource& source,std::span<const float> b,std::span<const float> x,std::size_t in,std::size_t out,std::span<float> y,const std::atomic_bool& c,bool precise)override{project<float>({}, {},b,x,in,out,0,y,c,precise,&identity,&source);return true;}
    bool convrot_source(const WeightIdentity& identity,const ByteWeightSource& source,std::span<const float> s,std::span<const float> b,std::span<const float> x,std::size_t in,std::size_t out,std::size_t g,std::span<float> y,const std::atomic_bool& c)override{project<std::uint8_t>({},s,b,x,in,out,g,y,c,false,&identity,&source);return true;}
    void dense_weight(const WeightIdentity& identity,std::span<const float> w,std::span<const float> b,std::span<const float> x,std::size_t in,std::size_t out,std::span<float> y,const std::atomic_bool& c,bool precise)override{project(w,{},b,x,in,out,0,y,c,precise,&identity);}
    void dense(std::span<const float> w,std::span<const float> b,std::span<const float> x,std::size_t in,std::size_t out,std::span<float> y,const std::atomic_bool& c,bool precise)override{project(w,{},b,x,in,out,0,y,c,precise);}
    void convrot(std::span<const std::uint8_t> w,std::span<const float> s,std::span<const float> b,std::span<const float> x,std::size_t in,std::size_t out,std::size_t g,std::span<float> y,const std::atomic_bool& c)override{project(w,s,b,x,in,out,g,y,c);}
    void convrot_weight(const WeightIdentity& identity,std::span<const std::uint8_t> w,std::span<const float> s,std::span<const float> b,std::span<const float> x,std::size_t in,std::size_t out,std::size_t g,std::span<float> y,const std::atomic_bool& c)override{project(w,s,b,x,in,out,g,y,c,false,&identity);}
};
}
std::shared_ptr<H3Compute> h3_cuda_compute(std::shared_ptr<Resources> r,std::vector<int> d){return std::make_shared<Compute>(std::move(r),std::move(d));}
}

namespace kadan {
std::shared_ptr<DenseCompute> cuda_dense_compute(std::shared_ptr<Resources> r,std::vector<int> devices,Workload workload){
 return std::make_shared<video::Compute>(std::move(r),std::move(devices),workload,true);
}
}
