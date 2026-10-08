#ifdef CUDA_API_PER_THREAD_DEFAULT_STREAM
#error "Decoder requires legacy default stream"
#endif
#include "kadan/cuda_decoder.hpp"
#include "linear_kernel.cuh"
#include "full_kernel.cuh"
#include "moe_kernel.cuh"
#include "decoder_kernel.cuh"
#include "fp8_kernel.cuh"
#include "nvfp4_kernel.cuh"
#include "layer_math_validation.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <thread>
#include <stdexcept>
namespace kadan::cuda {
namespace {
void require(bool x,const char* e){if(!x)throw std::invalid_argument(e);}
void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
struct Projection {const std::uint8_t *weights=nullptr,*scales=nullptr;float global=0;std::size_t rows=0,columns=0;};
}
struct Decoder::Impl {
    decoder::Config c;decoder::Plan p;std::shared_ptr<Resources> resources;int device;StateCursor cursor;
    std::thread::id owner=std::this_thread::get_id();Handle handle;void* storage=nullptr;
    bool loaded=false,pinned=false,poisoned=false,cleanup_failed=false,ready=false;
    detail::LinearBuffers linear{};detail::FullBuffers full{};detail::MoeBuffers moe{};
    std::array<std::array<Projection,3>,257> experts{};
    Impl(decoder::Config config,decoder::Plan layout,int d,std::shared_ptr<Resources> r,Handle h):c(config),p(layout),resources(std::move(r)),device(d),cursor(1,p.capacity),handle(h){}
    std::uint8_t* bytes(std::size_t offset){return static_cast<std::uint8_t*>(storage)+offset;}
    template<class T>T* pointer(std::size_t offset){return reinterpret_cast<T*>(bytes(offset));}
    float* work(std::size_t index){return pointer<float>(p.workspace)+index*p.hidden;}
    unsigned* flags(){return pointer<unsigned>(p.status);}
    void current(){require(std::this_thread::get_id()==owner,"decoder_thread_changed");int actual=-1;check(cudaGetDevice(&actual));require(actual==device,"decoder_device_changed");}
    void available(){require(handle&&!poisoned,"decoder_unavailable");try{current();}catch(const std::runtime_error&){poisoned=true;cursor.invalidate();throw;}}
    const std::uint16_t* upload(std::span<const float> values,std::size_t offset){
        auto* dst=pointer<std::uint16_t>(offset);std::array<std::uint16_t,1024> staging{};
        for(std::size_t at=0;at<values.size();at+=staging.size()){auto n=std::min(staging.size(),values.size()-at);for(std::size_t j=0;j<n;++j)staging[j]=std::uint16_t(std::bit_cast<std::uint32_t>(values[at+j])>>16);check(cudaMemcpy(dst+at,staging.data(),n*2,cudaMemcpyHostToDevice));}return dst;
    }
    void initialize(const decoder::Weights& w){
        current();int major=0,minor=0;check(cudaDeviceGetAttribute(&major,cudaDevAttrComputeCapabilityMajor,device));check(cudaDeviceGetAttribute(&minor,cudaDevAttrComputeCapabilityMinor,device));require(major==8&&minor==6,"requires_sm86");
        void* allocation=nullptr;check(cudaMalloc(&allocation,p.device_bytes));storage=allocation;
        const bool is_linear=c.attention==decoder::Attention::linear;
        const std::array<quantization::Matrix,4> projections=is_linear?std::array<quantization::Matrix,4>{w.linear.qkv,w.linear.z,w.linear.out,{}}:std::array<quantization::Matrix,4>{w.full.q_gate,w.full.key,w.full.value,w.full.out};
        for(std::size_t i=0;i<p.projection_count;++i){check(cudaMemcpy(bytes(p.projections[i].weights),projections[i].weights.data(),projections[i].weights.size_bytes(),cudaMemcpyHostToDevice));check(cudaMemcpy(bytes(p.projections[i].scale),projections[i].multipliers.data(),4,cudaMemcpyHostToDevice));}
        auto offset=p.auxiliary;auto aux=[&](std::span<const float> v){auto* result=upload(v,offset);offset+=v.size()*2;return result;};
        auto* scratch=pointer<float>(p.attention_scratch);
        if(is_linear){
            linear.input_norm=aux(w.linear.input_norm);linear.conv_weight=aux(w.linear.conv);linear.a_weight=aux(w.linear.a);linear.b_weight=aux(w.linear.b);linear.a_log=aux(w.linear.a_log);linear.dt_bias=aux(w.linear.dt_bias);linear.output_norm=aux(w.linear.output_norm);
            const auto& l=p.linear;linear.normalized=scratch+l.norm_offset;linear.qkv=scratch+l.qkv_offset;linear.z=scratch+l.z_offset;linear.a=scratch+l.a_offset;linear.b=scratch+l.b_offset;linear.core=scratch+l.core_offset;linear.gated=scratch+l.gate_offset;linear.projected=scratch+l.out_offset;
            linear.convolution=pointer<std::uint16_t>(p.state_first);linear.recurrent=pointer<float>(p.state_second);linear.status=flags();
        }else{
            full.input_norm=aux(w.full.input_norm);full.query_norm=aux(w.full.query_norm);full.key_norm=aux(w.full.key_norm);
            full.frequencies=pointer<float>(p.frequencies);check(cudaMemcpy(bytes(p.frequencies),w.full.frequencies.data(),w.full.frequencies.size_bytes(),cudaMemcpyHostToDevice));
            const auto& f=p.full;full.normalized=scratch+f.norm_offset;full.qg=scratch+f.qg_offset;full.key=scratch+f.k_offset;full.value=scratch+f.v_offset;full.query=scratch+f.q_offset;full.gate=scratch+f.gate_offset;full.probabilities=scratch+f.prob_offset;full.core=scratch+f.core_offset;full.gated=scratch+f.gated_offset;full.projected=scratch+f.out_offset;
            full.keys=pointer<std::uint16_t>(p.state_first);full.values=pointer<std::uint16_t>(p.state_second);full.status=flags();
        }
        upload(w.post_norm,p.post_norm);const auto& m=p.moe;auto base=p.moe_offset;
        moe.router=upload(w.moe.router,base+m.router_offset);moe.shared_gate=upload(w.moe.shared_gate,base+m.shared_gate_offset);
        auto matrix=[&](const quantization::Matrix& source,std::size_t at,std::size_t weights,std::size_t scales){
            check(cudaMemcpy(bytes(at+weights),source.weights.data(),source.weights.size_bytes(),cudaMemcpyHostToDevice));check(cudaMemcpy(bytes(at+scales),source.block_scales.data(),source.block_scales.size_bytes(),cudaMemcpyHostToDevice));
            return Projection{bytes(at+weights),bytes(at+scales),source.multipliers[0],source.rows,source.columns};};
        for(std::size_t e=0;e<=c.moe.experts;++e){const bool shared=e==c.moe.experts;const auto& source=shared?w.moe.shared:w.moe.experts[e];const auto& l=shared?m.shared_layout:m.routed_layout;const auto at=base+(shared?m.shared_offset:m.experts_offset+e*m.routed_layout.bytes);
            experts[e]={matrix(source.gate,at,l.gate_weights,l.gate_scales),matrix(source.up,at,l.up_weights,l.up_scales),matrix(source.down,at,l.down_weights,l.down_scales)};}
        auto* s=pointer<float>(base+m.scratch_offset);moe.logits=s+m.logits;moe.probabilities=s+m.probabilities;moe.top_weights=s+m.top_weights;moe.gate=s+m.gate;moe.up=s+m.up;moe.activation=s+m.activation;moe.down=s+m.down;moe.accumulator=s+m.accumulator;moe.shared=s+m.shared;moe.result=s+m.result;moe.shared_factor=s+m.shared_factor;moe.selected=pointer<unsigned>(base+m.indices_offset);moe.status=flags();
        zero();check(cudaStreamSynchronize(cudaStreamLegacy));resources->loaded(handle);loaded=true;
    }
    void zero(){check(cudaMemsetAsync(bytes(p.moe_offset+p.moe.scratch_offset),0,p.moe.device_bytes-p.moe.scratch_offset,cudaStreamLegacy));check(cudaMemsetAsync(bytes(p.state_first),0,p.device_bytes-p.state_first,cudaStreamLegacy));}
    void pin(){resources->pin(handle);pinned=true;}
    void unpin(){resources->unpin(handle);pinned=false;}
    void status(){check(cudaStreamSynchronize(cudaStreamLegacy));unsigned value=0;check(cudaMemcpy(&value,flags(),4,cudaMemcpyDeviceToHost));if(value)throw std::overflow_error("decoder_numeric_failure");}
    void fp8(std::size_t i,const float* x,float* y){const auto& v=p.projections[i];check(kadan_launch_fp8(bytes(v.weights),pointer<float>(v.scale),false,x,y,flags(),v.rows,v.columns));status();}
    // Private borrowed producers: coordinator validates the owner-bound active
    // capability on entry. No child owns admission, state lifetime or progress.
    void attention(StateStep token,const float* input){
        cursor.check_step(token);
        if(c.attention==decoder::Attention::linear){
            check(detail::linear_normalize(c.linear,linear,input));status();fp8(0,linear.normalized,linear.qkv);fp8(1,linear.normalized,linear.z);
            check(detail::linear_core(c.linear,linear));status();fp8(2,linear.gated,linear.projected);check(detail::linear_residual(c.linear,linear,input,work(0)));status();
        }else{
            check(detail::full_normalize(c.full,full,input));status();fp8(0,full.normalized,full.qg);fp8(1,full.normalized,full.key);fp8(2,full.normalized,full.value);
            check(detail::full_core(c.full,full,cursor.committed_tokens()));status();fp8(3,full.gated,full.projected);check(detail::full_residual(c.full,full,input,work(0)));status();
        }
    }
    void project(const Projection& m,const float* x,float* y){check(kadan_launch_nvfp4(m.weights,m.scales,m.global,x,y,flags(),m.rows,m.columns));}
    void expert(std::size_t e,std::size_t middle){project(experts[e][0],work(1),moe.gate);project(experts[e][1],work(1),moe.up);check(detail::moe_activate(middle,moe));status();project(experts[e][2],moe.activation,moe.down);}
    void mixture(StateStep token){
        cursor.check_step(token);check(detail::moe_route(c.moe,moe,work(1)));status();std::array<unsigned,8> selected{};
        check(cudaMemcpy(selected.data(),moe.selected,c.moe.top_k*sizeof(unsigned),cudaMemcpyDeviceToHost));std::sort(selected.begin(),selected.begin()+c.moe.top_k);
        for(std::size_t i=0;i<c.moe.top_k;++i){require(selected[i]<c.moe.experts&&(i==0||selected[i]!=selected[i-1]),"decoder_device_route");expert(selected[i],c.moe.intermediate);check(detail::moe_accumulate(c.moe,moe,selected[i]));status();}
        expert(c.moe.experts,c.moe.shared_intermediate);check(detail::moe_finish(c.moe,moe,work(2)));status();
    }
    void cancel(const std::atomic_bool* flag){if(flag&&flag->load(std::memory_order_relaxed))throw std::invalid_argument("decoder_cancelled");}
    bool invalidate()noexcept{
        cursor.invalidate();ready=false;if(!pinned)return true;
        try{check(cudaStreamSynchronize(cudaStreamLegacy));unpin();return true;}catch(...){poisoned=true;return false;}
    }
    void step(std::span<const float> x,std::span<float> y,const std::atomic_bool* cancelled){
        available();const auto token=cursor.begin();ready=false;
        try{
            require(x.size()==p.hidden&&y.size()==p.hidden,"decoder_input_shape");math::detail::output_alias(x,y,true);
            require(reinterpret_cast<std::uintptr_t>(x.data())%alignof(float)==0&&reinterpret_cast<std::uintptr_t>(y.data())%alignof(float)==0,"decoder_alignment");cancel(cancelled);
            pin();check(cudaMemsetAsync(flags(),0,4,cudaStreamLegacy));attention(token,x.data());cancel(cancelled);
            check(detail::decoder_norm(p.hidden,p.epsilon,pointer<std::uint16_t>(p.post_norm),work(0),work(1),flags()));status();mixture(token);cancel(cancelled);
            check(detail::decoder_residual(p.hidden,work(0),work(2),work(3),flags()));status();cancel(cancelled);
            cursor.written(token,0);cursor.ready_to_commit(token);
            check(cudaMemcpy(y.data(),work(3),p.hidden*4,cudaMemcpyDeviceToDevice));check(cudaStreamSynchronize(cudaStreamLegacy));cancel(cancelled);
            unpin();
            // Prevalidated, same-thread host transition. No fallible CUDA or
            // resource operation follows publication. Output is valid on return.
            cursor.commit(token);ready=true;
        }catch(const std::invalid_argument&){if(!invalidate())throw DeviceBufferQuarantine("decoder_borrowed_buffers_quarantined");throw;}
        catch(const std::overflow_error&){if(!invalidate())throw DeviceBufferQuarantine("decoder_borrowed_buffers_quarantined");throw;}
        catch(...){poisoned=true;if(!invalidate())throw DeviceBufferQuarantine("decoder_borrowed_buffers_quarantined");throw;}
    }
    void reset(){available();require(!cursor.active(),"decoder_active");pin();ready=false;try{zero();check(cudaStreamSynchronize(cudaStreamLegacy));unpin();cursor.reset();}catch(...){poisoned=true;cursor.invalidate();throw;}}
    bool cleanup()noexcept{
        if(!handle)return true;
        if(cleanup_failed)return false;
        try{if(storage){current();check(cudaStreamSynchronize(cudaStreamLegacy));}if(pinned)unpin();if(loaded){resources->begin_eviction(handle);loaded=false;}if(storage){check(cudaFree(storage));storage=nullptr;}resources->released(handle);handle=0;cursor.close();ready=false;return true;}
        catch(...){cleanup_failed=poisoned=true;cursor.invalidate();return false;}
    }
};
Decoder::Decoder(decoder::Config c,const decoder::Weights& w,int device,std::shared_ptr<Resources> r){
    const auto p=decoder::plan(c);decoder::validate_weights(c,w);require(r&&device>=0,"decoder_owner");auto request=r->snapshot().capacity;require(std::size_t(device)+1<request.size(),"decoder_unbudgeted_device");
    std::fill(request.begin(),request.end(),0);request[0]=sizeof(Impl);request[device+1]=p.device_bytes;auto h=r->reserve(Workload::llm,request);
    // Reserve BEFORE allocating charged metadata; construction failure before
    // ownership transfers has no CUDA work and releases the loading reservation.
    try{impl_=std::make_unique<Impl>(c,p,device,r,h);}catch(...){r->released(h);throw;}
    try{impl_->initialize(w);}catch(...){impl_->cleanup();throw;}
}
Decoder::~Decoder(){if(impl_)impl_->cleanup();}
std::size_t Decoder::host_metadata_bytes(){return sizeof(Impl);}
void Decoder::step_device(std::span<const float>x,std::span<float>y,const std::atomic_bool*c){require(bool(impl_),"decoder_closed");impl_->step(x,y,c);}
bool Decoder::valid()const{return impl_&&impl_->handle&&!impl_->poisoned&&impl_->cursor.valid();}
std::size_t Decoder::tokens()const{return impl_?impl_->cursor.committed_tokens():0;}
void Decoder::reset(){require(bool(impl_),"decoder_closed");impl_->reset();}
void Decoder::read_intermediates(std::span<float>a,std::span<float>u,std::span<float>m){
    require(bool(impl_),"decoder_closed");auto& i=*impl_;i.available();require(valid()&&i.ready&&a.size()==i.p.hidden&&u.size()==i.p.hidden&&m.size()==i.p.hidden,"decoder_diagnostic");i.pin();
    try{check(cudaMemcpy(a.data(),i.work(0),a.size_bytes(),cudaMemcpyDeviceToHost));check(cudaMemcpy(u.data(),i.work(1),u.size_bytes(),cudaMemcpyDeviceToHost));check(cudaMemcpy(m.data(),i.work(2),m.size_bytes(),cudaMemcpyDeviceToHost));i.unpin();}catch(...){i.poisoned=true;i.cursor.invalidate();throw;}
}
void Decoder::read_state(std::span<std::uint8_t>a,std::span<std::uint8_t>b){
    require(bool(impl_),"decoder_closed");auto& i=*impl_;i.available();require(valid()&&!i.cursor.active()&&a.size()==i.p.state_first_bytes&&b.size()==i.p.state_second_bytes,"decoder_state_read");i.pin();
    try{check(cudaMemcpy(a.data(),i.bytes(i.p.state_first),a.size_bytes(),cudaMemcpyDeviceToHost));check(cudaMemcpy(b.data(),i.bytes(i.p.state_second),b.size_bytes(),cudaMemcpyDeviceToHost));i.unpin();}catch(...){i.poisoned=true;i.cursor.invalidate();throw;}
}
void Decoder::close(){if(!impl_)return;if(!impl_->cleanup())throw std::runtime_error("decoder_cleanup_failed_reservation_retained");impl_.reset();}
} // namespace kadan::cuda
