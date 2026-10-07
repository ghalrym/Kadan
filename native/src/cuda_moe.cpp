#ifdef CUDA_API_PER_THREAD_DEFAULT_STREAM
#error "MoE requires legacy default stream"
#endif
#include "kadan/cuda_moe.hpp"
#include "moe_kernel.cuh"
#include "nvfp4_kernel.cuh"
#include "layer_math_validation.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <stdexcept>
#include <thread>
namespace kadan::cuda {
namespace {
void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
void require(bool x,const char* e){if(!x)throw std::invalid_argument(e);}
struct Projection {const std::uint8_t *weights=nullptr,*scales=nullptr;float global=0;std::size_t rows=0,columns=0;};
}
struct Moe::Impl {
    moe::Config c;moe::Plan p;std::shared_ptr<Resources> resources;int device;std::thread::id owner=std::this_thread::get_id();
    Handle handle=0;void* storage=nullptr;bool loaded=false,pinned=false,invalid=false,ready=false,poisoned=false,cleanup_failed=false;
    std::array<std::array<Projection,3>,257> experts{};detail::MoeBuffers b{};
    Impl(moe::Config shape,const moe::Weights& w,int d,std::shared_ptr<Resources> manager):c(shape),p(moe::plan(c)),resources(std::move(manager)),device(d){
        moe::validate_weights(c,w);require(resources&&device>=0,"moe_owner");auto request=resources->snapshot().capacity;
        require(std::size_t(device)+1<request.size(),"moe_unbudgeted_device");std::fill(request.begin(),request.end(),0);request[0]=sizeof(Impl);request[device+1]=p.device_bytes;
        handle=resources->reserve(Workload::llm,request);
        try{
            current();int major=0,minor=0;check(cudaDeviceGetAttribute(&major,cudaDevAttrComputeCapabilityMajor,device));check(cudaDeviceGetAttribute(&minor,cudaDevAttrComputeCapabilityMinor,device));require(major==8&&minor==6,"requires_sm86");
            void* allocation=nullptr;check(cudaMalloc(&allocation,p.device_bytes));storage=allocation;
            auto upload_bf=[&](std::span<const float> values,std::size_t offset){auto* dst=reinterpret_cast<std::uint16_t*>(bytes(offset));std::array<std::uint16_t,1024> staging{};
                for(std::size_t at=0;at<values.size();at+=staging.size()){const auto n=std::min(staging.size(),values.size()-at);for(std::size_t j=0;j<n;++j)staging[j]=std::uint16_t(std::bit_cast<std::uint32_t>(values[at+j])>>16);check(cudaMemcpy(dst+at,staging.data(),n*2,cudaMemcpyHostToDevice));}return dst;};
            b.router=upload_bf(w.router,p.router_offset);b.shared_gate=upload_bf(w.shared_gate,p.shared_gate_offset);
            auto upload=[&](const quantization::Matrix& m,std::size_t weight,std::size_t scale){Projection view{bytes(weight),bytes(scale),m.multipliers[0],m.rows,m.columns};
                check(cudaMemcpy(bytes(weight),m.weights.data(),m.weights.size_bytes(),cudaMemcpyHostToDevice));check(cudaMemcpy(bytes(scale),m.block_scales.data(),m.block_scales.size_bytes(),cudaMemcpyHostToDevice));return view;};
            for(std::size_t e=0;e<=c.experts;++e){const bool shared=e==c.experts;const auto& source=shared?w.shared:w.experts[e];const auto& l=shared?p.shared_layout:p.routed_layout;const auto base=shared?p.shared_offset:p.experts_offset+e*p.routed_layout.bytes;
                experts[e]={upload(source.gate,base+l.gate_weights,base+l.gate_scales),upload(source.up,base+l.up_weights,base+l.up_scales),upload(source.down,base+l.down_weights,base+l.down_scales)};}
            auto* scratch=reinterpret_cast<float*>(bytes(p.scratch_offset));
            b.logits=scratch+p.logits;b.probabilities=scratch+p.probabilities;b.top_weights=scratch+p.top_weights;b.gate=scratch+p.gate;b.up=scratch+p.up;b.activation=scratch+p.activation;
            b.down=scratch+p.down;b.accumulator=scratch+p.accumulator;b.shared=scratch+p.shared;b.result=scratch+p.result;b.shared_factor=scratch+p.shared_factor;
            b.selected=reinterpret_cast<unsigned*>(bytes(p.indices_offset));b.status=reinterpret_cast<unsigned*>(bytes(p.status_offset));
            check(cudaMemsetAsync(bytes(p.scratch_offset),0,p.device_bytes-p.scratch_offset,cudaStreamLegacy));check(cudaStreamSynchronize(cudaStreamLegacy));resources->loaded(handle);loaded=true;
        }catch(...){cleanup();throw;}
    }
    std::uint8_t* bytes(std::size_t offset){return static_cast<std::uint8_t*>(storage)+offset;}
    void current(){require(std::this_thread::get_id()==owner,"moe_thread_changed");int actual=-1;check(cudaGetDevice(&actual));require(actual==device,"moe_device_changed");}
    void available(){require(handle&&!poisoned,"moe_unavailable");try{current();}catch(const std::runtime_error&){poisoned=true;throw;}}
    void status(){check(cudaStreamSynchronize(cudaStreamLegacy));unsigned flags=0;check(cudaMemcpy(&flags,b.status,4,cudaMemcpyDeviceToHost));if(flags)throw std::overflow_error("moe_numeric_failure");}
    void pin(){resources->pin(handle);pinned=true;}
    void unpin(){resources->unpin(handle);pinned=false;}
    bool quiet()noexcept{if(!pinned)return true;try{check(cudaStreamSynchronize(cudaStreamLegacy));unpin();return true;}catch(...){poisoned=true;return false;}}
    void projection(const Projection& m,const float* x,float* y){check(kadan_launch_nvfp4(m.weights,m.scales,m.global,x,y,b.status,m.rows,m.columns));}
    void expert(std::size_t e,std::size_t middle,const float* x){projection(experts[e][0],x,b.gate);projection(experts[e][1],x,b.up);check(detail::moe_activate(middle,b));status();projection(experts[e][2],b.activation,b.down);}
    void forward(std::span<const float> x,std::span<float> y){
        available();require(!invalid,"moe_reset_required");ready=false;
        try{
            require(x.size()==c.hidden&&y.size()==c.hidden,"moe_input_shape");math::detail::output_alias(x,y,true);
            require(reinterpret_cast<std::uintptr_t>(x.data())%alignof(float)==0&&reinterpret_cast<std::uintptr_t>(y.data())%alignof(float)==0,"moe_alignment");
            pin();check(cudaMemsetAsync(b.status,0,4,cudaStreamLegacy));check(detail::moe_route(c,b,x.data()));status();
            // Bounded control metadata only; activations remain on device.
            std::array<unsigned,8> selected{};check(cudaMemcpy(selected.data(),b.selected,c.top_k*sizeof(unsigned),cudaMemcpyDeviceToHost));
            std::sort(selected.begin(),selected.begin()+c.top_k);
            for(std::size_t rank=0;rank<c.top_k;++rank){require(selected[rank]<c.experts&&(rank==0||selected[rank]!=selected[rank-1]),"moe_invalid_device_route");
                expert(selected[rank],c.intermediate,x.data());check(detail::moe_accumulate(c,b,selected[rank]));status();}
            expert(c.experts,c.shared_intermediate,x.data());check(detail::moe_finish(c,b,y.data()));status();unpin();ready=true;
        }catch(const std::invalid_argument&){invalid=true;if(!quiet())throw DeviceBufferQuarantine("moe_borrowed_buffers_quarantined");throw;}
        catch(const std::overflow_error&){invalid=true;if(!quiet())throw DeviceBufferQuarantine("moe_borrowed_buffers_quarantined");throw;}
        catch(...){poisoned=true;invalid=true;if(!quiet())throw DeviceBufferQuarantine("moe_borrowed_buffers_quarantined");throw;}
    }
    void reset(){available();pin();ready=false;try{check(cudaMemsetAsync(bytes(p.scratch_offset),0,p.device_bytes-p.scratch_offset,cudaStreamLegacy));check(cudaStreamSynchronize(cudaStreamLegacy));unpin();invalid=false;}catch(...){poisoned=true;throw;}}
    bool cleanup()noexcept{
        if(!handle)return true;
        if(cleanup_failed)return false;
        try{if(storage){current();check(cudaStreamSynchronize(cudaStreamLegacy));}if(pinned)unpin();if(loaded){resources->begin_eviction(handle);loaded=false;}
            if(storage){check(cudaFree(storage));storage=nullptr;}resources->released(handle);handle=0;ready=false;invalid=true;return true;
        }catch(...){cleanup_failed=poisoned=true;return false;}
    }
};
Moe::Moe(moe::Config c,const moe::Weights& w,int d,std::shared_ptr<Resources> r):impl_(std::make_unique<Impl>(c,w,d,std::move(r))){}
Moe::~Moe(){impl_->cleanup();}
std::size_t Moe::host_metadata_bytes(){return sizeof(Impl);}
void Moe::forward_device(std::span<const float> x,std::span<float> y){impl_->forward(x,y);}
void Moe::reset(){impl_->reset();}
bool Moe::valid()const{return impl_->handle&&!impl_->invalid&&!impl_->poisoned;}
void Moe::read_routes(std::span<unsigned> selected,std::span<float> logits,std::span<float> probabilities,std::span<float> top_weights){
    auto& i=*impl_;i.available();require(valid()&&i.ready&&selected.size()==i.c.top_k&&logits.size()==i.c.experts&&probabilities.size()==i.c.experts&&top_weights.size()==i.c.top_k,"moe_route_read");
    i.pin();try{check(cudaMemcpy(selected.data(),i.b.selected,selected.size_bytes(),cudaMemcpyDeviceToHost));check(cudaMemcpy(logits.data(),i.b.logits,logits.size_bytes(),cudaMemcpyDeviceToHost));check(cudaMemcpy(probabilities.data(),i.b.probabilities,probabilities.size_bytes(),cudaMemcpyDeviceToHost));check(cudaMemcpy(top_weights.data(),i.b.top_weights,top_weights.size_bytes(),cudaMemcpyDeviceToHost));i.unpin();}catch(...){i.poisoned=true;throw;}
}
void Moe::read_outputs(std::span<float> routed,std::span<float> shared,std::span<float> result){
    auto& i=*impl_;i.available();require(valid()&&i.ready&&routed.size()==i.c.hidden&&shared.size()==i.c.hidden&&result.size()==i.c.hidden,"moe_output_read");
    i.pin();try{check(cudaMemcpy(routed.data(),i.b.accumulator,routed.size_bytes(),cudaMemcpyDeviceToHost));check(cudaMemcpy(shared.data(),i.b.shared,shared.size_bytes(),cudaMemcpyDeviceToHost));check(cudaMemcpy(result.data(),i.b.result,result.size_bytes(),cudaMemcpyDeviceToHost));i.unpin();}catch(...){i.poisoned=true;throw;}
}
void Moe::close(){if(!impl_->cleanup())throw std::runtime_error("moe_cleanup_failed_reservation_retained");}
} // namespace kadan::cuda
