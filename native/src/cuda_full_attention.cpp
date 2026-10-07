#ifdef CUDA_API_PER_THREAD_DEFAULT_STREAM
#error "Full attention requires legacy default stream"
#endif
#include "kadan/cuda_full_attention.hpp"
#include "kadan/cuda_projection.hpp"
#include "kadan/cuda_state.hpp"
#include "full_kernel.cuh"
#include "layer_math_validation.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <thread>
#include <stdexcept>
namespace kadan::cuda {
namespace {
void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
void require(bool ok,const char* e){if(!ok)throw std::invalid_argument(e);}
}
struct FullAttention::Impl {
    full::Config c;full::Plan p;std::shared_ptr<Resources> resources;int device;std::thread::id owner=std::this_thread::get_id();
    Handle handle=0;void* storage=nullptr;bool loaded=false,pinned=false,invalid=false,poisoned=false,cleanup_failed=false;
    std::unique_ptr<Fp8Projection> qg,key,value,out;std::unique_ptr<SequenceState> state;detail::FullBuffers b{};
    Impl(full::Config config,const full::Weights& w,int ordinal,std::shared_ptr<Resources> manager):c(config),p(full::plan(c)),resources(std::move(manager)),device(ordinal){
        full::validate_weights(c,w);require(resources&&device>=0,"full_owner");auto request=resources->snapshot().capacity;
        require(std::size_t(device)+1<request.size(),"full_unbudgeted_device");
        const auto aux_bytes=(p.norm_weights*2+255)&~std::size_t{255};
        const auto frequency_bytes=(w.frequencies.size_bytes()+255)&~std::size_t{255};
        const auto scratch_offset=aux_bytes+frequency_bytes;
        const auto status_offset=(scratch_offset+p.scratch_floats*4+255)&~std::size_t{255};
        std::fill(request.begin(),request.end(),0);request[device+1]=status_offset+256;handle=resources->reserve(Workload::llm,request);
        try {
            current();qg=std::make_unique<Fp8Projection>(w.q_gate,device,resources);key=std::make_unique<Fp8Projection>(w.key,device,resources);value=std::make_unique<Fp8Projection>(w.value,device,resources);out=std::make_unique<Fp8Projection>(w.out,device,resources);
            checkpoint::TextArchitecture a{};a.layers=1;a.max_context=c.capacity;a.attention_heads=c.heads;a.kv_heads=c.kv_heads;a.head_dim=c.head_dim;
            a.key_heads=a.value_heads=a.key_dim=a.value_dim=a.conv_kernel=1;a.layer_types[0]=checkpoint::LayerKind::full_attention;
            state=std::make_unique<SequenceState>(a,c.capacity,device,resources);
            void* allocation=nullptr;check(cudaMalloc(&allocation,request[device+1]));storage=allocation;
            auto* packed=static_cast<std::uint16_t*>(storage);
            auto upload=[&](std::span<const float> values){
                auto* at=packed;std::array<std::uint16_t,1024> staging{};
                for(std::size_t offset=0;offset<values.size();offset+=staging.size()){
                    const auto n=std::min(staging.size(),values.size()-offset);
                    for(std::size_t j=0;j<n;++j)staging[j]=std::uint16_t(std::bit_cast<std::uint32_t>(values[offset+j])>>16);
                    check(cudaMemcpy(at+offset,staging.data(),n*sizeof(std::uint16_t),cudaMemcpyHostToDevice));
                }
                packed+=values.size();return at;
            };
            b.input_norm=upload(w.input_norm);b.query_norm=upload(w.query_norm);b.key_norm=upload(w.key_norm);
            require(std::size_t(packed-static_cast<std::uint16_t*>(storage))==p.norm_weights,"full_aux_layout");
            b.frequencies=reinterpret_cast<float*>(static_cast<std::uint8_t*>(storage)+aux_bytes);
            check(cudaMemcpy(const_cast<float*>(b.frequencies),w.frequencies.data(),w.frequencies.size_bytes(),cudaMemcpyHostToDevice));
            auto* next=reinterpret_cast<float*>(static_cast<std::uint8_t*>(storage)+scratch_offset);
            b.normalized=next+p.norm_offset;b.qg=next+p.qg_offset;b.key=next+p.k_offset;b.value=next+p.v_offset;b.query=next+p.q_offset;b.gate=next+p.gate_offset;
            b.probabilities=next+p.prob_offset;b.core=next+p.core_offset;b.gated=next+p.gated_offset;b.projected=next+p.out_offset;
            b.status=reinterpret_cast<unsigned*>(static_cast<std::uint8_t*>(storage)+status_offset);
            check(cudaStreamSynchronize(cudaStreamLegacy));resources->loaded(handle);loaded=true;
        }catch(...){cleanup();throw;}
    }
    void current(){require(std::this_thread::get_id()==owner,"full_thread_changed");int actual=-1;check(cudaGetDevice(&actual));require(actual==device,"full_device_changed");}
    void available(){require(handle&&!poisoned,"full_unavailable");try{current();}catch(const std::runtime_error&){poisoned=true;throw;}}
    void status(){check(cudaStreamSynchronize(cudaStreamLegacy));unsigned flags=0;check(cudaMemcpy(&flags,b.status,sizeof(flags),cudaMemcpyDeviceToHost));if(flags)throw std::overflow_error("full_numeric_failure");}
    bool invalidate(StateStep token)noexcept{
        invalid=true;
        try{state->abort(token);if(pinned){resources->unpin(handle);pinned=false;}return true;}
        catch(...){poisoned=true;return false;}
    }
    void step(std::span<const float> input,std::span<float> output){
        available();require(!invalid,"full_reset_required");
        StateStep token;
        try { token=state->begin(); } catch(const std::runtime_error&) { poisoned=true;throw; }
        try {
            require(input.size()==c.hidden&&output.size()==c.hidden,"full_input_shape");math::detail::output_alias(input,output,true);
            require(reinterpret_cast<std::uintptr_t>(input.data())%alignof(float)==0 && reinterpret_cast<std::uintptr_t>(output.data())%alignof(float)==0,"full_alignment");
            resources->pin(handle);pinned=true;const auto view=state->layer(token,0);b.keys=static_cast<std::uint16_t*>(view.key_history);b.values=static_cast<std::uint16_t*>(view.value_history);
            check(cudaMemsetAsync(b.status,0,sizeof(unsigned),cudaStreamLegacy));check(detail::full_normalize(c,b,input.data()));status();
            qg->matvec_device({b.normalized,c.hidden},{b.qg,2*p.queries});key->matvec_device({b.normalized,c.hidden},{b.key,p.kv});value->matvec_device({b.normalized,c.hidden},{b.value,p.kv});
            check(detail::full_core(c,b,state->committed_tokens()));status();out->matvec_device({b.gated,p.queries},{b.projected,c.hidden});
            check(detail::full_residual(c,b,input.data(),output.data()));status();state->written(token,0);state->commit(token);
            resources->unpin(handle);pinned=false;
        }catch(const std::invalid_argument&){
            if(!invalidate(token))throw DeviceBufferQuarantine("full_borrowed_buffers_quarantined");
            throw;
        }catch(const std::overflow_error&){
            if(!invalidate(token))throw DeviceBufferQuarantine("full_borrowed_buffers_quarantined");
            throw;
        }catch(...){
            const bool quiet=invalidate(token);poisoned=true;
            if(!quiet)throw DeviceBufferQuarantine("full_borrowed_buffers_quarantined");
            throw;
        }
    }
    bool cleanup()noexcept{
        if(!handle)return true;
        if(cleanup_failed)return false;
        try {
            if(storage || state || qg || key || value || out)current();
            if(storage)check(cudaStreamSynchronize(cudaStreamLegacy));
            if(state)state->close();
            if(qg)qg->close();
            if(key)key->close();
            if(value)value->close();
            if(out)out->close();
            if(pinned){resources->unpin(handle);pinned=false;}if(loaded){resources->begin_eviction(handle);loaded=false;}
            if(storage){check(cudaFree(storage));storage=nullptr;}resources->released(handle);handle=0;invalid=true;return true;
        }catch(...){cleanup_failed=true;poisoned=true;return false;}
    }
};
FullAttention::FullAttention(full::Config c,const full::Weights& w,int d,std::shared_ptr<Resources> r):impl_(std::make_unique<Impl>(c,w,d,std::move(r))){}
FullAttention::~FullAttention(){impl_->cleanup();}
void FullAttention::step_device(std::span<const float> x,std::span<float> y){impl_->step(x,y);}
void FullAttention::reset(){
    auto& i=*impl_;i.available();
    try {i.state->reset();i.invalid=false;} catch(const std::runtime_error&) {i.poisoned=true;throw;}
}
bool FullAttention::valid()const{return impl_->handle&&!impl_->invalid&&!impl_->poisoned&&impl_->state&&impl_->state->valid();}
std::size_t FullAttention::tokens()const{return impl_->state->committed_tokens();}
void FullAttention::read_state(std::span<std::uint16_t> keys,std::span<std::uint16_t> values){
    auto& i=*impl_;i.available();require(valid() && keys.size()==i.p.history_elements && values.size()==i.p.history_elements,"full_state_read");
    const auto& l=i.state->plan().layer[0];
    try {
        i.state->read_bytes(l.key_offset,{reinterpret_cast<std::uint8_t*>(keys.data()),keys.size_bytes()});
        i.state->read_bytes(l.value_offset,{reinterpret_cast<std::uint8_t*>(values.data()),values.size_bytes()});
    } catch(const std::runtime_error&) {i.poisoned=true;throw;}
}
void FullAttention::read_intermediates(std::span<float> probabilities,std::span<float> core,std::span<float> gated){
    auto& i=*impl_;i.available();
    require(valid() && tokens()>0 && probabilities.size()==i.c.heads*i.c.capacity && core.size()==i.p.queries && gated.size()==i.p.queries,"full_intermediate_read");
    try {
        check(cudaMemcpy(probabilities.data(),i.b.probabilities,probabilities.size_bytes(),cudaMemcpyDeviceToHost));
        check(cudaMemcpy(core.data(),i.b.core,core.size_bytes(),cudaMemcpyDeviceToHost));
        check(cudaMemcpy(gated.data(),i.b.gated,gated.size_bytes(),cudaMemcpyDeviceToHost));
    } catch(const std::runtime_error&) {i.poisoned=true;throw;}
}
void FullAttention::close(){if(!impl_->cleanup())throw std::runtime_error("full_cleanup_failed_reservation_retained");}
} // namespace kadan::cuda
