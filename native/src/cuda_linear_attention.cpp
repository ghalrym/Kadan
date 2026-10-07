#ifdef CUDA_API_PER_THREAD_DEFAULT_STREAM
#error "Linear sublayer requires legacy default stream"
#endif
#include "kadan/cuda_linear_attention.hpp"
#include "kadan/cuda_projection.hpp"
#include "kadan/cuda_state.hpp"
#include "linear_kernel.cuh"
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
struct LinearAttention::Impl {
    linear::Config c;linear::Plan p;std::shared_ptr<Resources> resources;int device;std::thread::id owner=std::this_thread::get_id();
    Handle handle=0;void* storage=nullptr;bool loaded=false,pinned=false,invalid=false,poisoned=false,cleanup_failed=false;
    std::unique_ptr<Fp8Projection> qkv,z,out;std::unique_ptr<SequenceState> state;detail::LinearBuffers b{};
    Impl(linear::Config config,const linear::Weights& w,int ordinal,std::shared_ptr<Resources> manager):c(config),p(linear::plan(c)),resources(std::move(manager)),device(ordinal){
        linear::validate_weights(c,w);require(resources&&device>=0,"linear_owner");auto request=resources->snapshot().capacity;
        require(std::size_t(device)+1<request.size(),"linear_unbudgeted_device");
        const auto aux_bytes=(p.aux_elements*2+255)&~std::size_t{255};
        const auto status_offset=(aux_bytes+p.scratch_floats*4+255)&~std::size_t{255};
        std::fill(request.begin(),request.end(),0);request[device+1]=status_offset+256;handle=resources->reserve(Workload::llm,request);
        try {
            current();qkv=std::make_unique<Fp8Projection>(w.qkv,device,resources);z=std::make_unique<Fp8Projection>(w.z,device,resources);out=std::make_unique<Fp8Projection>(w.out,device,resources);
            checkpoint::TextArchitecture a{};a.layers=1;a.max_context=c.capacity;a.attention_heads=a.kv_heads=1;a.head_dim=c.key_dim;
            a.key_heads=c.key_heads;a.value_heads=c.value_heads;a.key_dim=c.key_dim;a.value_dim=c.value_dim;a.conv_kernel=c.conv_kernel;a.layer_types[0]=checkpoint::LayerKind::linear_attention;
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
            b.input_norm=upload(w.input_norm);b.conv_weight=upload(w.conv);b.a_weight=upload(w.a);b.b_weight=upload(w.b);b.a_log=upload(w.a_log);b.dt_bias=upload(w.dt_bias);b.output_norm=upload(w.output_norm);
            require(std::size_t(packed-static_cast<std::uint16_t*>(storage))==p.aux_elements,"linear_aux_layout");
            auto* next=reinterpret_cast<float*>(static_cast<std::uint8_t*>(storage)+aux_bytes);
            b.normalized=next+p.norm_offset;b.qkv=next+p.qkv_offset;b.z=next+p.z_offset;b.a=next+p.a_offset;b.b=next+p.b_offset;
            b.core=next+p.core_offset;b.gated=next+p.gate_offset;b.projected=next+p.out_offset;b.status=reinterpret_cast<unsigned*>(static_cast<std::uint8_t*>(storage)+status_offset);
            check(cudaStreamSynchronize(cudaStreamLegacy));resources->loaded(handle);loaded=true;
        }catch(...){cleanup();throw;}
    }
    void current(){require(std::this_thread::get_id()==owner,"linear_thread_changed");int actual=-1;check(cudaGetDevice(&actual));require(actual==device,"linear_device_changed");}
    void available(){require(handle&&!poisoned,"linear_unavailable");try{current();}catch(const std::runtime_error&){poisoned=true;throw;}}
    void status(){check(cudaStreamSynchronize(cudaStreamLegacy));unsigned flags=0;check(cudaMemcpy(&flags,b.status,sizeof(flags),cudaMemcpyDeviceToHost));if(flags)throw std::overflow_error("linear_numeric_failure");}
    void invalidate(StateStep token)noexcept{
        invalid=true;try{state->abort(token);if(pinned){resources->unpin(handle);pinned=false;}}catch(...){poisoned=true;}
    }
    void step(std::span<const float> input,std::span<float> output){
        available();require(!invalid,"linear_reset_required");
        const auto token=state->begin();
        try {
            require(input.size()==c.hidden&&output.size()==c.hidden,"linear_input_shape");math::detail::output_alias(input,output,true);
            require(reinterpret_cast<std::uintptr_t>(input.data())%alignof(float)==0 && reinterpret_cast<std::uintptr_t>(output.data())%alignof(float)==0,"linear_alignment");
            resources->pin(handle);pinned=true;const auto view=state->layer(token,0);b.convolution=static_cast<std::uint16_t*>(view.convolution);b.recurrent=static_cast<float*>(view.recurrent);
            check(cudaMemsetAsync(b.status,0,sizeof(unsigned),cudaStreamLegacy));check(detail::linear_normalize(c,b,input.data()));status();
            qkv->matvec_device({b.normalized,c.hidden},{b.qkv,p.channels});z->matvec_device({b.normalized,c.hidden},{b.z,p.values});
            check(detail::linear_core(c,b));status();out->matvec_device({b.gated,p.values},{b.projected,c.hidden});
            check(detail::linear_residual(c,b,input.data(),output.data()));status();state->written(token,0);state->commit(token);
            resources->unpin(handle);pinned=false;
        }catch(const std::invalid_argument&){invalidate(token);throw;}catch(const std::overflow_error&){invalidate(token);throw;}
        catch(...){invalidate(token);poisoned=true;throw;}
    }
    bool cleanup()noexcept{
        if(!handle)return true;
        if(cleanup_failed)return false;
        try {
            if(storage || state || qkv || z || out)current();
            if(storage)check(cudaStreamSynchronize(cudaStreamLegacy));
            if(state)state->close();
            if(qkv)qkv->close();
            if(z)z->close();
            if(out)out->close();
            if(pinned){resources->unpin(handle);pinned=false;}if(loaded){resources->begin_eviction(handle);loaded=false;}
            if(storage){check(cudaFree(storage));storage=nullptr;}resources->released(handle);handle=0;invalid=true;return true;
        }catch(...){cleanup_failed=true;poisoned=true;return false;}
    }
};
LinearAttention::LinearAttention(linear::Config c,const linear::Weights& w,int d,std::shared_ptr<Resources> r):impl_(std::make_unique<Impl>(c,w,d,std::move(r))){}
LinearAttention::~LinearAttention(){impl_->cleanup();}
void LinearAttention::step_device(std::span<const float> x,std::span<float> y){impl_->step(x,y);}
void LinearAttention::reset(){auto& i=*impl_;i.available();i.state->reset();i.invalid=false;}
bool LinearAttention::valid()const{return impl_->handle&&!impl_->invalid&&!impl_->poisoned;}
std::size_t LinearAttention::tokens()const{return impl_->state->committed_tokens();}
void LinearAttention::read_state(std::span<std::uint16_t> conv,std::span<float> recurrent){
    auto& i=*impl_;i.available();require(!i.invalid && conv.size()==i.p.conv_elements && recurrent.size()==i.p.recurrent_elements,"linear_state_read");
    const auto& l=i.state->plan().layer[0];
    i.state->read_bytes(l.conv_offset,{reinterpret_cast<std::uint8_t*>(conv.data()),conv.size_bytes()});
    i.state->read_bytes(l.recurrent_offset,{reinterpret_cast<std::uint8_t*>(recurrent.data()),recurrent.size_bytes()});
}
void LinearAttention::close(){if(!impl_->cleanup())throw std::runtime_error("linear_cleanup_failed_reservation_retained");}
} // namespace kadan::cuda
