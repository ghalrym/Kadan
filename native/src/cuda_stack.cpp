#ifdef CUDA_API_PER_THREAD_DEFAULT_STREAM
#error "Stack requires legacy default stream"
#endif
#include "kadan/cuda_stack.hpp"
#include "decoder_producer.hpp"
#include "stack_kernel.cuh"
#include <optional>
#include <thread>
namespace kadan::cuda {
namespace {
void require(bool x,const char* e){if(!x)throw std::invalid_argument(e);}
void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
}
struct Stack::Impl {
    stack::Config c;stack::Plan p;std::shared_ptr<Resources> resources;int device;StateCursor cursor;
    std::array<std::optional<detail::DecoderProducer>,stack::layers> layer;
    std::thread::id owner=std::this_thread::get_id();Handle handle;void* storage=nullptr;float head_global=0;
    bool loaded=false,pinned=false,poisoned=false,cleanup_failed=false,ready=false,ended=false;
    Impl(stack::Config config,stack::Plan layout,int d,std::shared_ptr<Resources> r,Handle h):c(config),p(layout),resources(std::move(r)),device(d),cursor(stack::layers,p.capacity),handle(h){
        for(std::size_t i=0;i<stack::layers;++i)layer[i].emplace(c.layer[i],p.layer[i],cursor);
    }
    std::uint8_t* bytes(std::size_t offset){return static_cast<std::uint8_t*>(storage)+offset;}
    template<class T>T* pointer(std::size_t offset){return reinterpret_cast<T*>(bytes(offset));}
    void current(){require(std::this_thread::get_id()==owner,"stack_thread_changed");int actual=-1;check(cudaGetDevice(&actual));require(actual==device,"stack_device_changed");}
    void available(){require(handle&&!poisoned,"stack_unavailable");try{current();}catch(const std::runtime_error&){poisoned=true;cursor.invalidate();throw;}}
    void upload(std::span<const float>x,std::size_t offset){std::array<std::uint16_t,1024> staging{};auto*dst=pointer<std::uint16_t>(offset);for(std::size_t at=0;at<x.size();at+=staging.size()){auto n=std::min(staging.size(),x.size()-at);for(std::size_t j=0;j<n;++j)staging[j]=std::uint16_t(std::bit_cast<std::uint32_t>(x[at+j])>>16);check(cudaMemcpy(dst+at,staging.data(),n*2,cudaMemcpyHostToDevice));}}
    void initialize(const stack::Weights&w){
        current();int major=0,minor=0;check(cudaDeviceGetAttribute(&major,cudaDevAttrComputeCapabilityMajor,device));check(cudaDeviceGetAttribute(&minor,cudaDevAttrComputeCapabilityMinor,device));require(major==8&&minor==6,"requires_sm86");
        void* allocation=nullptr;check(cudaMalloc(&allocation,p.device_bytes));storage=allocation;
        for(std::size_t i=0;i<stack::layers;++i){layer[i]->storage=bytes(p.offset[i]);layer[i]->initialize(w.layer[i]);}
        upload(w.embedding,p.embedding);upload(w.final_norm,p.final_norm);check(cudaMemcpy(bytes(p.head_weights),w.head.weights.data(),w.head.weights.size_bytes(),cudaMemcpyHostToDevice));check(cudaMemcpy(bytes(p.head_scales),w.head.block_scales.data(),w.head.block_scales.size_bytes(),cudaMemcpyHostToDevice));head_global=w.head.multipliers[0];
        check(cudaMemsetAsync(bytes(p.embedded),0,p.device_bytes-p.embedded,cudaStreamLegacy));check(cudaStreamSynchronize(cudaStreamLegacy));resources->loaded(handle);loaded=true;
    }
    void pin(){resources->pin(handle);pinned=true;}void unpin(){resources->unpin(handle);pinned=false;}
    void cancel(const std::atomic_bool*f){if(f&&f->load(std::memory_order_relaxed))throw std::invalid_argument("stack_cancelled");}
    void status(){check(cudaStreamSynchronize(cudaStreamLegacy));unsigned flags=0;check(cudaMemcpy(&flags,bytes(p.status),4,cudaMemcpyDeviceToHost));if(flags)throw std::overflow_error("stack_numeric_failure");}
    bool invalidate()noexcept{cursor.invalidate();ready=false;if(!pinned)return true;try{check(cudaStreamSynchronize(cudaStreamLegacy));unpin();return true;}catch(...){poisoned=true;return false;}}
    stack::Selection step(unsigned input,bool stop,const std::atomic_bool* cancelled){
        available();require(!ended,"stack_eos");auto token=cursor.begin();ready=false;
        try{
            require(input<c.vocabulary,"stack_input_id");cancel(cancelled);pin();check(cudaMemsetAsync(bytes(p.status),0,4,cudaStreamLegacy));
            check(detail::stack_embedding(p.hidden,pointer<std::uint16_t>(p.embedding),input,pointer<float>(p.embedded)));status();const float*x=pointer<float>(p.embedded);
            for(std::size_t i=0;i<stack::layers;++i){layer[i]->run(token,x,cancelled);x=layer[i]->work(3);cursor.written(token,i);cancel(cancelled);}
            check(detail::decoder_norm(p.hidden,c.epsilon,pointer<std::uint16_t>(p.final_norm),x,pointer<float>(p.normalized),pointer<unsigned>(p.status)));status();
            check(kadan_launch_nvfp4(bytes(p.head_weights),bytes(p.head_scales),head_global,pointer<float>(p.normalized),pointer<float>(p.logits),pointer<unsigned>(p.status),c.vocabulary,p.hidden));status();
            check(detail::stack_select(c.vocabulary,pointer<float>(p.logits),pointer<unsigned>(p.selected),pointer<unsigned>(p.status)));status();cancel(cancelled);
            unsigned selected=0;check(cudaMemcpy(&selected,bytes(p.selected),4,cudaMemcpyDeviceToHost));check(cudaStreamSynchronize(cudaStreamLegacy));require(selected<c.vocabulary,"stack_device_selection");cancel(cancelled);
            stack::Selection result{selected,selected==c.eos};cursor.ready_to_commit(token);unpin();
            // Only this model cursor publishes. No child progress or fallible
            // CUDA/resource work remains. The return value is now consumable.
            cursor.commit(token);ready=true;ended=stop&&result.eos;return result;
        }catch(const std::invalid_argument&){if(!invalidate())throw DeviceBufferQuarantine("stack_arena_quarantined");throw;}
        catch(const std::overflow_error&){if(!invalidate())throw DeviceBufferQuarantine("stack_arena_quarantined");throw;}
        catch(...){poisoned=true;if(!invalidate())throw DeviceBufferQuarantine("stack_arena_quarantined");throw;}
    }
    void reset(){available();require(!cursor.active(),"stack_active");pin();ready=false;try{for(auto& l:layer)l->zero();check(cudaMemsetAsync(bytes(p.embedded),0,p.device_bytes-p.embedded,cudaStreamLegacy));check(cudaStreamSynchronize(cudaStreamLegacy));unpin();cursor.reset();ended=false;}catch(...){poisoned=true;cursor.invalidate();throw;}}
    bool cleanup()noexcept{
        if(!handle)return true;
        if(cleanup_failed)return false;
        try{if(storage){current();check(cudaStreamSynchronize(cudaStreamLegacy));}if(pinned)unpin();if(loaded){resources->begin_eviction(handle);loaded=false;}if(storage){check(cudaFree(storage));storage=nullptr;}resources->released(handle);handle=0;cursor.close();ready=false;return true;}
        catch(...){cleanup_failed=poisoned=true;cursor.invalidate();return false;}
    }
};
Stack::Stack(stack::Config c,const stack::Weights&w,int device,std::shared_ptr<Resources>r){
    auto p=stack::plan(c);stack::validate_weights(c,w);require(r&&device>=0,"stack_owner");auto request=r->snapshot().capacity;require(std::size_t(device)+1<request.size(),"stack_unbudgeted_device");std::fill(request.begin(),request.end(),0);request[0]=sizeof(Impl);request[device+1]=p.device_bytes;auto h=r->reserve(Workload::llm,request);
    try{impl_=std::make_unique<Impl>(c,p,device,r,h);}catch(...){r->released(h);throw;}
    try{impl_->initialize(w);}catch(...){impl_->cleanup();throw;}
}
Stack::~Stack(){if(impl_)impl_->cleanup();}
std::size_t Stack::host_metadata_bytes(){return sizeof(Impl);}
stack::Selection Stack::step(unsigned x,bool stop,const std::atomic_bool*c){require(bool(impl_),"stack_closed");return impl_->step(x,stop,c);}
bool Stack::valid()const{return impl_&&impl_->handle&&!impl_->poisoned&&impl_->cursor.valid();}
bool Stack::finished()const{return impl_&&impl_->ended;}std::size_t Stack::tokens()const{return impl_?impl_->cursor.committed_tokens():0;}
void Stack::reset(){require(bool(impl_),"stack_closed");impl_->reset();}
void Stack::read_layer(std::size_t index,std::span<float>out){require(bool(impl_),"stack_closed");auto&i=*impl_;i.available();require(valid()&&i.ready&&index<stack::layers&&out.size()==i.p.hidden,"stack_diagnostic");i.pin();try{check(cudaMemcpy(out.data(),i.layer[index]->work(3),out.size_bytes(),cudaMemcpyDeviceToHost));i.unpin();}catch(...){i.poisoned=true;i.cursor.invalidate();throw;}}
void Stack::read_state(std::size_t index,std::span<std::uint8_t>a,std::span<std::uint8_t>b){require(bool(impl_),"stack_closed");auto&i=*impl_;i.available();require(valid()&&index<stack::layers,"stack_state");const auto& p=i.p.layer[index];require(a.size()==p.state_first_bytes&&b.size()==p.state_second_bytes,"stack_state_shape");i.pin();try{check(cudaMemcpy(a.data(),i.layer[index]->bytes(p.state_first),a.size_bytes(),cudaMemcpyDeviceToHost));check(cudaMemcpy(b.data(),i.layer[index]->bytes(p.state_second),b.size_bytes(),cudaMemcpyDeviceToHost));i.unpin();}catch(...){i.poisoned=true;i.cursor.invalidate();throw;}}
void Stack::read_output(std::span<float>n,std::span<float>logits){require(bool(impl_),"stack_closed");auto&i=*impl_;i.available();require(valid()&&i.ready&&n.size()==i.p.hidden&&logits.size()==i.c.vocabulary,"stack_diagnostic");i.pin();try{check(cudaMemcpy(n.data(),i.bytes(i.p.normalized),n.size_bytes(),cudaMemcpyDeviceToHost));check(cudaMemcpy(logits.data(),i.bytes(i.p.logits),logits.size_bytes(),cudaMemcpyDeviceToHost));i.unpin();}catch(...){i.poisoned=true;i.cursor.invalidate();throw;}}
void Stack::close(){if(!impl_)return;if(!impl_->cleanup())throw std::runtime_error("stack_cleanup_failed_reservation_retained");impl_.reset();}
}
