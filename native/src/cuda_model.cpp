#ifdef CUDA_API_PER_THREAD_DEFAULT_STREAM
#error "Model requires legacy default stream"
#endif
#include "kadan/cuda_model.hpp"
#include "decoder_producer.hpp"
#include "stack_kernel.cuh"
#include <list>
#include <optional>
#include <thread>
namespace kadan::cuda {
namespace {
void require(bool v,const char* e){if(!v)throw std::invalid_argument(e);}
void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
std::size_t add(std::size_t a,std::size_t b){require(b<=SIZE_MAX-a,"model_budget_overflow");return a+b;}
}
struct Model::Impl final:model::Sink {
    ModelOptions options;int device;std::shared_ptr<Resources> resources;Handle handle;
    std::shared_ptr<checkpoint::MemoryBudget> metadata,staging;
    std::optional<checkpoint::ModelManifest> manifest;std::optional<model::Layout> layout;std::optional<StateCursor> cursor;
    std::pmr::list<detail::DecoderProducer> producers;std::array<detail::DecoderProducer*,256> layers{};
    std::thread::id thread=std::this_thread::get_id();void* storage=nullptr;float head_global=0;
    bool loaded=false,pinned=false,poisoned=false,cleanup_failed=false,ready=false,ended=false;
    Impl(ModelOptions o,int d,std::shared_ptr<Resources> r,Handle h):options(o),device(d),resources(std::move(r)),handle(h),metadata(std::make_shared<checkpoint::MemoryBudget>(o.metadata_bytes)),staging(std::make_shared<checkpoint::MemoryBudget>(o.staging_bytes)),producers(metadata.get()){}
    std::uint8_t* bytes(std::size_t offset){return static_cast<std::uint8_t*>(storage)+offset;}
    template<class T>T* pointer(std::size_t offset){return reinterpret_cast<T*>(bytes(offset));}
    void current(){require(thread==std::this_thread::get_id(),"model_thread_changed");int actual=-1;check(cudaGetDevice(&actual));require(actual==device,"model_device_changed");}
    void available(){require(handle&&!poisoned&&loaded,"model_unavailable");try{current();}catch(const std::runtime_error&){poisoned=true;cursor->invalidate();throw;}}
    void write(std::size_t offset,std::span<const std::uint8_t> data)override{
        require(offset<=layout->device_bytes()&&data.size()<=layout->device_bytes()-offset,"model_upload_bounds");check(cudaMemcpy(bytes(offset),data.data(),data.size(),cudaMemcpyHostToDevice));
    }
    void multiplier(const model::Binding& b,float value)override{if(b.layer<0)head_global=value;else layers[std::size_t(b.layer)]->experts[std::size_t(b.expert)][std::size_t(b.part)].global=value;}
    void initialize(const char* root,const std::atomic_bool* cancelled){
        cancel(cancelled);manifest.emplace(root,metadata);auto g=model::read_generation(root,manifest->architecture().vocab,metadata);layout.emplace(*manifest,options.capacity,g,metadata);require(options.staging_bytes>=layout->minimum_staging_bytes(),"model_staging_row");
        auto request=resources->snapshot().capacity;std::fill(request.begin(),request.end(),0);request[0]=Model::host_bytes(options);request[device+1]=add(layout->device_bytes(),options.device_headroom);resources->resize_loading(handle,request);
        cursor.emplace(layout->layers().size(),options.capacity);
        for(std::size_t i=0;i<layout->layers().size();++i){const auto& l=layout->layers()[i];producers.emplace_back(l.config,l.plan,*cursor);layers[i]=&producers.back();}
        cancel(cancelled);current();int major=0,minor=0;check(cudaDeviceGetAttribute(&major,cudaDevAttrComputeCapabilityMajor,device));check(cudaDeviceGetAttribute(&minor,cudaDevAttrComputeCapabilityMinor,device));require(major==8&&minor==6,"requires_sm86");
        std::size_t free=0,total=0;check(cudaMemGetInfo(&free,&total));require(free>=request[device+1],"model_physical_headroom");
        void* allocation=nullptr;check(cudaMalloc(&allocation,layout->device_bytes()));storage=allocation;
        for(std::size_t i=0;i<layout->layers().size();++i){layers[i]->storage=bytes(layout->layers()[i].offset);layers[i]->bind();}
        model::load(*layout,staging,*this,cancelled);zero();check(cudaStreamSynchronize(cudaStreamLegacy));cancel(cancelled);resources->loaded(handle);loaded=true;
    }
    void zero(){for(auto& p:producers)p.zero();check(cudaMemsetAsync(bytes(layout->embedded()),0,layout->device_bytes()-layout->embedded(),cudaStreamLegacy));}
    void pin(){resources->pin(handle);pinned=true;}void unpin(){resources->unpin(handle);pinned=false;}
    void cancel(const std::atomic_bool*f){require(!f||!f->load(std::memory_order_relaxed),"model_cancelled");}
    void status(){check(cudaStreamSynchronize(cudaStreamLegacy));unsigned v=0;check(cudaMemcpy(&v,bytes(layout->status()),4,cudaMemcpyDeviceToHost));if(v)throw std::overflow_error("model_numeric_failure");}
    bool invalidate()noexcept{cursor->invalidate();ready=false;if(!pinned)return true;try{check(cudaStreamSynchronize(cudaStreamLegacy));unpin();return true;}catch(...){poisoned=true;return false;}}
    stack::Selection step(unsigned input,bool stop,const std::atomic_bool* cancelled){
        available();require(!ended,"model_eos");auto token=cursor->begin();ready=false;
        try{const auto&a=manifest->architecture();require(input<a.vocab,"model_input_id");cancel(cancelled);pin();check(cudaMemsetAsync(bytes(layout->status()),0,4,cudaStreamLegacy));
            check(detail::stack_embedding(a.hidden,pointer<std::uint16_t>(layout->embedding()),input,pointer<float>(layout->embedded())));status();const float*x=pointer<float>(layout->embedded());
            for(std::size_t i=0;i<layout->layers().size();++i){layers[i]->run(token,x,cancelled);x=layers[i]->work(3);cursor->written(token,i);cancel(cancelled);}
            check(detail::decoder_norm(a.hidden,float(a.rms_epsilon),pointer<std::uint16_t>(layout->final_norm()),x,pointer<float>(layout->normalized()),pointer<unsigned>(layout->status())));status();
            check(kadan_launch_nvfp4_bf16(bytes(layout->head_weights()),bytes(layout->head_scales()),head_global,pointer<float>(layout->normalized()),pointer<float>(layout->logits()),pointer<unsigned>(layout->status()),a.vocab,a.hidden));status();
            check(detail::stack_select(a.vocab,pointer<float>(layout->logits()),pointer<unsigned>(layout->selected()),pointer<unsigned>(layout->status())));status();cancel(cancelled);
            unsigned selected=0;check(cudaMemcpy(&selected,bytes(layout->selected()),4,cudaMemcpyDeviceToHost));check(cudaStreamSynchronize(cudaStreamLegacy));require(selected<a.vocab,"model_device_selection");cancel(cancelled);
            stack::Selection result{selected,layout->generation().is_eos(selected)};cursor->ready_to_commit(token);unpin();cursor->commit(token);ready=true;ended=stop&&result.eos;return result;
        }catch(const std::invalid_argument&){if(!invalidate())throw DeviceBufferQuarantine("model_arena_quarantined");throw;}
        catch(const std::overflow_error&){if(!invalidate())throw DeviceBufferQuarantine("model_arena_quarantined");throw;}
        catch(...){poisoned=true;if(!invalidate())throw DeviceBufferQuarantine("model_arena_quarantined");throw;}
    }
    void reset(){available();require(!cursor->active(),"model_active");pin();ready=false;try{zero();check(cudaStreamSynchronize(cudaStreamLegacy));unpin();cursor->reset();ended=false;}catch(...){poisoned=true;cursor->invalidate();throw;}}
    bool cleanup()noexcept{
        if(!handle)return true;
        if(cleanup_failed)return false;
        try{if(storage){current();check(cudaStreamSynchronize(cudaStreamLegacy));}if(pinned)unpin();if(loaded){resources->begin_eviction(handle);loaded=false;}if(storage){check(cudaFree(storage));storage=nullptr;}
            // Return host allocations before releasing their reservation.
            producers.clear();layout.reset();manifest.reset();resources->released(handle);handle=0;if(cursor)cursor->close();ready=false;return true;
        }catch(...){cleanup_failed=poisoned=true;if(cursor)cursor->invalidate();return false;}
    }
};
std::size_t Model::host_bytes(ModelOptions o){require(o.metadata_bytes>0&&o.staging_bytes>=8&&o.staging_bytes<=32*1024*1024,"model_host_envelopes");return add(add(o.metadata_bytes,o.staging_bytes),control_headroom);}
Model::Model(const char* root,ModelOptions o,int device,std::shared_ptr<Resources> r,const std::atomic_bool* cancelled){
    require(r&&device>=0,"model_owner");auto request=r->snapshot().capacity;require(std::size_t(device)+1<request.size(),"model_unbudgeted_device");std::fill(request.begin(),request.end(),0);request[0]=host_bytes(o);auto h=r->reserve(Workload::llm,request);
    try{impl_=std::make_unique<Impl>(o,device,r,h);}catch(...){r->released(h);throw;}
    try{impl_->initialize(root,cancelled);}catch(...){if(!impl_->cleanup())throw DeviceBufferQuarantine("model_load_cleanup_failed_reservation_retained");throw;}
}
Model::~Model(){if(impl_)impl_->cleanup();}
stack::Selection Model::step(unsigned x,bool stop,const std::atomic_bool*c){require(bool(impl_),"model_closed");return impl_->step(x,stop,c);}
bool Model::valid()const{return impl_&&impl_->handle&&!impl_->poisoned&&impl_->loaded&&impl_->cursor->valid();}
bool Model::finished()const{return impl_&&impl_->ended;}std::size_t Model::tokens()const{return impl_&&impl_->cursor?impl_->cursor->committed_tokens():0;}
std::size_t Model::vocabulary()const{require(valid(),"model_unavailable");return impl_->manifest->architecture().vocab;}
std::size_t Model::device_bytes()const{require(valid(),"model_unavailable");return impl_->layout->device_bytes();}
void Model::reset(){require(bool(impl_),"model_closed");impl_->reset();}
void Model::close(){if(!impl_)return;if(!impl_->cleanup())throw std::runtime_error("model_cleanup_failed_reservation_retained");impl_.reset();}
void Model::read_logits(std::span<float> out){require(bool(impl_),"model_closed");auto&i=*impl_;i.available();require(valid()&&i.ready&&out.size()==vocabulary(),"model_logits_unavailable");i.pin();try{check(cudaMemcpy(out.data(),i.bytes(i.layout->logits()),out.size_bytes(),cudaMemcpyDeviceToHost));i.unpin();}catch(...){i.poisoned=true;i.cursor->invalidate();throw;}}
void Model::read_state(std::size_t index,std::span<std::uint8_t>a,std::span<std::uint8_t>b){require(bool(impl_),"model_closed");auto&i=*impl_;i.available();require(valid()&&index<i.layout->layers().size(),"model_state_unavailable");auto& p=*i.layers[index];require(a.size()==p.p.state_first_bytes&&b.size()==p.p.state_second_bytes,"model_state_shape");i.pin();try{check(cudaMemcpy(a.data(),p.bytes(p.p.state_first),a.size(),cudaMemcpyDeviceToHost));check(cudaMemcpy(b.data(),p.bytes(p.p.state_second),b.size(),cudaMemcpyDeviceToHost));i.unpin();}catch(...){i.poisoned=true;i.cursor->invalidate();throw;}}
} // namespace kadan::cuda
