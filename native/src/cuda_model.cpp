#ifdef CUDA_API_PER_THREAD_DEFAULT_STREAM
#error "Model requires legacy default stream"
#endif
#include "kadan/cuda_model.hpp"
#include "kadan/weight_backing.hpp"
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
    void* request_storage=nullptr;
    Handle weight_handle=0,state_handle=0;
    bool weight_loaded=false,state_loaded=false;
    std::size_t weight_bytes=0,state_bytes=0,global_weights=0,global_state=0;
    std::array<std::size_t,256> weight_offsets{},state_offsets{};
    std::shared_ptr<serving::WeightBacking> backing;
    bool loaded=false,pinned=false,poisoned=false,cleanup_failed=false,ready=false,ended=false;
    Impl(ModelOptions o,int d,std::shared_ptr<Resources> r,Handle h):options(o),device(d),resources(std::move(r)),handle(h),metadata(std::make_shared<checkpoint::MemoryBudget>(o.metadata_bytes)),staging(std::make_shared<checkpoint::MemoryBudget>(o.staging_bytes)),producers(metadata.get()){}
    std::uint8_t* bytes(std::size_t offset){
        if(!options.split_residency)return static_cast<std::uint8_t*>(storage)+offset;
        if(offset>=layout->embedded())return static_cast<std::uint8_t*>(request_storage)+global_state+offset-layout->embedded();
        if(offset>=layout->embedding())return static_cast<std::uint8_t*>(storage)+global_weights+offset-layout->embedding();
        const auto ls=layout->layers();
        auto it=std::upper_bound(ls.begin(),ls.end(),offset,[](auto at,const auto& l){return at<l.offset;});
        require(it!=ls.begin(),"model_split_offset");--it;auto index=std::size_t(it-ls.begin());
        auto local=offset-it->offset;const auto split=it->plan.moe_offset+it->plan.moe.scratch_offset;
        return local<split?static_cast<std::uint8_t*>(storage)+weight_offsets[index]+local:
            static_cast<std::uint8_t*>(request_storage)+state_offsets[index]+local-split;
    }
    template<class T>T* pointer(std::size_t offset){return reinterpret_cast<T*>(bytes(offset));}
    void current(){require(thread==std::this_thread::get_id(),"model_thread_changed");int actual=-1;check(cudaGetDevice(&actual));require(actual==device,"model_device_changed");}
    void available(){require(handle&&!poisoned&&loaded&&(!options.split_residency||(storage&&request_storage)),"model_unavailable");try{current();}catch(const std::runtime_error&){poisoned=true;cursor->invalidate();throw;}}
    void write(std::size_t offset,std::span<const std::uint8_t> data)override{
        require(offset<=layout->device_bytes()&&data.size()<=layout->device_bytes()-offset,"model_upload_bounds");check(cudaMemcpy(bytes(offset),data.data(),data.size(),cudaMemcpyHostToDevice));
    }
    void multiplier(const model::Binding& b,float value)override{if(b.layer<0)head_global=value;else layers[std::size_t(b.layer)]->experts[std::size_t(b.expert)][std::size_t(b.part)].global=value;}
    void initialize(const char* root,const std::atomic_bool* cancelled){
        cancel(cancelled);manifest.emplace(root,metadata);auto g=model::read_generation(root,manifest->architecture().vocab,metadata);layout.emplace(*manifest,options.capacity,g,metadata);require(options.staging_bytes>=layout->minimum_staging_bytes(),"model_staging_row");
        auto request=resources->snapshot().capacity;std::fill(request.begin(),request.end(),0);request[0]=Model::host_bytes(options);request[device+1]=options.split_residency?options.device_headroom:add(layout->device_bytes(),options.device_headroom);resources->resize_loading(handle,request);
        cursor.emplace(layout->layers().size(),options.capacity);
        for(std::size_t i=0;i<layout->layers().size();++i){const auto& l=layout->layers()[i];producers.emplace_back(l.config,l.plan,*cursor);layers[i]=&producers.back();}
        if(options.split_residency){
            for(std::size_t i=0;i<layout->layers().size();++i){const auto& p=layout->layers()[i].plan;auto split=p.moe_offset+p.moe.scratch_offset;
                weight_offsets[i]=weight_bytes;state_offsets[i]=state_bytes;weight_bytes=add(weight_bytes,split);state_bytes=add(state_bytes,p.device_bytes-split);}
            global_weights=weight_bytes;global_state=state_bytes;weight_bytes=add(weight_bytes,layout->embedded()-layout->embedding());state_bytes=add(state_bytes,layout->device_bytes()-layout->embedded());
            require(add(weight_bytes,state_bytes)==layout->device_bytes(),"model_split_plan");
            backing=std::make_shared<serving::WeightBacking>(resources,options.weight_ram_bytes,options.weight_cold_bytes,options.staging_bytes,serving::WeightBacking::model_entry_limit,true);
            manifest->backing(backing);begin_request(cancelled);resources->loaded(handle);loaded=true;return;
        }
        cancel(cancelled);current();int major=0,minor=0;check(cudaDeviceGetAttribute(&major,cudaDevAttrComputeCapabilityMajor,device));check(cudaDeviceGetAttribute(&minor,cudaDevAttrComputeCapabilityMinor,device));require(major==8&&minor==6,"requires_sm86");
        std::size_t free=0,total=0;check(cudaMemGetInfo(&free,&total));require(free>=request[device+1],"model_physical_headroom");
        void* allocation=nullptr;check(cudaMalloc(&allocation,layout->device_bytes()));storage=allocation;
        for(std::size_t i=0;i<layout->layers().size();++i){layers[i]->storage=bytes(layout->layers()[i].offset);layers[i]->bind();}
        model::load(*layout,staging,*this,cancelled);zero();check(cudaStreamSynchronize(cudaStreamLegacy));cancel(cancelled);resources->loaded(handle);loaded=true;
    }
    Footprint device_request(std::size_t bytes){auto r=resources->snapshot().capacity;std::fill(r.begin(),r.end(),0);r[device+1]=bytes;return r;}
    void bind_split(bool preserve){
        for(std::size_t i=0;i<layout->layers().size();++i){auto* p=layers[i];p->storage=static_cast<std::uint8_t*>(storage)+weight_offsets[i];
            p->request_storage=static_cast<std::uint8_t*>(request_storage)+state_offsets[i];p->split=p->p.moe_offset+p->p.moe.scratch_offset;p->bind(preserve);}
    }
    void begin_request(const std::atomic_bool* cancelled){
        require(options.split_residency&&!poisoned&&!cleanup_failed&&!request_storage,"model_request_state");current();cancel(cancelled);
        int major=0,minor=0;check(cudaDeviceGetAttribute(&major,cudaDevAttrComputeCapabilityMajor,device));check(cudaDeviceGetAttribute(&minor,cudaDevAttrComputeCapabilityMinor,device));require(major==8&&minor==6,"requires_sm86");
        const bool reload=!storage;
        require(backing->room_for_reservations(reload?2:1),"model_resident_slots");
        if(reload)weight_handle=resources->reserve(Workload::llm,device_request(weight_bytes));
        try{
            state_handle=resources->reserve(Workload::llm,device_request(state_bytes));
            std::size_t free=0,total=0;check(cudaMemGetInfo(&free,&total));require(free>=add(state_bytes,reload?add(weight_bytes,options.device_headroom):options.device_headroom),"model_physical_headroom");
            if(reload){void* allocation=nullptr;check(cudaMalloc(&allocation,weight_bytes));storage=allocation;}
            void* allocation=nullptr;check(cudaMalloc(&allocation,state_bytes));request_storage=allocation;bind_split(!reload);
            if(reload)model::load(*layout,staging,*this,cancelled);
            zero();check(cudaStreamSynchronize(cudaStreamLegacy));cancel(cancelled);cursor->reset();ended=false;ready=false;
            if(reload){resources->loaded(weight_handle);weight_loaded=true;}resources->loaded(state_handle);state_loaded=true;
        }catch(...){poisoned=true;throw;}
    }
    void end_request(){
        require(options.split_residency&&!pinned,"model_request_active");current();
        if(request_storage){check(cudaStreamSynchronize(cudaStreamLegacy));check(cudaFree(request_storage));request_storage=nullptr;}
        if(state_handle){if(state_loaded)resources->begin_eviction(state_handle);resources->released(state_handle);state_handle=0;state_loaded=false;}
        if(cursor)cursor->invalidate();
        ready=false;
    }
    void park(){
        require(options.split_residency,"model_not_split");end_request();
        if(storage){check(cudaStreamSynchronize(cudaStreamLegacy));check(cudaFree(storage));storage=nullptr;}
        if(weight_handle){if(weight_loaded)resources->begin_eviction(weight_handle);resources->released(weight_handle);weight_handle=0;weight_loaded=false;}
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
        try{if(options.split_residency){if(pinned)unpin();park();}if(storage){current();check(cudaStreamSynchronize(cudaStreamLegacy));}if(pinned)unpin();if(loaded){resources->begin_eviction(handle);loaded=false;}if(storage){check(cudaFree(storage));storage=nullptr;}
            // Return host allocations before releasing their reservation.
            producers.clear();layout.reset();manifest.reset();backing.reset();
            if(options.split_residency){require(resources->snapshot().used[device+1]==options.device_headroom,"context_has_other_device_owner");current();check(cudaDeviceReset());}
            resources->released(handle);handle=0;if(cursor)cursor->close();ready=false;return true;
        }catch(...){cleanup_failed=poisoned=true;if(cursor)cursor->invalidate();return false;}
    }
};
std::size_t Model::host_bytes(ModelOptions o){require(o.metadata_bytes>0&&o.staging_bytes>=8&&o.staging_bytes<=32*1024*1024,"model_host_envelopes");return add(add(add(o.metadata_bytes,o.staging_bytes),control_headroom),o.split_residency?serving::WeightBacking::model_control_bytes:0);}
Model::Model(const char* root,ModelOptions o,int device,std::shared_ptr<Resources> r,const std::atomic_bool* cancelled){
    require(r&&device>=0,"model_owner");auto request=r->snapshot().capacity;require(std::size_t(device)+1<request.size(),"model_unbudgeted_device");std::fill(request.begin(),request.end(),0);request[0]=host_bytes(o);if(o.split_residency){require(r->snapshot().used[device+1]==0,"split_requires_exclusive_context");request[device+1]=o.device_headroom;}auto h=r->reserve(Workload::llm,request);
    try{impl_=std::make_unique<Impl>(o,device,r,h);}catch(...){r->released(h);throw;}
    try{impl_->initialize(root,cancelled);}catch(...){if(!impl_->cleanup())throw DeviceBufferQuarantine("model_load_cleanup_failed_reservation_retained");throw;}
}
Model::~Model(){if(impl_)impl_->cleanup();}
stack::Selection Model::step(unsigned x,bool stop,const std::atomic_bool*c){require(bool(impl_),"model_closed");return impl_->step(x,stop,c);}
bool Model::valid()const{return impl_&&impl_->handle&&!impl_->poisoned&&impl_->loaded&&(!impl_->options.split_residency||(impl_->storage&&impl_->request_storage))&&impl_->cursor->valid();}
bool Model::finished()const{return impl_&&impl_->ended;}std::size_t Model::tokens()const{return impl_&&impl_->cursor?impl_->cursor->committed_tokens():0;}
std::size_t Model::vocabulary()const{require(valid(),"model_unavailable");return impl_->manifest->architecture().vocab;}
std::size_t Model::device_bytes()const{require(valid(),"model_unavailable");return impl_->layout->device_bytes();}
void Model::begin_request(const std::atomic_bool* c){require(bool(impl_),"model_closed");impl_->begin_request(c);}
void Model::end_request(){require(bool(impl_),"model_closed");try{impl_->end_request();}catch(...){impl_->poisoned=impl_->cleanup_failed=true;throw;}}
void Model::park(){require(bool(impl_),"model_closed");try{impl_->park();}catch(...){impl_->poisoned=impl_->cleanup_failed=true;throw;}}
serving::WeightBacking::Stats Model::cache_stats()const{return impl_&&impl_->backing?impl_->backing->stats():serving::WeightBacking::Stats{};}
std::size_t Model::retained_bytes()const{return impl_&&impl_->backing?impl_->backing->stats().ram:0;}
void Model::reset(){require(bool(impl_),"model_closed");impl_->reset();}
void Model::close(){if(!impl_)return;if(!impl_->cleanup())throw std::runtime_error("model_cleanup_failed_reservation_retained");impl_.reset();}
float Model::projection_multiplier(std::size_t item)const{
    require(bool(impl_),"model_closed");auto&i=*impl_;i.available();
    require(item<i.layout->bindings().size()&&i.manifest->items()[item].kind==checkpoint::ItemKind::nvfp4,"model_not_nvfp4");
    const auto& b=i.layout->bindings()[item];return b.layer<0?i.head_global:i.layers[std::size_t(b.layer)]->experts[std::size_t(b.expert)][std::size_t(b.part)].global;
}
void Model::read_logits(std::span<float> out){require(bool(impl_),"model_closed");auto&i=*impl_;i.available();require(valid()&&i.ready&&out.size()==vocabulary(),"model_logits_unavailable");i.pin();try{check(cudaMemcpy(out.data(),i.bytes(i.layout->logits()),out.size_bytes(),cudaMemcpyDeviceToHost));i.unpin();}catch(...){i.poisoned=true;i.cursor->invalidate();throw;}}
void Model::read_state(std::size_t index,std::span<std::uint8_t>a,std::span<std::uint8_t>b){require(bool(impl_),"model_closed");auto&i=*impl_;i.available();require(valid()&&index<i.layout->layers().size(),"model_state_unavailable");auto& p=*i.layers[index];require(a.size()==p.p.state_first_bytes&&b.size()==p.p.state_second_bytes,"model_state_shape");i.pin();try{check(cudaMemcpy(a.data(),p.bytes(p.p.state_first),a.size(),cudaMemcpyDeviceToHost));check(cudaMemcpy(b.data(),p.bytes(p.p.state_second),b.size(),cudaMemcpyDeviceToHost));i.unpin();}catch(...){i.poisoned=true;i.cursor->invalidate();throw;}}
} // namespace kadan::cuda
