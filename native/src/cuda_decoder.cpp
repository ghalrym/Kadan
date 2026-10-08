#ifdef CUDA_API_PER_THREAD_DEFAULT_STREAM
#error "Decoder requires legacy default stream"
#endif
#include "kadan/cuda_decoder.hpp"
#include "decoder_producer.hpp"
#include "layer_math_validation.hpp"
#include <thread>
namespace kadan::cuda {
namespace {
void require(bool x,const char* e){if(!x)throw std::invalid_argument(e);}
void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
}
struct Decoder::Impl {
    std::shared_ptr<Resources> resources;int device;StateCursor cursor;detail::DecoderProducer producer;
    std::thread::id owner=std::this_thread::get_id();Handle handle;
    bool loaded=false,pinned=false,poisoned=false,cleanup_failed=false,ready=false;
    Impl(decoder::Config c,decoder::Plan p,int d,std::shared_ptr<Resources> r,Handle h):resources(std::move(r)),device(d),cursor(1,p.capacity),producer(c,p,cursor),handle(h){}
    void current(){require(std::this_thread::get_id()==owner,"decoder_thread_changed");int actual=-1;check(cudaGetDevice(&actual));require(actual==device,"decoder_device_changed");}
    void available(){require(handle&&!poisoned,"decoder_unavailable");try{current();}catch(const std::runtime_error&){poisoned=true;cursor.invalidate();throw;}}
    void initialize(const decoder::Weights& weights){
        current();int major=0,minor=0;check(cudaDeviceGetAttribute(&major,cudaDevAttrComputeCapabilityMajor,device));check(cudaDeviceGetAttribute(&minor,cudaDevAttrComputeCapabilityMinor,device));require(major==8&&minor==6,"requires_sm86");
        void* allocation=nullptr;check(cudaMalloc(&allocation,producer.p.device_bytes));producer.storage=allocation;producer.initialize(weights);
        check(cudaStreamSynchronize(cudaStreamLegacy));resources->loaded(handle);loaded=true;
    }
    void pin(){resources->pin(handle);pinned=true;}
    void unpin(){resources->unpin(handle);pinned=false;}
    bool invalidate()noexcept{cursor.invalidate();ready=false;if(!pinned)return true;try{check(cudaStreamSynchronize(cudaStreamLegacy));unpin();return true;}catch(...){poisoned=true;return false;}}
    void step(std::span<const float> x,std::span<float> y,const std::atomic_bool* cancelled){
        available();const auto token=cursor.begin();ready=false;const auto& p=producer.p;
        try{
            require(x.size()==p.hidden&&y.size()==p.hidden,"decoder_input_shape");math::detail::output_alias(x,y,true);
            require(reinterpret_cast<std::uintptr_t>(x.data())%alignof(float)==0&&reinterpret_cast<std::uintptr_t>(y.data())%alignof(float)==0,"decoder_alignment");producer.cancel(cancelled);
            pin();producer.run(token,x.data(),cancelled);cursor.written(token,0);cursor.ready_to_commit(token);
            check(cudaMemcpy(y.data(),producer.work(3),p.hidden*4,cudaMemcpyDeviceToDevice));check(cudaStreamSynchronize(cudaStreamLegacy));producer.cancel(cancelled);unpin();
            // Prevalidated same-thread host publication; no CUDA/resources follow.
            cursor.commit(token);ready=true;
        }catch(const std::invalid_argument&){if(!invalidate())throw DeviceBufferQuarantine("decoder_borrowed_buffers_quarantined");throw;}
        catch(const std::overflow_error&){if(!invalidate())throw DeviceBufferQuarantine("decoder_borrowed_buffers_quarantined");throw;}
        catch(...){poisoned=true;if(!invalidate())throw DeviceBufferQuarantine("decoder_borrowed_buffers_quarantined");throw;}
    }
    void reset(){available();require(!cursor.active(),"decoder_active");pin();ready=false;try{producer.zero();check(cudaStreamSynchronize(cudaStreamLegacy));unpin();cursor.reset();}catch(...){poisoned=true;cursor.invalidate();throw;}}
    bool cleanup()noexcept{
        if(!handle)return true;
        if(cleanup_failed)return false;
        try{if(producer.storage){current();check(cudaStreamSynchronize(cudaStreamLegacy));}if(pinned)unpin();if(loaded){resources->begin_eviction(handle);loaded=false;}if(producer.storage){check(cudaFree(producer.storage));producer.storage=nullptr;}resources->released(handle);handle=0;cursor.close();ready=false;return true;}
        catch(...){cleanup_failed=poisoned=true;cursor.invalidate();return false;}
    }
};
Decoder::Decoder(decoder::Config c,const decoder::Weights& w,int device,std::shared_ptr<Resources> r){
    const auto p=decoder::plan(c);decoder::validate_weights(c,w);require(r&&device>=0,"decoder_owner");auto request=r->snapshot().capacity;require(std::size_t(device)+1<request.size(),"decoder_unbudgeted_device");
    std::fill(request.begin(),request.end(),0);request[0]=sizeof(Impl);request[device+1]=p.device_bytes;auto h=r->reserve(Workload::llm,request);
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
    require(bool(impl_),"decoder_closed");auto& i=*impl_;auto& b=i.producer;i.available();require(valid()&&i.ready&&a.size()==b.p.hidden&&u.size()==b.p.hidden&&m.size()==b.p.hidden,"decoder_diagnostic");i.pin();
    try{check(cudaMemcpy(a.data(),b.work(0),a.size_bytes(),cudaMemcpyDeviceToHost));check(cudaMemcpy(u.data(),b.work(1),u.size_bytes(),cudaMemcpyDeviceToHost));check(cudaMemcpy(m.data(),b.work(2),m.size_bytes(),cudaMemcpyDeviceToHost));i.unpin();}catch(...){i.poisoned=true;i.cursor.invalidate();throw;}
}
void Decoder::read_state(std::span<std::uint8_t>a,std::span<std::uint8_t>b){
    require(bool(impl_),"decoder_closed");auto& i=*impl_;auto& producer=i.producer;i.available();require(valid()&&!i.cursor.active()&&a.size()==producer.p.state_first_bytes&&b.size()==producer.p.state_second_bytes,"decoder_state_read");i.pin();
    try{check(cudaMemcpy(a.data(),producer.bytes(producer.p.state_first),a.size_bytes(),cudaMemcpyDeviceToHost));check(cudaMemcpy(b.data(),producer.bytes(producer.p.state_second),b.size_bytes(),cudaMemcpyDeviceToHost));i.unpin();}catch(...){i.poisoned=true;i.cursor.invalidate();throw;}
}
void Decoder::close(){if(!impl_)return;if(!impl_->cleanup())throw std::runtime_error("decoder_cleanup_failed_reservation_retained");impl_.reset();}
} // namespace kadan::cuda
