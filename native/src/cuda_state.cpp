#ifdef CUDA_API_PER_THREAD_DEFAULT_STREAM
#error "Kadan state storage requires legacy default-stream semantics"
#endif
#include "kadan/cuda_state.hpp"
#include <cuda_runtime_api.h>
#include <algorithm>
#include <stdexcept>
#include <string>
#include <thread>

namespace kadan::cuda {
namespace {
void check(cudaError_t error,const char* operation) {
    if (error!=cudaSuccess) throw std::runtime_error(std::string(operation)+": "+cudaGetErrorString(error));
}
void require(bool ok,const char* message) { if (!ok) throw std::invalid_argument(message); }
}
struct SequenceState::Impl {
    std::shared_ptr<Resources> resources;
    SequenceStatePlan layout;
    StateCursor cursor;
    std::thread::id owner=std::this_thread::get_id();
    int device;
    Handle handle=0;
    void* storage=nullptr;
    bool loaded=false,pinned=false,poisoned=false,cleanup_failed=false;
    Impl(const checkpoint::TextArchitecture& a,std::size_t capacity,int ordinal,std::shared_ptr<Resources> manager)
        :resources(std::move(manager)),cursor(a.layers,capacity),device(ordinal) {
        require(resources && device>=0,"invalid_cuda_resource_owner");
        auto request=resources->snapshot().capacity;
        require(static_cast<std::size_t>(device)+1<request.size(),"unbudgeted_cuda_device");
        std::array<std::size_t,256> placements{}; placements.fill(device);
        layout=plan_sequence_state(a,capacity,std::span(placements).first(a.layers),std::span(request).subspan(1));
        std::fill(request.begin(),request.end(),0); request[device+1]=bytes();
        handle=resources->reserve(Workload::llm,std::move(request));
        try {
            current_device(); int major=0,minor=0;
            check(cudaDeviceGetAttribute(&major,cudaDevAttrComputeCapabilityMajor,device),"state_device_attribute");
            check(cudaDeviceGetAttribute(&minor,cudaDevAttrComputeCapabilityMinor,device),"state_device_attribute");
            require(major==8 && minor==6,"requires_sm86");
            void* allocation=nullptr;
            check(cudaMalloc(&allocation,bytes()),"state_allocate"); storage=allocation;
            check(cudaMemsetAsync(storage,0,bytes(),cudaStreamLegacy),"state_initialize");
            check(cudaStreamSynchronize(cudaStreamLegacy),"state_initialize_sync");
            resources->loaded(handle); loaded=true;
        } catch (...) { cleanup(); throw; }
    }
    std::size_t bytes() const { return layout.device_bytes[device]; }
    std::uint8_t* pointer(std::size_t offset) const { return static_cast<std::uint8_t*>(storage)+offset; }
    void current_device() const {
        require(std::this_thread::get_id()==owner,"state_thread_changed");
        int current=-1; check(cudaGetDevice(&current),"state_current_device"); require(current==device,"state_device_changed");
    }
    void available() {
        require(handle && !poisoned,"state_storage_unavailable");
        // Wrong-thread/device API misuse may be corrected without mutating
        // state. A CUDA runtime failure instead poisons this arena.
        try { current_device(); } catch (const std::runtime_error&) { fail(); throw; }
    }
    void pin() { resources->pin(handle); pinned=true; }
    void unpin() { resources->unpin(handle); pinned=false; }
    void fail() noexcept { poisoned=true; cursor.invalidate(); }
    bool cleanup() noexcept {
        if (!handle) return true;
        if (cleanup_failed) return false;
        try {
            if (storage) { current_device(); check(cudaStreamSynchronize(cudaStreamLegacy),"state_close_sync"); }
            if (pinned) unpin();
            if (loaded) { resources->begin_eviction(handle); loaded=false; }
            if (storage) { check(cudaFree(storage),"state_free"); storage=nullptr; }
            resources->released(handle); handle=0; cursor.close(); return true;
        } catch (...) { cleanup_failed=true; fail(); return false; }
    }
};
SequenceState::SequenceState(const checkpoint::TextArchitecture& a,std::size_t capacity,int device,std::shared_ptr<Resources> resources)
    :impl_(std::make_unique<Impl>(a,capacity,device,std::move(resources))) {}
SequenceState::~SequenceState() { impl_->cleanup(); }
const SequenceStatePlan& SequenceState::plan() const { return impl_->layout; }
std::size_t SequenceState::committed_tokens() const { return impl_->cursor.committed_tokens(); }
StateStep SequenceState::begin() {
    auto& i=*impl_; i.available();
    const auto step=i.cursor.begin();
    try { i.pin(); } catch (...) { i.fail(); throw; }
    return step;
}
LayerStateView SequenceState::layer(StateStep step,std::size_t index) {
    auto& i=*impl_; i.available(); i.cursor.check_step(step); require(index<i.layout.layers,"state_layer_index");
    const auto& l=i.layout.layer[index]; LayerStateView view{l.kind};
    if (l.kind==checkpoint::LayerKind::full_attention) {
        const auto offset=i.cursor.committed_tokens()*l.kv_token_bytes;
        view.key_history=i.pointer(l.key_offset); view.value_history=i.pointer(l.value_offset);
        view.key_current=i.pointer(l.key_offset+offset); view.value_current=i.pointer(l.value_offset+offset);
        view.history_tokens=i.cursor.committed_tokens()+1; view.kv_token_bytes=l.kv_token_bytes;
    } else {
        view.convolution=i.pointer(l.conv_offset); view.recurrent=i.pointer(l.recurrent_offset);
        view.conv_bytes=l.conv_bytes; view.recurrent_bytes=l.recurrent_bytes;
    }
    return view;
}
void SequenceState::written(StateStep step,std::size_t index) { impl_->available(); impl_->cursor.written(step,index); }
void SequenceState::commit(StateStep step) {
    auto& i=*impl_; i.available(); i.cursor.ready_to_commit(step);
    try { check(cudaStreamSynchronize(cudaStreamLegacy),"state_commit_sync"); i.unpin(); i.cursor.commit(step); }
    catch (...) { i.fail(); throw; }
}
void SequenceState::abort(StateStep step) {
    auto& i=*impl_; i.available(); i.cursor.check_step(step);
    // Invalidate first; uncertain quiescence must never expose mixed layer state.
    i.cursor.abort(step);
    try { check(cudaStreamSynchronize(cudaStreamLegacy),"state_abort_sync"); i.unpin(); }
    catch (...) { i.fail(); throw; }
}
void SequenceState::reset() {
    auto& i=*impl_; i.available(); require(!i.cursor.active(),"state_reset_unavailable"); i.pin();
    try {
        check(cudaMemsetAsync(i.storage,0,i.bytes(),cudaStreamLegacy),"state_reset");
        check(cudaStreamSynchronize(cudaStreamLegacy),"state_reset_sync"); i.unpin(); i.cursor.reset();
    } catch (...) { i.fail(); throw; }
}
void SequenceState::read_bytes(std::size_t offset,std::span<std::uint8_t> destination) {
    auto& i=*impl_; i.available(); require(i.cursor.valid() && !i.cursor.active(),"state_read_unavailable");
    require(offset<=i.bytes() && destination.size()<=i.bytes()-offset,"state_read_bounds");
    i.pin();
    try {
        if (!destination.empty()) check(cudaMemcpy(destination.data(),i.pointer(offset),destination.size(),cudaMemcpyDeviceToHost),"state_read");
        i.unpin();
    } catch (...) { i.fail(); throw; }
}
void SequenceState::close() { if (!impl_->cleanup()) throw std::runtime_error("state_cleanup_failed_reservation_retained"); }
} // namespace kadan::cuda
