#pragma once
#include "kadan/resources.hpp"
#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <functional>
#include <future>
#include <memory>
#include <thread>

namespace kadan {
// The CUDA adapter and CPU-only unit fakes implement this lifecycle boundary.
// Failed allocate/create operations must leave no allocation/handle behind.
// These resource methods run on one owned thread; compute is supplied separately.
class DeviceOperations {
public:
    virtual ~DeviceOperations() = default;
    virtual void select(int device) = 0;
    virtual void allocate(Bytes bytes) = 0;
    virtual void free() = 0;
    virtual void create_handle() = 0;
    virtual void configure_handle() = 0;
    virtual void destroy_handle() = 0;
    virtual void synchronize() = 0;
    virtual void reset() = 0;
    virtual void release_weights() {}
};

// One exact-sized high-water allocation, bounded by the device ledger. Growth
// frees the old allocation before admitting a replacement: never two buffers.
// Context/library overhead remains covered by the caller's context reservation.
class DeviceWorkspace {
public:
    DeviceWorkspace(Resources& resources, int device, Workload workload,
                    std::unique_ptr<DeviceOperations> operations)
        : resources_(resources), device_(device), workload_(workload), operations_(std::move(operations)) {
        const auto capacity=resources_.snapshot().capacity;
        if(!operations_ || device<0 || std::size_t(device)+1>=capacity.size() || !capacity[device+1])
            throw std::invalid_argument("compute_workspace_device");
    }
    DeviceWorkspace(const DeviceWorkspace&)=delete;
    DeviceWorkspace& operator=(const DeviceWorkspace&)=delete;
    DeviceOperations& operations() { return *operations_; }
    Bytes retained_bytes() const { return bytes_; }
    bool quarantined() const { return quarantined_; }
    void activate() {
        healthy();touched_=true;
        try {operations_->select(device_);}catch(...){quarantined_=true;throw;}
    }
    void ensure(Bytes bytes) {
        healthy();
        if(!bytes)throw std::invalid_argument("compute_workspace_empty");
        if(bytes<=bytes_)return;
        try {
            // Even a failed selection may have initialized a context. Keep its
            // ownership until a successful explicit reset or process exit.
            activate();
            release_buffer();
            auto footprint=resources_.snapshot().capacity;
            std::fill(footprint.begin(),footprint.end(),0);footprint[device_+1]=bytes;
            reservation_=resources_.reserve(workload_,std::move(footprint));
            operations_->allocate(bytes);allocated_=true;bytes_=bytes;
            resources_.loaded(reservation_);resident_=true;resources_.pin(reservation_);pinned_=true;
            if(!handle_){operations_->create_handle();handle_=true;operations_->configure_handle();}
        } catch (...) {
            const auto error=std::current_exception();
            release(); // Cleanup failure takes precedence and retains accounting.
            std::rethrow_exception(error);
        }
    }
    void synchronize() {
        healthy();
        if(!touched_)return;
        try { operations_->synchronize(); }
        catch (...) { quarantined_=true;throw; }
    }
    // A request boundary, including cancellation. No scratch/handle residency
    // survives successful release; the execution thread can be reused.
    void release() {
        healthy();
        try {
            if(touched_)operations_->synchronize();
            // Destroy before freeing: the BLAS handle may reference workspace.
            if(handle_){operations_->destroy_handle();handle_=false;}
            release_buffer();
        } catch (...) { quarantined_=true;throw; }
    }
    // Dedicated worker context owner only. Caller releases context accounting
    // only AFTER every selected device acknowledges this operation.
    void reset() {
        release();
        try { operations_->release_weights(); }
        catch (...) { quarantined_=true;throw; }
        if(!touched_)return;
        try { operations_->reset();touched_=false; }
        catch (...) { quarantined_=true;throw; }
    }
private:
    void healthy() const { if(quarantined_)throw std::runtime_error("compute_workspace_cleanup_unconfirmed"); }
    void release_buffer() {
        try {
            if(allocated_){
                synchronize();
                // A handle's previous custom workspace must not outlive its buffer.
                // Growth retires the handle as well; ordinary reuse creates neither.
                if(handle_){operations_->destroy_handle();handle_=false;}
                operations_->free();allocated_=false;
            }
            if(reservation_){
                if(pinned_){resources_.unpin(reservation_);pinned_=false;}
                if(resident_){resources_.begin_eviction(reservation_);resident_=false;}
                resources_.released(reservation_);reservation_=0;
            }
            bytes_=0;
        } catch (...) {quarantined_=true;throw;}
    }
    Resources& resources_;
    int device_;
    Workload workload_;
    std::unique_ptr<DeviceOperations> operations_;
    Handle reservation_=0;
    Bytes bytes_=0;
    bool allocated_=false,handle_=false,pinned_=false,resident_=false,touched_=false,quarantined_=false;
};

// Bounded mailbox, persistent thread. The synchronous compute boundary joins
// every submitted future before host spans go out of scope. Destruction drains
// the one pending job and performs final cleanup on that same owning thread.
class DeviceThread {
public:
    DeviceThread(Resources& resources,int device,Workload workload,std::unique_ptr<DeviceOperations> operations)
        : workspace_(resources,device,workload,std::move(operations)),thread_([this]{run();}) {}
    ~DeviceThread() {
        {std::lock_guard lock(mutex_);stopping_=true;}
        ready_.notify_one();thread_.join();
    }
    DeviceThread(const DeviceThread&)=delete;
    DeviceThread& operator=(const DeviceThread&)=delete;
    std::future<void> submit_work(std::function<void(DeviceWorkspace&)> function,std::atomic_bool* failed=nullptr) {
        return submit([function=std::move(function),failed](DeviceWorkspace& workspace){
            try {function(workspace);workspace.synchronize();}
            catch (...) {if(failed)*failed=true;const auto error=std::current_exception();workspace.release();std::rethrow_exception(error);}
        });
    }
    std::future<void> submit(std::function<void(DeviceWorkspace&)> function) {
        std::packaged_task<void()> task([this,function=std::move(function)]{function(workspace_);});
        auto result=task.get_future();
        {std::lock_guard lock(mutex_);
            if(stopping_||pending_.valid())throw std::runtime_error("compute_device_thread_busy");
            pending_=std::move(task);
        }
        ready_.notify_one();return result;
    }
private:
    void run() noexcept {
        for(;;){
            std::packaged_task<void()> task;
            {std::unique_lock lock(mutex_);ready_.wait(lock,[this]{return stopping_||pending_.valid();});
                if(!pending_.valid())break;
                task=std::move(pending_);
            }
            task();
        }
        // Destructors cannot acknowledge cleanup failure. Leave uncertain
        // allocations charged; the parent must reap the worker before release.
        try {workspace_.reset();}catch(...){}
    }
    DeviceWorkspace workspace_;
    std::mutex mutex_;
    std::condition_variable ready_;
    bool stopping_=false;
    std::packaged_task<void()> pending_;
    std::thread thread_;
};
}
