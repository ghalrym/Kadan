#pragma once
#include "kadan/cleanup_error.hpp"
#include "kadan/checkpoint.hpp"
#include "kadan/resources.hpp"
#include <algorithm>
#include <atomic>
#include <map>
#include <span>

namespace kadan {
class WeightDeviceOperations {
public:
    virtual ~WeightDeviceOperations()=default;
    // Free physical bytes available for weights after the scratch allowance.
    virtual Bytes available_weight_bytes()=0;
    // nullptr means confirmed allocation OOM, without an outstanding allocation.
    virtual void* allocate_weight(Bytes)=0;
    virtual void copy_weight(void*,std::size_t,std::span<const std::byte>)=0;
    virtual void free_weight(void*)=0;
    virtual void synchronize_weights()=0;
};
// One selected device, used only on its owned thread. Exact F32/I8 bytes;
// scratch remains separate. The fixed envelope uses one ledger slot even for
// thousands of projections. Failed physical cleanup keeps that envelope charged.
class DeviceWeights {
public:
    struct Key {
        WeightIdentity source{};
        std::size_t first=0,count=0,width=0;
        auto operator<=>(const Key&) const=default;
    };
    static constexpr Bytes metadata_bytes=4*1024*1024;
    DeviceWeights(Resources& resources,int device,Workload workload,Bytes limit,WeightDeviceOperations& operations)
        : resources_(resources),device_(device),workload_(workload),limit_(limit),maximum_(limit),operations_(operations),
          metadata_(metadata_bytes),entries_(&metadata_){}
    DeviceWeights(const DeviceWeights&)=delete;
    DeviceWeights& operator=(const DeviceWeights&)=delete;
    Bytes reserved() const{return reservation_?limit_:0;}
    Bytes allocated() const{return allocated_;}
    bool quarantined() const{return quarantined_;}
    void bound(Bytes bytes){healthy();if(reservation_)throw std::runtime_error("weight_plan_already_loaded");limit_=std::min(maximum_,bytes);}
    Bytes capacity() const{return limit_;}
    Bytes maximum_capacity() const{return maximum_;}
    void* find(const Key& key) const{
        healthy();auto it=entries_.find(key);return it==entries_.end()?nullptr:it->second.pointer;
    }
    // nullptr explicitly selects caller-owned streaming scratch. A cache miss
    // cannot consume its scratch reserve or silently change representation.
    void* retain(const Key& key,std::span<const std::byte> source,const std::atomic_bool& cancel){
        healthy();stop(cancel);
        if(!key.width||key.count>std::numeric_limits<std::size_t>::max()/key.width||key.count*key.width!=source.size())
            throw std::invalid_argument("weight_representation_size");
        if(auto pointer=find(key))return pointer;
        if(source.empty()||source.size()>limit_-allocated_||source.size()>operations_.available_weight_bytes())return nullptr;
        if(!reservation_){
            auto footprint=resources_.snapshot().capacity;
            if(device_<0||std::size_t(device_)+1>=footprint.size())throw std::invalid_argument("weight_device");
            std::fill(footprint.begin(),footprint.end(),0);footprint[0]=metadata_bytes;footprint[device_+1]=limit_;
            reservation_=resources_.reserve(workload_,std::move(footprint));
        }
        auto entry=entries_.end();
        try{entry=entries_.try_emplace(key).first;}catch(const std::bad_alloc&){return nullptr;}
        try{
            entry->second.pointer=operations_.allocate_weight(source.size());
            if(!entry->second.pointer){entries_.erase(entry);return nullptr;}
            entry->second.bytes=source.size();
            allocated_+=source.size();
            for(std::size_t at=0;at<source.size();){
                stop(cancel);const auto count=std::min<std::size_t>(1024*1024,source.size()-at);
                operations_.copy_weight(entry->second.pointer,at,source.subspan(at,count));at+=count;
            }
            operations_.synchronize_weights();stop(cancel);return entry->second.pointer;
        }catch(...){
            const auto error=std::current_exception();
            rethrow_after_cleanup(error,[&]{try{drop(entry);}catch(...){quarantined_=true;throw;}});
        }
    }
    void release(){
        healthy();
        try{
            if(!entries_.empty())operations_.synchronize_weights();
            while(!entries_.empty())drop(entries_.begin());
            if(reservation_){resources_.released(reservation_);reservation_=0;}
        }catch(...){quarantined_=true;throw;}
    }
private:
    struct Entry {void* pointer=nullptr;Bytes bytes=0;};
    using Entries=std::pmr::map<Key,Entry>;
    void healthy() const{if(quarantined_)throw std::runtime_error("weight_cleanup_unconfirmed");}
    static void stop(const std::atomic_bool& cancel){if(cancel.load())throw std::runtime_error("weight_cancelled");}
    void drop(Entries::iterator entry){
        if(entry->second.pointer){operations_.synchronize_weights();operations_.free_weight(entry->second.pointer);allocated_-=entry->second.bytes;}
        entries_.erase(entry);
    }
    Resources& resources_;int device_;Workload workload_;Bytes limit_,maximum_;WeightDeviceOperations& operations_;
    checkpoint::MemoryBudget metadata_;Entries entries_;
    Handle reservation_=0;Bytes allocated_=0;bool quarantined_=false;
};
}
