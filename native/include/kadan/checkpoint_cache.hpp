#pragma once
#include "kadan/checkpoint.hpp"
#include "kadan/resources.hpp"
#include <algorithm>
#include <atomic>
#include <map>

namespace kadan::checkpoint {
// Exact serialized bytes, reusable across shard reopenings. One worker owns the
// cache; a caller must serialize reads/clear. Every payload and map allocation
// shares one admitted RAM envelope. Overflow streams without evicting hot data.
class ReadCache {
public:
    using Key=std::array<std::uint64_t,9>; // file identity plus payload interval
    using Source=std::function<void(std::size_t,std::span<std::uint8_t>)>;
    static constexpr std::size_t chunk_bytes=1024*1024;
    struct Stats {std::size_t allocated=0,entries=0;std::uint64_t hits=0,misses=0,source_bytes=0;};
    ReadCache(std::shared_ptr<Resources> resources,Bytes limit,Workload workload)
        : resources_(std::move(resources)),budget_(std::make_shared<MemoryBudget>(limit)),entries_(budget_.get()){
        if(!resources_||!limit||limit>std::numeric_limits<std::size_t>::max())throw std::invalid_argument("checkpoint_cache_budget");
        auto footprint=resources_->snapshot().capacity;std::fill(footprint.begin(),footprint.end(),0);footprint[0]=limit;
        reservation_=resources_->reserve(workload,std::move(footprint));
    }
    ~ReadCache(){entries_.clear();resources_->released(reservation_);}
    ReadCache(const ReadCache&)=delete;
    ReadCache& operator=(const ReadCache&)=delete;
    Stats stats() const{return {budget_->used(),entries_.size(),hits_,misses_,source_bytes_};}
    Bytes reserved_bytes() const{return budget_->limit();}
    void clear(){entries_.clear();}
    void read(const Key& key,std::size_t bytes,std::size_t offset,std::span<std::uint8_t> destination,
              const Source& source,const std::atomic_bool* cancel){
        stop(cancel);
        if(offset>bytes||destination.size()>bytes-offset)throw std::invalid_argument("checkpoint_cache_range");
        if(destination.empty())return;
        auto found=entries_.find(key);
        if(found!=entries_.end()){
            if(found->second.size()!=bytes)throw std::invalid_argument("checkpoint_cache_identity");
            ++hits_;copy_cached(found->second.data()+offset,destination,cancel);return;
        }
        ++misses_;
        if(bytes<=budget_->limit()-budget_->used()){
            // Map and vector use the same strict allocator. A metadata admission
            // failure also falls back to bounded file reads, with no extra cache.
            auto entry=entries_.end();
            try{entry=entries_.try_emplace(key,budget_.get(),bytes).first;}
            catch(const std::bad_alloc&){} // destination is separately caller-admitted
            if(entry!=entries_.end()){
                try{copy_source(0,{entry->second.data(),bytes},source,cancel);}
                catch(...){entries_.erase(entry);throw;}
                copy_cached(entry->second.data()+offset,destination,cancel);return;
            }
        }
        copy_source(offset,destination,source,cancel);
    }
private:
    struct Buffer {
        std::pmr::memory_resource* resource;
        std::uint8_t* bytes;
        std::size_t count;
        // Allocate without touching multi-GiB payload pages. Only cancellable
        // bounded source reads initialize them before the entry becomes usable.
        Buffer(std::pmr::memory_resource* r,std::size_t n):resource(r),
            bytes(static_cast<std::uint8_t*>(r->allocate(n,alignof(std::uint8_t)))),count(n){}
        ~Buffer(){resource->deallocate(bytes,count,alignof(std::uint8_t));}
        Buffer(const Buffer&)=delete;
        Buffer& operator=(const Buffer&)=delete;
        std::uint8_t* data(){return bytes;}
        std::size_t size() const{return count;}
    };
    static void stop(const std::atomic_bool* cancel){if(cancel&&cancel->load())throw std::runtime_error("checkpoint_cache_cancelled");}
    static void copy_cached(const std::uint8_t* source,std::span<std::uint8_t> destination,const std::atomic_bool* cancel){
        while(!destination.empty()){
            stop(cancel);const auto count=std::min(chunk_bytes,destination.size());
            std::copy_n(source,count,destination.data());source+=count;destination=destination.subspan(count);
        }
        stop(cancel);
    }
    void copy_source(std::size_t offset,std::span<std::uint8_t> destination,const Source& source,const std::atomic_bool* cancel){
        while(!destination.empty()){
            stop(cancel);const auto count=std::min(chunk_bytes,destination.size());
            source(offset,destination.first(count));source_bytes_+=count;stop(cancel);
            offset+=count;destination=destination.subspan(count);
        }
    }
    std::shared_ptr<Resources> resources_;
    std::shared_ptr<MemoryBudget> budget_;
    std::pmr::map<Key,Buffer> entries_;
    Handle reservation_=0;
    std::uint64_t hits_=0,misses_=0,source_bytes_=0;
};
}
