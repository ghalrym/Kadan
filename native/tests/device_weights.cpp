#include "kadan/device_weights.hpp"
#include <iostream>
using namespace kadan;
void check(bool b){if(!b)throw std::runtime_error("weight_test_failed");}
template<class F>void fails(F f){try{f();}catch(const std::exception&){return;}throw std::runtime_error("expected_failure");}
struct Fake final:WeightDeviceOperations {
    Resources& resources;std::map<void*,Bytes> pointers;unsigned copies=0,frees=0;std::string fail;
    std::atomic_bool* cancellation=nullptr;
    explicit Fake(Resources& r):resources(r){}
    void* allocate_weight(Bytes bytes)override{
        check(resources.snapshot().used[1]>=bytes&&resources.snapshot().used[0]>=DeviceWeights::metadata_bytes);
        if(fail=="allocate")throw std::runtime_error("allocate");
        // Distinct fake handles; these are never dereferenced or sent to a GPU.
        auto pointer=reinterpret_cast<void*>(std::uintptr_t(pointers.size()+1));pointers.emplace(pointer,bytes);return pointer;
    }
    void copy_weight(void* p,std::size_t offset,std::span<const std::byte> source)override{
        check(pointers.contains(p)&&source.size()<=1024*1024&&offset+source.size()<=pointers.at(p));++copies;
        if(cancellation)cancellation->store(true);
        if(fail=="copy")throw std::runtime_error("copy");
    }
    void free_weight(void* p)override{if(fail=="free")throw std::runtime_error("free");check(pointers.erase(p)==1);++frees;}
    void synchronize_weights()override{if(fail=="sync")throw std::runtime_error("sync");}
};
int main(){
    const Bytes host=DeviceWeights::metadata_bytes;std::atomic_bool cancel=false;
    std::array<float,8> values{};auto bytes=std::as_bytes(std::span(values));DeviceWeights::Key key{{1},0,8,4};
    {
        Resources r({host,1024});Fake gpu(r);DeviceWeights bank(r,0,Workload::image,64,gpu);
        auto first=bank.retain(key,bytes,cancel);check(first&&gpu.copies==1&&bank.allocated()==32&&bank.reserved()==64);
        check(bank.retain(key,bytes,cancel)==first&&gpu.copies==1);
        auto next=key;next.first=8;check(bank.retain(next,bytes,cancel));next.first=16;
        check(bank.retain(next,bytes,cancel)==nullptr&&bank.allocated()==64);
        check(r.snapshot().residents==1);bank.release();check(gpu.frees==2&&r.snapshot().residents==0);
        bank.bound(16);check(!bank.retain(key,bytes,cancel)&&r.snapshot().residents==0);
    }
    for(const auto* fail:{"allocate","copy"}){
        Resources r({host,1024});Fake gpu(r);DeviceWeights bank(r,0,Workload::video,64,gpu);gpu.fail=fail;
        fails([&]{bank.retain(key,bytes,cancel);});check(!bank.quarantined()&&bank.allocated()==0&&gpu.pointers.empty());
        gpu.fail.clear();bank.release();check(r.snapshot().residents==0);
    }
    {
        Resources r({host,1024});Fake gpu(r);DeviceWeights bank(r,0,Workload::video,64,gpu);gpu.cancellation=&cancel;
        fails([&]{bank.retain(key,bytes,cancel);});check(bank.allocated()==0&&gpu.pointers.empty());cancel=false;bank.release();
    }
    for(const auto* failure:{"free","sync"}){
        Resources r({host,1024});Fake gpu(r);DeviceWeights bank(r,0,Workload::image,64,gpu);
        bank.retain(key,bytes,cancel);gpu.fail=failure;fails([&]{bank.release();});
        check(bank.quarantined()&&r.snapshot().used==Footprint({host,64})&&bank.allocated()==32);
        gpu.fail.clear();fails([&]{bank.retain(key,bytes,cancel);});fails([&]{bank.release();});
    }
    {
        Resources r({host,1024});Fake gpu(r);DeviceWeights bank(r,0,Workload::image,64,gpu);
        auto invalid=key;invalid.count=1;fails([&]{bank.retain(invalid,bytes,cancel);});check(r.snapshot().residents==0);
    }
    std::cout<<"Mock immutable device-weight ownership passed (no CUDA or inference)\n";
}
