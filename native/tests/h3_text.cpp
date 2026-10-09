#include "kadan/h3_text.hpp"
#include <filesystem>
#include <iostream>
#include <vector>
using namespace kadan;
namespace {
void check(bool ok){if(!ok)throw std::runtime_error("test_failed");}
template<class F>void rejects(F action,const std::string& expected){try{action();}catch(const std::exception& e){if(std::string(e.what()).find(expected)!=std::string::npos)return;throw;}throw std::runtime_error("expected_"+expected);}
}
int main(int argc,char** argv) {
    check(argc==2);const std::string root=argv[1];std::atomic_bool cancel{false};
    using Encoder=video::H3TextEncoder;constexpr Bytes budget=256*1024*1024;
    auto ledger=std::make_shared<Resources>(Footprint{budget,0});
    auto input=ledger->reserve(Workload::video,{8,0});
    {
        std::array<std::uint32_t,2> ids{0,1};Encoder encoder(ledger);
        rejects([&]{encoder.execute(ids,root+"/absent",cancel);},"h3_text_not_loaded");
        cancel=true;rejects([&]{encoder.load(root.c_str(),"weights.safetensors",cancel);},"h3_text_cancelled");cancel=false;
        auto tiny=std::make_shared<Resources>(Footprint{Encoder::metadata_bytes-1});Encoder denied(tiny);
        rejects([&]{denied.load(root.c_str(),"weights.safetensors",cancel);},"exhausted");check(tiny->snapshot().used==Footprint({0}));
        encoder.load(root.c_str(),"weights.safetensors",cancel);
        const Bytes resident=8+Encoder::metadata_bytes;check(ledger->snapshot().used==Footprint({resident,0}));
        rejects([&]{encoder.execute({},root+"/empty",cancel);},"h3_text_token_limit");
        std::vector<std::uint32_t> too_many(Encoder::max_tokens+1);
        rejects([&]{encoder.execute(too_many,root+"/oversized",cancel);},"h3_text_token_limit");
        ids[0]=Encoder::vocab;rejects([&]{encoder.execute(ids,root+"/bad-id",cancel);},"h3_text_token_id");ids[0]=0;
        auto pressure=ledger->reserve(Workload::video,{budget-resident-1024*1024,0});
        rejects([&]{encoder.execute(ids,root+"/denied",cancel);},"exhausted");ledger->released(pressure);
        for(const std::string stage:{"projection","layer_completed"}) {
            bool seen=false;
            rejects([&]{encoder.execute(ids,root+"/cancel-"+stage,cancel,[&](const char* current,std::size_t){
                rejects([&]{encoder.unload();},"busy");
                rejects([&]{encoder.execute(ids,root+"/reentry",cancel);},"busy");
                if(stage==current){seen=true;cancel=true;}
            });},"h3_text_cancelled");cancel=false;check(seen && !std::filesystem::exists(root+"/cancel-"+stage));
            check(ledger->snapshot().used==Footprint({resident,0}));
        }
        rejects([&]{encoder.execute(ids,root+"/hook",cancel,[](const char*,std::size_t){throw std::runtime_error("hook_failure");});},"hook_failure");
        std::size_t layers=0;Bytes peak=0;
        encoder.execute(ids,root+"/result",cancel,[&](const char* stage,std::size_t count){
            const auto used=ledger->snapshot().used;check(used[1]==0 && used[0]<=budget);peak=std::max(peak,used[0]);
            if(std::string(stage)=="layer_completed")check(count==++layers);
        });check(layers==50 && peak>128*1024*1024);
        encoder.execute(std::span<const std::uint32_t>(ids.data(),1),root+"/reuse",cancel);
        check(ledger->snapshot().used==Footprint({resident,0}));
        rejects([&]{encoder.execute(std::span<const std::uint32_t>(ids.data(),1),root+"/late",cancel,[&](const char* stage,std::size_t){
            if(std::string(stage)=="publish")cancel=true;
        });},"h3_text_cancelled");cancel=false;
        check(!std::filesystem::exists(root+"/late") && ledger->snapshot().used==Footprint({resident,0}));
        encoder.unload();encoder.unload();check(ledger->snapshot().used==Footprint({8,0}));
        for(const auto& entry:std::filesystem::directory_iterator(root))check(entry.path().string().find(".partial-")==std::string::npos);
        std::cout<<"layers=50; peak_host_bytes="<<peak<<"; cancellation/reuse/cleanup passed\n";
    }
    ledger->released(input);check(ledger->snapshot().used==Footprint({0,0}));
}
