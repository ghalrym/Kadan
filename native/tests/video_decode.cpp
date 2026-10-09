#include "kadan/video.hpp"
#include <filesystem>
#include <chrono>
#include <iostream>
#include <limits>
#include <vector>
using namespace kadan;
namespace {
void check(bool ok){if(!ok)throw std::runtime_error("test_failed");}
template<class F>void rejects(F action,const std::string& expected){try{action();}catch(const std::exception& e){if(std::string(e.what()).find(expected)!=std::string::npos)return;throw;}throw std::runtime_error("expected_"+expected);}
}
int main(int argc,char** argv) {
    check(argc==2);const std::string root=argv[1];std::atomic_bool cancel{false};
    using Decoder=video::H3VideoDecoder;
    constexpr Bytes budget=512*1024*1024;
    auto ledger=std::make_shared<Resources>(Footprint{budget,0});
    constexpr Bytes input_bytes=48*sizeof(float);
    auto input=ledger->reserve(Workload::video,{input_bytes,0});
    {
        std::vector<float> values(48);
        for(std::size_t i=0;i<values.size();++i)values[i]=float(int(i%17)-8)/8;
        Decoder decoder(ledger);
        rejects([&]{decoder.execute(values,1,1,2,root+"/absent",cancel);},"video_not_loaded");
        cancel=true;rejects([&]{decoder.load(root.c_str(),"weights.safetensors",cancel);},"video_cancelled");cancel=false;
        auto low=std::make_shared<Resources>(Footprint{Decoder::weight_bytes-1});Decoder denied(low);
        rejects([&]{denied.load(root.c_str(),"weights.safetensors",cancel);},"exhausted");check(low->snapshot().used==Footprint({0}));
        decoder.load(root.c_str(),"weights.safetensors",cancel);
        const Bytes resident=input_bytes+Decoder::weight_bytes+video::H3DecoderInput::weight_bytes+video::H3DecoderInput::metadata_bytes;
        check(ledger->snapshot().used==Footprint({resident,0}));
        rejects([&]{decoder.load(root.c_str(),"weights.safetensors",cancel);},"video_already_loaded");
        rejects([&]{decoder.execute(values,0,1,2,root+"/zero",cancel);},"video_latent_shape_or_limit");
        rejects([&]{decoder.execute(values,std::numeric_limits<std::size_t>::max(),1,1,root+"/overflow",cancel);},"video_latent_shape_or_limit");
        rejects([&]{decoder.execute(values,1,1,1,root+"/shape",cancel);},"video_input_shape");
        auto pressure=ledger->reserve(Workload::video,{budget-resident-1,0});
        rejects([&]{decoder.execute(values,1,1,2,root+"/denied",cancel);},"exhausted");ledger->released(pressure);
        pressure=ledger->reserve(Workload::video,{budget-resident-1024*1024,0});
        rejects([&]{decoder.execute(values,1,1,2,root+"/block-denied",cancel);},"exhausted");
        ledger->released(pressure);check(ledger->snapshot().used==Footprint({resident,0}));
        for(const std::string stage:{"input","block_loaded","block_completed","publish"}) {
            bool seen=false;
            rejects([&]{decoder.execute(values,1,1,2,root+"/cancel-"+stage,cancel,[&](const char* current,std::size_t){
                rejects([&]{decoder.unload();},"busy");
                rejects([&]{decoder.execute(values,1,1,2,root+"/reentered",cancel);},"busy");
                if(stage==current){seen=true;cancel=true;}
            });},"video_cancelled");cancel=false;
            check(seen && !std::filesystem::exists(root+"/cancel-"+stage));
            check(ledger->snapshot().used==Footprint({resident,0}));
        }
        rejects([&]{decoder.execute(values,1,1,2,root+"/hook",cancel,[](const char*,std::size_t){throw std::runtime_error("hook_failure");});},"hook_failure");
        std::size_t completed=0;Bytes peak=0;
        decoder.execute(values,1,1,2,root+"/result",cancel,[&](const char* stage,std::size_t count){
            const auto used=ledger->snapshot().used;check(used[1]==0 && used[0]<=budget);peak=std::max(peak,used[0]);
            if(std::string(stage)=="block_completed")check(count==++completed);
        });check(completed==36 && peak>Decoder::weight_bytes+video::H3DecoderBlock::weight_bytes);
        // Decode a different axis shape without reloading the resident outer weights.
        decoder.execute(values,2,1,1,root+"/reuse",cancel);
        decoder.execute(values,1,2,1,root+"/height",cancel);
        rejects([&]{decoder.execute(values,1,1,2,root+"/result",cancel);},"video_artifact_publish");
        check(std::filesystem::exists(root+"/result"));
        values[0]=std::numeric_limits<float>::quiet_NaN();
        rejects([&]{decoder.execute(values,1,1,2,root+"/nan",cancel);},"video_nonfinite_input");
        check(ledger->snapshot().used==Footprint({resident,0}));
        values[0]=0;
        rejects([&]{decoder.execute(values,1,1,2,root+"/changed",cancel,[&](const char* stage,std::size_t){
            if(std::string(stage)=="input") {
                const auto path=root+"/weights.safetensors";
                std::filesystem::last_write_time(path,std::filesystem::last_write_time(path)+std::chrono::seconds(1));
            }
        });},"changed");
        check(!std::filesystem::exists(root+"/changed") && ledger->snapshot().used==Footprint({resident,0}));
        decoder.unload();decoder.unload();check(ledger->snapshot().used==Footprint({input_bytes,0}));
        for(const auto& entry:std::filesystem::directory_iterator(root))check(entry.path().string().find(".partial-")==std::string::npos);
    }
    ledger->released(input);check(ledger->snapshot().used==Footprint({0,0}));
    std::cout<<"36-block decoder cancellation, frame reconstruction, reuse and bounded host accounting passed\n";
}
