#include "kadan/video.hpp"
#include <filesystem>
#include <iostream>
#include <limits>
#include <vector>
using namespace kadan;
void check(bool value) {if(!value)throw std::runtime_error("test_failed");}
template<class F> void rejects(F f,const char* message) {
    try {f();}catch(const std::exception& e){check(std::string(e.what())==message);return;}
    throw std::runtime_error("expected_rejection");
}
int main(int argc,char** argv) {
    check(argc==2);const std::string root=argv[1];using Block=video::H3DecoderBlock;
    auto ledger=std::make_shared<Resources>(Footprint{288*1024*1024,0});
    std::atomic_bool cancel{false};
    const Bytes inputs=(4096+6)*sizeof(float);
    auto admission=ledger->reserve(Workload::video,{inputs,0});
    {
        std::vector<float> x(4096),coords{-0.5f,0.25f,0.75f,0.5f,-0.25f,-0.75f};
        for(std::size_t i=0;i<x.size();++i)x[i]=float(int(i%17)-8)/8;
        Block block(ledger);
        rejects([&]{block.execute(x,coords,root+"/absent",cancel);},"video_not_loaded");
        cancel=true;rejects([&]{block.load(root.c_str(),"weights.safetensors",cancel);},"video_cancelled");cancel=false;
        // FF failure must unwind all previously loaded QKV/RoPE/attention weights.
        rejects([&]{block.load(root.c_str(),"wrong.safetensors",cancel);},"video_tensor_layout");
        check(!block.loaded() && ledger->snapshot().used==Footprint({inputs,0}));
        auto denied=std::make_shared<Resources>(Footprint{Block::weight_bytes-1});
        Block small(denied);
        rejects([&]{small.load(root.c_str(),"weights.safetensors",cancel);},"exhausted");
        check(!small.loaded() && denied->snapshot().used==Footprint({0}));
        block.load(root.c_str(),"weights.safetensors",cancel);
        const auto resident=inputs+Block::weight_bytes;
        check(block.loaded() && ledger->snapshot().used==Footprint({resident,0}));
        rejects([&]{block.load(root.c_str(),"weights.safetensors",cancel);},"video_already_loaded");
        auto pressure=ledger->reserve(Workload::video,{288*1024*1024-resident-Block::intermediate_bytes+1,0});
        rejects([&]{block.execute(x,coords,root+"/denied",cancel);},"exhausted");
        check(!std::filesystem::exists(root+"/denied"));ledger->released(pressure);
        for(const std::string stage:{"qkv","rope","attention","feed_forward"}) {
            bool visited=false;
            rejects([&]{block.execute(x,coords,root+"/cancel-"+stage,cancel,[&](const char* current,std::size_t){
                rejects([&]{block.unload();},"busy");
                if(current==stage){visited=true;cancel=true;}
            });},"video_cancelled");
            check(visited && !std::filesystem::exists(root+"/cancel-"+stage));cancel=false;
            check(ledger->snapshot().used==Footprint({resident,0}));
        }
        rejects([&]{block.execute(x,coords,root+"/hook",cancel,[](const char*,std::size_t){throw std::runtime_error("hook_failure");});},"hook_failure");
        rejects([&]{block.execute(x,coords,root+"/late",cancel,[&](const char* stage,std::size_t count){
            if(std::string(stage)=="feed_forward" && count==2*(16384+2048))cancel=true;
        });},"video_cancelled");cancel=false;
        check(!std::filesystem::exists(root+"/late"));
        std::size_t seen=0;
        block.execute(x,coords,root+"/result",cancel,[&](const char* stage,std::size_t){
            ++seen;const std::string name=stage;
            const auto scratch=name=="qkv" ? video::H3DecoderQkv::scratch_bytes : name=="rope" ? video::H3QkRope::scratch_bytes : name=="attention" ? video::H3DecoderAttention::scratch_bytes : video::H3DecoderFeedForward::scratch_bytes;
            check(ledger->snapshot().used==Footprint({resident+Block::intermediate_bytes+scratch,0}));
            rejects([&]{block.unload();},"busy");
            rejects([&]{block.execute(x,coords,root+"/reentered",cancel);},"busy");
        });check(seen>0);
        rejects([&]{block.execute(x,coords,root+"/result",cancel);},"video_artifact_publish");
        // Resident reuse: one token after a two-token request, without reloading.
        block.execute(std::span<const float>(x.data(),2048),std::span<const float>(coords.data(),3),root+"/reuse",cancel);
        rejects([&]{block.execute({},coords,root+"/empty",cancel);},"video_input_shape");
        std::vector<float> oversized(6144);
        rejects([&]{block.execute(oversized,coords,root+"/oversized",cancel);},"video_input_limit");
        rejects([&]{block.execute(x,{},root+"/shape",cancel);},"video_coordinate_shape");
        coords[0]=2;rejects([&]{block.execute(x,coords,root+"/range",cancel);},"video_coordinate_range");coords[0]=0;
        x[0]=std::numeric_limits<float>::quiet_NaN();
        rejects([&]{block.execute(x,coords,root+"/nan",cancel);},"video_nonfinite_input");
        for(const auto& entry:std::filesystem::directory_iterator(root))check(entry.path().string().find(".partial-")==std::string::npos);
        check(ledger->snapshot().used==Footprint({resident,0}));block.unload();block.unload();
        check(ledger->snapshot().used==Footprint({inputs,0}));
        {Block scoped(ledger);scoped.load(root.c_str(),"weights.safetensors",cancel);}
        check(ledger->snapshot().used==Footprint({inputs,0}));
    }
    ledger->released(admission);check(ledger->snapshot().used==Footprint({0,0}));
    std::cout<<"Block cancellation at every stage, failure rollback, reuse, pins, accounting, cleanup passed\n";
}
