#include "kadan/video.hpp"
#include <filesystem>
#include <iostream>
#include <limits>
#include <vector>
using namespace kadan;
void check(bool value) { if (!value) throw std::runtime_error("test_failed"); }
template<class F> void rejects(F f, const char* message) {
    try { f(); } catch (const std::exception& e) { check(std::string(e.what())==message); return; }
    throw std::runtime_error("expected_rejection");
}
int main(int argc,char** argv) {
    check(argc==2); const std::string root=argv[1];
    using Stage=video::H3DecoderQkv;
    auto ledger=std::make_shared<Resources>(Footprint{64*1024*1024,0});
    std::atomic_bool cancel{false}; Stage stage(ledger);
    // Caller input remains charged through publication/cancellation.
    auto admission=ledger->reserve(Workload::video,{8*2048*sizeof(float),0});
    {
    std::vector<float> input(8*2048);
    for(std::size_t i=0;i<input.size();++i) input[i]=float(int(i%17)-8)/8;
    rejects([&]{stage.execute(input,root+"/absent",cancel);},"video_not_loaded");
    cancel=true; rejects([&]{stage.load(root.c_str(),"weights.safetensors",cancel);},"video_cancelled"); cancel=false;
    for(const auto& name:{"wrong.safetensors","shape.safetensors"})
        rejects([&]{stage.load(root.c_str(),name,cancel);},"video_tensor_layout");
    rejects([&]{stage.load(root.c_str(),"nan.safetensors",cancel);},"video_nonfinite_weight");
    check(ledger->snapshot().used[0]==input.size()*sizeof(float));
    auto small=std::make_shared<Resources>(Footprint{Stage::metadata_bytes});
    Stage denied(small); rejects([&]{denied.load(root.c_str(),"weights.safetensors",cancel);},"exhausted");
    check(small->snapshot().used[0]==0);
    stage.load(root.c_str(),"weights.safetensors",cancel);
    const auto resident=Stage::weight_bytes+input.size()*sizeof(float);
    check(ledger->snapshot().used==Footprint({resident,0}));
    rejects([&]{stage.load(root.c_str(),"weights.safetensors",cancel);},"video_already_loaded");
    auto pressure=ledger->reserve(Workload::video,{64*1024*1024-resident-Stage::scratch_bytes+1,0});
    rejects([&]{stage.execute(input,root+"/exhausted",cancel);},"exhausted");
    check(!std::filesystem::exists(root+"/exhausted"));
    ledger->released(pressure);
    check(ledger->snapshot().used==Footprint({resident,0}));
    std::size_t observed=0;
    stage.execute(input,root+"/result",cancel,[&](std::size_t rows){
        check(rows==observed+64);observed=rows;
        check(ledger->snapshot().used==Footprint({resident+Stage::scratch_bytes,0}));
        rejects([&]{stage.unload();},"busy");
    });
    check(observed==8*6144);
    rejects([&]{stage.execute(input,root+"/result",cancel);},"video_artifact_publish");
    rejects([&]{stage.execute({},root+"/empty",cancel);},"video_input_shape");
    rejects([&]{stage.execute(std::span<const float>(input.data(),1),root+"/bad",cancel);},"video_input_shape");
    std::vector<float> oversized(9*2048);
    rejects([&]{stage.execute(oversized,root+"/oversized",cancel);},"video_input_limit");
    rejects([&]{stage.execute(input,root+"/cancelled",cancel,[&](std::size_t rows){check(rows==64);cancel=true;});},"video_cancelled");
    check(!std::filesystem::exists(root+"/cancelled"));cancel=false;
    rejects([&]{stage.execute(input,root+"/late-cancel",cancel,[&](std::size_t rows){if(rows==8*6144)cancel=true;});},"video_cancelled");
    check(!std::filesystem::exists(root+"/late-cancel"));cancel=false;
    rejects([&]{stage.execute(input,root+"/hook-failed",cancel,[](std::size_t){throw std::runtime_error("hook_failure");});},"hook_failure");
    rejects([&]{stage.execute(input,root+"/missing/output",cancel);},"video_artifact_open");
    input[0]=std::numeric_limits<float>::infinity();
    rejects([&]{stage.execute(input,root+"/nonfinite",cancel);},"video_nonfinite_input");
    input[0]=std::numeric_limits<float>::max();
    rejects([&]{stage.execute(input,root+"/overflow",cancel);},"video_nonfinite_output");
    check(ledger->snapshot().used==Footprint({resident,0}));
    for(const auto& p:std::filesystem::directory_iterator(root)) check(p.path().string().find(".partial-")==std::string::npos);
    stage.unload();stage.unload();
    check(ledger->snapshot().used[0]==input.size()*sizeof(float));
    {Stage scoped(ledger);scoped.load(root.c_str(),"weights.safetensors",cancel);}
    check(ledger->snapshot().used[0]==input.size()*sizeof(float));
    }
    ledger->released(admission);check(ledger->snapshot().used==Footprint({0,0}));
    std::cout<<"QKV cancellation, pins, eviction, accounting and artifact cleanup passed\n";
}
