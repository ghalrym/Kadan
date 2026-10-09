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
    check(argc==2);
    const std::string root=argv[1];
    auto ledger=std::make_shared<Resources>(Footprint{8*1024*1024,0});
    std::atomic_bool cancel{false};
    video::H3DecoderInput stage(ledger);
    std::vector<float> input(3*24);
    for (std::size_t i=0;i<input.size();++i) input[i]=float(int(i%17)-8)/8;
    rejects([&]{stage.execute(input,root+"/absent",cancel);},"video_not_loaded");
    cancel=true;
    rejects([&]{stage.load(root.c_str(),"weights.safetensors",cancel);},"video_cancelled");
    check(ledger->snapshot().residents==0); cancel=false;
    rejects([&]{stage.load(root.c_str(),"wrong.safetensors",cancel);},"video_tensor_layout");
    rejects([&]{stage.load(root.c_str(),"nan.safetensors",cancel);},"video_nonfinite_weight");
    rejects([&]{stage.load(root.c_str(),"std.safetensors",cancel);},"video_latent_std");
    check(ledger->snapshot().residents==0);
    auto small=std::make_shared<Resources>(Footprint{video::H3DecoderInput::metadata_bytes});
    video::H3DecoderInput denied(small);
    rejects([&]{denied.load(root.c_str(),"weights.safetensors",cancel);},"exhausted");
    check(small->snapshot().residents==0);
    stage.load(root.c_str(),"weights.safetensors",cancel);
    check(stage.loaded());
    check(ledger->snapshot().used==Footprint({video::H3DecoderInput::weight_bytes,0}));
    rejects([&]{stage.load(root.c_str(),"weights.safetensors",cancel);},"video_already_loaded");
    std::vector<float> oversized((video::H3DecoderInput::max_tokens+1)*24);
    rejects([&]{stage.execute(oversized,root+"/oversized",cancel);},"video_input_limit");
    check(!std::filesystem::exists(root+"/oversized"));
    stage.execute(input,root+"/result",cancel,[&](std::size_t){
        rejects([&]{stage.unload();},"busy"); // Pins prevent eviction during execution.
    });
    rejects([&]{stage.execute(input,root+"/result",cancel);},"video_artifact_publish");
    rejects([&]{stage.execute({},root+"/empty",cancel);},"video_input_shape");
    rejects([&]{stage.execute(input,root+"/cancelled",cancel,[&](std::size_t n){if(n==1)cancel=true;});},"video_cancelled");
    check(!std::filesystem::exists(root+"/cancelled")); cancel=false;
    rejects([&]{stage.execute(input,root+"/hook-failed",cancel,[](std::size_t){throw std::runtime_error("hook_failure");});},"hook_failure");
    check(!std::filesystem::exists(root+"/hook-failed"));
    rejects([&]{stage.execute(input,root+"/missing-directory/output",cancel);},"video_artifact_open");
    input[0]=std::numeric_limits<float>::infinity();
    rejects([&]{stage.execute(input,root+"/nonfinite",cancel);},"video_nonfinite_input");
    check(!std::filesystem::exists(root+"/nonfinite"));
    check(ledger->snapshot().used==Footprint({video::H3DecoderInput::weight_bytes,0}));
    for(const auto& p:std::filesystem::directory_iterator(root)) check(p.path().string().find(".partial-")==std::string::npos);
    stage.unload(); stage.unload();
    check(ledger->snapshot().residents==0);
    { video::H3DecoderInput scoped(ledger); scoped.load(root.c_str(),"weights.safetensors",cancel); }
    check(ledger->snapshot().residents==0);
    std::cout<<"video lifecycle tests passed\n";
}
