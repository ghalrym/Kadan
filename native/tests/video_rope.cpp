#include "kadan/video.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <vector>
using namespace kadan;
void check(bool value){if(!value)throw std::runtime_error("test_failed");}
template<class F> void rejects(F f,const char* message){
    try{f();}catch(const std::exception& e){check(std::string(e.what())==message);return;}
    throw std::runtime_error("expected_rejection");
}
int main(int argc,char** argv){
    check(argc==2);const std::string root=argv[1];using Stage=video::H3QkRope;
    auto ledger=std::make_shared<Resources>(Footprint{256*1024,0});
    std::atomic_bool cancel{false};Stage stage(ledger);
    const Bytes input_bytes=(8*6144+8*3)*sizeof(float);
    auto admission=ledger->reserve(Workload::video,{input_bytes,0});
    {
    std::vector<float> qkv(8*6144),coordinates(8*3);
    for(std::size_t i=0;i<qkv.size();++i)qkv[i]=float(int(i%29)-14)/8;
    qkv[128]=-0.0f;
    for(std::size_t t=0;t<8;++t){coordinates[t*3]=float(t%2)-.5f;coordinates[t*3+1]=float((t/2)%2)-.5f;coordinates[t*3+2]=float(t/4)-.5f;}
    rejects([&]{stage.execute(qkv,coordinates,root+"/absent",cancel);},"video_not_loaded");
    cancel=true;rejects([&]{stage.load(cancel);},"video_cancelled");cancel=false;
    check(ledger->snapshot().used[0]==input_bytes);
    auto tiny=std::make_shared<Resources>(Footprint{Stage::resident_bytes-1});Stage denied(tiny);
    rejects([&]{denied.load(cancel);},"exhausted");check(tiny->snapshot().used[0]==0);
    stage.load(cancel);check(stage.loaded());
    rejects([&]{stage.load(cancel);},"video_already_loaded");
    const auto resident=input_bytes+Stage::resident_bytes;
    check(ledger->snapshot().used==Footprint({resident,0}));
    auto pressure=ledger->reserve(Workload::video,{256*1024-resident-Stage::scratch_bytes+1,0});
    rejects([&]{stage.execute(qkv,coordinates,root+"/exhausted",cancel);},"exhausted");
    check(!std::filesystem::exists(root+"/exhausted"));ledger->released(pressure);
    std::size_t observed=0;
    stage.execute(qkv,coordinates,root+"/result",cancel,[&](std::size_t heads){
        check(heads==++observed);check(ledger->snapshot().used==Footprint({resident+Stage::scratch_bytes,0}));
        rejects([&]{stage.unload();},"busy");
    });check(observed==256);
    rejects([&]{stage.execute(qkv,coordinates,root+"/result",cancel);},"video_artifact_publish");
    rejects([&]{stage.execute({},coordinates,root+"/empty",cancel);},"video_input_shape");
    rejects([&]{stage.execute({qkv.data(),1},coordinates,root+"/shape",cancel);},"video_input_shape");
    rejects([&]{stage.execute(qkv,{},root+"/coords",cancel);},"video_coordinate_shape");
    std::vector<float> oversized(9*6144);
    rejects([&]{stage.execute(oversized,coordinates,root+"/oversized",cancel);},"video_input_limit");
    for(std::size_t stop:{1,256}){
        rejects([&]{stage.execute(qkv,coordinates,root+"/cancel",cancel,[&](std::size_t n){if(n==stop)cancel=true;});},"video_cancelled");
        check(!std::filesystem::exists(root+"/cancel"));cancel=false;
    }
    cancel=true;rejects([&]{stage.execute(qkv,coordinates,root+"/pre-cancel",cancel);},"video_cancelled");cancel=false;
    rejects([&]{stage.execute(qkv,coordinates,root+"/hook",cancel,[](std::size_t){throw std::runtime_error("hook_failure");});},"hook_failure");
    rejects([&]{stage.execute(qkv,coordinates,root+"/missing/output",cancel);},"video_artifact_open");
    for(float bad:{2.0f,std::numeric_limits<float>::quiet_NaN()}){
        coordinates[0]=bad;rejects([&]{stage.execute(qkv,coordinates,root+"/range",cancel);},"video_coordinate_range");
    }coordinates[0]=-.5f;
    qkv[128]=std::numeric_limits<float>::infinity(); // V must be validated too.
    rejects([&]{stage.execute(qkv,coordinates,root+"/nonfinite",cancel);},"video_nonfinite_input");qkv[128]=-0.0f;
    qkv[0]=std::numeric_limits<float>::max();
    rejects([&]{stage.execute(qkv,coordinates,root+"/overflow",cancel);},"video_nonfinite_output");
    for(const auto& p:std::filesystem::directory_iterator(root))check(p.path().filename()=="result");
    check(ledger->snapshot().used==Footprint({resident,0}));
    stage.unload();stage.unload();check(!stage.loaded());
    check(ledger->snapshot().used[0]==input_bytes);
    {Stage scoped(ledger);scoped.load(cancel);}check(ledger->snapshot().used[0]==input_bytes);
    }
    ledger->released(admission);check(ledger->snapshot().used==Footprint({0,0}));
    std::cout<<"QK/RoPE admission, pins, cancellation, eviction and cleanup passed\n";
}
