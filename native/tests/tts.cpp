#include "kadan/tts.hpp"
#include <iostream>
#include <limits>
using namespace kadan;
void check(bool ok){if(!ok)throw std::runtime_error("test_failed");}
template<class F>void rejects(F f,const char* error){try{f();}catch(const std::exception& e){check(std::string(e.what())==error);return;}throw std::runtime_error("expected_failure");}
int main(int argc,char** argv){
    check(argc==2);const std::string root=argv[1];std::atomic_bool cancel{false};
    auto ledger=std::make_shared<Resources>(Footprint{64*1024*1024,0});
    tts::TextProjection heads(ledger);
    std::array<float,2048> marker{};
    rejects([&]{heads.execute(marker,cancel);},"tts_not_loaded");
    cancel=true;rejects([&]{heads.load(root.c_str(),"weights.safetensors",cancel);},"tts_cancelled");cancel=false;
    for(auto [file,error]:{std::pair{"wrong.safetensors","tts_tensor_layout"},std::pair{"nan.safetensors","tts_nonfinite_weight"}}){
        rejects([&]{heads.load(root.c_str(),file,cancel);},error);check(ledger->snapshot().residents==0);
    }
    auto small=std::make_shared<Resources>(Footprint{tts::TextProjection::metadata_bytes});
    tts::TextProjection denied(small);
    rejects([&]{denied.load(root.c_str(),"weights.safetensors",cancel);},"exhausted");check(small->snapshot().residents==0);
    heads.load(root.c_str(),"weights.safetensors",cancel);
    check(ledger->snapshot().used==Footprint({tts::TextProjection::weight_bytes,0}));
    rejects([&]{heads.load(root.c_str(),"weights.safetensors",cancel);},"tts_already_loaded");
    const auto remaining=ledger->snapshot().capacity[0]-ledger->snapshot().used[0];
    auto pressure=ledger->reserve(Workload::tts,{remaining,0});
    rejects([&]{heads.execute(marker,cancel);},"exhausted");
    ledger->released(pressure);
    const auto first=heads.execute(marker,cancel,[&]{rejects([&]{heads.unload();},"busy");});
    check(first==heads.execute(marker,cancel));
    rejects([&]{heads.execute({},cancel);},"tts_input_shape");
    rejects([&]{heads.execute(marker,cancel,[&]{cancel=true;});},"tts_cancelled");cancel=false;
    rejects([&]{heads.execute(marker,cancel,[]{throw std::runtime_error("hook_failure");});},"hook_failure");
    marker[0]=std::numeric_limits<float>::infinity();
    rejects([&]{heads.execute(marker,cancel);},"tts_nonfinite_input");
    check(ledger->snapshot().used==Footprint({tts::TextProjection::weight_bytes,0}));
    heads.unload();heads.unload();check(ledger->snapshot().residents==0);
    {tts::TextProjection scoped(ledger);scoped.load(root.c_str(),"weights.safetensors",cancel);}
    check(ledger->snapshot().residents==0);std::cout<<"tts lifecycle tests passed\n";
}
