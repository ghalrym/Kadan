#include "kadan/decision.hpp"
#include <iostream>
#include <limits>
using namespace kadan;
void check(bool ok){if(!ok)throw std::runtime_error("test_failed");}
template<class F>void rejects(F f,const char* error){try{f();}catch(const std::exception& e){check(std::string(e.what())==error);return;}throw std::runtime_error("expected_failure");}
int main(int argc,char** argv){
    check(argc==2);const std::string root=argv[1];std::atomic_bool cancel{false};
    auto ledger=std::make_shared<Resources>(Footprint{32*1024*1024,0});
    decision::TerminalHeads heads(ledger);
    std::array<float,1024> marker{};std::array<float,1028> action{};
    rejects([&]{heads.execute(marker,action,cancel);},"decision_not_loaded");
    cancel=true;rejects([&]{heads.load(root.c_str(),"weights.safetensors",cancel);},"decision_cancelled");cancel=false;
    for(auto [file,error]:{std::pair{"wrong.safetensors","decision_tensor_layout"},std::pair{"nan.safetensors","decision_nonfinite_weight"}}){
        rejects([&]{heads.load(root.c_str(),file,cancel);},error);check(ledger->snapshot().residents==0);
    }
    auto small=std::make_shared<Resources>(Footprint{decision::TerminalHeads::metadata_bytes});
    decision::TerminalHeads denied(small);
    rejects([&]{denied.load(root.c_str(),"weights.safetensors",cancel);},"exhausted");check(small->snapshot().residents==0);
    heads.load(root.c_str(),"weights.safetensors",cancel);
    check(ledger->snapshot().used==Footprint({decision::TerminalHeads::weight_bytes,0}));
    rejects([&]{heads.load(root.c_str(),"weights.safetensors",cancel);},"decision_already_loaded");
    const auto remaining=ledger->snapshot().capacity[0]-ledger->snapshot().used[0];
    auto pressure=ledger->reserve(Workload::decision,{remaining,0});
    rejects([&]{heads.execute(marker,action,cancel);},"exhausted");
    ledger->released(pressure);
    const auto first=heads.execute(marker,action,cancel,[&]{rejects([&]{heads.unload();},"busy");});
    check(first==heads.execute(marker,action,cancel));
    rejects([&]{heads.execute({},action,cancel);},"decision_input_shape");
    rejects([&]{heads.execute(marker,action,cancel,[&]{cancel=true;});},"decision_cancelled");cancel=false;
    rejects([&]{heads.execute(marker,action,cancel,[]{throw std::runtime_error("hook_failure");});},"hook_failure");
    marker[0]=std::numeric_limits<float>::infinity();
    rejects([&]{heads.execute(marker,action,cancel);},"decision_nonfinite_input");
    check(ledger->snapshot().used==Footprint({decision::TerminalHeads::weight_bytes,0}));
    heads.unload();heads.unload();check(ledger->snapshot().residents==0);
    {decision::TerminalHeads scoped(ledger);scoped.load(root.c_str(),"weights.safetensors",cancel);}
    check(ledger->snapshot().residents==0);std::cout<<"decision lifecycle tests passed\n";
}
