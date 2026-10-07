#include "full_fixture.hpp"
#include "kadan/linear_attention.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <source_location>
#include <stdexcept>
namespace {
void check(bool ok,std::source_location at=std::source_location::current()){if(!ok)throw std::runtime_error("line_"+std::to_string(at.line()));}
template<class F>void fails(F f){bool caught=false;try{f();}catch(const std::invalid_argument&){caught=true;}check(caught);}
template<class A,class B>void exact(const A& a,const B& b,std::source_location at=std::source_location::current()){check(a.size()==b.size()&&std::equal(a.begin(),a.end(),b.begin()),at);}
}
int main(){try{
    FullFixture f;auto p=kadan::full::plan(f.config);check(p.history_elements==48 && p.scratch_floats==190 && p.host_state_workspace_bytes==952);
    fails([&]{kadan::full::Reference bad(f.config,f.weights(),951);});
    kadan::full::Reference cpu(f.config,f.weights(),952);fails([&]{cpu.core();});
    std::array<float,3> y{};
    for(int replay=0;replay<2;++replay){
        for(std::size_t t=0;t<4;++t){
            cpu.step(full_golden::inputs[t],y);check(cpu.valid()&&cpu.tokens()==t+1);
            exact(y,full_golden::residual[t]);exact(cpu.keys(),full_golden::keys[t]);exact(cpu.values(),full_golden::values[t]);
            exact(cpu.probabilities(),full_golden::probabilities[t]);exact(cpu.core(),full_golden::core[t]);exact(cpu.gated(),full_golden::gated[t]);
        }
        // Capacity rejection occurs before begin: previously committed state survives.
        fails([&]{cpu.step(full_golden::inputs[0],y);});check(cpu.valid()&&cpu.tokens()==4);exact(cpu.keys(),full_golden::keys[3]);
        cpu.reset();check(cpu.valid()&&cpu.tokens()==0);for(auto x:cpu.keys())check(x==0);for(auto x:cpu.values())check(x==0);fails([&]{cpu.probabilities();});
    }
    auto alias=full_golden::inputs[0];cpu.step(alias,alias);exact(alias,full_golden::residual[0]);
    std::array<float,3> bad{std::numeric_limits<float>::quiet_NaN(),1,2};fails([&]{cpu.step(bad,y);});check(!cpu.valid()&&cpu.tokens()==1);
    fails([&]{cpu.step(full_golden::inputs[1],y);});cpu.reset();cpu.step(full_golden::inputs[0],y);exact(y,full_golden::residual[0]);
    // Finite BF16 Q/K scales overflow attention logits AFTER current KV append.
    FullFixture explosive;explosive.query_norm[0]=explosive.key_norm[0]=kadan::linear::bf16_round(1e20f);
    kadan::full::Reference failed(explosive.config,explosive.weights(),952);
    fails([&]{failed.step(full_golden::inputs[0],y);});check(!failed.valid()&&failed.tokens()==0);
    check(std::any_of(failed.keys().begin(),failed.keys().end(),[](auto x){return x!=0;}));
    failed.reset();check(failed.valid());for(auto x:failed.keys())check(x==0);for(auto x:failed.values())check(x==0);
    f.query_norm[0]=1.001f;fails([&]{kadan::full::validate_weights(f.config,f.weights());});
    auto shape=f.config;shape.heads=3;fails([&]{kadan::full::plan(shape);});shape=f.config;shape.rotary_dim=3;fails([&]{kadan::full::plan(shape);});
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
