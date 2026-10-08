#include "moe_fixture.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <source_location>
#include <stdexcept>
namespace {
void check(bool x,std::source_location at=std::source_location::current()){if(!x)throw std::runtime_error("line_"+std::to_string(at.line()));}
template<class F>void fails(F fn){bool caught=false;try{fn();}catch(const std::invalid_argument&){caught=true;}check(caught);}
template<class A,class B>void exact(const A& a,const B& b,std::source_location at=std::source_location::current()){check(a.size()==b.size()&&std::equal(a.begin(),a.end(),b.begin()),at);}
}
int main(){try{
    MoeFixture f;auto p=kadan::moe::plan(f.config);check(p.scratch_floats==171&&p.host_workspace_bytes==692&&p.device_bytes==9472);
    fails([&]{kadan::moe::Reference bad(f.config,f.weights(),691);});
    kadan::moe::Reference cpu(f.config,f.weights(),692);fails([&]{cpu.selected();});std::array<float,16> y{};
    for(int replay=0;replay<2;++replay){for(std::size_t t=0;t<5;++t){
        cpu.forward(moe_golden::inputs[t],y);check(cpu.valid());exact(y,moe_golden::result[t]);exact(cpu.result(),y);
        exact(cpu.selected(),moe_golden::selected[t]);exact(cpu.logits(),moe_golden::logits[t]);exact(cpu.top_weights(),moe_golden::top_weights[t]);
        exact(cpu.routed(),moe_golden::routed[t]);exact(cpu.shared(),moe_golden::shared[t]);
        for(std::size_t j=0;j<4;++j)check(std::abs(cpu.probabilities()[j]-moe_golden::probabilities[t][j])<=2e-7f);
    }cpu.reset();check(cpu.valid());fails([&]{cpu.result();});}
    // Exact tie: equal logits select ascending ids, weights remain exactly 1/2.
    cpu.forward(moe_golden::inputs[4],y);check(cpu.selected()[0]==0&&cpu.selected()[1]==1&&cpu.top_weights()[0]==.5f&&cpu.top_weights()[1]==.5f);
    auto alias=moe_golden::inputs[0];cpu.forward(alias,alias);exact(alias,moe_golden::result[0]);
    auto bad=moe_golden::inputs[0];bad[0]=std::numeric_limits<float>::quiet_NaN();y.fill(42);fails([&]{cpu.forward(bad,y);});check(!cpu.valid()&&y[0]==42);fails([&]{cpu.result();});fails([&]{cpu.forward(moe_golden::inputs[0],y);});cpu.reset();cpu.forward(moe_golden::inputs[0],y);exact(y,moe_golden::result[0]);
    MoeFixture explosive;for(std::size_t e=0;e<4;++e){explosive.globals[3*e]=1e20f;explosive.globals[3*e+1]=1e20f;}
    kadan::moe::Reference failure(explosive.config,explosive.weights(),692);y.fill(42);fails([&]{failure.forward(moe_golden::inputs[0],y);});check(!failure.valid()&&y[0]==42);failure.reset();check(failure.valid());
    f.router[0]=1.001f;fails([&]{kadan::moe::validate_weights(f.config,f.weights());});
    auto shape=f.config;shape.top_k=5;fails([&]{kadan::moe::plan(shape);});shape=f.config;shape.intermediate=17;fails([&]{kadan::moe::plan(shape);});
    // Large shapes are metadata-only, proving independent expert layout arithmetic.
    auto model=kadan::moe::plan({2048,256,8,512,512});check(model.device_bytes>0&&model.routed_layout.bytes==model.shared_layout.bytes);
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
