#include "decoder_fixture.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <source_location>
#include <stdexcept>
namespace {
void check(bool x,std::source_location at=std::source_location::current()){if(!x)throw std::runtime_error("line_"+std::to_string(at.line()));}
template<class F>void rejects(F fn){bool caught=false;try{fn();}catch(const std::exception&){caught=true;}check(caught);}
void matches(std::span<const float>a,std::span<const float>b,const char* label){for(std::size_t j=0;j<a.size();++j)if(a[j]!=b[j]){std::cerr<<label<<" j="<<j<<" actual="<<a[j]<<" expected="<<b[j]<<'\n';check(false);}}
}
int main(){try{
    for(auto kind:{kadan::decoder::Attention::linear,kadan::decoder::Attention::full}){
        DecoderFixture fixture;auto c=fixture.config(kind),bad=c;bad.moe.hidden=32;rejects([&]{kadan::decoder::plan(bad);});
        auto p=kadan::decoder::plan(c);rejects([&]{kadan::decoder::Reference r(c,fixture.weights(),p.host_numeric_bytes-1);});
        kadan::decoder::Reference r(c,fixture.weights(),p.host_numeric_bytes);const bool l=kind==kadan::decoder::Attention::linear;
        std::array<float,16> y{},a{},u{},m{};std::vector<std::uint8_t> first(p.state_first_bytes),second(p.state_second_bytes);
        for(int replay=0;replay<2;++replay){for(std::size_t t=0;t<3;++t){
            r.step(decoder_golden::inputs[t],y);check(r.valid()&&r.tokens()==t+1);r.read_intermediates(a,u,m);
            matches(a,l?decoder_golden::linear_attention[t]:decoder_golden::full_attention[t],"attention");
            matches(u,l?decoder_golden::linear_normalized[t]:decoder_golden::full_normalized[t],"norm");
            matches(m,l?decoder_golden::linear_mixture[t]:decoder_golden::full_mixture[t],"moe");
            matches(y,l?decoder_golden::linear_output[t]:decoder_golden::full_output[t],"output");
            r.read_state(first,second);
            if(l){check(std::memcmp(first.data(),decoder_golden::linear_first[t].data(),first.size())==0);for(std::size_t j=0;j<second.size()/4;++j){float v;std::memcpy(&v,second.data()+4*j,4);check(std::abs(v-decoder_golden::linear_second[t][j])<=2e-5f);}}
            else{check(std::memcmp(first.data(),decoder_golden::full_first[t].data(),first.size())==0);check(std::memcmp(second.data(),decoder_golden::full_second[t].data(),second.size())==0);}
        }r.reset();check(r.valid()&&r.tokens()==0);rejects([&]{r.read_intermediates(a,u,m);});}
        // Exact alias preserves the original input through both residuals.
        y=decoder_golden::inputs[0];r.step(y,y);matches(y,l?decoder_golden::linear_output[0]:decoder_golden::full_output[0],"alias");
        std::atomic_bool stop=true;y.fill(42);rejects([&]{r.step(decoder_golden::inputs[1],y,&stop);});check(!r.valid()&&r.tokens()==1&&y[0]==42);
        rejects([&]{r.step(decoder_golden::inputs[1],y);});rejects([&]{r.read_intermediates(a,u,m);});r.reset();
        // Immutable failing weights: finite projections overflow in MoE activation
        // after attention has already mutated state. No partial step is published.
        DecoderFixture failing_fixture;for(std::size_t e=0;e<4;++e){failing_fixture.moe.globals[3*e]=1e20f;failing_fixture.moe.globals[3*e+1]=1e20f;}
        kadan::decoder::Reference failing(c,failing_fixture.weights(),p.host_numeric_bytes);y.fill(42);
        rejects([&]{failing.step(decoder_golden::inputs[0],y);});check(!failing.valid()&&failing.tokens()==0);for(float v:y)check(v==42);
        rejects([&]{failing.read_state(first,second);});failing.reset();check(failing.valid()&&failing.tokens()==0);failing.read_state(first,second);for(auto v:first)check(v==0);for(auto v:second)check(v==0);
        std::cout<<(l?"linear":"full")<<" device arena "<<p.device_bytes<<", CPU numeric bytes "<<p.host_numeric_bytes<<'\n';
    }
    // Same owner-bound capability implementation used by the coordinator's
    // private producers: reject foreign, stale, duplicate and post-reset steps.
    kadan::StateCursor a(1,4),b(1,4);auto step=a.begin(),foreign=b.begin();rejects([&]{a.check_step(foreign);});a.written(step,0);rejects([&]{a.written(step,0);});a.commit(step);rejects([&]{a.check_step(step);});a.reset();rejects([&]{a.check_step(step);});
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
