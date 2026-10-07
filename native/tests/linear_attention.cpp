#include "linear_fixture.hpp"
#include <array>
#include <bit>
#include <cmath>
#include <iostream>
#include <limits>
#include <source_location>
#include <stdexcept>
namespace {
void check(bool ok,std::source_location at=std::source_location::current()){if(!ok)throw std::runtime_error("line_"+std::to_string(at.line()));}
template<class F>void fails(F fn,const char* why){bool caught=false;try{fn();}catch(const std::invalid_argument& e){check(std::string_view(e.what())==why);caught=true;}check(caught);}
void rounding(){
    using kadan::linear::bf16_round;
    check(bf16_round(1.00390625f)==1);check(bf16_round(1.01171875f)==1.015625f);
    check(bf16_round(-1.00390625f)==-1);check(std::signbit(bf16_round(-0.0f)));
    check(bf16_round(std::bit_cast<float>(std::uint32_t{0x00008000}))==0);
    check(std::bit_cast<std::uint32_t>(bf16_round(std::bit_cast<float>(std::uint32_t{0x00018000})))==0x00020000);
    fails([&]{bf16_round(std::numeric_limits<float>::max());},"linear_nonfinite");
}
void learned_gates(){
    LinearFixture f;f.a[0]=1;f.b[0]=-1;f.logs[0]=.5f;f.dt[0]=.25f;
    kadan::linear::Reference r(f.config,f.weights(),4096);std::array<float,2> x{1,1},y{};
    // Independent scalar closed form, now beta=.26953125 and decay=.084056817.
    r.step(x,y);check(std::abs(r.recurrent()[0]-.19688396155834198f)<2e-6);
    r.step(x,y);check(std::abs(r.recurrent()[0]-.34268563985824585f)<2e-6);
}
void vector_state(){
    LinearFixture f;f.config.key_heads=1;f.config.value_heads=2;f.config.key_dim=2;f.config.value_dim=2;
    f.qkv[3]=0;f.qkv[7]=0;f.a.resize(4);f.b.resize(4);f.logs.resize(2);f.dt.resize(2);f.gate={1,.5};
    kadan::linear::Reference r(f.config,f.weights(),4096);std::array<float,2> x{1,1},y{};r.step(x,y);
    // Non-square group layout: two 2x2 states. Second key coordinate is zero;
    // distinct value columns detect accidentally transposing [key,value].
    const auto s=r.recurrent();check(s.size()==8);
    check(std::abs(s[0]-.36523401737213135f)<2e-6 && std::abs(s[1]-.8789054155349731f)<2e-6);
    check(s[2]==0 && s[3]==0 && s[6]==0 && s[7]==0);
    check(std::abs(s[4]-1.429686188697815f)<2e-6 && std::abs(s[5]-1.9609355926513672f)<2e-6);
    check(y[0]>1 && y[1]>1);for(float value:y)check(kadan::linear::bf16_round(value)==value);
}
void asymmetric_sequence(){
    AsymmetricLinearFixture f;const auto p=kadan::linear::plan(f.config);
    check(p.conv_elements==60 && p.recurrent_elements==24 && p.host_state_workspace_bytes==496);
    kadan::linear::Reference r(f.config,f.weights(),p.host_state_workspace_bytes);
    for(int replay=0;replay<2;++replay){
        for(std::size_t t=0;t<4;++t){
            std::array<float,3> output{};r.step(linear_golden::inputs[t],output);
            check(output==linear_golden::residual[t]);
            for(std::size_t j=0;j<24;++j)check(std::abs(r.recurrent()[j]-linear_golden::state[t][j])<=2e-6f);
            for(std::size_t j=0;j<60;++j)check(r.convolution()[j]==linear_golden::history[t][j]);
            for(std::size_t j=0;j<12;++j){check(r.core()[j]==linear_golden::core[t][j]);check(r.gated()[j]==linear_golden::gated[t][j]);}
        }
        r.reset();check(r.valid()&&r.tokens()==0);
        for(auto v:r.convolution())check(v==0);for(auto v:r.recurrent())check(v==0);
    }
}
void sequence(){
    LinearFixture fixture;const auto p=kadan::linear::plan(fixture.config);check(p.conv_elements==16 && p.recurrent_elements==4);
    fails([&]{kadan::linear::Reference r(fixture.config,fixture.weights(),p.host_state_workspace_bytes-1);},"linear_host_budget");
    kadan::linear::Reference r(fixture.config,fixture.weights(),p.host_state_workspace_bytes);
    const std::array<std::array<float,2>,3> input{{{1,1},{1,-1},{-1,1}}};
    const std::array<std::array<float,2>,3> output{{{1.734375f,1.734375f},{1.734375f,-.73046875f},{-.73046875f,1.734375f}}};
    // Independently evaluated scalar closed form with decay=beta=1/2:
    // S' = .5*(1-.5*k*k)*S + .5*k*v, grouping heads [0,0,1,1].
    const float state[3][4]={{.36523401737213135f,.8789054155349731f,1.429686188697815f,1.9609355926513672f},
        {.7045896053314209f,1.6494134664535522f,.4941484332084656f,.6093866229057312f},
        {.2703893184661865f,.5471286773681641f,.7368164658546448f,1.03125f}};
    for(int replay=0;replay<2;++replay){
        for(std::size_t t=0;t<3;++t){std::array<float,2> y{};r.step(input[t],y);check(y==output[t] && r.tokens()==t+1 && r.valid());
            for(std::size_t h=0;h<4;++h)check(std::abs(r.recurrent()[h]-state[t][h])<2e-6);
            // Oldest/current raw Q projection slots, before convolution activation.
            for(std::size_t h=0;h<2;++h){const float previous=t?input[t-1][h]:0;
                check(r.convolution()[h*2]==std::uint16_t(std::bit_cast<std::uint32_t>(previous)>>16));
                check(r.convolution()[h*2+1]==std::uint16_t(std::bit_cast<std::uint32_t>(input[t][h])>>16));}
        }
        r.reset();check(r.tokens()==0 && r.valid());for(auto v:r.convolution())check(v==0);for(auto v:r.recurrent())check(v==0);
    }
    std::array<float,2> bad{std::numeric_limits<float>::quiet_NaN(),1},y{99,99};
    fails([&]{r.step(bad,y);},"linear_nonfinite");check(!r.valid() && y[0]==99);
    fails([&]{r.step(input[0],y);},"state_unavailable");r.reset();r.step(input[0],y);check(y==output[0]);
    auto alias=input[1];r.step(alias,alias);check(alias==output[1]);
    LinearFixture explosive;explosive.logs[0]=128; // Finite BF16 parameter, exp exceeds FP32.
    kadan::linear::Reference partial(explosive.config,explosive.weights(),4096);
    fails([&]{partial.step(input[0],y);},"linear_nonfinite");check(!partial.valid() && partial.convolution()[1]!=0 && partial.tokens()==0);
    partial.reset();for(auto v:partial.convolution())check(v==0);for(auto v:partial.recurrent())check(v==0);
    fixture.norm[0]=1.001f;fails([&]{kadan::linear::validate_weights(fixture.config,fixture.weights());},"linear_weight_bf16");
    auto c=fixture.config;c.value_heads=3;fails([&]{kadan::linear::plan(c);},"linear_shape");
}
}
int main(){try{rounding();asymmetric_sequence();sequence();vector_state();learned_gates();}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
