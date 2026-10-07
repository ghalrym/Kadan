#include "kadan/layer_math.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <source_location>
#include <stdexcept>
#include <vector>
namespace {
void check(bool ok,std::source_location at=std::source_location::current()) {if(!ok)throw std::runtime_error("line_"+std::to_string(at.line()));}
void near(float a,double b,double tolerance=2e-6) {check(std::abs(double(a)-b)<=tolerance*(1+std::abs(b)));}
template<class F> void fails(F fn,const char* message) {bool caught=false;try{fn();}catch(const std::invalid_argument& e){check(std::string_view(e.what())==message);caught=true;}check(caught);}
void normalization() {
    using namespace kadan::math;
    std::array<float,4> x{1,1,0,0},out{};std::array<float,2> weight{2,-1};Norm s{2,2,3,NormScale::direct};
    rms_norm(s,x,weight,out);near(out[0],1);near(out[1],-.5);check(out[2]==0 && out[3]==0);
    s.scale=NormScale::one_plus;rms_norm(s,x,weight,out);near(out[0],1.5);check(out[1]==0);
    rms_norm(s,x,weight,x);near(x[0],1.5);check(x[1]==0);
    std::vector<float> huge(2048,std::numeric_limits<float>::max()),ones(2048,1),answer(2048);
    rms_norm({1,2048,1e-6f,NormScale::direct},huge,ones,answer);for(float v:answer)near(v,1);
    s.epsilon=0;fails([&]{rms_norm(s,x,weight,out);},"math_epsilon");s.epsilon=3;
    s.scale=static_cast<NormScale>(4);fails([&]{rms_norm(s,x,weight,out);},"math_scale");s.scale=NormScale::direct;
    fails([&]{rms_norm(s,x,std::span<const float>(x).first(2),x);},"math_overlap");
    weight[1]=std::numeric_limits<float>::infinity();fails([&]{rms_norm(s,x,weight,out);},"math_nonfinite");
    fails([&]{rms_norm({1025,2,1,NormScale::direct},x,weight,out);},"math_shape");
}
void gates() {
    using namespace kadan::math;
    std::array<float,4> x{2,2,2,2},z{0,float(std::log(3.0)),1000,-1000},out{};
    gate(Gate::sigmoid,x,z,out);near(out[0],1);near(out[1],1.5);near(out[2],2);near(out[3],0);
    gate(Gate::silu,x,z,out);near(out[0],0);near(out[1],1.5*std::log(3.0));near(out[2],2000);near(out[3],0);
    gate(Gate::sigmoid,x,z,z);near(z[0],1);near(z[1],1.5);
    std::array<float,5> overlap{};
    fails([&]{gate(Gate::sigmoid,std::span(overlap).first(4),x,std::span(overlap).subspan(1));},"math_overlap");
    fails([&]{gate(static_cast<Gate>(9),x,z,out);},"math_gate");
    z[0]=std::numeric_limits<float>::quiet_NaN();fails([&]{gate(Gate::sigmoid,x,z,out);},"math_nonfinite");
    z.fill(std::numeric_limits<float>::max());x.fill(std::numeric_limits<float>::max());
    fails([&]{gate(Gate::silu,x,z,out);},"math_result_range");
    // Norm precedes SiLU, and direct weights differ from decoder's 1+weight.
    std::array<float,2> v{1,1},w{2,2},g{0,float(std::log(3.0))},n{},y{};
    rms_norm({1,2,3,NormScale::direct},v,w,n);gate(Gate::silu,n,g,y);
    near(y[0],0);near(y[1],.75*std::log(3.0));
}
void rotary() {
    using namespace kadan::math;
    std::array<float,2> f{};inverse_frequencies(10000,f);check(f[0]==1);near(f[1],.01);
    std::array<float,12> x{1,2,3,4,5,6,1,2,3,4,7,8},out{};Rotary s{2,6,4,262144};
    rope(s,0,f,x,out);check(x==out);
    rope(s,1,f,x,out);
    // Independent split-half golden pairs (1,3) and (2,4), not adjacent pairs.
    for(std::size_t b:{0u,6u}) {near(out[b],-1.9841106485555495);near(out[b+2],2.4623779024123156);
        near(out[b+1],1.959900667496664);near(out[b+3],4.019799668334994);}
    check(out[4]==5 && out[5]==6 && out[10]==7 && out[11]==8);
    rope(s,1,f,x,x);for(std::size_t i=0;i<x.size();++i)check(x[i]==out[i]);
    rope(s,262143,f,x,out);for(float v:out)check(std::isfinite(v));
    fails([&]{rope(s,262144,f,x,out);},"math_position");
    s.rotary_dim=3;fails([&]{rope(s,0,f,x,out);},"math_rotary_shape");s.rotary_dim=4;
    f[1]=0;fails([&]{rope(s,0,f,x,out);},"math_frequency_range");
    fails([&]{inverse_frequencies(1,f);},"math_frequency_shape");
    std::array<float,32> actual{};inverse_frequencies(10000000,actual);
    check(actual[0]==1 && actual.back()>0 && actual.back()<1e-6f);
    for(std::size_t i=1;i<actual.size();++i)check(actual[i]<actual[i-1]);
}
}
int main(){try{normalization();gates();rotary();}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
