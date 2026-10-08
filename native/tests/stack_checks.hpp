#pragma once
#include "stack_fixture.hpp"
#include <cmath>
#include <cstring>
#include <stdexcept>
namespace stack_test {
inline void check(bool ok,const char* error){if(!ok)throw std::runtime_error(error);}
inline std::span<const float> output(std::size_t layer,std::size_t t){switch(layer){case 0:return stack_golden::layer0_output[t];case 1:return stack_golden::layer1_output[t];case 2:return stack_golden::layer2_output[t];default:return stack_golden::layer3_output[t];}}
inline std::span<const std::uint16_t> first(std::size_t layer,std::size_t t){switch(layer){case 0:return stack_golden::layer0_first[t];case 1:return stack_golden::layer1_first[t];case 2:return stack_golden::layer2_first[t];default:return stack_golden::layer3_first[t];}}
inline std::span<const float> recurrent(std::size_t layer,std::size_t t){switch(layer){case 0:return stack_golden::layer0_second[t];case 1:return stack_golden::layer1_second[t];default:return stack_golden::layer2_second[t];}}
inline void exact(std::span<const float>a,std::span<const float>b,const char* label){check(a.size()==b.size(),"golden_shape");for(std::size_t j=0;j<a.size();++j)if(a[j]!=b[j])throw std::runtime_error(std::string(label)+" index="+std::to_string(j)+" actual="+std::to_string(a[j])+" expected="+std::to_string(b[j]));}
template<class Model>void verify(Model& model,const kadan::stack::Plan&p,std::size_t t){
    std::array<float,16> row{},normalized{},logits{};
    for(std::size_t i=0;i<4;++i){model.read_layer(i,row);exact(row,output(i,t),"layer_golden");std::vector<std::uint8_t>a(p.layer[i].state_first_bytes),b(p.layer[i].state_second_bytes);model.read_state(i,a,b);check(a.size()==first(i,t).size_bytes()&&std::memcmp(a.data(),first(i,t).data(),a.size())==0,"first_state_golden");
        if(i<3){auto expected=recurrent(i,t);check(b.size()==expected.size_bytes(),"recurrent_shape");for(std::size_t j=0;j<expected.size();++j){float v;std::memcpy(&v,b.data()+j*4,4);check(std::abs(v-expected[j])<=2e-5f,"recurrent_golden");}}
        else check(b.size()==stack_golden::layer3_second[t].size()*2&&std::memcmp(b.data(),stack_golden::layer3_second[t].data(),b.size())==0,"kv_value_golden");}
    model.read_output(normalized,logits);exact(normalized,stack_golden::normalized[t],"norm_golden");exact(logits,stack_golden::logits[t],"logits_golden");
}
template<class Model>void zero(Model& model,const kadan::stack::Plan&p){check(model.valid()&&!model.finished()&&model.tokens()==0,"reset_progress");for(std::size_t i=0;i<4;++i){std::vector<std::uint8_t>a(p.layer[i].state_first_bytes),b(p.layer[i].state_second_bytes);model.read_state(i,a,b);for(auto v:a)check(v==0,"reset_first");for(auto v:b)check(v==0,"reset_second");}}
}
