#pragma once
#include "kadan/linear_attention.hpp"
#include <vector>
// Deterministic small weights only, never database/checkpoint payloads.
struct LinearFixture {
    kadan::linear::Config config{2,2,4,1,1,2,8,1e-6f};
    std::vector<std::uint8_t> qkv,z,out;
    std::vector<float> norm,conv,a,b,logs,dt,gate;
    float scale=1;
    LinearFixture(){
        const auto p=kadan::linear::plan(config);qkv.resize(p.channels*config.hidden);z.resize(p.values*config.hidden);out.resize(config.hidden*p.values);
        // Q/K head 0 selects x0, head 1 selects x1. Value heads 0/1 belong
        // to key head 0; heads 2/3 belong to key head 1, with distinct values.
        for(std::size_t h=0;h<2;++h){qkv[h*2+h]=0x38;qkv[(h+2)*2+h]=0x38;}
        const std::uint8_t coefficients[]{0x38,0x40,0x44,0x48};
        for(std::size_t h=0;h<4;++h){qkv[(4+h)*2+h/2]=coefficients[h];z[h*2+h/2]=0x38;}
        out[0]=0x38;out[4+2]=0x38;
        norm.resize(2);conv.resize(p.conv_elements);for(std::size_t i=0;i<p.channels;++i){conv[i*2]=.5f;conv[i*2+1]=1;}
        a.resize(8);b.resize(8);logs.resize(4);dt.resize(4);gate.assign(1,1);
    }
    kadan::linear::Weights weights()const{
        using kadan::quantization::Encoding;const auto p=kadan::linear::plan(config);
        return {{Encoding::modelopt_fp8,p.channels,config.hidden,qkv,{},std::span(&scale,1)},
                {Encoding::modelopt_fp8,p.values,config.hidden,z,{},std::span(&scale,1)},
                {Encoding::modelopt_fp8,config.hidden,p.values,out,{},std::span(&scale,1)},norm,conv,a,b,logs,dt,gate};
    }
};

#include "linear_golden.hpp"
struct AsymmetricLinearFixture : LinearFixture {
    AsymmetricLinearFixture(){
        config={3,2,4,2,3,3,8,1e-6f};
        using namespace linear_golden;
        this->qkv.assign(linear_golden::qkv.begin(),linear_golden::qkv.end());
        this->z.assign(linear_golden::z.begin(),linear_golden::z.end());
        this->out.assign(linear_golden::out.begin(),linear_golden::out.end());
        this->norm.assign(linear_golden::norm.begin(),linear_golden::norm.end());
        this->conv.assign(linear_golden::conv.begin(),linear_golden::conv.end());
        this->a.assign(linear_golden::a.begin(),linear_golden::a.end());this->b.assign(linear_golden::b.begin(),linear_golden::b.end());
        this->logs.assign(linear_golden::logs.begin(),linear_golden::logs.end());this->dt.assign(linear_golden::dt.begin(),linear_golden::dt.end());this->gate.assign(linear_golden::gate.begin(),linear_golden::gate.end());
    }
};
