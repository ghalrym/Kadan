#pragma once
#include "decoder_fixture.hpp"
#include "kadan/stack.hpp"
#include "stack_golden.hpp"
struct StackFixture {
    std::array<DecoderFixture,4> layer;
    std::array<std::uint8_t,128> packed=stack_golden::head_packed;
    std::array<std::uint8_t,16> scales=stack_golden::head_scales;float global=.75f;
    StackFixture(){for(std::size_t i=0;i<4;++i){layer[i].scale=.5f+.25f*i;for(auto& v:layer[i].moe.globals)v*=.5f+.125f*i;}}
    kadan::stack::Config config()const{
        kadan::stack::Config c{};for(std::size_t i=0;i<4;++i){c.layer[i]=layer[i].config(i==3?kadan::decoder::Attention::full:kadan::decoder::Attention::linear);c.layer[i].linear.capacity=c.layer[i].full.capacity=8;}
        c.vocabulary=16;c.eos=stack_golden::eos;c.epsilon=1e-6f;return c;
    }
    kadan::stack::Weights weights()const{return {{layer[0].weights(),layer[1].weights(),layer[2].weights(),layer[3].weights()},stack_golden::embedding,stack_golden::final_norm,{kadan::quantization::Encoding::modelopt_nvfp4,16,16,packed,scales,std::span(&global,1)}};}
};
