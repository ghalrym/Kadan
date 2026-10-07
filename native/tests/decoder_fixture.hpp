#pragma once
#include "kadan/decoder.hpp"
#include "decoder_golden.hpp"
#include "moe_fixture.hpp"
struct DecoderFixture {
    MoeFixture moe;float scale=1,frequency=1;
    kadan::decoder::Config config(kadan::decoder::Attention attention)const{return {attention,{16,2,2,2,2,3,4,1e-6f},{16,2,1,4,2,4,1e-6f},moe.config};}
    kadan::decoder::Weights weights()const{
        using namespace decoder_golden;using kadan::quantization::Encoding;
        auto mat=[&](auto& bytes,std::size_t rows,std::size_t cols){return kadan::quantization::Matrix{Encoding::modelopt_fp8,rows,cols,bytes,{},std::span(&scale,1)};};
        return {{mat(lqkv,12,16),mat(lz,4,16),mat(lout,16,4),innorm,conv,la,lb,logs,dt,gate},
            {mat(fqg,16,16),mat(fk,4,16),mat(fv,4,16),mat(fout,16,8),innorm,qn,kn,std::span(&frequency,1)},moe.weights(),postnorm};
    }
};
