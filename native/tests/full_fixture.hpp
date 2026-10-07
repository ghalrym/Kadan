#pragma once
#include "kadan/full_attention.hpp"
#include "full_golden.hpp"
#include <vector>
struct FullFixture {
    kadan::full::Config config{3,4,2,6,4,4,1e-6f};
    std::vector<std::uint8_t> qg{full_golden::qg.begin(),full_golden::qg.end()},key{full_golden::key.begin(),full_golden::key.end()},value{full_golden::value.begin(),full_golden::value.end()},out{full_golden::out.begin(),full_golden::out.end()};
    std::vector<float> input_norm{full_golden::input_norm.begin(),full_golden::input_norm.end()},query_norm{full_golden::query_norm.begin(),full_golden::query_norm.end()},key_norm{full_golden::key_norm.begin(),full_golden::key_norm.end()};
    float scale=1;
    kadan::full::Weights weights()const{
        using kadan::quantization::Encoding;const auto p=kadan::full::plan(config);
        return {{Encoding::modelopt_fp8,2*p.queries,config.hidden,qg,{},std::span(&scale,1)},
            {Encoding::modelopt_fp8,p.kv,config.hidden,key,{},std::span(&scale,1)},
            {Encoding::modelopt_fp8,p.kv,config.hidden,value,{},std::span(&scale,1)},
            {Encoding::modelopt_fp8,config.hidden,p.queries,out,{},std::span(&scale,1)},input_norm,query_norm,key_norm,full_golden::frequencies};
    }
};
