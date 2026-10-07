#pragma once
#include "kadan/moe.hpp"
#include "moe_golden.hpp"
#include <array>
#include <vector>
struct MoeFixture {
    kadan::moe::Config config{16,4,2,16,32};
    std::array<std::vector<std::uint8_t>,15> packed,scales;
    std::array<float,15> globals{};std::array<float,64> router{};std::array<float,16> shared_gate{};
    std::array<kadan::moe::Expert,4> experts{};kadan::moe::Expert shared{};
    MoeFixture(){
        using kadan::quantization::Encoding;
        for(std::size_t e=0;e<=4;++e){std::array<kadan::quantization::Matrix,3> projections{};const auto middle=e==4?32:16;
            for(std::size_t role=0;role<3;++role){const std::size_t rows=role==2?16:middle,columns=role==2?middle:16,n=3*e+role;
                packed[n].resize(rows*columns/2);scales[n].resize(rows*columns/16);globals[n]=float(n%4+2)/4;
                for(std::size_t r=0;r<rows;++r){for(std::size_t j=0;j<columns;++j){const auto code=(r+2*j+3*e+role)%5+((r+j+e+role)%3==0?8:0);packed[n][r*columns/2+j/2]|=std::uint8_t(code<<(4*(j%2)));}
                    for(std::size_t block=0;block<columns/16;++block)scales[n][r*columns/16+block]=(r+block+e+role)%2?0x28:0x20;}
                projections[role]={Encoding::modelopt_nvfp4,rows,columns,packed[n],scales[n],std::span(&globals[n],1)};
            }
            kadan::moe::Expert view{projections[0],projections[1],projections[2]};if(e==4)shared=view;else experts[e]=view;
        }
        for(std::size_t e=0;e<4;++e){router[e*16+e]=1;router[e*16+4]=.125f*(float(e)-1);}
        for(std::size_t j=0;j<16;++j)shared_gate[j]=(j%2?-1:1)*.03125f*float(1+j%3);
    }
    MoeFixture(const MoeFixture&)=delete;MoeFixture& operator=(const MoeFixture&)=delete;
    kadan::moe::Weights weights()const{return {router,shared_gate,experts,shared};}
};
