#include "kadan/auto_placement.hpp"
#include <cassert>
#include <iostream>
using namespace kadan::model;
int main(){
    const std::vector<LayerBytes> layers{{60,10},{60,10},{60,10}};
    auto one=auto_place(layers,20,10,4,{{0,260},{1,260}},8);
    assert(!one.streaming&&one.allocations.size()==1);
    auto multi=auto_place(layers,20,10,4,{{0,180},{1,180}},8);
    assert(!multi.streaming&&multi.allocations.size()==2);
    auto offload=auto_place(layers,20,10,4,{{0,140}},8);
    assert(offload.streaming&&offload.allocations.at(0).weights==80);
    assert(offload.state_offsets[0]!=offload.state_offsets[1]);
    assert(offload.weight_offsets[0]==offload.weight_offsets[1]);
    bool rejected=false;try{auto_place(layers,20,10,4,{{0,100}},8);}catch(const std::runtime_error&){rejected=true;}assert(rejected);
    for(auto p:{one,multi,offload})for(auto [d,a]:p.allocations){(void)d;assert(a.weights+a.state+8<=(p.streaming?140:p.allocations.size()==1?260:180));}
    auto sparse=auto_place(layers,20,10,4,{{3,180},{7,180}},8);assert(sparse.allocations.contains(3)&&sparse.allocations.contains(7));
    auto again=auto_place(layers,20,10,4,{{0,180},{1,180}},8);assert(again.devices==multi.devices);
    std::cout<<"Automatic Qwen placement unit checks passed\n";
}
