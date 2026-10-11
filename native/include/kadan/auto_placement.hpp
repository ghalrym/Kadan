#pragma once
#include <algorithm>
#include <cstddef>
#include <limits>
#include <map>
#include <numeric>
#include <stdexcept>
#include <vector>
namespace kadan::model {
struct LayerBytes {std::size_t weights,state;};
struct DevicePlan {std::size_t weights=0,state=0;};
struct AutoPlacement {
    int primary=0;bool streaming=false;
    std::vector<int> devices;
    std::vector<std::size_t> weight_offsets,state_offsets;
    std::map<int,DevicePlan> allocations;
    std::size_t global_weight_offset=0,global_state_offset=0;
};
inline std::size_t checked_add(std::size_t a,std::size_t b){
    if(b>std::numeric_limits<std::size_t>::max()-a)throw std::overflow_error("placement_overflow");
    return a+b;
}
// Complete decoder layers are indivisible. State is retained separately even
// when every layer's immutable weights share a bounded streaming slot.
inline AutoPlacement auto_place(const std::vector<LayerBytes>& layers,
        std::size_t global_weights,std::size_t global_state,std::size_t transfer,
        const std::map<int,std::size_t>& budgets,std::size_t headroom){
    if(layers.empty()||layers.size()>256||!global_weights||!global_state||!transfer||budgets.empty())
        throw std::invalid_argument("placement_shape");
    for(auto l:layers)if(!l.weights||!l.state)throw std::invalid_argument("placement_layer");
    for(auto [d,b]:budgets)if(d<0||d>=64||!b)throw std::invalid_argument("placement_device");
    std::vector<int> anchors;for(auto [d,b]:budgets){(void)b;anchors.push_back(d);}
    std::stable_sort(anchors.begin(),anchors.end(),[&](int a,int b){return budgets.at(a)>budgets.at(b);});
    std::vector<std::size_t> order(layers.size());std::iota(order.begin(),order.end(),0);
    std::stable_sort(order.begin(),order.end(),[&](auto a,auto b){return checked_add(layers[a].weights,layers[a].state)>checked_add(layers[b].weights,layers[b].state);});
    // Prefer one GPU, then real layer distribution, then streaming offload.
    for(int mode=0;mode<3;++mode)for(int anchor:anchors){
        AutoPlacement p;p.primary=anchor;p.streaming=mode==2;
        p.devices.resize(layers.size());p.weight_offsets.resize(layers.size());p.state_offsets.resize(layers.size());
        p.allocations[anchor]={global_weights,checked_add(global_state,transfer)};
        auto fits=[&](int d,DevicePlan v){return checked_add(checked_add(v.weights,v.state),headroom)<=budgets.at(d);};
        if(!fits(anchor,p.allocations.at(anchor)))continue;
        bool ok=true;
        for(auto i:order){
            int chosen=-1;DevicePlan next{};std::size_t best=0;
            for(int d:anchors){
                if(mode==0&&d!=anchor)continue;
                auto it=p.allocations.find(d);DevicePlan v=it==p.allocations.end()?DevicePlan{0,transfer}:it->second;
                const auto base=d==anchor?global_weights:0;
                v.weights=p.streaming?checked_add(base,std::max(v.weights-base,layers[i].weights)):checked_add(v.weights,layers[i].weights);
                v.state=checked_add(v.state,layers[i].state);
                if(!fits(d,v))continue;
                auto left=budgets.at(d)-headroom-v.weights-v.state;
                if(chosen<0||left>best){chosen=d;next=v;best=left;}
            }
            if(chosen<0){ok=false;break;}
            auto it=p.allocations.find(chosen);DevicePlan previous=it==p.allocations.end()?DevicePlan{0,transfer}:it->second;
            p.devices[i]=chosen;p.weight_offsets[i]=p.streaming?(chosen==anchor?global_weights:0):previous.weights;
            p.state_offsets[i]=previous.state;p.allocations[chosen]=next;
        }
        if(ok)return p;
    }
    throw std::runtime_error("Qwen execution requires more memory for retained context state, global tensors and one complete decoder layer; no supported device placement fits");
}
} // namespace kadan::model
