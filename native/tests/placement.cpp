#include "kadan/model_manifest.hpp"
#include <algorithm>
#include <array>
#include <iostream>
#include <limits>
#include <source_location>
#include <stdexcept>
#include <vector>
using namespace kadan::checkpoint;
namespace {
void check(bool ok,std::source_location at=std::source_location::current()) {
    if (!ok) throw std::runtime_error("check_line_"+std::to_string(at.line()));
}
template<class F> void fails(F fn,const char* message) {
    bool caught=false; try { fn(); } catch(const std::invalid_argument& e) {check(std::string_view(e.what())==message);caught=true;} check(caught);
}
}
int main() {
    try {
        constexpr auto cached=ExpertPolicy::host_cached,resident=ExpertPolicy::fully_resident;
        auto memory=std::make_shared<MemoryBudget>(65536);
        std::vector<ModelItem> items;
        auto item=[&](int layer,int expert,std::uint64_t host,std::uint64_t device) {
            items.emplace_back(std::pmr::get_default_resource()); auto& i=items.back();
            i.layer=layer;i.expert=expert;i.payload_bytes=host;i.device_bytes=device;
        };
        item(-1,-1,100,256);
        // Four logical layers, two experts each, three projection descriptors per expert.
        // Each group: 60 payload bytes, 768 device bytes; each layer: 120 / 1536.
        for(int l=0;l<4;++l) {item(l,-1,40,256);for(int e=0;e<2;++e) for(int p=0;p<3;++p) {item(l,e,20,256);items.back().rows=p;}}
        std::array<std::size_t,3> caps{100000,100000,100000},head{128,256,512},slots{1,2,0},state{512,1024,0};
        std::array<std::size_t,4> map{0,1,0,1};
        std::array<ExpertPolicy,4> policy{resident,cached,cached,resident};
        PlacementOptions o{caps,head,slots,map,0,100000,100,policy,state,200};
        auto plan=[&]{return plan_model_placement(items,4,2,memory,o);};
        const auto initial=memory->used();
        {
            auto p=plan();check(p.host_bytes==66076 && p.expert_host_bytes==240 && p.metadata_bytes==65536);
            check(p.staging_bytes==100 && p.host_headroom_bytes==200 && p.max_transfer_bytes==100);
            const auto a=p.devices()[0],b=p.devices()[1],c=p.devices()[2];
            check(a.resident_bytes==768 && a.expert_resident_bytes==1536 && a.expert_cache_bytes==768 && a.state_workspace_bytes==512 && a.total_bytes==3712);
            check(b.resident_bytes==512 && b.expert_resident_bytes==1536 && b.expert_cache_bytes==1536 && b.state_workspace_bytes==1024 && b.total_bytes==4864);
            check(c.total_bytes==512 && c.expert_resident_bytes==0 && c.expert_cache_bytes==0);
            for(std::size_t i=0;i<items.size();++i) check(p.item_devices()[i]==(items[i].layer<0?0:map[items[i].layer]));
            check(p.layer_expert_policy()[2]==cached && p.layer_expert_policy()[3]==resident);
        }
        check(memory->used()==initial);
        std::reverse(items.begin(),items.end()); // Group aggregation cannot rely on adjacency.
        {auto p=plan();check(p.devices()[0].total_bytes==3712 && p.devices()[1].total_bytes==4864);}
        for(auto& i:items) if(i.layer==0 && i.expert>=0) i.device_bytes=1024;
        {auto p=plan();check(p.devices()[0].expert_resident_bytes==6144 && p.devices()[0].expert_cache_bytes==768 && p.devices()[0].total_bytes==8320);}
        for(auto& i:items) i.device_bytes=256;
        // Interleave projection roles so a larger cached group is never adjacent.
        for(auto& i:items) if(i.layer==2 && i.expert==0) i.device_bytes=(i.rows+1)*256;
        std::stable_sort(items.begin(),items.end(),[](const auto& a,const auto& b){return a.rows<b.rows;});
        {auto p=plan();check(p.devices()[0].expert_cache_bytes==1536 && p.devices()[0].total_bytes==4480);
         check(p.devices()[1].total_bytes==4864 && p.expert_host_bytes==240);}
        for(auto& i:items) i.device_bytes=256;
        policy.fill(resident);slots={0,0,0};
        {auto p=plan();check(p.host_bytes==65836 && p.expert_host_bytes==0);
         check(p.devices()[0].expert_resident_bytes==3072 && p.devices()[0].total_bytes==4480);
         check(p.devices()[1].expert_resident_bytes==3072 && p.devices()[1].total_bytes==4864);}
        caps={4480,4864,512};o.host_capacity=65836; {auto p=plan();check(p.host_bytes==o.host_capacity);}
        --caps[0];fails(plan,"placement_device_budget");++caps[0];
        --o.host_capacity;fails(plan,"placement_host_budget");++o.host_capacity;
        --o.staging_bytes;fails(plan,"placement_staging_budget");++o.staging_bytes;
        slots[0]=1;fails(plan,"placement_headroom_or_slots");slots[0]=0;
        policy.fill(cached);slots={1,2,0};caps.fill(100000);o.host_capacity=100000;
        {auto p=plan();check(p.expert_host_bytes==480 && p.host_bytes==66316);
         check(p.devices()[0].expert_resident_bytes==0 && p.devices()[0].total_bytes==2176);
         check(p.devices()[1].total_bytes==3328);}
        o.layer_expert_policy={}; {auto p=plan();check(p.host_bytes==66316);}
        o.layer_expert_policy=policy;
        slots[0]=0;fails(plan,"placement_headroom_or_slots");slots[0]=1;
        policy[0]=static_cast<ExpertPolicy>(99);fails(plan,"placement_policy");policy[0]=cached;
        o.layer_expert_policy=std::span(policy).first(3);fails(plan,"placement_shape");o.layer_expert_policy=policy;
        o.device_state_workspace=std::span(state).first(2);fails(plan,"placement_shape");o.device_state_workspace=state;
        map[0]=3;fails(plan,"placement_device");map[0]=0;
        const auto saved_layer=items[0].layer;
        items[0].layer=4;fails(plan,"placement_item");items[0].layer=saved_layer;
        items[0].device_bytes=std::numeric_limits<std::uint64_t>::max();fails(plan,"manifest_overflow");items[0].device_bytes=256;
        o.host_headroom=std::numeric_limits<std::size_t>::max();fails(plan,"manifest_overflow");o.host_headroom=200;
        state[0]=std::numeric_limits<std::size_t>::max();fails(plan,"manifest_overflow");state[0]=512;
        check(memory->used()==initial);
        auto tiny=std::make_shared<MemoryBudget>(1);bool exhausted=false;
        try {plan_model_placement(items,4,2,tiny,o);} catch(const std::bad_alloc&) {exhausted=true;}
        check(exhausted && tiny->used()==0);
    } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
