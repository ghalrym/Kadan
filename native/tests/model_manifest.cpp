#include "kadan/model_manifest.hpp"
#include <algorithm>
#include <array>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <source_location>
#include <string>
#include <string_view>

namespace {
void check(bool ok, std::source_location at = std::source_location::current()) { if (!ok) throw std::runtime_error("check_failed_line_" + std::to_string(at.line())); }
template<class Error=std::invalid_argument,class F> void fails(F fn,std::string_view expected) {
    bool caught=false; try { fn(); } catch (const Error& e) { check(e.what()==expected); caught=true; } check(caught);
}
}
int main(int argc,char** argv) {
    if (argc!=2) return 2;
    using namespace kadan::checkpoint;
    try {
        auto budget=std::make_shared<MemoryBudget>(16*1024*1024);
        std::unique_ptr<Projection> retained;
        {
            ModelManifest model(argv[1],budget);
            check(model.items().size()==44 && model.tensor_count()==117 && model.excluded_tensor_count()==2);
            const auto& a=model.architecture(); check(a.hidden==16 && a.layers==2 && a.experts==2);
            std::array<std::size_t,2> cap{1024*1024,1024*1024},headroom{1024,1024},slots{1,1},layers{0,1};
            PlacementOptions options{cap,headroom,slots,layers,0,32*1024*1024,1024*1024};
            const auto initial=budget->used();
            {
                auto plan=model.place(options); check(plan.item_devices().size()==44);
                check(plan.host_bytes==plan.metadata_bytes+plan.expert_host_bytes+plan.staging_bytes);
                check(plan.expert_host_bytes>0 && plan.max_transfer_bytes==1032);
                const auto d0=plan.devices()[0]; check(d0.total_bytes==d0.resident_bytes+d0.expert_cache_bytes+1024);
                cap[0]=d0.total_bytes-1;
                fails([&]{ model.place(options); },"placement_device_budget"); cap[0]=1024*1024;
                auto bad=options; bad.host_capacity=plan.host_bytes-1;
                fails([&]{ model.place(bad); },"placement_host_budget");
                bad=options; bad.staging_bytes=plan.max_transfer_bytes-1;
                fails([&]{ model.place(bad); },"placement_staging_budget");
                bad=options; bad.io_device=2; fails([&]{ model.place(bad); },"placement_shape");
                layers[1]=2; fails([&]{ model.place(options); },"placement_device"); layers[1]=1;
                headroom[0]=0; fails([&]{ model.place(options); },"placement_headroom_or_slots"); headroom[0]=1024;
                slots[0]=0; fails([&]{ model.place(options); },"placement_headroom_or_slots"); slots[0]=1;
                bad=options; bad.staging_bytes=std::numeric_limits<std::size_t>::max(); bad.host_capacity=bad.staging_bytes;
                fails([&]{ model.place(bad); },"manifest_overflow");
            }
            check(budget->used()==initial);
            {
                std::array<ExpertPolicy,2> policies{ExpertPolicy::fully_resident,ExpertPolicy::fully_resident};
                std::array<std::size_t,2> no_slots{0,0};
                auto resident=options;resident.layer_expert_policy=policies;resident.expert_slots=no_slots;
                auto plan=model.place(resident);
                // Six 16x16 NVFP4 projection slabs per layer: five aligned 256-byte
                // regions each. Payload per projection is 128+16+4+4 = 152 bytes.
                check(plan.expert_host_bytes==0 && plan.host_bytes==17*1024*1024);
                for(const auto& d:plan.devices()) check(d.expert_resident_bytes==7680 && d.expert_cache_bytes==0);
                policies[1]=ExpertPolicy::host_cached;no_slots[1]=1;
                auto mixed=model.place(resident);check(mixed.expert_host_bytes==912);
                check(mixed.devices()[0].expert_resident_bytes==7680 && mixed.devices()[0].expert_cache_bytes==0);
                check(mixed.devices()[1].expert_resident_bytes==0 && mixed.devices()[1].expert_cache_bytes==3840);
            }
            check(budget->used()==initial);
            auto find=[&](std::string_view name) { auto items=model.items(); auto it=std::find_if(items.begin(),items.end(),[&](const auto& x){return x.name==name;}); check(it!=items.end()); return std::size_t(it-items.begin()); };
            const auto fp4=find("lm_head"),fp8=find("model.language_model.layers.0.linear_attn.in_proj_qkv");
            auto loading=std::make_shared<MemoryBudget>(13);
            retained=std::make_unique<Projection>(model.load_projection_rows(fp4,0,1,13,std::make_shared<MemoryBudget>(13)));
            const auto metadata_before=budget->used();
            {
                auto retained=model.load_projection_rows(fp4,0,1,13,loading);
                check(loading->used()==13 && budget->used()==metadata_before);
                fails<std::bad_alloc>([&]{ model.load_projection_rows(fp4,0,1,13,loading); },"std::bad_alloc");
            }
            check(loading->used()==0 && budget->used()==metadata_before);
            const auto dense=find("model.language_model.embed_tokens.weight");
            for (auto id:{fp4,fp8}) {
                auto p=model.load_projection_rows(id,1,1,256);
                const auto result=kadan::quantization::matvec(p.view(),std::array<float,16>{1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1},4);
                check(result[0]==16 && model.read_input_scale(id)==1);
                fails([&]{ model.load_projection_rows(id,0,1,0); },"payload_budget");
            }
            const auto dense_info=model.primary_tensor(dense),fp4_info=model.primary_tensor(fp4),fp8_info=model.primary_tensor(fp8);
            check(dense_info.dtype==Dtype::bf16 && dense_info.rank==2 && dense_info.shape[0]==32 && dense_info.shape[1]==16);
            check(fp4_info.dtype==Dtype::u8 && fp4_info.shape[1]==8);
            check(fp8_info.dtype==Dtype::fp8 && fp8_info.shape[0]==64 && fp8_info.shape[1]==16);
            fails([&]{ model.primary_tensor(model.items().size()); },"manifest_item_index");
            std::array<std::uint8_t,2> bytes{}; model.read_dense(dense,0,bytes); check(bytes[0]==0x80 && bytes[1]==0x3f);
            fails([&]{ model.read_dense(dense,1024,bytes); },"tensor_read_bounds");
            fails([&]{ model.load_projection_rows(dense,0,1,256); },"not_projection_item");
            fails([&]{ model.read_dense(fp8,0,bytes); },"not_dense_item");
            fails([&]{ model.read_input_scale(dense); },"not_projection_item");
            // Modifying a pinned shard after construction invalidates planning/reads.
            { std::ofstream changed(std::string(argv[1])+"/a.safetensors",std::ios::binary|std::ios::app); changed.put(0); }
            fails([&]{ model.place(options); },"checkpoint_changed");
        }
        check(budget->used()==0);
        check(retained->view().columns==16 && kadan::quantization::decode_rows(retained->view(),0,1,64)[0]==1);
    } catch (const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
}
