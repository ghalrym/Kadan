#pragma once
#include "kadan/model.hpp"
#include <bit>
#include <deque>
#include <map>
#include <string>
#include <cstring>
struct ModelImage: kadan::model::Sink {
    const kadan::model::Layout& layout;std::vector<std::uint8_t> arena;std::vector<float> scales;std::size_t writes=0,max_write=0;
    ModelImage(const kadan::model::Layout& l):layout(l),arena(l.device_bytes()),scales(l.bindings().size()){
        if(l.device_bytes()>2*1024*1024)throw std::runtime_error("fixture_only_image");
    }
    void write(std::size_t at,std::span<const std::uint8_t> data)override{if(at>arena.size()||data.size()>arena.size()-at)throw std::runtime_error("image_bounds");std::copy(data.begin(),data.end(),arena.begin()+at);++writes;max_write=std::max(max_write,data.size());}
    void multiplier(const kadan::model::Binding& b,float v)override{scales[b.item]=v;}
    const kadan::model::Binding& binding(const std::string& name)const{auto items=layout.manifest().items();for(auto&b:layout.bindings())if(items[b.item].name==std::string_view(name))return b;throw std::runtime_error("test_binding_missing");}
    std::span<const std::uint8_t> raw(std::size_t at,std::size_t n)const{return std::span(arena).subspan(at,n);}
    kadan::quantization::Matrix matrix(const std::string& name){auto&b=binding(name);const auto&i=layout.manifest().items()[b.item];bool fp8=i.kind==kadan::checkpoint::ItemKind::fp8;
        if(fp8)std::memcpy(&scales[b.item],arena.data()+b.scales,4);
        return {fp8?kadan::quantization::Encoding::modelopt_fp8:kadan::quantization::Encoding::modelopt_nvfp4,i.rows,i.columns,raw(b.weights,i.rows*i.columns/(fp8?1:2)),fp8?std::span<const std::uint8_t>{}:raw(b.scales,i.rows*i.columns/16),std::span(&scales[b.item],1)};
    }
    std::vector<float> dense(const std::string& name){auto& b=binding(name);auto n=layout.manifest().items()[b.item].payload_bytes;std::vector<float> v(n/2);for(std::size_t j=0;j<v.size();++j){auto at=b.weights+2*j;v[j]=std::bit_cast<float>((std::uint32_t(arena[at])|(std::uint32_t(arena[at+1])<<8))<<16);}return v;}
};
struct ModelCpuLayer {
    std::deque<std::vector<float>> dense;std::vector<kadan::moe::Expert> experts;std::unique_ptr<kadan::decoder::Reference> reference;
    ModelCpuLayer(ModelImage& image,std::size_t i){auto base=std::string("model.language_model.layers.")+std::to_string(i);const auto&l=image.layout.layers()[i];
        auto d=[&](const char*s)->std::span<const float>{dense.push_back(image.dense(base+s));return dense.back();};auto m=[&](const char*s){return image.matrix(base+s);};kadan::decoder::Weights w;
        if(l.config.attention==kadan::decoder::Attention::linear){w.linear={m(".linear_attn.in_proj_qkv"),m(".linear_attn.in_proj_z"),m(".linear_attn.out_proj"),d(".input_layernorm.weight"),d(".linear_attn.conv1d.weight"),d(".linear_attn.in_proj_a.weight"),d(".linear_attn.in_proj_b.weight"),d(".linear_attn.A_log"),d(".linear_attn.dt_bias"),d(".linear_attn.norm.weight")};}
        else{dense.emplace_back(l.config.full.rotary_dim/2);auto&f=dense.back();std::memcpy(f.data(),image.arena.data()+l.offset+l.plan.frequencies,f.size()*4);w.full={m(".self_attn.q_proj"),m(".self_attn.k_proj"),m(".self_attn.v_proj"),m(".self_attn.o_proj"),d(".input_layernorm.weight"),d(".self_attn.q_norm.weight"),d(".self_attn.k_norm.weight"),f};}
        for(std::size_t e=0;e<=l.config.moe.experts;++e){auto prefix=base+(e==l.config.moe.experts?".mlp.shared_expert":".mlp.experts."+std::to_string(e));kadan::moe::Expert expert{image.matrix(prefix+".gate_proj"),image.matrix(prefix+".up_proj"),image.matrix(prefix+".down_proj")};if(e==l.config.moe.experts)w.moe.shared=expert;else experts.push_back(expert);}
        w.moe.experts=experts;w.moe.router=d(".mlp.gate.weight");w.moe.shared_gate=d(".mlp.shared_expert_gate.weight");w.post_norm=d(".post_attention_layernorm.weight");reference=std::make_unique<kadan::decoder::Reference>(l.config,w,l.plan.host_numeric_bytes);
    }
};
