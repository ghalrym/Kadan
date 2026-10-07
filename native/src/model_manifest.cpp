#include "kadan/model_manifest.hpp"
#include "metadata_json.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace kadan::checkpoint {
namespace {
void require(bool ok,const char* message) { if (!ok) throw std::invalid_argument(message); }
std::uint64_t add(std::uint64_t a,std::uint64_t b) { require(b<=UINT64_MAX-a,"manifest_overflow"); return a+b; }
std::uint64_t mul(std::uint64_t a,std::uint64_t b) { require(!b || a<=UINT64_MAX/b,"manifest_overflow"); return a*b; }
std::uint64_t align(std::uint64_t n) { return add(n,255)&~std::uint64_t{255}; }
std::size_t dimension(const json::Value& c,std::string_view key,std::size_t max=1048576) {
    const auto n=c.at(key).integer(); require(n>0 && n<=max,"architecture_dimension"); return n;
}
TextArchitecture architecture(const json::Value& config) {
    require(config.at("model_type").string()=="qwen3_5_moe","unsupported_model_type");
    const auto& names=config.at("architectures");
    require(names.kind==json::Value::Kind::array && names.children.size()==1 &&
        names.children[0].string()=="Qwen3_5MoeForConditionalGeneration","unsupported_architecture");
    require(!config.at("tie_word_embeddings").boolean(),"tied_embeddings_unsupported");
    const auto& c=config.at("text_config");
    require(c.at("model_type").string()=="qwen3_5_moe_text" && c.at("dtype").string()=="bfloat16" &&
        c.at("hidden_act").string()=="silu" && c.at("mamba_ssm_dtype").string()=="float32","unsupported_text_contract");
    require(!c.at("attention_bias").boolean() && c.at("attention_dropout").number()==0 &&
        c.at("attn_output_gate").boolean() && !c.at("tie_word_embeddings").boolean() && c.at("use_cache").boolean(),"unsupported_text_flags");
    TextArchitecture a{};
    a.hidden=dimension(c,"hidden_size"); a.vocab=dimension(c,"vocab_size");
    a.layers=dimension(c,"num_hidden_layers",256); a.experts=dimension(c,"num_experts",1024);
    a.experts_per_token=dimension(c,"num_experts_per_tok",a.experts);
    a.intermediate=dimension(c,"moe_intermediate_size"); a.shared_intermediate=dimension(c,"shared_expert_intermediate_size");
    a.attention_heads=dimension(c,"num_attention_heads",1024); a.kv_heads=dimension(c,"num_key_value_heads",1024);
    a.head_dim=dimension(c,"head_dim",4096); a.key_heads=dimension(c,"linear_num_key_heads",1024);
    a.value_heads=dimension(c,"linear_num_value_heads",1024); a.key_dim=dimension(c,"linear_key_head_dim",4096);
    a.value_dim=dimension(c,"linear_value_head_dim",4096); a.conv_kernel=dimension(c,"linear_conv_kernel_dim",1024);
    a.max_context=dimension(c,"max_position_embeddings",2147483647);
    a.bos_token=c.at("bos_token_id").integer(); a.eos_token=c.at("eos_token_id").integer();
    require(a.bos_token<a.vocab && a.eos_token<a.vocab,"token_id_range");
    require(a.hidden%16==0 && a.intermediate%16==0 && a.shared_intermediate%16==0 &&
        a.attention_heads%a.kv_heads==0 && a.value_heads%a.key_heads==0,"architecture_alignment");
    a.rms_epsilon=c.at("rms_norm_eps").number(); require(a.rms_epsilon>0,"rms_epsilon");
    const auto& rope=c.at("rope_parameters");
    require(rope.at("rope_type").string()=="default" && rope.at("mrope_interleaved").boolean(),"unsupported_rope");
    a.rope_theta=rope.at("rope_theta").number(); const auto fraction=rope.at("partial_rotary_factor").number();
    require(a.rope_theta>0 && fraction>0 && fraction<=1 && fraction==c.at("partial_rotary_factor").number(),"rope_parameters");
    const auto rotary=fraction*a.head_dim; require(rotary==std::floor(rotary) && rotary>=2,"rotary_dimension");
    a.rotary_dim=static_cast<std::size_t>(rotary); require(a.rotary_dim%2==0,"rotary_dimension");
    const auto& sections=rope.at("mrope_section"); require(sections.kind==json::Value::Kind::array && sections.children.size()==3,"rope_sections");
    std::uint64_t sum=0;
    for (std::size_t i=0;i<3;++i) { a.rope_sections[i]=sections.children[i].integer(); sum=add(sum,a.rope_sections[i]); }
    require(sum==a.rotary_dim/2,"rope_sections");
    const auto interval=dimension(c,"full_attention_interval",256);
    const auto& layers=c.at("layer_types"); require(layers.kind==json::Value::Kind::array && layers.children.size()==a.layers,"layer_types");
    for (std::size_t i=0;i<a.layers;++i) {
        const bool full=(i+1)%interval==0;
        require(layers.children[i].string()==(full?"full_attention":"linear_attention"),"layer_types");
        a.layer_types[i]=full?LayerKind::full_attention:LayerKind::linear_attention;
    }
    const auto& quant=config.at("quantization_config");
    require(quant.at("quant_method").string()=="modelopt" && quant.at("quant_algo").string()=="MIXED_PRECISION" &&
        quant.at("producer").at("name").string()=="modelopt","unsupported_quantization");
    return a;
}
}
struct ModelManifest::Impl {
    struct Entry { std::string_view name; std::size_t shard; bool used=false; };
    std::shared_ptr<MemoryBudget> budget;
    TextArchitecture arch{};
    std::pmr::vector<std::unique_ptr<Shard>> shards;
    std::pmr::vector<Entry> entries;
    std::pmr::vector<ModelItem> items;
    std::pmr::vector<std::string_view> quant_targets;
    std::size_t tensors=0, excluded=0;
    std::uint64_t bytes=0;
    explicit Impl(std::shared_ptr<MemoryBudget> quota) : budget(std::move(quota)),shards(budget.get()),entries(budget.get()),items(budget.get()),quant_targets(budget.get()) {}
    std::pmr::string name(std::string_view prefix,std::string_view suffix) const {
        std::pmr::string out(prefix,budget.get()); out+=suffix; return out;
    }
    Entry& entry(std::string_view name) {
        auto it=std::lower_bound(entries.begin(),entries.end(),name,[](const auto& e,auto n){return e.name<n;});
        require(it!=entries.end() && it->name==name,"manifest_missing_tensor"); return *it;
    }
    TensorInfo expect(std::string_view name,Dtype dtype,std::initializer_list<std::uint64_t> shape,std::size_t& shard) {
        auto& e=entry(name); require(!e.used,"manifest_duplicate_role");
        auto info=shards[e.shard]->tensor(name);
        require(info.dtype==dtype && info.rank==shape.size() && std::equal(shape.begin(),shape.end(),info.shape.begin()),"manifest_tensor_shape_or_dtype");
        e.used=true; shard=e.shard; return info;
    }
    void dense(std::string_view name,std::initializer_list<std::uint64_t> shape,int layer) {
        ModelItem item(budget.get()); item.name=name; item.layer=layer; item.kind=ItemKind::dense;
        auto info=expect(name,Dtype::bf16,shape,item.shard);
        item.payload_bytes=info.bytes; item.device_bytes=align(info.bytes); items.push_back(std::move(item));
    }
    void projection(std::string_view prefix,ItemKind kind,std::size_t rows,std::size_t columns,int layer,int expert,
                    const json::Value& quant,std::string_view declaration) {
        const auto& declared=quant.at(declaration);
        require(declared.at("quant_algo").string()==(kind==ItemKind::fp8?"FP8":"W4A16_NVFP4"),"projection_quantization_declaration");
        if (kind==ItemKind::nvfp4) require(declared.at("group_size").integer()==16,"projection_group_size");
        quant_targets.push_back(declared.name);
        ModelItem item(budget.get()); item.name=prefix; item.kind=kind; item.rows=rows; item.columns=columns; item.layer=layer; item.expert=expert;
        const auto weight=expect(name(prefix,".weight"),kind==ItemKind::fp8?Dtype::fp8:Dtype::u8,
                                 {rows,kind==ItemKind::fp8?columns:columns/2},item.shard);
        std::size_t other=0; std::uint64_t scale_bytes=0;
        if (kind==ItemKind::nvfp4) {
            require(columns%16==0,"projection_alignment");
            auto block=expect(name(prefix,".weight_scale"),Dtype::fp8,{rows,columns/16},other);
            require(other==item.shard,"split_projection_unsupported"); scale_bytes=block.bytes;
            auto global=expect(name(prefix,".weight_scale_2"),Dtype::fp32,{},other);
            require(other==item.shard,"split_projection_unsupported"); scale_bytes=add(scale_bytes,global.bytes);
        } else {
            // This checkpoint contract declares scalar FP8 scales. Row-scaled
            // variants need a separately reviewed manifest profile, not inference.
            auto scale=expect(name(prefix,".weight_scale"),Dtype::fp32,{},other);
            require(other==item.shard,"split_projection_unsupported"); scale_bytes=scale.bytes;
        }
        const auto calibration=expect(name(prefix,".input_scale"),Dtype::fp32,{},other);
        require(other==item.shard,"split_projection_unsupported");
        item.payload_bytes=add(add(weight.bytes,scale_bytes),calibration.bytes);
        // Matches the current projection owner's slab, excluding calibration
        // (retained as metadata/host data; activation quantization not executed).
        auto offset=align(weight.bytes);
        if (kind==ItemKind::nvfp4) offset=align(add(offset,scale_bytes-4));
        else offset=align(add(offset,scale_bytes));
        offset=align(add(offset,mul(columns,4))); offset=align(add(offset,mul(rows,4)));
        item.device_bytes=align(add(offset,4)); items.push_back(std::move(item));
    }
    void bind(const json::Value& quant) {
        const auto& a=arch;
        dense("model.language_model.embed_tokens.weight",{a.vocab,a.hidden},-1);
        dense("model.language_model.norm.weight",{a.hidden},-1);
        projection("lm_head",ItemKind::nvfp4,a.vocab,a.hidden,-1,-1,quant,"lm_head");
        for (std::size_t l=0;l<a.layers;++l) {
            const auto base=name("model.language_model.layers.",std::to_string(l));
            dense(name(base,".input_layernorm.weight"),{a.hidden},l);
            dense(name(base,".post_attention_layernorm.weight"),{a.hidden},l);
            dense(name(base,".mlp.gate.weight"),{a.experts,a.hidden},l);
            dense(name(base,".mlp.shared_expert_gate.weight"),{1,a.hidden},l);
            const auto project=[&](std::string_view suffix,std::size_t rows,std::size_t cols) {
                auto prefix=name(base,suffix); projection(prefix,ItemKind::fp8,rows,cols,l,-1,quant,prefix);
            };
            if (a.layer_types[l]==LayerKind::full_attention) {
                project(".self_attn.q_proj",mul(mul(a.attention_heads,a.head_dim),2),a.hidden);
                project(".self_attn.k_proj",mul(a.kv_heads,a.head_dim),a.hidden);
                project(".self_attn.v_proj",mul(a.kv_heads,a.head_dim),a.hidden);
                project(".self_attn.o_proj",a.hidden,mul(a.attention_heads,a.head_dim));
                dense(name(base,".self_attn.q_norm.weight"),{a.head_dim},l);
                dense(name(base,".self_attn.k_norm.weight"),{a.head_dim},l);
            } else {
                const auto key=mul(a.key_heads,a.key_dim),value=mul(a.value_heads,a.value_dim);
                const auto qkv=add(mul(key,2),value);
                project(".linear_attn.in_proj_qkv",qkv,a.hidden);
                project(".linear_attn.in_proj_z",value,a.hidden);
                project(".linear_attn.out_proj",a.hidden,value);
                dense(name(base,".linear_attn.in_proj_a.weight"),{a.value_heads,a.hidden},l);
                dense(name(base,".linear_attn.in_proj_b.weight"),{a.value_heads,a.hidden},l);
                dense(name(base,".linear_attn.A_log"),{a.value_heads},l);
                dense(name(base,".linear_attn.dt_bias"),{a.value_heads},l);
                dense(name(base,".linear_attn.norm.weight"),{a.value_dim},l);
                dense(name(base,".linear_attn.conv1d.weight"),{qkv,1,a.conv_kernel},l);
            }
            for (std::size_t e=0;e<=a.experts;++e) {
                const bool shared=e==a.experts; const auto middle=shared?a.shared_intermediate:a.intermediate;
                const auto family=name(base,shared?".mlp.shared_expert":".mlp.experts");
                auto prefix=shared?name(family,""):name(name(family,"."),std::to_string(e));
                for (auto part : {std::string_view("gate_proj"),std::string_view("up_proj"),std::string_view("down_proj")}) {
                    const auto full=name(name(prefix,"."),part); const bool down=part=="down_proj";
                    projection(full,ItemKind::nvfp4,down?a.hidden:middle,down?middle:a.hidden,l,shared?-1:static_cast<int>(e),quant,shared?std::string_view(full):std::string_view(family));
                }
            }
        }
        for (const auto& e:entries) if (!e.used) {
            require(e.name.starts_with("model.visual.") || e.name.starts_with("mtp."),"unexpected_text_tensor"); ++excluded;
        }
        require(quant.kind==json::Value::Kind::object,"quantization_declarations");
        std::sort(quant_targets.begin(),quant_targets.end());
        const auto end=std::unique(quant_targets.begin(),quant_targets.end());
        require(static_cast<std::size_t>(end-quant_targets.begin())==quant.children.size(),"unused_quantization_declaration");
        quant_targets.clear(); quant_targets.shrink_to_fit(); // Views refer to the construction-only config DOM.
    }
};
ModelManifest::ModelManifest(const char* root,std::shared_ptr<MemoryBudget> budget,ManifestLimits limits) {
    require(budget && limits.shards>0 && limits.shards<=64 && limits.tensors>0 && limits.tensors<=200000 &&
        limits.shard.header_bytes>0 && limits.shard.header_bytes<=16*1024*1024 && limits.shard.tensors>0 && limits.shard.tensors<=65536 &&
        limits.config_bytes>0 && limits.config_bytes<=1024*1024 && limits.index_bytes>0 && limits.index_bytes<=16*1024*1024,"manifest_limits");
    impl_=std::make_unique<Impl>(budget);
    const auto config=json::read(root,"config.json",limits.config_bytes,budget); impl_->arch=::kadan::checkpoint::architecture(config);
    {
        const auto index=json::read(root,"model.safetensors.index.json",limits.index_bytes,budget);
        require(index.kind==json::Value::Kind::object && index.children.size()==2,"index_fields");
        const auto& map=index.at("weight_map");
        require(map.kind==json::Value::Kind::object && !map.children.empty() && map.children.size()<=limits.tensors,"index_tensor_limit");
        std::pmr::vector<std::pmr::string> names(budget.get());
        for (const auto& entry:map.children) {
            auto shard=entry.string();
            if (std::find(names.begin(),names.end(),shard)==names.end()) {
                require(names.size()<limits.shards,"shard_limit"); names.emplace_back(shard);
            }
        }
        std::sort(names.begin(),names.end());
        for (const auto& name:names) {
            impl_->shards.push_back(std::make_unique<Shard>(root,name,budget,limits.shard));
            impl_->tensors+=impl_->shards.back()->tensor_count();
            require(impl_->tensors<=limits.tensors,"manifest_tensor_limit");
        }
        require(impl_->tensors==map.children.size(),"index_coverage");
        for (const auto& entry:map.children) {
            auto name=entry.string(); const auto it=std::lower_bound(names.begin(),names.end(),name);
            const auto shard=static_cast<std::size_t>(it-names.begin());
            const auto info=impl_->shards[shard]->tensor(entry.name);
            impl_->entries.push_back({info.name,shard,false}); impl_->bytes=add(impl_->bytes,info.bytes);
        }
        require(index.at("metadata").at("total_size").integer()==impl_->bytes,"index_total_size");
    }
    impl_->bind(config.at("quantization_config").at("quantized_layers"));
    for (const auto& shard:impl_->shards) shard->check_unchanged();
    impl_->entries.clear(); impl_->entries.shrink_to_fit();
}
ModelManifest::~ModelManifest()=default;
const TextArchitecture& ModelManifest::architecture() const { return impl_->arch; }
std::span<const ModelItem> ModelManifest::items() const { return impl_->items; }
std::size_t ModelManifest::tensor_count() const { return impl_->tensors; }
std::size_t ModelManifest::excluded_tensor_count() const { return impl_->excluded; }
std::uint64_t ModelManifest::checkpoint_bytes() const { return impl_->bytes; }
TensorInfo ModelManifest::primary_tensor(std::size_t item) const {
    require(item<impl_->items.size(),"manifest_item_index");
    const auto& entry=impl_->items[item];
    if (entry.kind==ItemKind::dense) return impl_->shards[entry.shard]->tensor(entry.name);
    return impl_->shards[entry.shard]->tensor(impl_->name(entry.name,".weight"));
}
Placement::Placement(std::shared_ptr<MemoryBudget> budget):budget_(std::move(budget)),devices_(budget_.get()),item_devices_(budget_.get()) {}
Placement ModelManifest::place(const PlacementOptions& options) const {
    const auto devices=options.device_capacity.size();
    require(devices>0 && devices<=64 && options.device_headroom.size()==devices && options.expert_slots.size()==devices &&
        options.layer_devices.size()==impl_->arch.layers && options.io_device<devices,"placement_shape");
    for (auto device:options.layer_devices) require(device<devices,"placement_device");
    for (const auto& shard:impl_->shards) shard->check_unchanged();
    Placement out(impl_->budget); out.devices_.resize(devices); out.item_devices_.reserve(impl_->items.size());
    std::array<std::uint64_t,64> expert_max{};
    std::uint64_t group=0; int last_layer=-1,last_expert=-1;
    std::size_t last_device=0;
    for (const auto& item:impl_->items) {
        const auto device=item.layer<0?options.io_device:options.layer_devices[item.layer]; out.item_devices_.push_back(device);
        out.max_transfer_bytes=std::max(out.max_transfer_bytes,item.payload_bytes);
        if (item.expert<0) out.devices_[device].resident_bytes=add(out.devices_[device].resident_bytes,item.device_bytes);
        else {
            out.expert_host_bytes=add(out.expert_host_bytes,item.payload_bytes);
            if (item.layer!=last_layer || item.expert!=last_expert) {
                expert_max[last_device]=std::max(expert_max[last_device],group); group=0;
                last_layer=item.layer; last_expert=item.expert; last_device=device;
            }
            group=add(group,item.device_bytes);
        }
    }
    expert_max[last_device]=std::max(expert_max[last_device],group);
    for (std::size_t d=0;d<devices;++d) {
        require(options.device_headroom[d]>0 && options.expert_slots[d]<=impl_->arch.experts && (!expert_max[d] || options.expert_slots[d]>0),"placement_headroom_or_slots");
        auto& target=out.devices_[d]; target.headroom_bytes=options.device_headroom[d];
        target.expert_cache_bytes=mul(expert_max[d],options.expert_slots[d]);
        target.total_bytes=add(add(target.resident_bytes,target.expert_cache_bytes),target.headroom_bytes);
        require(target.total_bytes<=options.device_capacity[d],"placement_device_budget");
    }
    require(options.staging_bytes>=out.max_transfer_bytes,"placement_staging_budget");
    out.staging_bytes=options.staging_bytes; out.metadata_bytes=impl_->budget->limit();
    out.host_bytes=add(add(out.expert_host_bytes,out.metadata_bytes),out.staging_bytes);
    require(out.host_bytes<=options.host_capacity,"placement_host_budget");
    return out;
}
Projection ModelManifest::load_projection_rows(std::size_t item,std::size_t first,std::size_t count,std::size_t budget,std::shared_ptr<MemoryBudget> payload_memory) const {
    require(item<impl_->items.size() && impl_->items[item].kind!=ItemKind::dense,"not_projection_item");
    const auto& entry=impl_->items[item];
    return impl_->shards[entry.shard]->load_modelopt_rows(entry.name,first,count,budget,std::move(payload_memory));
}
void ModelManifest::read_dense(std::size_t item,std::size_t offset,std::span<std::uint8_t> destination) const {
    require(item<impl_->items.size() && impl_->items[item].kind==ItemKind::dense,"not_dense_item");
    const auto& entry=impl_->items[item]; impl_->shards[entry.shard]->read_tensor(entry.name,offset,destination);
}
float ModelManifest::read_input_scale(std::size_t item) const {
    require(item<impl_->items.size() && impl_->items[item].kind!=ItemKind::dense,"not_projection_item");
    const auto& entry=impl_->items[item]; std::array<std::uint8_t,4> bytes{};
    impl_->shards[entry.shard]->read_tensor(impl_->name(entry.name,".input_scale"),0,bytes);
    const auto scale=quantization::fp32_le(bytes); require(std::isfinite(scale) && scale>0,"invalid_input_scale"); return scale;
}
} // namespace kadan::checkpoint
