#include "kadan/model.hpp"
#include "metadata_json.hpp"
#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>
#include <numeric>
#include <string>
namespace kadan::model {
namespace {
void require(bool v,const char* e){if(!v)throw std::invalid_argument(e);}
std::size_t add(std::size_t a,std::size_t b){require(b<=SIZE_MAX-a,"model_layout_overflow");return a+b;}
void cancel(const std::atomic_bool* f){require(!f||!f->load(std::memory_order_relaxed),"model_cancelled");}
std::span<const std::uint8_t> bytes(std::span<const float> v){return {reinterpret_cast<const std::uint8_t*>(v.data()),v.size_bytes()};}
}
bool Generation::is_eos(unsigned id)const{require(count<=eos.size(),"model_eos_ids");return std::find(eos.begin(),eos.begin()+count,id)!=eos.begin()+count;}
Generation read_generation(const char* root,std::size_t vocab,const std::shared_ptr<checkpoint::MemoryBudget>& budget){
    require(bool(budget),"model_metadata_budget");const auto json=checkpoint::json::read(root,"generation_config.json",65536,budget);Generation g;
    auto append=[&](const checkpoint::json::Value& v){auto n=v.integer();require(n<vocab&&g.count<g.eos.size()&&!g.is_eos(unsigned(n)),"model_eos_ids");g.eos[g.count++]=unsigned(n);};
    const auto& eos=json.at("eos_token_id");if(eos.kind==checkpoint::json::Value::Kind::array)for(const auto& v:eos.children)append(v);else append(eos);
    require(g.count>0,"model_eos_ids");return g;
}
Layout::Layout(const checkpoint::ModelManifest& manifest,std::size_t capacity,Generation generation,std::shared_ptr<checkpoint::MemoryBudget> budget)
    :manifest_(manifest),budget_(std::move(budget)),layers_(budget_.get()),bindings_(budget_.get()),capacity_(capacity),generation_(generation){
    require(bool(budget_),"model_metadata_budget");const auto&a=manifest.architecture();
    require(capacity>0&&capacity<=a.max_context&&a.vocab<=262144,"model_capacity_or_vocabulary");require(generation.count>0&&generation.count<=generation.eos.size(),"model_eos_ids");
    for(std::size_t i=0;i<generation.count;++i){require(generation.eos[i]<a.vocab,"model_eos_ids");for(std::size_t j=0;j<i;++j)require(generation.eos[j]!=generation.eos[i],"model_eos_ids");}
    auto region=[&](std::size_t n){auto at=end_;end_=add(end_,add(n,255)&~std::size_t(255));return at;};
    layers_.reserve(a.layers);
    for(std::size_t i=0;i<a.layers;++i){decoder::Config c{};c.attention=a.layer_types[i]==checkpoint::LayerKind::linear_attention?decoder::Attention::linear:decoder::Attention::full;
        c.linear={a.hidden,a.key_heads,a.value_heads,a.key_dim,a.value_dim,a.conv_kernel,capacity,float(a.rms_epsilon),true};
        c.full={a.hidden,a.attention_heads,a.kv_heads,a.head_dim,a.rotary_dim,capacity,float(a.rms_epsilon),true};
        c.moe={a.hidden,a.experts,a.experts_per_token,a.intermediate,a.shared_intermediate,true};auto p=decoder::plan(c);layers_.push_back({c,p,region(p.device_bytes)});}
    embedding_=region(a.vocab*a.hidden*2);final_norm_=region(a.hidden*2);head_weights_=region(a.vocab*a.hidden/2);head_scales_=region(a.vocab*a.hidden/16);
    embedded_=region(a.hidden*4);normalized_=region(a.hidden*4);logits_=region(a.vocab*4);selected_=region(4);status_=region(4);
    const auto items=manifest.items();for(const auto& item:items)if(item.kind!=checkpoint::ItemKind::dense)minimum_staging_=std::max(minimum_staging_,add(item.kind==checkpoint::ItemKind::fp8?item.columns:item.columns/2+item.columns/16,4));bindings_.resize(items.size(),Binding{SIZE_MAX,0,0,-1,-1,-1});
    std::pmr::vector<std::size_t> index(budget_.get());index.resize(items.size());std::iota(index.begin(),index.end(),0);
    std::sort(index.begin(),index.end(),[&](auto x,auto y){return items[x].name<items[y].name;});
    auto bind=[&](std::string_view name,std::size_t offset,std::size_t scales=0,int layer=-1,int expert=-1,int part=-1){
        auto it=std::lower_bound(index.begin(),index.end(),name,[&](auto id,auto n){return items[id].name<n;});require(it!=index.end()&&items[*it].name==name,"model_missing_binding");auto id=*it;require(bindings_[id].item==SIZE_MAX,"model_duplicate_binding");
        require(offset<end_&&(scales==0||scales<end_),"model_binding_bounds");bindings_[id]={id,offset,scales,layer,expert,part};};
    bind("model.language_model.embed_tokens.weight",embedding_);bind("model.language_model.norm.weight",final_norm_);bind("lm_head",head_weights_,head_scales_);
    for(std::size_t i=0;i<a.layers;++i){const auto& l=layers_[i];const auto&p=l.plan;auto base=std::string("model.language_model.layers.")+std::to_string(i);const auto start=l.offset;
        auto aux=start+p.auxiliary;auto dense=[&](const char* suffix,std::size_t n){bind(base+suffix,aux);aux+=n*2;};
        dense(".input_layernorm.weight",a.hidden);bind(base+".post_attention_layernorm.weight",start+p.post_norm);
        bind(base+".mlp.gate.weight",start+p.moe_offset+p.moe.router_offset);bind(base+".mlp.shared_expert_gate.weight",start+p.moe_offset+p.moe.shared_gate_offset);
        auto projection=[&](const char* suffix,std::size_t n){bind(base+suffix,start+p.projections[n].weights,start+p.projections[n].scale);};
        if(l.config.attention==decoder::Attention::linear){
            projection(".linear_attn.in_proj_qkv",0);projection(".linear_attn.in_proj_z",1);projection(".linear_attn.out_proj",2);
            dense(".linear_attn.conv1d.weight",p.linear.conv_elements);dense(".linear_attn.in_proj_a.weight",a.value_heads*a.hidden);dense(".linear_attn.in_proj_b.weight",a.value_heads*a.hidden);
            dense(".linear_attn.A_log",a.value_heads);dense(".linear_attn.dt_bias",a.value_heads);dense(".linear_attn.norm.weight",a.value_dim);
        }else{projection(".self_attn.q_proj",0);projection(".self_attn.k_proj",1);projection(".self_attn.v_proj",2);projection(".self_attn.o_proj",3);dense(".self_attn.q_norm.weight",a.head_dim);dense(".self_attn.k_norm.weight",a.head_dim);}
        for(std::size_t e=0;e<=a.experts;++e){bool shared=e==a.experts;const auto& x=shared?p.moe.shared_layout:p.moe.routed_layout;auto at=start+p.moe_offset+(shared?p.moe.shared_offset:p.moe.experts_offset+e*x.bytes);
            auto prefix=base+(shared?".mlp.shared_expert":".mlp.experts."+std::to_string(e));
            bind(prefix+".gate_proj",at+x.gate_weights,at+x.gate_scales,int(i),int(e),0);bind(prefix+".up_proj",at+x.up_weights,at+x.up_scales,int(i),int(e),1);bind(prefix+".down_proj",at+x.down_weights,at+x.down_scales,int(i),int(e),2);}
    }
    for(const auto& b:bindings_)require(b.item!=SIZE_MAX,"model_unbound_item");
}
float bf16_weight(const quantization::Matrix& m,std::size_t r,std::size_t j){
    require(r<m.rows&&j<m.columns&&m.rows<=SIZE_MAX/m.columns,"model_weight_index");
    const bool fp8=m.encoding==quantization::Encoding::modelopt_fp8;
    require(fp8||m.encoding==quantization::Encoding::modelopt_nvfp4,"model_weight_encoding");
    require(m.multipliers.size()==1&&std::isfinite(m.multipliers[0])&&m.multipliers[0]>0,"model_weight_scale");
    require(m.weights.size()==m.rows*m.columns/(fp8?1:2),"model_weight_shape");
    if(!fp8)require(m.columns%16==0&&m.block_scales.size()==m.rows*(m.columns/16),"model_weight_shape");
    float local;
    if(m.encoding==quantization::Encoding::modelopt_fp8)local=quantization::e4m3fn(m.weights[r*m.columns+j]);
    else{auto code=(m.weights[r*(m.columns/2)+j/2]>>(4*(j%2)))&15;local=quantization::e2m1(code)*quantization::e4m3fn(m.block_scales[r*(m.columns/16)+j/16]);}
    // Both products are FP32, as in the existing Python weight-only decoder.
    return linear::bf16_round(local*m.multipliers[0]);
}
void load(const Layout& layout,std::shared_ptr<checkpoint::MemoryBudget> staging,Sink& sink,const std::atomic_bool* cancelled){
    load_range(layout,std::move(staging),sink,0,layout.device_bytes(),cancelled);
}
void load_range(const Layout& layout,std::shared_ptr<checkpoint::MemoryBudget> staging,Sink& sink,std::size_t first_offset,std::size_t last_offset,const std::atomic_bool* cancelled){
    require(first_offset<last_offset&&last_offset<=layout.device_bytes(),"model_load_range");
    static_assert(std::endian::native==std::endian::little);
    require(staging&&staging->limit()>=std::max<std::size_t>(8,layout.minimum_staging_bytes()),"model_staging_budget");const auto& manifest=layout.manifest();const auto items=manifest.items();const auto limit=staging->limit();
    for(const auto& b:layout.bindings()){
        cancel(cancelled);if(b.weights<first_offset||b.weights>=last_offset)continue;const auto& item=items[b.item];
        if(item.kind==checkpoint::ItemKind::dense){
            std::pmr::vector<std::uint8_t> buffer(staging.get());buffer.resize(std::min<std::size_t>(item.payload_bytes,limit&~std::size_t(1)));
            for(std::size_t at=0;at<item.payload_bytes;at+=buffer.size()){cancel(cancelled);auto n=std::min<std::size_t>(buffer.size(),item.payload_bytes-at);auto chunk=std::span(buffer).first(n);manifest.read_dense(b.item,at,chunk);
                for(std::size_t j=0;j<n;j+=2){unsigned bits=unsigned(chunk[j])|(unsigned(chunk[j+1])<<8);require(std::isfinite(std::bit_cast<float>(std::uint32_t(bits)<<16)),"model_dense_nonfinite");}
                sink.write(b.weights+at,chunk);}
        }else{
            (void)manifest.read_input_scale(b.item); // Validate calibration; weight-only path deliberately does not quantize activations.
            const bool fp8=item.kind==checkpoint::ItemKind::fp8;auto row_bytes=fp8?item.columns:item.columns/2+item.columns/16;require(row_bytes<=limit-4,"model_staging_row");auto rows=(limit-4)/row_bytes;
            for(std::size_t first=0;first<item.rows;first+=rows){cancel(cancelled);auto count=std::min(rows,item.rows-first);auto owned=manifest.load_projection_rows(b.item,first,count,limit,staging);auto m=owned.view();
                // Finite positive scales and canonical bytes are already checked
                // by the shard loader. Monotone magnitude codes let us validate
                // the largest weight per FP4 block (or scalar-scaled FP8 tile),
                // without decoding every parameter or allocating dense scratch.
                if(fp8){unsigned largest=0;for(auto value:m.weights)largest=std::max(largest,unsigned(value&127));(void)linear::bf16_round(quantization::e4m3fn(std::uint8_t(largest))*m.multipliers[0]);}
                else for(std::size_t block=0;block<m.block_scales.size();++block){unsigned largest=0;for(std::size_t j=0;j<8;++j){auto value=m.weights[block*8+j];largest=std::max({largest,unsigned(value&7),unsigned((value>>4)&7)});}float local=quantization::e2m1(std::uint8_t(largest))*quantization::e4m3fn(m.block_scales[block]);(void)linear::bf16_round(local*m.multipliers[0]);}
                sink.write(b.weights+first*(fp8?item.columns:item.columns/2),m.weights);
                if(fp8)sink.write(b.scales,bytes(m.multipliers));else{sink.write(b.scales+first*(item.columns/16),m.block_scales);sink.multiplier(b,m.multipliers[0]);}
            }
        }
    }
    std::array<float,128> frequency{};const auto&a=manifest.architecture();
    for(const auto& l:layout.layers())if(l.offset>=first_offset&&l.offset<last_offset&&l.config.attention==decoder::Attention::full){
        for(std::size_t j=0;j<a.rotary_dim/2;++j){frequency[j]=1/std::pow(float(a.rope_theta),float(2*j)/float(a.rotary_dim));require(std::isfinite(frequency[j])&&frequency[j]>0&&frequency[j]<=1,"model_rope_frequency");}
        sink.write(l.offset+l.plan.frequencies,bytes(std::span<const float>(frequency).first(a.rotary_dim/2)));}
    cancel(cancelled);manifest.check_unchanged();
}
} // namespace kadan::model
