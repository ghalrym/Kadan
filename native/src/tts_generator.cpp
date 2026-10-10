#include "kadan/tts_generator.hpp"
#include "kadan/checkpoint.hpp"
#include <algorithm>
#include <bit>
#include <cmath>
#include <map>
#include <vector>
namespace kadan::tts {
namespace {
void need(bool b,const char* e){if(!b)throw std::runtime_error(e);}
void stop(const std::atomic_bool& c){need(!c.load(),"tts_generation_cancelled");}
Footprint host(Resources& r,Bytes n){auto f=r.snapshot().capacity;std::fill(f.begin(),f.end(),0);f[0]=n;return f;}
struct Lease{Resources& r;Handle h;Lease(Resources& a,Bytes b):r(a),h(r.reserve(Workload::tts,host(r,b))){}~Lease(){if(h)r.released(h);}};
struct Buffer{Lease lease;std::unique_ptr<float[]> data;std::size_t size;Buffer(Resources& r,std::size_t n):lease(r,n*4),data(std::make_unique<float[]>(n)),size(n){}std::span<float> span(){return {data.get(),size};}};
struct Active{bool& busy;Active(bool& b):busy(b){need(!b,"busy");b=true;}~Active(){busy=false;}};
struct Pin{Resources& r;Handle h;Pin(Resources& a,Handle b):r(a),h(b){r.pin(h);}~Pin(){r.unpin(h);}};
struct Spec{std::string name;std::vector<std::uint64_t> shape;std::size_t count=1,offset=0;};
void finite(std::span<const float> x){for(float v:x)need(std::isfinite(v),"tts_generation_nonfinite");}
void config(GeneratorConfig d){for(auto s:{d.talker,d.predictor})need(s.state>0&&s.state<=2048&&s.heads>0&&s.heads<=16&&s.kv_heads>0&&s.kv_heads<=s.heads&&s.heads%s.kv_heads==0&&s.head_dim>0&&s.head_dim<=128&&s.head_dim%2==0&&s.layers>0&&s.layers<=28&&s.intermediate>0&&s.intermediate<=6144,"tts_generation_config");need(d.text_vocabulary>0&&d.text_vocabulary<=151936&&d.codec_vocabulary>0&&d.codec_vocabulary<=3072&&d.codebooks>=2&&d.codebooks<=16&&d.entries>0&&d.entries<=2048&&d.entries<d.codec_vocabulary,"tts_generation_config");}
std::vector<Spec> layout(GeneratorConfig d){std::vector<Spec> s;auto add=[&](std::string p,std::initializer_list<std::uint64_t> shape){s.push_back({std::move(p),shape});};auto linear=[&](std::string p,std::size_t in,std::size_t out,bool bias=false){add(p+".weight",{out,in});if(bias)add(p+".bias",{out});};
    add("talker.model.text_embedding.weight",{d.text_vocabulary,d.talker.state});add("talker.model.codec_embedding.weight",{d.codec_vocabulary,d.talker.state});for(auto n:{"linear_fc1","linear_fc2"})linear(std::string("talker.text_projection.")+n,d.talker.state,d.talker.state,true);linear("talker.codec_head",d.talker.state,d.codec_vocabulary);
    if(d.predictor.state!=d.talker.state)linear("talker.code_predictor.small_to_mtp_projection",d.talker.state,d.predictor.state,true);
    for(std::size_t i=0;i<d.codebooks-1;++i){add("talker.code_predictor.model.codec_embedding."+std::to_string(i)+".weight",{d.entries,d.talker.state});linear("talker.code_predictor.lm_head."+std::to_string(i),d.predictor.state,d.entries);}
    for(bool small:{false,true}){const auto a=small?d.predictor:d.talker;const std::string root=small?"talker.code_predictor.model":"talker.model";add(root+".norm.weight",{a.state});for(std::size_t i=0;i<a.layers;++i){const auto p=root+".layers."+std::to_string(i);for(auto n:{"input_layernorm","post_attention_layernorm"})add(p+"."+n+".weight",{a.state});for(auto n:{"q_norm","k_norm"})add(p+".self_attn."+n+".weight",{a.head_dim});linear(p+".self_attn.q_proj",a.state,a.heads*a.head_dim);for(auto n:{"k_proj","v_proj"})linear(p+".self_attn."+n,a.state,a.kv_heads*a.head_dim);linear(p+".self_attn.o_proj",a.heads*a.head_dim,a.state);for(auto n:{"gate_proj","up_proj"})linear(p+".mlp."+n,a.state,a.intermediate);linear(p+".mlp.down_proj",a.intermediate,a.state);}}
    std::size_t at=0;for(auto& v:s){for(auto n:v.shape)v.count*=n;v.offset=at;at+=v.count;}return s;
}
struct Cache{
    Buffer keys,values;StackConfig d;std::size_t capacity,length=0;
    Cache(Resources& r,StackConfig a,std::size_t maximum):keys(r,a.layers*maximum*a.kv_heads*a.head_dim),values(r,a.layers*maximum*a.kv_heads*a.head_dim),d(a),capacity(maximum){}
};
float silu(float x){double v=x;return float(v>=0?v/(1+std::exp(-v)):v*std::exp(v)/(1+std::exp(v)));}
}
struct CodeGenerator::Impl{
    Lease metadata;GeneratorConfig d;std::vector<Spec> specs;std::map<std::string,std::size_t,std::less<>> index;std::unique_ptr<Buffer> weights;
    Impl(Resources& r,GeneratorConfig a):metadata(r,16*1024*1024),d(a),specs(layout(a)){for(std::size_t i=0;i<specs.size();++i)index.emplace(specs[i].name,i);}
    std::span<const float> w(const std::string& p){auto& s=specs.at(index.at(p));return {weights->data.get()+s.offset,s.count};}
    void linear(const std::string& p,std::span<const float> x,std::span<float> y,const std::atomic_bool& cancel,bool bias=false){auto a=w(p+".weight"),b=bias?w(p+".bias"):std::span<const float>{};need(a.size()==x.size()*y.size(),"tts_generation_projection_shape");for(std::size_t o=0;o<y.size();++o){if(o%32==0)stop(cancel);float v=bias?b[o]:0;for(std::size_t c=0;c<x.size();++c)v+=a[o*x.size()+c]*x[c];y[o]=v;}finite(y);}
    void norm(const std::string& p,std::span<const float> x,std::span<float> y){auto a=w(p+".weight");need(x.size()==a.size()&&y.size()==x.size(),"tts_generation_norm_shape");double mean=0;for(auto v:x)mean+=double(v)*v;const float scale=float(1/std::sqrt(mean/x.size()+1e-6));for(std::size_t i=0;i<x.size();++i)y[i]=x[i]*scale*a[i];finite(y);}
    void embedding(const std::string& p,std::uint32_t id,std::span<float> y){auto a=w(p+".weight");need(std::size_t(id)<a.size()/y.size(),"tts_generation_token_range");std::copy_n(a.data()+id*y.size(),y.size(),y.data());}
    void text(Resources& r,std::uint32_t id,std::span<float> output,const std::atomic_bool& cancel){Buffer first(r,d.talker.state),second(r,d.talker.state);embedding("talker.model.text_embedding",id,first.span());linear("talker.text_projection.linear_fc1",first.span(),second.span(),cancel,true);for(auto& v:second.span())v=silu(v);linear("talker.text_projection.linear_fc2",second.span(),output,cancel,true);}
    void step(Resources& r,const std::string& root,Cache& cache,std::span<const float> input,std::span<float> output,const std::atomic_bool& cancel){
        auto a=cache.d;need(cache.length<cache.capacity&&input.size()==a.state&&output.size()==a.state,"tts_generation_context");const auto qw=a.heads*a.head_dim,kw=a.kv_heads*a.head_dim;
        Buffer x(r,a.state),n(r,a.state),q(r,qw),k(r,kw),v(r,kw),mix(r,qw),delta(r,a.state),gate(r,a.intermediate),up(r,a.intermediate),scores(r,cache.length+1);
        std::copy(input.begin(),input.end(),x.data.get());
        for(std::size_t layer=0;layer<a.layers;++layer){stop(cancel);const auto p=root+".layers."+std::to_string(layer);norm(p+".input_layernorm",x.span(),n.span());linear(p+".self_attn.q_proj",n.span(),q.span(),cancel);linear(p+".self_attn.k_proj",n.span(),k.span(),cancel);linear(p+".self_attn.v_proj",n.span(),v.span(),cancel);
            for(auto item:{std::pair{q.span(),a.heads},std::pair{k.span(),a.kv_heads}}){for(std::size_t h=0;h<item.second;++h){auto head=item.first.subspan(h*a.head_dim,a.head_dim);norm(p+(item.second==a.heads&&item.first.data()==q.data.get()?".self_attn.q_norm":".self_attn.k_norm"),head,head);for(std::size_t c=0;c<a.head_dim/2;++c){float angle=float(cache.length)*std::pow(1000000.f,-2.f*float(c)/float(a.head_dim)),co=std::cos(angle),si=std::sin(angle),u=head[c],z=head[c+a.head_dim/2];head[c]=u*co-z*si;head[c+a.head_dim/2]=z*co+u*si;}}}
            auto offset=(layer*cache.capacity+cache.length)*kw;std::copy(k.span().begin(),k.span().end(),cache.keys.data.get()+offset);std::copy(v.span().begin(),v.span().end(),cache.values.data.get()+offset);
            for(std::size_t h=0;h<a.heads;++h){stop(cancel);auto kh=h/(a.heads/a.kv_heads);float maximum=-INFINITY;for(std::size_t t=0;t<=cache.length;++t){float value=0;auto at=(layer*cache.capacity+t)*kw+kh*a.head_dim;for(std::size_t c=0;c<a.head_dim;++c)value+=q.data[h*a.head_dim+c]*cache.keys.data[at+c];value/=std::sqrt(float(a.head_dim));scores.data[t]=value;maximum=std::max(maximum,value);}double total=0;for(std::size_t t=0;t<=cache.length;++t){scores.data[t]=std::exp(scores.data[t]-maximum);total+=scores.data[t];}need(std::isfinite(total)&&total>0,"tts_generation_nonfinite");for(std::size_t c=0;c<a.head_dim;++c){float value=0;for(std::size_t t=0;t<=cache.length;++t)value+=float(scores.data[t]/total)*cache.values.data[(layer*cache.capacity+t)*kw+kh*a.head_dim+c];mix.data[h*a.head_dim+c]=value;}}
            linear(p+".self_attn.o_proj",mix.span(),delta.span(),cancel);for(std::size_t i=0;i<a.state;++i)x.data[i]+=delta.data[i];norm(p+".post_attention_layernorm",x.span(),n.span());linear(p+".mlp.gate_proj",n.span(),gate.span(),cancel);linear(p+".mlp.up_proj",n.span(),up.span(),cancel);for(std::size_t i=0;i<a.intermediate;++i)gate.data[i]=silu(gate.data[i])*up.data[i];linear(p+".mlp.down_proj",gate.span(),delta.span(),cancel);for(std::size_t i=0;i<a.state;++i)x.data[i]+=delta.data[i];finite(x.span());
        }
        norm(root+".norm",x.span(),output);stop(cancel);++cache.length;
    }
};
CodeGenerator::CodeGenerator(std::shared_ptr<Resources> r):resources_(std::move(r)){need(bool(resources_),"tts_resources_required");}
CodeGenerator::~CodeGenerator(){unload();}
void CodeGenerator::load(const char* root,const std::string& name,GeneratorConfig d,const std::atomic_bool& cancel){Active active(busy_);stop(cancel);need(!model_,"tts_generation_loaded");config(d);auto m=std::make_unique<Impl>(*resources_,d);Lease parser(*resources_,16*1024*1024);checkpoint::Shard shard(root,name,std::make_shared<checkpoint::MemoryBudget>(16*1024*1024),{2*1024*1024,4096,4096});for(const auto& s:m->specs){auto t=shard.tensor(s.name);need((t.dtype==checkpoint::Dtype::bf16||t.dtype==checkpoint::Dtype::fp32)&&t.rank==s.shape.size()&&std::equal(s.shape.begin(),s.shape.end(),t.shape.begin()),"tts_generation_tensor_layout");}auto& last=m->specs.back();need(last.offset+last.count<=3ULL*1024*1024*1024,"tts_generation_weight_limit");m->weights=std::make_unique<Buffer>(*resources_,last.offset+last.count);Lease staging(*resources_,4096);std::array<std::uint8_t,4096> bytes;
    for(const auto& s:m->specs){auto type=shard.tensor(s.name).dtype;std::size_t width=type==checkpoint::Dtype::bf16?2:4;for(std::size_t at=0;at<s.count;){stop(cancel);auto count=std::min(bytes.size()/width,s.count-at);shard.read_tensor(s.name,at*width,{bytes.data(),count*width});for(std::size_t i=0;i<count;++i){std::uint32_t bits=0;for(std::size_t j=0;j<width;++j)bits|=std::uint32_t(bytes[i*width+j])<<(8*j);float v=std::bit_cast<float>(width==2?bits<<16:bits);need(std::isfinite(v),"tts_generation_nonfinite_weight");m->weights->data[s.offset+at+i]=v;}at+=count;}}
    shard.check_unchanged();stop(cancel);resources_->loaded(m->weights->lease.h);model_=std::move(m);
}
void CodeGenerator::unload(){need(!busy_,"busy");if(model_){resources_->begin_eviction(model_->weights->lease.h);model_.reset();}}
GeneratedCodes CodeGenerator::generate(std::span<const std::uint32_t> text,const VoicePrompt& voice,std::span<std::uint32_t> codes,const std::atomic_bool& cancel,const Hook& hook,std::size_t minimum,float penalty){Active active(busy_);stop(cancel);need(bool(model_),"tts_generation_not_loaded");auto& m=*model_;auto d=m.d;need(!text.empty()&&text.size()<=2048&&!codes.empty()&&codes.size()%d.codebooks==0&&codes.size()/d.codebooks<=300,"tts_generation_shape");const auto maximum=codes.size()/d.codebooks;need(minimum<=maximum&&std::isfinite(penalty)&&penalty>=1&&penalty<=2,"tts_generation_policy");for(auto id:text)need(id<d.text_vocabulary,"tts_generation_token_range");for(auto id:voice.role)need(id<d.text_vocabulary,"tts_generation_token_range");for(auto id:{voice.text_bos,voice.text_eos,voice.text_pad})need(id<d.text_vocabulary,"tts_generation_token_range");for(auto id:{voice.think,voice.think_bos,voice.language,voice.think_eos,voice.speaker,voice.codec_pad,voice.codec_bos,voice.codec_eos})need(id<d.codec_vocabulary,"tts_generation_token_range");need(voice.codec_eos>=d.entries,"tts_generation_eos");Pin pin(*resources_,m.weights->lease.h);
    Cache main(*resources_,d.talker,text.size()+11+maximum),small(*resources_,d.predictor,d.codebooks+1);Buffer hidden(*resources_,d.talker.state),input(*resources_,d.talker.state),pad(*resources_,d.talker.state),bos(*resources_,d.talker.state),end(*resources_,d.talker.state),codec(*resources_,d.talker.state),aggregate(*resources_,d.talker.state),subinput(*resources_,d.predictor.state),subhidden(*resources_,d.predictor.state),logits(*resources_,std::max(d.codec_vocabulary,d.entries));
    Lease repeated_admission(*resources_,d.codec_vocabulary);std::vector<std::uint8_t> repeated(d.codec_vocabulary);
    m.text(*resources_,voice.text_pad,pad.span(),cancel);m.text(*resources_,voice.text_bos,bos.span(),cancel);m.text(*resources_,voice.text_eos,end.span(),cancel);
    auto main_step=[&]{m.step(*resources_,"talker.model",main,input.span(),hidden.span(),cancel);};
    for(auto id:voice.role){m.text(*resources_,id,input.span(),cancel);main_step();}
    const std::array<std::uint32_t,6> prefix{voice.think,voice.think_bos,voice.language,voice.think_eos,voice.speaker,voice.codec_pad};
    for(std::size_t i=0;i<prefix.size();++i){m.embedding("talker.model.codec_embedding",prefix[i],codec.span());for(std::size_t c=0;c<d.talker.state;++c)input.data[c]=codec.data[c]+(i+1==prefix.size()?bos.data[c]:pad.data[c]);main_step();}
    m.embedding("talker.model.codec_embedding",voice.codec_pad,codec.span());for(auto id:text){m.text(*resources_,id,input.span(),cancel);for(std::size_t c=0;c<d.talker.state;++c)input.data[c]+=codec.data[c];main_step();}for(std::size_t c=0;c<d.talker.state;++c)input.data[c]=end.data[c]+codec.data[c];main_step();m.embedding("talker.model.codec_embedding",voice.codec_bos,codec.span());for(std::size_t c=0;c<d.talker.state;++c)input.data[c]=pad.data[c]+codec.data[c];main_step();if(hook)hook("prefill",main.length);
    auto small_step=[&](std::span<const float> x){if(d.predictor.state!=d.talker.state)m.linear("talker.code_predictor.small_to_mtp_projection",x,subinput.span(),cancel,true);else std::copy(x.begin(),x.end(),subinput.data.get());m.step(*resources_,"talker.code_predictor.model",small,subinput.span(),subhidden.span(),cancel);};
    for(std::size_t frame=0;frame<maximum;++frame){stop(cancel);m.linear("talker.codec_head",hidden.span(),logits.span().first(d.codec_vocabulary),cancel);std::uint32_t first=0;float best=-INFINITY;for(std::size_t id=0;id<d.codec_vocabulary;++id){if(id>=d.entries&&id!=voice.codec_eos)continue;if(id==voice.codec_eos&&frame<minimum)continue;float value=logits.data[id];if(repeated[id])value=value<0?value*penalty:value/penalty;if(value>best){best=value;first=std::uint32_t(id);}}
        if(first==voice.codec_eos){return {frame,true};}
        repeated[first]=1;codes[frame*d.codebooks]=first;m.embedding("talker.model.codec_embedding",first,codec.span());std::copy(codec.span().begin(),codec.span().end(),aggregate.data.get());small.length=0;small_step(hidden.span());small_step(codec.span());
        for(std::size_t group=1;group<d.codebooks;++group){m.linear("talker.code_predictor.lm_head."+std::to_string(group-1),subhidden.span(),logits.span().first(d.entries),cancel);auto id=std::uint32_t(std::max_element(logits.data.get(),logits.data.get()+d.entries)-logits.data.get());codes[frame*d.codebooks+group]=id;m.embedding("talker.code_predictor.model.codec_embedding."+std::to_string(group-1),id,codec.span());for(std::size_t c=0;c<d.talker.state;++c)aggregate.data[c]+=codec.data[c];if(group+1<d.codebooks)small_step(codec.span());}
        if(hook){hook("frame",frame);}
        stop(cancel);for(std::size_t c=0;c<d.talker.state;++c)input.data[c]=aggregate.data[c]+pad.data[c];main_step();
    }
    return {maximum,false};
}
}
