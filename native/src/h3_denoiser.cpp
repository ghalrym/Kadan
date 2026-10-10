#include "kadan/h3_denoiser.hpp"
#include "h3_quant_marker.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <limits>
#include <vector>
namespace kadan::video {
namespace {
using Shard=checkpoint::Shard;using Dtype=checkpoint::Dtype;
void require(bool value,const char* why){if(!value)throw std::runtime_error(why);}
void stop(const std::atomic_bool& flag){require(!flag.load(),"h3_denoiser_cancelled");}
Footprint host(Resources& ledger,Bytes bytes){auto p=ledger.snapshot().capacity;std::fill(p.begin(),p.end(),0);p[0]=bytes;return p;}
struct Admission {
    Resources& ledger;Handle handle;
    Admission(Resources& r,Bytes n):ledger(r),handle(r.reserve(Workload::video,host(r,n))){}
    ~Admission(){if(handle)ledger.released(handle);}
};
struct Active {bool& busy;explicit Active(bool& b):busy(b){require(!b,"busy");b=true;}~Active(){busy=false;}};
void shape(Shard& shard,const std::string& name,std::initializer_list<std::uint64_t> dimensions){
    const auto t=shard.tensor(name);require(t.rank==dimensions.size() && std::equal(dimensions.begin(),dimensions.end(),t.shape.begin()),"h3_denoiser_shape");
}
void finite(std::span<const float> x){for(float v:x)require(std::isfinite(v),"h3_denoiser_nonfinite");}
void read_dense(Shard& shard,const std::string& name,std::span<float> output,const std::atomic_bool& cancel,std::size_t first=0){
    const auto t=shard.tensor(name);require(t.dtype==Dtype::fp32 || t.dtype==Dtype::bf16,"h3_denoiser_dense_dtype");
    std::array<std::uint8_t,4096> raw;const std::size_t width=t.dtype==Dtype::fp32?4:2;
    for(std::size_t offset=0;offset<output.size();){
        stop(cancel);const auto count=std::min(raw.size()/width,output.size()-offset);
        shard.read_tensor(name,(first+offset)*width,{raw.data(),count*width});
        for(std::size_t i=0;i<count;++i){std::uint32_t bits=0;for(std::size_t b=0;b<width;++b)bits|=std::uint32_t(raw[i*width+b])<<(8*b);if(width==2)bits<<=16;output[offset+i]=std::bit_cast<float>(bits);}
        offset+=count;
    }
    finite(output);
}
void rotate(std::span<double> values,std::size_t group){
    require((group==64 || group==256) && values.size()%group==0,"h3_denoiser_rotation_shape");
    for(std::size_t g=0;g<values.size();g+=group)for(std::size_t stride=1;stride<group;stride*=4)
        for(std::size_t start=0;start<group;start+=stride*4)for(std::size_t i=0;i<stride;++i){
            auto* p=values.data()+g+start+i;double a=p[0],b=p[stride],c=p[2*stride],d=p[3*stride];
            p[0]=((a+b)+(c-d))*.5;p[stride]=((a+b)+(d-c))*.5;p[2*stride]=((a+c)+(d-b))*.5;p[3*stride]=((b+c)+(d-a))*.5;
        }
}
std::int8_t quantize(float x){const float lo=std::floor(x),fraction=x-lo;int n=int(lo);if(fraction>.5f || (fraction==.5f && n%2))++n;return std::int8_t(std::clamp(n,-127,127));}
std::size_t quant_group(Shard& shard,const std::string& prefix){
    const auto info=shard.tensor(prefix+".comfy_quant");require(info.dtype==Dtype::u8 && info.rank==1 && info.bytes<=256,"h3_denoiser_quant_marker");
    std::array<std::uint8_t,256> raw;shard.read_tensor(prefix+".comfy_quant",0,{raw.data(),std::size_t(info.bytes)});
    const auto group=detail::convrot_group({reinterpret_cast<const char*>(raw.data()),std::size_t(info.bytes)});
    require(group!=0,"h3_denoiser_quant_marker");return group;
}
// One projection payload resident at a time, released before the admission.
void project_base(Shard& shard,const std::string& prefix,std::span<const float> input,std::size_t in,std::size_t out,std::span<float> output,bool bias,Resources& resources,const std::atomic_bool& cancel,const H3Denoiser::Hook& hook,H3Compute* compute){
    require(in>0 && in<=28672 && out>0 && out<=96768 && input.size()%in==0 && output.size()==input.size()/in*out,"h3_denoiser_projection_shape");
    finite(input);shape(shard,prefix+".weight",{out,in});const auto info=shard.tensor(prefix+".weight");
    const bool quantized=info.dtype==Dtype::i8;require(quantized || info.dtype==Dtype::bf16 || info.dtype==Dtype::fp32,"h3_denoiser_projection_dtype");
    const auto rows=input.size()/in;
    // Fixed read scratch and hook/control envelope included; tensor memory uses exact counts.
    const Bytes bytes=quantized?in*out+out*8+in*9+4096:in*64*4+out*4+4096;
    Admission admitted(resources,bytes);
    auto bias_values=std::make_unique<float[]>(out);if(bias){shape(shard,prefix+".bias",{out});read_dense(shard,prefix+".bias",{bias_values.get(),out},cancel);}
    if(quantized){
        shape(shard,prefix+".weight_scale",{out,1});require(shard.tensor(prefix+".weight_scale").dtype==Dtype::fp32,"h3_denoiser_scale_dtype");const auto group=quant_group(shard,prefix);
        require(in%group==0,"h3_denoiser_rotation_shape");auto weights=std::make_unique<std::uint8_t[]>(in*out);auto scales=std::make_unique<float[]>(out);auto rotated=std::make_unique<double[]>(in);auto codes=std::make_unique<std::int8_t[]>(in);
        for(std::size_t offset=0;offset<in*out;){stop(cancel);const auto count=std::min<std::size_t>(1024*1024,in*out-offset);shard.read_tensor(prefix+".weight",offset,{weights.get()+offset,count});offset+=count;}
        read_dense(shard,prefix+".weight_scale",{scales.get(),out},cancel);for(std::size_t r=0;r<out;++r)require(scales[r]>=0,"h3_denoiser_scale");
        if(compute){compute->convrot({weights.get(),in*out},{scales.get(),out},{bias_values.get(),out},input,in,out,group,output,cancel);return;}
        for(std::size_t t=0;t<rows;++t){
            stop(cancel);for(std::size_t c=0;c<in;++c)rotated[c]=input[t*in+c];rotate({rotated.get(),in},group);float maximum=1e-10f;
            for(std::size_t c=0;c<in;++c){rotated[c]=float(rotated[c]);require(std::isfinite(rotated[c]),"h3_denoiser_nonfinite");maximum=std::max(maximum,std::abs(float(rotated[c])));}
            const float scale=maximum/127;for(std::size_t c=0;c<in;++c)codes[c]=quantize(float(rotated[c])/scale);
            for(std::size_t r=0;r<out;++r){if(r%32==0)stop(cancel);std::int32_t sum=0;for(std::size_t c=0;c<in;++c)sum+=std::int32_t(std::bit_cast<std::int8_t>(weights[r*in+c]))*codes[c];output[t*out+r]=(float(sum)*scale)*scales[r]+bias_values[r];if((r+1)%256==0 && hook)hook("projection",r+1);}
        }
    }else{
        if(compute){Admission dense_admission(resources,in*out*sizeof(float));auto weights=std::make_unique<float[]>(in*out);read_dense(shard,prefix+".weight",{weights.get(),in*out},cancel);compute->dense({weights.get(),in*out},{bias_values.get(),out},input,in,out,output,cancel,true);return;}
        auto block=std::make_unique<float[]>(in*64);
        for(std::size_t r=0;r<out;r+=64){stop(cancel);const auto count=std::min<std::size_t>(64,out-r);read_dense(shard,prefix+".weight",{block.get(),count*in},cancel,r*in);
            for(std::size_t t=0;t<rows;++t)for(std::size_t j=0;j<count;++j){double sum=0;for(std::size_t c=0;c<in;++c)sum+=double(input[t*in+c])*block[j*in+c];output[t*out+r+j]=float(sum)+bias_values[r+j];}
            if(hook)hook("projection",r+count);
        }
    }
    finite(output);
}
void project(Shard& shard,Shard* turbo,const std::string& prefix,std::span<const float> x,std::size_t in,std::size_t out,std::span<float> y,bool bias,bool adapted,Resources& resources,const std::atomic_bool& cancel,const H3Denoiser::Hook& hook,H3Compute* compute){
    project_base(shard,prefix,x,in,out,y,bias,resources,cancel,hook,compute);
    if(!turbo || !adapted)return;
    const auto p="diffusion_model."+prefix;const auto a=turbo->tensor(p+".lora_A.weight");require(a.dtype==Dtype::bf16 && a.rank==2 && a.shape[1]==in && a.shape[0]>0 && a.shape[0]<=384,"h3_denoiser_lora_layout");const auto rank=std::size_t(a.shape[0]);
    shape(*turbo,p+".lora_B.weight",{out,rank});shape(*turbo,p+".alpha",{});float alpha;read_dense(*turbo,p+".alpha",{&alpha,1},cancel);require(alpha>=0,"h3_denoiser_lora_alpha");const float scale=alpha/float(rank);
    const auto rows=x.size()/in;Admission admitted(resources,rows*(rank+out)*sizeof(float));auto tmp=std::make_unique<float[]>(rows*rank),delta=std::make_unique<float[]>(rows*out);
    project_base(*turbo,p+".lora_A",x,in,rank,{tmp.get(),rows*rank},false,resources,cancel,hook,compute);
    project_base(*turbo,p+".lora_B",{tmp.get(),rows*rank},rank,out,{delta.get(),rows*out},false,resources,cancel,hook,compute);
    for(std::size_t i=0;i<y.size();++i)y[i]+=delta[i]*scale;
    finite(y);
}
void rms(std::span<const float> x,std::span<const float> weight,std::span<float> y){
    require(x.size()==y.size() && x.size()%weight.size()==0,"h3_denoiser_norm_shape");
    for(std::size_t at=0;at<x.size();at+=weight.size()){double squares=0;for(std::size_t c=0;c<weight.size();++c)squares+=double(x[at+c])*x[at+c];double scale=1/std::sqrt(squares/weight.size()+1e-5);for(std::size_t c=0;c<weight.size();++c)y[at+c]=double(x[at+c])*scale*weight[c];}finite(y);
}
void norm(Shard& shard,const std::string& name,std::span<const float> x,std::span<float> y,std::span<float> scratch,const std::atomic_bool& cancel){shape(shard,name,{scratch.size()});read_dense(shard,name,scratch,cancel);rms(x,scratch,y);}
void rope(std::span<float> x,std::span<const float> positions,std::span<const float> frequencies){
    for(std::size_t t=0;t<positions.size()/3;++t)for(std::size_t h=0;h<56;++h)for(std::size_t c=0;c<48;++c){const double angle=double(positions[t*3+c/16])*frequencies[c%16];auto* p=x.data()+(t*56+h)*128;const double a=p[c],b=p[c+48];p[c]=a*std::cos(angle)-b*std::sin(angle);p[c+48]=b*std::cos(angle)+a*std::sin(angle);}
}
void attention(std::span<const float> q,std::span<const float> k,std::span<const float> v,std::span<float> y,std::size_t tokens,const std::atomic_bool& cancel){
    std::array<double,H3Denoiser::max_tokens> probability;
    for(std::size_t t=0;t<tokens;++t)for(std::size_t h=0;h<56;++h){stop(cancel);double maximum=-INFINITY;
        for(std::size_t j=0;j<tokens;++j){double sum=0;for(std::size_t c=0;c<128;++c)sum+=double(q[(t*56+h)*128+c])*k[(j*56+h)*128+c];probability[j]=sum/std::sqrt(128.0);maximum=std::max(maximum,probability[j]);}
        double total=0;for(std::size_t j=0;j<tokens;++j){probability[j]=std::exp(probability[j]-maximum);total+=probability[j];}
        for(std::size_t c=0;c<128;++c){double sum=0;for(std::size_t j=0;j<tokens;++j)sum+=probability[j]/total*v[(j*56+h)*128+c];y[(t*56+h)*128+c]=sum;}
    }finite(y);
}
}
H3Denoiser::H3Denoiser(std::shared_ptr<Resources> r,std::shared_ptr<H3Compute> compute):resources_(std::move(r)),compute_(std::move(compute)){require(bool(resources_),"h3_denoiser_resources");}
H3Denoiser::~H3Denoiser(){unload();}
void H3Denoiser::load(const char* root,const std::string& name,const std::atomic_bool& cancel){
    stop(cancel);require(!executing_,"busy");require(!loaded(),"h3_denoiser_already_loaded");Admission admitted(*resources_,metadata_bytes);
    auto shard=std::make_unique<Shard>(root,name,std::make_shared<checkpoint::MemoryBudget>(metadata_bytes),checkpoint::Limits{1024*1024,4096,8192});
    shape(*shard,"condition_proj.weight",{hidden,text_width});shape(*shard,"video_patch_proj.weight",{hidden,video_width});shape(*shard,"audio_patch_proj.weight",{hidden,audio_width});shape(*shard,"rope.inv_freq",{16});
    for(std::size_t layer=0;layer<layers;++layer){stop(cancel);auto p="blocks."+std::to_string(layer)+".";shape(*shard,p+"adaln_proj.linear.weight",{96768,2688});shape(*shard,p+"attn.qkv_proj.weight",{21504,hidden});shape(*shard,p+"mlp.fc1.weight",{28672,hidden});}
    shard->check_unchanged();stop(cancel);shard_=std::move(shard);metadata_=admitted.handle;admitted.handle=0;
}
void H3Denoiser::load_turbo(const char* root,const std::string& name,const std::atomic_bool& cancel){
    stop(cancel);require(!executing_,"busy");require(loaded() && !turbo_,"h3_denoiser_turbo_state");Admission admitted(*resources_,metadata_bytes);
    auto shard=std::make_unique<Shard>(root,name,std::make_shared<checkpoint::MemoryBudget>(metadata_bytes),checkpoint::Limits{1024*1024,2048,8192});
    for(std::size_t layer=0;layer<layers;++layer){stop(cancel);auto p="diffusion_model.blocks."+std::to_string(layer)+".";shape(*shard,p+"attn.qkv_proj.lora_A.weight",{384,hidden});shape(*shard,p+"attn.qkv_proj.lora_B.weight",{21504,384});}
    shard->check_unchanged();stop(cancel);turbo_=std::move(shard);turbo_metadata_=admitted.handle;admitted.handle=0;
}
void H3Denoiser::unload(){require(!executing_,"busy");turbo_.reset();shard_.reset();if(turbo_metadata_){resources_->released(turbo_metadata_);turbo_metadata_=0;}if(metadata_){resources_->released(metadata_);metadata_=0;}}
void H3Denoiser::execute(const Input& input,std::span<float> video_out,std::span<float> audio_out,const std::atomic_bool& cancel,const Hook& hook){
    stop(cancel);require(loaded(),"h3_denoiser_not_loaded");Active active(executing_);
    require(!input.text.empty() && input.text.size()%text_width==0 && input.video.size()%video_width==0 && input.audio.size()%audio_width==0,"h3_denoiser_input_shape");
    const auto nt=input.text.size()/text_width,nv=input.video.size()/video_width,na=input.audio.size()/audio_width,n=nt+nv+na;
    require(nt<=(compute_?H3Tokenizer::max_tokens:512) && n<=(compute_?107856+1206+H3Tokenizer::max_tokens:max_tokens) && n>nt && input.positions.size()==n*3 && input.timesteps.size()==n && input.tags.size()==n && video_out.size()==input.video.size() && audio_out.size()==input.audio.size(),"h3_denoiser_token_limit");
    finite(input.text);finite(input.video);finite(input.audio);finite(input.positions);finite(input.timesteps);
    std::array<float,3> times{};std::size_t time_count=0;Admission indices(*resources_,n*sizeof(std::size_t));auto time_ids=std::make_unique<std::size_t[]>(n);
    for(std::size_t i=0;i<n;++i){require(input.tags[i]<3 && input.timesteps[i]>=0 && input.timesteps[i]<=1,"h3_denoiser_timestep");std::size_t id=0;while(id<time_count && times[id]!=input.timesteps[i])++id;if(id==time_count){require(time_count<3,"h3_denoiser_time_count");times[time_count++]=input.timesteps[i];}time_ids[i]=id;}
    constexpr std::size_t inner=7168,ff=14336;
    // Three unique timesteps, all three AdaLN modalities; fixed scratch ledger
    // also covers stack arrays, checkpoint read chunks and norm vectors.
    constexpr std::size_t fixed=3*(256+5376+2688+96768)+hidden+16;
    constexpr std::size_t per_token=hidden*3+inner*7+ff*2;
    Admission admitted(*resources_,(n*per_token+fixed)*4+max_tokens*(sizeof(double)+sizeof(std::size_t))+16384);
    auto memory=std::make_unique<float[]>(n*per_token+fixed);std::size_t offset=0;
    auto span=[&](std::size_t count){std::span<float> s(memory.get()+offset,count);offset+=count;return s;};
    auto current=span(n*hidden),normalized=span(n*hidden),projected=span(n*hidden),qkv=span(n*inner*3),q=span(n*inner),k=span(n*inner),v=span(n*inner),attended=span(n*inner),mlp=span(n*ff*2);
    auto frequencies=span(16),norm_weight=span(hidden),t_freq=span(3*256),t_hidden=span(3*5376),t_embed=span(3*2688),adaln=span(3*96768);
    require(offset<=n*per_token+fixed,"h3_denoiser_scratch");shard_->check_unchanged();if(turbo_)turbo_->check_unchanged();read_dense(*shard_,"rope.inv_freq",frequencies,cancel);
    auto linear=[&](const std::string& p,std::span<const float> x,std::size_t in,std::size_t out,std::span<float> y,bool bias=false,bool adapted=false){project(*shard_,turbo_.get(),p,x,in,out,y,bias,adapted,*resources_,cancel,hook,compute_.get());};
    auto attn=[&](const std::string& p,std::size_t count,bool positional){
        linear(p+".qkv_proj",normalized.first(count*hidden),hidden,inner*3,qkv.first(count*inner*3),false,true);
        // ComfyUI base weights and the ComfyUI Turbo adapter store [QKV,head,128].
        // The original Hugging Face grouped layout is a different checkpoint contract.
        for(std::size_t t=0;t<count;++t)for(std::size_t h=0;h<56;++h)for(std::size_t c=0;c<128;++c){auto at=(t*56+h)*128+c;const auto row=t*inner*3+h*128+c;q[at]=qkv[row];k[at]=qkv[row+inner];v[at]=qkv[row+2*inner];}
        norm(*shard_,p+".q_norm.weight",q.first(count*inner),q.first(count*inner),norm_weight.first(128),cancel);norm(*shard_,p+".k_norm.weight",k.first(count*inner),k.first(count*inner),norm_weight.first(128),cancel);
        if(positional){rope(q.first(count*inner),input.positions,frequencies);rope(k.first(count*inner),input.positions,frequencies);}
        if(compute_)compute_->attention(q.first(count*inner),k.first(count*inner),v.first(count*inner),count,56,56,128,false,attended.first(count*inner),cancel,true);
        else attention(q.first(count*inner),k.first(count*inner),v.first(count*inner),attended.first(count*inner),count,cancel);
        linear(p+".out_proj",attended.first(count*inner),inner,hidden,projected.first(count*hidden),false,true);
    };
    auto feed=[&](const std::string& p,std::size_t count){
        linear(p+".fc1",normalized.first(count*hidden),hidden,ff*2,mlp.first(count*ff*2),false,true);
        // Compact gate*value in place; destination never reaches a future row.
        for(std::size_t t=0;t<count;++t)for(std::size_t c=0;c<ff;++c){const double gate=mlp[t*ff*2+c];mlp[t*ff+c]=float(gate/(1+std::exp(-gate)))*mlp[t*ff*2+ff+c];}
        linear(p+".fc2",mlp.first(count*ff),ff,hidden,projected.first(count*hidden),false,true);
    };
    linear("condition_proj",input.text,text_width,hidden,current.first(nt*hidden),true);
    for(std::size_t b=0;b<2;++b){auto p="token_refiner.blocks."+std::to_string(b);norm(*shard_,p+".norm1.weight",current.first(nt*hidden),normalized.first(nt*hidden),norm_weight,cancel);attn(p+".attn",nt,false);for(std::size_t i=0;i<nt*hidden;++i)current[i]+=projected[i];norm(*shard_,p+".norm2.weight",current.first(nt*hidden),normalized.first(nt*hidden),norm_weight,cancel);feed(p+".mlp",nt);for(std::size_t i=0;i<nt*hidden;++i)current[i]+=projected[i];if(hook)hook("refiner_completed",b+1);}
    norm(*shard_,"token_refiner.final_norm.weight",current.first(nt*hidden),current.first(nt*hidden),norm_weight,cancel);
    if(nv)linear("video_patch_proj",input.video,video_width,hidden,current.subspan(nt*hidden,nv*hidden),true);
    if(na)linear("audio_patch_proj",input.audio,audio_width,hidden,current.subspan((nt+nv)*hidden,na*hidden),true);
    for(std::size_t t=0;t<time_count;++t)for(std::size_t c=0;c<128;++c){const double angle=times[t]*std::exp(-std::log(10000.0)*double(c)/128);t_freq[t*256+c]=std::cos(angle);t_freq[t*256+c+128]=std::sin(angle);}
    linear("time_embedder.proj_in",t_freq.first(time_count*256),256,5376,t_hidden.first(time_count*5376),true);for(std::size_t i=0;i<time_count*5376;++i){double a=t_hidden[i];t_hidden[i]=a/(1+std::exp(-a));}
    linear("time_embedder.proj_out",t_hidden.first(time_count*5376),5376,2688,t_embed.first(time_count*2688),true);for(std::size_t i=0;i<time_count*2688;++i){double a=t_embed[i];t_embed[i]=a/(1+std::exp(-a));}
    auto modulate=[&](std::size_t group){for(std::size_t t=0;t<n;++t){const auto base=(time_ids[t]*3+input.tags[t])*6*hidden;for(std::size_t c=0;c<hidden;++c)normalized[t*hidden+c]=normalized[t*hidden+c]*(1+adaln[base+(group+1)*hidden+c])+adaln[base+group*hidden+c];}};
    auto residual=[&](std::size_t group){for(std::size_t t=0;t<n;++t){const auto base=(time_ids[t]*3+input.tags[t])*6*hidden;for(std::size_t c=0;c<hidden;++c)current[t*hidden+c]+=projected[t*hidden+c]*adaln[base+group*hidden+c];}finite(current);};
    for(std::size_t b=0;b<layers;++b){stop(cancel);auto p="blocks."+std::to_string(b);linear(p+".adaln_proj.linear",t_embed.first(time_count*2688),2688,96768,adaln.first(time_count*96768),true);norm(*shard_,p+".norm1.weight",current,normalized,norm_weight,cancel);modulate(0);attn(p+".attn",n,true);residual(2);norm(*shard_,p+".norm2.weight",current,normalized,norm_weight,cancel);modulate(3);feed(p+".mlp",n);residual(5);if(hook)hook("block_completed",b+1);}
    linear("final_layer.adaln_proj.linear",t_embed.first(time_count*2688),2688,hidden*2,adaln.first(time_count*hidden*2),true);norm(*shard_,"final_layer.norm.weight",current,normalized,norm_weight,cancel);
    for(std::size_t t=0;t<n;++t)for(std::size_t c=0;c<hidden;++c)normalized[t*hidden+c]=normalized[t*hidden+c]*(1+adaln[(time_ids[t]*2+1)*hidden+c])+adaln[time_ids[t]*2*hidden+c];
    // Compute selected rows only; dense F64 reductions have no batch-dependent kernel choice.
    if(nv)linear("final_layer.video_out",normalized.subspan(nt*hidden,nv*hidden),hidden,video_width,video_out,true);
    if(na)linear("final_layer.audio_out",normalized.subspan((nt+nv)*hidden,na*hidden),hidden,audio_width,audio_out,true);
    stop(cancel);shard_->check_unchanged();if(turbo_)turbo_->check_unchanged();if(hook)hook("completed",n);stop(cancel);
}
}
