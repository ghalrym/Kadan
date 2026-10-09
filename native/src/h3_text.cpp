#include "kadan/h3_text.hpp"
#include "h3_quant_marker.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <fcntl.h>
#include <unistd.h>
namespace kadan::video {
namespace {
void require(bool ok,const char* error){if(!ok)throw std::runtime_error(error);}
void stop(const std::atomic_bool& flag){require(!flag.load(),"h3_text_cancelled");}
Footprint host(Resources& r,Bytes bytes){auto p=r.snapshot().capacity;std::fill(p.begin(),p.end(),0);p[0]=bytes;return p;}
struct Admission {
    Resources& r;Handle h;
    Admission(Resources& ledger,Bytes bytes):r(ledger),h(r.reserve(Workload::video,host(r,bytes))){}
    ~Admission(){if(h)r.released(h);}
};
struct Active {bool& busy;explicit Active(bool& b):busy(b){require(!busy,"busy");busy=true;}~Active(){busy=false;}};
void shape(checkpoint::Shard& shard,const std::string& name,checkpoint::Dtype dtype,std::initializer_list<std::uint64_t> dimensions) {
    const auto info=shard.tensor(name);require(info.dtype==dtype && info.rank==dimensions.size(),"h3_text_tensor_layout");
    require(std::equal(dimensions.begin(),dimensions.end(),info.shape.begin()),"h3_text_tensor_layout");
}
float bf16(std::uint16_t bits){const auto value=std::bit_cast<float>(std::uint32_t(bits)<<16);require(std::isfinite(value),"h3_text_nonfinite_weight");return value;}
void dense(checkpoint::Shard& shard,const std::string& name,std::span<float> destination,Resources& resources,const std::atomic_bool& cancel,std::size_t first=0) {
    const auto info=shard.tensor(name);require(info.dtype==checkpoint::Dtype::bf16,"h3_text_tensor_layout");
    Admission admitted(resources,4096);std::array<std::uint8_t,4096> bytes;
    for(std::size_t at=0;at<destination.size();) {
        stop(cancel);const auto count=std::min<std::size_t>(bytes.size()/2,destination.size()-at);
        shard.read_tensor(name,(first+at)*2,{bytes.data(),count*2});
        for(std::size_t i=0;i<count;++i)destination[at+i]=bf16(std::uint16_t(bytes[i*2])|(std::uint16_t(bytes[i*2+1])<<8));
        at+=count;
    }
}
// F64 accumulation keeps INT8 rounding independent of F32 reduction order.
// Independent radix-4 regular Hadamard transform: H4 has one negative entry
// in each row (at column 3-row). Four normalized stages implement H256.
void rotate(std::span<double> values) {
    require(values.size()%256==0,"h3_text_rotation_shape");
    for(std::size_t group=0;group<values.size();group+=256)
        for(std::size_t stride=1;stride<256;stride*=4)
            for(std::size_t start=0;start<256;start+=stride*4)
                for(std::size_t i=0;i<stride;++i) {
                    auto* a=values.data()+group+start+i;
                    const double x=a[0],y=a[stride],z=a[2*stride],w=a[3*stride];
                    a[0]=((x+y)+(z-w))*0.5f;
                    a[stride]=((x+y)+(w-z))*0.5f;
                    a[2*stride]=((x+z)+(w-y))*0.5f;
                    a[3*stride]=((y+z)+(w-x))*0.5f;
                }
}
// Explicit nearest-even avoids dependence on the process floating-point mode.
std::int8_t quantize(float value) {
    const float low=std::floor(value),fraction=value-low;
    int result=int(low);
    if(fraction>0.5f || (fraction==0.5f && result%2!=0))++result;
    return static_cast<std::int8_t>(std::clamp(result,-127,127));
}
void linear(checkpoint::Shard& shard,const std::string& prefix,std::span<const float> input,
    std::size_t in,std::size_t out,std::span<float> destination,Resources& resources,
    const std::atomic_bool& cancel,const H3TextEncoder::Hook& hook,H3Compute* compute) {
    stop(cancel);require(input.size()%in==0 && destination.size()==input.size()/in*out,"h3_text_projection_shape");
    require(in<=25600 && out<=25600 && in%256==0,"h3_text_projection_limit");
    shape(shard,prefix+".weight",checkpoint::Dtype::i8,{out,in});
    shape(shard,prefix+".weight_scale",checkpoint::Dtype::fp32,{out,1});
    const auto marker=shard.tensor(prefix+".comfy_quant");
    require(marker.dtype==checkpoint::Dtype::u8 && marker.rank==1 && marker.bytes<=256,"h3_text_quant_marker");
    Admission admission(resources,in*out+out*4+in*9+256);
    auto weights=std::make_unique<std::uint8_t[]>(in*out);
    auto scales=std::make_unique<float[]>(out);
    auto rotated=std::make_unique<double[]>(in);
    auto codes=std::make_unique<std::int8_t[]>(in);
    std::array<std::uint8_t,256> marker_bytes;
    shard.read_tensor(prefix+".comfy_quant",0,{marker_bytes.data(),std::size_t(marker.bytes)});
    const std::string_view marker_text(reinterpret_cast<const char*>(marker_bytes.data()),std::size_t(marker.bytes));
    require(detail::convrot_group(marker_text)==256,"h3_text_quant_marker");
    for(std::size_t offset=0;offset<in*out;) {
        stop(cancel);const auto count=std::min<std::size_t>(1024*1024,in*out-offset);
        shard.read_tensor(prefix+".weight",offset,{weights.get()+offset,count});offset+=count;
    }
    static_assert(std::endian::native==std::endian::little);
    shard.read_tensor(prefix+".weight_scale",0,{reinterpret_cast<std::uint8_t*>(scales.get()),out*4});
    for(std::size_t r=0;r<out;++r)require(std::isfinite(scales[r]) && scales[r]>=0,"h3_text_scale");
    if(compute){compute->convrot({weights.get(),in*out},{scales.get(),out},{},input,in,out,256,destination,cancel);return;}
    for(std::size_t token=0;token<input.size()/in;++token) {
        stop(cancel);
        for(std::size_t c=0;c<in;++c){const auto value=input[token*in+c];require(std::isfinite(value),"h3_text_nonfinite_input");rotated[c]=value;}
        rotate({rotated.get(),in});float maximum=1e-10f;
        for(std::size_t c=0;c<in;++c){rotated[c]=float(rotated[c]);require(std::isfinite(rotated[c]),"h3_text_nonfinite_output");maximum=std::max(maximum,std::abs(float(rotated[c])));}
        const float scale=maximum/127.0f;
        for(std::size_t c=0;c<in;++c)codes[c]=quantize(float(rotated[c])/scale);
        for(std::size_t r=0;r<out;++r) {
            if(r%32==0)stop(cancel);
            std::int32_t sum=0;
            for(std::size_t c=0;c<in;++c)sum+=std::int32_t(std::bit_cast<std::int8_t>(weights[r*in+c]))*codes[c];
            const float value=(float(sum)*scale)*scales[r];require(std::isfinite(value),"h3_text_nonfinite_output");
            destination[token*out+r]=value;
            if((r+1)%32==0 && hook)hook("projection",token*out+r+1);
        }
    }
}
void rms(std::span<const float> input,std::span<const float> weights,std::span<float> output) {
    require(input.size()==output.size() && input.size()%weights.size()==0,"h3_text_norm_shape");
    for(std::size_t at=0;at<input.size();at+=weights.size()) {
        double squares=0;for(std::size_t c=0;c<weights.size();++c)squares+=double(input[at+c])*input[at+c];
        require(std::isfinite(squares),"h3_text_nonfinite_output");
        const double inverse=1.0/std::sqrt(squares/double(weights.size())+1e-6);
        for(std::size_t c=0;c<weights.size();++c)output[at+c]=(input[at+c]*inverse)*weights[c];
    }
}
void rope(std::span<float> values,std::size_t heads,std::size_t tokens) {
    for(std::size_t t=0;t<tokens;++t)for(std::size_t h=0;h<heads;++h)for(std::size_t c=0;c<64;++c) {
        const double angle=double(t)/std::pow(5000000.0,double(c)/64.0);
        auto* p=values.data()+(t*heads+h)*128;const double x=p[c],y=p[c+64];
        p[c]=x*std::cos(angle)-y*std::sin(angle);p[c+64]=y*std::cos(angle)+x*std::sin(angle);
    }
}
void attention(std::span<const float> q,std::span<const float> k,std::span<const float> v,
    std::span<float> output,std::size_t tokens,const std::atomic_bool& cancel) {
    std::array<double,H3TextEncoder::max_tokens> probabilities;
    for(std::size_t t=0;t<tokens;++t)for(std::size_t h=0;h<64;++h) {
        stop(cancel);double maximum=-INFINITY;
        for(std::size_t j=0;j<=t;++j) {
            double dot=0;for(std::size_t c=0;c<128;++c)dot+=double(q[(t*64+h)*128+c])*k[(j*8+h/8)*128+c];
            probabilities[j]=dot/std::sqrt(128.0);maximum=std::max(maximum,probabilities[j]);
        }
        double denominator=0;for(std::size_t j=0;j<=t;++j){probabilities[j]=std::exp(probabilities[j]-maximum);denominator+=probabilities[j];}
        for(std::size_t c=0;c<128;++c) {
            double result=0;for(std::size_t j=0;j<=t;++j)result+=(probabilities[j]/denominator)*v[(j*8+h/8)*128+c];
            require(std::isfinite(result),"h3_text_nonfinite_output");output[(t*64+h)*128+c]=result;
        }
    }
}
struct Output {
    std::string temp;int fd;
    explicit Output(const std::string& name):temp(name+".partial-XXXXXX"),fd(mkstemp(temp.data())){require(fd>=0,"h3_text_output_open");}
    ~Output(){if(fd>=0)close(fd);unlink(temp.c_str());}
    void write(const void* source,std::size_t bytes) {
        const auto* p=static_cast<const char*>(source);
        while(bytes){auto n=::write(fd,p,bytes);if(n<0 && errno==EINTR)continue;require(n>0,"h3_text_output_write");p+=n;bytes-=n;}
    }
    void publish(const std::string& name){require(fsync(fd)==0,"h3_text_output_sync");require(link(temp.c_str(),name.c_str())==0,"h3_text_output_publish");}
};
}
H3TextEncoder::H3TextEncoder(std::shared_ptr<Resources> resources,std::shared_ptr<H3Compute> compute):resources_(std::move(resources)),compute_(std::move(compute)){require(bool(resources_),"h3_text_resources_required");}
H3TextEncoder::~H3TextEncoder(){unload();}
Footprint H3TextEncoder::host(Bytes bytes) const{return video::host(*resources_,bytes);}
void H3TextEncoder::load(const char* root,const std::string& basename,const std::atomic_bool& cancel) {
    stop(cancel);require(!executing_,"busy");require(!loaded(),"h3_text_already_loaded");
    Admission metadata(*resources_,metadata_bytes);
    auto shard=std::make_unique<checkpoint::Shard>(root,basename,std::make_shared<checkpoint::MemoryBudget>(metadata_bytes),checkpoint::Limits{1024*1024,2048,4096});
    shape(*shard,"model.embed_tokens.weight",checkpoint::Dtype::bf16,{vocab,hidden});
    for(std::size_t layer=0;layer<layers;++layer) {
        stop(cancel);const auto p="model.layers."+std::to_string(layer)+".";
        shape(*shard,p+"input_layernorm.weight",checkpoint::Dtype::bf16,{hidden});
        shape(*shard,p+"post_attention_layernorm.weight",checkpoint::Dtype::bf16,{hidden});
        shape(*shard,p+"self_attn.q_norm.weight",checkpoint::Dtype::bf16,{128});
        shape(*shard,p+"self_attn.k_norm.weight",checkpoint::Dtype::bf16,{128});
    }
    shard->check_unchanged();stop(cancel);shard_=std::move(shard);metadata_=metadata.h;metadata.h=0;
}
void H3TextEncoder::unload(){require(!executing_,"busy");shard_.reset();if(metadata_){resources_->released(metadata_);metadata_=0;}}
void H3TextEncoder::execute(std::span<const std::uint32_t> ids,const std::string& output,
    const std::atomic_bool& cancel,const Hook& hook) {
    stop(cancel);require(!executing_,"busy");require(loaded(),"h3_text_not_loaded");
    require(!ids.empty() && ids.size()<=max_tokens,"h3_text_token_limit");
    for(auto id:ids)require(id<vocab,"h3_text_token_id");
    Active active(executing_);const auto tokens=ids.size();
    constexpr std::size_t per_token=hidden*3+8192*2+1024*2+25600*2;
    constexpr std::size_t norms=hidden*2+128*2;
    Admission admitted(*resources_,(tokens*per_token+norms)*sizeof(float)+max_tokens*sizeof(double));
    auto memory=std::make_unique<float[]>(tokens*per_token+norms);
    std::size_t offset=0;
    auto span=[&](std::size_t width){auto result=std::span<float>(memory.get()+offset,tokens*width);offset+=tokens*width;return result;};
    auto current=span(hidden),normalized=span(hidden),projected=span(hidden),q=span(8192),k=span(1024),v=span(1024),attended=span(8192),gate=span(25600),up=span(25600);
    auto norm=std::span<float>(memory.get()+offset,hidden),post=std::span<float>(norm.data()+hidden,hidden),qn=std::span<float>(post.data()+hidden,128),kn=std::span<float>(qn.data()+128,128);
    shard_->check_unchanged();
    for(std::size_t t=0;t<tokens;++t)dense(*shard_,"model.embed_tokens.weight",current.subspan(t*hidden,hidden),*resources_,cancel,ids[t]*hidden);
    for(std::size_t layer=0;layer<layers;++layer) {
        stop(cancel);const auto p="model.layers."+std::to_string(layer)+".";
        dense(*shard_,p+"input_layernorm.weight",norm,*resources_,cancel);
        dense(*shard_,p+"post_attention_layernorm.weight",post,*resources_,cancel);
        dense(*shard_,p+"self_attn.q_norm.weight",qn,*resources_,cancel);
        dense(*shard_,p+"self_attn.k_norm.weight",kn,*resources_,cancel);
        rms(current,norm,normalized);
        auto project=[&](const char* name,std::span<const float> x,std::size_t in,std::size_t out,std::span<float> y){linear(*shard_,p+name,x,in,out,y,*resources_,cancel,hook,compute_.get());};
        project("self_attn.q_proj",normalized,hidden,8192,q);
        project("self_attn.k_proj",normalized,hidden,1024,k);
        project("self_attn.v_proj",normalized,hidden,1024,v);
        rms(q,qn,q);rms(k,kn,k);rope(q,64,tokens);rope(k,8,tokens);
        if(compute_)compute_->attention(q,k,v,tokens,64,8,128,true,attended,cancel,true);
        else attention(q,k,v,attended,tokens,cancel);
        project("self_attn.o_proj",attended,8192,hidden,projected);
        for(std::size_t i=0;i<current.size();++i)current[i]+=projected[i];
        rms(current,post,normalized);
        project("mlp.gate_proj",normalized,hidden,25600,gate);
        project("mlp.up_proj",normalized,hidden,25600,up);
        for(std::size_t i=0;i<gate.size();++i){const double a=gate[i];gate[i]=float(a/(1+std::exp(-a)))*up[i];}
        project("mlp.down_proj",gate,25600,hidden,projected);
        for(std::size_t i=0;i<current.size();++i){current[i]+=projected[i];require(std::isfinite(current[i]),"h3_text_nonfinite_output");}
        if(hook)hook("layer_completed",layer+1);
    }
    stop(cancel);shard_->check_unchanged();Output artifact(output);
    const auto header="KADAN_H3_CONDITIONING_F32_V1\n"+std::to_string(tokens)+" 5120\nF32LE\n";
    artifact.write(header.data(),header.size());artifact.write(current.data(),current.size_bytes());
    if(hook)hook("publish",tokens);
    stop(cancel);shard_->check_unchanged();artifact.publish(output);
}
}
