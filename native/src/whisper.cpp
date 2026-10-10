#include "kadan/whisper.hpp"
#include "kadan/checkpoint.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <limits>
#include <map>
#include <vector>
#ifdef KADAN_WHISPER_BLAS
#include <cblas.h>
#endif
namespace kadan::stt {
namespace {
void require(bool b,const char* e){if(!b)throw std::runtime_error(e);}
void stop(const std::atomic_bool& c){require(!c.load(),"whisper_cancelled");}
Footprint host(Resources& r,Bytes n){auto p=r.snapshot().capacity;std::fill(p.begin(),p.end(),0);p[0]=n;return p;}
struct Lease {
    Resources& r;Handle h;
    Lease(Resources& r_,Bytes n):r(r_),h(r.reserve(Workload::speech,host(r,n))){}
    ~Lease(){if(h)r.released(h);}
};
struct Buffer {
    Lease lease;std::unique_ptr<float[]> data;std::size_t size;
    Buffer(Resources& r,std::size_t n):lease(r,n*sizeof(float)),data(std::make_unique<float[]>(n)),size(n){}
    std::span<float> span(){return {data.get(),size};}
};
struct Active {bool& busy;explicit Active(bool& b):busy(b){require(!b,"busy");b=true;}~Active(){busy=false;}};
struct Pin {Resources& r;Handle h;Pin(Resources& r_,Handle h_):r(r_),h(h_){r.pin(h);}~Pin(){r.unpin(h);}};
void finite(std::span<const float> x){for(float v:x)require(std::isfinite(v),"whisper_nonfinite");}
bool overlap(std::span<const float> a,std::span<float> b){auto x=reinterpret_cast<std::uintptr_t>(a.data()),y=reinterpret_cast<std::uintptr_t>(b.data());return x<y ? y-x<a.size_bytes() : x-y<b.size_bytes();}
float gelu(float x){return .5f*x*(1.f+std::erf(x*float(1/std::sqrt(2.))));}
float half(std::uint16_t b){const auto sign=b>>15,exponent=(b>>10)&31,mantissa=b&1023;require(exponent!=31,"whisper_nonfinite_weight");float v=exponent ? std::ldexp(float(1024+mantissa),int(exponent)-25):std::ldexp(float(mantissa),-24);return sign ? -v:v;}
struct Spec {std::string name;std::vector<std::uint64_t> shape;std::size_t count=1,offset=0;};
std::vector<Spec> specs(WhisperDimensions d){
    std::vector<Spec> s;
    auto add=[&](std::string name,std::initializer_list<std::uint64_t> shape){s.push_back({std::move(name),shape});};
    auto norm=[&](const std::string& p,std::size_t n){add(p+".weight",{n});add(p+".bias",{n});};
    auto linear=[&](const std::string& p,std::size_t in,std::size_t out,bool bias=true){add(p+".weight",{out,in});if(bias)add(p+".bias",{out});};
    auto attention=[&](const std::string& p,std::size_t n){linear(p+".query",n,n);linear(p+".key",n,n,false);linear(p+".value",n,n);linear(p+".out",n,n);};
    add("encoder.conv1.weight",{d.audio_state,d.mels,3});add("encoder.conv1.bias",{d.audio_state});
    add("encoder.conv2.weight",{d.audio_state,d.audio_state,3});add("encoder.conv2.bias",{d.audio_state});
    add("encoder.positional_embedding",{d.audio_context,d.audio_state});
    add("decoder.token_embedding.weight",{d.vocabulary,d.text_state});add("decoder.positional_embedding",{d.text_context,d.text_state});
    for(bool decoder:{false,true}){
        const auto n=decoder?d.text_state:d.audio_state,layers=decoder?d.text_layers:d.audio_layers;
        const std::string root=decoder?"decoder":"encoder";
        for(std::size_t i=0;i<layers;++i){const auto p=root+".blocks."+std::to_string(i);
            norm(p+".attn_ln",n);attention(p+".attn",n);
            if(decoder){norm(p+".cross_attn_ln",n);attention(p+".cross_attn",n);}
            norm(p+".mlp_ln",n);linear(p+".mlp.0",n,4*n);linear(p+".mlp.2",4*n,n);
        }
        norm(root+(decoder?".ln":".ln_post"),n);
    }
    std::size_t offset=0;for(auto& v:s){for(auto n:v.shape)v.count*=n;v.offset=offset;offset+=v.count;}
    return s;
}
void validate(WhisperDimensions d){
    require(d.mels>0&&d.mels<=128&&d.audio_context>0&&d.audio_context<=1500&&d.text_context>0&&d.text_context<=448,"whisper_dimensions");
    require(d.audio_state>0&&d.audio_state<=1280&&d.text_state==d.audio_state&&d.audio_heads>0&&d.audio_heads<=32&&d.text_heads>0&&d.text_heads<=32,"whisper_dimensions");
    require(d.audio_state%d.audio_heads==0&&d.text_state%d.text_heads==0&&d.audio_layers>0&&d.audio_layers<=32&&d.text_layers>0&&d.text_layers<=32&&d.vocabulary>0&&d.vocabulary<=52000,"whisper_dimensions");
}
}
struct Whisper::Impl {
    Lease metadata;WhisperDimensions d;std::vector<Spec> layout;std::map<std::string,std::size_t,std::less<>> index;
    std::unique_ptr<Buffer> weights;
    Impl(Resources& r,WhisperDimensions dims):metadata(r,8*1024*1024),d(dims),layout(specs(dims)){
        for(std::size_t i=0;i<layout.size();++i)index.emplace(layout[i].name,i);
    }
    std::span<const float> w(const std::string& name)const{const auto& s=layout.at(index.at(name));return {weights->data.get()+s.offset,s.count};}
    void linear(const std::string& p,std::span<const float> x,std::span<float> y,std::size_t in,std::size_t out,const std::atomic_bool& cancel,bool bias=true){
        const auto matrix=w(p+".weight");const auto b=bias?w(p+".bias"):std::span<const float>{};
#ifdef KADAN_WHISPER_BLAS
        for(std::size_t row=0;row<x.size()/in;row+=16){
            stop(cancel);const auto count=std::min<std::size_t>(16,x.size()/in-row);
            cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasTrans,int(count),int(out),int(in),1,x.data()+row*in,int(in),matrix.data(),int(in),0,y.data()+row*out,int(out));
            if(bias)for(std::size_t i=0;i<count;++i)for(std::size_t o=0;o<out;++o)y[(row+i)*out+o]+=b[o];
        }
#else
        for(std::size_t row=0;row<x.size()/in;++row)for(std::size_t o=0;o<out;++o){if(o%32==0)stop(cancel);float v=bias?b[o]:0;
            for(std::size_t c=0;c<in;++c)v+=x[row*in+c]*matrix[o*in+c];
            y[row*out+o]=v;}
#endif
        finite(y);
    }
    void norm(const std::string& p,std::span<const float> x,std::span<float> y,std::size_t n,const std::atomic_bool& cancel){
        const auto a=w(p+".weight"),b=w(p+".bias");
        for(std::size_t row=0;row<x.size()/n;++row){stop(cancel);double sum=0;for(std::size_t c=0;c<n;++c)sum+=x[row*n+c];double mean=sum/n,var=0;
            for(std::size_t c=0;c<n;++c){double v=x[row*n+c]-mean;var+=v*v;}
            const float scale=float(1/std::sqrt(var/n+1e-5));for(std::size_t c=0;c<n;++c)y[row*n+c]=(x[row*n+c]-float(mean))*scale*a[c]+b[c];}
        finite(y);
    }
    void attention(Resources& r,const std::string& p,std::span<const float> x,std::span<const float> source,std::span<float> output,std::size_t n,std::size_t heads,bool causal,const std::atomic_bool& cancel){
        const auto rows=x.size()/n,keys=source.size()/n,width=n/heads;
        Buffer q(r,x.size()),k(r,source.size()),v(r,source.size()),mixed(r,x.size()),scores(r,keys);
        linear(p+".query",x,q.span(),n,n,cancel);linear(p+".key",source,k.span(),n,n,cancel,false);linear(p+".value",source,v.span(),n,n,cancel);
        const float scale=1/std::sqrt(float(width));
        for(std::size_t row=0;row<rows;++row)for(std::size_t h=0;h<heads;++h){stop(cancel);float maximum=-std::numeric_limits<float>::infinity();const auto limit=causal?row+1:keys;
            for(std::size_t j=0;j<limit;++j){float score=0;for(std::size_t c=0;c<width;++c)score+=q.data[row*n+h*width+c]*k.data[j*n+h*width+c];score*=scale;scores.data[j]=score;maximum=std::max(maximum,score);}
            double total=0;for(std::size_t j=0;j<limit;++j){scores.data[j]=std::exp(scores.data[j]-maximum);total+=scores.data[j];}
            require(std::isfinite(total)&&total>0,"whisper_nonfinite");
            for(std::size_t c=0;c<width;++c){float value=0;for(std::size_t j=0;j<limit;++j)value+=float(scores.data[j]/total)*v.data[j*n+h*width+c];mixed.data[row*n+h*width+c]=value;}
        }
        linear(p+".out",mixed.span(),output,n,n,cancel);
    }
    void block(Resources& r,const std::string& p,std::span<float> x,std::span<const float> audio,std::size_t n,std::size_t heads,const std::atomic_bool& cancel){
        Buffer normalized(r,x.size()),delta(r,x.size());norm(p+".attn_ln",x,normalized.span(),n,cancel);
        attention(r,p+".attn",normalized.span(),normalized.span(),delta.span(),n,heads,!audio.empty(),cancel);
        for(std::size_t i=0;i<x.size();++i)x[i]+=delta.data[i];
        if(!audio.empty()){norm(p+".cross_attn_ln",x,normalized.span(),n,cancel);attention(r,p+".cross_attn",normalized.span(),audio,delta.span(),n,heads,false,cancel);for(std::size_t i=0;i<x.size();++i)x[i]+=delta.data[i];}
        norm(p+".mlp_ln",x,normalized.span(),n,cancel);Buffer expanded(r,4*x.size());linear(p+".mlp.0",normalized.span(),expanded.span(),n,4*n,cancel);
        for(auto& value:expanded.span())value=gelu(value);
        linear(p+".mlp.2",expanded.span(),delta.span(),4*n,n,cancel);for(std::size_t i=0;i<x.size();++i)x[i]+=delta.data[i];finite(x);
    }
    void logits(Resources& r,std::span<const std::uint32_t> tokens,std::span<const float> audio,std::span<float> output,const std::atomic_bool& cancel,const Hook& hook){
        require(!tokens.empty()&&tokens.size()<=d.text_context&&output.size()==d.vocabulary&&audio.size()==d.audio_context*d.audio_state,"whisper_input_shape");
        finite(audio);for(auto id:tokens)require(id<d.vocabulary,"whisper_token_range");
        Buffer x(r,tokens.size()*d.text_state),normalized(r,x.size);const auto embedding=w("decoder.token_embedding.weight"),position=w("decoder.positional_embedding");
        for(std::size_t row=0;row<tokens.size();++row)for(std::size_t c=0;c<d.text_state;++c)x.data[row*d.text_state+c]=embedding[tokens[row]*d.text_state+c]+position[row*d.text_state+c];
        for(std::size_t i=0;i<d.text_layers;++i){stop(cancel);if(hook)hook("decoder",i);block(r,"decoder.blocks."+std::to_string(i),x.span(),audio,d.text_state,d.text_heads,cancel);}
        norm("decoder.ln",x.span(),normalized.span(),d.text_state,cancel);
        for(std::size_t id=0;id<d.vocabulary;++id){if(id%32==0)stop(cancel);float value=0;for(std::size_t c=0;c<d.text_state;++c)value+=normalized.data[(tokens.size()-1)*d.text_state+c]*embedding[id*d.text_state+c];output[id]=value;}finite(output);stop(cancel);
    }
};
Whisper::Whisper(std::shared_ptr<Resources> r):resources_(std::move(r)){require(bool(resources_),"whisper_resources_required");}
Whisper::~Whisper(){unload();}
bool Whisper::loaded()const{return bool(model_);}
void Whisper::load(const char* root,const std::string& name,WhisperDimensions dims,const std::atomic_bool& cancel){
    Active active(busy_);stop(cancel);require(!model_,"whisper_already_loaded");validate(dims);
#ifdef KADAN_WHISPER_BLAS
    require(openblas_get_num_threads()==1,"whisper_blas_thread_limit");
#endif
    auto m=std::make_unique<Impl>(*resources_,dims);Lease parser(*resources_,16*1024*1024);
    checkpoint::Shard shard(root,name,std::make_shared<checkpoint::MemoryBudget>(16*1024*1024),{4*1024*1024,4096,4096});
    require(shard.tensor_count()==m->layout.size(),"whisper_tensor_count");
    for(const auto& s:m->layout){const auto t=shard.tensor(s.name);require((t.dtype==checkpoint::Dtype::fp32||t.dtype==checkpoint::Dtype::fp16||t.dtype==checkpoint::Dtype::bf16)&&t.rank==s.shape.size()&&std::equal(s.shape.begin(),s.shape.end(),t.shape.begin()),"whisper_tensor_layout");}
    const auto& last=m->layout.back();m->weights=std::make_unique<Buffer>(*resources_,last.offset+last.count);
    Lease staging(*resources_,4096);std::array<std::uint8_t,4096> bytes;
    for(const auto& s:m->layout){const auto dtype=shard.tensor(s.name).dtype;const std::size_t width=dtype==checkpoint::Dtype::fp32?4:2;
        for(std::size_t at=0;at<s.count;){stop(cancel);auto n=std::min(bytes.size()/width,s.count-at);shard.read_tensor(s.name,at*width,{bytes.data(),n*width});
            for(std::size_t i=0;i<n;++i){std::uint32_t bits=0;for(std::size_t j=0;j<width;++j)bits|=std::uint32_t(bytes[i*width+j])<<(8*j);
                float value=dtype==checkpoint::Dtype::fp32?std::bit_cast<float>(bits):dtype==checkpoint::Dtype::bf16?std::bit_cast<float>(bits<<16):half(std::uint16_t(bits));
                require(std::isfinite(value),"whisper_nonfinite_weight");m->weights->data[s.offset+at+i]=value;}
            at+=n;}
    }
    shard.check_unchanged();stop(cancel);resources_->loaded(m->weights->lease.h);model_=std::move(m);
}
void Whisper::unload(){require(!busy_,"busy");if(!model_)return;resources_->begin_eviction(model_->weights->lease.h);model_.reset();}
void Whisper::encode(std::span<const float> mel,std::span<float> encoded,const std::atomic_bool& cancel,const Hook& hook){
    Active active(busy_);stop(cancel);require(bool(model_),"whisper_not_loaded");auto& m=*model_;auto d=m.d;const auto frames=2*d.audio_context;
    require(mel.size()==d.mels*frames&&encoded.size()==d.audio_context*d.audio_state,"whisper_input_shape");require(!overlap(mel,encoded),"whisper_buffer_overlap");finite(mel);Pin pin(*resources_,m.weights->lease.h);
    Buffer first(*resources_,frames*d.audio_state),second(*resources_,encoded.size());
    auto convolution=[&](const char* p,std::span<const float> x,std::span<float> y,std::size_t in,std::size_t time,std::size_t stride,bool band_major){
        const auto w=m.w(std::string(p)+".weight"),b=m.w(std::string(p)+".bias");const auto out_time=(time+stride-1)/stride;
        for(std::size_t t=0;t<out_time;++t){stop(cancel);for(std::size_t o=0;o<d.audio_state;++o){float value=b[o];for(std::size_t c=0;c<in;++c)for(int k=0;k<3;++k){const auto source=std::ptrdiff_t(t*stride)+k-1;if(source>=0&&source<std::ptrdiff_t(time))value+=x[band_major?c*time+source:source*in+c]*w[(o*in+c)*3+k];}y[t*d.audio_state+o]=gelu(value);}}
        finite(y);
    };
    if(hook)hook("convolution",0);
    convolution("encoder.conv1",mel,first.span(),d.mels,frames,1,true);
    if(hook)hook("convolution",1);
    convolution("encoder.conv2",first.span(),second.span(),d.audio_state,frames,2,false);
    auto position=m.w("encoder.positional_embedding");for(std::size_t i=0;i<second.size;++i)second.data[i]+=position[i];
    for(std::size_t i=0;i<d.audio_layers;++i){stop(cancel);if(hook)hook("encoder",i);m.block(*resources_,"encoder.blocks."+std::to_string(i),second.span(),{},d.audio_state,d.audio_heads,cancel);}
    m.norm("encoder.ln_post",second.span(),encoded,d.audio_state,cancel);stop(cancel);
}
void Whisper::decode(std::span<const std::uint32_t> tokens,std::span<const float> encoded,std::span<float> logits,const std::atomic_bool& cancel,const Hook& hook){
    Active active(busy_);stop(cancel);require(bool(model_),"whisper_not_loaded");require(!overlap(encoded,logits),"whisper_buffer_overlap");Pin pin(*resources_,model_->weights->lease.h);model_->logits(*resources_,tokens,encoded,logits,cancel,hook);
}
std::size_t Whisper::greedy(std::span<const std::uint32_t> prompt,std::span<const float> encoded,std::span<std::uint32_t> output,std::uint32_t eos,std::span<const std::uint32_t> suppressed,const std::atomic_bool& cancel,const Hook& hook,std::span<const std::uint32_t> first_suppressed){
    Active active(busy_);stop(cancel);require(bool(model_),"whisper_not_loaded");const auto d=model_->d;
    require(!prompt.empty()&&!output.empty()&&prompt.size()<=d.text_context&&output.size()<=d.text_context-prompt.size()+1&&eos<d.vocabulary,"whisper_generation_shape");
    auto source_address=reinterpret_cast<std::uintptr_t>(encoded.data());
    auto output_address=reinterpret_cast<std::uintptr_t>(output.data());
    require(source_address<output_address ? output_address-source_address>=encoded.size_bytes() : source_address-output_address>=output.size_bytes(),"whisper_buffer_overlap");
    for(auto id:suppressed)require(id<d.vocabulary,"whisper_token_range");
    for(auto id:first_suppressed)require(id<d.vocabulary,"whisper_token_range");
    Pin pin(*resources_,model_->weights->lease.h);Lease token_admission(*resources_,d.text_context*sizeof(std::uint32_t));std::vector<std::uint32_t> tokens;tokens.reserve(d.text_context);tokens.assign(prompt.begin(),prompt.end());Buffer logits(*resources_,d.vocabulary);
    for(std::size_t i=0;i<output.size();++i){model_->logits(*resources_,tokens,encoded,logits.span(),cancel,hook);for(auto id:suppressed)logits.data[id]=-std::numeric_limits<float>::infinity();
        if(i==0)for(auto id:first_suppressed)logits.data[id]=-std::numeric_limits<float>::infinity();
        const auto best=std::max_element(logits.data.get(),logits.data.get()+d.vocabulary);require(std::isfinite(*best),"whisper_all_tokens_suppressed");const auto id=std::uint32_t(best-logits.data.get());output[i]=id;if(hook)hook("token",i);stop(cancel);if(id==eos||i+1==output.size())return i+1;tokens.push_back(id);}
    return output.size();
}
}
