#include "kadan/image_text.hpp"
#include "kadan/checkpoint.hpp"
#include "kadan/checkpoint_floats.hpp"
#include <algorithm>
#include <bit>
#include <cmath>
#include <map>
#include <set>
#ifdef KADAN_IMAGE_BLAS
#include <cblas.h>
#endif
namespace kadan::image {
namespace {
void need(bool b,const char* e){if(!b)throw std::runtime_error(e);}
void stop(const std::atomic_bool& c){need(!c.load(),"image_text_cancelled");}
Footprint host(Resources& r,Bytes n){auto p=r.snapshot().capacity;std::fill(p.begin(),p.end(),0);p[0]=n;return p;}
struct Lease{Resources& r;Handle h;Lease(Resources& a,Bytes n):r(a),h(r.reserve(Workload::image,host(r,n))){}~Lease(){r.released(h);}};
struct Buffer{Lease lease;std::unique_ptr<float[]> data;std::size_t n;Buffer(Resources& r,std::size_t count,bool overwrite=false):lease(r,count*4),data(overwrite?std::make_unique_for_overwrite<float[]>(count):std::make_unique<float[]>(count)),n(count){}std::span<float> span(){return {data.get(),n};}};
struct Busy{bool& b;Busy(bool& flag):b(flag){need(!b,"busy");b=true;}~Busy(){b=false;}};
struct Pin{Resources& r;Handle h;Pin(Resources& a,Handle id):r(a),h(id){r.pin(h);}~Pin(){r.unpin(h);}};
void finite(std::span<const float> x){for(float v:x)need(std::isfinite(v),"image_text_nonfinite");}
float silu(float x){double v=x;return float(v>=0?v/(1+std::exp(-v)):v*std::exp(v)/(1+std::exp(v)));}
struct Spec{std::string name;std::vector<std::uint64_t> shape;std::size_t offset=0,count=1,shard=0;};
std::vector<Spec> layout(TextConfig d){std::vector<Spec> out;auto add=[&](std::string p,std::initializer_list<std::uint64_t> shape){out.push_back({std::move(p),shape});};add("embed_tokens.weight",{d.vocabulary,d.state});for(std::size_t i=0;i<d.layers;++i){const auto p="layers."+std::to_string(i)+".";for(auto n:{"input_layernorm","post_attention_layernorm"})add(p+n+".weight",{d.state});for(auto n:{"q_norm","k_norm"})add(p+"self_attn."+n+".weight",{d.head_dim});add(p+"self_attn.q_proj.weight",{d.state,d.state});for(auto n:{"k_proj","v_proj"})add(p+"self_attn."+n+".weight",{d.kv_heads*d.head_dim,d.state});add(p+"self_attn.o_proj.weight",{d.state,d.state});for(auto n:{"gate_proj","up_proj"})add(p+"mlp."+n+".weight",{d.intermediate,d.state});add(p+"mlp.down_proj.weight",{d.state,d.intermediate});}
 std::size_t at=0;for(auto& s:out){for(auto n:s.shape)s.count*=n;s.offset=at;at+=s.count;}return out;
}
}
struct TextEncoder::Impl{
 std::shared_ptr<DenseCompute> compute;
 Lease metadata;TextConfig d;std::vector<Spec> specs;std::map<std::string,std::size_t,std::less<>> names;std::unique_ptr<Buffer> weights;
 Impl(Resources& r,TextConfig config,std::shared_ptr<DenseCompute> c):compute(std::move(c)),metadata(r,16*1024*1024),d(config),specs(layout(config)){for(std::size_t i=0;i<specs.size();++i)names.emplace(specs[i].name,i);}
 std::span<const float> weight(const std::string& n){const auto& s=specs.at(names.at(n));return {weights->data.get()+s.offset,s.count};}
 void linear(const std::string& name,std::span<const float> x,std::span<float> y,std::size_t rows,std::size_t in,std::size_t out,const std::atomic_bool& cancel){auto w=weight(name+".weight");need(w.size()==in*out&&x.size()==rows*in&&y.size()==rows*out,"image_text_projection_shape");
  if(compute){compute->dense(w,{},x,in,out,y,cancel);finite(y);return;}
#ifdef KADAN_IMAGE_BLAS
  openblas_set_num_threads(1);
  for(std::size_t at=0;at<rows;at+=16){stop(cancel);auto count=std::min(std::size_t(16),rows-at);cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasTrans,int(count),int(out),int(in),1,x.data()+at*in,int(in),w.data(),int(in),0,y.data()+at*out,int(out));}
#else
  for(std::size_t t=0;t<rows;++t)for(std::size_t o=0;o<out;++o){if(o%32==0)stop(cancel);float v=0;for(std::size_t c=0;c<in;++c)v+=x[t*in+c]*w[o*in+c];y[t*out+o]=v;}
#endif
  finite(y);}

 void norm(const std::string& name,std::span<const float> x,std::span<float> y,std::size_t rows,std::size_t width,const std::atomic_bool& cancel){auto w=weight(name+".weight");need(w.size()==width&&x.size()==rows*width&&y.size()==x.size(),"image_text_norm_shape");for(std::size_t t=0;t<rows;++t){stop(cancel);double variance=0;for(std::size_t c=0;c<width;++c)variance+=double(x[t*width+c])*x[t*width+c];float scale=float(1/std::sqrt(variance/width+1e-6));for(std::size_t c=0;c<width;++c)y[t*width+c]=x[t*width+c]*scale*w[c];}finite(y);}
 void rotate(std::span<float> x,std::size_t rows,std::size_t heads,const std::atomic_bool& cancel){for(std::size_t t=0;t<rows;++t){stop(cancel);for(std::size_t h=0;h<heads;++h){auto row=x.subspan((t*heads+h)*d.head_dim,d.head_dim);for(std::size_t c=0;c<d.head_dim/2;++c){float angle=float(t)*std::pow(5000000.f,-2.f*float(c)/float(d.head_dim)),co=std::cos(angle),si=std::sin(angle),a=row[c],b=row[c+d.head_dim/2];row[c]=a*co-b*si;row[c+d.head_dim/2]=b*co+a*si;}}}}
};
TextEncoder::TextEncoder(std::shared_ptr<Resources> r,std::shared_ptr<DenseCompute> compute):resources_(std::move(r)),compute_(std::move(compute)){need(bool(resources_),"image_resources");}
TextEncoder::~TextEncoder(){unload();}
void TextEncoder::load(const char* root,std::span<const std::string> files,TextConfig d,const std::atomic_bool& cancel,const Hook& hook){Busy active(busy_);stop(cancel);need(!model_,"image_text_load_state");need(d.vocabulary>0&&d.vocabulary<=151936&&d.state>0&&d.state<=4096&&d.layers>0&&d.layers<=36&&d.heads>0&&d.heads<=32&&d.kv_heads>0&&d.heads%d.kv_heads==0&&d.head_dim>0&&d.head_dim<=128&&d.head_dim%2==0&&d.state==d.heads*d.head_dim&&d.intermediate>0&&d.intermediate<=12288,"image_text_config");need(!files.empty()&&files.size()<=4&&std::set<std::string>(files.begin(),files.end()).size()==files.size(),"image_text_shards");auto m=std::make_unique<Impl>(*resources_,d,compute_);Lease parser(*resources_,files.size()*16*1024*1024);std::vector<std::unique_ptr<checkpoint::Shard>> shards;
 for(const auto& file:files)shards.push_back(std::make_unique<checkpoint::Shard>(root,file,std::make_shared<checkpoint::MemoryBudget>(16*1024*1024),checkpoint::Limits{2*1024*1024,4096,4096}));
 for(auto& spec:m->specs){auto name="model.language_model."+spec.name;std::size_t found=shards.size();for(std::size_t i=0;i<shards.size();++i){try{shards[i]->tensor(name);}catch(const std::invalid_argument& e){if(std::string(e.what())=="missing_tensor")continue;throw;}need(found==shards.size(),"image_text_duplicate_tensor");found=i;}need(found<shards.size(),"image_text_missing_tensor");spec.shard=found;auto t=shards[found]->tensor(name);need((t.dtype==checkpoint::Dtype::bf16||t.dtype==checkpoint::Dtype::fp32)&&t.rank==spec.shape.size()&&std::equal(spec.shape.begin(),spec.shape.end(),t.shape.begin()),"image_text_tensor_layout");}
 const auto& last=m->specs.back();m->weights=std::make_unique<Buffer>(*resources_,last.offset+last.count,true);Lease staging(*resources_,checkpoint::float_read_buffer_bytes);auto bytes=std::make_unique_for_overwrite<std::uint8_t[]>(checkpoint::float_read_buffer_bytes);
 std::size_t loaded=0; for(const auto& spec:m->specs){auto& shard=*shards[spec.shard];auto name="model.language_model."+spec.name;std::size_t reported=0;checkpoint::read_floats(shard.tensor(name).dtype,{m->weights->data.get()+spec.offset,spec.count},{bytes.get(),checkpoint::float_read_buffer_bytes},cancel,[&](std::size_t offset,std::span<std::uint8_t> out){shard.read_tensor(name,offset,out);},[&](std::size_t n){if(hook && n>=reported+64*1024*1024){hook(loaded+n);reported=n;}});loaded+=shard.tensor(name).bytes;if(hook)hook(loaded);}
 for(const auto& shard:shards){shard->check_unchanged();}
 stop(cancel);resources_->loaded(m->weights->lease.h);model_=std::move(m);
}
void TextEncoder::unload(){need(!busy_,"busy");if(model_){resources_->begin_eviction(model_->weights->lease.h);model_.reset();}}
void TextEncoder::execute(std::span<const std::uint32_t> ids,std::span<float> output,const std::atomic_bool& cancel,const Hook& hook){Busy active(busy_);stop(cancel);need(bool(model_),"image_text_not_loaded");auto& m=*model_;auto d=m.d;need(!ids.empty()&&ids.size()<=2048&&output.size()==ids.size()*d.state,"image_text_shape");for(auto id:ids)need(id<d.vocabulary,"image_text_token_range");const auto rows=ids.size(),kv=d.kv_heads*d.head_dim;Pin pin(*resources_,m.weights->lease.h);
 Buffer x(*resources_,rows*d.state),n(*resources_,rows*d.state),q(*resources_,rows*d.state),k(*resources_,rows*kv),v(*resources_,rows*kv),mix(*resources_,rows*d.state),delta(*resources_,rows*d.state),gate(*resources_,rows*d.intermediate),up(*resources_,rows*d.intermediate),scores(*resources_,rows);auto embedding=m.weight("embed_tokens.weight");for(std::size_t t=0;t<rows;++t)std::copy_n(embedding.data()+ids[t]*d.state,d.state,x.data.get()+t*d.state);
 for(std::size_t layer=0;layer<d.layers;++layer){stop(cancel);const auto p="layers."+std::to_string(layer)+".";m.norm(p+"input_layernorm",x.span(),n.span(),rows,d.state,cancel);m.linear(p+"self_attn.q_proj",n.span(),q.span(),rows,d.state,d.state,cancel);m.linear(p+"self_attn.k_proj",n.span(),k.span(),rows,d.state,kv,cancel);m.linear(p+"self_attn.v_proj",n.span(),v.span(),rows,d.state,kv,cancel);m.norm(p+"self_attn.q_norm",q.span(),q.span(),rows*d.heads,d.head_dim,cancel);m.norm(p+"self_attn.k_norm",k.span(),k.span(),rows*d.kv_heads,d.head_dim,cancel);m.rotate(q.span(),rows,d.heads,cancel);m.rotate(k.span(),rows,d.kv_heads,cancel);
  if(!(m.compute&&m.compute->attend(q.span(),k.span(),v.span(),rows,rows,d.heads,d.kv_heads,d.head_dim,rows,0,0,mix.span(),cancel)))
  for(std::size_t t=0;t<rows;++t)for(std::size_t h=0;h<d.heads;++h){stop(cancel);const auto kh=h/(d.heads/d.kv_heads);float maximum=-INFINITY;for(std::size_t s=0;s<=t;++s){float value=0;for(std::size_t c=0;c<d.head_dim;++c)value+=q.data[t*d.state+h*d.head_dim+c]*k.data[s*kv+kh*d.head_dim+c];scores.data[s]=value/std::sqrt(float(d.head_dim));maximum=std::max(maximum,scores.data[s]);}double sum=0;for(std::size_t s=0;s<=t;++s){scores.data[s]=std::exp(scores.data[s]-maximum);sum+=scores.data[s];}need(sum>0&&std::isfinite(sum),"image_text_nonfinite");for(std::size_t c=0;c<d.head_dim;++c){float value=0;for(std::size_t s=0;s<=t;++s)value+=float(scores.data[s]/sum)*v.data[s*kv+kh*d.head_dim+c];mix.data[t*d.state+h*d.head_dim+c]=value;}}
  m.linear(p+"self_attn.o_proj",mix.span(),delta.span(),rows,d.state,d.state,cancel);for(std::size_t i=0;i<x.n;++i)x.data[i]+=delta.data[i];m.norm(p+"post_attention_layernorm",x.span(),n.span(),rows,d.state,cancel);m.linear(p+"mlp.gate_proj",n.span(),gate.span(),rows,d.state,d.intermediate,cancel);m.linear(p+"mlp.up_proj",n.span(),up.span(),rows,d.state,d.intermediate,cancel);for(std::size_t i=0;i<gate.n;++i)gate.data[i]=silu(gate.data[i])*up.data[i];m.linear(p+"mlp.down_proj",gate.span(),delta.span(),rows,d.intermediate,d.state,cancel);for(std::size_t i=0;i<x.n;++i)x.data[i]+=delta.data[i];finite(x.span());if(hook)hook(layer);
 }
 stop(cancel);std::copy(x.span().begin(),x.span().end(),output.begin());
}
}
