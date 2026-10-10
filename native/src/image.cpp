#include "kadan/image.hpp"
#include "kadan/checkpoint.hpp"
#include "kadan/checkpoint_floats.hpp"
#include <algorithm>
#include <bit>
#include <cmath>
#include <map>
#include <set>
#include <vector>
#ifdef KADAN_IMAGE_BLAS
#include <cblas.h>
#endif
namespace kadan::image {
namespace {
void need(bool b,const char* e){if(!b)throw std::runtime_error(e);}
void stop(const std::atomic_bool& c){need(!c.load(),"image_cancelled");}
Footprint host(Resources& r,Bytes n){auto p=r.snapshot().capacity;std::fill(p.begin(),p.end(),0);p[0]=n;return p;}
struct Lease{Resources& r;Handle h;Lease(Resources& a,Bytes n):r(a),h(r.reserve(Workload::image,host(r,n))){}~Lease(){r.released(h);}};
struct Buffer{Lease lease;std::unique_ptr<float[]> data;std::size_t n;Buffer(Resources& r,std::size_t count,bool overwrite=false):lease(r,count*4),data(overwrite?std::make_unique_for_overwrite<float[]>(count):std::make_unique<float[]>(count)),n(count){}std::span<float> span(){return {data.get(),n};}};
struct Busy{bool& b;Busy(bool& flag):b(flag){need(!b,"busy");b=true;}~Busy(){b=false;}};
struct Pin{Resources& r;Handle h;Pin(Resources& a,Handle id):r(a),h(id){r.pin(h);}~Pin(){r.unpin(h);}};
void finite(std::span<const float> x){for(float v:x)need(std::isfinite(v),"image_nonfinite");}
float silu(float x){double v=x;return float(v>=0?v/(1+std::exp(-v)):v*std::exp(v)/(1+std::exp(v)));}
struct Spec{std::string name;std::vector<std::uint64_t> shape;std::size_t offset=0,count=1,shard=0;WeightIdentity identity{};};
std::vector<Spec> layout(BlockConfig d){std::vector<Spec> out;auto add=[&](std::string p,std::initializer_list<std::uint64_t> shape){out.push_back({std::move(p),shape});};
 for(auto n:{"to_q","to_k","to_v","to_out.0"})add(std::string("attn.")+n+".weight",{d.state,d.state});
 for(auto n:{"norm_q","norm_k"})add(std::string("attn.")+n+".weight",{d.head_dim});
 for(auto n:{"proj","gate_layer"}){add(std::string("img_mlp.")+n+".weight",{d.intermediate,d.state});}
 add("img_mlp.out.weight",{d.state,d.intermediate});
 std::size_t offset=0;for(auto& s:out){for(auto n:s.shape)s.count*=n;s.offset=offset;offset+=s.count;}return out;
}
void validate(BlockConfig d){need(d.state>0&&d.state<=4096&&d.heads>0&&d.heads<=32&&d.head_dim>0&&d.head_dim<=128&&d.state==d.heads*d.head_dim&&d.intermediate>0&&d.intermediate<=12288,"image_config");std::size_t sum=0;for(auto a:d.axes){need(a%2==0,"image_rope_config");sum+=a;}need(sum==d.head_dim,"image_rope_config");}
}
struct TransformerBlock::Impl{
 std::shared_ptr<DenseCompute> compute;
 Lease metadata;BlockConfig d;std::vector<Spec> specs;std::map<std::string,std::size_t,std::less<>> names;std::unique_ptr<Buffer> weights;
 Impl(Resources& r,BlockConfig config,std::shared_ptr<DenseCompute> c):compute(std::move(c)),metadata(r,1024*1024),d(config),specs(layout(config)){for(std::size_t i=0;i<specs.size();++i)names.emplace(specs[i].name,i);}
 std::span<const float> weight(const std::string& n){const auto& s=specs.at(names.at(n));return {weights->data.get()+s.offset,s.count};}
 void linear(const std::string& name,std::span<const float> x,std::span<float> y,std::size_t rows,std::size_t in,std::size_t out,const std::atomic_bool& cancel){auto w=weight(name+".weight");need(w.size()==in*out&&x.size()==rows*in&&y.size()==rows*out,"image_projection_shape");
  if(compute){compute->dense_weight(specs.at(names.at(name+".weight")).identity,w,{},x,in,out,y,cancel);finite(y);return;}
#ifdef KADAN_IMAGE_BLAS
  openblas_set_num_threads(1);
  for(std::size_t at=0;at<rows;at+=16){stop(cancel);auto count=std::min(std::size_t(16),rows-at);cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasTrans,int(count),int(out),int(in),1,x.data()+at*in,int(in),w.data(),int(in),0,y.data()+at*out,int(out));}
#else
  for(std::size_t t=0;t<rows;++t)for(std::size_t o=0;o<out;++o){if(o%32==0)stop(cancel);float v=0;for(std::size_t c=0;c<in;++c)v+=x[t*in+c]*w[o*in+c];y[t*out+o]=v;}
#endif
  finite(y);}

 void norm_mod(std::span<const float> x,std::span<const float> modulation,std::span<float> y,std::size_t rows,std::size_t text,std::size_t stage,const std::atomic_bool& cancel){for(std::size_t t=0;t<rows;++t){stop(cancel);double mean=0,variance=0;for(std::size_t c=0;c<d.state;++c)mean+=x[t*d.state+c];mean/=d.state;for(std::size_t c=0;c<d.state;++c){double v=x[t*d.state+c]-mean;variance+=v*v;}float scale=float(1/std::sqrt(variance/d.state+1e-6));auto offset=(t<text?4*d.state:0)+stage*2*d.state;for(std::size_t c=0;c<d.state;++c)y[t*d.state+c]=float(x[t*d.state+c]-mean)*scale*(1+modulation[offset+c]);}finite(y);}
 void rotate(std::span<float> x,const std::string& name,std::size_t rows,std::size_t text,std::size_t height,std::size_t width,const std::atomic_bool& cancel){auto norm=weight(name+".weight");for(std::size_t t=0;t<rows;++t){stop(cancel);std::array<long,3> positions{long(t),long(t),long(t)};if(t>=text)positions={long(text),long((t-text)/width)-long(height-height/2),long((t-text)%width)-long(width-width/2)};
  for(std::size_t h=0;h<d.heads;++h){auto row=x.subspan(t*d.state+h*d.head_dim,d.head_dim);double variance=0;for(float v:row)variance+=double(v)*v;float scale=float(1/std::sqrt(variance/d.head_dim+1e-6));for(std::size_t c=0;c<d.head_dim;++c)row[c]*=scale*norm[c];std::size_t offset=0;
   for(std::size_t axis=0;axis<3;++axis){for(std::size_t c=0;c<d.axes[axis];c+=2){float angle=float(positions[axis])*std::pow(10000.f,-float(c)/float(d.axes[axis])),co=std::cos(angle),si=std::sin(angle),a=row[offset+c],b=row[offset+c+1];row[offset+c]=a*co-b*si;row[offset+c+1]=b*co+a*si;}offset+=d.axes[axis];}
  }
 }finite(x);}
};
TransformerBlock::TransformerBlock(std::shared_ptr<Resources> r,std::shared_ptr<DenseCompute> compute):resources_(std::move(r)),compute_(std::move(compute)){need(bool(resources_),"image_resources");}
TransformerBlock::~TransformerBlock(){unload();}
void TransformerBlock::load(const char* root,const std::string& filename,std::size_t block,BlockConfig d,const std::atomic_bool& cancel){const std::array<std::string,1> files{filename};load(root,files,block,d,cancel);}
void TransformerBlock::load(const char* root,std::span<const std::string> files,std::size_t block,BlockConfig d,const std::atomic_bool& cancel){Busy active(busy_);stop(cancel);need(!model_&&block<32,"image_load_state");validate(d);need(!files.empty()&&files.size()<=4&&std::set<std::string>(files.begin(),files.end()).size()==files.size(),"image_shards");auto m=std::make_unique<Impl>(*resources_,d,compute_);Lease parser(*resources_,files.size()*16*1024*1024);std::vector<std::unique_ptr<checkpoint::Shard>> shards;
 for(const auto& filename:files){shards.push_back(std::make_unique<checkpoint::Shard>(root,filename,std::make_shared<checkpoint::MemoryBudget>(16*1024*1024),checkpoint::Limits{2*1024*1024,4096,4096}));}
 auto prefix="transformer_blocks."+std::to_string(block)+".";
 for(auto& spec:m->specs){std::size_t found=shards.size();for(std::size_t i=0;i<shards.size();++i){try{shards[i]->tensor(prefix+spec.name);}catch(const std::invalid_argument& e){if(std::string(e.what())=="missing_tensor")continue;throw;}need(found==shards.size(),"image_duplicate_tensor");found=i;}need(found<shards.size(),"image_missing_tensor");spec.shard=found;spec.identity=shards[found]->tensor_identity(prefix+spec.name);auto t=shards[found]->tensor(prefix+spec.name);need((t.dtype==checkpoint::Dtype::bf16||t.dtype==checkpoint::Dtype::fp32)&&t.rank==spec.shape.size()&&std::equal(spec.shape.begin(),spec.shape.end(),t.shape.begin()),"image_tensor_layout");}
 const auto& last=m->specs.back();m->weights=std::make_unique<Buffer>(*resources_,last.offset+last.count,true);Lease staging(*resources_,checkpoint::float_read_buffer_bytes);auto bytes=std::make_unique_for_overwrite<std::uint8_t[]>(checkpoint::float_read_buffer_bytes);
 for(const auto& spec:m->specs){auto& shard=*shards[spec.shard];auto name=prefix+spec.name;checkpoint::read_floats(shard.tensor(name).dtype,{m->weights->data.get()+spec.offset,spec.count},{bytes.get(),checkpoint::float_read_buffer_bytes},cancel,[&](std::size_t offset,std::span<std::uint8_t> out){shard.read_tensor(name,offset,out);},[](std::size_t){});}
 for(const auto& shard:shards){shard->check_unchanged();}
 stop(cancel);resources_->loaded(m->weights->lease.h);model_=std::move(m);
}
void TransformerBlock::unload(){need(!busy_,"busy");if(model_){resources_->begin_eviction(model_->weights->lease.h);model_.reset();}}
void TransformerBlock::execute(std::span<const float> input,std::span<const float> modulation,std::size_t text,std::size_t height,std::size_t width,std::span<float> output,const std::atomic_bool& cancel,const Hook& hook){Busy active(busy_);stop(cancel);need(bool(model_),"image_not_loaded");auto& m=*model_;auto d=m.d;need(text>0&&text<=2048&&height>0&&height<=256&&width>0&&width<=256&&text+height*width<=32768,"image_token_layout");const auto rows=text+height*width;need(input.size()==rows*d.state&&output.size()==input.size()&&modulation.size()==8*d.state,"image_shape");finite(input);finite(modulation);Pin pin(*resources_,m.weights->lease.h);
 Buffer x(*resources_,input.size()),n(*resources_,input.size()),q(*resources_,input.size()),k(*resources_,input.size()),v(*resources_,input.size()),mix(*resources_,input.size()),delta(*resources_,input.size()),scores(*resources_,rows),gate(*resources_,rows*d.intermediate),up(*resources_,rows*d.intermediate);std::copy(input.begin(),input.end(),x.data.get());
 m.norm_mod(x.span(),modulation,n.span(),rows,text,0,cancel);for(auto item:{std::pair{"attn.to_q",q.span()},std::pair{"attn.to_k",k.span()},std::pair{"attn.to_v",v.span()}})m.linear(item.first,n.span(),item.second,rows,d.state,d.state,cancel);m.rotate(q.span(),"attn.norm_q",rows,text,height,width,cancel);m.rotate(k.span(),"attn.norm_k",rows,text,height,width,cancel);
 if(hook)hook("qkv");
 stop(cancel);
 if(!(m.compute&&m.compute->attend(q.span(),k.span(),v.span(),rows,rows,d.heads,d.heads,d.head_dim,text,0,0,mix.span(),cancel)))
 for(std::size_t t=0;t<rows;++t){const auto allowed=t<text?t+1:rows;for(std::size_t h=0;h<d.heads;++h){stop(cancel);float maximum=-INFINITY;for(std::size_t s=0;s<allowed;++s){float value=0;for(std::size_t c=0;c<d.head_dim;++c)value+=q.data[t*d.state+h*d.head_dim+c]*k.data[s*d.state+h*d.head_dim+c];scores.data[s]=value/std::sqrt(float(d.head_dim));maximum=std::max(maximum,scores.data[s]);}double sum=0;for(std::size_t s=0;s<allowed;++s){scores.data[s]=std::exp(scores.data[s]-maximum);sum+=scores.data[s];}need(sum>0&&std::isfinite(sum),"image_nonfinite");for(std::size_t c=0;c<d.head_dim;++c){float value=0;for(std::size_t s=0;s<allowed;++s)value+=float(scores.data[s]/sum)*v.data[s*d.state+h*d.head_dim+c];mix.data[t*d.state+h*d.head_dim+c]=value;}}}
 m.linear("attn.to_out.0",mix.span(),delta.span(),rows,d.state,d.state,cancel);for(std::size_t t=0;t<rows;++t)for(std::size_t c=0;c<d.state;++c)x.data[t*d.state+c]+=std::tanh(modulation[(t<text?4*d.state:0)+d.state+c])*delta.data[t*d.state+c];if(hook)hook("attention");
 m.norm_mod(x.span(),modulation,n.span(),rows,text,1,cancel);m.linear("img_mlp.gate_layer",n.span(),gate.span(),rows,d.state,d.intermediate,cancel);m.linear("img_mlp.proj",n.span(),up.span(),rows,d.state,d.intermediate,cancel);for(std::size_t i=0;i<gate.n;++i)gate.data[i]=silu(gate.data[i])*up.data[i];m.linear("img_mlp.out",gate.span(),delta.span(),rows,d.intermediate,d.state,cancel);
 for(std::size_t t=0;t<rows;++t)for(std::size_t c=0;c<d.state;++c)x.data[t*d.state+c]+=std::tanh(modulation[(t<text?4*d.state:0)+3*d.state+c])*delta.data[t*d.state+c];
 finite(x.span());if(hook)hook("feed_forward");stop(cancel);std::copy(x.span().begin(),x.span().end(),output.begin());
}
}
