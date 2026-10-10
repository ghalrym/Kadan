#include "kadan/image_denoiser.hpp"
#include "kadan/checkpoint.hpp"
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
void stop(const std::atomic_bool& c){need(!c.load(),"image_cancelled");}
Footprint host(Resources& r,Bytes n){auto p=r.snapshot().capacity;std::fill(p.begin(),p.end(),0);p[0]=n;return p;}
struct Lease{Resources& r;Handle h;Lease(Resources& a,Bytes n):r(a),h(r.reserve(Workload::image,host(r,n))){}~Lease(){r.released(h);}};
struct Buffer{Lease lease;std::unique_ptr<float[]> data;std::size_t n;Buffer(Resources& r,std::size_t count):lease(r,count*4),data(std::make_unique<float[]>(count)),n(count){}std::span<float> span(){return {data.get(),n};}};
struct Busy{bool& b;Busy(bool& flag):b(flag){need(!b,"busy");b=true;}~Busy(){b=false;}};
struct Pin{Resources& r;Handle h;Pin(Resources& a,Handle id):r(a),h(id){r.pin(h);}~Pin(){r.unpin(h);}};
void finite(std::span<const float> x){for(float v:x)need(std::isfinite(v),"image_nonfinite");}
float silu(float x){double v=x;return float(v>=0?v/(1+std::exp(-v)):v*std::exp(v)/(1+std::exp(v)));}
struct Spec{std::string name;std::vector<std::uint64_t> shape;std::size_t offset=0,count=1,shard=0;};
std::vector<Spec> layout(DenoiserConfig d){std::vector<Spec> out;auto add=[&](std::string p,std::initializer_list<std::uint64_t> shape){out.push_back({std::move(p),shape});};const auto n=d.block.state;
 add("img_in.weight",{n,d.channels});add("txt_in.text_norm.weight",{d.context});add("txt_in.in_layer.weight",{n,d.context});add("txt_in.out_layer.weight",{n,n});add("time_text_embed.timestep_embedder.linear_1.weight",{n,256});add("time_text_embed.timestep_embedder.linear_2.weight",{n,n});add("modulation.1.weight",{4*n,n});add("norm_out.linear.weight",{n,n});add("proj_out.weight",{d.channels,n});
 std::size_t at=0;for(auto& s:out){for(auto n:s.shape)s.count*=n;s.offset=at;at+=s.count;}return out;
}
std::size_t locate(const std::vector<std::unique_ptr<checkpoint::Shard>>& shards,const std::string& name){std::size_t found=shards.size();for(std::size_t i=0;i<shards.size();++i){try{shards[i]->tensor(name);}catch(const std::invalid_argument& e){if(std::string(e.what())=="missing_tensor")continue;throw;}need(found==shards.size(),"image_duplicate_tensor");found=i;}need(found<shards.size(),"image_missing_tensor");return found;}
}
struct Denoiser::Impl{
 Lease metadata;DenoiserConfig d;std::vector<Spec> specs;std::map<std::string,std::size_t,std::less<>> names;std::unique_ptr<Buffer> weights;std::vector<std::unique_ptr<TransformerBlock>> blocks;
 Impl(Resources& r,DenoiserConfig config):metadata(r,16*1024*1024),d(config),specs(layout(config)){for(std::size_t i=0;i<specs.size();++i)names.emplace(specs[i].name,i);}
 std::span<const float> weight(const std::string& n){const auto& s=specs.at(names.at(n));return {weights->data.get()+s.offset,s.count};}
 void linear(const std::string& name,std::span<const float> x,std::span<float> y,std::size_t rows,std::size_t in,std::size_t out,const std::atomic_bool& cancel){auto w=weight(name+".weight");need(w.size()==in*out&&x.size()==rows*in&&y.size()==rows*out,"image_projection_shape");
#ifdef KADAN_IMAGE_BLAS
  openblas_set_num_threads(1);
  for(std::size_t at=0;at<rows;at+=16){stop(cancel);auto count=std::min(std::size_t(16),rows-at);cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasTrans,int(count),int(out),int(in),1,x.data()+at*in,int(in),w.data(),int(in),0,y.data()+at*out,int(out));}
#else
  for(std::size_t t=0;t<rows;++t)for(std::size_t o=0;o<out;++o){if(o%32==0)stop(cancel);float v=0;for(std::size_t c=0;c<in;++c)v+=x[t*in+c]*w[o*in+c];y[t*out+o]=v;}
#endif
  finite(y);}

};
Denoiser::Denoiser(std::shared_ptr<Resources> r):resources_(std::move(r)){need(bool(resources_),"image_resources");}
Denoiser::~Denoiser(){unload();}
void Denoiser::load(const char* root,std::span<const std::string> files,DenoiserConfig d,const std::atomic_bool& cancel){Busy active(busy_);stop(cancel);need(!model_,"image_load_state");need(d.layers>0&&d.layers<=32&&d.context>0&&d.context<=4096&&d.channels>0&&d.channels<=64&&d.block.state>0&&d.block.state<=4096&&files.size()>0&&files.size()<=4,"image_denoiser_config");need(std::set<std::string>(files.begin(),files.end()).size()==files.size(),"image_duplicate_shard");auto m=std::make_unique<Impl>(*resources_,d);Lease parser(*resources_,files.size()*16*1024*1024);std::vector<std::unique_ptr<checkpoint::Shard>> shards;
 for(const auto& file:files)shards.push_back(std::make_unique<checkpoint::Shard>(root,file,std::make_shared<checkpoint::MemoryBudget>(16*1024*1024),checkpoint::Limits{2*1024*1024,4096,4096}));
 for(auto& s:m->specs){s.shard=locate(shards,s.name);auto t=shards[s.shard]->tensor(s.name);need((t.dtype==checkpoint::Dtype::bf16||t.dtype==checkpoint::Dtype::fp32)&&t.rank==s.shape.size()&&std::equal(s.shape.begin(),s.shape.end(),t.shape.begin()),"image_tensor_layout");}
 const auto& last=m->specs.back();m->weights=std::make_unique<Buffer>(*resources_,last.offset+last.count);Lease staging(*resources_,4096);std::array<std::uint8_t,4096> bytes;
 for(const auto& s:m->specs){auto& shard=*shards[s.shard];auto type=shard.tensor(s.name).dtype;const std::size_t width=type==checkpoint::Dtype::bf16?2:4;for(std::size_t at=0;at<s.count;){stop(cancel);auto count=std::min(bytes.size()/width,s.count-at);shard.read_tensor(s.name,at*width,{bytes.data(),count*width});for(std::size_t i=0;i<count;++i){std::uint32_t bits=0;for(std::size_t j=0;j<width;++j)bits|=std::uint32_t(bytes[i*width+j])<<(8*j);float v=std::bit_cast<float>(width==2?bits<<16:bits);need(std::isfinite(v),"image_nonfinite_weight");m->weights->data[s.offset+at+i]=v;}at+=count;}}
 for(std::size_t i=0;i<d.layers;++i){stop(cancel);auto block=std::make_unique<TransformerBlock>(resources_);block->load(root,files,i,d.block,cancel);m->blocks.push_back(std::move(block));}
 for(const auto& shard:shards){shard->check_unchanged();}
 stop(cancel);resources_->loaded(m->weights->lease.h);model_=std::move(m);
}
void Denoiser::unload(){need(!busy_,"busy");if(model_){resources_->begin_eviction(model_->weights->lease.h);model_.reset();}}
void Denoiser::execute(std::span<const float> latent,std::span<const float> condition,std::size_t text,std::size_t height,std::size_t width,float timestep,std::span<float> velocity,const std::atomic_bool& cancel,const Hook& hook){Busy active(busy_);stop(cancel);need(bool(model_),"image_not_loaded");auto& m=*model_;auto d=m.d;const auto state=d.block.state;need(text>0&&text<=2048&&height>0&&height<=256&&width>0&&width<=256&&height*width%4==0&&text+height*width<=32768&&std::isfinite(timestep)&&timestep>=0&&timestep<=1,"image_denoiser_layout");const auto images=height*width,rows=text+images;need(latent.size()==images*d.channels&&velocity.size()==latent.size()&&condition.size()==text*d.context,"image_shape");finite(latent);finite(condition);Pin pin(*resources_,m.weights->lease.h);
 Buffer normalized(*resources_,text*d.context),projected(*resources_,text*state),x(*resources_,rows*state),next(*resources_,rows*state),frequency(*resources_,512),time1(*resources_,2*state),time2(*resources_,2*state),modulation(*resources_,8*state),scale(*resources_,state),final(*resources_,images*state),out(*resources_,velocity.size());
 auto norm=m.weight("txt_in.text_norm.weight");for(std::size_t t=0;t<text;++t){stop(cancel);double mean=0;for(std::size_t c=0;c<d.context;++c)mean+=double(condition[t*d.context+c])*condition[t*d.context+c];float s=float(1/std::sqrt(mean/d.context+1e-6));for(std::size_t c=0;c<d.context;++c)normalized.data[t*d.context+c]=condition[t*d.context+c]*s*(1+norm[c]);}
 m.linear("txt_in.in_layer",normalized.span(),projected.span(),text,d.context,state,cancel);for(float& v:projected.span())v=.5f*v*(1+std::tanh(.7978845608028654f*(v+.044715f*v*v*v)));m.linear("txt_in.out_layer",projected.span(),x.span().first(text*state),text,state,state,cancel);m.linear("img_in",latent,x.span().subspan(text*state),images,d.channels,state,cancel);
 for(std::size_t row=0;row<2;++row)for(std::size_t c=0;c<128;++c){float angle=(row?0.f:timestep*1000.f)*std::exp(-std::log(10000.f)*float(c)/128.f);frequency.data[row*256+c]=std::cos(angle);frequency.data[row*256+128+c]=std::sin(angle);}
 m.linear("time_text_embed.timestep_embedder.linear_1",frequency.span(),time1.span(),2,256,state,cancel);for(float& v:time1.span())v=silu(v);m.linear("time_text_embed.timestep_embedder.linear_2",time1.span(),time2.span(),2,state,state,cancel);for(float& v:time2.span())v=silu(v);m.linear("modulation.1",time2.span(),modulation.span(),2,state,4*state,cancel);if(hook)hook("conditioning",0);
 for(std::size_t i=0;i<d.layers;++i){m.blocks[i]->execute(x.span(),modulation.span(),text,height,width,next.span(),cancel);std::swap(x.data,next.data);if(hook)hook("block",i);stop(cancel);}
 m.linear("norm_out.linear",time2.span().first(state),scale.span(),1,state,state,cancel);
 for(std::size_t t=0;t<images;++t){stop(cancel);auto input=x.span().subspan((text+t)*state,state);double mean=0,variance=0;for(float v:input)mean+=v;mean/=state;for(float v:input)variance+=(v-mean)*(v-mean);float n=float(1/std::sqrt(variance/state+1e-6));for(std::size_t c=0;c<state;++c)final.data[t*state+c]=float(input[c]-mean)*n*(1+scale.data[c]);}
 m.linear("proj_out",final.span(),out.span(),images,state,d.channels,cancel);stop(cancel);std::copy(out.span().begin(),out.span().end(),velocity.begin());
}
}
