#include "kadan/image_vae.hpp"
#include "kadan/checkpoint.hpp"
#include "kadan/checkpoint_floats.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <map>
#ifdef KADAN_IMAGE_BLAS
#include <cblas.h>
#endif
namespace kadan::image {
namespace {
void need(bool b,const char* e){if(!b)throw std::runtime_error(e);}
void stop(const std::atomic_bool& c){need(!c.load(),"image_vae_cancelled");}
Footprint host(Resources& r,Bytes n){auto p=r.snapshot().capacity;std::fill(p.begin(),p.end(),0);p[0]=n;return p;}
struct Lease{Resources& r;Handle h;Lease(Resources& a,Bytes n):r(a),h(r.reserve(Workload::image,host(r,n))){}~Lease(){r.released(h);}};
struct Buffer{Lease lease;std::unique_ptr<float[]> data;std::size_t n;Buffer(Resources& r,std::size_t count,bool overwrite=false):lease(r,count*4),data(overwrite?std::make_unique_for_overwrite<float[]>(count):std::make_unique<float[]>(count)),n(count){}std::span<float> span(){return {data.get(),n};}};
struct Tensor{std::size_t c,h,w;Buffer values;Tensor(Resources& r,std::size_t channels,std::size_t height,std::size_t width):c(channels),h(height),w(width),values(r,c*h*w){}std::span<float> span(){return values.span();}};
using T=std::unique_ptr<Tensor>;
struct Busy{bool& b;Busy(bool& f):b(f){need(!b,"busy");b=true;}~Busy(){b=false;}};
struct Pin{Resources& r;Handle h;Pin(Resources& a,Handle id):r(a),h(id){r.pin(h);}~Pin(){r.unpin(h);}};
void finite(std::span<const float> x){for(float v:x)need(std::isfinite(v),"image_vae_nonfinite");}
float silu(float x){double v=x;return float(v>=0?v/(1+std::exp(-v)):v*std::exp(v)/(1+std::exp(v)));}
struct Spec{std::string name;std::vector<std::uint64_t> shape;std::size_t offset=0,count=1;};
std::vector<Spec> layout(VaeConfig d){std::vector<Spec> out;auto add=[&](std::string p,std::initializer_list<std::uint64_t> s){out.push_back({std::move(p),s});};auto conv=[&](std::string p,std::size_t in,std::size_t n,std::size_t k){add(p+".weight",{n,in,k,k});add(p+".bias",{n});};auto norm=[&](std::string p,std::size_t n,bool image=false){if(image)add(p+".gamma",{n,1,1});else add(p+".gamma",{n,1,1,1});};auto res=[&](std::string p,std::size_t in,std::size_t n){norm(p+".norm1",in);conv(p+".conv1",in,n,3);norm(p+".norm2",n);conv(p+".conv2",n,n,3);if(in!=n)conv(p+".conv_shortcut",in,n,1);};
 conv("post_quant_conv",d.latent,d.latent,1);conv("decoder.conv_in",d.latent,d.base*8,3);res("decoder.mid_block.resnets.0",d.base*8,d.base*8);norm("decoder.mid_block.attentions.0.norm",d.base*8,true);conv("decoder.mid_block.attentions.0.to_qkv",d.base*8,d.base*24,1);conv("decoder.mid_block.attentions.0.proj",d.base*8,d.base*8,1);res("decoder.mid_block.resnets.1",d.base*8,d.base*8);
 const std::array<std::size_t,6> dims{8,8,8,4,2,1};for(std::size_t i=0;i<5;++i){const auto p="decoder.up_blocks."+std::to_string(i);for(std::size_t j=0;j<=d.residuals;++j)res(p+".resnets."+std::to_string(j),d.base*dims[j?i+1:i],d.base*dims[i+1]);if(i<4)conv(p+".upsampler.resample.1",d.base*dims[i+1],d.base*dims[i+1],3);}
 norm("decoder.norm_out",d.base);conv("decoder.conv_out",d.base,d.channels,3);std::size_t at=0;for(auto& s:out){for(auto n:s.shape)s.count*=n;s.offset=at;at+=s.count;}return out;
}
}
struct VaeDecoder::Impl{
 std::shared_ptr<DenseCompute> compute;
 Resources& r;Lease metadata;VaeConfig d;std::vector<Spec> specs;std::map<std::string,std::size_t,std::less<>> names;std::unique_ptr<Buffer> weights;
 Impl(Resources& a,VaeConfig config,std::shared_ptr<DenseCompute> c):compute(std::move(c)),r(a),metadata(a,8*1024*1024),d(config),specs(layout(config)){for(std::size_t i=0;i<specs.size();++i)names.emplace(specs[i].name,i);}
 std::span<const float> weight(const std::string& n){const auto& s=specs.at(names.at(n));return {weights->data.get()+s.offset,s.count};}
 T make(std::size_t c,std::size_t h,std::size_t w){return std::make_unique<Tensor>(r,c,h,w);}
 T conv(const std::string& p,Tensor& x,const std::atomic_bool& cancel){const auto& spec=specs.at(names.at(p+".weight"));auto out=std::size_t(spec.shape[0]),kernel=std::size_t(spec.shape[2]),k=x.c*kernel*kernel,points=x.h*x.w;need(spec.shape[1]==x.c,"image_vae_conv_shape");auto weights_=weight(p+".weight"),bias=weight(p+".bias");auto y=make(out,x.h,x.w);constexpr std::size_t tile=64;Buffer columns(r,k*tile),result(r,out*tile);
  if(compute){
   for(std::size_t at=0;at<points;at+=tile){stop(cancel);const auto count=std::min(tile,points-at);
    for(std::size_t t=0;t<count;++t)for(std::size_t ic=0;ic<x.c;++ic)for(std::size_t ky=0;ky<kernel;++ky)for(std::size_t kx=0;kx<kernel;++kx){const auto py=std::ptrdiff_t((at+t)/x.w+ky)-std::ptrdiff_t(kernel/2),px=std::ptrdiff_t((at+t)%x.w+kx)-std::ptrdiff_t(kernel/2);columns.data[t*k+(ic*kernel+ky)*kernel+kx]=(py>=0&&px>=0&&py<std::ptrdiff_t(x.h)&&px<std::ptrdiff_t(x.w))?x.values.data[ic*points+std::size_t(py)*x.w+std::size_t(px)]:0;}
    compute->dense(weights_,bias,columns.span().first(count*k),k,out,result.span().first(count*out),cancel);
    for(std::size_t t=0;t<count;++t)for(std::size_t oc=0;oc<out;++oc)y->values.data[oc*points+at+t]=result.data[t*out+oc];
   }finite(y->span());return y;
  }
  for(std::size_t at=0;at<points;at+=tile){stop(cancel);auto count=std::min(tile,points-at);for(std::size_t ic=0;ic<x.c;++ic)for(std::size_t ky=0;ky<kernel;++ky)for(std::size_t kx=0;kx<kernel;++kx)for(std::size_t t=0;t<count;++t){auto py=std::ptrdiff_t((at+t)/x.w+ky)-std::ptrdiff_t(kernel/2),px=std::ptrdiff_t((at+t)%x.w+kx)-std::ptrdiff_t(kernel/2);columns.data[((ic*kernel+ky)*kernel+kx)*count+t]=(py>=0&&px>=0&&py<std::ptrdiff_t(x.h)&&px<std::ptrdiff_t(x.w))?x.values.data[ic*points+std::size_t(py)*x.w+std::size_t(px)]:0;}
#ifdef KADAN_IMAGE_BLAS
   cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,int(out),int(count),int(k),1,weights_.data(),int(k),columns.data.get(),int(count),0,result.data.get(),int(count));
#else
   for(std::size_t oc=0;oc<out;++oc){stop(cancel);for(std::size_t t=0;t<count;++t){float v=0;for(std::size_t j=0;j<k;++j)v+=weights_[oc*k+j]*columns.data[j*count+t];result.data[oc*count+t]=v;}}
#endif
   for(std::size_t oc=0;oc<out;++oc)for(std::size_t t=0;t<count;++t)y->values.data[oc*points+at+t]=result.data[oc*count+t]+bias[oc];
  }finite(y->span());return y;
 }
 T norm(const std::string& p,Tensor& x,bool activate,const std::atomic_bool& cancel){auto g=weight(p+".gamma");need(g.size()==x.c,"image_vae_norm_shape");auto y=make(x.c,x.h,x.w);auto points=x.h*x.w;for(std::size_t t=0;t<points;++t){if(t%64==0)stop(cancel);double sum=0;for(std::size_t c=0;c<x.c;++c)sum+=double(x.values.data[c*points+t])*x.values.data[c*points+t];float scale=float(std::sqrt(double(x.c))/std::max(std::sqrt(sum),1e-12));for(std::size_t c=0;c<x.c;++c){float v=x.values.data[c*points+t]*scale*g[c];y->values.data[c*points+t]=activate?silu(v):v;}}return y;}
 void add(Tensor& x,Tensor& y,const std::atomic_bool& cancel){need(x.c==y.c&&x.h==y.h&&x.w==y.w,"image_vae_add_shape");for(std::size_t i=0;i<x.values.n;++i){if(i%4096==0)stop(cancel);x.values.data[i]+=y.values.data[i];}finite(x.span());}
 T residual(const std::string& p,Tensor& x,const std::atomic_bool& cancel){auto n=norm(p+".norm1",x,true,cancel),a=conv(p+".conv1",*n,cancel);n.reset();n=norm(p+".norm2",*a,true,cancel);a.reset();auto y=conv(p+".conv2",*n,cancel);n.reset();if(x.c==y->c)add(*y,x,cancel);else{auto shortcut=conv(p+".conv_shortcut",x,cancel);add(*y,*shortcut,cancel);}return y;}
 T attention(Tensor& x,const std::atomic_bool& cancel){std::string p="decoder.mid_block.attentions.0";auto n=norm(p+".norm",x,false,cancel),qkv=conv(p+".to_qkv",*n,cancel);n.reset();auto mix=make(x.c,x.h,x.w);auto points=x.h*x.w;Buffer scores(r,points);bool accelerated=false;
 if(compute){Buffer packed(r,points*x.c*4);auto qs=packed.span().first(points*x.c),ks=packed.span().subspan(points*x.c,points*x.c),vs=packed.span().subspan(2*points*x.c,points*x.c),ys=packed.span().subspan(3*points*x.c);
  for(std::size_t t=0;t<points;++t){stop(cancel);for(std::size_t c=0;c<x.c;++c){qs[t*x.c+c]=qkv->values.data[c*points+t];ks[t*x.c+c]=qkv->values.data[(x.c+c)*points+t];vs[t*x.c+c]=qkv->values.data[(2*x.c+c)*points+t];}}
  accelerated=compute->attend(qs,ks,vs,points,points,1,1,x.c,0,0,0,ys,cancel);
  if(accelerated)for(std::size_t t=0;t<points;++t)for(std::size_t c=0;c<x.c;++c)mix->values.data[c*points+t]=ys[t*x.c+c];
 }
 if(!accelerated){for(std::size_t t=0;t<points;++t){stop(cancel);float maximum=-INFINITY;for(std::size_t s=0;s<points;++s){float v=0;for(std::size_t c=0;c<x.c;++c)v+=qkv->values.data[c*points+t]*qkv->values.data[(x.c+c)*points+s];scores.data[s]=v/std::sqrt(float(x.c));maximum=std::max(maximum,scores.data[s]);}double sum=0;for(std::size_t s=0;s<points;++s){scores.data[s]=std::exp(scores.data[s]-maximum);sum+=scores.data[s];}need(sum>0&&std::isfinite(sum),"image_vae_attention");for(std::size_t c=0;c<x.c;++c){float v=0;for(std::size_t s=0;s<points;++s)v+=float(scores.data[s]/sum)*qkv->values.data[(2*x.c+c)*points+s];mix->values.data[c*points+t]=v;}}}
 auto y=conv(p+".proj",*mix,cancel);add(*y,x,cancel);return y;}
 T upsample(Tensor& x,const std::atomic_bool& cancel){auto y=make(x.c,x.h*2,x.w*2);for(std::size_t c=0;c<x.c;++c){stop(cancel);for(std::size_t h=0;h<y->h;++h)for(std::size_t w=0;w<y->w;++w)y->values.data[(c*y->h+h)*y->w+w]=x.values.data[(c*x.h+h/2)*x.w+w/2];}return y;}
 void shortcut(Tensor& out,Tensor& in,std::size_t factor,const std::atomic_bool& cancel){auto repeats=out.c*factor*4/in.c;need(repeats>0&&out.c*factor*4%in.c==0,"image_vae_shortcut_shape");for(std::size_t c=0;c<out.c;++c){stop(cancel);for(std::size_t h=0;h<out.h;++h)for(std::size_t w=0;w<out.w;++w){auto ic=(c*factor*4+(factor-1)*4+(h%2)*2+w%2)/repeats;out.values.data[(c*out.h+h)*out.w+w]+=in.values.data[(ic*in.h+h/2)*in.w+w/2];}}finite(out.span());}
};
VaeDecoder::VaeDecoder(std::shared_ptr<Resources> r,std::shared_ptr<DenseCompute> compute):resources_(std::move(r)),compute_(std::move(compute)){need(bool(resources_),"image_resources");}
VaeDecoder::~VaeDecoder(){unload();}
void VaeDecoder::load(const char* root,const std::string& file,VaeConfig d,const std::atomic_bool& cancel,const Hook& hook){Busy active(busy_);stop(cancel);need(!model_,"image_vae_load_state");need(d.base>0&&d.base<=144&&d.latent>0&&d.latent<=64&&d.channels>0&&d.channels<=4&&d.residuals<=2,"image_vae_config");auto m=std::make_unique<Impl>(*resources_,d,compute_);Lease parser(*resources_,16*1024*1024);checkpoint::Shard shard(root,file,std::make_shared<checkpoint::MemoryBudget>(16*1024*1024),checkpoint::Limits{2*1024*1024,4096,4096});
 for(const auto& spec:m->specs){auto t=shard.tensor(spec.name);need((t.dtype==checkpoint::Dtype::bf16||t.dtype==checkpoint::Dtype::fp32)&&t.rank==spec.shape.size()&&std::equal(spec.shape.begin(),spec.shape.end(),t.shape.begin()),"image_vae_tensor_layout");}const auto& last=m->specs.back();m->weights=std::make_unique<Buffer>(*resources_,last.offset+last.count,true);Lease staging(*resources_,checkpoint::float_read_buffer_bytes);auto bytes=std::make_unique_for_overwrite<std::uint8_t[]>(checkpoint::float_read_buffer_bytes);
 std::size_t loaded=0; for(const auto& spec:m->specs){auto name=spec.name;std::size_t reported=0;checkpoint::read_floats(shard.tensor(name).dtype,{m->weights->data.get()+spec.offset,spec.count},{bytes.get(),checkpoint::float_read_buffer_bytes},cancel,[&](std::size_t offset,std::span<std::uint8_t> out){shard.read_tensor(name,offset,out);},[&](std::size_t n){if(hook && n>=reported+64*1024*1024){hook(loaded+n);reported=n;}});loaded+=shard.tensor(name).bytes;if(hook)hook(loaded);}shard.check_unchanged();stop(cancel);resources_->loaded(m->weights->lease.h);model_=std::move(m);
#ifdef KADAN_IMAGE_BLAS
 openblas_set_num_threads(1);
#endif
}
void VaeDecoder::unload(){need(!busy_,"busy");if(model_){resources_->begin_eviction(model_->weights->lease.h);model_.reset();}}
void VaeDecoder::decode(std::span<const float> input,std::size_t h,std::size_t w,std::span<float> output,const std::atomic_bool& cancel,const Hook& hook){Busy active(busy_);stop(cancel);need(bool(model_),"image_vae_not_loaded");auto& m=*model_;need(h>0&&w>0&&h<=192&&w<=192&&h*w<=18432&&input.size()==m.d.latent*h*w&&output.size()==m.d.channels*h*w*256,"image_vae_shape");finite(input);Pin pin(*resources_,m.weights->lease.h);auto x=m.make(m.d.latent,h,w);std::copy(input.begin(),input.end(),x->span().begin());x=m.conv("post_quant_conv",*x,cancel);x=m.conv("decoder.conv_in",*x,cancel);x=m.residual("decoder.mid_block.resnets.0",*x,cancel);x=m.attention(*x,cancel);x=m.residual("decoder.mid_block.resnets.1",*x,cancel);
 for(std::size_t i=0;i<5;++i){const auto p="decoder.up_blocks."+std::to_string(i);auto original=std::move(x);x=m.residual(p+".resnets.0",*original,cancel);for(std::size_t j=1;j<=m.d.residuals;++j)x=m.residual(p+".resnets."+std::to_string(j),*x,cancel);if(i<4){x=m.upsample(*x,cancel);x=m.conv(p+".upsampler.resample.1",*x,cancel);m.shortcut(*x,*original,i<3?2:1,cancel);}if(hook)hook(i);}
 x=m.norm("decoder.norm_out",*x,true,cancel);x=m.conv("decoder.conv_out",*x,cancel);stop(cancel);std::transform(x->span().begin(),x->span().end(),output.begin(),[](float v){return std::clamp(v,-1.f,1.f);});
}
}
