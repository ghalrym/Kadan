#include "kadan/h3_audio.hpp"
#include "kadan/checkpoint.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <map>
#include <vector>
#ifdef KADAN_H3_AUDIO_BLAS
#include <cblas.h>
#endif
namespace kadan::video {
namespace {
void need(bool b,const char* e){if(!b)throw std::runtime_error(e);}
void stop(const std::atomic_bool& c){need(!c.load(),"h3_audio_cancelled");}
Footprint host(Resources& r,Bytes n){auto f=r.snapshot().capacity;std::fill(f.begin(),f.end(),0);f[0]=n;return f;}
struct Lease{Resources& r;Handle h;Lease(Resources& a,Bytes n):r(a),h(r.reserve(Workload::video,host(r,n))){}~Lease(){r.released(h);}};
struct Matrix{Lease lease;std::size_t rows,cols;std::unique_ptr<float[]> data;Matrix(Resources& r,std::size_t t,std::size_t c):lease(r,t*c*4),rows(t),cols(c),data(std::make_unique<float[]>(t*c)){}std::span<float> span(){return {data.get(),rows*cols};}};
using M=std::unique_ptr<Matrix>;
M make(Resources& r,std::size_t t,std::size_t c){return std::make_unique<Matrix>(r,t,c);}
struct Active{bool& b;Active(bool& a):b(a){need(!b,"busy");b=true;}~Active(){b=false;}};
struct Pin{Resources& r;Handle h;Pin(Resources& a,Handle id):r(a),h(id){r.pin(h);}~Pin(){r.unpin(h);}};
void finite(std::span<const float> x){for(float v:x)need(std::isfinite(v),"h3_audio_nonfinite");}
struct Spec{std::string name;std::vector<std::uint64_t> shape;std::size_t count=1,offset=0;};
std::vector<Spec> layout(H3AudioConfig d){
 std::vector<Spec> s;auto add=[&](std::string p,std::initializer_list<std::uint64_t> a){s.push_back({std::move(p),a});};
 auto conv=[&](std::string p,std::size_t in,std::size_t out,std::size_t k,bool transpose=false,bool bias=true){add(p+".weight",{transpose?in:out,transpose?out:in,k});if(bias)add(p+".bias",{out});};
 auto act=[&](std::string p,std::size_t c){add(p+".act.alpha",{c});add(p+".act.beta",{c});add(p+".upsample.filter",{1,1,12});add(p+".downsample.lowpass.filter",{1,1,12});};
 add("latents_mean",{32});add("latents_std",{32});conv("dec_in_proj",32,d.projection,1);conv("decoder.conv_pre",d.projection,d.initial,7);
 for(std::size_t i=0;i<7;++i){const auto c=d.initial>>(i+1);conv("decoder.ups."+std::to_string(i)+".0",c*2,c,i<2?9:4,true);
  for(std::size_t j=0;j<3;++j){const auto p="decoder.resblocks."+std::to_string(i*3+j);const auto k=std::array<std::size_t,3>{3,7,11}[j];
   for(std::size_t a=0;a<3;++a){conv(p+".convs1."+std::to_string(a),c,c,k);conv(p+".convs2."+std::to_string(a),c,c,k);}
   for(std::size_t a=0;a<6;++a)act(p+".activations."+std::to_string(a),c);
  }
 }
 act("decoder.activation_post",d.initial/128);conv("decoder.conv_post",d.initial/128,1,7,false,false);
 std::size_t at=0;for(auto& x:s){for(auto n:x.shape)x.count*=n;x.offset=at;at+=x.count;}return s;
}
}
struct H3AudioDecoder::Impl{
 Resources& r;Lease metadata;H3AudioConfig d;std::shared_ptr<DenseCompute> compute;std::vector<Spec> specs;std::map<std::string,std::size_t,std::less<>> names;M weights;
 Impl(Resources& a,H3AudioConfig c,std::shared_ptr<DenseCompute> backend):r(a),metadata(a,2*1024*1024),d(c),compute(std::move(backend)),specs(layout(c)){for(std::size_t i=0;i<specs.size();++i)names.emplace(specs[i].name,i);}
 std::span<const float> w(const std::string& p){const auto& s=specs.at(names.at(p));return {weights->data.get()+s.offset,s.count};}
 void dense(std::span<const float> a,std::span<const float> b,std::span<const float> x,std::size_t in,std::size_t out,std::span<float> y,const std::atomic_bool& c){
  stop(c);if(compute){compute->dense(a,b,x,in,out,y,c);return;}
#ifdef KADAN_H3_AUDIO_BLAS
  cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasTrans,int(x.size()/in),int(out),int(in),1,x.data(),int(in),a.data(),int(in),0,y.data(),int(out));
  if(!b.empty())for(std::size_t i=0;i<y.size();++i)y[i]+=b[i%out];
#else
  for(std::size_t t=0;t<x.size()/in;++t){stop(c);for(std::size_t o=0;o<out;++o){float v=b.empty()?0:b[o];for(std::size_t k=0;k<in;++k)v+=x[t*in+k]*a[o*in+k];y[t*out+o]=v;}}
#endif
  stop(c);
 }
 M conv(const std::string& p,Matrix& x,std::size_t out,std::size_t kernel,std::size_t dilation,const std::atomic_bool& cancel,bool bias=true){
  auto y=make(r,x.rows,out),packed=make(r,32,x.cols*kernel);const auto a=w(p+".weight"),b=bias?w(p+".bias"):std::span<const float>{};const auto pad=(kernel-1)*dilation/2;
  for(std::size_t at=0;at<x.rows;at+=32){stop(cancel);const auto count=std::min<std::size_t>(32,x.rows-at);
   for(std::size_t t=0;t<count;++t)for(std::size_t c=0;c<x.cols;++c)for(std::size_t k=0;k<kernel;++k){const auto source=std::ptrdiff_t(at+t+k*dilation)-std::ptrdiff_t(pad);packed->data[(t*x.cols+c)*kernel+k]=source>=0&&source<std::ptrdiff_t(x.rows)?x.data[std::size_t(source)*x.cols+c]:0;}
   dense(a,b,packed->span().first(count*x.cols*kernel),x.cols*kernel,out,y->span().subspan(at*out,count*out),cancel);
  }finite(y->span());return y;
 }
 M up(const std::string& p,Matrix& x,std::size_t out,std::size_t rate,std::size_t kernel,const std::atomic_bool& cancel){
  auto y=make(r,x.rows*rate,out),transposed=make(r,out*kernel,x.cols),values=make(r,32,out*kernel);auto a=w(p+".weight"),b=w(p+".bias");const auto pad=(kernel-rate)/2;
  for(std::size_t t=0;t<y->rows;++t){if(t%256==0)stop(cancel);std::copy(b.begin(),b.end(),y->data.get()+t*out);}
  for(std::size_t o=0;o<out;++o){stop(cancel);for(std::size_t k=0;k<kernel;++k)for(std::size_t c=0;c<x.cols;++c)transposed->data[(o*kernel+k)*x.cols+c]=a[(c*out+o)*kernel+k];}
  for(std::size_t at=0;at<x.rows;at+=32){stop(cancel);const auto count=std::min<std::size_t>(32,x.rows-at);dense(transposed->span(),{},x.span().subspan(at*x.cols,count*x.cols),x.cols,out*kernel,values->span().first(count*out*kernel),cancel);
   for(std::size_t t=0;t<count;++t)for(std::size_t k=0;k<kernel;++k){const auto dest=std::ptrdiff_t((at+t)*rate+k)-std::ptrdiff_t(pad);if(dest>=0&&dest<std::ptrdiff_t(y->rows))for(std::size_t o=0;o<out;++o)y->data[std::size_t(dest)*out+o]+=values->data[(t*out+o)*kernel+k];}
  }finite(y->span());return y;
 }
 M activation(const std::string& p,Matrix& x,const std::atomic_bool& cancel){
  auto upsample=make(r,x.rows*2,x.cols),y=make(r,x.rows,x.cols);auto filter=w(p+".upsample.filter"),down=w(p+".downsample.lowpass.filter"),alpha=w(p+".act.alpha"),beta=w(p+".act.beta");
  // Replicate-pad 5, stride-2 transpose conv, scale2, crop15 each side.
  for(std::size_t t=0;t<upsample->rows;++t){if(t%32==0)stop(cancel);for(std::size_t c=0;c<x.cols;++c){float v=0;
   for(std::size_t k=0;k<12;++k){const auto numerator=std::ptrdiff_t(t+15)-std::ptrdiff_t(k);if(numerator%2==0){const auto source=std::clamp(numerator/2-5,std::ptrdiff_t(0),std::ptrdiff_t(x.rows-1));v+=x.data[std::size_t(source)*x.cols+c]*filter[k];}}
   v*=2;const auto a=std::exp(alpha[c]),b=std::exp(beta[c]);need(std::isfinite(a)&&std::isfinite(b),"h3_audio_activation_range");const auto s=std::sin(a*v);upsample->data[t*x.cols+c]=v+s*s/(b+1e-9f);
  }}
  // Replicate-pad (5,6), stride-2 cross-correlation.
  for(std::size_t t=0;t<x.rows;++t){if(t%32==0)stop(cancel);for(std::size_t c=0;c<x.cols;++c){float v=0;for(std::size_t k=0;k<12;++k){const auto source=std::clamp(std::ptrdiff_t(2*t+k)-5,std::ptrdiff_t(0),std::ptrdiff_t(upsample->rows-1));v+=upsample->data[std::size_t(source)*x.cols+c]*down[k];}y->data[t*x.cols+c]=v;}}
  finite(y->span());return y;
 }
 M block(const std::string& p,Matrix& input,std::size_t kernel,const std::atomic_bool& cancel){
  auto x=make(r,input.rows,input.cols);std::copy(input.span().begin(),input.span().end(),x->data.get());
  for(std::size_t j=0;j<3;++j){auto a=activation(p+".activations."+std::to_string(2*j),*x,cancel);auto b=conv(p+".convs1."+std::to_string(j),*a,x->cols,kernel,2*j+1,cancel);a.reset();a=activation(p+".activations."+std::to_string(2*j+1),*b,cancel);b.reset();b=conv(p+".convs2."+std::to_string(j),*a,x->cols,kernel,1,cancel);for(std::size_t i=0;i<x->span().size();++i){if(i%4096==0)stop(cancel);x->data[i]+=b->data[i];}finite(x->span());}
  return x;
 }
};
H3AudioDecoder::H3AudioDecoder(std::shared_ptr<Resources> r,std::shared_ptr<DenseCompute> c):resources_(std::move(r)),compute_(std::move(c)){need(bool(resources_),"h3_audio_resources");}
H3AudioDecoder::~H3AudioDecoder(){unload();}
void H3AudioDecoder::load(const char* root,const std::string& file,H3AudioConfig d,const std::atomic_bool& cancel,std::shared_ptr<checkpoint::ReadCache> cache){
 Active active(busy_);stop(cancel);need(!model_,"h3_audio_loaded");need(d.projection>0&&d.projection<=2048&&d.initial>=128&&d.initial<=1024&&(d.initial&(d.initial-1))==0,"h3_audio_config");
 auto m=std::make_unique<Impl>(*resources_,d,compute_);Lease parser(*resources_,4*1024*1024);checkpoint::Shard shard(root,file,std::make_shared<checkpoint::MemoryBudget>(4*1024*1024),{1024*1024,2048,4096});shard.cache_reads(std::move(cache),&cancel);
 for(auto& s:m->specs){const auto t=shard.tensor(s.name);need(t.dtype==checkpoint::Dtype::fp32&&t.rank==s.shape.size()&&std::equal(s.shape.begin(),s.shape.end(),t.shape.begin()),"h3_audio_tensor_layout");}
 const auto& last=m->specs.back();m->weights=make(*resources_,1,last.offset+last.count);Lease transfer(*resources_,4096);std::array<std::uint8_t,4096> bytes;
 for(const auto& s:m->specs)for(std::size_t at=0;at<s.count;){stop(cancel);const auto n=std::min<std::size_t>(1024,s.count-at);shard.read_tensor(s.name,at*4,{bytes.data(),n*4});for(std::size_t i=0;i<n;++i){std::uint32_t bits=0;for(std::size_t b=0;b<4;++b)bits|=std::uint32_t(bytes[i*4+b])<<(b*8);float v=std::bit_cast<float>(bits);need(std::isfinite(v),"h3_audio_nonfinite_weight");m->weights->data[s.offset+at+i]=v;}at+=n;}
 for(float v:m->w("latents_std")){need(v>0,"h3_audio_std");}shard.check_unchanged();stop(cancel);resources_->loaded(m->weights->lease.h);model_=std::move(m);
}
void H3AudioDecoder::unload(){need(!busy_,"busy");if(model_){resources_->begin_eviction(model_->weights->lease.h);model_.reset();}}
void H3AudioDecoder::decode(std::span<const float> input,std::size_t time,std::span<float> output,const std::atomic_bool& cancel,const Hook& hook){
 Active active(busy_);stop(cancel);need(bool(model_),"h3_audio_not_loaded");need(time>0&&time<=max_latents&&input.size()==time*64&&output.size()==time*1600,"h3_audio_shape");
 const auto a=reinterpret_cast<std::uintptr_t>(input.data()),b=reinterpret_cast<std::uintptr_t>(output.data());need(a<b?b-a>=input.size_bytes():a-b>=output.size_bytes(),"h3_audio_alias");finite(input);auto& m=*model_;Pin pin(*resources_,m.weights->lease.h);
 for(std::size_t channel=0;channel<2;++channel){stop(cancel);auto x=make(*resources_,time,32);auto mean=m.w("latents_mean"),std=m.w("latents_std");for(std::size_t t=0;t<time;++t)for(std::size_t c=0;c<32;++c)x->data[t*32+c]=input[(channel*time+t)*32+c]*std[c]+mean[c];
  if(hook){hook("normalized",channel);}stop(cancel);auto y=m.conv("dec_in_proj",*x,m.d.projection,1,1,cancel);x.reset();x=m.conv("decoder.conv_pre",*y,m.d.initial,7,1,cancel);y.reset();
  for(std::size_t i=0;i<7;++i){const auto c=m.d.initial>>(i+1);y=m.up("decoder.ups."+std::to_string(i)+".0",*x,c,i<2?5:2,i<2?9:4,cancel);x.reset();auto sum=make(*resources_,y->rows,c);
   for(std::size_t j=0;j<3;++j){auto branch=m.block("decoder.resblocks."+std::to_string(i*3+j),*y,std::array<std::size_t,3>{3,7,11}[j],cancel);for(std::size_t k=0;k<sum->span().size();++k){if(k%4096==0)stop(cancel);sum->data[k]+=branch->data[k];}}
   for(float& v:sum->span()){v/=3;}x=std::move(sum);y.reset();if(hook)hook("upsample",channel*7+i);stop(cancel);
  }
  y=m.activation("decoder.activation_post",*x,cancel);x.reset();x=m.conv("decoder.conv_post",*y,1,7,1,cancel,false);y.reset();
  for(std::size_t t=0;t<x->rows;++t){if(t%4096==0)stop(cancel);output[t*2+channel]=std::clamp(x->data[t],-1.0f,1.0f);}if(hook)hook("waveform",channel);
 }stop(cancel);
}
}
