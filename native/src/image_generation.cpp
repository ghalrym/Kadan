#include "kadan/image_generation.hpp"
#include "kadan/image_tokenizer.hpp"
#include "kadan/image_text.hpp"
#include "kadan/image_denoiser.hpp"
#include "kadan/image_vae.hpp"
#include "kadan/image_schedule.hpp"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <random>
namespace kadan::image {
namespace {
void need(bool b,const char* e){if(!b)throw std::runtime_error(e);}
void stop(const std::atomic_bool& c){need(!c.load(),"image_cancelled");}
Footprint host(Resources& r,Bytes n){auto p=r.snapshot().capacity;std::fill(p.begin(),p.end(),0);p[0]=n;return p;}
struct Lease{Resources& r;Handle h;Lease(Resources& a,Bytes n):r(a),h(r.reserve(Workload::image,host(r,n))){}~Lease(){r.released(h);}};
struct Busy{bool& b;Busy(bool& f):b(f){need(!b,"busy");b=true;}~Busy(){b=false;}};
using Json=nlohmann::json;
Json read(const std::string& path){std::ifstream f(path,std::ios::binary|std::ios::ate);need(bool(f)&&f.tellg()>0&&f.tellg()<=65536,"image_config_size");std::string s(std::size_t(f.tellg()),'\0');f.seekg(0);f.read(s.data(),s.size());need(bool(f),"image_config_read");return Json::parse(s);}
constexpr const char* prefix="<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n";
}
struct Generator::Impl {
 std::shared_ptr<DenseCompute> compute;
 Lease metadata;PromptTokenizer tokenizer;TextEncoder text;Denoiser denoiser;VaeDecoder vae;std::array<float,64> mean,stddev;
 Impl(std::shared_ptr<Resources> r,std::shared_ptr<DenseCompute> c):compute(c),metadata(*r,32*1024*1024),tokenizer(r),text(r,c),denoiser(r,c),vae(r,c){}
};
Generator::Generator(std::shared_ptr<Resources> r,std::shared_ptr<DenseCompute> compute):resources_(std::move(r)),compute_(std::move(compute)){need(bool(resources_),"image_resources");}
Generator::~Generator(){unload();}
void Generator::load(const std::string& root,const std::atomic_bool& cancel,const Hook& hook){Busy active(busy_);stop(cancel);need(!model_,"image_generator_loaded");auto m=std::make_unique<Impl>(resources_,compute_);
 auto text=read(root+"/text_encoder/config.json").at("text_config");need(text.at("vocab_size")==151936&&text.at("hidden_size")==4096&&text.at("num_hidden_layers")==36&&text.at("num_attention_heads")==32&&text.at("num_key_value_heads")==8&&text.at("head_dim")==128&&text.at("intermediate_size")==12288&&text.at("rope_theta")==5000000&&text.at("rms_norm_eps")==1e-6&&!text.at("attention_bias").get<bool>(),"image_text_config_contract");
 auto v=read(root+"/vae/config.json");need(v.at("decoder_base_dim")==144&&v.at("z_dim")==64&&v.at("dim_mult")==Json::array({1,2,4,8,8})&&v.at("num_res_blocks")==2&&v.at("out_channels")==4&&v.at("is_residual")==true&&v.at("patch_size").is_null()&&v.at("temperal_downsample")==Json::array({false,true,true,true}),"image_vae_config_contract");need(v.at("latents_mean").size()==64&&v.at("latents_std").size()==64,"image_latent_normalization");for(std::size_t i=0;i<64;++i){m->mean[i]=v.at("latents_mean")[i];m->stddev[i]=v.at("latents_std")[i];need(std::isfinite(m->mean[i])&&std::isfinite(m->stddev[i])&&m->stddev[i]>0,"image_latent_normalization");}
 auto s=read(root+"/scheduler/scheduler_config.json");need(s.at("_class_name")=="FlowMatchEulerDiscreteScheduler"&&s.at("base_image_seq_len")==256&&s.at("max_image_seq_len")==8192&&s.at("base_shift")==.5&&s.at("max_shift")==.9&&s.at("shift_terminal")==.02&&s.at("time_shift_type")=="exponential"&&s.at("use_dynamic_shifting")==true&&s.at("num_train_timesteps")==1000,"image_scheduler_contract");for(auto key:{"invert_sigmas","stochastic_sampling","use_beta_sigmas","use_exponential_sigmas","use_karras_sigmas"})need(s.at(key)==false,"image_scheduler_contract");
 auto t=read(root+"/transformer/config.json");need(t.at("num_layers")==32&&t.at("num_attention_heads")==32&&t.at("attention_head_dim")==128&&t.at("in_channels")==64&&t.at("context_in_dim")==4096&&t.at("axes_dims_rope")==Json::array({16,56,56})&&t.at("patch_size")==1&&t.at("mlp_ratio")==3&&t.at("eps")==1e-6&&t.at("causal_condition")==true&&t.at("out_channels")==64,"image_transformer_contract");
 m->tokenizer.load(root+"/processor/tokenizer.json",cancel);if(hook)hook("tokenizer_loaded",0);
 const std::array<std::string,4> text_files{"model-00001-of-00004.safetensors","model-00002-of-00004.safetensors","model-00003-of-00004.safetensors","model-00004-of-00004.safetensors"};auto tr=root+"/text_encoder";m->text.load(tr.c_str(),text_files,{},cancel,[&](std::size_t bytes){if(hook)hook("text_loading_mib",bytes/(1024*1024));});if(hook)hook("text_loaded",0);
 const std::array<std::string,2> denoiser_files{"diffusion_pytorch_model-00001-of-00002.safetensors","diffusion_pytorch_model-00002-of-00002.safetensors"};auto dr=root+"/transformer";m->denoiser.load(dr.c_str(),denoiser_files,{},cancel,hook);if(hook)hook("denoiser_loaded",0);auto vr=root+"/vae";m->vae.load(vr.c_str(),"diffusion_pytorch_model.safetensors",{},cancel,[&](std::size_t bytes){if(hook)hook("vae_loading_mib",bytes/(1024*1024));});if(hook)hook("vae_loaded",0);stop(cancel);model_=std::move(m);
}
void Generator::unload(){need(!busy_,"busy");model_.reset();}
void Generator::validate(const ImageRequest& request){need(!request.prompt.empty()&&request.prompt.size()<=8000&&request.width>=32&&request.height>=32&&request.width<=3072&&request.height<=3072&&request.width*request.height<=4608*1024&&request.width%32==0&&request.height%32==0&&request.steps>=2&&request.steps<=100,"image_request_bounds");}
void Generator::generate(const ImageRequest& request,std::span<float> output,const std::atomic_bool& cancel,const Hook& hook){Busy active(busy_);stop(cancel);need(bool(model_),"image_generator_not_loaded");validate(request);need(output.size()==request.width*request.height*4,"image_output_shape");auto& m=*model_;auto h=request.height/16,w=request.width/16,points=h*w;
 Lease admitted(*resources_,64*1024*1024);std::array<std::uint32_t,2048> tokens;auto raw=std::string(prefix)+"<|im_start|>user\n"+request.prompt+"<|im_end|>\n<|im_start|>assistant\n";auto count=m.tokenizer.encode(raw,tokens,cancel);std::array<std::uint32_t,64> system;auto drop=m.tokenizer.encode(prefix,system,cancel);need(drop==14&&count>drop,"image_prompt_prefix");std::vector<float> features(count*4096);m.text.execute(std::span(tokens).first(count),features,cancel,[&](std::size_t i){if(hook)hook("text_layer",i);});auto condition=std::span(features).subspan(drop*4096);const auto text_count=count-drop;
 std::vector<float> latent(points*64),velocity(points*64),decoded(64*points);std::mt19937_64 rng(request.seed);auto uniform=[&](){return (double(rng()>>11)+.5)/9007199254740992.;};for(std::size_t i=0;i<latent.size();i+=2){double radius=std::sqrt(-2*std::log(uniform())),angle=6.2831853071795864769*uniform();latent[i]=float(radius*std::cos(angle));if(i+1<latent.size())latent[i+1]=float(radius*std::sin(angle));}auto times=schedule(points,request.steps);
 for(std::size_t i=0;i<request.steps;++i){stop(cancel);m.denoiser.execute(latent,condition,text_count,h,w,times.sigma[i],velocity,cancel,[&](const char* phase,std::size_t j){if(hook)hook(phase,j);});const float dt=times.sigma[i+1]-times.sigma[i];for(std::size_t j=0;j<latent.size();++j){latent[j]+=dt*velocity[j];need(std::isfinite(latent[j]),"image_nonfinite_latent");}if(hook)hook("step_completed",i+1);}
 for(std::size_t p=0;p<points;++p)for(std::size_t c=0;c<64;++c)decoded[c*points+p]=latent[p*64+c]*m.stddev[c]+m.mean[c];
 m.vae.decode(decoded,h,w,output,cancel,[&](std::size_t i){if(hook)hook("vae_up_block",i);});stop(cancel);
}
}
