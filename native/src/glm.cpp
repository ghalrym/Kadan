// GLM text equations/reference: Transformers 536ecc007387a50e77603bb5d92100e9b07514cc,
// models/glm5_next/modeling_glm5_next.py. Original bounded C++ execution.
#include "kadan/glm.hpp"
#include "kadan/checkpoint.hpp"
#include "kadan/weight_inventory.hpp"
#include "metadata_json.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <numeric>
#include <set>
#include <string>
namespace kadan::glm {
namespace {
constexpr std::size_t H=4096,Q=8192,heads=64,dim=128,metadata=512ULL<<20,scratch=128ULL<<20;
constexpr std::string_view base="model.language_model.";
void need(bool b,const char* e){if(!b)throw std::runtime_error(e);}
using Vec=std::vector<float>;
float bf(float x){auto u=std::bit_cast<std::uint32_t>(x);u+=0x7fff+((u>>16)&1);return std::bit_cast<float>(u&0xffff0000);}
void cast(Vec& x){for(auto& v:x){need(std::isfinite(v),"glm_nonfinite");v=bf(v);}}
float sigmoid(float x){return x>=0?1/(1+std::exp(-x)):std::exp(x)/(1+std::exp(x));}
void norm(Vec& x,std::span<const float> w,float eps=1e-5,bool rounded=true){need(w.empty()||w.size()==x.size(),"glm_norm_shape");float ss=0;for(float v:x)ss+=v*v;auto inv=1/std::sqrt(ss/x.size()+eps);for(std::size_t i=0;i<x.size();++i)x[i]=(rounded?bf(x[i]*inv):x[i]*inv)*(w.empty()?1:w[i]);if(rounded)cast(x);}
struct Tensor {checkpoint::Shard* shard;checkpoint::TensorInfo info;};
struct Store {
 std::shared_ptr<checkpoint::MemoryBudget> budget=std::make_shared<checkpoint::MemoryBudget>(metadata);
 std::vector<std::unique_ptr<checkpoint::Shard>> shards;std::map<std::string,Tensor,std::less<>> tensors;
 std::size_t packed=0,execution=0;std::atomic_bool cancel{false};
 Store(const char* root){
  auto cfg=checkpoint::json::read(root,"config.json",16<<20,budget);need(cfg.at("model_type").string()=="glm5_next","glm_model_type");const auto& t=cfg.at("text_config");
  for(auto [key,value]:std::initializer_list<std::pair<const char*,std::size_t>>{{"hidden_size",H},{"num_hidden_layers",45},{"vocab_size",154880},{"num_attention_heads",64},{"q_lora_rank",1536},{"kv_lora_rank",512},{"qk_nope_head_dim",256},{"qk_rope_head_dim",0},{"v_head_dim",256},{"n_routed_experts",288},{"num_experts_per_tok",8},{"first_k_dense_replace",3},{"intermediate_size",12288},{"moe_intermediate_size",2048},{"n_shared_experts",1},{"hc_mult",4},{"hc_sinkhorn_iters",20},{"index_head_dim",128},{"index_n_heads",32},{"index_topk",2048},{"index_kpool",4},{"n_group",1},{"topk_group",1}})need(t.at(key).integer()==value,"glm_config_shape");
  need(t.at("rms_norm_eps").number()==1e-5&&t.at("hc_eps").number()==1e-6&&t.at("routed_scaling_factor").number()==2.5&&t.at("swiglu_limit").number()==10,"glm_config_math");
  need(t.at("mhc").boolean()&&t.at("mla_use_nope").boolean()&&t.at("norm_topk_prob").boolean()&&t.at("index_kpool_compress").boolean()&&t.at("index_kpool_always_select_tail").boolean(),"glm_config_flags");
  need(t.at("scoring_func").string()=="sigmoid"&&t.at("topk_method").string()=="noaux_tc","glm_router_config");
  const auto& linear=t.at("linear_attn_config");need(linear.at("num_heads").integer()==64&&linear.at("head_dim").integer()==128&&linear.at("short_conv_kernel_size").integer()==4&&linear.at("gate_lower_bound").number()==-5,"glm_kda_config");
  const auto& types=t.at("layer_types").children;need(types.size()==45,"glm_layer_count");for(std::size_t i=0;i<45;++i)need(types[i].string()==(i%4==3?"deepseek_sparse_attention":"linear_attention"),"glm_layer_type");
  const auto& quant=cfg.at("quantization_config");need(quant.at("quant_method").string()=="compressed-tensors"&&quant.at("format").string()=="nvfp4-pack-quantized","glm_quant_format");
  need(t.at("max_position_embeddings").integer()==1048576&&t.at("hidden_act").string()=="silu","glm_context_activation");
  const auto& mlpt=t.at("mlp_layer_types").children;const auto& idxt=t.at("indexer_types").children;need(mlpt.size()==45&&idxt.size()==45,"glm_block_types");for(std::size_t i=0;i<45;++i)need(mlpt[i].string()==(i<3?"dense":"sparse")&&idxt[i].string()=="full","glm_block_type");
  const auto& groups=quant.at("config_groups").children;need(groups.size()==1,"glm_quant_groups");const auto& group=groups[0];const auto& w=group.at("weights");need(group.at("format").string()=="nvfp4-pack-quantized"&&group.at("input_activations").kind==checkpoint::json::Value::Kind::null&&group.at("output_activations").kind==checkpoint::json::Value::Kind::null&&w.at("group_size").integer()==16&&w.at("num_bits").integer()==4&&w.at("type").string()=="float"&&w.at("strategy").string()=="tensor_group"&&w.at("symmetric").boolean()&&!w.at("dynamic").boolean()&&w.at("scale_dtype").string()=="torch.float8_e4m3fn","glm_quant_contract");
  auto generation=checkpoint::json::read(root,"generation_config.json",65536,budget);const auto& eos=generation.at("eos_token_id").children;need(eos.size()==3&&eos[0].integer()==154820&&eos[1].integer()==154827&&eos[2].integer()==154829,"glm_eos_contract");
  auto index=checkpoint::json::read(root,"model.safetensors.index.json",64<<20,budget);std::set<std::string> files;for(const auto& entry:index.at("weight_map").children)files.emplace(entry.string());need(files.size()<=128,"glm_shard_count");
  for(const auto& file:files){auto s=std::make_unique<checkpoint::Shard>(root,file,budget,checkpoint::Limits{16<<20,131072,4096});for(std::size_t i=0;i<s->tensor_count();++i){auto v=s->tensor_at(i);if(!(v.name.starts_with(base)||v.name=="lm_head.weight")||v.name.starts_with("model.language_model.layers.45."))continue;need(tensors.emplace(std::string(v.name),Tensor{s.get(),v}).second,"glm_duplicate_tensor");packed=checkpoint::weight_bytes_add(packed,v.bytes);execution=checkpoint::weight_bytes_add(execution,v.dtype==checkpoint::Dtype::u8?v.bytes*8:v.bytes*4/checkpoint::storage_width(v.dtype));}shards.push_back(std::move(s));}
  auto matrix=[&](const std::string& n,std::size_t rows,std::size_t cols){shape(n,{rows,cols});need(get(n).info.dtype==checkpoint::Dtype::bf16,"glm_matrix_dtype");};
  auto vector=[&](const std::string& n,std::size_t count,checkpoint::Dtype dtype){shape(n,{count});need(get(n).info.dtype==dtype,"glm_vector_dtype");};
  matrix(std::string(base)+"embed_tokens.weight",154880,H);matrix("lm_head.weight",154880,H);vector(std::string(base)+"norm.weight",H,checkpoint::Dtype::bf16);
  for(std::size_t i=0;i<45;++i){const auto p=std::string(base)+"layers."+std::to_string(i)+".";for(auto tag:{"attn","ffn"}){auto n=p+"hc_"+tag;matrix(n+"_fn",24,16384);vector(n+"_base",24,checkpoint::Dtype::fp32);vector(n+"_scale",3,checkpoint::Dtype::fp32);}vector(p+"input_layernorm.weight",H,checkpoint::Dtype::bf16);vector(p+"post_attention_layernorm.weight",H,checkpoint::Dtype::bf16);
   auto a=p+"self_attn.";if(i%4!=3){for(auto tag:{"q","k","v"}){matrix(a+tag+"_proj.weight",Q,H);shape(a+tag+"_conv1d.weight",{Q,1,4});need(get(a+tag+"_conv1d.weight").info.dtype==checkpoint::Dtype::bf16,"glm_conv_dtype");}for(auto tag:{"f","g"}){matrix(a+tag+"_a_proj.weight",128,H);matrix(a+tag+"_b_proj.weight",Q,128);}matrix(a+"b_proj.weight",64,H);matrix(a+"o_proj.weight",H,Q);vector(a+"o_norm.weight",128,checkpoint::Dtype::bf16);vector(a+"A_log",64,checkpoint::Dtype::fp32);vector(a+"dt_bias",Q,checkpoint::Dtype::fp32);
   }else{matrix(a+"q_a_proj.weight",1536,H);matrix(a+"q_b_proj.weight",16384,1536);vector(a+"q_a_layernorm.weight",1536,checkpoint::Dtype::bf16);matrix(a+"kv_a_proj_with_mqa.weight",512,H);matrix(a+"kv_b_proj.weight",32768,512);vector(a+"kv_a_layernorm.weight",512,checkpoint::Dtype::bf16);matrix(a+"o_proj.weight",H,16384);auto idx=a+"indexer.";matrix(idx+"wq_b.weight",4096,1536);matrix(idx+"wk.weight",128,H);matrix(idx+"weights_proj.weight",32,H);matrix(idx+"index_kpool_compress_gate",128,H);matrix(idx+"index_kpool_compress_ape",4,128);vector(idx+"k_norm.weight",128,checkpoint::Dtype::bf16);vector(idx+"k_norm.bias",128,checkpoint::Dtype::bf16);}
   auto mlp=p+"mlp.";auto dense=[&](const std::string& n,std::size_t width){matrix(n+"gate_proj.weight",width,H);matrix(n+"up_proj.weight",width,H);matrix(n+"down_proj.weight",H,width);};if(i<3)dense(mlp,12288);else{dense(mlp+"shared_experts.",2048);matrix(mlp+"gate.weight",288,H);vector(mlp+"gate.e_score_correction_bias",288,checkpoint::Dtype::fp32);for(std::size_t e=0;e<288;++e)for(auto tag:{"gate","up","down"}){auto n=mlp+"experts."+std::to_string(e)+"."+tag+"_proj";bool down=std::string_view(tag)=="down";auto rows=down?H:2048,cols=down?2048:H;shape(n+".weight_packed",{rows,cols/2});shape(n+".weight_scale",{rows,cols/16});shape(n+".weight_global_scale",{1});need(get(n+".weight_packed").info.dtype==checkpoint::Dtype::u8&&get(n+".weight_scale").info.dtype==checkpoint::Dtype::fp8&&get(n+".weight_global_scale").info.dtype==checkpoint::Dtype::fp32,"glm_quant_tensor_dtype");}}
  }
 }
 Tensor& get(const std::string& n){auto it=tensors.find(n);need(it!=tensors.end(),"glm_missing_tensor");return it->second;}
 void shape(const std::string& n,std::initializer_list<std::uint64_t> sizes){const auto& t=get(n).info;need(t.rank==sizes.size()&&std::equal(sizes.begin(),sizes.end(),t.shape.begin()),"glm_tensor_shape");}
 Vec read(const std::string& n,std::size_t first=0,std::size_t count=0){auto& t=get(n);if(!count)count=checkpoint::execution_weight_bytes(t.info,checkpoint::WeightRepresentation::f32)/4;need(count<=131072,"glm_small_tensor_bound");Vec out(count);t.shard->read_float_tensor(n,first,out,cancel);return out;}
 void source(const std::string& n,std::size_t first,std::span<float> out){auto it=tensors.find(n);if(it!=tensors.end()){it->second.shard->read_float_tensor(n,first,out,cancel);return;}
  need(n.ends_with(".weight"),"glm_source_name");auto prefix=n.substr(0,n.size()-7);auto& t=get(prefix+".weight_packed");const auto cols=t.info.shape[1]*2;need(first%cols==0&&out.size()%cols==0,"glm_quant_row_alignment");
  auto& scale=get(prefix+".weight_scale");auto multiplier=read(prefix+".weight_global_scale",0,1)[0];need(std::isfinite(multiplier)&&multiplier>0&&std::isfinite(1.f/multiplier),"glm_global_scale");multiplier=1.f/multiplier;
  constexpr std::size_t tile_rows=64;std::vector<std::uint8_t> packedrow(tile_rows*cols/2),blocks(tile_rows*cols/16);
  for(std::size_t row=0;row<out.size()/cols;row+=tile_rows){need(!cancel.load(),"glm_cancelled");const auto rows=std::min(tile_rows,out.size()/cols-row);t.shard->read_tensor(prefix+".weight_packed",(first/cols+row)*cols/2,std::span(packedrow).first(rows*cols/2));scale.shard->read_tensor(prefix+".weight_scale",(first/cols+row)*cols/16,std::span(blocks).first(rows*cols/16));for(std::size_t b=0;b<rows*cols/16;++b){auto v=quantization::e4m3fn(blocks[b]);need(std::isfinite(v)&&v>=0,"glm_block_scale");}for(std::size_t c=0;c<rows*cols;++c){auto value=bf(quantization::e2m1((packedrow[c/2]>>(4*(c%2)))&15)*quantization::e4m3fn(blocks[c/16])*multiplier);need(std::isfinite(value),"glm_weight_nonfinite");out[row*cols+c]=value;}}


 }
 Vec project(DenseCompute& compute,const std::string& n,std::span<const float> x,std::size_t out,bool rounded=true){auto it=tensors.find(n);auto key=it!=tensors.end()?n:n.substr(0,n.size()-7)+".weight_packed";auto& t=get(key);const auto in=t.info.shape[1]*(it!=tensors.end()?1:2);need(in&&x.size()%in==0,"glm_projection_input");shape(key,{out,it!=tensors.end()?in:in/2});Vec y(out*(x.size()/in));auto source_fn=[&](std::size_t first,std::span<float> dst){source(n,first,dst);};need(compute.dense_source(t.shard->tensor_identity(key),source_fn,{},x,in,out,y,cancel,false),"glm_cuda_source_required");if(rounded)cast(y);return y;}
};
struct State {Vec recurrent,conv,latent,pools,tail_key,tail_gate;};
}
std::size_t state_bytes(std::size_t context){need(context&&context<=serving::max_resident_capacity,"glm_context_bound");return 34ULL*(64*128*128+3*8192*4)*4+11ULL*(context*512+((context+3)/4)*128+8*128)*4;}
Plan plan(const char* root,std::size_t context){auto bytes=state_bytes(context);Store s(root);return {context,metadata+scratch+bytes,s.packed,s.tensors.size(),s.execution};}
struct Engine::Impl {
 Plan p;WorkerCompute execution;Handle memory=0;std::unique_ptr<Store> weights;std::array<State,45> states;std::size_t tokens=0;bool active=false;
 Impl(const char* root,Plan plan,const ComputePlan& cp):p(plan),execution(cp,Workload::llm){need(bool(execution.compute),"glm_cuda_required");memory=execution.resources->reserve(Workload::llm,host_footprint(*execution.resources,p.host-(48ULL<<20)));try{weights=std::make_unique<Store>(root);need(weights->packed==p.packed&&weights->tensors.size()==p.tensors,"glm_preflight_changed");for(std::size_t i=0;i<45;++i){auto& s=states[i];if(i%4!=3){s.recurrent.resize(64*128*128);s.conv.resize(3*8192*4);}else{s.latent.resize(p.context*512);s.pools.resize(((p.context+3)/4)*128);s.tail_key.resize(4*128);s.tail_gate.resize(4*128);}}execution.compute->prepare_weights(p.execution_weights);}catch(...){weights.reset();execution.resources->released(memory);memory=0;throw;}}
 ~Impl(){weights.reset();for(auto& s:states)s={};if(memory)execution.resources->released(memory);}
 Vec project(const std::string& n,std::span<const float> x,std::size_t out,bool rounded=true){return weights->project(*execution.compute,n,x,out,rounded);}
 Vec rms(const std::string& n,Vec x){norm(x,weights->read(n));return x;}
 struct Mixing{Vec input,post,comb;};
 Mixing mix(const std::string& prefix,const Vec& streams){Vec x=streams;norm(x,{},1e-5,false);auto coeff=project(prefix+"_fn",x,24,false);auto bias=weights->read(prefix+"_base"),scale=weights->read(prefix+"_scale");Mixing m{Vec(H),Vec(4),Vec(16)};for(std::size_t a=0;a<4;++a){float pre=sigmoid(coeff[a]*scale[0]+bias[a])+1e-6f;m.post[a]=bf(2*sigmoid(coeff[4+a]*scale[1]+bias[4+a]));for(std::size_t j=0;j<H;++j)m.input[j]+=pre*streams[a*H+j];float max=-INFINITY;for(std::size_t b=0;b<4;++b){auto k=a*4+b;m.comb[k]=coeff[8+k]*scale[2]+bias[8+k];max=std::max(max,m.comb[k]);}float sum=0;for(std::size_t b=0;b<4;++b)sum+=std::exp(m.comb[a*4+b]-max);for(std::size_t b=0;b<4;++b)m.comb[a*4+b]=std::exp(m.comb[a*4+b]-max)/sum+1e-6f;}
  auto normalize=[&](bool columns){for(std::size_t a=0;a<4;++a){float sum=1e-6f;for(std::size_t b=0;b<4;++b)sum+=m.comb[columns?b*4+a:a*4+b];for(std::size_t b=0;b<4;++b)m.comb[columns?b*4+a:a*4+b]/=sum;}};normalize(true);for(int i=1;i<20;++i){normalize(false);normalize(true);}cast(m.input);cast(m.comb);return m;
 }
 void combine(Vec& streams,const Mixing& m,const Vec& y){Vec next(streams.size());for(std::size_t b=0;b<4;++b)for(std::size_t j=0;j<H;++j){float v=0;for(std::size_t a=0;a<4;++a)v+=m.comb[a*4+b]*streams[a*H+j];next[b*H+j]=bf(bf(v)+bf(m.post[b]*y[j]));}streams.swap(next);}
 Vec kda(const std::string& prefix,State& s,const Vec& x){std::array<Vec,3> qkv;for(std::size_t which=0;which<3;++which){std::string tag=which==0?"q":which==1?"k":"v";qkv[which]=project(prefix+tag+"_proj.weight",x,Q);auto conv=weights->read(prefix+tag+"_conv1d.weight");for(std::size_t j=0;j<Q;++j){auto* h=s.conv.data()+(which*Q+j)*4;std::move(h+1,h+4,h);h[3]=qkv[which][j];float v=0;for(std::size_t k=0;k<4;++k)v+=h[k]*conv[j*4+k];v=bf(v);qkv[which][j]=bf(v*sigmoid(v));}}
  auto f=project(prefix+"f_b_proj.weight",project(prefix+"f_a_proj.weight",x,128),Q);auto bias=weights->read(prefix+"dt_bias"),alog=weights->read(prefix+"A_log"),beta=project(prefix+"b_proj.weight",x,64);auto gate=project(prefix+"g_b_proj.weight",project(prefix+"g_a_proj.weight",x,128),Q),normw=weights->read(prefix+"o_norm.weight");Vec output(Q);
  for(std::size_t h=0;h<heads;++h){float qnorm=1e-6f,knorm=1e-6f;for(std::size_t j=0;j<dim;++j){qnorm+=qkv[0][h*dim+j]*qkv[0][h*dim+j];knorm+=qkv[1][h*dim+j]*qkv[1][h*dim+j];}qnorm=1/std::sqrt(qnorm)/std::sqrt(float(dim));knorm=1/std::sqrt(knorm);Vec memory(dim),delta(dim);auto* state=s.recurrent.data()+h*dim*dim;for(std::size_t k=0;k<dim;++k){auto i=h*dim+k;auto decay=std::exp(-5*sigmoid(std::exp(alog[h])*(f[i]+bias[i])));for(std::size_t v=0;v<dim;++v){state[k*dim+v]*=decay;memory[v]+=state[k*dim+v]*qkv[1][i]*knorm;}}
   for(std::size_t v=0;v<dim;++v)delta[v]=(qkv[2][h*dim+v]-memory[v])*bf(sigmoid(beta[h]));
   for(std::size_t k=0;k<dim;++k)for(std::size_t v=0;v<dim;++v){state[k*dim+v]+=qkv[1][h*dim+k]*knorm*delta[v];output[h*dim+v]+=state[k*dim+v]*qkv[0][h*dim+k]*qnorm;}
   Vec row(output.begin()+h*dim,output.begin()+(h+1)*dim);cast(row);norm(row,normw,1e-5,false);for(std::size_t v=0;v<dim;++v)output[h*dim+v]=bf(row[v]*sigmoid(gate[h*dim+v]));
  }return project(prefix+"o_proj.weight",output,H);
 }
 Vec mlp(const std::string& prefix,const Vec& x,std::size_t intermediate){auto gate=project(prefix+"gate_proj.weight",x,intermediate),up=project(prefix+"up_proj.weight",x,intermediate);for(std::size_t i=0;i<intermediate;++i){auto g=std::min(10.f,gate[i]);gate[i]=bf(bf(g*sigmoid(g))*std::clamp(up[i],-10.f,10.f));}return project(prefix+"down_proj.weight",gate,H);}
 Vec moe(const std::string& prefix,const Vec& x){auto scores=project(prefix+"gate.weight",x,288,false),bias=weights->read(prefix+"gate.e_score_correction_bias");std::array<std::size_t,288> order;std::iota(order.begin(),order.end(),0);for(auto& v:scores)v=sigmoid(v);std::partial_sort(order.begin(),order.begin()+8,order.end(),[&](auto a,auto b){return scores[a]+bias[a]>scores[b]+bias[b];});float sum=1e-20f;for(std::size_t i=0;i<8;++i)sum+=scores[order[i]];std::sort(order.begin(),order.begin()+8);Vec y(H);for(std::size_t i=0;i<8;++i){auto expert=mlp(prefix+"experts."+std::to_string(order[i])+".",x,2048);auto factor=scores[order[i]]/sum*2.5f;for(std::size_t j=0;j<H;++j)y[j]=bf(y[j]+bf(expert[j]*factor));}auto shared=mlp(prefix+"shared_experts.",x,2048);for(std::size_t j=0;j<H;++j)y[j]=bf(y[j]+shared[j]);return y;}
 Vec dsa(const std::string&,State&,const Vec&);
};
Vec Engine::Impl::dsa(const std::string& prefix,State& s,const Vec& x){
 auto qr=rms(prefix+"q_a_layernorm.weight",project(prefix+"q_a_proj.weight",x,1536));auto q=project(prefix+"q_b_proj.weight",qr,16384);auto latent=rms(prefix+"kv_a_layernorm.weight",project(prefix+"kv_a_proj_with_mqa.weight",x,512));std::copy(latent.begin(),latent.end(),s.latent.begin()+tokens*512);
 const auto ip=prefix+"indexer.";auto iq=project(ip+"wq_b.weight",qr,4096),key=project(ip+"wk.weight",x,128),scale=weights->read(ip+"k_norm.weight"),bias=weights->read(ip+"k_norm.bias");float mean=std::accumulate(key.begin(),key.end(),0.f)/128,variance=0;for(auto v:key)variance+=(v-mean)*(v-mean);for(std::size_t i=0;i<128;++i)key[i]=bf((key[i]-mean)/std::sqrt(variance/128+1e-6f)*scale[i]+bias[i]);
 auto gate=project(ip+"index_kpool_compress_gate",x,128);auto tail=tokens%4;std::copy(key.begin(),key.end(),s.tail_key.begin()+tail*128);std::copy(gate.begin(),gate.end(),s.tail_gate.begin()+tail*128);
 const auto pools=(tokens+1)/4;if(tail==3){auto ape=weights->read(ip+"index_kpool_compress_ape");for(std::size_t d=0;d<128;++d){float max=-INFINITY;for(std::size_t j=0;j<4;++j)max=std::max(max,s.tail_gate[j*128+d]+ape[j*128+d]);float total=0;for(std::size_t j=0;j<4;++j)total+=std::exp(s.tail_gate[j*128+d]+ape[j*128+d]-max);float value=0;for(std::size_t j=0;j<4;++j)value+=bf(bf(std::exp(s.tail_gate[j*128+d]+ape[j*128+d]-max)/total)*s.tail_key[j*128+d]);s.pools[(pools-1)*128+d]=bf(value);}}
 auto headweight=project(ip+"weights_proj.weight",x,32);Vec poolscore(pools);for(std::size_t pidx=0;pidx<pools;++pidx)for(std::size_t h=0;h<32;++h){float dot=0;for(std::size_t d=0;d<128;++d)dot+=iq[h*128+d]*s.pools[pidx*128+d];poolscore[pidx]+=std::max(0.f,dot/std::sqrt(128.f))*headweight[h]/std::sqrt(32.f);}
 std::vector<std::size_t> order(pools);std::iota(order.begin(),order.end(),0);auto chosen=std::min<std::size_t>(512,pools);std::partial_sort(order.begin(),order.begin()+chosen,order.end(),[&](auto a,auto b){return poolscore[a]>poolscore[b];});std::vector<std::size_t> selected;selected.reserve(2051);for(std::size_t i=0;i<chosen;++i)for(std::size_t j=0;j<4;++j)selected.push_back(order[i]*4+j);for(std::size_t j=pools*4;j<=tokens;++j)selected.push_back(j);std::sort(selected.begin(),selected.end());need(!selected.empty()&&selected.size()<=2051,"glm_sparse_indices");
 // Keep only compressed KV between tokens. Two bounded expansion passes preserve
 // BF16 softmax probabilities without retaining expanded per-head K/V history.
 Vec scores(selected.size()*64);
 auto expand=[&](auto use){for(std::size_t first=0;first<selected.size();first+=16){const auto count=std::min<std::size_t>(16,selected.size()-first);Vec latents(count*512);for(std::size_t i=0;i<count;++i)std::copy_n(s.latent.begin()+selected[first+i]*512,512,latents.begin()+i*512);auto kv=project(prefix+"kv_b_proj.weight",latents,32768);for(std::size_t i=0;i<count;++i)use(first+i,std::span<const float>(kv).subspan(i*32768,32768));}};
 expand([&](std::size_t i,std::span<const float> kv){for(std::size_t h=0;h<64;++h){float value=0;for(std::size_t d=0;d<256;++d)value+=q[h*256+d]*kv[h*512+d];scores[h*selected.size()+i]=bf(bf(value)/16);}});
 for(std::size_t h=0;h<64;++h){auto first=scores.begin()+h*selected.size();auto last=first+selected.size();auto max=*std::max_element(first,last);float sum=0;for(auto it=first;it!=last;++it)sum+=std::exp(*it-max);for(auto it=first;it!=last;++it)*it=bf(std::exp(*it-max)/sum);}
 Vec output(16384);expand([&](std::size_t i,std::span<const float> kv){for(std::size_t h=0;h<64;++h)for(std::size_t d=0;d<256;++d)output[h*256+d]+=scores[h*selected.size()+i]*kv[h*512+256+d];});cast(output);return project(prefix+"o_proj.weight",output,H);
}
Engine::Engine(const char* root,Plan p,const ComputePlan& cp):impl_(std::make_unique<Impl>(root,p,cp)){}
Engine::~Engine()=default;
std::shared_ptr<Resources> Engine::resources()const{return impl_->execution.resources;}
serving::Info Engine::info()const{return {154880,impl_->p.context,impl_->p.execution_weights,impl_->p.host};}
void Engine::reset(){auto& m=*impl_;m.tokens=0;for(std::size_t i=0;i<45;++i){std::fill(m.states[i].recurrent.begin(),m.states[i].recurrent.end(),0);std::fill(m.states[i].conv.begin(),m.states[i].conv.end(),0);}}
void Engine::begin_request(){for(const auto& shard:impl_->weights->shards)shard->check_unchanged();impl_->execution.resume();reset();impl_->active=true;}
void Engine::end_request(){impl_->execution.idle();impl_->active=false;}
void Engine::park(){need(!impl_->active,"glm_active_park");impl_->execution.park();}
serving::Token Engine::step(unsigned token,bool){auto& m=*impl_;need(m.active&&token<154880&&m.tokens<m.p.context,"glm_step_state");auto x=m.weights->read(std::string(base)+"embed_tokens.weight",token*H,H);Vec streams;streams.reserve(4*H);for(int i=0;i<4;++i)streams.insert(streams.end(),x.begin(),x.end());for(std::size_t i=0;i<45;++i){auto prefix=std::string(base)+"layers."+std::to_string(i)+".";auto mix=m.mix(prefix+"hc_attn",streams);auto input=m.rms(prefix+"input_layernorm.weight",mix.input);auto output=i%4==3?m.dsa(prefix+"self_attn.",m.states[i],input):m.kda(prefix+"self_attn.",m.states[i],input);m.combine(streams,mix,output);mix=m.mix(prefix+"hc_ffn",streams);input=m.rms(prefix+"post_attention_layernorm.weight",mix.input);output=i<3?m.mlp(prefix+"mlp.",input,12288):m.moe(prefix+"mlp.",input);m.combine(streams,mix,output);}
 for(std::size_t j=0;j<H;++j)x[j]=bf((streams[j]+streams[H+j]+streams[2*H+j]+streams[3*H+j])/4);
 x=m.rms(std::string(base)+"norm.weight",x);auto logits=m.project("lm_head.weight",x,154880);auto chosen=std::size_t(std::max_element(logits.begin(),logits.end())-logits.begin());++m.tokens;return {unsigned(chosen),is_eos(unsigned(chosen)),m.tokens};}
void Engine::close(){auto r=resources();impl_->active=false;impl_->execution.park();impl_.reset();for(auto b:r->snapshot().used)need(b==0,"glm_cleanup_unconfirmed");}
}
