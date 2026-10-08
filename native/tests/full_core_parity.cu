#include "full_core_reference.cuh"
#include <bit>
#include <charconv>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <vector>
using kadan::cuda::detail::FullBuffers;
namespace {
void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
std::vector<void*> owned;
template<class T>T* allocate(std::size_t n){T*p;check(cudaMalloc(&p,n*sizeof(T)));owned.push_back(p);check(cudaMemset(p,0,n*sizeof(T)));return p;}
template<class T>void upload(T*p,const std::vector<T>&v){check(cudaMemcpy(p,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice));}
template<class T>void exact(T*a,T*b,std::size_t n,const char* label){std::vector<T>x(n),y(n);check(cudaMemcpy(x.data(),a,n*sizeof(T),cudaMemcpyDeviceToHost));check(cudaMemcpy(y.data(),b,n*sizeof(T),cudaMemcpyDeviceToHost));if(std::memcmp(x.data(),y.data(),n*sizeof(T)))throw std::runtime_error(label);}
}
int main(int argc,char**argv){
 if(argc!=4||std::string_view(argv[1])!="--allow-gpu-validation"||std::string_view(argv[2])!="--device")return 2;
 try{
 int device=-1;std::string_view arg(argv[3]);auto[end,error]=std::from_chars(arg.data(),arg.data()+arg.size(),device);if(error!=std::errc{}||end!=arg.data()+arg.size()||device<0||device>=64)return 2;
 check(cudaSetDevice(device));kadan::full::Config c{2048,16,2,256,64,65536,1e-6f};
 const auto q=c.heads*c.head_dim,kv=c.kv_heads*c.head_dim;
 auto* norm=allocate<std::uint16_t>(c.head_dim);auto* frequencies=allocate<float>(c.rotary_dim/2);
 std::vector<float> freq(c.rotary_dim/2);for(std::size_t i=0;i<freq.size();++i)freq[i]=1.f/float(i+1);upload(frequencies,freq);
 auto make=[&](){FullBuffers b{};b.query_norm=norm;b.key_norm=norm;b.frequencies=frequencies;b.status=allocate<unsigned>(1);
 b.qg=allocate<float>(2*q);b.key=allocate<float>(kv);b.value=allocate<float>(kv);b.query=allocate<float>(q);b.gate=allocate<float>(q);b.probabilities=allocate<float>(c.heads*c.capacity);b.core=allocate<float>(q);b.gated=allocate<float>(q);b.keys=allocate<std::uint16_t>(c.capacity*kv);b.values=allocate<std::uint16_t>(c.capacity*kv);return b;};auto a=make(),b=make();
 unsigned seed=1947;auto random=[&](){seed=1664525*seed+1013904223;return float(int(seed>>16)%257-128)/1024;};
 std::vector<std::uint16_t> history(512*kv);for(auto&v:history)v=std::uint16_t(std::bit_cast<unsigned>(random())>>16);upload(a.keys,history);upload(b.keys,history);for(auto&v:history)v=std::uint16_t(std::bit_cast<unsigned>(random())>>16);upload(a.values,history);upload(b.values,history);
 // Begin dirty; later shorter positions must clear a previously longer prefix.
 check(cudaMemset(a.probabilities,0x3f,c.heads*c.capacity*sizeof(float)));
 check(cudaMemset(b.probabilities,0x3f,c.heads*c.capacity*sizeof(float)));
 const std::size_t positions[]={0,1,31,127,128,129,178,511,1,c.capacity-1,1,511};
 for(std::size_t test=0;test<12;++test){const auto position=positions[test];std::vector<float> qg(2*q),key(kv),value(kv);for(auto&v:qg)v=random();for(auto&v:key)v=random();for(auto&v:value)v=random();if(test==11)qg[0]=std::numeric_limits<float>::infinity();
 check(cudaMemset(a.status,0,sizeof(unsigned)));check(cudaMemset(b.status,0,sizeof(unsigned)));
 upload(a.qg,qg);upload(b.qg,qg);upload(a.key,key);upload(b.key,key);upload(a.value,value);upload(b.value,value);
 check(baseline::full_core(c,a,position));check(kadan::cuda::detail::full_core(c,b,position));check(cudaDeviceSynchronize());
 exact(a.query,b.query,q,"query");exact(a.gate,b.gate,q,"gate");exact(a.keys,b.keys,c.capacity*kv,"keys");exact(a.values,b.values,c.capacity*kv,"values");exact(a.probabilities,b.probabilities,c.heads*c.capacity,"probabilities/tail padding");exact(a.core,b.core,q,"core");exact(a.gated,b.gated,q,"gated");exact(a.status,b.status,1,"status");
 std::vector<float> probabilities(c.heads*c.capacity);check(cudaMemcpy(probabilities.data(),b.probabilities,probabilities.size()*sizeof(float),cudaMemcpyDeviceToHost));
 for(std::size_t head=0;head<c.heads;++head)for(std::size_t t=position+1;t<c.capacity;++t)if(probabilities[head*c.capacity+t]!=0.f)throw std::runtime_error("stale probability tail");
 unsigned flags=0;check(cudaMemcpy(&flags,b.status,sizeof(flags),cudaMemcpyDeviceToHost));
 if((test<11&&flags!=0)||(test==11&&flags==0))throw std::runtime_error("unexpected per-case status");}
 for(auto p:owned)check(cudaFree(p));
 std::cout<<"PASS: 12 full-shape attention cases (16 query heads, 2 KV heads, head dim 256, capacity 65536), positions 0/1/31/127/128/129/178/511/capacity-1, dirty probability buffers, shorter-after-longer and nonfinite input; bitwise query/KV/probability-padding/core/gated/status parity\n";
 }catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}
}
