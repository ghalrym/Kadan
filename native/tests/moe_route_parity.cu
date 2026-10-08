#include "moe_route_reference.cuh"
#include <vector>
#include <bit>
#include <iostream>
#include <stdexcept>
#include <cstring>
#include <string>
#include <string_view>
using kadan::cuda::detail::MoeBuffers;
void ck(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
std::vector<void*> owned;
template<class T>T* alloc(size_t n){T*p;ck(cudaMalloc(&p,n*sizeof(T)));owned.push_back(p);ck(cudaMemset(p,0,n*sizeof(T)));return p;}
template<class T>void upload(T*p,const std::vector<T>&v){ck(cudaMemcpy(p,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice));}
template<class T>void exact(T*a,T*b,size_t n,const char* label){std::vector<T>x(n),y(n);ck(cudaMemcpy(x.data(),a,n*sizeof(T),cudaMemcpyDeviceToHost));ck(cudaMemcpy(y.data(),b,n*sizeof(T),cudaMemcpyDeviceToHost));if(memcmp(x.data(),y.data(),n*sizeof(T)))throw std::runtime_error(label);}
int main(int argc,char**argv){
 if(argc!=4||std::string_view(argv[1])!="--allow-gpu-validation"||std::string_view(argv[2])!="--device")return 2;
 try{const int device=std::stoi(argv[3]);if(device<0||device>=64)return 2;ck(cudaSetDevice(device));kadan::moe::Config c{};c.hidden=2048;c.experts=256;c.top_k=8;
 auto*x=alloc<float>(c.hidden);auto*w=alloc<uint16_t>(c.hidden*c.experts);auto*g=alloc<uint16_t>(c.hidden);
 auto make=[&](){MoeBuffers b{};b.router=w;b.shared_gate=g;b.status=alloc<unsigned>(1);b.accumulator=alloc<float>(c.hidden);b.logits=alloc<float>(c.experts);b.probabilities=alloc<float>(c.experts);b.top_weights=alloc<float>(c.top_k);b.selected=alloc<unsigned>(c.top_k);b.shared_factor=alloc<float>(1);return b;};auto a=make(),b=make();
 unsigned seed=12345;auto rand=[&](){seed=1664525*seed+1013904223;return float(int(seed>>16)%257-128)/1024;};
 for(int test=0;test<6;++test){std::vector<float> hv(c.hidden);std::vector<uint16_t> hw(c.hidden*c.experts),hg(c.hidden);
 for(auto&v:hv)v=rand();for(auto&v:hw)v=uint16_t(std::bit_cast<unsigned>(test==0?0.f:rand())>>16);for(auto&v:hg)v=uint16_t(std::bit_cast<unsigned>(rand())>>16);
 upload(x,hv);upload(w,hw);upload(g,hg);baseline::reference_route<<<1,1>>>(c,a,x);ck(cudaGetLastError());ck(kadan::cuda::detail::moe_route(c,b,x));ck(cudaDeviceSynchronize());
 exact(a.logits,b.logits,c.experts,"logits");exact(a.probabilities,b.probabilities,c.experts,"probabilities");exact(a.selected,b.selected,c.top_k,"selected");exact(a.top_weights,b.top_weights,c.top_k,"weights");exact(a.shared_factor,b.shared_factor,1,"shared factor");exact(a.accumulator,b.accumulator,c.hidden,"accumulator");exact(a.status,b.status,1,"status");}
 for(auto p:owned)ck(cudaFree(p));std::cout<<"PASS: six full-shape 256x2048 router cases, including ties, bitwise identical logits/probabilities/top-k/weights/shared factor/status\n";
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
