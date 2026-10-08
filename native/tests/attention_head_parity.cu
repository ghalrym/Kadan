#include "linear_normalization_reference.cuh"
#include "full_normalization_reference.cuh"
#include <bit>
#include <charconv>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <vector>
using namespace kadan::cuda::detail;
void ck(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
std::vector<void*> owned;
template<class T>T* alloc(std::size_t n){T*p;ck(cudaMalloc(&p,n*sizeof(T)));owned.push_back(p);return p;}
template<class T>void upload(T*p,const std::vector<T>&v){ck(cudaMemcpy(p,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice));}
template<class T>void exact(T*a,T*b,std::size_t n){std::vector<T>x(n),y(n);ck(cudaMemcpy(x.data(),a,n*sizeof(T),cudaMemcpyDeviceToHost));ck(cudaMemcpy(y.data(),b,n*sizeof(T),cudaMemcpyDeviceToHost));if(std::memcmp(x.data(),y.data(),n*sizeof(T)))throw std::runtime_error("attention head bitwise mismatch");}
int main(int argc,char**argv){
 if(argc!=4||std::string_view(argv[1])!="--allow-gpu-validation"||std::string_view(argv[2])!="--device")return 2;
 try{int device=-1;std::string_view arg(argv[3]);auto[end,error]=std::from_chars(arg.data(),arg.data()+arg.size(),device);if(error!=std::errc{}||end!=arg.data()+arg.size()||device<0||device>=64)return 2;ck(cudaSetDevice(device));unsigned cases=0;
 for(unsigned dim:{1u,3u,128u,256u,4096u,4097u}){
 kadan::linear::Config lc{2048,2,4,dim,dim,4,1024,1e-6f};kadan::full::Config fc{2048,4,2,dim,2*std::min(32u,dim/2),1024,1e-6f};
 auto*weights=alloc<std::uint16_t>(dim);auto*frequencies=alloc<float>(32);std::vector<float> freq(32);for(unsigned j=0;j<32;++j)freq[j]=1.f/float(j+1);upload(frequencies,freq);
 auto make_linear=[&](){LinearBuffers b{};b.output_norm=weights;b.qkv=alloc<float>(8*dim);b.core=alloc<float>(4*dim);b.z=alloc<float>(4*dim);b.gated=alloc<float>(4*dim);b.status=alloc<unsigned>(1);return b;};auto la=make_linear(),lb=make_linear();
 auto make_full=[&](){FullBuffers b{};b.query_norm=b.key_norm=weights;b.frequencies=frequencies;b.qg=alloc<float>(8*dim);b.key=alloc<float>(2*dim);b.query=alloc<float>(4*dim);b.gate=alloc<float>(4*dim);b.status=alloc<unsigned>(1);return b;};auto fa=make_full(),fb=make_full();
 for(unsigned mode=0;mode<7;++mode){std::vector<float> qkv(8*dim),core(4*dim),z(4*dim),key(2*dim);std::vector<std::uint16_t> w(dim);for(unsigned j=0;j<8*dim;++j)qkv[j]=float(int(j%29)-14)/64.f;for(unsigned j=0;j<4*dim;++j){core[j]=float(int(j%31)-15)/64.f;z[j]=float(int(j%17)-8)/32.f;}for(unsigned j=0;j<2*dim;++j)key[j]=float(int(j%23)-11)/64.f;for(unsigned j=0;j<dim;++j)w[j]=std::uint16_t(std::bit_cast<unsigned>(float(j%7)/16.f)>>16);
 if(mode==1){for(auto&v:qkv)v=0;for(auto&v:core)v=0;for(auto&v:key)v=0;}
 if(mode==2||mode==3){qkv[0]=core[0]=key[0]=mode==2?INFINITY:NAN;}
 if(mode==4){for(auto&v:qkv)v=std::bit_cast<float>(0x7f7f0000u);for(auto&v:core)v=std::bit_cast<float>(0x7f7f0000u);for(auto&v:key)v=std::bit_cast<float>(0x7f7f0000u);}
 if(mode==5)w[0]=0x7fc1;if(mode==6){for(auto&v:qkv)v=-0.f;for(auto&v:core)v=-0.f;for(auto&v:key)v=-0.f;}
 upload(weights,w);for(auto b:{la,lb}){upload(b.qkv,qkv);upload(b.core,core);upload(b.z,z);ck(cudaMemset(b.status,0,4));}for(auto b:{fa,fb}){upload(b.qg,qkv);upload(b.key,key);ck(cudaMemset(b.status,0,4));}
 baseline_linear::normalize_qk<<<lc.key_heads,1>>>(lc,la);ck(cudaGetLastError());baseline_linear::gated_norm<<<lc.value_heads,1>>>(lc,la);ck(cudaGetLastError());ck(linear_normalize_qk(lc,lb));ck(linear_gated_norm(lc,lb));
 baseline_full::prepare<<<fc.heads,1>>>(fc,fa,178);ck(cudaGetLastError());ck(full_prepare(fc,fb,178));ck(cudaDeviceSynchronize());exact(la.qkv,lb.qkv,8*dim);exact(la.gated,lb.gated,4*dim);exact(la.status,lb.status,1);exact(fa.query,fb.query,4*dim);exact(fa.key,fb.key,2*dim);exact(fa.gate,fb.gate,4*dim);exact(fa.status,fb.status,1);unsigned flags;ck(cudaMemcpy(&flags,fb.status,4,cudaMemcpyDeviceToHost));if((mode>=2&&mode<=5)!=(flags!=0))throw std::runtime_error("full head expected numeric status");++cases;}
 for(void*p:owned)ck(cudaFree(p));owned.clear();}
 std::cout<<"PASS: "<<cases<<" combined attention-head cases: odd/full/shared-fallback dimensions, zero/signed zero, Inf/NaN, intermediate overflow, invalid norm weights; bitwise query/key/gate/QK/gated/status\n";
 }catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
