#include "linear_normalization_reference.cuh"
#include <bit>
#include <charconv>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <vector>
using kadan::cuda::detail::LinearBuffers;
namespace {
void ck(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
std::vector<void*> owned;
template<class T>T* alloc(std::size_t n){T*p;ck(cudaMalloc(&p,n*sizeof(T)));owned.push_back(p);return p;}
template<class T>void upload(T*p,const std::vector<T>&v){ck(cudaMemcpy(p,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice));}
template<class T>void exact(T*a,T*b,std::size_t n){std::vector<T>x(n),y(n);ck(cudaMemcpy(x.data(),a,n*sizeof(T),cudaMemcpyDeviceToHost));ck(cudaMemcpy(y.data(),b,n*sizeof(T),cudaMemcpyDeviceToHost));if(std::memcmp(x.data(),y.data(),n*sizeof(T)))throw std::runtime_error("auxiliary bitwise parity");}
}
int main(int argc,char**argv){
 if(argc!=4||std::string_view(argv[1])!="--allow-gpu-validation"||std::string_view(argv[2])!="--device")return 2;
 try{int device=-1;std::string_view arg(argv[3]);auto[end,error]=std::from_chars(arg.data(),arg.data()+arg.size(),device);if(error!=std::errc{}||end!=arg.data()+arg.size()||device<0||device>=64)return 2;ck(cudaSetDevice(device));unsigned cases=0;
 for(unsigned hidden:{2u,2048u,4096u,4097u})for(unsigned heads:{1u,4u,32u}){kadan::linear::Config c{hidden,heads==32?16u:1u,heads,128,128,4,1024,1e-6f};const auto values=c.value_heads*c.value_dim,channels=2*c.key_heads*c.key_dim+values;
 auto*x=alloc<float>(c.hidden);auto*wa=alloc<std::uint16_t>(heads*c.hidden);auto*wb=alloc<std::uint16_t>(heads*c.hidden);
 auto make=[&](){LinearBuffers b{};b.normalized=x;b.a_weight=wa;b.b_weight=wb;b.qkv=alloc<float>(channels);b.z=alloc<float>(values);b.a=alloc<float>(heads);b.b=alloc<float>(heads);b.status=alloc<unsigned>(1);return b;};auto a=make(),b=make();
 for(unsigned mode=0;mode<5;++mode){std::vector<float> input(c.hidden),qkv(channels),z(values);std::vector<std::uint16_t> weights(heads*c.hidden);for(unsigned j=0;j<c.hidden;++j)input[j]=float(int(j%63)-31)/512.f;for(std::size_t j=0;j<channels;++j)qkv[j]=float(int(j%31)-15)/137.f;for(std::size_t j=0;j<values;++j)z[j]=float(int(j%29)-14)/139.f;for(std::size_t j=0;j<weights.size();++j)weights[j]=std::uint16_t(std::bit_cast<unsigned>(float(int(j%127)-63)/1024.f)>>16);
 if(mode==1)for(auto&v:input)v=0.f;if(mode==2)input[0]=INFINITY;if(mode==3){for(auto&v:input)v=2.f;for(auto&v:weights)v=0x7f7f;}
 if(mode==4)input[0]=NAN;
 upload(x,input);upload(wa,weights);upload(wb,weights);upload(a.qkv,qkv);upload(b.qkv,qkv);upload(a.z,z);upload(b.z,z);ck(cudaMemset(a.status,0,4));ck(cudaMemset(b.status,0,4));baseline_linear::auxiliary<<<(channels+127)/128,128>>>(c,a);ck(cudaGetLastError());ck(kadan::cuda::detail::linear_auxiliary(c,b));ck(cudaDeviceSynchronize());exact(a.a,b.a,heads);exact(a.b,b.b,heads);exact(a.qkv,b.qkv,channels);exact(a.z,b.z,values);exact(a.status,b.status,1);unsigned flags;ck(cudaMemcpy(&flags,b.status,4,cudaMemcpyDeviceToHost));if((mode>=2)!=(flags!=0))throw std::runtime_error("auxiliary status");++cases;}
 for(auto p:owned)ck(cudaFree(p));owned.clear();}
 std::cout<<"PASS: "<<cases<<" auxiliary projection cases, actual 2048x32 shape, BF16 outputs and per-case numeric flags\n";
 }catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}
}
