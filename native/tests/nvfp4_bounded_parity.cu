#include "nvfp4_reference.cuh"
#include <bit>
#include <charconv>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <vector>
namespace {
void ck(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
}
int main(int argc,char**argv){
 if(argc!=4||std::string_view(argv[1])!="--allow-gpu-validation"||std::string_view(argv[2])!="--device")return 2;
 try{int device=-1;std::string_view arg(argv[3]);auto[end,error]=std::from_chars(arg.data(),arg.data()+arg.size(),device);if(error!=std::errc{}||end!=arg.data()+arg.size()||device<0||device>=64)return 2;ck(cudaSetDevice(device));
 const float bound=std::numeric_limits<float>::max()/4096.f;
 const float globals[]={0.f,.75f,-.75f,std::numeric_limits<float>::denorm_min(),bound,-bound,std::nextafter(bound,INFINITY),std::numeric_limits<float>::max(),std::numeric_limits<float>::max()};
 unsigned cases=0;
 for(auto rows:{3u,257u,248320u}){const unsigned columns=rows==3?32:2048;const auto wn=std::size_t(rows)*columns/2,sn=std::size_t(rows)*columns/16;
 std::vector<unsigned char> hw(wn),hs(sn);std::vector<float> hx(columns),a(rows),b(rows);for(std::size_t i=0;i<wn;++i)hw[i]=unsigned(i*17+31)%256;for(std::size_t i=0;i<sn;++i)hs[i]=unsigned(i*13)%127;
 unsigned char *w,*s;float*x,*ya,*yb;unsigned*flags;ck(cudaMalloc(&w,wn));ck(cudaMalloc(&s,sn));ck(cudaMalloc(&x,columns*4));ck(cudaMalloc(&ya,rows*4));ck(cudaMalloc(&yb,rows*4));ck(cudaMalloc(&flags,8));ck(cudaMemcpy(w,hw.data(),wn,cudaMemcpyHostToDevice));ck(cudaMemcpy(s,hs.data(),sn,cudaMemcpyHostToDevice));
 for(unsigned k=0;k<9;++k){if(rows==248320&&k!=1)continue;for(unsigned j=0;j<columns;++j)hx[j]=k>=4?1e-38f:float(int(j%13)-6)/8.f;ck(cudaMemcpy(x,hx.data(),columns*4,cudaMemcpyHostToDevice));
 if(k==8)ck(cudaMemset(w,0,wn));
 for(bool bf16:{false,true}){ck(cudaMemset(flags,0,8));ck((bf16?baseline_launch_nvfp4_bf16:baseline_launch_nvfp4)(w,s,globals[k],x,ya,flags,rows,columns));ck((bf16?kadan_launch_nvfp4_bf16:kadan_launch_nvfp4)(w,s,globals[k],x,yb,flags+1,rows,columns));ck(cudaDeviceSynchronize());ck(cudaMemcpy(a.data(),ya,rows*4,cudaMemcpyDeviceToHost));ck(cudaMemcpy(b.data(),yb,rows*4,cudaMemcpyDeviceToHost));unsigned f[2];ck(cudaMemcpy(f,flags,8,cudaMemcpyDeviceToHost));if(std::memcmp(a.data(),b.data(),rows*4)||f[0]!=f[1])throw std::runtime_error("projection bitwise parity");if((k<7||k==8)&&f[1]!=0)throw std::runtime_error("valid projection status");if(k==7&&f[1]==0)throw std::runtime_error("overflow not reported");++cases;}}
 for(void*p:{static_cast<void*>(w),static_cast<void*>(s),static_cast<void*>(x),static_cast<void*>(ya),static_cast<void*>(yb),static_cast<void*>(flags)})ck(cudaFree(p));}
 std::cout<<"PASS: "<<cases<<" bitwise NVFP4 cases, both BF16 modes, signs/subnormal/boundary/fallback/overflow and 248320x2048 head shape\n";
 }catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}
}
