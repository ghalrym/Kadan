#include "nvfp4_kernel.cuh"
#include "moe_kernel.cuh"
#include <charconv>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <vector>
void ck(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
std::vector<void*> owned;
template<class T>T* alloc(std::size_t n){T*p;ck(cudaMalloc(&p,n*sizeof(T)));owned.push_back(p);return p;}
template<class T>void upload(T*p,const std::vector<T>&x){ck(cudaMemcpy(p,x.data(),x.size()*sizeof(T),cudaMemcpyHostToDevice));}
template<class T>void exact(T*a,T*b,std::size_t n){std::vector<T>x(n),y(n);ck(cudaMemcpy(x.data(),a,n*sizeof(T),cudaMemcpyDeviceToHost));ck(cudaMemcpy(y.data(),b,n*sizeof(T),cudaMemcpyDeviceToHost));if(std::memcmp(x.data(),y.data(),n*sizeof(T)))throw std::runtime_error("fused bitwise output/status mismatch");}
int main(int argc,char**argv){
 if(argc!=4||std::string_view(argv[1])!="--allow-gpu-validation"||std::string_view(argv[2])!="--device")return 2;
 try{int device=-1;std::string_view arg(argv[3]);auto[end,error]=std::from_chars(arg.data(),arg.data()+arg.size(),device);if(error!=std::errc{}||end!=arg.data()+arg.size()||device<0||device>=64)return 2;ck(cudaSetDevice(device));unsigned cases=0;
 for(unsigned rows:{3u,257u,2048u})for(unsigned cols:{32u,512u,2048u}){
 auto*w=alloc<std::uint8_t>(rows*cols/2);auto*s=alloc<std::uint8_t>(rows*cols/16);auto*uw=alloc<std::uint8_t>(rows*cols/2);auto*us=alloc<std::uint8_t>(rows*cols/16);auto*x=alloc<float>(cols);auto*a=alloc<float>(rows);auto*b=alloc<float>(rows);auto*u=alloc<float>(rows);auto*v=alloc<float>(rows);auto*aa=alloc<float>(rows);auto*bb=alloc<float>(rows);auto*f=alloc<unsigned>(2);auto*selected=alloc<unsigned>(8);auto*weights=alloc<float>(8);
 std::vector<std::uint8_t> hw(rows*cols/2),hs(rows*cols/16),huw(rows*cols/2),hus(rows*cols/16);for(std::size_t j=0;j<hw.size();++j)hw[j]=unsigned(j*17+31)%256;for(std::size_t j=0;j<hs.size();++j)hs[j]=unsigned(j*13)%127;for(std::size_t j=0;j<huw.size();++j)huw[j]=unsigned(j*29+13)%256;for(std::size_t j=0;j<hus.size();++j)hus[j]=unsigned(j*7+4)%127;upload(w,hw);upload(s,hs);upload(uw,huw);upload(us,hus);upload(selected,std::vector<unsigned>{6,2,7,3,1,0,5,4});upload(weights,std::vector<float>{.125f,.25f,.0625f,.5f,.03125f,.25f,0.f,.125f});
 for(bool bf16:{false,true})for(unsigned mode=0;mode<7;++mode){std::vector<float> hx(cols),acc(rows);for(unsigned j=0;j<cols;++j)hx[j]=float(int(j%13)-6)/8.f;for(unsigned j=0;j<rows;++j)acc[j]=float(int(j%11)-5)/8.f;
 float global=.75f,other=-.5f;if(mode==1)global=std::numeric_limits<float>::max();if(mode==2)other=std::numeric_limits<float>::max();if(mode==3)hx[0]=INFINITY;if(mode==4)hx[0]=NAN;if(mode==5)for(auto&z:acc)z=std::numeric_limits<float>::max();if(mode==6)global=0.f;upload(x,hx);ck(cudaMemset(f,0,8));
 auto fn=bf16?kadan_launch_nvfp4_bf16:kadan_launch_nvfp4;ck(fn(w,s,global,x,a,f,rows,cols));ck(fn(uw,us,other,x,u,f,rows,cols));ck(kadan_launch_nvfp4_pair(w,s,global,x,b,f+1,rows,cols,bf16,{uw,us,other,v}));ck(cudaDeviceSynchronize());exact(a,b,rows);exact(u,v,rows);exact(f,f+1,1);
 upload(aa,acc);upload(bb,acc);ck(cudaMemset(f,0,8));ck(fn(w,s,global,x,a,f,rows,cols));kadan::moe::Config c{};c.hidden=rows;c.top_k=8;kadan::cuda::detail::MoeBuffers mb{};mb.down=a;mb.accumulator=aa;mb.selected=selected;mb.top_weights=weights;mb.status=f;const unsigned expert=mode==6?99:unsigned(mode);
 ck(kadan::cuda::detail::moe_accumulate(c,mb,expert));ck(kadan_launch_nvfp4_accumulate(w,s,global,x,b,f+1,rows,cols,bf16,{selected,weights,8,expert,bb}));ck(cudaDeviceSynchronize());exact(a,b,rows);exact(aa,bb,rows);exact(f,f+1,1);++cases;}
 for(void*p:owned)ck(cudaFree(p));owned.clear();}
 std::cout<<"PASS: "<<cases<<" fused pair/accumulation cases: distinct gate/up weight and scale buffers, both BF16 modes, shape boundaries, bounded/fallback scales, Inf/NaN, accumulator overflow and absent expert\n";
 }catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
