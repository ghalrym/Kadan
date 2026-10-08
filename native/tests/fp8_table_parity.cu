#include "fp8_reference.cuh"
#include <charconv>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <vector>
void ck(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
int main(int argc,char**argv){
 if(argc!=4||std::string_view(argv[1])!="--allow-gpu-validation"||std::string_view(argv[2])!="--device")return 2;
 try{int device=-1;std::string_view arg(argv[3]);auto[end,error]=std::from_chars(arg.data(),arg.data()+arg.size(),device);if(error!=std::errc{}||end!=arg.data()+arg.size()||device<0||device>=64)return 2;ck(cudaSetDevice(device));unsigned cases=0;
 for(unsigned rows:{1u,3u,257u,4096u})for(unsigned cols:{17u,2048u,4096u}){
 unsigned char*w;float *scales,*x,*a,*b;unsigned*f;ck(cudaMalloc(&w,rows*cols));ck(cudaMalloc(&scales,rows*4));ck(cudaMalloc(&x,cols*4));ck(cudaMalloc(&a,rows*4));ck(cudaMalloc(&b,rows*4));ck(cudaMalloc(&f,8));
 std::vector<unsigned char> hw(rows*cols);std::vector<float> hs(rows),hx(cols),ha(rows),hb(rows);
 for(bool bf16:{false,true})for(bool row_scales:{false,true})for(unsigned mode=0;mode<8;++mode){
 for(std::size_t j=0;j<hw.size();++j){hw[j]=(j*17+11)%256;if((hw[j]&127)==127)--hw[j];if(mode==7)hw[j]=0;}
 for(unsigned j=0;j<cols;++j)hx[j]=float(int(j%13)-6)/8.f;
 for(unsigned j=0;j<rows;++j){hs[j]=j%2?-.5f:.75f;if(mode==1)hs[j]=0.f;if(mode==2)hs[j]=std::numeric_limits<float>::denorm_min();if(mode==5||mode==7)hs[j]=std::numeric_limits<float>::max();if(mode==6)hs[j]=std::numeric_limits<float>::max()/512.f;}
 if(mode==3)hx[0]=INFINITY;if(mode==4)hx[0]=NAN;
 ck(cudaMemcpy(w,hw.data(),hw.size(),cudaMemcpyHostToDevice));ck(cudaMemcpy(scales,hs.data(),rows*4,cudaMemcpyHostToDevice));ck(cudaMemcpy(x,hx.data(),cols*4,cudaMemcpyHostToDevice));ck(cudaMemset(f,0,8));
 auto old=bf16?old_launch_fp8_bf16:old_launch_fp8;auto candidate=bf16?kadan_launch_fp8_bf16:kadan_launch_fp8;ck(old(w,scales,row_scales,x,a,f,rows,cols));ck(candidate(w,scales,row_scales,x,b,f+1,rows,cols));ck(cudaDeviceSynchronize());ck(cudaMemcpy(ha.data(),a,rows*4,cudaMemcpyDeviceToHost));ck(cudaMemcpy(hb.data(),b,rows*4,cudaMemcpyDeviceToHost));unsigned flags[2];ck(cudaMemcpy(flags,f,8,cudaMemcpyDeviceToHost));if(std::memcmp(ha.data(),hb.data(),rows*4)||flags[0]!=flags[1])throw std::runtime_error("FP8 table bitwise mismatch");if(mode==7&&flags[1])throw std::runtime_error("unused table entry incorrectly flagged");++cases;}
 for(void*p:{(void*)w,(void*)scales,(void*)x,(void*)a,(void*)b,(void*)f})ck(cudaFree(p));}
 std::cout<<"PASS: "<<cases<<" FP8 table cases: row/scalar scales, both BF16 modes, partial blocks/columns, signs/zero/subnormal/overflow/Inf/NaN and unused overflowing table entries\n";
 }catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
