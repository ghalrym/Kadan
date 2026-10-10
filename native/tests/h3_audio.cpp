#include "kadan/h3_audio.hpp"
#include "dense_reference.hpp"
#include <fstream>
#include <iostream>
#include <limits>
#include <vector>
using namespace kadan;
void check(bool b){if(!b)throw std::runtime_error("h3_audio_test_failed");}
template<class F>void rejects(F f,const char* e){try{f();}catch(const std::exception& x){if(x.what()!=std::string(e))throw;return;}throw std::runtime_error("expected_failure");}
int main(int argc,char** argv){try{
 auto r=std::make_shared<Resources>(Footprint{2ULL<<30});std::atomic_bool cancel=false;video::H3AudioConfig d{8,128};
 if(argc==6){if(std::string(argv[5])=="full")d={};else check(std::string(argv[5])=="small");std::ifstream in(argv[2],std::ios::binary|std::ios::ate);check(bool(in)&&in.tellg()>0&&in.tellg()%(64*4)==0);const auto count=std::size_t(in.tellg())/4,time=count/64;check(time<=video::H3AudioDecoder::max_latents);const auto caller=r->reserve(Workload::video,{(count+time*1600)*4});{
  std::vector<float> input(count),output(time*1600);in.seekg(0);in.read(reinterpret_cast<char*>(input.data()),count*4);check(bool(in));video::H3AudioDecoder decoder(r);decoder.load(argv[1],argv[4],d,cancel);decoder.decode(input,time,output,cancel);decoder.unload();std::ofstream out(argv[3],std::ios::binary);out.write(reinterpret_cast<char*>(output.data()),output.size()*4);check(bool(out));
 }r->released(caller);check(r->snapshot().residents==0);return 0;}
 check(argc==2);auto compute=test_dense();video::H3AudioDecoder decoder(r,compute);std::vector<float> input(128),output(3200,99);auto caller=r->reserve(Workload::video,{(128+3200)*4});
 rejects([&]{decoder.decode(input,2,output,cancel);},"h3_audio_not_loaded");cancel=true;rejects([&]{decoder.load(argv[1],"weights.safetensors",d,cancel);},"h3_audio_cancelled");cancel=false;
 for(auto pair:{std::pair{"wrong.safetensors","h3_audio_tensor_layout"},std::pair{"nan.safetensors","h3_audio_nonfinite_weight"},std::pair{"std.safetensors","h3_audio_std"}}){rejects([&]{decoder.load(argv[1],pair.first,d,cancel);},pair.second);check(r->snapshot().residents==1);}
 decoder.load(argv[1],"weights.safetensors",d,cancel);auto baseline=r->snapshot().used;std::size_t hooks=0;
 decoder.decode(input,2,output,cancel,[&](const char*,std::size_t){++hooks;rejects([&]{decoder.unload();},"busy");});check(hooks==18);for(float v:output)check(v==0);check(r->snapshot().used==baseline);
 rejects([&]{decoder.decode({},0,{},cancel);},"h3_audio_shape");rejects([&]{decoder.decode(input,641,output,cancel);},"h3_audio_shape");rejects([&]{decoder.decode(input,2,{},cancel);},"h3_audio_shape");
 rejects([&]{decoder.decode(std::span(output).first(128),2,output,cancel);},"h3_audio_alias");
 input[0]=std::numeric_limits<float>::infinity();rejects([&]{decoder.decode(input,2,output,cancel);},"h3_audio_nonfinite");input[0]=0;
 auto pressure=r->reserve(Workload::video,{r->snapshot().capacity[0]-baseline[0]});rejects([&]{decoder.decode(input,2,output,cancel);},"exhausted");r->released(pressure);check(r->snapshot().used==baseline);
 rejects([&]{decoder.decode(input,2,output,cancel,[&](const char* p,std::size_t i){if(std::string(p)=="upsample"&&i==1)cancel=true;});},"h3_audio_cancelled");cancel=false;check(r->snapshot().used==baseline);
 rejects([&]{decoder.decode(input,2,output,cancel,[](const char*,std::size_t){throw std::runtime_error("hook_failure");});},"hook_failure");check(r->snapshot().used==baseline);
 if(compute){compute->fail=true;rejects([&]{decoder.decode(input,2,output,cancel);},"test_compute_failure");compute->fail=false;check(r->snapshot().used==baseline);}
 decoder.unload();decoder.unload();r->released(caller);check(r->snapshot().residents==0);if(compute)check(compute->calls>0);std::cout<<"PASS H3 audio shapes, pinning, cancellation, failure cleanup and bounded admission\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
