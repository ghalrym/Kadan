#include "kadan/h3_generation.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>
using namespace kadan;
using namespace kadan::video;
void check(bool b){if(!b)throw std::runtime_error("pipeline_test_failed");}
int main(int argc,char** argv){try{check(argc==2);std::filesystem::path root(argv[1]);auto path=[&](const char* n){return (root/n).string();};H3GenerationPaths paths{path("tokenizer.json"),path("text.safetensors"),path("base.safetensors"),path("turbo.safetensors"),path("vae.safetensors")};auto r=std::make_shared<Resources>(Footprint{1024ULL*1024*1024});H3Generation generation(r);H3GenerationRequest req;req.prompt=std::string(1,'\0');req.output=path("result.y4m");req.width=32;req.height=32;req.updates=1;std::atomic_bool cancel=false;
 bool cancelled=false;try{generation.execute(paths,req,cancel,[&](const char* stage,std::size_t){if(std::string_view(stage)=="tokenized")cancel=true;});}catch(const std::exception& e){cancelled=std::string_view(e.what())=="h3_text_cancelled";}check(cancelled&&!std::filesystem::exists(req.output)&&r->snapshot().used[0]==0);cancel=false;
 Bytes peak=0;std::size_t steps=0;generation.execute(paths,req,cancel,[&](const char* stage,std::size_t count){peak=std::max(peak,r->snapshot().used[0]);if(std::string_view(stage)=="denoise_completed")steps=count;});check(steps==1&&peak>260000000&&r->snapshot().used[0]==0);std::ifstream file(req.output,std::ios::binary);std::string line;std::getline(file,line);check(line=="YUV4MPEG2 W32 H32 F24:1 Ip A1:1 C444 XCOLORRANGE=FULL");for(int f=0;f<5;++f){std::getline(file,line);check(line=="FRAME");std::string pixels(32*32*3,'\0');file.read(pixels.data(),pixels.size());check(bool(file));}check(file.peek()==std::char_traits<char>::eof());for(const auto& entry:std::filesystem::directory_iterator(root))check(!entry.path().filename().string().starts_with(".kadan-h3-"));std::cout<<"PASS complete tokenizer, 50 text layers, 2 refiners, 50 denoiser blocks, Euler update, 36 VAE blocks, temporal crop and 5 frames; sampled_peak="<<peak<<" final=0\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
