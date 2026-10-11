#include "kadan/image_denoiser.hpp"
#include <bit>
#include <fstream>
#include <iostream>
#include <vector>
#include <fcntl.h>
#include <unistd.h>
namespace{void check(bool b,const char* e){if(!b)throw std::runtime_error(e);}void read(const char* p,std::span<float> values){std::ifstream f(p,std::ios::binary|std::ios::ate);check(bool(f)&&f.tellg()==std::streamoff(values.size_bytes()),"image_file_size");f.seekg(0);f.read(reinterpret_cast<char*>(values.data()),values.size_bytes());check(bool(f),"image_file_read");}}
int main(int argc,char** argv){static_assert(std::endian::native==std::endian::little);try{
 check(argc>=7&&argc<=10,"usage: kadan-image-denoise ROOT CONFIG LATENT_F32 CONDITION_F32 OUTPUT_F32 SHARD...");kadan::image::DenoiserConfig d;std::size_t text=0,height=0,width=0;float timestep=0;std::ifstream f(argv[2]);f>>d.block.state>>d.block.heads>>d.block.head_dim>>d.block.intermediate>>d.block.axes[0]>>d.block.axes[1]>>d.block.axes[2]>>d.layers>>d.context>>d.channels>>text>>height>>width>>timestep;std::string extra;check(bool(f)&&!(f>>extra),"image_config_file");std::vector<std::string> files;for(int i=6;i<argc;++i)files.emplace_back(argv[i]);
 auto r=std::make_shared<kadan::Resources>(kadan::Footprint{96ULL*1024*1024*1024});std::atomic_bool cancel{false};{
 kadan::image::Denoiser model(r);std::cerr<<"native_load_started\n";model.load(argv[1],files,d,cancel);std::cerr<<"native_loaded\n";check(text>0&&text<=2048&&height>0&&height<=256&&width>0&&width<=256&&text+height*width<=32768,"image_token_layout");const auto count=height*width*d.channels;
 struct Admission{kadan::Resources& r;kadan::Handle h;~Admission(){r.released(h);}} admitted{*r,r->reserve(kadan::Workload::image,{4*(2*count+text*d.context)})};std::vector<float> input(count),condition(text*d.context),output(count);read(argv[3],input);read(argv[4],condition);model.execute(input,condition,text,height,width,timestep,output,cancel,[](const char* phase,std::size_t i){std::cerr<<phase<<' '<<i<<'\n';});model.unload();
 int fd=open(argv[5],O_CREAT|O_EXCL|O_WRONLY|O_CLOEXEC|O_NOFOLLOW,0600);check(fd>=0,"image_output_create");std::size_t at=0;auto bytes=reinterpret_cast<const char*>(output.data());while(at<output.size()*4){auto n=write(fd,bytes+at,output.size()*4-at);if(n<=0){close(fd);unlink(argv[5]);throw std::runtime_error("image_output_write");}at+=n;}if(close(fd)!=0){unlink(argv[5]);throw std::runtime_error("image_output_close");}
 }
 std::cout<<"{\"resident_bytes\":"<<r->snapshot().used[0]<<",\"gpu_execution\":false,\"full_denoiser\":true,\"full_image_generation\":false}\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
