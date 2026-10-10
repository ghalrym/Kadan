#include "kadan/image_vae.hpp"
#include <fstream>
#include <iostream>
#include <vector>
#include <fcntl.h>
#include <unistd.h>
namespace {void need(bool b,const char* e){if(!b)throw std::runtime_error(e);}}
int main(int argc,char** argv){try{need(argc==6,"usage: kadan-image-decode ROOT SHARD CONFIG INPUT_F32 OUTPUT_F32");kadan::image::VaeConfig d;std::size_t h,w;std::ifstream config(argv[3]);config>>d.base>>d.latent>>d.channels>>d.residuals>>h>>w;std::string extra;need(bool(config)&&!(config>>extra)&&h>0&&w>0&&h<=128&&w<=128&&h*w<=4096&&d.latent<=64&&d.channels<=4,"image_vae_config_file");auto r=std::make_shared<kadan::Resources>(kadan::Footprint{8ULL*1024*1024*1024});std::atomic_bool cancel{false};{
 kadan::image::VaeDecoder model(r);model.load(argv[1],argv[2],d,cancel);auto n=d.latent*h*w,m=d.channels*h*w*256;struct Admission{kadan::Resources& r;kadan::Handle h;~Admission(){r.released(h);}} admitted{*r,r->reserve(kadan::Workload::image,{(n+m)*4})};std::vector<float> input(n),output(m);std::ifstream source(argv[4],std::ios::binary|std::ios::ate);need(bool(source)&&source.tellg()==std::streamoff(n*4),"image_vae_input_size");source.seekg(0);source.read(reinterpret_cast<char*>(input.data()),n*4);need(bool(source),"image_vae_input_read");model.decode(input,h,w,output,cancel,[](std::size_t i){std::cerr<<"vae_up_block "<<i<<'\n';});model.unload();int fd=open(argv[5],O_CREAT|O_EXCL|O_WRONLY|O_CLOEXEC|O_NOFOLLOW,0600);need(fd>=0,"image_vae_output_create");auto bytes=reinterpret_cast<const char*>(output.data());std::size_t at=0;while(at<m*4){auto count=write(fd,bytes+at,m*4-at);if(count<=0){close(fd);unlink(argv[5]);throw std::runtime_error("image_vae_output_write");}at+=count;}if(close(fd)!=0){unlink(argv[5]);throw std::runtime_error("image_vae_output_close");}}
 std::cout<<"{\"resident_bytes\":"<<r->snapshot().used[0]<<",\"gpu_execution\":false,\"full_vae_decoder\":true,\"full_image_generation\":false}\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
