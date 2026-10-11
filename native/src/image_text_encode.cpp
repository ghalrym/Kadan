#include "kadan/image_text.hpp"
#include <bit>
#include <fstream>
#include <iostream>
#include <vector>
#include <fcntl.h>
#include <unistd.h>
namespace{void check(bool b,const char* e){if(!b)throw std::runtime_error(e);}}
int main(int argc,char** argv){static_assert(std::endian::native==std::endian::little);try{
 check(argc>=6&&argc<=9,"usage: kadan-image-text ROOT CONFIG IDS_U32 OUTPUT_F32 SHARD...");kadan::image::TextConfig d;std::ifstream config(argv[2]);config>>d.vocabulary>>d.state>>d.layers>>d.heads>>d.kv_heads>>d.head_dim>>d.intermediate;std::string extra;check(bool(config)&&!(config>>extra),"image_text_config_file");std::vector<std::string> files;for(int i=5;i<argc;++i)files.emplace_back(argv[i]);auto r=std::make_shared<kadan::Resources>(kadan::Footprint{64ULL*1024*1024*1024});std::atomic_bool cancel{false};{
 kadan::image::TextEncoder model(r);model.load(argv[1],files,d,cancel);std::ifstream source(argv[3],std::ios::binary|std::ios::ate);check(bool(source)&&source.tellg()>0&&source.tellg()%4==0&&source.tellg()<=2048*4,"image_text_input_size");auto count=std::size_t(source.tellg())/4;
 struct Admission{kadan::Resources& r;kadan::Handle h;~Admission(){r.released(h);}} admitted{*r,r->reserve(kadan::Workload::image,{count*4*(1+d.state)})};std::vector<std::uint32_t> ids(count);std::vector<float> output(count*d.state);source.seekg(0);source.read(reinterpret_cast<char*>(ids.data()),count*4);check(bool(source),"image_text_input_read");model.execute(ids,output,cancel,[](std::size_t i){std::cerr<<"text_layer "<<i<<'\n';});model.unload();
 int fd=open(argv[4],O_CREAT|O_EXCL|O_WRONLY|O_CLOEXEC|O_NOFOLLOW,0600);check(fd>=0,"image_text_output_create");std::size_t at=0;auto bytes=reinterpret_cast<const char*>(output.data());while(at<output.size()*4){auto n=write(fd,bytes+at,output.size()*4-at);if(n<=0){close(fd);unlink(argv[4]);throw std::runtime_error("image_text_output_write");}at+=n;}if(close(fd)!=0){unlink(argv[4]);throw std::runtime_error("image_text_output_close");}
 }
 std::cout<<"{\"resident_bytes\":"<<r->snapshot().used[0]<<",\"gpu_execution\":false,\"pre_final_norm\":true,\"full_image_generation\":false}\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
