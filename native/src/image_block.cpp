#include "kadan/image.hpp"
#include <bit>
#include <fstream>
#include <iostream>
#include <vector>
#include <fcntl.h>
#include <unistd.h>
namespace{void check(bool b,const char* e){if(!b)throw std::runtime_error(e);}void read(const char* p,std::span<float> values){std::ifstream f(p,std::ios::binary|std::ios::ate);check(bool(f)&&f.tellg()==std::streamoff(values.size_bytes()),"image_file_size");f.seekg(0);f.read(reinterpret_cast<char*>(values.data()),values.size_bytes());check(bool(f),"image_file_read");}}
int main(int argc,char** argv){static_assert(std::endian::native==std::endian::little);try{
 check(argc==7,"usage: kadan-image-block ROOT SHARD CONFIG INPUT_F32 MODULATION_F32 OUTPUT_F32");kadan::image::BlockConfig d;std::size_t text=0,height=0,width=0;std::ifstream f(argv[3]);f>>d.state>>d.heads>>d.head_dim>>d.intermediate>>d.axes[0]>>d.axes[1]>>d.axes[2]>>text>>height>>width;std::string extra;check(bool(f)&&!(f>>extra),"image_config_file");
 auto r=std::make_shared<kadan::Resources>(kadan::Footprint{4ULL*1024*1024*1024});std::atomic_bool cancel{false};{
 kadan::image::TransformerBlock block(r);block.load(argv[1],argv[2],0,d,cancel);check(text>0&&text<=2048&&height>0&&height<=256&&width>0&&width<=256&&text+height*width<=32768,"image_token_layout");const auto count=(text+height*width)*d.state;
 struct Admission{kadan::Resources& r;kadan::Handle h;~Admission(){r.released(h);}} admitted{*r,r->reserve(kadan::Workload::image,{4*(2*count+8*d.state)})};std::vector<float> input(count),mod(8*d.state),output(count);read(argv[4],input);read(argv[5],mod);block.execute(input,mod,text,height,width,output,cancel);block.unload();
 int fd=open(argv[6],O_CREAT|O_EXCL|O_WRONLY|O_CLOEXEC|O_NOFOLLOW,0600);check(fd>=0,"image_output_create");std::size_t at=0;auto bytes=reinterpret_cast<const char*>(output.data());while(at<output.size()*4){auto n=write(fd,bytes+at,output.size()*4-at);if(n<=0){close(fd);unlink(argv[6]);throw std::runtime_error("image_output_write");}at+=n;}if(close(fd)!=0){unlink(argv[6]);throw std::runtime_error("image_output_close");}
 }
 std::cout<<"{\"resident_bytes\":"<<r->snapshot().used[0]<<",\"gpu_execution\":false,\"full_image_generation\":false}\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
