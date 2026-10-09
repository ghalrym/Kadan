#include "kadan/h3_denoiser.hpp"
#include <bit>
#include <csignal>
#include <fstream>
#include <iostream>
#include <fcntl.h>
#include <unistd.h>
namespace {
std::atomic_bool cancelled=false;
void signal_handler(int){cancelled.store(true);}
void check(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}
struct Artifact {
    std::string temporary;int fd;
    explicit Artifact(const std::string& name):temporary(name+".partial-XXXXXX"),fd(mkstemp(temporary.data())){check(fd>=0,"output_open");}
    ~Artifact(){if(fd>=0)close(fd);unlink(temporary.c_str());}
    void write(const void* p,std::size_t n){auto* b=static_cast<const char*>(p);while(n){if(cancelled.load())throw std::runtime_error("cancelled");auto k=::write(fd,b,n);if(k<0&&errno==EINTR)continue;check(k>0,"output_write");b+=k;n-=k;}}
    void publish(const std::string& name){check(!cancelled.load(),"cancelled");check(fsync(fd)==0,"output_sync");check(link(temporary.c_str(),name.c_str())==0,"output_publish");}
};
}
int main(int argc,char** argv){
    try{
        check(argc==5 || argc==7,"usage: kadan-h3-denoise ROOT SHARD INPUT OUTPUT [TURBO_ROOT TURBO_SHARD]");
        static_assert(std::endian::native==std::endian::little);std::signal(SIGINT,signal_handler);std::signal(SIGTERM,signal_handler);
        std::ifstream input(argv[3],std::ios::binary);std::string magic;std::getline(input,magic);check(magic=="KADAN_H3_DENOISER_INPUT_V1","input_header");std::size_t nt=0,nv=0,na=0;input>>nt>>nv>>na;check(bool(input) && input.get()=='\n',"input_shape");
        check(nt>0 && nt<=512 && nv>0 && nv<=1024 && na<=1024 && nt+nv+na<=1024,"input_limit");const auto n=nt+nv+na;const auto features=nt*5120+nv*96+na*32;const auto floats=features+n*4;
        auto resources=std::make_shared<kadan::Resources>(kadan::Footprint{1024ULL*1024*1024});const auto caller=resources->reserve(kadan::Workload::video,{(floats+nv*96+na*32)*4+n*4});
        {
            auto data=std::make_unique<float[]>(floats);auto tags=std::make_unique<std::uint32_t[]>(n);auto output=std::make_unique<float[]>(nv*96+na*32);
            input.read(reinterpret_cast<char*>(data.get()),floats*4);input.read(reinterpret_cast<char*>(tags.get()),n*4);check(bool(input) && input.peek()==std::char_traits<char>::eof(),"input_bytes");
            kadan::video::H3Denoiser model(resources);model.load(argv[1],argv[2],cancelled);if(argc==7)model.load_turbo(argv[5],argv[6],cancelled);
            kadan::video::H3Denoiser::Input request{{data.get(),nt*5120},{data.get()+nt*5120,nv*96},{data.get()+nt*5120+nv*96,na*32},{data.get()+features,n*3},{data.get()+features+n*3,n},{tags.get(),n}};
            model.execute(request,{output.get(),nv*96},{output.get()+nv*96,na*32},cancelled,[](const char* phase,std::size_t count){if(std::string_view(phase)=="block_completed")std::cerr<<"block "<<count<<"/50\n";});
            model.unload();Artifact artifact(argv[4]);const auto header="KADAN_H3_VELOCITY_F32_V1\n"+std::to_string(nv)+" "+std::to_string(na)+"\n";artifact.write(header.data(),header.size());artifact.write(output.get(),(nv*96+na*32)*4);artifact.publish(argv[4]);
        }
        resources->released(caller);std::cout<<"{\"blocks\":50,\"refiner_blocks\":2,\"resident_bytes\":"<<resources->snapshot().used[0]<<",\"full_video_generation\":false}\n";return 0;
    }catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
}
