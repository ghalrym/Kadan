#include "kadan/tts_audio.hpp"
#include <bit>
#include <fstream>
#include <iostream>
#include <vector>
#include <fcntl.h>
#include <unistd.h>
int main(int argc,char** argv){
    static_assert(std::endian::native==std::endian::little);
    try{
        if(argc!=6)throw std::runtime_error("usage: kadan-tts-audio-decode ROOT SHARD CONFIG CODES_U32 OUTPUT_F32");
        kadan::tts::AudioConfig d;std::ifstream configuration(argv[3]);configuration>>d.codebooks>>d.entries>>d.code_dim>>d.latent>>d.state>>d.heads>>d.head_dim>>d.layers>>d.intermediate>>d.decoder>>d.window;std::string extra;
        if(!configuration||(configuration>>extra))throw std::runtime_error("tts_audio_config_file");
        auto r=std::make_shared<kadan::Resources>(kadan::Footprint{2ULL*1024*1024*1024});std::size_t frames=0;
        {
            std::atomic_bool cancel{false};kadan::tts::AudioDecoder decoder(r);decoder.load(argv[1],argv[2],d,cancel);
            std::ifstream source(argv[4],std::ios::binary|std::ios::ate);if(!source)throw std::runtime_error("tts_audio_input_file");const auto bytes=source.tellg();
            if(bytes<=0||bytes%std::streamoff(d.codebooks*4)!=0||bytes>std::streamoff(d.codebooks*4*300))throw std::runtime_error("tts_audio_code_shape");frames=std::size_t(bytes)/(d.codebooks*4);
            struct Admission{kadan::Resources& r;kadan::Handle h;~Admission(){r.released(h);}} admitted{*r,r->reserve(kadan::Workload::tts,{std::size_t(bytes)+frames*1920*4})};
            std::vector<std::uint32_t> codes(std::size_t(bytes)/4);std::vector<float> output(frames*1920);source.seekg(0);source.read(reinterpret_cast<char*>(codes.data()),bytes);if(!source)throw std::runtime_error("tts_audio_input_read");
            decoder.decode(codes,output,cancel);decoder.unload();
            int fd=open(argv[5],O_CREAT|O_EXCL|O_WRONLY|O_CLOEXEC|O_NOFOLLOW,0600);if(fd<0)throw std::runtime_error("tts_audio_output_create");auto count=write(fd,output.data(),output.size()*4);const auto closed=close(fd);if(count!=std::ptrdiff_t(output.size()*4)||closed){unlink(argv[5]);throw std::runtime_error("tts_audio_output_write");}
        }
        std::cout<<"{\"frames\":"<<frames<<",\"samples\":"<<frames*1920<<",\"sample_rate\":24000,\"full_tts_generation\":false,\"gpu_execution\":false,\"resident_bytes\":"<<r->snapshot().used[0]<<"}\n";
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
