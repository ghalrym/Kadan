#include "h3_cli.hpp"
#include "kadan/video.hpp"
#include <charconv>
#include <csignal>
#include <fstream>
#include <iostream>
#include <string_view>
#include <vector>
namespace {
std::atomic_bool cancelled{false};
static_assert(std::atomic_bool::is_always_lock_free);
void stop(int) {cancelled.store(true);}
std::size_t dimension(const char* text) {
    std::string_view value(text);std::size_t result=0;
    const auto parsed=std::from_chars(value.data(),value.data()+value.size(),result);
    if(parsed.ec!=std::errc{} || parsed.ptr!=value.data()+value.size() || !result || result>4096)
        throw std::runtime_error("video_latent_shape_or_limit");
    return result;
}
}
int main(int argc,char** argv) {
    try {
        if(argc!=8)throw std::runtime_error("usage: kadan-video-decode VAE_ROOT SHARD NORMALIZED_LATENTS_F32LE T H W OUTPUT_FRAMES");
        const auto t=dimension(argv[4]),h=dimension(argv[5]),w=dimension(argv[6]);
        if(t>4096/h || t*h>4096/w)throw std::runtime_error("video_latent_shape_or_limit");
        const auto elements=t*h*w*24;
        auto execution=kadan::video::h3_execution(1024ULL*1024*1024);auto resources=execution.resources;
        std::signal(SIGINT,stop);std::signal(SIGTERM,stop);
        struct Admission {kadan::Resources& r;kadan::Handle h;~Admission(){r.released(h);}};
        {
            Admission input{*resources,resources->reserve(kadan::Workload::video,kadan::video::h3_host(*resources,elements*sizeof(float)))};
            std::vector<float> latents(elements);
            std::ifstream stream(argv[3],std::ios::binary);
            stream.read(reinterpret_cast<char*>(latents.data()),elements*sizeof(float));
            if(!stream || stream.peek()!=std::char_traits<char>::eof())throw std::runtime_error("video_input_shape");
            kadan::video::H3VideoDecoder decoder(resources,execution.compute);
            decoder.load(argv[1],argv[2],cancelled);
            decoder.execute(latents,t,h,w,argv[7],cancelled,[](const char* stage,std::size_t count){
                if(std::string_view(stage)=="block_completed")std::cerr<<"decoded_block="<<count<<'\n';
            });
            decoder.unload();
        }
        std::cout<<"vae_decode_complete=true; full_video_generation=false;  resident_bytes="<<resources->snapshot().used[0]<<'\n';
        return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
