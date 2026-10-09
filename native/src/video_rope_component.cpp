#include "kadan/video.hpp"
#include <fstream>
#include <iostream>
#include <vector>
namespace {
std::size_t input_size(const char* path,std::size_t stride,std::size_t maximum) {
    std::ifstream stream(path,std::ios::binary|std::ios::ate);
    const auto bytes=stream.tellg();
    if (!stream || bytes<=0 || std::size_t(bytes)>maximum || std::size_t(bytes)%stride)
        throw std::runtime_error("input_shape_or_limit");
    return std::size_t(bytes);
}
void read(const char* path,std::vector<float>& values) {
    std::ifstream stream(path,std::ios::binary);
    stream.read(reinterpret_cast<char*>(values.data()),values.size()*sizeof(float));
    if (!stream || stream.peek()!=std::char_traits<char>::eof()) throw std::runtime_error("input_read");
}
}
int main(int argc,char** argv) {
    try {
        if(argc!=4) throw std::runtime_error("usage: kadan-video-rope-component QKV_F32LE COORDINATES_F32LE OUTPUT_COMPONENT");
        using Stage=kadan::video::H3QkRope;
        const auto qbytes=input_size(argv[1],Stage::width*4,Stage::max_tokens*Stage::width*4);
        const auto cbytes=input_size(argv[2],3*4,Stage::max_tokens*3*4);
        if(qbytes/(Stage::width*4)!=cbytes/(3*4)) throw std::runtime_error("coordinate_shape");
        auto resources=std::make_shared<kadan::Resources>(kadan::Footprint{256*1024});
        struct Admission {
            kadan::Resources& ledger; kadan::Handle handle;
            ~Admission(){ledger.released(handle);}
        };
        {
            Admission input{*resources,resources->reserve(kadan::Workload::video,{qbytes+cbytes})};
            std::vector<float> qkv(qbytes/4),coordinates(cbytes/4);
            read(argv[1],qkv);read(argv[2],coordinates);
            std::atomic_bool cancel{false};Stage stage(resources);stage.load(cancel);
            stage.execute(qkv,coordinates,argv[3],cancel);stage.unload();
        }
        std::cout<<"H3 QK RMSNorm/RoPE executed; full_video_generation=false; gpu_execution=false; resident_bytes="<<resources->snapshot().used[0]<<'\n';
        return 0;
    } catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
