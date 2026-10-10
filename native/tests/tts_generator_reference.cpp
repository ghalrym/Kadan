#include "kadan/tts_generator.hpp"
#include <iostream>
#include <vector>
int main(int argc,char** argv){try{
    if(argc!=3&&argc!=4)throw std::runtime_error("usage: reference ROOT FRAMES");
    auto r=std::make_shared<kadan::Resources>(kadan::Footprint{64*1024*1024});std::atomic_bool cancel{false};
    kadan::tts::CodeGenerator m(r);kadan::tts::GeneratorConfig d{{8,2,1,4,2,16},{4,2,1,4,2,8},32,24,4,16};
    kadan::tts::VoicePrompt v{{0,1,2},3,4,5,16,17,18,19,20,21,22,23};
    const auto frames=std::stoul(argv[2]);if(!frames||frames>8)throw std::runtime_error("frames");std::vector<std::uint32_t> codes(frames*4),text{6,7};
    const auto variant=argc==4?std::stoul(argv[3]):0;std::vector<std::uint32_t> instruction;
    if(variant&1){v.automatic_language=true;v.nothink=16;}
    if(variant&2){v.speaker=19;v.language=17;instruction={8,9,10};}
    m.load(argv[1],"weights.safetensors",d,cancel);auto result=m.generate(text,v,codes,cancel,{},frames,1.05f,instruction);m.unload();
    std::cout<<"{\"frames\":"<<result.frames<<",\"stopped\":"<<(result.stopped?"true":"false")<<",\"resident_bytes\":"<<r->snapshot().used[0]<<",\"codes\":[";
    for(std::size_t i=0;i<result.frames*4;++i){if(i)std::cout<<',';std::cout<<codes[i];}std::cout<<"]}\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
