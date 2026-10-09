#include "kadan/tts.hpp"
#include <bit>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>
int main(int argc,char** argv) {
    static_assert(std::endian::native==std::endian::little && sizeof(float)==4);
    try {
        if(argc!=4) throw std::runtime_error("usage: kadan-tts-component ROOT SHARD INPUT_F32LE");
        auto resources=std::make_shared<kadan::Resources>(kadan::Footprint{64*1024*1024});
        std::array<float,2048> input;
        std::ifstream source(argv[3],std::ios::binary|std::ios::ate);
        if(!source || source.tellg()!=sizeof(input)) throw std::runtime_error("tts_input_size");
        struct Reservation {std::shared_ptr<kadan::Resources> r;kadan::Handle h;~Reservation(){r->released(h);}};
        std::array<float,2048> output;
        {
            Reservation admission{resources,resources->reserve(kadan::Workload::tts,{sizeof(input)})};
            source.seekg(0);source.read(reinterpret_cast<char*>(input.data()),sizeof(input));
            if(!source) throw std::runtime_error("tts_input_read");
            std::atomic_bool cancel{false};kadan::tts::TextProjection heads(resources);
            heads.load(argv[1],argv[2],cancel);
            output=heads.execute(input,cancel);
        }
        std::cout<<std::setprecision(9)<<"{\"projected\":[";
        for(std::size_t i=0;i<output.size();++i){if(i)std::cout<<',';std::cout<<output[i];}
        std::cout<<"],\"full_tts_generation\":false,\"gpu_execution\":false,\"resident_bytes\":"<<resources->snapshot().used[0]<<"}\n";
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
