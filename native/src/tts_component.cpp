#include "kadan/tts.hpp"
#include <bit>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>
int main(int argc,char** argv) {
    static_assert(std::endian::native==std::endian::little && sizeof(float)==4);
    try {
        if(argc!=4 && argc!=5) throw std::runtime_error("usage: kadan-tts-component ROOT SHARD INPUT_F32LE [--sequence]");
        const bool sequence=argc==5;
        if(sequence && std::string(argv[4])!="--sequence") throw std::runtime_error("tts_mode");
        auto resources=std::make_shared<kadan::Resources>(kadan::Footprint{96*1024*1024});
        constexpr std::size_t row_bytes=kadan::tts::TextProjection::hidden*sizeof(float);
        std::ifstream source(argv[3],std::ios::binary|std::ios::ate);
        if(!source) throw std::runtime_error("tts_input_size");
        const auto bytes=source.tellg();
        if(bytes<=0 || bytes%row_bytes!=0 || bytes>std::streamoff(row_bytes*(sequence ? kadan::tts::TextProjection::max_tokens : 1)))
            throw std::runtime_error("tts_input_size");
        const auto count=static_cast<std::size_t>(bytes)/sizeof(float);
        struct Reservation {std::shared_ptr<kadan::Resources> r;kadan::Handle h;~Reservation(){r->released(h);}};
        std::cout.exceptions(std::ios::badbit|std::ios::failbit);
        {
            Reservation admission{resources,resources->reserve(kadan::Workload::tts,{2*count*sizeof(float)})};
            std::vector<float> input(count),output(count);
            source.seekg(0);source.read(reinterpret_cast<char*>(input.data()),bytes);
            if(!source) throw std::runtime_error("tts_input_read");
            std::atomic_bool cancel{false};kadan::tts::TextProjection heads(resources);
            heads.load(argv[1],argv[2],cancel);
            heads.execute_sequence(input,output,cancel);
            heads.unload();
            std::cout<<std::setprecision(9)<<"{\"projected\":[";
            for(std::size_t i=0;i<output.size();++i){if(i)std::cout<<',';std::cout<<output[i];}
            std::cout<<"],\"resident_bytes_at_publication\":"<<resources->snapshot().used[0];
            std::cout.flush(); // Keep caller admission until output has been published.
            std::vector<float>().swap(output);std::vector<float>().swap(input); // Physical heap release before ledger release.
        }
        std::cout<<",\"tokens\":"<<count/kadan::tts::TextProjection::hidden<<",\"full_tts_generation\":false,\"gpu_execution\":false,\"resident_bytes\":"<<resources->snapshot().used[0]<<"}\n";
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
