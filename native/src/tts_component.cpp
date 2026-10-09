#include "kadan/tts.hpp"
#include <bit>
#include <fstream>
#include <iomanip>
#include <iostream>
int main(int argc,char** argv) {
    static_assert(std::endian::native==std::endian::little && sizeof(float)==4);
    try {
        if(argc!=4) throw std::runtime_error("usage: kadan-tts-component ROOT SHARD INPUT_F32LE");
        auto resources=std::make_shared<kadan::Resources>(kadan::Footprint{64*1024*1024});
        using Buffer=std::array<float,2048>;
        std::ifstream source(argv[3],std::ios::binary|std::ios::ate);
        if(!source || source.tellg()!=sizeof(Buffer)) throw std::runtime_error("tts_input_size");
        struct Reservation {std::shared_ptr<kadan::Resources> r;kadan::Handle h;~Reservation(){r->released(h);}};
        std::cout.exceptions(std::ios::badbit|std::ios::failbit);
        {
            Reservation admission{resources,resources->reserve(kadan::Workload::tts,{2*sizeof(Buffer)})};
            auto input=std::make_unique<Buffer>();
            source.seekg(0);source.read(reinterpret_cast<char*>(input->data()),sizeof(Buffer));
            if(!source) throw std::runtime_error("tts_input_read");
            std::atomic_bool cancel{false};kadan::tts::TextProjection heads(resources);
            heads.load(argv[1],argv[2],cancel);
            // Direct prvalue initialization elides an additional retained output copy.
            std::unique_ptr<Buffer> output(new Buffer(heads.execute(*input,cancel)));
            heads.unload();
            std::cout<<std::setprecision(9)<<"{\"projected\":[";
            for(std::size_t i=0;i<output->size();++i){if(i)std::cout<<',';std::cout<<(*output)[i];}
            std::cout<<"],\"resident_bytes_at_publication\":"<<resources->snapshot().used[0];
            std::cout.flush(); // Keep caller admission until output has been published.
            output.reset();input.reset(); // Physical heap release before ledger release.
        }
        std::cout<<",\"full_tts_generation\":false,\"gpu_execution\":false,\"resident_bytes\":"<<resources->snapshot().used[0]<<"}\n";
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
