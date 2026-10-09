#include "kadan/decision.hpp"
#include <bit>
#include <fstream>
#include <iomanip>
#include <iostream>
int main(int argc,char** argv) {
    static_assert(std::endian::native==std::endian::little && sizeof(float)==4);
    try {
        if(argc!=4) throw std::runtime_error("usage: kadan-decision-component ROOT SHARD INPUT_F32LE");
        auto resources=std::make_shared<kadan::Resources>(kadan::Footprint{32*1024*1024});
        using Input=std::array<float,2052>;
        using Output=std::array<float,3>;
        std::ifstream source(argv[3],std::ios::binary|std::ios::ate);
        if(!source || source.tellg()!=sizeof(Input)) throw std::runtime_error("decision_input_size");
        struct Reservation {std::shared_ptr<kadan::Resources> r;kadan::Handle h;~Reservation(){r->released(h);}};
        std::cout.exceptions(std::ios::badbit|std::ios::failbit);
        {
            Reservation admission{resources,resources->reserve(kadan::Workload::decision,{sizeof(Input)+sizeof(Output)})};
            auto input=std::make_unique<Input>();
            source.seekg(0);source.read(reinterpret_cast<char*>(input->data()),sizeof(Input));
            if(!source) throw std::runtime_error("decision_input_read");
            std::atomic_bool cancel{false};kadan::decision::TerminalHeads heads(resources);
            heads.load(argv[1],argv[2],cancel);
            // Direct prvalue initialization elides an additional retained output copy.
            std::unique_ptr<Output> output(new Output(heads.execute({input->data(),1024},{input->data()+1024,1028},cancel)));
            heads.unload();
            std::cout<<std::setprecision(9)<<"{\"score\":"<<(*output)[0]<<",\"actions\":["<<(*output)[1]<<','<<(*output)[2]
                     <<"],\"resident_bytes_at_publication\":"<<resources->snapshot().used[0];
            std::cout.flush(); // Keep caller admission until output has been published.
            output.reset();input.reset(); // Physical heap release before ledger release.
        }
        std::cout<<",\"full_decision_inference\":false,\"gpu_execution\":false,\"resident_bytes\":"<<resources->snapshot().used[0]<<"}\n";
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
