#include "kadan/decision.hpp"
#include <bit>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>
int main(int argc,char** argv) {
    static_assert(std::endian::native==std::endian::little && sizeof(float)==4);
    try {
        if(argc!=4) throw std::runtime_error("usage: kadan-decision-component ROOT SHARD INPUT_F32LE");
        auto resources=std::make_shared<kadan::Resources>(kadan::Footprint{32*1024*1024});
        std::array<float,2052> input;
        std::ifstream source(argv[3],std::ios::binary|std::ios::ate);
        if(!source || source.tellg()!=sizeof(input)) throw std::runtime_error("decision_input_size");
        struct Reservation {std::shared_ptr<kadan::Resources> r;kadan::Handle h;~Reservation(){r->released(h);}};
        std::array<float,3> output;
        {
            Reservation admission{resources,resources->reserve(kadan::Workload::decision,{sizeof(input)})};
            source.seekg(0);source.read(reinterpret_cast<char*>(input.data()),sizeof(input));
            if(!source) throw std::runtime_error("decision_input_read");
            std::atomic_bool cancel{false};kadan::decision::TerminalHeads heads(resources);
            heads.load(argv[1],argv[2],cancel);
            output=heads.execute({input.data(),1024},{input.data()+1024,1028},cancel);
        }
        std::cout<<std::setprecision(9)<<"{\"score\":"<<output[0]<<",\"actions\":["<<output[1]<<','<<output[2]
                 <<"],\"full_decision_inference\":false,\"gpu_execution\":false,\"resident_bytes\":"<<resources->snapshot().used[0]<<"}\n";
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
