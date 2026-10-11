#include "h3_cli.hpp"
#include "kadan/h3_text.hpp"
#include <csignal>
#include <fstream>
#include <iostream>
#include <vector>
namespace {
std::atomic_bool cancelled{false};
static_assert(std::atomic_bool::is_always_lock_free);
void stop(int){cancelled.store(true);}
}
int main(int argc,char** argv) {
    try {
        if(argc!=5)throw std::runtime_error("usage: kadan-h3-text-encode CHECKPOINT_ROOT SHARD TOKEN_IDS_U32LE OUTPUT_FEATURES");
        std::ifstream stream(argv[3],std::ios::binary|std::ios::ate);
        const auto size=stream.tellg();
        if(!stream || size<=0 || size%4 || size>std::streamoff(kadan::video::H3TextEncoder::max_tokens*4))throw std::runtime_error("h3_text_token_limit");
        auto execution=kadan::video::h3_execution(512ULL*1024*1024);auto ledger=execution.resources;
        std::signal(SIGINT,stop);std::signal(SIGTERM,stop);
        struct Admission {kadan::Resources& r;kadan::Handle h;~Admission(){r.released(h);}};
        {
            Admission input{*ledger,ledger->reserve(kadan::Workload::video,kadan::video::h3_host(*ledger,std::size_t(size)))};
            std::vector<std::uint32_t> ids(std::size_t(size)/4);
            stream.seekg(0);stream.read(reinterpret_cast<char*>(ids.data()),size);
            if(!stream || stream.peek()!=std::char_traits<char>::eof())throw std::runtime_error("h3_text_input_read");
            kadan::video::H3TextEncoder encoder(ledger,execution.compute);encoder.load(argv[1],argv[2],cancelled);
            encoder.execute(ids,argv[4],cancelled,[](const char* stage,std::size_t count){
                if(std::string_view(stage)=="layer_completed")std::cerr<<"encoded_layer="<<count<<'\n';
            });
            encoder.unload();
        }
        std::cout<<"conditioning_layers=50; compute=selected_f32_convrot_int8; production_bf16_parity=false; full_video_generation=false; resident_bytes="<<ledger->snapshot().used[0]<<'\n';
        return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
