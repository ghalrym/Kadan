#include "kadan/h3_generation.hpp"
#include <csignal>
#include <iostream>
namespace {std::atomic_bool cancel=false;void signal_handler(int){cancel.store(true);}}
int main(int argc,char** argv){try{if(argc!=8)throw std::runtime_error("usage: kadan-h3-generate TOKENIZER TEXT DENOISER TURBO VAE PROMPT OUTPUT.y4m");std::signal(SIGINT,signal_handler);std::signal(SIGTERM,signal_handler);auto resources=std::make_shared<kadan::Resources>(kadan::Footprint{1536ULL*1024*1024});kadan::video::H3Generation generation(resources);kadan::video::H3GenerationRequest request;request.prompt=argv[6];request.output=argv[7];generation.execute({argv[1],argv[2],argv[3],argv[4],argv[5]},request,cancel,[](const char* stage,std::size_t count){std::cerr<<stage<<' '<<count<<'\n';});std::cout<<"{\"frames\":22,\"width\":64,\"height\":64,\"updates\":8,\"audio\":false,\"resident_bytes\":"<<resources->snapshot().used[0]<<"}\n";return 0;}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
