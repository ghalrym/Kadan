#include "h3_cli.hpp"
#include "kadan/h3_generation.hpp"
#include "kadan/h3_profile.hpp"
#include <charconv>
#include <csignal>
#include <cstdlib>
#include <iostream>
namespace {std::atomic_bool cancel=false;void signal_handler(int){cancel.store(true);}std::size_t number(const char* s){std::string_view text(s);std::size_t n=0;auto r=std::from_chars(text.data(),text.data()+text.size(),n);if(r.ec!=std::errc{}||r.ptr!=text.data()+text.size())throw std::invalid_argument("h3_cli_integer");return n;}}
int main(int argc,char** argv){try{
    if(argc!=8&&argc!=11)throw std::runtime_error("usage: kadan-h3-generate TOKENIZER TEXT DENOISER TURBO VAE PROMPT OUTPUT [SHORT_EDGE ASPECT SECONDS]; KADAN_H3_DEVICES selects CUDA");
    std::signal(SIGINT,signal_handler);std::signal(SIGTERM,signal_handler);auto execution=kadan::video::h3_execution(1536ULL*1024*1024);auto resources=execution.resources;
    kadan::video::H3Generation generation(resources,execution.compute);kadan::video::H3GenerationRequest request;request.prompt=argv[6];request.output=argv[7];
    if(argc==11){const auto p=kadan::video::h3_api_profile(number(argv[8]),argv[9],number(argv[10]));request.width=p.width;request.height=p.height;request.frames=p.frames;request.updates=4;}
    const char* audio=std::getenv("KADAN_H3_AUDIO_VAE");
    generation.execute({argv[1],argv[2],argv[3],argv[4],argv[5],audio?audio:""},request,cancel,[](const char* stage,std::size_t count){std::cerr<<stage<<' '<<count<<'\n';});
    std::cout<<"{\"frames\":"<<request.frames<<",\"width\":"<<request.width<<",\"height\":"<<request.height<<",\"updates\":"<<request.updates<<",\"audio\":"<<(audio&&*audio?"true":"false")<<",\"cuda\":"<<(execution.compute?"true":"false")<<",\"resident_bytes\":"<<resources->snapshot().used[0]<<"}\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
