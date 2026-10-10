#include "kadan/stt.hpp"
#include <bit>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <csignal>
namespace {
volatile std::sig_atomic_t interrupted=0;
void stop(int){interrupted=1;}
std::size_t size(std::ifstream& stream){
    if(!stream || stream.tellg()<0)throw std::runtime_error("stt_input_file");
    return static_cast<std::size_t>(stream.tellg());
}
void read(std::ifstream& stream,float* destination,std::size_t bytes){
    stream.seekg(0);stream.read(reinterpret_cast<char*>(destination),static_cast<std::streamsize>(bytes));
    if(!stream)throw std::runtime_error("stt_input_read");
}
}
int main(int argc,char** argv){
    static_assert(std::endian::native==std::endian::little && sizeof(float)==4);
    try {
        if(argc!=4 && argc!=5)throw std::runtime_error("usage: kadan-stt-component BINS FILTER_F32LE PCM16KHZ_F32LE [--window]");
        const bool window=argc==5;
        if(window && std::string(argv[4])!="--window")throw std::runtime_error("stt_mode");
        const std::string argument=argv[1];
        if(argument!="80" && argument!="128")throw std::runtime_error("stt_mel_bins");
        const std::size_t bins=argument=="80"?80:128;
        std::ifstream bank(argv[2],std::ios::binary|std::ios::ate),pcm(argv[3],std::ios::binary|std::ios::ate);
        const auto bank_bytes=size(bank),pcm_bytes=size(pcm);
        if(bank_bytes!=bins*201*4 || pcm_bytes%4)throw std::runtime_error("stt_input_size");
        const auto samples=pcm_bytes/4;
        if(samples>kadan::stt::LogMel::max_samples)throw std::runtime_error("stt_sample_count");
        const auto frames=kadan::stt::LogMel::frames(window?kadan::stt::LogMel::max_samples:samples),values=bins*frames;
        auto ledger=std::make_shared<kadan::Resources>(kadan::Footprint{8*1024*1024});
        struct Admission{std::shared_ptr<kadan::Resources> r;kadan::Handle h;~Admission(){r->released(h);}};
        std::cout.exceptions(std::ios::badbit|std::ios::failbit);
        std::signal(SIGINT,stop);std::signal(SIGTERM,stop);
        {
            Admission io{ledger,ledger->reserve(kadan::Workload::speech,{bank_bytes+pcm_bytes+values*4})};
            auto filters=std::make_unique<float[]>(bank_bytes/4),input=std::make_unique<float[]>(samples),output=std::make_unique<float[]>(values);
            read(bank,filters.get(),bank_bytes);read(pcm,input.get(),pcm_bytes);
            std::atomic_bool cancel{interrupted!=0};kadan::stt::LogMel frontend(ledger);
            frontend.load(bins,{filters.get(),bank_bytes/4},cancel);
            const auto observe=[&](std::size_t){cancel=interrupted!=0;};
            if(window)frontend.execute_window({input.get(),samples},{output.get(),values},cancel,observe);
            else frontend.execute({input.get(),samples},{output.get(),values},cancel,observe);
            frontend.unload();
            if(interrupted)throw std::runtime_error("stt_cancelled");
            std::cout<<std::setprecision(9)<<"{\"bins\":"<<bins<<",\"frames\":"<<frames<<",\"mel\":[";
            for(std::size_t i=0;i<values;++i){if(i)std::cout<<',';std::cout<<output[i];}
            std::cout<<"],\"resident_bytes_at_publication\":"<<ledger->snapshot().used[0];
            std::cout.flush();
            output.reset();input.reset();filters.reset();
        }
        std::cout<<",\"full_transcription\":false,\"gpu_execution\":false,\"resident_bytes\":"<<ledger->snapshot().used[0]<<"}\n";
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
