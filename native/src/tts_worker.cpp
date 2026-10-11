#include "kadan/worker_compute.hpp"
#include "kadan/tts_generator.hpp"
#include "kadan/tts_controls.hpp"
#include "kadan/tts_audio.hpp"
#include "kadan/tts_tokenizer.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <csignal>
#include <fstream>
#include <iostream>
#include <vector>
#include <fcntl.h>
#include <unistd.h>
namespace {
constexpr kadan::Bytes budget=2ULL*1024*1024*1024;
std::atomic_bool cancelled{false};
void signal_cancel(int){cancelled.store(true);}
void check(bool value,const char* message){if(!value)throw std::runtime_error(message);}
struct Admission{kadan::Resources& r;kadan::Handle h;Admission(kadan::Resources& resources,kadan::Bytes n):r(resources),h(r.reserve(kadan::Workload::tts,kadan::host_footprint(r,n))){}~Admission(){r.released(h);}};
struct Request{std::string script,speaker,language,instruction;};
Request request_file(const std::string& path){std::ifstream f(path,std::ios::binary|std::ios::ate);check(bool(f)&&f.tellg()>0&&f.tellg()<=256*1024,"tts_request_size");std::string data(std::size_t(f.tellg()),'\0');f.seekg(0);f.read(data.data(),data.size());check(bool(f),"tts_request_read");auto j=nlohmann::json::parse(data);check(j.is_object()&&j.size()==4,"tts_request_fields");Request q{j.at("script"),j.at("speaker"),j.at("language"),j.at("instruction")};check(!q.script.empty()&&q.script.size()<=32000&&q.speaker.size()<=32&&q.language.size()<=32&&q.instruction.size()<=8000,"tts_request_bounds");return q;}

void publish(const std::string& path,std::span<const float> samples){
    std::vector<std::uint8_t> bytes(44+samples.size()*2);auto word=[&](std::size_t at,std::uint32_t n,std::size_t width){for(std::size_t i=0;i<width;++i)bytes[at+i]=std::uint8_t(n>>(8*i));};
    auto tag=[&](std::size_t at,const char* text){std::copy_n(text,4,bytes.begin()+at);};
    tag(0,"RIFF");word(4,36+samples.size()*2,4);tag(8,"WAVE");tag(12,"fmt ");word(16,16,4);word(20,1,2);word(22,1,2);word(24,24000,4);word(28,48000,4);word(32,2,2);word(34,16,2);tag(36,"data");word(40,samples.size()*2,4);
    for(std::size_t i=0;i<samples.size();++i){check(std::isfinite(samples[i]),"tts_nonfinite_audio");auto value=std::clamp(std::lround(samples[i]*32768),-32768L,32767L);word(44+2*i,std::uint16_t(value),2);}
    check(!cancelled.load(),"tts_cancelled");int fd=open(path.c_str(),O_CREAT|O_EXCL|O_WRONLY|O_CLOEXEC|O_NOFOLLOW,0600);check(fd>=0,"tts_output_create");std::size_t at=0;
    while(at<bytes.size()){auto n=write(fd,bytes.data()+at,bytes.size()-at);if(n<=0||cancelled.load()){close(fd);unlink(path.c_str());throw std::runtime_error("tts_output_write");}at+=n;}
    if(close(fd)!=0){unlink(path.c_str());throw std::runtime_error("tts_output_close");}
}
}
int main(int argc,char** argv){
    static_assert(std::atomic_bool::is_always_lock_free);
    try{
        if(argc==2&&std::string(argv[1])=="--capabilities"){std::cout<<"tts 2 "<<(kadan::worker_compute_plan(budget,"KADAN_TTS_DEVICES").devices.empty()?"cpu":"cuda")<<" multilingual custom 300 "<<budget<<'\n';return 0;}
        check(argc==4,"usage: kadan-tts-worker MODEL_ROOT TOKENIZER_JSON WORKSPACE");
        std::signal(SIGTERM,signal_cancel);std::signal(SIGINT,signal_cancel);
        kadan::WorkerCompute execution(kadan::worker_compute_plan(budget,"KADAN_TTS_DEVICES"),kadan::Workload::tts);auto r=execution.resources;
        {
            Admission process_workspace(*r,256*1024*1024);
            kadan::tts::VoiceControls controls(std::string(argv[1])+"/config.json");
            kadan::tts::TextTokenizer tokenizer(r);tokenizer.load(argv[2],cancelled);std::cerr<<"tokenizer_loaded\n";
            kadan::tts::CodeGenerator generator(r,execution.compute);generator.load(argv[1],"model.safetensors",{},cancelled);std::cerr<<"talker_loaded\n";
            kadan::tts::AudioDecoder decoder(r,execution.compute);const auto audio_root=std::string(argv[1])+"/speech_tokenizer";decoder.load(audio_root.c_str(),"model.safetensors",{},cancelled);std::cerr<<"decoder_loaded\n";
            const auto baseline=r->snapshot().used[0];std::cout<<"ready 1 "<<baseline<<'\n'<<std::flush;
            while(!cancelled.load()){
                std::string command;char c;while(std::cin.get(c)&&c!='\n'){check(command.size()<32,"tts_command_bound");command+=c;}
                if(!std::cin&&command.empty())break;
                if(command=="quit")break;
                if(command=="park"){execution.park();std::cout<<"parked\n"<<std::flush;continue;}if(command=="resume"){execution.resume();std::cout<<"resumed\n"<<std::flush;continue;}check(command=="speak","tts_command");execution.resume();std::size_t frames=0;
                {
                    Admission admitted(*r,2*1024*1024+2*2048*4+300*16*4+300*1920*6+44);
                    auto request=request_file(std::string(argv[3])+"/request.json");auto voice=controls.resolve(request.speaker,request.language);auto& text=request.script;std::array<std::uint32_t,2048> tokens;auto count=tokenizer.encode(text,tokens,cancelled);std::vector<std::uint32_t> codes(300*16);
                    std::array<std::uint32_t,2048> instruction{};std::size_t instruction_count=0;if(!request.instruction.empty())instruction_count=tokenizer.encode("<|im_start|>user\n"+request.instruction+"<|im_end|>\n",instruction,cancelled);
                    auto generated=generator.generate(std::span(tokens).first(count),voice,codes,cancelled,[](const char* phase,std::size_t i){std::cerr<<phase<<' '<<i<<'\n';},2,1.05f,std::span(instruction).first(instruction_count));
                    check(generated.stopped&&generated.frames>0,"tts_incomplete_generation");frames=generated.frames;std::vector<float> audio(frames*1920);
                    decoder.decode(std::span(codes).first(frames*16),audio,cancelled,[](const char* phase,std::size_t i){std::cerr<<"audio_"<<phase<<' '<<i<<'\n';});
                    publish(std::string(argv[3])+"/audio.wav",audio);
                }
                execution.idle();const auto retained=execution.compute?execution.compute->retained_weights():kadan::Footprint{};check(r->snapshot().used[0]==baseline+(retained.empty()?0:retained[0]),"tts_scratch_leak");std::cout<<"done "<<44+frames*1920*2<<' '<<frames<<' '<<baseline<<'\n'<<std::flush;
            }
        }
        execution.park();check(r->snapshot().used[0]==0,"tts_cleanup_leak");execution.idle();std::cout<<"closed 0\n"<<std::flush;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';std::cout<<"error\n"<<std::flush;return 1;}
}
