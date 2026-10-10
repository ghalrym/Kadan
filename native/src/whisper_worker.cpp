#include "kadan/whisper.hpp"
#include "kadan/whisper_tokens.hpp"
#include "kadan/stt.hpp"
#include <bit>
#include <fstream>
#include <iostream>
#include <vector>
#include <fcntl.h>
#include <unistd.h>
namespace {
constexpr kadan::Bytes budget=16ULL*1024*1024*1024;
void check(bool ok,const char* error){if(!ok)throw std::runtime_error(error);}
struct Admission{kadan::Resources& r;kadan::Handle h;Admission(kadan::Resources& r_,kadan::Bytes n):r(r_),h(r.reserve(kadan::Workload::speech,{n})){}~Admission(){r.released(h);}};
void read(const std::string& path,std::span<float> out){std::ifstream file(path,std::ios::binary|std::ios::ate);check(bool(file)&&file.tellg()==std::streamoff(out.size_bytes()),"whisper_input_file");file.seekg(0);file.read(reinterpret_cast<char*>(out.data()),out.size_bytes());check(bool(file),"whisper_input_read");}
void publish(const std::string& path,const std::string& text){int fd=open(path.c_str(),O_CREAT|O_EXCL|O_WRONLY|O_CLOEXEC|O_NOFOLLOW,0600);check(fd>=0,"whisper_output_exists");std::size_t at=0;while(at<text.size()){auto count=write(fd,text.data()+at,text.size()-at);if(count<=0){close(fd);unlink(path.c_str());throw std::runtime_error("whisper_output_write");}at+=count;}if(close(fd)!=0){unlink(path.c_str());throw std::runtime_error("whisper_output_close");}}
}
int main(int argc,char** argv){
    static_assert(std::endian::native==std::endian::little);
    try{
        if(argc==2&&std::string(argv[1])=="--capabilities"){std::cout<<"whisper 1 cpu en 480000 "<<budget<<'\n';return 0;}
        check(argc==6,"usage: kadan-whisper-worker MODEL_ROOT SHARD DIMENSIONS ASSET_ROOT WORKSPACE");
        auto r=std::make_shared<kadan::Resources>(kadan::Footprint{budget});
        {
            Admission process_workspace(*r,256*1024*1024);
            kadan::stt::WhisperDimensions d{};std::ifstream dims(argv[3]);dims>>d.mels>>d.audio_context>>d.audio_state>>d.audio_heads>>d.audio_layers>>d.vocabulary>>d.text_context>>d.text_state>>d.text_heads>>d.text_layers;std::string extra;
            check(bool(dims)&&!(dims>>extra)&&d.audio_context==1500&&(d.mels==80||d.mels==128),"whisper_dimensions_file");
            std::atomic_bool cancel{false};kadan::stt::Whisper model(r);model.load(argv[1],argv[2],d,cancel);
            kadan::stt::WhisperTokens tokens(r);tokens.load((std::string(argv[4])+"/english.tokens").c_str(),d.vocabulary,cancel);
            check(tokens.prompt().size()<=d.text_context,"whisper_prompt_context");
            kadan::stt::LogMel frontend(r);
            {Admission filters(*r,d.mels*201*4);std::vector<float> values(d.mels*201);read(std::string(argv[4])+"/mel-"+std::to_string(d.mels)+".f32",values);frontend.load(d.mels,values,cancel);}
            const auto baseline=r->snapshot().used[0];std::cout<<"ready 1 "<<baseline<<'\n'<<std::flush;
            while(true){std::string command;char c;while(std::cin.get(c)&&c!='\n'){check(command.size()<32,"whisper_command_bound");command+=c;}
                if(!std::cin&&command.empty())break;
                if(command=="quit")break;
                check(command=="transcribe","whisper_command");
                std::size_t text_bytes=0,produced=0;
                {
                    const std::string input=std::string(argv[5])+"/pcm.f32",output=std::string(argv[5])+"/text.txt";
                    std::ifstream file(input,std::ios::binary|std::ios::ate);check(bool(file),"whisper_pcm_file");auto bytes=file.tellg();check(bytes>=0&&bytes%4==0&&bytes<=480000*4,"whisper_pcm_bound");
                    const auto samples=std::size_t(bytes)/4,max_tokens=std::min<std::size_t>(224,d.text_context-tokens.prompt().size()+1);
                    Admission admitted(*r,4*(samples+d.mels*3000+d.audio_context*d.audio_state+max_tokens)+kadan::stt::WhisperTokens::max_text_bytes);
                    std::vector<float> pcm(samples),mel(d.mels*3000),encoded(d.audio_context*d.audio_state);std::vector<std::uint32_t> generated(max_tokens);
                    read(input,pcm);frontend.execute_window(pcm,mel,cancel);model.encode(mel,encoded,cancel,[](const char* stage,std::size_t i){if(std::string_view(stage)=="encoder")std::cerr<<"encoder_layer "<<i<<'\n';});
                    produced=model.greedy(tokens.prompt(),encoded,generated,tokens.eos(),tokens.suppressed(),cancel,[](const char* stage,std::size_t i){if(std::string_view(stage)=="token")std::cerr<<"generated_token "<<i<<'\n';},tokens.first_suppressed());
                    check(produced&&generated[produced-1]==tokens.eos(),"whisper_token_limit");
                    auto text=tokens.decode(std::span(generated).first(produced));text_bytes=text.size();publish(output,text);
                }
                check(r->snapshot().used[0]==baseline,"whisper_scratch_leak");std::cout<<"done "<<text_bytes<<' '<<produced<<' '<<baseline<<'\n'<<std::flush;
            }
        }
        check(r->snapshot().used[0]==0,"whisper_cleanup_leak");std::cout<<"closed 0\n"<<std::flush;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';std::cout<<"error\n"<<std::flush;return 1;}
}
