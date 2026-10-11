#include "kadan/whisper.hpp"
#include <bit>
#include <charconv>
#include <fstream>
#include <iostream>
#include <vector>
#include <fcntl.h>
#include <unistd.h>
namespace {
std::size_t integer(const char* text){std::size_t n=0;std::string_view s(text);const auto result=std::from_chars(s.data(),s.data()+s.size(),n);if(result.ec!=std::errc{}||result.ptr!=s.data()+s.size())throw std::runtime_error("whisper_argument");return n;}
template<class T> void read(const char* path,std::span<T> output){std::ifstream in(path,std::ios::binary|std::ios::ate);if(!in||in.tellg()!=std::streamoff(output.size_bytes()))throw std::runtime_error("whisper_input_file");in.seekg(0);in.read(reinterpret_cast<char*>(output.data()),output.size_bytes());if(!in)throw std::runtime_error("whisper_input_read");}
template<class T>void write(const std::string& path,std::span<T> values){int fd=open(path.c_str(),O_CREAT|O_EXCL|O_WRONLY|O_CLOEXEC|O_NOFOLLOW,0600);if(fd<0)throw std::runtime_error("whisper_output_create");const char* data=reinterpret_cast<const char*>(values.data());std::size_t at=0;while(at<values.size_bytes()){auto n=::write(fd,data+at,values.size_bytes()-at);if(n<=0){close(fd);unlink(path.c_str());throw std::runtime_error("whisper_output_write");}at+=n;}if(close(fd)!=0){unlink(path.c_str());throw std::runtime_error("whisper_output_close");}}
}
int main(int argc,char** argv){
    static_assert(std::endian::native==std::endian::little);
    try{
        if(argc!=9)throw std::runtime_error("usage: kadan-whisper-network ROOT SHARD DIMS MEL_F32 TOKENS_U32 OUTPUT_PREFIX MAX_NEW EOS");
        kadan::stt::WhisperDimensions d{};std::ifstream dimensions(argv[3]);dimensions>>d.mels>>d.audio_context>>d.audio_state>>d.audio_heads>>d.audio_layers>>d.vocabulary>>d.text_context>>d.text_state>>d.text_heads>>d.text_layers;
        std::string extra;if(!dimensions||(dimensions>>extra))throw std::runtime_error("whisper_dimensions_file");
        const auto maximum=integer(argv[7]),eos=integer(argv[8]);if(maximum==0||maximum>448||eos>=d.vocabulary)throw std::runtime_error("whisper_generation_shape");
        std::ifstream tokens_file(argv[5],std::ios::binary|std::ios::ate);if(!tokens_file)throw std::runtime_error("whisper_token_file");const auto bytes=tokens_file.tellg();if(bytes<=0||bytes%4!=0||bytes>448*4)throw std::runtime_error("whisper_token_file");const auto count=std::size_t(bytes)/4;
        auto resources=std::make_shared<kadan::Resources>(kadan::Footprint{16ULL*1024*1024*1024});
        {
            std::atomic_bool cancel{false};kadan::stt::Whisper model(resources);model.load(argv[1],argv[2],d,cancel);
            struct Reservation {kadan::Resources& r;kadan::Handle h;~Reservation(){r.released(h);}};
            const auto mel_count=d.mels*2*d.audio_context,audio_count=d.audio_context*d.audio_state;
            Reservation admitted{*resources,resources->reserve(kadan::Workload::speech,{4*(mel_count+audio_count+d.vocabulary+count+maximum)})};
            std::vector<float> mel(mel_count),encoded(audio_count),logits(d.vocabulary);std::vector<std::uint32_t> tokens(count),generated(maximum);
            read(argv[4],std::span(mel));read(argv[5],std::span(tokens));
            model.encode(mel,encoded,cancel);model.decode(tokens,encoded,logits,cancel);
            const auto produced=model.greedy(tokens,encoded,generated,std::uint32_t(eos),{},cancel);
            const std::string prefix(argv[6]);write(prefix+".encoded.f32",std::span(encoded));write(prefix+".logits.f32",std::span(logits));write(prefix+".tokens.u32",std::span(generated).first(produced));
            model.unload();std::cout<<"{\"generated_tokens\":"<<produced<<",\"gpu_execution\":false,\"transcript\":false";
        }
        std::cout<<",\"resident_bytes\":"<<resources->snapshot().used[0]<<"}\n";
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
