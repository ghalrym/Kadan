#include "kadan/whisper.hpp"
#include <array>
#include <iostream>
#include <limits>
#include <vector>
using namespace kadan;
void check(bool b){if(!b)throw std::runtime_error("test_failed");}
template<class F>void rejects(F f,const char* text){try{f();}catch(const std::exception& e){if(e.what()==std::string(text))return;throw;}throw std::runtime_error("expected_failure");}
int main(int argc,char** argv){
    check(argc==2);auto r=std::make_shared<Resources>(Footprint{64*1024*1024,0});std::atomic_bool cancel{false};stt::Whisper model(r);
    stt::WhisperDimensions d{4,5,8,2,2,17,7,8,2,2};std::vector<float> mel(40,.125f),audio(40),logits(17);std::array<std::uint32_t,2> prompt{1,2};
    rejects([&]{model.encode(mel,audio,cancel);},"whisper_not_loaded");
    cancel=true;rejects([&]{model.load(argv[1],"weights.safetensors",d,cancel);},"whisper_cancelled");cancel=false;
    auto invalid=d;invalid.audio_heads=3;rejects([&]{model.load(argv[1],"weights.safetensors",invalid,cancel);},"whisper_dimensions");
    rejects([&]{model.load(argv[1],"wrong.safetensors",d,cancel);},"whisper_tensor_layout");check(r->snapshot().residents==0);
    rejects([&]{model.load(argv[1],"nan.safetensors",d,cancel);},"whisper_nonfinite_weight");check(r->snapshot().residents==0);
    auto tiny=std::make_shared<Resources>(Footprint{24*1024*1024});stt::Whisper denied(tiny);
    rejects([&]{denied.load(argv[1],"weights.safetensors",d,cancel);},"exhausted");check(tiny->snapshot().residents==0);
    model.load(argv[1],"weights.safetensors",d,cancel);const auto resident=r->snapshot().used;
    rejects([&]{model.load(argv[1],"weights.safetensors",d,cancel);},"whisper_already_loaded");
    std::size_t blocks=0;
    model.encode(mel,audio,cancel,[&](const char* stage,std::size_t){if(std::string(stage)=="encoder")++blocks;rejects([&]{model.unload();},"busy");rejects([&]{model.encode(mel,audio,cancel);},"busy");});
    check(blocks==2);for(float v:audio)check(v==0);
    model.decode(prompt,audio,logits,cancel);for(float v:logits)check(v==0);
    std::array<std::uint32_t,3> generated{};
    check(model.greedy(prompt,audio,generated,0,{},cancel)==1&&generated[0]==0);
    const std::array<std::uint32_t,1> suppress{0};check(model.greedy(prompt,audio,generated,16,suppress,cancel)==3);for(auto id:generated)check(id==1);
    const std::array<std::uint32_t,2> first_only{0,1};
    check(model.greedy(prompt,audio,generated,16,{},cancel,{},first_only)==3);
    check(generated[0]==2&&generated[1]==0&&generated[2]==0);
    std::array<std::uint32_t,17> all;for(std::size_t i=0;i<all.size();++i)all[i]=i;
    rejects([&]{model.greedy(prompt,audio,generated,16,all,cancel);},"whisper_all_tokens_suppressed");
    rejects([&]{model.encode(mel,mel,cancel);},"whisper_buffer_overlap");
    rejects([&]{model.decode(prompt,audio,std::span(audio).first(17),cancel);},"whisper_buffer_overlap");
    rejects([&]{model.decode({},audio,logits,cancel);},"whisper_input_shape");
    prompt[1]=17;rejects([&]{model.decode(prompt,audio,logits,cancel);},"whisper_token_range");prompt[1]=2;
    mel[0]=std::numeric_limits<float>::infinity();rejects([&]{model.encode(mel,audio,cancel);},"whisper_nonfinite");mel[0]=0;
    auto pressure=r->reserve(Workload::speech,{r->snapshot().capacity[0]-resident[0],0});
    rejects([&]{model.encode(mel,audio,cancel);},"exhausted");rejects([&]{model.decode(prompt,audio,logits,cancel);},"exhausted");r->released(pressure);
    for(const auto operation:{0,1,2}){
        const auto hook=[&](const char*,std::size_t){cancel=true;};
        rejects([&]{if(operation==0)model.encode(mel,audio,cancel,hook);else if(operation==1)model.decode(prompt,audio,logits,cancel,hook);else model.greedy(prompt,audio,generated,16,{},cancel,hook);},"whisper_cancelled");cancel=false;check(r->snapshot().used==resident);
    }
    rejects([&]{model.encode(mel,audio,cancel,[](const char*,std::size_t){throw std::runtime_error("hook_failure");});},"hook_failure");check(r->snapshot().used==resident);
    model.unload();model.unload();check(r->snapshot().residents==0);
    {stt::Whisper scoped(r);scoped.load(argv[1],"half.safetensors",d,cancel);scoped.encode(mel,audio,cancel);scoped.decode(prompt,audio,logits,cancel);}
    check(r->snapshot().residents==0);std::cout<<"Whisper full network lifecycle passed\n";
}
