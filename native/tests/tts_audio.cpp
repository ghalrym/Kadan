#include "dense_reference.hpp"
#include "kadan/tts_audio.hpp"
#include <iostream>
#include <vector>
using namespace kadan;
void check(bool b){if(!b)throw std::runtime_error("audio_test_failed");}
template<class F>void rejects(F f,const char* error){try{f();}catch(const std::exception& e){check(e.what()==std::string(error));return;}throw std::runtime_error("expected_failure");}
int main(int argc,char** argv){
    check(argc==2);auto r=std::make_shared<Resources>(Footprint{64*1024*1024});auto compute=test_dense();tts::AudioDecoder decoder(r,compute);tts::AudioConfig d{4,16,8,8,8,2,4,2,16,32,3};std::atomic_bool cancel{false};std::vector<std::uint32_t> codes(8);std::vector<float> waveform(3840,99);
    rejects([&]{decoder.decode(codes,waveform,cancel);},"tts_audio_not_loaded");cancel=true;rejects([&]{decoder.load(argv[1],"weights.safetensors",d,cancel);},"tts_audio_cancelled");cancel=false;
    rejects([&]{decoder.load(argv[1],"wrong.safetensors",d,cancel);},"tts_audio_tensor_layout");check(r->snapshot().residents==0);
    rejects([&]{decoder.load(argv[1],"nan.safetensors",d,cancel);},"tts_audio_nonfinite_weight");check(r->snapshot().residents==0);
    decoder.load(argv[1],"weights.safetensors",d,cancel);auto resident=r->snapshot().used;std::size_t stages=0;
    decoder.decode(codes,waveform,cancel,[&](const char*,std::size_t){++stages;rejects([&]{decoder.unload();},"busy");});check(stages==9);for(float v:waveform)check(v==0);check(r->snapshot().used==resident);
    if(compute){compute->fail=true;rejects([&]{decoder.decode(codes,waveform,cancel);},"test_compute_failure");compute->fail=false;check(r->snapshot().used==resident);}
    auto pressure=r->reserve(Workload::tts,{r->snapshot().capacity[0]-resident[0]});rejects([&]{decoder.decode(codes,waveform,cancel);},"exhausted");r->released(pressure);
    codes[0]=16;rejects([&]{decoder.decode(codes,waveform,cancel);},"tts_audio_code_range");codes[0]=0;
    rejects([&]{decoder.decode({},waveform,cancel);},"tts_audio_code_shape");rejects([&]{decoder.decode(codes,{},cancel);},"tts_audio_output_shape");
    rejects([&]{decoder.decode(codes,waveform,cancel,[&](const char*,std::size_t){cancel=true;});},"tts_audio_cancelled");cancel=false;check(r->snapshot().used==resident);
    rejects([&]{decoder.decode(codes,waveform,cancel,[](const char*,std::size_t){throw std::runtime_error("hook_failure");});},"hook_failure");check(r->snapshot().used==resident);
    decoder.unload();decoder.unload();check(r->snapshot().residents==0);check_dense(compute);std::cout<<"audio decoder lifecycle passed\n";
}
