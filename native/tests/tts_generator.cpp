#include "dense_reference.hpp"
#include "kadan/tts_generator.hpp"
#include <iostream>
#include <vector>
using namespace kadan;
void check(bool b){if(!b)throw std::runtime_error("generator_test_failed");}
template<class F>void rejects(F f,const char* expected){try{f();}catch(const std::exception& e){if(e.what()!=std::string(expected))throw;return;}throw std::runtime_error("expected_failure");}
int main(int argc,char** argv){
    check(argc==2);auto r=std::make_shared<Resources>(Footprint{64*1024*1024});auto compute=test_dense();tts::CodeGenerator model(r,compute);
    tts::GeneratorConfig d{{8,2,1,4,2,16},{4,2,1,4,2,8},32,24,4,16};
    tts::VoicePrompt voice{{0,1,2},3,4,5,16,17,18,19,20,21,22,23};
    std::atomic_bool cancel{false};std::vector<std::uint32_t> text{6,7},codes(16,99);
    rejects([&]{model.generate(text,voice,codes,cancel);},"tts_generation_not_loaded");
    cancel=true;rejects([&]{model.load(argv[1],"valid.safetensors",d,cancel);},"tts_generation_cancelled");cancel=false;
    for(auto pair:{std::pair{"wrong.safetensors","tts_generation_tensor_layout"},std::pair{"nan.safetensors","tts_generation_nonfinite_weight"}}){rejects([&]{model.load(argv[1],pair.first,d,cancel);},pair.second);check(r->snapshot().used[0]==0);}
    model.load(argv[1],"valid.safetensors",d,cancel);auto resident=r->snapshot().used;
    std::size_t hooks=0;auto result=model.generate(text,voice,codes,cancel,[&](const char*,std::size_t){++hooks;rejects([&]{model.unload();},"busy");rejects([&]{model.generate(text,voice,codes,cancel);},"busy");});
    check(result.frames==4&&!result.stopped&&hooks==5);for(auto v:codes)check(v==0);check(r->snapshot().used==resident);
    auto again=model.generate(text,voice,codes,cancel);check(again.frames==4&&!again.stopped);check(r->snapshot().used==resident);
    for(bool during_frame:{false,true}){rejects([&]{model.generate(text,voice,codes,cancel,[&](const char* phase,std::size_t){if((std::string(phase)=="frame")==during_frame)cancel=true;});},"tts_generation_cancelled");cancel=false;check(r->snapshot().used==resident);}
    rejects([&]{model.generate(text,voice,codes,cancel,[](const char*,std::size_t){throw std::runtime_error("hook_failure");});},"hook_failure");check(r->snapshot().used==resident);
    if(compute){compute->fail=true;rejects([&]{model.generate(text,voice,codes,cancel);},"test_compute_failure");compute->fail=false;check(r->snapshot().used==resident);}
    auto pressure=r->reserve(Workload::tts,{r->snapshot().capacity[0]-resident[0]});rejects([&]{model.generate(text,voice,codes,cancel);},"exhausted");r->released(pressure);check(r->snapshot().used==resident);
    text[0]=32;rejects([&]{model.generate(text,voice,codes,cancel);},"tts_generation_token_range");text[0]=6;
    rejects([&]{model.generate({},voice,codes,cancel);},"tts_generation_shape");
    model.unload();check(r->snapshot().used[0]==0);model.load(argv[1],"eos.safetensors",d,cancel);
    auto stopped=model.generate(text,voice,codes,cancel);check(stopped.frames==2&&stopped.stopped);
    auto empty=model.generate(text,voice,codes,cancel,{},0);check(empty.frames==0&&empty.stopped);
    model.unload();model.unload();check(r->snapshot().used[0]==0);check_dense(compute);std::cout<<"generator lifecycle passed\n";
}
