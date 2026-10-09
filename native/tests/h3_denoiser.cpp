#include "kadan/h3_denoiser.hpp"
#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>
using Model=kadan::video::H3Denoiser;
void check(bool v){if(!v)throw std::runtime_error("assertion_failed");}
template<class F>void rejects(F f,const std::string& message){try{f();}catch(const std::exception& e){check(std::string(e.what())==message);return;}throw std::runtime_error("expected_rejection");}
int main(int argc,char** argv){try{
    check(argc==2);std::atomic_bool cancel=false;auto resources=std::make_shared<kadan::Resources>(kadan::Footprint{512ULL*1024*1024});Model model(resources);
    model.load(argv[1],"base.safetensors",cancel);model.load_turbo(argv[1],"turbo.safetensors",cancel);check(model.turbo_loaded());
    const auto retained=resources->snapshot().used[0];check(retained==Model::metadata_bytes*2);
    std::vector<float> text(5120),video(96),audio(32),positions(9),times{1,.25f,.3f},vo(96),ao(32);std::vector<std::uint32_t> tags{1,0,2};
    for(std::size_t i=0;i<4;++i)text[i]=float(i+1)*.125f;
    for(std::size_t i=0;i<video.size();++i)video[i]=float(int(i%9)-4)*.125f;
    for(std::size_t i=0;i<audio.size();++i)audio[i]=float(int(i%7)-3)*.0625f;
    Model::Input input{text,video,audio,positions,times,tags};
    auto execute=[&](const Model::Hook& hook={}){model.execute(input,vo,ao,cancel,hook);};
    cancel=true;rejects([&]{execute();},"h3_denoiser_cancelled");cancel=false;
    rejects([&]{execute([&](const char* phase,std::size_t){if(std::string_view(phase)=="refiner_completed")cancel=true;});},"h3_denoiser_cancelled");cancel=false;check(resources->snapshot().used[0]==retained);
    std::size_t blocks=0,refiners=0;kadan::Bytes peak=0;
    execute([&](const char* phase,std::size_t count){peak=std::max(peak,resources->snapshot().used[0]);if(std::string_view(phase)=="block_completed")blocks=count;if(std::string_view(phase)=="refiner_completed")refiners=count;rejects([&]{model.unload();},"busy");});
    check(blocks==50 && refiners==2 && resources->snapshot().used[0]==retained && peak>260000000);
    std::ofstream out(std::string(argv[1])+"/result.f32",std::ios::binary);out.write(reinterpret_cast<const char*>(vo.data()),vo.size()*4);out.write(reinterpret_cast<const char*>(ao.data()),ao.size()*4);check(bool(out));out.close();
    rejects([&]{execute([](const char* phase,std::size_t){if(std::string_view(phase)=="block_completed")throw std::runtime_error("hook_failure");});},"hook_failure");check(resources->snapshot().used[0]==retained);
    tags[0]=3;rejects([&]{execute();},"h3_denoiser_timestep");tags[0]=1;
    times[1]=std::nanf("");rejects([&]{execute();},"h3_denoiser_nonfinite");times[1]=.25f;
    model.unload();check(resources->snapshot().used[0]==0);
    auto small=std::make_shared<kadan::Resources>(kadan::Footprint{20ULL*1024*1024});Model denied(small);denied.load(argv[1],"base.safetensors",cancel);denied.load_turbo(argv[1],"turbo.safetensors",cancel);
    rejects([&]{denied.execute(input,vo,ao,cancel);},"exhausted");check(small->snapshot().used[0]==retained);denied.unload();check(small->snapshot().used[0]==0);
    std::cout<<"All 50 blocks, 2 refiners, Turbo, cancellation, exceptions, reentrancy and admission cleanup passed; peak="<<peak<<'\n';return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
