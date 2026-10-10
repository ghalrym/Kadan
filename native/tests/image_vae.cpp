#include "kadan/image_vae.hpp"
#include <iostream>
#include <vector>
using namespace kadan;
void check(bool b){if(!b)throw std::runtime_error("vae_test_failed");}
template<class F>void rejects(F f,const char* expected){try{f();}catch(const std::exception& e){if(e.what()!=std::string(expected))throw;return;}throw std::runtime_error("expected_failure");}
int main(int argc,char** argv){check(argc==2);auto r=std::make_shared<Resources>(Footprint{64*1024*1024});image::VaeDecoder m(r);image::VaeConfig d{2,2,4,0};std::atomic_bool cancel{false};std::vector<float> input(2,.5f),output(1024,99);
 rejects([&]{m.decode(input,1,1,output,cancel);},"image_vae_not_loaded");cancel=true;rejects([&]{m.load(argv[1],"model.safetensors",d,cancel);},"image_vae_cancelled");cancel=false;m.load(argv[1],"model.safetensors",d,cancel);auto resident=r->snapshot().used;std::size_t stages=0;m.decode(input,1,1,output,cancel,[&](std::size_t){++stages;rejects([&]{m.unload();},"busy");});check(stages==5);for(auto v:output)check(v==.25f);check(r->snapshot().used==resident);
 std::fill(output.begin(),output.end(),99);rejects([&]{m.decode(input,1,1,output,cancel,[&](std::size_t){cancel=true;});},"image_vae_cancelled");cancel=false;for(auto v:output)check(v==99);check(r->snapshot().used==resident);auto pressure=r->reserve(Workload::image,{r->snapshot().capacity[0]-resident[0]});rejects([&]{m.decode(input,1,1,output,cancel);},"exhausted");r->released(pressure);check(r->snapshot().used==resident);m.unload();m.unload();check(r->snapshot().used[0]==0);std::cout<<"VAE lifecycle passed\n";
}
