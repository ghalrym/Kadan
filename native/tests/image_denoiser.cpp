#include "dense_reference.hpp"
#include "kadan/image_denoiser.hpp"
#include <iostream>
#include <vector>
using namespace kadan;
void check(bool b){if(!b)throw std::runtime_error("denoiser_test_failed");}
template<class F>void rejects(F f,const char* expected){try{f();}catch(const std::exception& e){if(e.what()!=std::string(expected))throw;return;}throw std::runtime_error("expected_failure");}
int main(int argc,char** argv){check(argc==2);auto r=std::make_shared<Resources>(Footprint{128*1024*1024});auto compute=test_dense();image::Denoiser m(r,compute);image::DenoiserConfig d{{16,2,8,48,{2,2,4}},2,16,4};std::array<std::string,2> files{"a.safetensors","b.safetensors"};std::atomic_bool cancel{false};std::vector<float> input(16,1),condition(32,.1f),output(16,99);
 rejects([&]{m.execute(input,condition,2,2,2,.5f,output,cancel);},"image_not_loaded");cancel=true;rejects([&]{m.load(argv[1],files,d,cancel);},"image_cancelled");cancel=false;
 std::array<std::string,2> wrong{"wrong.safetensors","b.safetensors"};rejects([&]{m.load(argv[1],wrong,d,cancel);},"image_tensor_layout");check(r->snapshot().used[0]==0);
 m.load(argv[1],files,d,cancel);auto resident=r->snapshot().used;std::size_t stages=0;m.execute(input,condition,2,2,2,.5f,output,cancel,[&](const char*,std::size_t){++stages;rejects([&]{m.unload();},"busy");});check(stages==3);for(float v:output)check(v==0);check(r->snapshot().used==resident);
 for(auto stage:{"conditioning","block"}){std::fill(output.begin(),output.end(),99);rejects([&]{m.execute(input,condition,2,2,2,.5f,output,cancel,[&](const char* phase,std::size_t){if(std::string(phase)==stage)cancel=true;});},"image_cancelled");cancel=false;for(float v:output)check(v==99);check(r->snapshot().used==resident);}
 auto pressure=r->reserve(Workload::image,{r->snapshot().capacity[0]-resident[0]});rejects([&]{m.execute(input,condition,2,2,2,.5f,output,cancel);},"exhausted");r->released(pressure);check(r->snapshot().used==resident);
 rejects([&]{m.execute(input,condition,2,2,2,2.f,output,cancel);},"image_denoiser_layout");m.unload();m.unload();check(r->snapshot().used[0]==0);check_dense(compute);std::cout<<"denoiser lifecycle passed\n";
}
