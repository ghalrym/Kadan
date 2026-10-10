#include "kadan/image_text.hpp"
#include <iostream>
#include <vector>
using namespace kadan;
void check(bool b){if(!b)throw std::runtime_error("text_test_failed");}
template<class F>void rejects(F f,const char* expected){try{f();}catch(const std::exception& e){if(e.what()!=std::string(expected))throw;return;}throw std::runtime_error("expected_failure");}
int main(int argc,char** argv){check(argc==2);auto r=std::make_shared<Resources>(Footprint{128*1024*1024});image::TextEncoder m(r);image::TextConfig d{32,16,2,2,1,8,48};std::array<std::string,2> files{"a.safetensors","b.safetensors"};std::atomic_bool cancel{false};std::vector<std::uint32_t> ids{1,3};std::vector<float> output(32,99);
 rejects([&]{m.execute(ids,output,cancel);},"image_text_not_loaded");cancel=true;rejects([&]{m.load(argv[1],files,d,cancel);},"image_text_cancelled");cancel=false;
 m.load(argv[1],files,d,cancel);auto resident=r->snapshot().used;std::size_t stages=0;m.execute(ids,output,cancel,[&](std::size_t){++stages;rejects([&]{m.unload();},"busy");});check(stages==2);for(std::size_t i=0;i<32;++i)check(output[i]==(i<16?.25f:.5f));check(r->snapshot().used==resident);
 std::fill(output.begin(),output.end(),99);rejects([&]{m.execute(ids,output,cancel,[&](std::size_t){cancel=true;});},"image_text_cancelled");cancel=false;for(float v:output)check(v==99);check(r->snapshot().used==resident);
 auto pressure=r->reserve(Workload::image,{r->snapshot().capacity[0]-resident[0]});rejects([&]{m.execute(ids,output,cancel);},"exhausted");r->released(pressure);check(r->snapshot().used==resident);ids[0]=32;rejects([&]{m.execute(ids,output,cancel);},"image_text_token_range");m.unload();m.unload();check(r->snapshot().used[0]==0);std::cout<<"text conditioner lifecycle passed\n";
}
