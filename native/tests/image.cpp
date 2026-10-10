#include "dense_reference.hpp"
#include "kadan/image.hpp"
#include <iostream>
#include <vector>
using namespace kadan;
void check(bool b){if(!b)throw std::runtime_error("image_test_failed");}
template<class F>void rejects(F f,const char* expected){try{f();}catch(const std::exception& e){if(e.what()!=std::string(expected))throw;return;}throw std::runtime_error("expected_failure");}
int main(int argc,char** argv){check(argc==2);auto r=std::make_shared<Resources>(Footprint{64*1024*1024});auto compute=test_dense();image::TransformerBlock b(r,compute);image::BlockConfig d{16,2,8,48,{2,2,4}};std::atomic_bool cancel{false};std::vector<float> input(96,1),mod(128,.1f),output(96,99);
 rejects([&]{b.execute(input,mod,2,2,2,output,cancel);},"image_not_loaded");
 for(auto pair:{std::pair{"wrong.safetensors","image_tensor_layout"},std::pair{"nan.safetensors","image_nonfinite_weight"}}){rejects([&]{b.load(argv[1],pair.first,0,d,cancel);},pair.second);check(r->snapshot().used[0]==0);}
 cancel=true;rejects([&]{b.load(argv[1],"weights.safetensors",0,d,cancel);},"image_cancelled");cancel=false;b.load(argv[1],"weights.safetensors",0,d,cancel);auto resident=r->snapshot().used;
 b.execute(input,mod,2,2,2,output,cancel,[&](const char*){rejects([&]{b.unload();},"busy");});check(output==input&&r->snapshot().used==resident);
 for(auto stage:{"qkv","attention","feed_forward"}){std::fill(output.begin(),output.end(),99);rejects([&]{b.execute(input,mod,2,2,2,output,cancel,[&](const char* phase){if(std::string(phase)==stage)cancel=true;});},"image_cancelled");cancel=false;for(auto v:output)check(v==99);check(r->snapshot().used==resident);}
 rejects([&]{b.execute(input,mod,2,2,2,output,cancel,[](const char*){throw std::runtime_error("hook");});},"hook");check(r->snapshot().used==resident);
 auto pressure=r->reserve(Workload::image,{r->snapshot().capacity[0]-resident[0]});rejects([&]{b.execute(input,mod,2,2,2,output,cancel);},"exhausted");r->released(pressure);check(r->snapshot().used==resident);
 rejects([&]{b.execute(input,mod,0,2,2,output,cancel);},"image_token_layout");rejects([&]{b.execute(input,{},2,2,2,output,cancel);},"image_shape");b.unload();b.unload();check(r->snapshot().used[0]==0);check_dense(compute);std::cout<<"image block lifecycle passed\n";
}
