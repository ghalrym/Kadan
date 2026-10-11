// Checkpoint parsing/lifecycle only. Never calls execute or creates a device.
#include "kadan/image.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <unistd.h>
void check(bool b){if(!b)throw std::runtime_error("test_failed");}
int main(){
 char directory[]="/tmp/kadan-weight-load-XXXXXX";check(mkdtemp(directory));
 struct Cleanup{const char* path;~Cleanup(){std::filesystem::remove_all(path);}}cleanup{directory};
 std::ostringstream header;header<<'{';std::size_t bytes=0,count=0;
 auto add=[&](std::string name,bool vector){
  if(count++)header<<',';
  const auto size=vector?4:8;
  header<<"\"transformer_blocks.7."<<name<<".weight\":{\"dtype\":\"BF16\",\"shape\":"<<(vector?"[2]":"[2,2]")<<",\"data_offsets\":["<<bytes<<','<<bytes+size<<"]}";
  bytes+=size;
 };
 for(auto name:{"attn.to_q","attn.to_k","attn.to_v","attn.to_out.0","img_mlp.proj","img_mlp.gate_layer","img_mlp.out"})add(name,false);
 for(auto name:{"attn.norm_q","attn.norm_k"})add(name,true);
 header<<'}';const auto text=header.str();
 {std::ofstream file(std::string(directory)+"/weights.safetensors",std::ios::binary);for(unsigned i=0;i<8;++i)file.put(char(std::uint64_t(text.size())>>(8*i)));file<<text;for(std::size_t i=0;i<bytes;++i)file.put(0);}
 auto resources=std::make_shared<kadan::Resources>(kadan::Footprint{64*1024*1024});
 kadan::image::TransformerBlock block(resources);std::atomic_bool cancel=false;
 block.load(directory,"weights.safetensors",7,{2,1,2,2,{0,0,2}},cancel);
 check(resources->snapshot().used[0]>0);block.unload();check(resources->snapshot().used[0]==0);
 std::cout<<"qualified block checkpoint names and release passed (no execution)\n";
}
