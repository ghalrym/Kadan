#include "kadan/image_generation.hpp"
#include <filesystem>
#include <array>
#include <limits>
#include <vector>
using namespace kadan::image;
void need(bool b){if(!b)throw std::runtime_error("generation_test_failed");}
template<class F>void reject(F f){try{f();}catch(const std::exception&){return;}throw std::runtime_error("expected_failure");}
int main(int argc,char** argv){need(argc==2);auto root=std::string(argv[1]);std::atomic_bool cancel{false};ImageRequest q{"red ball",32,32,2,0};Generator::validate(q);
 for(auto shape:std::array<std::array<std::size_t,2>,4>{{{2048,2048},{2400,1792},{1792,2400},{2752,1536}}})Generator::validate({"ball",shape[0],shape[1],50,0});
 reject([&]{Generator::validate({"ball",3072,3072,50,0});});reject([&]{Generator::validate({"ball",3104,32,50,0});});
 q.steps=1;reject([&]{Generator::validate(q);});q.steps=2;q.width=33;reject([&]{Generator::validate(q);});q.width=32;q.prompt.clear();reject([&]{Generator::validate(q);});std::vector<float> rgba(16);for(std::size_t i=0;i<4;++i){rgba[i]=-1;rgba[4+i]=1;rgba[8+i]=0;rgba[12+i]=1;}
 publish_png(root+"/result.png",rgba,2,2,cancel);auto size=std::filesystem::file_size(root+"/result.png");reject([&]{publish_png(root+"/result.png",rgba,2,2,cancel);});need(std::filesystem::file_size(root+"/result.png")==size);cancel=true;reject([&]{publish_png(root+"/cancel.png",rgba,2,2,cancel);});need(!std::filesystem::exists(root+"/cancel.png"));cancel=false;
 // Full original wide profile exercises publication buffers and PNG dimensions without model/GPU work.
 {std::vector<float> large(2752*1536*4,0);publish_png(root+"/wide.png",large,1536,2752,cancel);}
 rgba[0]=std::numeric_limits<float>::quiet_NaN();reject([&]{publish_png(root+"/nan.png",rgba,2,2,cancel);});need(!std::filesystem::exists(root+"/nan.png"));
}
