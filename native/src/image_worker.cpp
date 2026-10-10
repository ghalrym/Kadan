#include "kadan/image_generation.hpp"
#include <nlohmann/json.hpp>
#include <csignal>
#include <fstream>
#include <iostream>
#include <vector>
namespace {
constexpr kadan::Bytes budget=80ULL*1024*1024*1024;std::atomic_bool cancel{false};void signal_cancel(int){cancel.store(true);}void need(bool b,const char* e){if(!b)throw std::runtime_error(e);}
struct Admission{kadan::Resources& r;kadan::Handle h;Admission(kadan::Resources& a,kadan::Bytes n):r(a),h(r.reserve(kadan::Workload::image,{n})){}~Admission(){r.released(h);}};
kadan::image::ImageRequest read(const std::string& path){std::ifstream f(path,std::ios::binary|std::ios::ate);need(bool(f)&&f.tellg()>0&&f.tellg()<=64000,"image_request_file");std::string text(std::size_t(f.tellg()),'\0');f.seekg(0);f.read(text.data(),text.size());need(bool(f),"image_request_read");auto j=nlohmann::json::parse(text);need(j.is_object()&&j.size()==5&&j.at("width").is_number_unsigned()&&j.at("height").is_number_unsigned()&&j.at("steps").is_number_unsigned()&&j.at("seed").is_number_unsigned(),"image_request_fields");kadan::image::ImageRequest q{j.at("prompt"),j.at("width"),j.at("height"),j.at("steps"),j.at("seed")};kadan::image::Generator::validate(q);return q;}
}
int main(int argc,char** argv){static_assert(std::atomic_bool::is_always_lock_free);try{
 if(argc==2&&std::string(argv[1])=="--capabilities"){std::cout<<"image 1 cpu 32 256 100 "<<budget<<'\n';return 0;}need(argc==3,"usage: kadan-image-worker MODEL_ROOT WORKSPACE");std::signal(SIGTERM,signal_cancel);std::signal(SIGINT,signal_cancel);auto r=std::make_shared<kadan::Resources>(kadan::Footprint{budget});{
 Admission process(*r,256*1024*1024);kadan::image::Generator model(r);auto hook=[](const char* phase,std::size_t i){std::cerr<<phase<<' '<<i<<'\n';};model.load(argv[1],cancel,hook);auto baseline=r->snapshot().used[0];std::cout<<"ready 1 "<<baseline<<'\n'<<std::flush;
 while(!cancel.load()){std::string command;char c;while(std::cin.get(c)&&c!='\n'){need(command.size()<32,"image_command_bound");command+=c;}if((!std::cin&&command.empty())||command=="quit")break;need(command=="generate","image_command");{
 Admission request_memory(*r,8*1024*1024);auto request=read(std::string(argv[2])+"/request.json");std::vector<float> rgba(request.height*request.width*4);model.generate(request,rgba,cancel,hook);kadan::image::publish_png(std::string(argv[2])+"/image.png",rgba,request.height,request.width,cancel);
 }need(r->snapshot().used[0]==baseline,"image_scratch_leak");std::cout<<"done "<<baseline<<'\n'<<std::flush;}
 }need(r->snapshot().used[0]==0,"image_cleanup_leak");std::cout<<"closed 0\n"<<std::flush;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';std::cout<<"error\n"<<std::flush;return 1;}}
