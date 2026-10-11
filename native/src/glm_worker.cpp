#include "kadan/glm.hpp"
#include <iostream>
#include <csignal>
#include <map>
namespace {
void need(bool b,const char* e){if(!b)throw std::runtime_error(e);}
kadan::ComputePlan compute(std::string_view text,std::size_t host){kadan::ComputePlan p{{host},{}};while(!text.empty()){auto end=text.find(',');auto entry=text.substr(0,end);auto sep=entry.find(':');need(sep!=std::string_view::npos,"glm_device_budget");auto d=kadan::serving::number(entry.substr(0,sep)),bytes=kadan::serving::number(entry.substr(sep+1));need(d<2&&bytes>=768ULL*1024*1024,"glm_device_floor");need(std::find(p.devices.begin(),p.devices.end(),int(d))==p.devices.end(),"glm_duplicate_device");p.devices.push_back(int(d));p.capacity.resize(std::max(p.capacity.size(),d+2));p.capacity[d+1]=bytes;if(end==std::string_view::npos)break;text.remove_prefix(end+1);}need(!p.devices.empty(),"glm_cuda_required");return p;}
}
int main(int argc,char** argv){try{need(std::signal(SIGPIPE,SIG_IGN)!=SIG_ERR,"glm_sigpipe");
 if(argc==5&&std::string_view(argv[1])=="--plan-glm"){auto p=kadan::glm::plan(argv[2],kadan::serving::number(argv[3]));auto cp=compute(argv[4],p.host);std::cout<<"plan 5 "<<p.host<<' '<<p.execution_weights<<" 154880 "<<p.context<<' '<<p.packed<<' '<<p.tensors<<' '<<cp.devices.size();for(auto d:cp.devices)std::cout<<' '<<d<<' '<<cp.capacity[d+1];std::cout<<'\n';return 0;}
 if(argc==6&&std::string_view(argv[1])=="--serve-glm"){auto p=kadan::glm::plan(argv[2],kadan::serving::number(argv[3]));auto stamp=std::to_string(p.host)+":"+std::to_string(p.packed)+":"+std::to_string(p.tensors)+":"+std::to_string(p.execution_weights);need(stamp==argv[5],"glm_preflight_changed");auto cp=compute(argv[4],p.host);kadan::glm::Engine engine(argv[2],p,cp);kadan::serving::resident_session(engine,engine.resources(),kadan::serving::session_identity(),std::cin,std::cout);return 0;}
 throw std::runtime_error("usage: kadan-glm-worker --plan-glm ROOT CONTEXT BUDGETS | --serve-glm ROOT CONTEXT BUDGETS STAMP");
 }catch(const std::exception& e){std::cerr<<e.what()<<'\n';std::cout<<"error\n"<<std::flush;return 1;}}
