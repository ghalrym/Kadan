#include "kadan/model_worker.hpp"
#include <sstream>
#include <iostream>
#include <thread>
#include <chrono>
using namespace kadan::serving;
void check(bool value){if(!value)throw std::runtime_error("test_failed");}
struct Fake:ResidentEngine {
    std::thread::id owner=std::this_thread::get_id();
    std::size_t count=0,starts=0,ends=0,parks=0;bool closed=false,fail_end=false,block=false;
    void owned(){check(owner==std::this_thread::get_id());}
    Info info()const override{return {16,4,128,256};}
    void reset()override{owned();count=0;}
    void begin_request()override{owned();++starts;count=0;}
    void end_request()override{owned();if(fail_end)throw std::runtime_error("cleanup");++ends;}
    void park()override{owned();++parks;}
    Token step(unsigned token,bool)override{owned();if(block){std::cout<<"running\n"<<std::flush;for(;;)std::this_thread::sleep_for(std::chrono::seconds(1));}return {token,false,++count};}
    void close()override{owned();closed=true;}
};
const std::string nonce="0123456789abcdef0123456789abcdef";
std::string command(std::string op,unsigned id,std::string rest=""){return op+" "+nonce+" "+std::to_string(id)+rest+"\n";}
std::string run(Fake& fake,std::string input,bool failure=false){
    std::istringstream in(input);std::ostringstream out;bool failed=false;
    try{resident_session(fake,std::make_shared<kadan::Resources>(kadan::Footprint{1024,1024}),nonce,in,out);}catch(...){failed=true;}
    check(failed==failure);return out.str();
}
int main(int argc,char**){
    if(argc>1){Fake fake;fake.block=true;resident_session(fake,std::make_shared<kadan::Resources>(kadan::Footprint{1024,1024}),nonce,std::cin,std::cout);return 0;}
    {Fake f;auto out=run(f,command("submit",0)+command("submit",0)+command("submit",0)+command("cancel",2)+command("start",1)+command("step",1," 2 0")+command("end",1)+command("start",3)+command("end",3)+command("park",0)+command("close",0));
        check(f.starts==2&&f.ends==2&&f.parks&&f.closed);check(out.find("token "+nonce+" 1 2 0 1")!=std::string::npos);}
    {Fake f;run(f,command("submit",0)+command("submit",0)+command("start",2),true);check(f.starts==0);}
    {Fake f;run(f,"submit ffffffffffffffffffffffffffffffff 0\n",true);check(f.starts==0);}
    {Fake f;auto out=run(f,command("submit",0)+command("start",1)+command("end",1)+command("submit",0)+command("start",2)+command("step",1," 2 0"),true);check(out.find("token ")==std::string::npos);}
    {Fake f;run(f,command("submit",0)+command("start",1)+command("cancel",1),true);}
    {Fake f;f.fail_end=true;auto out=run(f,command("submit",0)+command("start",1)+command("end",1),true);check(out.find("ended ")==std::string::npos&&out.find("closed ")==std::string::npos);}
    {Fake f;run(f,command("submit",0)+command("start",1));check(f.closed&&f.ends==1);}
    {Fake f;std::string in;for(int i=0;i<33;++i)in+=command("submit",0);run(f,in,true);check(f.starts==0);}
    check(session_identity()!=session_identity());std::cout<<"resident session tests passed\n";
}
