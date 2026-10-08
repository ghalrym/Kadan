#include "kadan/model_worker.hpp"
#include <sstream>
#include <stdexcept>
#include <iostream>
struct Fake: kadan::serving::Engine {
    std::size_t count=0,steps=0,resets=0;bool closed=false,fail_close=false,fail_step=false,bad_progress=false;
    kadan::serving::Info info()const override{return {16,2,128,256};}
    void reset()override{count=0;++resets;}
    kadan::serving::Token step(unsigned id,bool)override{++steps;if(fail_step)throw std::runtime_error("injected");return {id,id==7,++count+unsigned(bad_progress)};}
    void close()override{if(fail_close)throw std::runtime_error("cleanup");closed=true;}
};
void check(bool value){if(!value)throw std::runtime_error("test_failed");}
std::string run(Fake& e,const std::string& s,bool failure=false){std::istringstream in(s);std::ostringstream out;bool failed=false;try{kadan::serving::session(e,in,out);}catch(const std::exception&){failed=true;}check(failed==failure);return out.str();}
int main(){try{
    {Fake e;auto out=run(e,"step 2 0\nstep 7 1\nreset\nstep 3 0\nclose\n");check(e.closed&&e.steps==3&&e.resets==1);check(out=="ready 1 16 2 128 256\ntoken 2 0 1\ntoken 7 1 2\nok reset\ntoken 3 0 1\nclosed 0\n");}
    {Fake e;check(run(e,"").ends_with("closed 0\n")&&e.closed);}
    for(const auto& input:{"step 16 0\n","step -1 0\n","step 1 2\n","step 1 0 extra\n","step 1\n","step 1 0","reset \n","wat\n","\n","step 1 0\r\n","step 18446744073709551616 0\n"}){Fake e;run(e,input,true);check(!e.closed&&e.steps==0);}
    {Fake e;run(e,std::string(65,'a')+"\n",true);check(e.steps==0);}
    {Fake e;run(e,"step 7 1\nstep 2 0\n",true);check(e.steps==1);}
    {Fake e;run(e,"step 7 0\nstep 2 0\nstep 3 0\n",true);check(e.steps==2);}
    {Fake e;e.fail_close=true;check(run(e,"close\n",true).find("closed")==std::string::npos);}
    {Fake e;e.fail_step=true;check(run(e,"step 2 0\n",true).find("token")==std::string::npos);}
    {Fake e;e.bad_progress=true;check(run(e,"step 2 0\n",true).find("token")==std::string::npos);}
    {Fake e;std::istringstream in("step 2 0\n");std::ostringstream out;out.setstate(std::ios::badbit);bool failed=false;try{kadan::serving::session(e,in,out);}catch(...){failed=true;}check(failed&&e.steps==0);}
    std::cout<<"bounded model worker protocol tests passed\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
