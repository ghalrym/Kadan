// Mock protocol/control only: no model math, loading or CUDA execution.
#include "kadan/model_worker.hpp"
#include "kadan/glm.hpp"
#include <sstream>
#include <iostream>
void check(bool ok){if(!ok)throw std::runtime_error("test_failed");}
struct MockEngine : kadan::serving::ResidentEngine {
 std::size_t capacity=1048576,count=0;bool closed=false;
 kadan::serving::Info info()const override{return {154880,capacity,0,0};}
 void reset()override{count=0;}
 void begin_request()override{reset();}
 void end_request()override{}
 void park()override{}
 void close()override{closed=true;}
 kadan::serving::Token step(unsigned,bool)override{return {154820,kadan::glm::is_eos(154820),++count};}
};
int main(){
 static_assert(kadan::serving::max_capacity==262144);
 static_assert(kadan::serving::max_resident_capacity==1048576);
 static_assert(kadan::glm::is_eos(154820)&&kadan::glm::is_eos(154827)&&kadan::glm::is_eos(154829)&&!kadan::glm::is_eos(154821));
 const std::string id(32,'a');
 auto resources=std::make_shared<kadan::Resources>(kadan::Footprint{1024});MockEngine engine;
 std::istringstream in("submit "+id+" 0\nstart "+id+" 1\nstep "+id+" 1 42 0\nend "+id+" 1\nclose "+id+" 0\n");std::ostringstream out;
 kadan::serving::resident_session(engine,resources,id,in,out);
 check(out.str().find("ready 2 "+id+" 154880 1048576")!=std::string::npos);
 check(out.str().find("token "+id+" 1 154820 1 1")!=std::string::npos);
 check(engine.closed&&resources->snapshot().residents==0);
 engine.capacity=1048577;std::istringstream empty;bool rejected=false;
 try{kadan::serving::resident_session(engine,resources,id,empty,out);}catch(const std::runtime_error&){rejected=true;}
 check(rejected);std::cout<<"resident context and prefill EOS protocol passed (mock only)\n";
}
