#include "kadan/h3_generation.hpp"
#include "kadan/generation_queue.hpp"
#include <nlohmann/json.hpp>
#include <array>
#include <cerrno>
#include <charconv>
#include <chrono>
#include <csignal>
#include <fcntl.h>
#include <iostream>
#include <poll.h>
#include <unistd.h>
#include <unordered_set>
namespace {
using Json=nlohmann::json;
using Queue=kadan::serving::GenerationQueue;
std::atomic_bool cancelled=false;
void request_stop(int){cancelled.store(true);}
struct Io {std::shared_ptr<kadan::Resources> resources; kadan::Handle id; explicit Io(std::shared_ptr<kadan::Resources> r):resources(std::move(r)),id(resources->reserve(kadan::Workload::video,{1024*1024})){} ~Io(){resources->released(id);std::cerr<<"resident_bytes="<<resources->snapshot().used[0]<<'\n';}};
void check(bool b,const char* why){if(!b)throw std::runtime_error(why);}
std::size_t number(const Json& object,const char* key,std::size_t fallback){if(!object.contains(key))return fallback;const auto& v=object.at(key);check(v.is_number_unsigned(),"h3_request_integer");return v.get<std::size_t>();}
std::string string(const Json& object,const char* key,std::size_t limit){check(object.contains(key)&&object.at(key).is_string(),"h3_request_string");auto out=object.at(key).get<std::string>();check(!out.empty()&&out.size()<=limit&&out.find('\0')==std::string::npos,"h3_request_string");return out;}
bool read_byte(char& c){while(!cancelled.load()){pollfd p{STDIN_FILENO,POLLIN,0};int ready=poll(&p,1,100);if(ready<0&&errno==EINTR)continue;check(ready>=0,"h3_stdin_poll");if(!ready)continue;auto n=read(STDIN_FILENO,&c,1);if(n==1)return true;if(n==0)return false;if(errno==EINTR)continue;throw std::runtime_error("h3_stdin_read");}return false;}
void publish(const std::string& result){const auto line=result+'\n';std::size_t at=0;const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(30);while(at<line.size()){if(cancelled.load())return;check(std::chrono::steady_clock::now()<deadline,"h3_stdout_timeout");pollfd p{STDOUT_FILENO,POLLOUT,0};auto ready=poll(&p,1,100);if(ready<0&&errno==EINTR)continue;check(ready>=0&&!(p.revents&(POLLERR|POLLHUP|POLLNVAL)),"h3_stdout_poll");if(!ready)continue;auto n=write(STDOUT_FILENO,line.data()+at,line.size()-at);if(n>0)at+=std::size_t(n);else check(n<0&&(errno==EINTR||errno==EAGAIN),"h3_stdout_write");}}
}
int main(int argc,char** argv){
 static_assert(std::atomic_bool::is_always_lock_free);std::cerr.tie(nullptr);
 try{
  check(argc==6,"usage: kadan-h3-worker TOKENIZER TEXT DENOISER TURBO VAE (JSON lines on stdin)");std::signal(SIGINT,request_stop);std::signal(SIGTERM,request_stop);std::signal(SIGPIPE,SIG_IGN);const auto flags=fcntl(STDOUT_FILENO,F_GETFL);check(flags>=0&&fcntl(STDOUT_FILENO,F_SETFL,flags|O_NONBLOCK)==0,"h3_stdout_nonblocking");
  auto resources=std::make_shared<kadan::Resources>(kadan::Footprint{1536ULL*1024*1024});Io io(resources);Queue queue(resources);kadan::video::H3Generation model(resources);const kadan::video::H3GenerationPaths paths{argv[1],argv[2],argv[3],argv[4],argv[5]};
  {
   auto run=[&](const std::string& frame){
    std::array<std::unordered_set<std::string>,17> keys;std::size_t events=0;const auto object=Json::parse(frame,[&](int depth,Json::parse_event_t event,Json& v){check(depth>=0&&depth<16&&++events<20000,"h3_request_json_limit");if(event==Json::parse_event_t::object_start)keys[depth+1].clear();if(event==Json::parse_event_t::key)check(keys[depth].insert(v.get<std::string>()).second,"h3_request_duplicate_key");return true;});check(object.is_object()&&object.size()<=7,"h3_request_object");for(auto it=object.begin();it!=object.end();++it)check(it.key()=="prompt"||it.key()=="output"||it.key()=="width"||it.key()=="height"||it.key()=="frames"||it.key()=="updates"||it.key()=="seed","h3_request_field");
    kadan::video::H3GenerationRequest request;request.prompt=string(object,"prompt",8192);request.output=string(object,"output",4096);request.width=number(object,"width",64);request.height=number(object,"height",64);request.frames=number(object,"frames",22);request.updates=number(object,"updates",8);request.seed=number(object,"seed",0);
    // No idle tensor residency: each stage streams and charges its actual peak
    // into the SAME ledger before allocation. The FIFO ticket holds only model
    // identity, not a second overlapping copy of that execution reservation.
    const auto id=queue.submit({"h3/streamed",kadan::Workload::video,{0}});std::exception_ptr failure;
    while(queue.pending()){auto action=queue.poll();if(action.kind==Queue::Kind::load){queue.loaded(id,true);}else if(action.kind==Queue::Kind::execute){try{model.execute(paths,request,cancelled);}catch(...){failure=std::current_exception();}if(cancelled.load())queue.cancel(id);queue.completed(id,false);}else if(action.kind==Queue::Kind::cleanup){queue.cleaned(action.reservation,true);}else throw std::runtime_error("h3_queue_stalled");}
    if(failure)std::rethrow_exception(failure);
    return Json{{"output",request.output},{"frames",request.frames},{"width",request.width},{"height",request.height},{"audio",false},{"resident_bytes",resources->snapshot().used[0]-1024*1024}}.dump();
   };
   std::string frame;char c;while(read_byte(c)){if(c!='\n'){check(frame.size()<65536,"h3_request_size");frame+=c;}else if(!frame.empty()){std::string reply;try{reply=run(frame);}catch(const std::exception& e){reply=Json{{"error",e.what()}}.dump();}publish(reply);frame.clear();}}
   check(frame.empty()||cancelled.load(),"h3_incomplete_frame");queue.stop();auto action=queue.poll();if(action.kind==Queue::Kind::cleanup)queue.cleaned(action.reservation,true);
  }
  return 0;
 }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
