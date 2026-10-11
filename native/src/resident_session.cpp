#include "kadan/model_worker.hpp"
#include "kadan/generation_queue.hpp"
#include <array>
#include <cerrno>
#include <istream>
#include <ostream>
#include <sstream>
#include <sys/random.h>

namespace kadan::serving {
namespace {
void require(bool v,const char* why){if(!v)throw std::runtime_error(why);}
void flush(std::ostream& out){out.flush();require(bool(out),"output_failed");}
bool frame(std::istream& in,std::string& line){
    line.clear();char c;
    while(in.get(c)){if(c=='\n')return true;require(c>=32&&c<=126&&line.size()<160,"invalid_frame");line+=c;}
    require(in.eof()&&line.empty(),"incomplete_frame");return false;
}
}
std::string session_identity(){
    std::array<unsigned char,16> bytes{};std::size_t at=0;
    while(at<bytes.size()){auto n=::getrandom(bytes.data()+at,bytes.size()-at,0);if(n<0&&errno==EINTR)continue;require(n>0,"session_entropy");at+=std::size_t(n);}
    constexpr char digits[]="0123456789abcdef";std::string result;result.reserve(32);
    for(auto b:bytes){result+=digits[b>>4];result+=digits[b&15];}return result;
}
void resident_session(ResidentEngine& engine,std::shared_ptr<Resources> resources,std::string_view identity,
                      std::istream& input,std::ostream& output){
    require(identity.size()==32&&identity.find_first_not_of("0123456789abcdef")==std::string_view::npos,"session_identity");
    GenerationQueue queue(resources);
    auto zero=resources->snapshot().capacity;std::fill(zero.begin(),zero.end(),0);
    const auto info=engine.info();
    require(info.vocabulary>0&&info.vocabulary<=262144&&info.capacity>0&&info.capacity<=max_resident_capacity,"invalid_info");
    output<<"ready 2 "<<identity<<' '<<info.vocabulary<<' '<<info.capacity<<'\n';flush(output);
    Handle active=0;std::size_t committed=0;bool ended=false;std::string line;
    auto cleanup=[&]{auto action=queue.poll();if(action.kind==GenerationQueue::Kind::cleanup){engine.park();queue.cleaned(action.reservation,true);}};
    while(frame(input,line)){
        std::istringstream fields(line);std::string op,session,id_text,extra;fields>>op>>session>>id_text;
        require(session==identity&&!id_text.empty(),"stale_session");const auto id=number(id_text);
        if(op=="submit"){
            require(id==0&&!(fields>>extra),"invalid_submit");
            // The executor owns all physical reservations in this SAME ledger;
            // the zero-byte queue token expresses lifecycle ownership only.
            auto request=queue.submit({"configured-qwen",Workload::llm,zero});
            output<<"queued "<<identity<<' '<<request<<'\n';
        }else if(op=="start"){
            require(!active&&id!=0&&!(fields>>extra),"invalid_start");
            auto action=queue.poll();require(action.request==id&&(action.kind==GenerationQueue::Kind::load||action.kind==GenerationQueue::Kind::execute),"fifo_head_required");
            engine.begin_request();if(action.kind==GenerationQueue::Kind::load)queue.loaded(id,true);
            active=id;committed=0;ended=false;output<<"started "<<identity<<' '<<id<<'\n';
        }else if(op=="step"){
            std::string token,flag;fields>>token>>flag;require(active==id&&id&&!(fields>>extra)&&(flag=="0"||flag=="1"),"invalid_step");
            auto value=number(token);require(value<info.vocabulary&&!ended&&committed<info.capacity,"context_finished");
            auto result=engine.step(unsigned(value),flag=="1");require(result.id<info.vocabulary&&result.committed==committed+1,"engine_contract");
            committed=result.committed;ended=result.eos&&flag=="1";
            output<<"token "<<identity<<' '<<id<<' '<<result.id<<' '<<unsigned(result.eos)<<' '<<committed<<'\n';
        }else if(op=="end"){
            require(active==id&&id&&!(fields>>extra),"invalid_end");engine.end_request();queue.completed(id,true);active=0;
            output<<"ended "<<identity<<' '<<id<<'\n';
        }else if(op=="cancel"){
            require(id&&active!=id&&!(fields>>extra),"active_cancel_requires_termination");
            require(queue.cancel(id),"stale_request");output<<"cancelled "<<identity<<' '<<id<<'\n';
        }else if(op=="park"){
            require(id==0&&!active&&!queue.pending()&&!(fields>>extra),"invalid_park");queue.evict_idle();cleanup();engine.park();
            output<<"parked "<<identity<<" 0\n";
        }else if(op=="cache"){
            require(id==0&&!active&&!(fields>>extra),"invalid_cache");
            auto s=engine.cache_stats();
            output<<"cache "<<identity<<" 0 "<<s.capacity<<' '<<s.ram<<' '<<s.cold<<' '<<s.hits<<' '<<s.misses<<' '<<s.hit_bytes<<' '<<s.source_bytes<<' '<<s.evictions<<' '<<s.entries<<'\n';
        }else if(op=="close"){
            require(id==0&&!(fields>>extra),"invalid_close");break;
        }else throw std::runtime_error("unknown_command");
        flush(output);
    }
    if(active){engine.end_request();queue.completed(active,false);active=0;}
    queue.stop();cleanup();engine.close();
    auto state=resources->snapshot();require(state.residents==0,"reservation_leak");for(auto n:state.used)require(n==0,"reservation_leak");
    output<<"closed "<<identity<<" 0\n";flush(output);
}
} // namespace kadan::serving
