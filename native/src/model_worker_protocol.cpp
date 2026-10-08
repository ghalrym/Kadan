#include "kadan/model_worker.hpp"
#include <charconv>
#include <istream>
#include <ostream>
#include <stdexcept>
#include <string>
namespace kadan::serving {
namespace {
void require(bool value,const char* why){if(!value)throw std::runtime_error(why);}
void flush(std::ostream& output){output.flush();require(bool(output),"output_failed");}
bool frame(std::istream& input,std::string& line){
    line.clear();char c;
    while(input.get(c)){
        if(c=='\n')return true;
        require(c>=32&&c<=126&&line.size()<max_frame,"invalid_frame");line.push_back(c);
    }
    require(input.eof()&&line.empty(),"incomplete_frame");return false;
}
}
std::size_t number(std::string_view v){
    require(!v.empty()&&v.size()<=20,"invalid_number");
    std::size_t n=0;auto[end,error]=std::from_chars(v.data(),v.data()+v.size(),n);
    require(error==std::errc{}&&end==v.data()+v.size(),"invalid_number");return n;
}
void session(Engine& engine,std::istream& input,std::ostream& output){
    const auto info=engine.info();require(info.vocabulary>0&&info.vocabulary<=262144&&info.capacity>0&&info.capacity<=max_capacity,"invalid_info");
    output<<"ready 1 "<<info.vocabulary<<' '<<info.capacity<<' '<<info.arena_bytes<<' '<<info.host_bytes<<'\n';flush(output);
    std::size_t committed=0;bool finished=false;std::string line;
    while(frame(input,line)){
        if(line=="close")break;
        if(line=="reset"){engine.reset();committed=0;finished=false;output<<"ok reset\n";flush(output);continue;}
        require(line.starts_with("step "),"unknown_command");
        const auto split=line.find(' ',5);require(split!=std::string::npos,"invalid_step");
        const auto token=number(std::string_view(line).substr(5,split-5));const auto flag=std::string_view(line).substr(split+1);
        require(flag=="0"||flag=="1","invalid_stop");require(token<info.vocabulary,"token_range");
        require(!finished&&committed<info.capacity,"context_finished");
        auto selected=engine.step(unsigned(token),flag=="1");
        require(selected.id<info.vocabulary&&selected.committed==committed+1,"engine_contract");
        committed=selected.committed;finished=(selected.eos&&flag=="1");
        output<<"token "<<selected.id<<' '<<unsigned(selected.eos)<<' '<<committed<<'\n';flush(output);
    }
    engine.close();output<<"closed 0\n";flush(output);
}
}
